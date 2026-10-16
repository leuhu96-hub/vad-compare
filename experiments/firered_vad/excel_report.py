"""Báo cáo Excel cho FireRedVAD theo đúng bố cục sheet "KQ du lieu that" (mục A, B, C) của file
Tong_hop_danh_gia_VAD_test.xlsx, để so trực tiếp với YAMNet.

    # chỉ có audio + GT + model: tự chia dev/test (30 % dev, theo file, phân tầng category, seed 3)
    python excel_report.py --audio-root "D:/data/phase_1" --gt-root "D:/data/Groundtruth" \
        --model-dir models -o firered_report.xlsx

    # toàn bộ là test, ngưỡng cố định
    python excel_report.py --audio-root ... --gt-root ... --model-dir models --dev-ratio 0 --threshold 0.5

    # dùng đúng chia dev/test của file YAMNet và thêm mục so sánh D
    python excel_report.py --xlsx Tong_hop_danh_gia_VAD_test.xlsx --audio-root ... --gt-root ... --model-dir models

Giữ nguyên quy trình của file YAMNet:
  * chia dev / test: đọc sheet "DL theo file" của --xlsx, hoặc --split-csv, hoặc tự chia (--dev-ratio)
  * GT bin 0.5 s (đa số thời lượng), cờ biên ±1 bin quanh điểm chuyển GT
  * quét ngưỡng 0.01 … 0.95 (bước 0.01), chọn ngưỡng trên dev theo DCF-min hoặc F1-max
  * DCF = w·MR + (1−w)·FAR (w = 0.75), CI 95% bootstrap theo file (1000 lần, seed 0)

Ba cấu hình FireRed (tương ứng A0 … A4 của YAMNet):
  F0  prob 10 ms ≥ ngưỡng, không hậu xử lý
  F1  hậu xử lý gốc FireRed: làm mượt 5 frame, máy trạng thái 0.2 s / 0.2 s, lùi đầu đoạn, cắt đoạn > 20 s
  F2  prob 10 ms ≥ ngưỡng + hậu xử lý A4 của YAMNet: pad 100 / 120 ms → drop → merge < 500 ms

Quyết định 10 ms → bin 0.5 s: --rebin overlap (mặc định; bin = speech nếu phủ ≥ --min-overlap, mặc định 20 %)
hoặc --rebin midpoint (tâm bin nằm trong đoạn speech).

Prob của model được cache trong --cache (chạy lại chỉ mất vài giây). Excel sinh ra dùng công thức
SUMIFS trên sheet "DL quet nguong": đổi cấu hình (B7), tiêu chí (B8), trọng số DCF (B6) là tự tính lại.
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import vad_excel  # noqa: E402
from dataset_reader import file_key  # noqa: E402

FRAME = 0.01
THRESHOLDS = vad_excel.THRESHOLDS          # dùng chung (có thể được thêm ngưỡng lẻ từ --threshold)


# =========================================================================== FireRed prob (cache)
def get_probs(items, args, log=print):
    model = None
    os.makedirs(args.cache, exist_ok=True)
    for i, it in enumerate(items, 1):
        h = hashlib.md5(f"{os.path.abspath(it['audio'])}|{args.onnx or args.model_dir}|{args.mode}".encode()).hexdigest()[:8]
        cp = os.path.join(args.cache, f"{file_key(it['audio'])}_{h}.npz")
        if os.path.exists(cp) and not args.force:
            z = np.load(cp)
            it["probs"], it["dur"] = z["probs"].astype(np.float64), float(z["duration"])
            continue
        if model is None:
            from firered_onnx import FireRedOnnx

            model = FireRedOnnx(model_dir=args.model_dir, mode=args.mode, threads=args.threads,
                                onnx_path=args.onnx, cmvn_path=args.cmvn)
            log(f"[model] {model.onnx_path} ({model.mode})")
        r = model.run_file(it["audio"])
        it["probs"], it["dur"] = r.probs.astype(np.float64), float(r.duration)
        np.savez_compressed(cp, probs=r.probs.astype(np.float32), duration=r.duration)
        log(f"[{i}/{len(items)}] {os.path.basename(it['audio'])} ({r.duration:.1f} s) "
            f"onnx {r.timings.get('7_onnx', 0):.2f} s")


# =========================================================================== post-processing (10 ms)
def _runs(b):
    """-> starts, ends (exclusive) của các đoạn 1 trong mảng 0/1."""
    d = np.diff(np.r_[0, b.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _from_runs(n, starts, ends):
    out = np.zeros(n, np.int8)
    for s, e in zip(starts, ends):
        out[s:e] = 1
    return out


def fsm(b, min_sp, min_sil):
    """Máy trạng thái FireRed (giống _state_machine của upstream), cài theo run-length."""
    n = len(b)
    if min_sp <= 0 and min_sil <= 0:
        return b.astype(np.int8)
    msp, msil = max(min_sp, 1), max(min_sil, 1)
    dec = np.zeros(n, np.int8)
    idx = np.flatnonzero(np.diff(b)) + 1
    speaking = False
    for s, e in zip(np.r_[0, idx], np.r_[idx, n]):
        L = e - s
        if not speaking:
            if b[s] == 1 and L > msp:
                dec[s:e] = 1
                speaking = True
        elif b[s] == 1:
            dec[s:e] = 1
        elif L > msil:
            dec[s:s + msil] = 1
            speaking = False
        else:
            dec[s:e] = 1
    return dec


def fix_smooth_start(dec, w):
    if w <= 1 or len(dec) < 2:
        return dec
    new = dec.copy()
    for t in np.flatnonzero((dec[1:] == 1) & (dec[:-1] == 0)) + 1:
        new[max(0, t - w):t] = 1
    return new


def split_long(dec, probs, max_frame):
    """Giống _split_long của upstream (kể cả cách đổi giây -> frame bằng int())."""
    starts, ends = _runs(dec)
    if not len(starts):
        return dec
    T = len(dec)
    new = dec.copy()
    for j, (s0, e0) in enumerate(zip(starts, ends)):
        a = int(round(s0 * FRAME, 3) / FRAME)
        e_s = round(T * FRAME + 0.025, 3) if e0 == T else round(e0 * FRAME, 3)
        b = int(e_s / FRAME)
        if b - a <= max_frame:
            continue
        seg, start = probs[a:b], 0
        while len(seg) - start > max_frame:
            w0, w1 = int(start + max_frame / 2), int(start + max_frame)
            k = w0 + int(np.argmin(seg[w0:w1]))
            new[a + k] = 0
            start = k + 1
    return new


def smooth(p, w):
    from firered_onnx import _smooth

    return _smooth(p, w)


def firered_post(sm, probs, thr, post):
    b = (sm >= thr).astype(np.int8)
    dec = fsm(b, post.min_speech_frame, post.min_silence_frame)
    dec = fix_smooth_start(dec, max(1, post.smooth_window_size))
    return split_long(dec, probs, post.max_speech_frame)


def rules_post(on, pad_b, pad_a, drop, merge, order):
    """Hậu xử lý kiểu YAMNet A4 trên frame 10 ms (đơn vị frame). merge: lấp khoảng hở < merge."""
    T = len(on)
    s, e = _runs(on)
    for step in order:
        if not len(s):
            break
        if step == "pad":
            s, e = np.maximum(s - pad_b, 0), np.minimum(e + pad_a, T)
            keep_s, keep_e = [s[0]], [e[0]]
            for a, b in zip(s[1:], e[1:]):
                if a <= keep_e[-1]:
                    keep_e[-1] = max(keep_e[-1], b)
                else:
                    keep_s.append(a)
                    keep_e.append(b)
            s, e = np.array(keep_s), np.array(keep_e)
        elif step == "drop":
            k = (e - s) >= drop
            s, e = s[k], e[k]
        elif step == "merge":
            keep_s, keep_e = [s[0]], [e[0]]
            for a, b in zip(s[1:], e[1:]):
                if a - keep_e[-1] < merge:
                    keep_e[-1] = max(keep_e[-1], b)
                else:
                    keep_s.append(a)
                    keep_e.append(b)
            s, e = np.array(keep_s), np.array(keep_e)
    return _from_runs(T, s, e)


# =========================================================================== bins (10 ms -> bin GT)
class Binner:
    """Quyết định 10 ms -> bin GT.
    Frame cuối phủ thêm 25 ms (cửa sổ 25 ms), giống decision_to_segment của FireRed: đoạn speech chạm
    frame cuối kéo tới min(T·10 ms + 25 ms, độ dài file). Nhờ vậy khớp từng bin với run.py."""

    def __init__(self, n, dur, bin_s, rule, min_ov, n_frames):
        self.n, self.fpb, self.T = n, int(round(bin_s / FRAME)), n_frames
        edges = np.arange(n + 1) * bin_s
        self.width = np.minimum(edges[1:], dur) - edges[:-1]
        self.rule, self.min_ov = rule, min_ov
        mid = (edges[:-1] + np.minimum(edges[1:], dur)) / 2
        self.mid = np.floor(mid / FRAME + 1e-9).astype(int)
        self.mid_t = mid
        t0 = round(n_frames * FRAME, 3)                       # giống decisions_to_segments (làm tròn ms)
        self.tail_end = min(round(t0 + 0.025, 3), dur)
        self.tail_start = t0
        self.tail_cov = np.clip(np.minimum(edges[1:], self.tail_end) - np.maximum(edges[:-1], t0), 0, None)

    def _pad(self, x, fill=0):
        L = self.n * self.fpb
        out = np.full(L, fill, dtype=x.dtype)
        m = min(L, len(x))
        out[:m] = x[:m]
        return out

    def _mid_vals(self, x, fill):
        """Giá trị tại tâm bin; tâm rơi vào đuôi 25 ms sau frame cuối thì lấy frame cuối."""
        v = np.full(self.n, fill, dtype=np.float64)
        inside = self.mid < len(x)
        v[inside] = x[self.mid[inside]]
        tail = ~inside & (self.mid_t >= self.tail_start) & (self.mid_t < self.tail_end) & (len(x) > 0)
        if tail.any():
            v[tail] = x[-1]
        return v

    def bins(self, dec):
        if self.rule == "midpoint":
            return self._mid_vals(dec, 0) > 0
        cov = self._pad(dec).reshape(self.n, self.fpb).sum(1) * FRAME
        if len(dec) and dec[-1]:
            cov = cov + self.tail_cov
        return cov >= self.min_ov * self.width - 1e-9

    def bins_all_thr(self, probs, thr):
        """F0 cho mọi ngưỡng một lần: -> [n_thr, n] bool."""
        if self.rule == "midpoint":
            return self._mid_vals(probs, -1.0)[None, :] >= thr[:, None]
        P = self._pad(probs, -1.0).reshape(self.n, self.fpb)
        cov = (P[None, :, :] >= thr[:, None, None]).sum(2) * FRAME
        if len(probs):
            cov = cov + (probs[-1] >= thr)[:, None] * self.tail_cov[None, :]
        return cov >= self.min_ov * self.width[None, :] - 1e-9


# =========================================================================== spec cho vad_excel
class FireRedSpec:
    name, short, abbr, sheet = "FireRedVAD", "FireRed", "FR", "KQ FireRed"
    configs = ["F0", "F1", "F2"]
    default_config, default_thr = "F1", 0.4
    frame_desc = "10 ms"
    here, out_name = HERE, "firered_report.xlsx"
    doc = __doc__

    @staticmethod
    def add_args(ap):
        ap.add_argument("--model-dir", default=os.path.join(HERE, "models"))
        ap.add_argument("--onnx")
        ap.add_argument("--cmvn")
        ap.add_argument("--mode", default="auto", choices=["auto", "nonstream", "stream", "stream_cached"])
        ap.add_argument("--f2-pad-before", type=float, default=0.10)
        ap.add_argument("--f2-pad-after", type=float, default=0.12)
        ap.add_argument("--f2-drop", type=float, default=0.0)
        ap.add_argument("--f2-merge", type=float, default=0.5)
        ap.add_argument("--f2-order", default="pad,drop,merge")

    @staticmethod
    def prepare(args):
        args.f2_order = [s.strip() for s in args.f2_order.split(",") if s.strip()]

    @staticmethod
    def model_label(args):
        return args.onnx or args.model_dir

    @staticmethod
    def descriptions(args):
        return {
            "F0": "Prob 10 ms ≥ ngưỡng, không hậu xử lý",
            "F1": "Hậu xử lý gốc FireRed: làm mượt 5 frame, máy trạng thái 0.2 s / 0.2 s, lùi đầu đoạn 50 ms, "
                  "cắt đoạn > 20 s",
            "F2": f"Prob 10 ms ≥ ngưỡng + hậu xử lý A4 của YAMNet (pad {args.f2_pad_before * 1000:.0f}/"
                  f"{args.f2_pad_after * 1000:.0f} ms, merge < {args.f2_merge * 1000:.0f} ms)",
        }

    @staticmethod
    def pp_note(args):
        return (f"F2 = pad {args.f2_pad_before * 1000:.0f}/{args.f2_pad_after * 1000:.0f} ms, drop "
                f"{args.f2_drop * 1000:.0f} ms, merge < {args.f2_merge * 1000:.0f} ms, thứ tự "
                f"{' → '.join(args.f2_order)}.")

    @staticmethod
    def load_probs(items, args):
        get_probs(items, args)

    @staticmethod
    def decide(it, args, thr, n):
        from firered_onnx import PostConfig

        post = PostConfig()
        B = Binner(n, it["dur"], args.bin, args.rebin, args.min_overlap, len(it["probs"]))
        p = it["probs"]
        sm = smooth(p, max(1, post.smooth_window_size))
        fr = lambda s: int(round(s / FRAME))  # noqa: E731
        return {
            "F0": B.bins_all_thr(p, thr),
            "F1": np.stack([B.bins(firered_post(sm, p, t, post)) for t in thr]),
            "F2": np.stack([B.bins(rules_post((p >= t).astype(np.int8), fr(args.f2_pad_before),
                                              fr(args.f2_pad_after), fr(args.f2_drop), fr(args.f2_merge),
                                              args.f2_order)) for t in thr]),
        }


SPEC = FireRedSpec()


def main(argv=None):
    return vad_excel.main(SPEC, argv)


if __name__ == "__main__":
    main()
