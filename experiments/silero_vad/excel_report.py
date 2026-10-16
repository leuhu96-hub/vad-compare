"""Báo cáo Excel cho Silero VAD, cùng bố cục sheet "KQ du lieu that" (mục A, B, C) của file
Tong_hop_danh_gia_VAD_test.xlsx (YAMNet) và báo cáo FireRed (experiments/firered_vad/excel_report.py).

    # chỉ có audio + GT + model: tự chia dev/test (30 % dev, theo file, phân tầng category, seed 3)
    python excel_report.py --audio-root "D:/data/phase_1" --gt-root "D:/data/Groundtruth" \
        --model models/silero_vad.onnx -o silero_report.xlsx

    # toàn bộ là test, ngưỡng cố định
    python excel_report.py --audio-root ... --gt-root ... --dev-ratio 0 --threshold 0.5

    # dùng đúng chia dev/test của file YAMNet và thêm mục so sánh D
    python excel_report.py --xlsx Tong_hop_danh_gia_VAD_test.xlsx --audio-root ... --gt-root ...

Ba cấu hình Silero (tương ứng A0 … A4 của YAMNet, F0 … F2 của FireRed):
  S0  prob mỗi khối 32 ms ≥ ngưỡng, không hậu xử lý
  S1  hậu xử lý gốc Silero (get_speech_timestamps): vào khi prob ≥ ngưỡng, ra khi < ngưỡng − 0.15,
      lặng ≥ 100 ms mới cắt, bỏ đoạn ≤ 250 ms, pad 30 ms mỗi phía
  S2  prob 32 ms ≥ ngưỡng + hậu xử lý A4 của YAMNet: pad 100 / 120 ms → drop → merge < 500 ms

Các tham số chung (--rebin, --min-overlap, --dcf-weight, --criterion, --threshold, --dev-ratio…) giống hệt
báo cáo FireRed; xem experiments/common/vad_excel.py.
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
from silero_onnx import CHUNK, SR, default_model_path  # noqa: E402

THRESHOLDS = vad_excel.THRESHOLDS


def get_probs(items, args, log=print):
    model = None
    os.makedirs(args.cache, exist_ok=True)
    for i, it in enumerate(items, 1):
        h = hashlib.md5(f"{os.path.abspath(it['audio'])}|silero|{os.path.abspath(args.model)}".encode()).hexdigest()[:8]
        cp = os.path.join(args.cache, f"{file_key(it['audio'])}_silero_{h}.npz")
        if os.path.exists(cp) and not args.force:
            z = np.load(cp)
            it["probs"], it["n_samples"] = z["probs"].astype(np.float64), int(z["n_samples"])
            it["dur"] = it["n_samples"] / SR
            continue
        if model is None:
            from silero_onnx import SileroOnnx

            model = SileroOnnx(args.model, threads=args.threads)
            args.model_desc = f"{model.model_path} [{model.kind}]"
            log(f"[model] {model.describe()}")
        r = model.run_file(it["audio"])
        it["probs"], it["n_samples"], it["dur"] = r.probs.astype(np.float64), r.n_samples, r.duration
        np.savez_compressed(cp, probs=r.probs, n_samples=r.n_samples)
        log(f"[{i}/{len(items)}] {os.path.basename(it['audio'])} ({r.duration:.1f} s) onnx {r.timings['onnx']:.2f} s")


def frame_segments(on, n_samples):
    """Khối 32 ms đang bật -> [(start_s, end_s)]; khối k phủ mẫu [512k, min(512(k+1), N))."""
    d = np.diff(np.r_[0, on.astype(np.int8), 0])
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(a * CHUNK / SR, min(b * CHUNK, n_samples) / SR) for a, b in zip(s, e)]


class SileroSpec:
    name, short, abbr, sheet = "Silero VAD", "Silero", "SL", "KQ Silero"
    configs = ["S0", "S1", "S2"]
    default_config, default_thr = "S1", 0.5
    frame_desc = "khối 32 ms"
    here, out_name = HERE, "silero_report.xlsx"
    doc = __doc__

    @staticmethod
    def add_args(ap):
        ap.add_argument("--model", default=default_model_path(),
                        help="file .onnx có sẵn (tên tuỳ ý) hoặc thư mục chứa nó; tự nhận v5/v6, v4, sequence")
        ap.add_argument("--min-speech-ms", type=int, default=250, help="S1: bỏ đoạn ngắn hơn")
        ap.add_argument("--min-silence-ms", type=int, default=100, help="S1: lặng tối thiểu để cắt đoạn")
        ap.add_argument("--speech-pad-ms", type=int, default=30, help="S1: pad mỗi phía")
        ap.add_argument("--max-speech-s", type=float, default=float("inf"), help="S1: cắt đoạn dài hơn")
        ap.add_argument("--neg-offset", type=float, default=0.15,
                        help="S1: ngưỡng ra = max(ngưỡng − giá trị này, 0.01) (mặc định của Silero)")
        ap.add_argument("--s2-pad-before", type=float, default=0.10)
        ap.add_argument("--s2-pad-after", type=float, default=0.12)
        ap.add_argument("--s2-drop", type=float, default=0.0)
        ap.add_argument("--s2-merge", type=float, default=0.5)
        ap.add_argument("--s2-order", default="pad,drop,merge")

    @staticmethod
    def prepare(args):
        args.s2_order = [s.strip() for s in args.s2_order.split(",") if s.strip()]

    @staticmethod
    def model_label(args):
        return getattr(args, "model_desc", args.model)

    @staticmethod
    def descriptions(args):
        return {
            "S0": "Prob mỗi khối 32 ms ≥ ngưỡng, không hậu xử lý",
            "S1": f"Hậu xử lý gốc Silero (get_speech_timestamps): ra khi < ngưỡng − {args.neg_offset:g}, "
                  f"lặng ≥ {args.min_silence_ms} ms mới cắt, bỏ đoạn ≤ {args.min_speech_ms} ms, "
                  f"pad {args.speech_pad_ms} ms",
            "S2": f"Prob 32 ms ≥ ngưỡng + hậu xử lý A4 của YAMNet (pad {args.s2_pad_before * 1000:.0f}/"
                  f"{args.s2_pad_after * 1000:.0f} ms, merge < {args.s2_merge * 1000:.0f} ms)",
        }

    @staticmethod
    def pp_note(args):
        return (f"S2 = pad {args.s2_pad_before * 1000:.0f}/{args.s2_pad_after * 1000:.0f} ms, drop "
                f"{args.s2_drop * 1000:.0f} ms, merge < {args.s2_merge * 1000:.0f} ms, thứ tự "
                f"{' → '.join(args.s2_order)}.")

    @staticmethod
    def load_probs(items, args):
        get_probs(items, args)

    @staticmethod
    def decide(it, args, thr, n):
        from silero_utils import get_speech_timestamps_from_probs

        p, N, dur = it["probs"], it["n_samples"], it["dur"]
        plist = [float(x) for x in p]
        bins = lambda segs: vad_excel.seg_bins(segs, n, dur, args.bin, args.rebin, args.min_overlap)  # noqa: E731
        D0, D1, D2 = [], [], []
        for t in thr:
            raw = frame_segments(p >= t, N)
            D0.append(bins(raw))
            ts = get_speech_timestamps_from_probs(
                plist, sampling_rate=SR, threshold=float(t), min_speech_duration_ms=args.min_speech_ms,
                max_speech_duration_s=args.max_speech_s, min_silence_duration_ms=args.min_silence_ms,
                speech_pad_ms=args.speech_pad_ms, neg_threshold=max(float(t) - args.neg_offset, 0.01),
                audio_length_samples=N)
            D1.append(bins(vad_excel.merge_segs([(d["start"] / SR, d["end"] / SR) for d in ts])))
            D2.append(bins(vad_excel.rules_segs(raw, dur, args.s2_pad_before, args.s2_pad_after,
                                                args.s2_drop, args.s2_merge, args.s2_order)))
        return {"S0": np.stack(D0), "S1": np.stack(D1), "S2": np.stack(D2)}


SPEC = SileroSpec()


def main(argv=None):
    return vad_excel.main(SPEC, argv)


if __name__ == "__main__":
    main()
