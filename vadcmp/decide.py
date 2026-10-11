"""Từ score theo frame/hop -> segment speech. Hai phương pháp, chọn bằng `postprocess.method`:

rules (mặc định, giống lib.so hiện tại)
    threshold -> segment -> pad / merge / drop theo `order`.
    Lưu ý: order = pad -> merge -> drop làm drop/merge xét trên đoạn ĐÃ pad, nên
    `min_speech` thực tế = min_speech - (pad_before + pad_after) và
    `merge_gap` thực tế = merge_gap + (pad_before + pad_after).

fsm (kiểu FireRedVAD, tổng quát cho mọi hop)
    1. smooth_s      : trung bình trượt nhìn về trước (FireRed: 5 frame = 50 ms). 0 = tắt.
    2. threshold     : ngưỡng vào speech (thr_on).
       thr_off       : ngưỡng ra (hysteresis). Đang nói thì chỉ coi là im khi score < thr_off.
    3. máy trạng thái trên chuỗi 0/1 (giống VadPostprocessor):
         IM LẶNG --1--> CÓ THỂ NÓI --1 liên tục dài hơn min_speech--> ĐANG NÓI
         ĐANG NÓI --0--> CÓ THỂ DỪNG --0 liên tục dài hơn min_silence--> IM LẶNG
       => đoạn ngắn hơn min_speech bị bỏ, khoảng lặng ngắn hơn min_silence bị lấp,
          XÉT TRÊN CHUỖI THÔ (trước khi nới đoạn).
    4. hangover      : FireRed giữ "đang nói" thêm min_silence sau khi tiếng dừng
                       (= pad_after ẩn bằng min_silence). hangover=false cắt đuôi về
                       frame speech cuối cùng -> không phình đoạn, giảm FAR.
    5. bù trễ làm mượt: lùi đầu đoạn smooth_s (như FireRed _fix_smooth_window_start).
    6. pad_before / pad_after: nới đoạn, làm SAU CÙNG. Cho phép giá trị âm = co đoạn
                       (bù độ nhoè của cửa sổ dài như YAMNet 0.96 s).
    7. max_speech_s  : cắt đoạn dài hơn ngưỡng tại điểm score thấp nhất (0 = tắt).
"""
from __future__ import annotations

import numpy as np

from vadcmp.timeline import FrameScores, binarize, merge_segments, postprocess

FSM_DEFAULTS = {"smooth_s": 0.0, "thr_off": None, "min_speech": 0.2, "min_silence": 0.2,
                "hangover": True, "pad_before": 0.0, "pad_after": 0.0, "max_speech": 0.0}


def _runs(b: np.ndarray):
    """-> list[(value, start, end_exclusive)]"""
    if len(b) == 0:
        return []
    idx = np.flatnonzero(np.diff(b)) + 1
    starts = np.r_[0, idx]
    ends = np.r_[idx, len(b)]
    return [(int(b[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def hysteresis(p: np.ndarray, thr_on: float, thr_off: float | None) -> np.ndarray:
    """Vào speech khi p >= thr_on; đang speech thì chỉ ra khi p < thr_off; ở giữa -> giữ trạng thái cũ."""
    p = np.asarray(p, dtype=np.float64)
    if thr_off is None or thr_off >= thr_on:
        return (p >= thr_on).astype(np.int8)
    val = np.full(len(p) + 1, -1, np.int8)
    val[0] = 0
    val[1:][p >= thr_on] = 1
    val[1:][p < thr_off] = 0
    idx = np.where(val >= 0, np.arange(len(val)), 0)
    return val[np.maximum.accumulate(idx)][1:].astype(np.int8)


def causal_mean(p: np.ndarray, w: int) -> np.ndarray:
    """Trung bình trượt nhìn về trước w frame (đầu file chia cho số frame đã có), như FireRed."""
    p = np.asarray(p, dtype=np.float64)
    if w <= 1:
        return p
    c = np.cumsum(np.r_[0.0, p])
    idx = np.arange(1, len(p) + 1)
    lo = np.maximum(0, idx - w)
    return (c[idx] - c[lo]) / (idx - lo)


def state_machine(b: np.ndarray, min_sp: int, min_sil: int):
    """Máy trạng thái FireRed, cài theo run-length (O(số run)).
    -> (decisions, tail) với tail[i] = 1 nếu frame i chỉ là 'có thể dừng' (hangover)."""
    n = len(b)
    dec = np.zeros(n, np.int8)
    tail = np.zeros(n, np.int8)
    speaking = False
    for v, s, e in _runs(b):
        L = e - s
        if not speaking:
            if v == 1 and L > min_sp:          # cần > min_sp frame 1 liên tục (giống FireRed)
                dec[s:e] = 1
                speaking = True
        else:
            if v == 1:
                dec[s:e] = 1
            elif L > min_sil:                  # đủ dài -> về im lặng; min_sil frame đầu vẫn là 1
                dec[s:s + min_sil] = 1
                tail[s:s + min_sil] = 1
                speaking = False
            else:                              # khoảng lặng ngắn -> lấp
                dec[s:e] = 1
                if e == n:
                    tail[s:e] = 1
    return dec, tail


def fsm_config(pp: dict) -> dict:
    return {**FSM_DEFAULTS, **{k: v for k, v in (pp or {}).items() if k in FSM_DEFAULTS}}


def fsm_prepare(fs: FrameScores, smooth_s: float):
    """-> (fs đã tiled, hop, w, p đã làm mượt). Tách riêng để `tune` dùng lại."""
    fs = fs.tiled()
    hop = float(np.median(np.diff(fs.start))) if len(fs.start) > 1 else 0.01
    w = max(1, int(round(float(smooth_s) / hop)))
    return fs, hop, w, causal_mean(fs.prob, w)


def fsm_segments(fs: FrameScores, p: np.ndarray, b: np.ndarray, duration: float, hop: float,
                 w: int, cfg: dict) -> np.ndarray:
    """Bước 3..7 từ chuỗi nhị phân b (sau threshold/hysteresis)."""
    min_sp = _frames(cfg["min_speech"], hop)
    min_sil = _frames(cfg["min_silence"], hop)
    dec, tail = state_machine(b, min_sp, min_sil)
    if not cfg["hangover"]:
        dec = dec & ~(tail.astype(bool) & (b == 0))
    if w > 1:                                   # bù trễ làm mượt ở đầu đoạn
        on = np.flatnonzero(np.diff(np.r_[0, dec]) == 1)
        for t in on:
            dec[max(0, t - w):t] = 1
    on = dec.astype(bool)
    if not on.any():
        return np.zeros((0, 2))
    seg = merge_segments(np.stack([fs.start[on], fs.end[on]], axis=1), gap=1e-6)
    if cfg["pad_before"] or cfg["pad_after"]:
        seg[:, 0] = np.maximum(seg[:, 0] - float(cfg["pad_before"]), 0.0)
        seg[:, 1] = np.minimum(seg[:, 1] + float(cfg["pad_after"]), duration)
        seg = merge_segments(seg[seg[:, 1] > seg[:, 0]])   # pad âm (co đoạn) có thể làm đoạn rỗng
    mx = float(cfg["max_speech"])
    if mx > 0:
        seg = _split_long(seg, fs, p, mx)
    return seg


def _frames(sec, hop) -> int:
    """Số frame nguyên nằm trong `sec` giây. Đoạn phải DÀI HƠN số frame này (giống FireRed:
    min_speech 0.2 s @10 ms = 20 frame -> cần >= 21 frame; @80 ms = 2 frame -> cần >= 3 hop = 0.24 s)."""
    return int(np.floor(float(sec) / hop + 1e-6))


def _thr_off(v):
    return None if v in (None, "none", "None", "") else float(v)


def decide_fsm(fs: FrameScores, duration: float, threshold: float, pp: dict) -> np.ndarray:
    if len(fs.prob) == 0:
        return np.zeros((0, 2))
    cfg = fsm_config(pp)
    fs, hop, w, p = fsm_prepare(fs, cfg["smooth_s"])
    b = hysteresis(p, threshold, _thr_off(cfg["thr_off"]))
    return fsm_segments(fs, p, b, duration, hop, w, cfg)


def _split_long(seg, fs, p, mx):
    out = []
    for s, e in seg:
        while e - s > mx:
            m = (fs.start >= s + mx / 2) & (fs.end <= s + mx)
            if not m.any():
                break
            k = np.flatnonzero(m)[np.argmin(p[m])]
            out.append((s, fs.start[k]))
            s = fs.end[k]
        out.append((s, e))
    return np.array(out).reshape(-1, 2)


def decide(fs: FrameScores, duration: float, threshold: float, pp: dict | None) -> np.ndarray:
    """Điểm vào chung: FrameScores -> segments (N, 2)."""
    if pp and pp.get("method", "rules") == "fsm":
        return decide_fsm(fs, duration, threshold, pp)
    seg = binarize(fs, threshold)
    if pp:
        seg = postprocess(seg, duration, pp.get("pad_before", 0), pp.get("pad_after", 0),
                          pp.get("merge_gap", 0), pp.get("min_speech", 0),
                          pp.get("order", ("pad", "merge", "drop")))
    return seg


def describe(pp) -> str:
    """Mô tả ngắn cấu hình post-processing (cho report / log)."""
    if pp in (None, "none", False):
        return "không"
    if not isinstance(pp, dict):
        return str(pp)
    if pp.get("method", "rules") == "fsm":
        c = fsm_config(pp)
        off = c["thr_off"]
        parts = [f"FSM kiểu FireRed: min_speech {c['min_speech']} s, min_silence {c['min_silence']} s",
                 "hangover " + ("bật (giữ thêm min_silence)" if c["hangover"] else "tắt")]
        if off not in (None, "none"):
            parts.append(f"thr_off {off}")
        if c["smooth_s"]:
            parts.append(f"làm mượt {c['smooth_s']} s")
        if c["pad_before"] or c["pad_after"]:
            parts.append(f"pad {c['pad_before']}/{c['pad_after']} s")
        if c["max_speech"]:
            parts.append(f"cắt đoạn > {c['max_speech']} s")
        return ", ".join(parts)
    order = pp.get("order", ("pad", "merge", "drop"))
    return (f"pad trước {pp.get('pad_before', 0)} s, pad sau {pp.get('pad_after', 0)} s, "
            f"merge gap < {pp.get('merge_gap', 0)} s, bỏ segment < {pp.get('min_speech', 0)} s "
            f"(thứ tự: {' → '.join(order)})")
