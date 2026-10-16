"""YAMNet bước 6–7: gộp score cửa sổ thành score mỗi hop, rồi hậu xử lý thành đoạn speech.
Không phụ thuộc vadcmp. Cũng đọc file score có sẵn (vd. output của lib.so).

Bước 6 – gộp cửa sổ -> hop ("rescore per hop"):
  mean      trung bình các cửa sổ phủ hop
  max       lớn nhất
  triangle  trung bình có trọng số tam giác: trọng số = 1 − |tâm hop − tâm cửa sổ| / (độ dài cửa sổ / 2)
  rồi làm mượt trung bình trượt `smooth_hops` hop (đối xứng, đầu/cuối file chia cho số hop có thật).
  Mặc định triangle + 5 hop = cấu hình A1–A4 trong file tổng hợp YAMNet (sheet Snapshot).

Bước 7 – hậu xử lý (đơn vị giây):
  y0_segments  score hop ≥ ngưỡng -> đoạn
  y1_segments  lib hiện tại: pad trước / sau -> drop -> merge (thứ tự tuỳ chọn)
  y2_segments  máy trạng thái kiểu FireRed: hysteresis (ngưỡng ra), đoạn thô phải dài hơn min_speech,
               lặng dài hơn min_silence mới cắt, hangover bật / tắt, pad sau cùng (cho phép âm)
"""
from __future__ import annotations

import os
import re

import numpy as np


# =========================================================================== đọc score có sẵn
_SPLIT = re.compile(r"[\s,;]+")


def read_window_file(path):
    """-> starts, ends, scores (giây). Dòng: `index, start, end, score` hoặc `start end score`; bỏ header."""
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            toks = [t for t in _SPLIT.split(line.strip()) if t]
            try:
                nums = [float(t) for t in toks]
            except ValueError:
                continue
            if len(nums) >= 3:
                rows.append(nums[-3:])
    a = np.array(rows, float).reshape(-1, 3)
    return a[:, 0], a[:, 1], a[:, 2]


def index_score_files(root):
    idx = {}
    for dp, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".txt", ".csv")):
                idx.setdefault(os.path.splitext(f)[0].strip().lower(), []).append(os.path.join(dp, f))
    return idx


def find_score_file(item, root, pattern=None, index=None):
    """pattern (tuỳ chọn) dùng {stem} {category} {Category} {index}, vd. '{Category}_{index}.txt';
    không có pattern thì tìm file cùng tên với file audio dưới root."""
    stem = os.path.splitext(os.path.basename(item["audio"]))[0]
    if pattern:
        m = re.search(r"(\d+)$", stem)
        cat = item.get("category", "") or ""
        p = pattern.format(stem=stem, category=cat, Category=cat[:1].upper() + cat[1:],
                           index=m.group(1) if m else "")
        p = p if os.path.isabs(p) else os.path.join(root, p)
        return p if os.path.exists(p) else None
    hits = (index or {}).get(stem.strip().lower(), [])
    return hits[0] if len(hits) == 1 else None


# =========================================================================== bước 6: cửa sổ -> hop
def rescore(starts, window_s, scores, hop_s, duration, method="triangle", smooth_hops=5):
    """Score cửa sổ -> score mỗi hop [ceil(duration / hop)]. Hop không có cửa sổ nào phủ = 0."""
    n_hops = max(1, int(np.ceil(duration / hop_s - 1e-9)))
    span = max(1, int(round(window_s / hop_s)))
    first = np.round(np.asarray(starts) / hop_s).astype(int)
    acc, wsum = np.zeros(n_hops), np.zeros(n_hops)
    out = np.full(n_hops, -np.inf) if method == "max" else None
    offs = np.arange(span)
    if method == "triangle":
        w = 1 - np.abs((offs + 0.5) * hop_s - window_s / 2) / (window_s / 2)
        w = np.clip(w, 1e-6, None)
    else:
        w = np.ones(span)
    for i0, s in zip(first, scores):
        a, b = max(i0, 0), min(i0 + span, n_hops)
        if a >= b:
            continue
        ww = w[a - i0:b - i0]
        if method == "max":
            out[a:b] = np.maximum(out[a:b], s)
        else:
            acc[a:b] += ww * s
        wsum[a:b] += ww
    if method == "max":
        hops = np.where(wsum > 0, out, 0.0)
    elif method in ("mean", "triangle"):
        hops = np.where(wsum > 0, acc / np.maximum(wsum, 1e-12), 0.0)
    else:
        raise ValueError(f"rescore lạ: {method}")
    if smooth_hops and smooth_hops > 1:
        k = int(smooth_hops)
        c = np.cumsum(np.r_[0.0, hops])
        i = np.arange(n_hops)
        lo, hi = np.maximum(0, i - (k - 1) // 2), np.minimum(n_hops, i + k // 2 + 1)
        hops = (c[hi] - c[lo]) / (hi - lo)
    return hops


# =========================================================================== bước 7: hậu xử lý
def hop_segments(on, hop_s, dur):
    """Hop đang bật -> [(start_s, end_s)]; hop i phủ [i·hop, min((i+1)·hop, dur))."""
    d = np.diff(np.r_[0, np.asarray(on).astype(np.int8), 0])
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(a * hop_s, min(b * hop_s, dur)) for a, b in zip(s, e) if a * hop_s < dur]


def merge_segs(segs, gap=0.0):
    """Hợp các đoạn; nối đoạn cách nhau < gap giây."""
    out = []
    for a, b in sorted(segs):
        if out and (a <= out[-1][1] or a - out[-1][1] < gap):
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def y0_segments(hops, thr, hop_s, dur):
    return hop_segments(hops >= thr, hop_s, dur)


def y1_segments(hops, thr, hop_s, dur, pad_before=0.10, pad_after=0.12, drop=0.0, merge=0.5,
                order=("pad", "drop", "merge")):
    segs = y0_segments(hops, thr, hop_s, dur)
    for step in order:
        if not segs:
            break
        if step == "pad":
            segs = merge_segs([(max(0.0, a - pad_before), min(dur, b + pad_after)) for a, b in segs])
        elif step == "drop":
            segs = [(a, b) for a, b in segs if b - a >= drop - 1e-9]
        elif step == "merge":
            segs = merge_segs(segs, merge)
        else:
            raise ValueError(f"bước hậu xử lý lạ: {step}")
    return segs


def hysteresis(p, thr_on, thr_off=None):
    """Vào speech khi p >= thr_on; đang speech chỉ ra khi p < thr_off; ở giữa giữ trạng thái cũ."""
    p = np.asarray(p, dtype=np.float64)
    if thr_off is None or thr_off >= thr_on:
        return (p >= thr_on).astype(np.int8)
    val = np.full(len(p) + 1, -1, np.int8)
    val[0] = 0
    val[1:][p >= thr_on] = 1
    val[1:][p < thr_off] = 0
    idx = np.where(val >= 0, np.arange(len(val)), 0)
    return val[np.maximum.accumulate(idx)][1:].astype(np.int8)


def state_machine(b, min_sp, min_sil):
    """Máy trạng thái FireRed theo run-length. -> (dec, tail); tail = frame 'có thể dừng' (hangover)."""
    n = len(b)
    dec, tail = np.zeros(n, np.int8), np.zeros(n, np.int8)
    if n == 0:
        return dec, tail
    idx = np.flatnonzero(np.diff(b)) + 1
    speaking = False
    for s, e in zip(np.r_[0, idx], np.r_[idx, n]):
        L = e - s
        if not speaking:
            if b[s] == 1 and L > min_sp:
                dec[s:e] = 1
                speaking = True
        elif b[s] == 1:
            dec[s:e] = 1
        elif L > min_sil:
            dec[s:s + min_sil] = 1
            tail[s:s + min_sil] = 1
            speaking = False
        else:
            dec[s:e] = 1
            if e == n:
                tail[s:e] = 1
    return dec, tail


def y2_segments(hops, thr, hop_s, dur, thr_off=None, min_speech=0.1, min_silence=0.16, hangover=True,
                pad_before=0.0, pad_after=-0.08):
    """Máy trạng thái kiểu FireRed trên score hop. Số frame = floor(giây / hop); đoạn phải DÀI HƠN số đó."""
    b = hysteresis(hops, thr, thr_off)
    frames = lambda sec: int(np.floor(float(sec) / hop_s + 1e-6))  # noqa: E731
    dec, tail = state_machine(b, frames(min_speech), frames(min_silence))
    if not hangover:
        dec = dec & ~(tail.astype(bool) & (b == 0))
    segs = hop_segments(dec, hop_s, dur)
    if segs and (pad_before or pad_after):
        segs = [(max(0.0, a - pad_before), min(dur, b_ + pad_after)) for a, b_ in segs]
        segs = merge_segs([(a, b_) for a, b_ in segs if b_ > a])
    return segs
