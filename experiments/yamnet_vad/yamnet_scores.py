"""Score YAMNet (cửa sổ dài, nhảy hop ngắn) -> score theo hop, giống phần đầu pipeline của lib.so.

Hai nguồn score cửa sổ:
  * model     : chạy file .tflite / .onnx của bạn (dùng lại vadcmp/models/yamnet.py: đọc input shape,
                softmax / logits, speech_index, model lượng tử hoá …)
  * precomputed: file kết quả có sẵn (vd. output của lib.so), mỗi dòng `index, start, end, score`
                (hoặc `start end score`), một file cho mỗi file audio

Rồi gộp các cửa sổ chồng nhau thành score mỗi hop ("rescore per hop"):
  mean      trung bình các cửa sổ phủ hop
  max       lớn nhất
  triangle  trung bình có trọng số tam giác theo khoảng cách từ tâm hop tới tâm cửa sổ
            (trọng số = 1 − |tâm hop − tâm cửa sổ| / (độ dài cửa sổ / 2))
và làm mượt trung bình trượt `smooth_hops` hop (đối xứng, đầu/cuối file chia cho số hop có thật).
Mặc định triangle + 5 hop = cấu hình A1–A4 ghi trong file tổng hợp YAMNet (sheet Snapshot).
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
SR = 16000


# =========================================================================== nguồn 1: model
class YamnetModel:
    """Bọc vadcmp.models.yamnet.YamnetVAD để lấy score TỪNG CỬA SỔ (chưa gộp)."""

    def __init__(self, model_path, hop_s=0.08, speech_index=1, output="auto", output_index=0,
                 input_samples=None, threads=1, tail="drop"):
        sys.path.insert(0, ROOT)
        from vadcmp.models.yamnet import YamnetVAD

        cfg = {"model_path": os.path.abspath(model_path), "hop_s": hop_s, "speech_index": speech_index,
               "output": output, "output_index": output_index}
        if input_samples:
            cfg["input_samples"] = int(input_samples)
        self.m = YamnetVAD("yamnet", cfg, num_threads=threads, base_dir=os.getcwd())
        self.m.load()
        self.hop = int(round(hop_s * SR))
        self.n_in = self.m.n_in
        self.window_s = self.n_in / SR
        self.tail = tail
        self.model_path = model_path

    def describe(self):
        return (f"{self.model_path} [{self.m.backend}, input {self.m.input_shape}, cửa sổ {self.n_in} mẫu = "
                f"{self.window_s:.3f} s, hop {self.hop / SR:g} s, speech_index {self.m.speech_index}]")

    def window_scores(self, x16):
        """-> starts_s, score. tail=drop: chỉ cửa sổ đủ mẫu (giống lib: 'bỏ phần dư');
        tail=pad: thêm một cửa sổ đệm 0 để phủ phần cuối."""
        n = len(x16)
        if n >= self.n_in:
            n_win = 1 + (n - self.n_in) // self.hop
        else:
            n_win = 0
        x = x16
        if self.tail == "pad" and (n_win == 0 or (n_win - 1) * self.hop + self.n_in < n):
            n_win += 1
            x = np.pad(x16, (0, max(0, (n_win - 1) * self.hop + self.n_in - n)))
        scores = np.empty(n_win)
        for i in range(n_win):
            scores[i] = self.m._to_prob(self.m._infer(x[i * self.hop:i * self.hop + self.n_in]))
        return np.arange(n_win) * self.hop / SR, scores


# =========================================================================== nguồn 2: file có sẵn
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


# =========================================================================== gộp cửa sổ -> hop
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
