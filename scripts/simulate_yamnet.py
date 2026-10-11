"""MÔ PHỎNG score kiểu YAMNet (cửa sổ 0.96 s, hop 0.08 s) từ prob 10 ms của một model khác đã cache.

Dùng để thử `tune` / so sánh post-processing khi chưa có model YAMNet thật trong repo.
KHÔNG phải YAMNet: chỉ bắt chước hình dạng score của 1 classifier cửa sổ dài:
    frac  = trung bình prob nguồn trong cửa sổ 0.96 s       (cửa sổ dài -> "nhoè" quanh biên)
    score = sigmoid(k * (frac - c)) + nhiễu N(0, sigma)     (cửa sổ có ~c speech đã bị gọi là speech)
Ghi ra định dạng precomputed `index,start,end,score` (giống output lib.so), đọc bằng model
`type: precomputed` -> rescore per hop (mean 12 cửa sổ) y như pipeline thật.

    python scripts/simulate_yamnet.py -c config.demo.yaml --source firered -o data/demo/yamnet_sim
"""
from __future__ import annotations

import argparse
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from vadcmp.config import load_config, resolve  # noqa: E402
from vadcmp.dataset import read_manifest  # noqa: E402
from vadcmp.runner import load_cached  # noqa: E402


def simulate(fs, dur, window=0.96, hop=0.08, k=10.0, c=0.35, sigma=0.07, seed=0):
    rng = np.random.default_rng(seed)
    t = np.r_[fs.start, fs.end[-1:]]
    cum = np.r_[0.0, np.cumsum(fs.prob * (fs.end - fs.start))]
    n = max(1, int(np.ceil(max(dur - window, 0) / hop)) + 1)
    starts = np.arange(n) * hop
    a = np.clip(starts, 0, dur)
    b = np.clip(starts + window, 0, dur)
    frac = (np.interp(b, t, cum) - np.interp(a, t, cum)) / window   # phần thiếu (cuối file) = im lặng
    score = 1 / (1 + np.exp(-k * (frac - c))) + rng.normal(0, sigma, n)
    return starts, starts + window, np.clip(score, 0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default="config.demo.yaml")
    ap.add_argument("--source", default="firered", help="model nguồn (đã có cache)")
    ap.add_argument("-o", "--out", default="data/demo/yamnet_sim")
    ap.add_argument("--k", type=float, default=10.0)
    ap.add_argument("--c", type=float, default=0.35)
    ap.add_argument("--sigma", type=float, default=0.07)
    a = ap.parse_args()
    cfg = load_config(a.config)
    mc = next(m for m in cfg["models"] if m["name"] == a.source)
    os.makedirs(a.out, exist_ok=True)
    n = 0
    for it in read_manifest(resolve(cfg, cfg["data"]["manifest"])):
        r = load_cached(cfg, mc, it)
        if r is None:
            continue
        fs, timing = r
        stem = os.path.splitext(os.path.basename(it["audio"]))[0]
        s, e, sc = simulate(fs, timing["duration"], k=a.k, c=a.c, sigma=a.sigma,
                            seed=zlib.crc32(stem.encode()))
        with open(os.path.join(a.out, stem + ".txt"), "w") as f:
            for i, (x, y, z) in enumerate(zip(s, e, sc)):
                f.write(f"{i},{x:.2f},{y:.2f},{z:.4f}\n")
        n += 1
    print(f"{n} file -> {a.out}")


if __name__ == "__main__":
    main()
