"""Command line:

    python -m vadcmp manifest --audio-root data/audio --gt-root data/Groundtruth -o data/manifest.csv
    python -m vadcmp infer    -c config.yaml [--models silero,ten] [--force] [--limit 20]
    python -m vadcmp evaluate -c config.yaml [--models ...]      # metrics + report from cache
    python -m vadcmp run      -c config.yaml                     # infer + evaluate
"""
from __future__ import annotations

import argparse
import os
import sys

from vadcmp.config import load_config, resolve
from vadcmp.dataset import read_manifest, scan, write_manifest


def _models(s):
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


def cmd_manifest(a):
    rows, unmatched, info = scan(a.audio_root, a.gt_root, a.gt_dir)
    write_manifest(rows, a.output)
    cats = sorted({r["category"] for r in rows})
    print(f"GT: {info['n_gt']} file trong '{a.gt_dir}/'", end="")
    ign = info["ignored"]
    if ign:
        print(f"; bỏ qua {sum(ign.values())} file .txt ở {len(ign)} folder khác "
              f"(vd. {', '.join(sorted(ign)[:4])}{' ...' if len(ign) > 4 else ''})")
    else:
        print()
    print(f"{len(rows)} cặp audio↔GT, {len(cats)} category: {', '.join(cats) or '-'} -> {a.output}")
    if unmatched:
        print(f"{len(unmatched)} file audio không ghép được (0 hoặc >1 GT trùng tên):")
        for p, n in unmatched[:20]:
            print(f"   [{n} match] {p}")
    if info["unused_gt"]:
        print(f"{len(info['unused_gt'])} file GT không có audio tương ứng, vd. {info['unused_gt'][0]}")


def cmd_infer(a):
    from vadcmp.runner import run_inference

    cfg = load_config(a.config)
    n_err = run_inference(cfg, _models(a.models), a.force, a.limit)
    if n_err:
        print(f"{n_err} lỗi trong lúc chạy - xem log phía trên (VADCMP_DEBUG=1 để in traceback)")


def cmd_evaluate(a):
    from vadcmp.metrics import collect_bins, summarize
    from vadcmp.report import write_report

    cfg = load_config(a.config)
    bins, speed, order, items = collect_bins(cfg, _models(a.models))
    overall, bycat, perfile, speed_sum = summarize(bins, speed, order, cfg)
    path = write_report(cfg, bins, overall, bycat, perfile, speed_sum, order, items)
    cols = ["model", "n_files", "auc", "f1", "f1_excl_boundary", "precision", "recall",
            "miss_rate", "far", "rtf"]
    with __import__("pandas").option_context("display.width", 200, "display.precision", 4):
        print(overall[cols].sort_values("auc", ascending=False).to_string(index=False))
    print(f"\nReport: {path}")


def cmd_run(a):
    cmd_infer(a)
    cmd_evaluate(a)


def main(argv=None):
    p = argparse.ArgumentParser(prog="vadcmp", description="So sánh các model VAD")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("manifest", help="ghép audio với ground truth -> manifest.csv")
    m.add_argument("--audio-root", required=True)
    m.add_argument("--gt-root", required=True)
    m.add_argument("--gt-dir", default="ground_truth",
                   help="chỉ đọc GT trong thư mục có tên này (mặc định: ground_truth)")
    m.add_argument("-o", "--output", default="data/manifest.csv")
    m.set_defaults(func=cmd_manifest)

    for name, fn, h in (("infer", cmd_infer, "chạy model, lưu cache"),
                        ("evaluate", cmd_evaluate, "tính metric + report từ cache"),
                        ("run", cmd_run, "infer + evaluate")):
        s = sub.add_parser(name, help=h)
        s.add_argument("-c", "--config", default="config.yaml")
        s.add_argument("--models", help="danh sách tên model, cách nhau dấu phẩy")
        if name != "evaluate":
            s.add_argument("--force", action="store_true", help="bỏ qua cache, chạy lại")
            s.add_argument("--limit", type=int, help="chỉ chạy N file đầu (thử nhanh)")
        s.set_defaults(func=fn)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
