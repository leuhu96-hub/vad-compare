"""Dò tham số post-processing trên score đã cache (không chạy lại model).

So sánh cấu hình hiện tại của model (thường là `rules`: pad -> merge -> drop) với lưới
cấu hình `fsm` kiểu FireRed (vadcmp/decide.py), chấm trên đúng lưới bin GT như `evaluate`.

    python -m vadcmp tune -c config.yaml --model yamnet [--objective far] [--holdout 0.3]

Mục tiêu:
    far : FAR nhỏ nhất với điều kiện miss <= miss của baseline + --max-miss-increase
    f1  : F1 lớn nhất
Kết quả: <output_dir>/tune_<model>.csv (mọi cấu hình) + in top cấu hình và đoạn YAML để dán vào config.
--holdout 0.3: chọn cấu hình trên 70% file, báo cáo lại trên 30% còn lại (tránh "học thuộc" dataset).
"""
from __future__ import annotations

import hashlib
import itertools
import os
import time

import numpy as np
import pandas as pd

from vadcmp.config import model_postprocess, resolve
from vadcmp.dataset import read_manifest
from vadcmp.decide import decide, describe, fsm_prepare, fsm_segments, hysteresis
from vadcmp.groundtruth import boundary_mask, gt_grid, parse_groundtruth
from vadcmp.runner import load_cached
from vadcmp.timeline import segments_to_bins

DEFAULT_GRID = {
    "thr": "-0.1,-0.05,0,0.05,0.1",     # cộng vào threshold hiện tại của model
    "thr_off": "none,0.1,0.2",          # thr_off = thr - giá trị này (none = không hysteresis)
    "min_speech": "0.1,0.2,0.3",
    "min_silence": "0.16,0.32,0.48",
    "hangover": "1,0",
    "pad_before": "0,0.08",
    "pad_after": "0,0.08",
    "smooth_s": "0",
}


def _floats(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _is_test(item_id: str, frac: float) -> bool:
    if frac <= 0:
        return False
    h = int(hashlib.md5(item_id.encode()).hexdigest()[:8], 16) % 1000
    return h < frac * 1000


class _File:
    __slots__ = ("id", "category", "fs", "dur", "edges", "labels", "keep", "bnd", "test", "prep")


def load_files(cfg, mc, gt_bin, holdout, log=print):
    items = read_manifest(resolve(cfg, cfg["data"]["manifest"]))
    tol = int(cfg["scoring"]["boundary_tolerance"])
    out = []
    for it in items:
        r = load_cached(cfg, mc, it)
        if r is None:
            continue
        fs, timing = r
        f = _File()
        f.id, f.category = it["id"], it["category"]
        f.fs = fs.smoothed(float(mc.get("smooth_s", 0.0)))
        f.dur = timing["duration"]
        gt = parse_groundtruth(it["gt"], gt_bin)
        f.edges, labels = gt_grid(gt, gt_bin, f.dur)
        f.keep = labels >= 0
        f.labels = labels[f.keep].astype(np.int8)
        f.bnd = boundary_mask(labels, tol)[f.keep]
        f.test = _is_test(it["id"], holdout)
        f.prep = {}
        out.append(f)
    if not out:
        raise SystemExit(f"Không có cache cho {mc['name']} - chạy `python -m vadcmp infer` trước.")
    log(f"[tune] {mc['name']}: {len(out)}/{len(items)} file có cache"
        + (f", {sum(f.test for f in out)} file để kiểm tra (holdout)" if holdout > 0 else ""))
    return out


def _counts(f: _File, seg, min_overlap):
    d = segments_to_bins(seg, f.edges, min_overlap)[f.keep]
    y = f.labels
    tp = int(((d == 1) & (y == 1)).sum())
    fp = int(((d == 1) & (y == 0)).sum())
    fn = int(((d == 0) & (y == 1)).sum())
    tn = int(((d == 0) & (y == 0)).sum())
    fp_b = int(((d == 1) & (y == 0) & f.bnd).sum())
    fn_b = int(((d == 0) & (y == 1) & f.bnd).sum())
    neg_core = int(((y == 0) & ~f.bnd).sum())
    return np.array([tp, fp, fn, tn, fp_b, fn_b, neg_core])


def _metrics(c, prefix=""):
    tp, fp, fn, tn, fp_b, fn_b, neg_core = (int(x) for x in c)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        prefix + "f1": 2 * p * r / (p + r) if p + r else 0.0,
        prefix + "miss": fn / (tp + fn) if tp + fn else 0.0,
        prefix + "far": fp / (fp + tn) if fp + tn else 0.0,
        # FAR chỉ tính bin không-speech nằm xa điểm chuyển GT ("báo nhầm thật", không phải lệch biên)
        prefix + "far_core": (fp - fp_b) / neg_core if neg_core else 0.0,
        prefix + "fp_boundary_share": fp_b / fp if fp else 0.0,
    }


def _eval_segments(files, seg_fn, min_overlap):
    tr = np.zeros(7, np.int64)
    te = np.zeros(7, np.int64)
    for f in files:
        c = _counts(f, seg_fn(f), min_overlap)
        if f.test:
            te += c
        else:
            tr += c
    return tr, te


def tune(cfg, model_name, grid=None, objective="far", holdout=0.0, max_miss_increase=0.0,
         gt_bin=None, top=8, log=print):
    mcs = [m for m in cfg["models"] if m["name"] == model_name]
    if not mcs:
        raise SystemExit(f"Không có model tên {model_name!r} trong config")
    mc = mcs[0]
    g = {**DEFAULT_GRID, **{k: v for k, v in (grid or {}).items() if v is not None}}
    gt_bin = float(gt_bin or cfg["data"]["gt_bin"])
    min_overlap = float(cfg["scoring"]["min_overlap"])
    files = load_files(cfg, mc, gt_bin, holdout, log)
    has_test = any(f.test for f in files)
    thr0 = float(mc["threshold"])
    pp0 = model_postprocess(cfg, mc)

    rows = []

    def add(name, params, tr, te):
        row = {"config": name, **params, **_metrics(tr)}
        if has_test:
            row.update(_metrics(te, "test_"))
        rows.append(row)

    # baseline = đúng như `evaluate` đang làm
    tr, te = _eval_segments(files, lambda f: decide(f.fs, f.dur, thr0, pp0), min_overlap)
    add("baseline", {"method": (pp0 or {}).get("method", "rules") if pp0 else "none",
                     "thr": thr0, "desc": describe(pp0)}, tr, te)
    # baseline đổi thứ tự: drop/merge trên đoạn thô, pad sau cùng (cùng tham số)
    if pp0 and pp0.get("method", "rules") == "rules":
        pp_re = {**pp0, "order": ["drop", "merge", "pad"]}
        tr, te = _eval_segments(files, lambda f: decide(f.fs, f.dur, thr0, pp_re), min_overlap)
        add("baseline_reordered", {"method": "rules", "thr": thr0, "desc": describe(pp_re)}, tr, te)

    thrs = sorted({round(min(max(thr0 + float(d), 0.01), 0.99), 4) for d in _floats(g["thr"])})
    offs = [None if o.lower() == "none" else float(o) for o in _floats(g["thr_off"])]
    combos = list(itertools.product(
        [float(x) for x in _floats(g["smooth_s"])], thrs, offs,
        [float(x) for x in _floats(g["min_speech"])], [float(x) for x in _floats(g["min_silence"])],
        [x.lower() in ("1", "true", "yes") for x in _floats(g["hangover"])],
        [float(x) for x in _floats(g["pad_before"])], [float(x) for x in _floats(g["pad_after"])]))
    log(f"[tune] baseline: {describe(pp0)} @ thr {thr0}; thử {len(combos)} cấu hình fsm ...")
    t0 = time.time()
    for i, (sm, thr, off, msp, msil, ho, pb, pa) in enumerate(combos):
        thr_off = None if off is None else round(thr - off, 4)
        cfg_fsm = {"method": "fsm", "smooth_s": sm, "thr_off": thr_off, "min_speech": msp,
                   "min_silence": msil, "hangover": ho, "pad_before": pb, "pad_after": pa,
                   "max_speech": 0.0}

        def seg_fn(f, sm=sm, thr=thr, thr_off=thr_off, c=cfg_fsm):
            key = ("prep", sm)
            if key not in f.prep:
                f.prep[key] = fsm_prepare(f.fs, sm)
            fs, hop, w, p = f.prep[key]
            kb = ("b", sm, thr, thr_off)
            if kb not in f.prep:
                f.prep[kb] = hysteresis(p, thr, thr_off)
            return fsm_segments(fs, p, f.prep[kb], f.dur, hop, w, c)

        tr, te = _eval_segments(files, seg_fn, min_overlap)
        add(f"fsm_{i:04d}", {"method": "fsm", "thr": thr, "thr_off": thr_off, "smooth_s": sm,
                             "min_speech": msp, "min_silence": msil, "hangover": ho,
                             "pad_before": pb, "pad_after": pa}, tr, te)
        if (i + 1) % 200 == 0:
            log(f"[tune]   {i + 1}/{len(combos)} ({time.time() - t0:.0f} s)")

    df = pd.DataFrame(rows)
    base = df.iloc[0]
    if objective == "far":
        ok = df["miss"] <= base["miss"] + max_miss_increase + 1e-12
        ranked = df[ok].sort_values(["far", "f1"], ascending=[True, False])
    else:
        ranked = df.sort_values(["f1", "far"], ascending=[False, True])
    df["rank"] = np.nan
    df.loc[ranked.index, "rank"] = np.arange(1, len(ranked) + 1)
    out_dir = resolve(cfg, cfg["output_dir"])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"tune_{model_name}.csv")
    df.sort_values("rank").to_csv(path, index=False)

    show = ["config", "thr", "thr_off", "min_speech", "min_silence", "hangover", "pad_before", "pad_after",
            "f1", "miss", "far", "far_core", "fp_boundary_share"]
    if has_test:
        show += ["test_f1", "test_miss", "test_far"]
    head = pd.concat([df[df.config.str.startswith("baseline")], ranked.head(top)]).drop_duplicates("config")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 4):
        log(head[[c for c in show if c in head]].to_string(index=False))
    best = ranked.iloc[0] if len(ranked) else None
    if best is not None and best["method"] == "fsm":
        log("\nDán vào model trong config.yaml:")
        log(yaml_snippet(best))
    log(f"\n{len(df)} cấu hình -> {path}")
    return df, best


def yaml_snippet(row) -> str:
    off = row["thr_off"]
    off = "null" if off is None or (isinstance(off, float) and np.isnan(off)) else f"{off:g}"
    return (f"    threshold: {row['thr']:g}\n"
            f"    postprocess:\n"
            f"      method: fsm\n"
            f"      thr_off: {off}\n"
            f"      smooth_s: {row['smooth_s']:g}\n"
            f"      min_speech: {row['min_speech']:g}\n"
            f"      min_silence: {row['min_silence']:g}\n"
            f"      hangover: {str(bool(row['hangover'])).lower()}\n"
            f"      pad_before: {row['pad_before']:g}\n"
            f"      pad_after: {row['pad_after']:g}")
