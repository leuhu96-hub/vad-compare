"""Score cached model outputs against the ground-truth grid."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from vadcmp.config import model_postprocess, resolve
from vadcmp.dataset import read_manifest
from vadcmp.groundtruth import boundary_mask, gt_grid, parse_groundtruth
from vadcmp.runner import load_cached, selected_models
from vadcmp.timeline import bin_scores, binarize, postprocess, segments_to_bins


def _safe_div(a, b):
    return float(a) / float(b) if b else float("nan")


def binary_metrics(label, dec, boundary=None) -> dict:
    label = np.asarray(label).astype(bool)
    dec = np.asarray(dec).astype(bool)
    tp = int((label & dec).sum()); fp = int((~label & dec).sum())
    tn = int((~label & ~dec).sum()); fn = int((label & ~dec).sum())
    p, r = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    out = {
        "precision": p, "recall": r,
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "miss_rate": _safe_div(fn, tp + fn),        # FNR = 1 - recall: speech bins missed
        "far": _safe_div(fp, fp + tn),              # false alarm rate = 1 - specificity
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
    if boundary is not None:
        boundary = np.asarray(boundary).astype(bool)
        err = label != dec
        out["err_boundary"] = int((err & boundary).sum())
        out["err_fp_true"] = int((err & ~boundary & dec).sum())
        out["err_fn_true"] = int((err & ~boundary & ~dec).sum())
        keep = ~boundary
        l2, d2 = label[keep], dec[keep]
        tp2 = int((l2 & d2).sum()); fp2 = int((~l2 & d2).sum()); fn2 = int((l2 & ~d2).sum())
        tn2 = int((~l2 & ~d2).sum())
        out["f1_excl_boundary"] = _safe_div(2 * tp2, 2 * tp2 + fp2 + fn2)
        out["miss_rate_excl_boundary"] = _safe_div(fn2, tp2 + fn2)
        out["far_excl_boundary"] = _safe_div(fp2, fp2 + tn2)
    return out


def score_metrics(label, score) -> dict:
    label = np.asarray(label).astype(int)
    score = np.asarray(score, dtype=float)
    if len(np.unique(label)) < 2:
        return {"auc": float("nan"), "ap": float("nan"), "youden_thr": float("nan"),
                "youden_tpr": float("nan"), "youden_fpr": float("nan"), "f1_at_youden": float("nan"),
                "miss_rate_at_youden": float("nan"), "far_at_youden": float("nan")}
    fpr, tpr, thr = roc_curve(label, score)
    j = int(np.argmax(tpr - fpr))
    t = float(min(thr[j], 1.0))
    return {
        "auc": float(roc_auc_score(label, score)),
        "ap": float(average_precision_score(label, score)),
        "youden_thr": t, "youden_tpr": float(tpr[j]), "youden_fpr": float(fpr[j]),
        "miss_rate_at_youden": float(1 - tpr[j]), "far_at_youden": float(fpr[j]),
        "f1_at_youden": binary_metrics(label, score >= t)["f1"],
    }


def collect_bins(cfg: dict, only=None, log=print):
    """One row per (model, file, GT bin). Uses only files every selected model has."""
    items = read_manifest(resolve(cfg, cfg["data"]["manifest"]))
    mcfgs = selected_models(cfg, only)
    sc = cfg["scoring"]
    bin_size = float(cfg["data"]["gt_bin"])

    cached = {}
    for mc in mcfgs:
        for it in items:
            r = load_cached(cfg, mc, it)
            if r is not None:
                cached[(mc["name"], it["id"])] = r
    have = [mc for mc in mcfgs if any((mc["name"], it["id"]) in cached for it in items)]
    load_err = {}
    lp = os.path.join(resolve(cfg, cfg["output_dir"]), "load_times.json")
    if os.path.exists(lp):
        load_err = {k: v.get("error", "") for k, v in json.load(open(lp)).items()}
    for mc in mcfgs:
        if mc not in have:
            why = load_err.get(mc["name"]) or "chưa có cache - chạy `infer` trước"
            log(f"[eval] bỏ qua {mc['name']}: {why}")
    common = [it for it in items if all((mc["name"], it["id"]) in cached for mc in have)]
    if len(common) < len(items):
        log(f"[eval] using {len(common)}/{len(items)} files that every model processed")

    rows, speed = [], []
    gt_cache = {}
    for it in common:
        gt = gt_cache.get(it["gt"])
        if gt is None:
            gt = gt_cache[it["gt"]] = parse_groundtruth(it["gt"], bin_size)
        for mc in have:
            fs, timing = cached[(mc["name"], it["id"])]
            dur = timing["duration"]
            edges, labels = gt_grid(gt, bin_size, dur)
            bnd = boundary_mask(labels, int(sc["boundary_tolerance"]))
            fs_s = fs.smoothed(float(mc.get("smooth_s", 0.0)))
            thr = float(mc["threshold"])
            scores = bin_scores(fs_s, edges, sc["bin_agg"])
            seg = binarize(fs_s, thr)
            pp = model_postprocess(cfg, mc)
            if pp:
                seg = postprocess(seg, dur, pp["pad_before"], pp["pad_after"], pp["merge_gap"],
                                  pp["min_speech"], pp.get("order", ("pad", "merge", "drop")))
            dec_pp = segments_to_bins(seg, edges, float(sc["min_overlap"]))
            keep = labels >= 0
            n = int(keep.sum())
            rows.append(pd.DataFrame({
                "model": mc["name"], "id": it["id"], "category": it["category"],
                "bin": np.arange(len(labels))[keep], "t0": edges[:-1][keep],
                "label": labels[keep], "score": scores[keep],
                "dec_raw": (scores[keep] >= thr).astype(np.int8), "dec_pp": dec_pp[keep],
                "boundary": bnd[keep],
            }))
            speed.append({"model": mc["name"], "id": it["id"], "category": it["category"],
                          "duration": dur, "wall": timing["wall"], "cpu": timing["cpu"], "n_bins": n})
    if not rows:
        raise SystemExit("Nothing to evaluate - run `python -m vadcmp infer -c <config>` first.")
    order = [mc["name"] for mc in have]
    return pd.concat(rows, ignore_index=True), pd.DataFrame(speed), order, common



def summarize(bins: pd.DataFrame, speed: pd.DataFrame, order, cfg: dict):
    def group_metrics(df):
        d = {"n_bins": len(df), "speech_ratio": float(df["label"].mean())}
        d.update(score_metrics(df["label"], df["score"]))
        d.update(binary_metrics(df["label"], df["dec_pp"], df["boundary"]))
        raw = binary_metrics(df["label"], df["dec_raw"])
        d["f1_raw"], d["miss_rate_raw"], d["far_raw"] = raw["f1"], raw["miss_rate"], raw["far"]
        return d

    overall = []
    for m in order:
        df = bins[bins.model == m]
        d = {"model": m, "n_files": df["id"].nunique()}
        d.update(group_metrics(df))
        overall.append(d)
    overall = pd.DataFrame(overall)

    bycat = []
    for (m, c), df in bins.groupby(["model", "category"], sort=False):
        d = {"model": m, "category": c, "n_files": df["id"].nunique()}
        d.update(group_metrics(df))
        bycat.append(d)
    bycat = pd.DataFrame(bycat)

    perfile = []
    for (m, i), df in bins.groupby(["model", "id"], sort=False):
        d = {"model": m, "id": i, "category": df["category"].iloc[0], "n_bins": len(df)}
        bm = binary_metrics(df["label"], df["dec_pp"], df["boundary"])
        d.update({k: bm[k] for k in ("f1", "precision", "recall", "specificity", "miss_rate", "far",
                                     "accuracy", "err_boundary", "err_fp_true", "err_fn_true")})
        d["auc"] = score_metrics(df["label"], df["score"])["auc"]
        perfile.append(d)
    perfile = pd.DataFrame(perfile)

    load = {}
    lp = os.path.join(resolve(cfg, cfg["output_dir"]), "load_times.json")
    if os.path.exists(lp):
        load = json.load(open(lp))
    sp = []
    for m in order:
        s = speed[speed.model == m]
        tot = s["duration"].sum()
        wall = s["wall"].sum(min_count=1)   # NaN when speed was not measured (precomputed)
        cpu = s["cpu"].sum(min_count=1)
        per = s["wall"] / s["duration"]
        sp.append({
            "model": m, "audio_s": tot,
            "rtf": wall / tot if tot else float("nan"),
            "rtf_median": float(per.median()), "rtf_p95": float(per.quantile(0.95)),
            "cpu_rtf": cpu / tot if tot else float("nan"),
            "x_realtime": tot / wall if wall and wall > 0 else float("nan"),
            "load_s": load.get(m, {}).get("load_s", float("nan")),
        })
    speed_sum = pd.DataFrame(sp)
    overall = overall.merge(speed_sum[["model", "rtf", "x_realtime"]], on="model", how="left")
    return overall, bycat, perfile, speed_sum


def roc_points(bins: pd.DataFrame, order, category=None):
    out = {}
    for m in order:
        df = bins[bins.model == m]
        if category is not None:
            df = df[df.category == category]
        if df["label"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(df["label"], df["score"])
        out[m] = (fpr, tpr, roc_auc_score(df["label"], df["score"]))
    return out
