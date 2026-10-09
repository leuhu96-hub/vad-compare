"""Chạy FireRedVAD (ONNX, từng bước) cho cả folder dataset hoặc file lẻ.

Dataset (audio + ground truth, xem dataset_reader.py):
    python run.py --audio-root phase_1 --gt-root Groundtruth -o out
    python run.py --audio-root phase_1 --gt-root Groundtruth --onnx my_vad.onnx --plot

File / folder audio lẻ (không có GT):
    python run.py song.m4a other_folder/ -o out

Output trong -o:
    segments/<quality>/<name>.txt   start end Speech               (segment sau post-processing)
    bins/<quality>/<name>.txt       0 0.5 Speech / 0.5 1 Non-Speech (bin 0.5 s, cùng định dạng GT)
    probs/<quality>/<name>.csv      (--save-probs) frame, start_s, end_s, prob, smoothed, decision
    plots/<quality>/<name>.png      (--plot) waveform, log-mel, CMVN, prob, segment
    steps/<quality>/<name>.npz      (--dump) tensor trung gian bước 1..8
    results.csv                     mỗi file: thông tin input, số segment, metric so với GT, RTF
    summary.csv                     (có GT) metric gộp: tất cả / theo quality / category / quality×category
    dataset_report.txt              (có GT) kết quả ghép audio <-> GT
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset_reader import (AUDIO_EXTS, Item, gt_to_bins, load_dataset,  # noqa: E402
                            read_ground_truth, segments_to_bins, write_bins)
from firered_onnx import FRAME_SHIFT_S, FireRedOnnx, PostConfig  # noqa: E402

METRIC_COLS = ["n_bins", "tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1",
               "miss_rate", "far"]


# ============================================================================ metrics
def confusion(gt_bins, pred_bins):
    n = min(len(gt_bins), len(pred_bins))
    tp = fp = tn = fn = 0
    for g, p in zip(gt_bins[:n], pred_bins[:n]):
        if g < 0:
            continue
        if g == 1:
            tp, fn = tp + (p == 1), fn + (p == 0)
        else:
            fp, tn = fp + (p == 1), tn + (p == 0)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def scores(c):
    def div(a, b):
        return a / b if b else float("nan")
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    return {"n_bins": tp + fp + tn + fn, **c,
            "accuracy": div(tp + tn, tp + fp + tn + fn), "precision": div(tp, tp + fp),
            "recall": div(tp, tp + fn), "f1": div(2 * tp, 2 * tp + fp + fn),
            "miss_rate": div(fn, tp + fn), "far": div(fp, fp + tn)}


def summarize(rows):
    """Gộp TP/FP/TN/FN (micro) theo nhóm."""
    groups = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "files": 0, "audio_s": 0.0})
    for r in rows:
        if r.get("tp") == "":
            continue
        for key in (("all", "", ""), ("quality", r["quality"], ""), ("category", "", r["category"]),
                    ("quality×category", r["quality"], r["category"])):
            g = groups[key]
            for k in ("tp", "fp", "tn", "fn"):
                g[k] += r[k]
            g["files"] += 1
            g["audio_s"] += r["duration_s"]
    order = {"all": 0, "quality": 1, "category": 2, "quality×category": 3}
    out = []
    for (level, q, c), g in sorted(groups.items(), key=lambda kv: (order[kv[0][0]], kv[0][1], kv[0][2])):
        out.append({"group": level, "quality": q, "category": c, "files": g["files"],
                    "audio_s": round(g["audio_s"], 2), **scores(g)})
    return out


# ============================================================================ outputs
def _path(out_dir, kind, quality, name, ext):
    d = os.path.join(out_dir, kind, quality) if quality else os.path.join(out_dir, kind)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name + ext)


def write_probs(path, r):
    t0, t1 = r.frame_times()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "start_s", "end_s", "prob", "smoothed", "decision"])
        for i in range(len(r.probs)):
            w.writerow([i, f"{t0[i]:.3f}", f"{t1[i]:.3f}", f"{r.probs[i]:.6f}",
                        f"{r.smoothed[i]:.6f}", int(r.decisions[i])])


def plot_result(r, path, gt_bins=None, bin_s=0.5):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = r.steps
    fig, ax = plt.subplots(4, 1, figsize=(12, 8.5), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1.4, 1.4, 1.2]})
    if "3_16k" in s:
        ax[0].plot(np.arange(len(s["3_16k"])) / 16000, s["3_16k"], lw=0.4, color="#52514e")
    ax[0].set_title(f"{os.path.basename(r.path)}   input: {r.orig_sr} Hz, {r.channels} kênh  ->  16 kHz mono",
                    loc="left", fontsize=10, fontweight="bold")
    ext = [0, len(r.probs) * FRAME_SHIFT_S, 0, 80]
    if "5g_logmel" in s:
        ax[1].imshow(s["5g_logmel"].T, origin="lower", aspect="auto", extent=ext, cmap="Blues")
        ax[1].set_ylabel("log-mel")
    if "6_cmvn" in s:
        ax[2].imshow(s["6_cmvn"].T, origin="lower", aspect="auto", extent=ext, cmap="RdBu_r", vmin=-3, vmax=3)
        ax[2].set_ylabel("sau CMVN")
    if gt_bins:
        for i, g in enumerate(gt_bins):
            if g == 1:
                ax[3].axvspan(i * bin_s, (i + 1) * bin_s, color="#e3e1db", lw=0)
    t = np.arange(len(r.probs)) * FRAME_SHIFT_S
    ax[3].plot(t, r.probs, lw=0.8, color="#8a8984", label="prob")
    ax[3].plot(t, r.smoothed, lw=1.4, color="#2a78d6", label="smoothed")
    for a, b in r.segments:
        ax[3].axvspan(a, b, ymin=0, ymax=0.06, color="#0b0b0b", lw=0)
    ax[3].set_ylim(-0.05, 1.05)
    ax[3].set_xlabel("giây  (nền xám = GT speech, vạch đen = segment dự đoán)")
    ax[3].legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_csv(path, rows):
    if not rows:
        return
    cols = list(rows[0])
    for r in rows[1:]:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


# ============================================================================ main
def list_audio(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for dp, dirs, files in os.walk(p):
                dirs.sort()
                out += [os.path.join(dp, f) for f in sorted(files) if f.lower().endswith(AUDIO_EXTS)]
        else:
            out.append(p)
    return [Item(os.path.splitext(os.path.basename(p))[0], p, "", "", "") for p in out]


def main():
    ap = argparse.ArgumentParser(description="FireRedVAD ONNX cho folder dataset hoặc file lẻ")
    ap.add_argument("inputs", nargs="*", help="file/folder audio lẻ (khi không dùng --audio-root)")
    ap.add_argument("--audio-root", help="vd. phase_1  (cấu trúc <audio-root>/<quality>/<name>.mp4)")
    ap.add_argument("--gt-root", help="vd. Groundtruth  (cấu trúc <gt-root>/<category>/ground_truth/<name>.txt)")
    ap.add_argument("--gt-dir", default="ground_truth", help="tên thư mục chứa GT")
    ap.add_argument("--bin", type=float, default=0.5, help="độ dài bin GT (giây)")
    ap.add_argument("-o", "--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--model-dir", default=os.path.join(HERE, "models"),
                    help="thư mục chứa .onnx + cmvn.ark (tên chuẩn hoặc tên bất kỳ)")
    ap.add_argument("--onnx", help="file .onnx có sẵn (tên tuỳ ý)")
    ap.add_argument("--cmvn", help="cmvn.ark (mặc định: tìm cạnh file .onnx)")
    ap.add_argument("--mode", default="auto", choices=["auto", "nonstream", "stream", "stream_cached"])
    ap.add_argument("--chunk", type=int, default=10, help="stream_cached: số frame 10 ms mỗi lần gọi")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--downmix", default="mean", help="mean | 0 | 1 ... (chọn kênh)")
    ap.add_argument("--res-type", default="soxr_hq", help="librosa.resample res_type")
    ap.add_argument("--dither", default="auto", help="auto (=1.0 nếu input < 16 kHz) | số | 0")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--smooth", type=int, default=5, help="frame")
    ap.add_argument("--min-speech", type=int, default=20, help="frame")
    ap.add_argument("--min-silence", type=int, default=20, help="frame")
    ap.add_argument("--max-speech", type=int, default=2000, help="frame")
    ap.add_argument("--merge-silence", type=int, default=0, help="frame")
    ap.add_argument("--extend-speech", type=int, default=0, help="frame")
    ap.add_argument("--save-probs", action="store_true", help="lưu prob từng frame")
    ap.add_argument("--plot", action="store_true", help="vẽ hình mỗi file")
    ap.add_argument("--dump", action="store_true", help="lưu tensor trung gian (.npz)")
    ap.add_argument("--limit", type=int, help="chỉ chạy N file đầu (thử nhanh)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    if a.audio_root or a.gt_root:
        if not (a.audio_root and a.gt_root):
            ap.error("cần cả --audio-root và --gt-root")
        ds = load_dataset(a.audio_root, a.gt_root, a.gt_dir)
        report = ds.report()
        print(report + "\n")
        with open(os.path.join(a.out, "dataset_report.txt"), "w", encoding="utf-8") as f:
            f.write(report + "\n")
        items, has_gt = ds.items, True
    elif a.inputs:
        items, has_gt = list_audio(a.inputs), False
    else:
        ap.error("đưa file/folder audio, hoặc --audio-root + --gt-root")
    if a.limit:
        items = items[:a.limit]

    post = PostConfig(a.smooth, a.threshold, a.min_speech, a.max_speech, a.min_silence,
                      a.merge_silence, a.extend_speech)
    vad = FireRedOnnx(a.model_dir, a.mode, a.threads, a.chunk, post, a.downmix, a.res_type, a.dither,
                      onnx_path=a.onnx, cmvn_path=a.cmvn)
    print(f"Model: {vad.onnx_path} ({vad.mode}), CMVN: {vad.cmvn_path}")

    rows, t_all = [], time.perf_counter()
    for k, it in enumerate(items, 1):
        tag = f"[{k}/{len(items)}] {os.path.join(it.quality, os.path.basename(it.audio))}"
        try:
            r = vad.run_file(it.audio, keep_steps=a.plot or a.dump)
        except Exception as e:
            print(f"{tag}: LỖI {type(e).__name__}: {e}")
            rows.append({"name": it.name, "quality": it.quality, "category": it.category,
                         "audio": it.audio, "gt": it.gt, "error": f"{type(e).__name__}: {e}"})
            continue
        q, name = it.quality, it.name
        with open(_path(a.out, "segments", q, name, ".txt"), "w") as f:
            for s, e in r.segments:
                f.write(f"{s:.3f} {e:.3f} Speech\n")
        pred_bins = segments_to_bins(r.segments, r.duration, a.bin)
        write_bins(_path(a.out, "bins", q, name, ".txt"), pred_bins, a.bin, r.duration)
        if a.save_probs:
            write_probs(_path(a.out, "probs", q, name, ".csv"), r)
        if a.dump:
            np.savez_compressed(_path(a.out, "steps", q, name, ".npz"), orig_sr=r.orig_sr, channels=r.channels,
                                **{kk: np.asarray(v) for kk, v in r.steps.items()})

        row = {"name": name, "quality": q, "category": it.category, "audio": it.audio, "gt": it.gt,
               "orig_sr": r.orig_sr, "channels": r.channels, "duration_s": round(r.duration, 3),
               "dither": vad.last_dither, "n_segments": len(r.segments),
               "speech_s": round(sum(e - s for s, e in r.segments), 3)}
        gt_bins = None
        msg = ""
        if has_gt:
            try:
                gt_bins = gt_to_bins(read_ground_truth(it.gt), a.bin)
                sc = scores(confusion(gt_bins, pred_bins))
                row.update(sc)
                row["gt_len_s"] = round(len(gt_bins) * a.bin, 3)
                msg = f", F1 {sc['f1']:.3f}, miss {sc['miss_rate']:.3f}, FAR {sc['far']:.3f}"
                if abs(len(gt_bins) - len(pred_bins)) > 1:
                    msg += f"  (GT {len(gt_bins) * a.bin:.1f} s ≠ audio {r.duration:.1f} s)"
            except Exception as e:
                row["error"] = f"GT: {type(e).__name__}: {e}"
                msg = f", LỖI GT: {e}"
        if a.plot:
            plot_result(r, _path(a.out, "plots", q, name, ".png"), gt_bins, a.bin)
        proc = sum(v for kk, v in r.timings.items() if kk != "1_load")
        row["rtf"] = round(proc / max(r.duration, 1e-9), 5)
        rows.append(row)
        print(f"{tag}: {r.orig_sr} Hz x {r.channels} kênh, {r.duration:.1f} s, "
              f"{len(r.segments)} segment{msg}")

    write_csv(os.path.join(a.out, "results.csv"), rows)
    if has_gt:
        summ = summarize([r for r in rows if "tp" in r])
        write_csv(os.path.join(a.out, "summary.csv"), summ)
        print()
        for s in summ:
            if s["group"] in ("all", "quality", "category"):
                label = s["group"] if s["group"] == "all" else f"{s['group']}={s['quality'] or s['category']}"
                print(f"{label:<24} {s['files']:>4} file  F1 {s['f1']:.4f}  P {s['precision']:.4f}  "
                      f"R {s['recall']:.4f}  miss {s['miss_rate']:.4f}  FAR {s['far']:.4f}")
    n_err = sum(1 for r in rows if "error" in r)
    print(f"\n{len(rows) - n_err}/{len(rows)} file OK, {time.perf_counter() - t_all:.1f} s -> {a.out}")


if __name__ == "__main__":
    main()
