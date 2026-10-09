"""Chạy FireRedVAD (ONNX, từng bước) trên file hoặc folder audio có sr / số kênh bất kỳ.

    python run.py input.m4a
    python run.py data/Audio --mode stream_cached --chunk 10 --dump --plot -o out
    python run.py song.mp3 --threshold 0.5 --min-speech 20 --min-silence 20

Output cho mỗi file <stem> trong thư mục -o:
    <stem>_probs.csv       frame, start_s, end_s, prob, smoothed, decision   (mỗi frame 10 ms)
    <stem>_segments.txt    start end Speech                                 (sau post-processing)
    <stem>_bins.txt        0 0.5 Speech / 0.5 1 Non-Speech                  (bin 0.5 s, so với GT)
    <stem>_steps.npz       (--dump) toàn bộ tensor trung gian của các bước 1..8
    <stem>.png             (--plot) waveform, log-mel, CMVN, prob, segment
và summary.csv (thông tin input + thời gian từng bước).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from firered_onnx import (FRAME_SHIFT_S, FireRedOnnx, PostConfig,  # noqa: E402
                          describe_session)

AUDIO_EXTS = {".wav", ".mp3", ".mp4", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm",
              ".mkv", ".mov", ".amr", ".3gp", ".wma", ".aiff", ".aif", ".caf"}
HERE = os.path.dirname(os.path.abspath(__file__))


def list_inputs(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for dp, dirs, files in os.walk(p):
                dirs.sort()
                out += [os.path.join(dp, f) for f in sorted(files)
                        if os.path.splitext(f)[1].lower() in AUDIO_EXTS]
        else:
            out.append(p)
    return out


def segments_to_bins(segs, dur, bin_s=0.5, min_overlap=0.5):
    n = int(np.ceil(dur / bin_s - 1e-9))
    cov = np.zeros(n)
    for s, e in segs:
        for i in range(int(s // bin_s), min(n, int(np.ceil(e / bin_s)))):
            a, b = i * bin_s, min((i + 1) * bin_s, dur)
            cov[i] += max(0.0, min(e, b) - max(s, a))
    width = np.minimum((np.arange(n) + 1) * bin_s, dur) - np.arange(n) * bin_s
    return [(i * bin_s, min((i + 1) * bin_s, dur), int(cov[i] >= min_overlap * width[i] - 1e-9)) for i in range(n)]


def write_outputs(r, out_dir, stem, dump, plot):
    t0, t1 = r.frame_times()
    with open(os.path.join(out_dir, f"{stem}_probs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "start_s", "end_s", "prob", "smoothed", "decision"])
        for i in range(len(r.probs)):
            w.writerow([i, f"{t0[i]:.3f}", f"{t1[i]:.3f}", f"{r.probs[i]:.6f}",
                        f"{r.smoothed[i]:.6f}", int(r.decisions[i])])
    with open(os.path.join(out_dir, f"{stem}_segments.txt"), "w") as f:
        for s, e in r.segments:
            f.write(f"{s:.3f} {e:.3f} Speech\n")
    with open(os.path.join(out_dir, f"{stem}_bins.txt"), "w") as f:
        for s, e, d in segments_to_bins(r.segments, r.duration):
            f.write(f"{s:g} {e:g} {'Speech' if d else 'Non-Speech'}\n")
    if dump:
        np.savez_compressed(os.path.join(out_dir, f"{stem}_steps.npz"),
                            orig_sr=r.orig_sr, channels=r.channels,
                            **{k: np.asarray(v) for k, v in r.steps.items()})
    if plot:
        plot_result(r, os.path.join(out_dir, f"{stem}.png"))


def plot_result(r, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = r.steps
    fig, ax = plt.subplots(4, 1, figsize=(12, 8.5), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1.4, 1.4, 1.2]})
    x16 = s.get("3_16k")
    if x16 is not None:
        t = np.arange(len(x16)) / 16000
        ax[0].plot(t, x16, lw=0.4, color="#52514e")
    ax[0].set_title(f"{os.path.basename(r.path)}   input: {r.orig_sr} Hz, {r.channels} kênh  ->  16 kHz mono",
                    loc="left", fontsize=10, fontweight="bold")
    ext = [0, len(r.probs) * FRAME_SHIFT_S, 0, 80]
    if "5g_logmel" in s:
        ax[1].imshow(s["5g_logmel"].T, origin="lower", aspect="auto", extent=ext, cmap="Blues")
        ax[1].set_ylabel("log-mel\n(bước 5)")
    if "6_cmvn" in s:
        ax[2].imshow(s["6_cmvn"].T, origin="lower", aspect="auto", extent=ext, cmap="RdBu_r", vmin=-3, vmax=3)
        ax[2].set_ylabel("sau CMVN\n(bước 6)")
    t = np.arange(len(r.probs)) * FRAME_SHIFT_S
    ax[3].plot(t, r.probs, lw=0.8, color="#8a8984", label="prob (bước 7)")
    ax[3].plot(t, r.smoothed, lw=1.4, color="#2a78d6", label="smoothed")
    for a, b in r.segments:
        ax[3].axvspan(a, b, ymin=0, ymax=0.06, color="#0b0b0b", lw=0)
    ax[3].set_ylim(-0.05, 1.05)
    ax[3].set_xlabel("giây  (vạch đen = segment sau post-processing, bước 8)")
    ax[3].legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="FireRedVAD ONNX từng bước, input sr/kênh bất kỳ")
    ap.add_argument("inputs", nargs="+", help="file audio hoặc folder")
    ap.add_argument("-o", "--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--model-dir", default=os.path.join(HERE, "models"))
    ap.add_argument("--mode", default="nonstream", choices=["nonstream", "stream", "stream_cached"])
    ap.add_argument("--chunk", type=int, default=10, help="stream_cached: số frame 10 ms mỗi lần gọi")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--downmix", default="mean", help="mean | 0 | 1 ... (chọn kênh)")
    ap.add_argument("--res-type", default="soxr_hq", help="librosa.resample res_type")
    ap.add_argument("--dither", default="auto",
                    help="auto (=1.0 nếu input < 16 kHz, ngược lại 0) | số (đơn vị int16) | 0 để tắt")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--smooth", type=int, default=5, help="cửa sổ làm mượt (frame)")
    ap.add_argument("--min-speech", type=int, default=20, help="frame")
    ap.add_argument("--min-silence", type=int, default=20, help="frame")
    ap.add_argument("--max-speech", type=int, default=2000, help="frame")
    ap.add_argument("--merge-silence", type=int, default=0, help="frame")
    ap.add_argument("--extend-speech", type=int, default=0, help="frame")
    ap.add_argument("--dump", action="store_true", help="lưu tensor trung gian vào <stem>_steps.npz")
    ap.add_argument("--plot", action="store_true", help="vẽ <stem>.png")
    ap.add_argument("--show-model", action="store_true", help="in input/output của file ONNX")
    a = ap.parse_args()

    post = PostConfig(a.smooth, a.threshold, a.min_speech, a.max_speech, a.min_silence,
                      a.merge_silence, a.extend_speech)
    vad = FireRedOnnx(a.model_dir, a.mode, a.threads, a.chunk, post, a.downmix, a.res_type, a.dither)
    if a.show_model:
        print(f"[{a.mode}] {describe_session(vad.sess)}")
    os.makedirs(a.out, exist_ok=True)
    files = list_inputs(a.inputs)
    rows = []
    for k, path in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            r = vad.run_file(path, keep_steps=a.dump or a.plot)
        except Exception as e:
            print(f"[{k}/{len(files)}] LỖI {path}: {type(e).__name__}: {e}")
            continue
        write_outputs(r, a.out, stem, a.dump, a.plot)
        proc = sum(v for kk, v in r.timings.items() if kk != "1_load")
        speech = sum(e - s for s, e in r.segments)
        print(f"[{k}/{len(files)}] {os.path.basename(path)}: {r.orig_sr} Hz x {r.channels} kênh, "
              f"{r.duration:.2f} s -> {len(r.probs)} frame, {len(r.segments)} segment "
              f"({speech:.2f} s speech), RTF {proc / max(r.duration, 1e-9):.4f}")
        rows.append({"file": path, "orig_sr": r.orig_sr, "channels": r.channels,
                     "duration_s": round(r.duration, 3), "frames": len(r.probs),
                     "n_segments": len(r.segments), "speech_s": round(speech, 3),
                     "dither": vad.last_dither,
                     **{f"t_{kk}": round(v, 5) for kk, v in r.timings.items()}})
    if rows:
        with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
