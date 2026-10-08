"""Build a small demo dataset from the public TEN VAD testset (30 wav + .scv labels),
so the whole pipeline can be tried before plugging in your own data.

    git clone --depth 1 https://github.com/TEN-framework/ten-vad.git /tmp/ten-vad
    python scripts/make_demo_dataset.py /tmp/ten-vad/testset data/demo

Each file is also re-encoded (needs ffmpeg) into a different container / sample rate /
channel layout, one per category, to exercise the decoder:
    wav_16k_mono | mp3_44k_stereo | m4a_48k_stereo
GT is written like your files: Model/<category>/ground_truth/<stem>.txt with lines
"0 0.5 Speech" / "0.5 1 Non-Speech 1" (the 4th column is ignored by vadcmp).
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from vadcmp.dataset import write_manifest  # noqa: E402
from vadcmp.groundtruth import gt_grid  # noqa: E402

VARIANTS = {
    "wav_16k_mono": (".wav", []),
    "mp3_44k_stereo": (".mp3", ["-ar", "44100", "-ac", "2", "-c:a", "libmp3lame", "-b:a", "128k"]),
    "m4a_48k_stereo": (".m4a", ["-ar", "48000", "-ac", "2", "-c:a", "aac", "-b:a", "128k"]),
}


def read_scv(path):
    toks = open(path).read().strip().split(",")[1:]
    return np.array([(float(toks[i]), float(toks[i + 1]), int(toks[i + 2])) for i in range(0, len(toks) - 2, 3)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("testset")
    ap.add_argument("out")
    ap.add_argument("--bin", type=float, default=0.5)
    a = ap.parse_args()
    wavs = sorted(glob.glob(os.path.join(a.testset, "*.wav")))
    if not wavs:
        raise SystemExit("no .wav found")
    has_ffmpeg = shutil.which("ffmpeg") is not None
    rows = []
    for k, wav in enumerate(wavs):
        stem = os.path.splitext(os.path.basename(wav))[0]
        scv = os.path.splitext(wav)[0] + ".scv"
        if not os.path.exists(scv):
            continue
        cat = list(VARIANTS)[k % len(VARIANTS)] if has_ffmpeg else "wav_16k_mono"
        ext, args = VARIANTS[cat]
        adir = os.path.join(a.out, "audio", cat)
        gdir = os.path.join(a.out, "Model", cat, "ground_truth")
        os.makedirs(adir, exist_ok=True)
        os.makedirs(gdir, exist_ok=True)
        dst = os.path.join(adir, stem + ext)
        if ext == ".wav":
            shutil.copy(wav, dst)
        else:
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", wav, *args, dst], check=True)
        edges, labels = gt_grid(read_scv(scv), a.bin)
        gp = os.path.join(gdir, f"{stem}.txt")
        with open(gp, "w") as f:
            for s, e, lab in zip(edges[:-1], edges[1:], labels):
                if lab >= 0:
                    f.write(f"{s:g} {e:g} Speech\n" if lab == 1 else f"{s:g} {e:g} Non-Speech 1\n")
        rows.append({"audio": dst, "gt": gp, "category": cat})
    write_manifest(rows, os.path.join(a.out, "manifest.csv"))
    print(f"{len(rows)} files -> {a.out}/manifest.csv")


if __name__ == "__main__":
    main()
