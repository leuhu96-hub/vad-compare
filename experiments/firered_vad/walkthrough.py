"""Chạy từng bước và in shape / dtype / khoảng giá trị của mỗi tensor.

    python walkthrough.py audio.m4a [--mode nonstream|stream|stream_cached]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import firered_onnx as fo  # noqa: E402


def show(step, name, a, note=""):
    a = np.asarray(a)
    rng = f"[{a.min():.4g}, {a.max():.4g}]" if a.size else "[]"
    print(f"{step:<5} {name:<22} shape {str(a.shape):<16} {str(a.dtype):<8} {rng:<26} {note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--model-dir", default=os.path.join(HERE, "models"))
    ap.add_argument("--onnx", help="file .onnx có sẵn (tên tuỳ ý)")
    ap.add_argument("--cmvn", help="cmvn.ark (mặc định: cạnh file .onnx)")
    ap.add_argument("--mode", default="auto", choices=["auto", *fo.MODEL_FILES])
    ap.add_argument("--chunk", type=int, default=10)
    a = ap.parse_args()

    x, sr = fo.step1_load_audio(a.audio)
    show("1", "load", x, f"{sr} Hz, {x.shape[0]} kênh, {x.shape[1] / sr:.2f} s")
    mono = fo.step2_downmix(x)
    show("2", "downmix", mono, "trung bình các kênh")
    x16 = fo.step3_resample(mono, sr)
    show("3", "resample -> 16 kHz", x16, f"{len(x16) / 16000:.2f} s")
    xi = fo.step4_int16_scale(x16)
    show("4", "int16 scale", xi, "×32768")
    if sr < 16000:
        xi = fo.step4b_dither(xi, 1.0)
        show("4b", "dither 1.0", xi, "vì input < 16 kHz")
    feat, st = fo.step5_fbank(xi, return_steps=True)
    show("5a", "frames", st["5a_frames"], "25 ms / 10 ms, snip_edges")
    show("5b", "remove DC", st["5b_dc"])
    show("5c", "pre-emphasis 0.97", st["5c_preemph"])
    show("5d", "povey window", st["5d_window"])
    show("5e", "|FFT 512|^2", st["5e_power"])
    show("5f", "mel 80", st["5f_mel"])
    show("5g", "log -> fbank", feat)
    vad = fo.FireRedOnnx(a.model_dir, a.mode, onnx_path=a.onnx, cmvn_path=a.cmvn)
    mean, istd = vad.mean, vad.istd
    show("6", "cmvn mean / istd", np.stack([mean, istd]), f"từ {vad.cmvn_path}")
    feat_n = fo.step6_cmvn(feat, mean, istd)
    show("6", "CMVN", feat_n, f"mean {feat_n.mean():.3f}, std {feat_n.std():.3f}")
    sess = vad.sess
    print(f"\n      ONNX {vad.onnx_path}  -> mode {vad.mode}\n{fo.describe_session(sess)}\n")
    if vad.mode == "nonstream":
        probs = fo.step7_onnx_nonstream(sess, feat_n)
    elif vad.mode == "stream":
        probs = fo.step7_onnx_stream_full(sess, feat_n)
    else:
        probs = fo.step7_onnx_stream_cached(sess, feat_n, a.chunk)
    show("7", "probs", probs, f"1 giá trị / 10 ms ({vad.mode})")
    sm, binary, dec, segs = fo.step8_postprocess(probs, fo.PostConfig(), len(x16) / 16000)
    show("8", "smoothed", sm, "cửa sổ 5 frame")
    show("8", "decisions", dec, f"{dec.mean() * 100:.1f}% frame là speech")
    print(f"8     segments ({len(segs)}): " + ", ".join(f"{s:.2f}-{e:.2f}" for s, e in segs[:12])
          + (" ..." if len(segs) > 12 else ""))


if __name__ == "__main__":
    main()
