"""Kiểm chứng từng bước của firered_onnx.py.

    python check_steps.py <file_16k_mono.wav> [--upstream-repo /path/to/FireRedVAD]

1. step5 fbank (numpy)       vs kaldi-native-fbank           (nếu đã cài)
2. step6 CMVN                in thống kê mean/istd; so với kaldiio nếu đã cài
3. step7 stream_cached       vs stream (cả chuỗi)           với chunk 1 / 10 / 37 frame
4. sr / số kênh bất kỳ        tạo các bản 8k..48k, 1..6 kênh từ file gốc -> so prob với bản 16k mono,
                              kiểm tra lệch thời gian (cross-correlation), số frame; thử nhiều res_type
5. step8 post-process        vs VadPostprocessor của upstream (nếu có --upstream-repo,
                              chỉ import file .py để đối chiếu, không dùng torch)
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
import types
from fractions import Fraction

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import firered_onnx as fo  # noqa: E402

OK, WARN = "PASS", "WARN"


def line(name, status, msg):
    print(f"[{status}] {name:<34} {msg}")


def check_fbank(x16):
    xi = fo.step4_int16_scale(x16)
    ours = fo.step5_fbank(xi)
    try:
        ref = fo.step5_fbank_knf(xi)
    except ImportError:
        line("fbank vs kaldi-native-fbank", WARN, "chưa cài kaldi-native-fbank -> bỏ qua")
        return
    d = np.abs(ours - ref)
    st = OK if ours.shape == ref.shape and d.max() < 1e-2 else "FAIL"
    line("fbank vs kaldi-native-fbank", st, f"shape {ours.shape}, max |diff| {d.max():.2e}, mean {d.mean():.2e}")


def check_cmvn(model_dir):
    p = os.path.join(model_dir, "cmvn.ark")
    mean, istd = fo.load_cmvn(p)
    line("cmvn.ark", OK, f"dim {len(mean)}, mean [{mean.min():.2f}, {mean.max():.2f}], "
                         f"std [{(1 / istd).min():.2f}, {(1 / istd).max():.2f}]")
    try:
        import kaldiio
    except ImportError:
        return
    st = kaldiio.load_mat(p)
    ours = fo.read_kaldi_matrix(p)
    line("cmvn vs kaldiio", OK if np.allclose(st, ours) else "FAIL", f"max |diff| {np.abs(st - ours).max():.2e}")


def check_stream_cache(model_dir, feat):
    full = fo.step7_onnx_stream_full(fo._session(os.path.join(model_dir, fo.MODEL_FILES["stream"])), feat)
    sess = fo._session(os.path.join(model_dir, fo.MODEL_FILES["stream_cached"]))
    for c in (1, 10, 37):
        p = fo.step7_onnx_stream_cached(sess, feat, c)
        d = np.abs(p - full)
        line(f"stream_cached chunk={c:<3} vs stream", OK if d.max() < 1e-4 else "FAIL",
             f"{len(p)} frame, max |diff| {d.max():.2e}")


def _make_variant(x16, sr, ch):
    """16 kHz mono -> sr / ch. Dùng FFT resample (giữ nguyên dải 0-8 kHz) để chỉ đo ảnh hưởng của
    pipeline, rồi nhân bản kênh. Lưu WAV float32."""
    from scipy.signal import resample as fft_resample

    y = fft_resample(x16, int(round(len(x16) * sr / 16000))).astype(np.float32) if sr != 16000 else x16
    return np.repeat(y[:, None], ch, axis=1)


def _lag(a, b, max_lag=64):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    lags = range(-max_lag, max_lag + 1)
    c = [np.dot(a[max(0, k):n + min(0, k)], b[max(0, -k):n - max(0, k)]) for k in lags]
    return list(lags)[int(np.argmax(c))]


def check_any_sr(x16, model_dir, res_types):
    vad0 = fo.FireRedOnnx(model_dir, "nonstream")
    ref = vad0.run_array(x16, 16000)
    print(f"       gốc 16 kHz mono: {len(ref.probs)} frame, {len(ref.segments)} segment")
    cases = [(16000, 2), (22050, 1), (44100, 2), (48000, 6), (8000, 1)]
    with tempfile.TemporaryDirectory() as td:
        for rt in res_types:
            vad = fo.FireRedOnnx(model_dir, "nonstream", res_type=rt)
            print(f"   res_type = {rt}")
            for sr, ch in cases:
                path = os.path.join(td, f"v_{sr}_{ch}.wav")
                if not os.path.exists(path):
                    wavfile.write(path, sr, _make_variant(x16, sr, ch))
                try:
                    r = vad.run_file(path, keep_steps=True)   # đủ bước 1 (librosa.load) -> 8
                except ImportError as e:
                    line("sr/kênh bất kỳ", WARN, f"cần librosa: {e}")
                    return
                n = min(len(r.probs), len(ref.probs))
                d = np.abs(r.probs[:n] - ref.probs[:n])
                agree = (r.decisions[:n] == ref.decisions[:n]).mean() * 100
                lag = _lag(r.steps["3_16k"], x16)
                if sr < 16000:
                    st = OK if agree >= 90 else WARN
                    note = f"  (dither={vad.last_dither}; 8 kHz không có dải 4-8 kHz nên không thể giống hệt)"
                else:
                    st, note = (OK if lag == 0 and len(r.probs) == len(ref.probs) and agree >= 99 else "FAIL"), ""
                line(f"{sr:>5} Hz x {ch} kênh", st,
                     f"đọc {r.orig_sr} Hz x {r.channels} kênh, lệch {lag} mẫu, frame {len(r.probs)}/{len(ref.probs)}, "
                     f"mean |Δprob| {d.mean():.4f}, max {d.max():.3f}, quyết định khớp {agree:.1f}%{note}")


def check_post_vs_upstream(repo, probs):
    pkg = os.path.join(repo, "fireredvad")
    if not os.path.isdir(pkg):
        line("post-process vs upstream", "FAIL", f"không thấy {pkg}")
        return
    # nạp module mà không chạy fireredvad/__init__.py (cần torch)
    for name, sub in (("fireredvad", pkg), ("fireredvad.core", os.path.join(pkg, "core"))):
        m = types.ModuleType(name)
        m.__path__ = [sub]
        sys.modules.setdefault(name, m)
    spec = importlib.util.spec_from_file_location("fireredvad.core.vad_postprocessor",
                                                  os.path.join(pkg, "core", "vad_postprocessor.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for cfg in (fo.PostConfig(), fo.PostConfig(3, 0.5, 10, 300, 10, 15, 5)):
        up = mod.VadPostprocessor(cfg.smooth_window_size, cfg.speech_threshold, cfg.min_speech_frame,
                                  cfg.max_speech_frame, cfg.min_silence_frame, cfg.merge_silence_frame,
                                  cfg.extend_speech_frame)
        dec_up = up.process(probs.tolist())
        seg_up = up.decision_to_segment(dec_up, len(probs) * 0.01)
        _, _, dec, seg = fo.step8_postprocess(probs, cfg, len(probs) * 0.01)
        same = list(dec) == list(dec_up) and seg == seg_up
        line("post-process vs upstream", OK if same else "FAIL",
             f"thr={cfg.speech_threshold} smooth={cfg.smooth_window_size} max_speech={cfg.max_speech_frame}: "
             f"{len(seg)} segment, decisions {'giống hệt' if same else 'KHÁC'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="file tham chiếu (nên là wav 16 kHz mono có speech)")
    ap.add_argument("--model-dir", default=os.path.join(HERE, "models"))
    ap.add_argument("--upstream-repo", help="đường dẫn repo FireRedVAD đã clone (chỉ để đối chiếu)")
    ap.add_argument("--res-types", default="soxr_hq,fft",
                    help="các librosa res_type cần thử, vd. soxr_hq,soxr_vhq,kaiser_best,polyphase,fft")
    a = ap.parse_args()

    x, sr = fo.step1_load_audio(a.wav)
    x16 = fo.step3_resample(fo.step2_downmix(x), sr)
    print(f"Input: {a.wav} ({sr} Hz, {x.shape[0]} kênh, {len(x16) / 16000:.2f} s)\n")
    check_fbank(x16)
    check_cmvn(a.model_dir)
    mean, istd = fo.load_cmvn(os.path.join(a.model_dir, "cmvn.ark"))
    feat = fo.step6_cmvn(fo.step5_fbank(fo.step4_int16_scale(x16)), mean, istd)
    check_stream_cache(a.model_dir, feat)
    print("\nsr / số kênh bất kỳ (so với bản 16 kHz mono):")
    check_any_sr(x16, a.model_dir, [t.strip() for t in a.res_types.split(",") if t.strip()])
    if a.upstream_repo:
        probs = fo.step7_onnx_nonstream(fo._session(os.path.join(a.model_dir, fo.MODEL_FILES["nonstream"])), feat)
        print()
        check_post_vs_upstream(a.upstream_repo, probs.astype(np.float64))


if __name__ == "__main__":
    main()
