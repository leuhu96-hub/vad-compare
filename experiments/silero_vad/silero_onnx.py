"""Silero VAD chạy thẳng file .onnx (không cần torch / package silero-vad).

Pipeline (giống silero_vad.OnnxWrapper + get_speech_timestamps):
  1. đọc audio bất kỳ (sr, số kênh)         -> dùng chung firered_onnx.step1_load_audio (librosa)
  2. trộn về 1 kênh (mean)                   -> step2_downmix
  3. resample về 16 kHz                      -> step3_resample (librosa soxr_hq)
  4. cắt khối 512 mẫu (32 ms), khối cuối pad 0
  5. mỗi khối: input = [64 mẫu cuối của khối trước | 512 mẫu] (1, 576) float32 trong [-1, 1]
               state (2, 1, 128) mang từ khối trước sang, sr = 16000 (int64)
               -> output (1, 1) = xác suất speech của khối, state mới
  6. hậu xử lý gốc: silero_utils.get_speech_timestamps_from_probs (chép nguyên văn từ silero-vad)
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "firered_vad"))
from firered_onnx import step1_load_audio, step2_downmix, step3_resample  # noqa: E402

SR = 16000
CHUNK = 512           # 32 ms
CONTEXT = 64


def default_model_path():
    for p in (os.path.join(HERE, "models", "silero_vad.onnx"),
              os.path.join(HERE, "..", "..", "models", "silero_vad.onnx")):
        if os.path.exists(p):
            return os.path.normpath(p)
    return os.path.join(HERE, "models", "silero_vad.onnx")


@dataclass
class Result:
    path: str
    orig_sr: int
    channels: int
    n_samples: int             # số mẫu 16 kHz
    probs: np.ndarray          # [ceil(n_samples / 512)] mỗi khối 32 ms
    timings: dict = field(default_factory=dict)

    @property
    def duration(self):
        return self.n_samples / SR


class SileroOnnx:
    def __init__(self, model_path: str | None = None, threads: int = 1, downmix: str = "mean",
                 res_type: str = "soxr_hq"):
        import onnxruntime as ort

        self.model_path = model_path or default_model_path()
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Không thấy model Silero: {self.model_path} (chỉ định bằng --model)")
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = threads
        self.sess = ort.InferenceSession(self.model_path, so, providers=["CPUExecutionProvider"])
        names = {i.name for i in self.sess.get_inputs()}
        if not {"input", "state", "sr"} <= names:
            raise ValueError(f"{self.model_path}: cần input/state/sr (Silero v5+), model có {sorted(names)}")
        self.downmix, self.res_type = downmix, res_type

    def probs_16k(self, x16: np.ndarray) -> np.ndarray:
        n = len(x16)
        pad = (-n) % CHUNK
        x = np.pad(x16.astype(np.float32), (0, pad))
        state = np.zeros((2, 1, 128), np.float32)
        context = np.zeros((1, CONTEXT), np.float32)
        sr = np.array(SR, dtype=np.int64)
        out = np.empty(len(x) // CHUNK, np.float32)
        for k in range(len(out)):
            inp = np.concatenate([context, x[None, k * CHUNK:(k + 1) * CHUNK]], axis=1)
            p, state = self.sess.run(None, {"input": inp, "state": state, "sr": sr})
            out[k] = p[0, 0]
            context = inp[:, -CONTEXT:]
        return out

    def run_file(self, path: str) -> Result:
        tm = {}
        t = time.perf_counter()
        x, sr = step1_load_audio(path)
        tm["load"] = time.perf_counter() - t
        t = time.perf_counter()
        x16 = step3_resample(step2_downmix(x, self.downmix), sr, SR, self.res_type)
        tm["resample"] = time.perf_counter() - t
        t = time.perf_counter()
        probs = self.probs_16k(x16)
        tm["onnx"] = time.perf_counter() - t
        return Result(path, sr, x.shape[0], len(x16), probs, tm)


def speech_segments(probs, n_samples, threshold=0.5, min_speech_ms=250, min_silence_ms=100,
                    speech_pad_ms=30, max_speech_s=float("inf"), neg_threshold=None):
    """Hậu xử lý gốc Silero -> [(start_s, end_s)]."""
    from silero_utils import get_speech_timestamps_from_probs

    ts = get_speech_timestamps_from_probs(
        [float(p) for p in probs], sampling_rate=SR, threshold=threshold,
        min_speech_duration_ms=min_speech_ms, max_speech_duration_s=max_speech_s,
        min_silence_duration_ms=min_silence_ms, speech_pad_ms=speech_pad_ms,
        neg_threshold=neg_threshold, audio_length_samples=n_samples)
    return [(d["start"] / SR, d["end"] / SR) for d in ts]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Silero VAD qua ONNX cho 1 file")
    ap.add_argument("audio")
    ap.add_argument("--model")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    r = SileroOnnx(a.model).run_file(a.audio)
    print(f"{a.audio}: {r.orig_sr} Hz, {r.channels} kênh, {r.duration:.2f} s, {len(r.probs)} khối 32 ms")
    for s, e in speech_segments(r.probs, r.n_samples, a.threshold):
        print(f"{s:.3f} {e:.3f} Speech")
