"""Silero VAD chạy thẳng file .onnx (không cần torch / package silero-vad).

Pipeline (giống silero_vad.OnnxWrapper + get_speech_timestamps):
  1. đọc audio bất kỳ (sr, số kênh)         -> dùng chung firered_onnx.step1_load_audio (librosa)
  2. trộn về 1 kênh (mean)                   -> step2_downmix
  3. resample về 16 kHz                      -> step3_resample (librosa soxr_hq)
  4. cắt khối 512 mẫu (32 ms), khối cuối pad 0
  5. mỗi khối: input = [64 mẫu cuối của khối trước | 512 mẫu] (1, 576) float32 trong [-1, 1]
               state (2, 1, 128) mang từ khối trước sang, sr = 16000 (int64)
               -> output (1, 1) = xác suất speech của khối, state mới
     Model có sẵn tên tuỳ ý: loại được nhận tự động từ chữ ký ONNX (detect_kind): v5/v6 (kể cả _op15,
     _half, _op18_ifless, _openvino_16k không có sr), v4 (h/c, không ngữ cảnh), _16k_sequence (nhiều khối/lần).
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


def resolve_model(path: str | None) -> str:
    """File .onnx có sẵn (tên tuỳ ý) hoặc thư mục chứa nó. Thư mục: ưu tiên silero_vad.onnx, rồi *silero*.onnx,
    rồi file .onnx duy nhất."""
    path = path or default_model_path()
    if os.path.isdir(path):
        onnx = sorted(f for f in os.listdir(path) if f.lower().endswith(".onnx"))
        pick = ([f for f in onnx if f.lower() == "silero_vad.onnx"] or [f for f in onnx if "silero" in f.lower()]
                or (onnx if len(onnx) == 1 else []))
        if not pick:
            raise FileNotFoundError(f"{path}: không tìm thấy file .onnx của Silero (có: {onnx or 'không có'})")
        path = os.path.join(path, pick[0])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không thấy model Silero: {path} (chỉ định bằng --model <file.onnx | thư mục>)")
    return path


def detect_kind(sess) -> str:
    """Nhận loại model từ chữ ký ONNX:
      v5        input, state [2,B,128] (+ sr)  -> có 64 mẫu ngữ cảnh   (silero_vad.onnx v5/v6, _op15, _half,
                                                                         _op18_ifless, _openvino_16k)
      v4        input, sr, h, c [2,B,64]       -> không ngữ cảnh         (silero_vad.onnx bản v4)
      sequence  input [L,576], h, c [1,1,128]  -> nhiều khối một lần     (silero_vad_16k_sequence.onnx)"""
    ins = {i.name for i in sess.get_inputs()}
    outs = {o.name for o in sess.get_outputs()}
    if "state" in ins and "input" in ins:
        return "v5"
    if {"input", "h", "c"} <= ins and "speech_probs" in outs:
        return "sequence"
    if {"input", "h", "c"} <= ins:
        return "v4"
    raise ValueError("Không nhận ra model Silero. Input: " + ", ".join(
        f"{i.name}{i.shape}" for i in sess.get_inputs()) + "; output: " + ", ".join(sorted(outs)))


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


_NP = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(double)": np.float64}


class SileroOnnx:
    def __init__(self, model_path: str | None = None, threads: int = 1, downmix: str = "mean",
                 res_type: str = "soxr_hq", block_frames: int = 512):
        import onnxruntime as ort

        self.model_path = resolve_model(model_path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = threads
        so.log_severity_level = 3                       # ẩn cảnh báo "Removing initializer" của vài bản export
        self.sess = ort.InferenceSession(self.model_path, so, providers=["CPUExecutionProvider"])
        self.kind = detect_kind(self.sess)
        self.inputs = {i.name: i for i in self.sess.get_inputs()}
        self.has_sr = "sr" in self.inputs
        self.dtype = _NP.get(self.inputs["input"].type, np.float32)
        self.downmix, self.res_type, self.block_frames = downmix, res_type, block_frames

    def describe(self) -> str:
        sig = ", ".join(f"{i.name}{i.shape}" for i in self.sess.get_inputs())
        return f"{self.model_path} [{self.kind}] ({sig})"

    def _st(self, name, shape):
        t = self.inputs[name]
        return np.zeros(shape, _NP.get(t.type, np.float32))

    def probs_16k(self, x16: np.ndarray) -> np.ndarray:
        n = len(x16)
        pad = (-n) % CHUNK
        x = np.pad(x16.astype(np.float32), (0, pad))
        n_frames = len(x) // CHUNK
        out = np.empty(n_frames, np.float32)
        sr = np.array(SR, dtype=np.int64)
        if self.kind == "sequence":                     # nhiều khối một lần gọi, ngữ cảnh 64 mẫu như v5
            frames = x.reshape(n_frames, CHUNK)
            ctx = np.zeros((n_frames, CONTEXT), np.float32)
            if n_frames > 1:
                ctx[1:] = frames[:-1, -CONTEXT:]
            blocks = np.concatenate([ctx, frames], axis=1).astype(self.dtype)
            h, c = self._st("h", (1, 1, 128)), self._st("c", (1, 1, 128))
            for a in range(0, n_frames, self.block_frames):
                p, h, c = self.sess.run(["speech_probs", "hn", "cn"],
                                        {"input": blocks[a:a + self.block_frames], "h": h, "c": c})
                out[a:a + self.block_frames] = np.asarray(p).reshape(-1)
            return out
        if self.kind == "v4":                           # không ngữ cảnh, state h/c [2,1,64]
            h, c = self._st("h", (2, 1, 64)), self._st("c", (2, 1, 64))
            for k in range(n_frames):
                feed = {"input": x[None, k * CHUNK:(k + 1) * CHUNK].astype(self.dtype), "h": h, "c": c}
                if self.has_sr:
                    feed["sr"] = sr
                p, h, c = self.sess.run(None, feed)
                out[k] = np.asarray(p).reshape(-1)[0]
            return out
        state = self._st("state", (2, 1, 128))          # v5 / v6
        context = np.zeros((1, CONTEXT), np.float32)
        for k in range(n_frames):
            inp = np.concatenate([context, x[None, k * CHUNK:(k + 1) * CHUNK]], axis=1)
            feed = {"input": inp.astype(self.dtype), "state": state}
            if self.has_sr:
                feed["sr"] = sr
            p, state = self.sess.run(None, feed)
            out[k] = np.asarray(p).reshape(-1)[0]
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
    ap.add_argument("--model", help="file .onnx có sẵn (tên tuỳ ý) hoặc thư mục chứa nó")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    m = SileroOnnx(a.model)
    print("Model:", m.describe())
    r = m.run_file(a.audio)
    print(f"{a.audio}: {r.orig_sr} Hz, {r.channels} kênh, {r.duration:.2f} s, {len(r.probs)} khối 32 ms")
    for s, e in speech_segments(r.probs, r.n_samples, a.threshold):
        print(f"{s:.3f} {e:.3f} Speech")
