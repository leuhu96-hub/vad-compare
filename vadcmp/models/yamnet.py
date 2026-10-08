"""Your YAMNet transfer-learning VAD (speech / non-speech), run on raw audio.

Supports .tflite (ai-edge-litert, tflite-runtime or tensorflow) and .onnx models whose
input is a waveform window (float32, 16 kHz, [-1, 1]).

Pipeline (same as the on-device lib, minus the decode step which vadcmp.audio does):
    sliding windows of `window_s` (0.96 s) every `hop_s` (0.08 s)
    -> model -> speech prob per window
    -> overlap handling: every hop gets the mean (or max) of the windows covering it
    -> FrameScores at hop resolution (post-processing is applied later, like for all models)
"""
from __future__ import annotations

import os

import numpy as np

from vadcmp.models.base import VADModel, ort_session
from vadcmp.timeline import FrameScores, window_scores_to_hops


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _tflite_interpreter(path: str, num_threads: int):
    errors = []
    for mod, attr in (("ai_edge_litert.interpreter", "Interpreter"),
                      ("tflite_runtime.interpreter", "Interpreter"),
                      ("tensorflow.lite", "Interpreter")):
        try:
            m = __import__(mod, fromlist=[attr])
            return getattr(m, attr)(model_path=path, num_threads=num_threads)
        except ImportError as e:
            errors.append(f"{mod}: {e}")
    raise ImportError("No TFLite runtime found. pip install ai-edge-litert\n" + "\n".join(errors))


class YamnetVAD(VADModel):
    type_name = "yamnet"
    default_threshold = 0.6026  # Youden-optimal threshold found earlier on your data

    def load(self):
        self.window_s = float(self.cfg.get("window_s", 0.96))
        self.hop_s = float(self.cfg.get("hop_s", 0.08))
        self.speech_index = int(self.cfg.get("speech_index", 1))
        self.output_mode = self.cfg.get("output", "auto")      # auto | probs | logits
        self.overlap = self.cfg.get("overlap", "mean")         # mean | max
        self.output_index = int(self.cfg.get("output_index", 0))
        path = self.path("model_path")
        if not path:
            raise ValueError("yamnet: set model_path to your .tflite or .onnx model")
        if not os.path.exists(path):
            raise FileNotFoundError(f"yamnet: model not found: {path} (copy your .tflite/.onnx there)")
        if path.endswith(".onnx"):
            self.backend = "onnx"
            self.sess = ort_session(path, self.num_threads)
            inp = self.sess.get_inputs()[0]
            self.input_name = inp.name
            self.input_shape = [d if isinstance(d, int) else -1 for d in inp.shape]
        else:
            self.backend = "tflite"
            self.interp = _tflite_interpreter(path, self.num_threads)
            self.interp.allocate_tensors()
            det = self.interp.get_input_details()[0]
            self.in_idx = det["index"]
            self.in_dtype = det["dtype"]
            self.in_quant = det.get("quantization", (0.0, 0))
            self.out_det = self.interp.get_output_details()[self.output_index]
            shape = list(det.get("shape_signature", det["shape"]))
            self.input_shape = [int(d) if d > 0 else -1 for d in shape]
        n_cfg = self.cfg.get("input_samples")
        fixed = [d for d in self.input_shape if d > 1]
        if n_cfg:
            self.n_in = int(n_cfg)
        elif fixed:
            self.n_in = fixed[-1]
        else:
            self.n_in = int(round(self.window_s * 16000))
        self._shape = [1 if d in (-1, 1) else d for d in self.input_shape]
        if self._shape and self._shape[-1] != self.n_in:
            self._shape[-1] = self.n_in
        if self.backend == "tflite" and -1 in self.input_shape:
            self.interp.resize_tensor_input(self.in_idx, self._shape)
            self.interp.allocate_tensors()

    def _infer(self, window: np.ndarray) -> np.ndarray:
        x = window.reshape(self._shape).astype(np.float32)
        if self.backend == "onnx":
            return np.asarray(self.sess.run(None, {self.input_name: x})[self.output_index])
        if self.in_dtype != np.float32:
            scale, zero = self.in_quant
            x = np.round(x / (scale or 1.0) + zero).astype(self.in_dtype)
        self.interp.set_tensor(self.in_idx, x)
        self.interp.invoke()
        y = self.interp.get_tensor(self.out_det["index"])
        q = self.out_det.get("quantization", (0.0, 0))
        if y.dtype != np.float32 and q[0]:
            y = (y.astype(np.float32) - q[1]) * q[0]
        return y

    def _to_prob(self, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.float64)
        if y.ndim > 1:  # e.g. (frames, classes) -> average over frames
            y = y.reshape(-1, y.shape[-1]).mean(axis=0)
        y = y.reshape(-1)
        if y.size == 1:
            v = y[0]
            return float(v if self.output_mode == "probs" or 0 <= v <= 1 else 1 / (1 + np.exp(-v)))
        is_prob = self.output_mode == "probs" or (
            self.output_mode == "auto" and y.min() >= 0 and abs(y.sum() - 1) < 1e-3)
        p = y if is_prob else _softmax(y)
        return float(p[self.speech_index])

    def process(self, audio: np.ndarray) -> FrameScores:
        dur = len(audio) / 16000
        hop = int(round(self.hop_s * 16000))
        n_hops = max(1, int(np.ceil(len(audio) / hop)))
        x = np.pad(audio, (0, max(0, self.n_in - len(audio))))
        n_win = 1 + max(0, (len(x) - self.n_in)) // hop
        # one extra window if the tail is not covered
        if (n_win - 1) * hop + self.n_in < len(audio):
            n_win += 1
            x = np.pad(x, (0, (n_win - 1) * hop + self.n_in - len(x)))
        scores = np.empty(n_win)
        for i in range(n_win):
            scores[i] = self._to_prob(self._infer(x[i * hop:i * hop + self.n_in]))
        hops = window_scores_to_hops(scores, self.hop_s, self.window_s, n_hops, self.overlap)
        return FrameScores.from_hops(hops, self.hop_s, duration=dur)
