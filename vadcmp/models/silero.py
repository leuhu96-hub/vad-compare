"""Silero VAD (v5/v6) via ONNX Runtime - same streaming logic as silero_vad.OnnxWrapper,
without the torch dependency. 512-sample chunks (32 ms) + 64-sample context at 16 kHz."""
from __future__ import annotations

import numpy as np

from vadcmp.models.base import VADModel, ort_session
from vadcmp.timeline import FrameScores


class SileroVAD(VADModel):
    type_name = "silero"
    default_threshold = 0.5
    CHUNK = 512
    CONTEXT = 64

    def load(self):
        path = self.path("model_path", "models/silero_vad.onnx")
        self.sess = ort_session(path, self.num_threads)
        self.sr = np.array(16000, dtype=np.int64)

    def process(self, audio: np.ndarray) -> FrameScores:
        n = self.CHUNK
        dur = len(audio) / 16000
        pad = (-len(audio)) % n
        x = np.pad(audio, (0, pad)).astype(np.float32)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, self.CONTEXT), dtype=np.float32)
        probs = np.empty(len(x) // n, dtype=np.float32)
        for k in range(len(probs)):
            inp = np.concatenate([context, x[None, k * n:(k + 1) * n]], axis=1)
            out, state = self.sess.run(None, {"input": inp, "state": state, "sr": self.sr})
            probs[k] = out[0, 0]
            context = inp[:, -self.CONTEXT:]
        return FrameScores.from_hops(probs, n / 16000, duration=dur)
