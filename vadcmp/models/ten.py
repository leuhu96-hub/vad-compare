"""TEN VAD (Agora). pip install git+https://github.com/TEN-framework/ten-vad.git
Linux needs libc++ (sudo apt install libc++1)."""
from __future__ import annotations

import numpy as np

from vadcmp.audio import to_int16
from vadcmp.models.base import VADModel
from vadcmp.timeline import FrameScores


class TenVAD(VADModel):
    type_name = "ten"
    default_threshold = 0.5

    def load(self):
        from ten_vad import TenVad

        self._cls = TenVad
        self.hop = int(self.cfg.get("hop_size", 256))  # 160 (10 ms) or 256 (16 ms)
        self._cls(self.hop, self.threshold)  # fail fast if the native lib is missing

    def process(self, audio: np.ndarray) -> FrameScores:
        vad = self._cls(self.hop, self.threshold)  # new handle = reset state
        n = self.hop
        dur = len(audio) / 16000
        pcm = to_int16(np.pad(audio, (0, (-len(audio)) % n)))
        k = len(pcm) // n
        probs = np.empty(k, dtype=np.float32)
        for i in range(k):
            frame = np.ascontiguousarray(pcm[i * n:(i + 1) * n])
            probs[i], _ = vad.process(frame)
        del vad
        return FrameScores.from_hops(probs, n / 16000, duration=dur)
