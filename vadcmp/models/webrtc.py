"""WebRTC VAD (GMM, binary decisions). pip install webrtcvad-wheels"""
from __future__ import annotations

import numpy as np

from vadcmp.audio import to_int16
from vadcmp.models.base import VADModel
from vadcmp.timeline import FrameScores


class WebRTCVAD(VADModel):
    type_name = "webrtc"
    default_threshold = 0.5

    def load(self):
        import webrtcvad

        self.mode = int(self.cfg.get("mode", 2))           # 0 (least aggressive) .. 3
        self.frame_ms = int(self.cfg.get("frame_ms", 30))   # 10, 20 or 30
        assert self.frame_ms in (10, 20, 30), "webrtc frame_ms must be 10, 20 or 30"
        self.vad = webrtcvad.Vad(self.mode)

    def process(self, audio: np.ndarray) -> FrameScores:
        import webrtcvad

        self.vad = webrtcvad.Vad(self.mode)  # fresh state per file
        n = 16000 * self.frame_ms // 1000
        dur = len(audio) / 16000
        pcm = to_int16(np.pad(audio, (0, (-len(audio)) % n)))
        k = len(pcm) // n
        dec = np.empty(k, dtype=np.float32)
        for i in range(k):
            dec[i] = 1.0 if self.vad.is_speech(pcm[i * n:(i + 1) * n].tobytes(), 16000) else 0.0
        return FrameScores.from_hops(dec, n / 16000, duration=dur)
