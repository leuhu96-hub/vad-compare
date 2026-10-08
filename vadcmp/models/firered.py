"""FireRedVAD (Xiaohongshu FireRedTeam) - non-streaming or streaming ONNX model.

Files (from github.com/FireRedTeam/FireRedVAD/pretrained_models/onnx_models):
    fireredvad_vad.onnx         non-streaming (default)
    fireredvad_stream_vad.onnx  streaming, caches handled internally
    cmvn.ark
Features: Kaldi fbank 80-dim, 25/10 ms, on int16-scaled samples, then global CMVN.
Output: per-10ms speech probability (already sigmoid).
"""
from __future__ import annotations

import os

import numpy as np

from vadcmp.audio import to_int16
from vadcmp.kaldi_fbank import KaldiCMVN, fbank
from vadcmp.models.base import VADModel, ort_session
from vadcmp.timeline import FrameScores


class FireRedVAD(VADModel):
    type_name = "firered"
    default_threshold = 0.4  # upstream FireRedVadConfig.speech_threshold

    def load(self):
        model_dir = self.path("model_dir", "models/firered")
        onnx_name = self.cfg.get("onnx", "fireredvad_vad.onnx")
        self.sess = ort_session(os.path.join(model_dir, onnx_name), self.num_threads)
        self.cmvn = KaldiCMVN(os.path.join(model_dir, self.cfg.get("cmvn", "cmvn.ark")))
        self.input_name = self.sess.get_inputs()[0].name
        self.chunk_frames = int(self.cfg.get("chunk_max_frame", 30000))

    def process(self, audio: np.ndarray) -> FrameScores:
        dur = len(audio) / 16000
        feats = self.cmvn(fbank(to_int16(audio).astype(np.float32)))
        if len(feats) == 0:
            return FrameScores.from_hops(np.zeros(0), 0.01, duration=dur)
        probs = []
        for a in range(0, len(feats), self.chunk_frames):
            chunk = feats[a:a + self.chunk_frames][None]
            out = self.sess.run(None, {self.input_name: chunk})[0]
            probs.append(np.asarray(out).reshape(-1))
        return FrameScores.from_hops(np.concatenate(probs), 0.01, duration=dur)
