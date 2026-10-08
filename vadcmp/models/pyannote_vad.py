"""pyannote segmentation (default pyannote/segmentation-3.0) used as a VAD.

pip install pyannote.audio ; accept the model's user conditions on Hugging Face and set
HF_TOKEN (or `token:` in config). Speech prob = 1 - P(no active speaker) for powerset
models, max over speakers otherwise.
"""
from __future__ import annotations

import os

import numpy as np

from vadcmp.models.base import VADModel
from vadcmp.timeline import FrameScores


class PyannoteVAD(VADModel):
    type_name = "pyannote"
    default_threshold = 0.5

    def load(self):
        import torch
        from pyannote.audio import Inference, Model

        torch.set_num_threads(self.num_threads)
        ckpt = self.cfg.get("checkpoint", "pyannote/segmentation-3.0")
        local = self.path("checkpoint") if ckpt and os.path.exists(self.path("checkpoint")) else ckpt
        token = self.cfg.get("token") or os.environ.get("HF_TOKEN")
        try:
            model = Model.from_pretrained(local, token=token)
        except TypeError:  # pyannote.audio < 4
            model = Model.from_pretrained(local, use_auth_token=token)
        if model is None:
            raise RuntimeError("pyannote model failed to load - check HF_TOKEN and model access")
        specs = model.specifications
        specs = specs[0] if isinstance(specs, (tuple, list)) else specs
        self.powerset = bool(getattr(specs, "powerset", False))
        kw = {"skip_conversion": True} if self.powerset else {}
        step = self.cfg.get("step")
        if step:
            kw["step"] = float(step)
        self.inference = Inference(model, device=torch.device("cpu"), **kw)
        self.torch = torch

    def process(self, audio: np.ndarray) -> FrameScores:
        dur = len(audio) / 16000
        with self.torch.inference_mode():
            out = self.inference({"waveform": self.torch.from_numpy(audio)[None], "sample_rate": 16000})
        data = np.asarray(out.data, dtype=np.float64)
        sw = out.sliding_window
        if self.powerset:
            speech = 1.0 - np.exp(data[:, 0])          # class 0 = empty speaker set (log-prob)
        else:
            speech = data.max(axis=1)
        speech = np.clip(speech, 0.0, 1.0)
        centers = sw.start + sw.duration / 2 + np.arange(len(speech)) * sw.step
        return FrameScores.from_centers(centers, speech, sw.step, duration=dur)
