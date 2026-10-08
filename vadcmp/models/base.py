from __future__ import annotations

import os

import numpy as np

from vadcmp.timeline import FrameScores


class VADModel:
    """Every adapter turns a mono float32 waveform at 16 kHz into FrameScores.

    Subclasses implement `load()` (called once, timed as load time) and
    `process(audio)` (called per file, timed for RTF). State must be reset per file.
    """

    type_name = "base"
    default_threshold = 0.5

    def __init__(self, name: str, cfg: dict, num_threads: int = 1, base_dir: str = "."):
        self.name = name
        self.cfg = dict(cfg)
        self.num_threads = int(cfg.get("num_threads", num_threads))
        self.base_dir = base_dir
        self.sample_rate = 16000
        self.threshold = float(cfg.get("threshold", self.default_threshold))

    def path(self, key: str, default: str | None = None) -> str | None:
        p = self.cfg.get(key, default)
        if p is None:
            return None
        p = os.path.expanduser(str(p))
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(self.base_dir, p))

    def load(self) -> None:
        pass

    def process(self, audio: np.ndarray) -> FrameScores:
        raise NotImplementedError

    # optional: adapters that read pre-computed results need to know which file
    def set_context(self, item: dict) -> None:
        self._item = item

    @property
    def measures_speed(self) -> bool:
        return True


def ort_session(path: str, num_threads: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.log_severity_level = 3
    return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
