"""Model registry. Imports are lazy so a missing optional dependency only affects its model."""
from __future__ import annotations

import importlib

REGISTRY = {
    "silero": ("vadcmp.models.silero", "SileroVAD"),
    "webrtc": ("vadcmp.models.webrtc", "WebRTCVAD"),
    "ten": ("vadcmp.models.ten", "TenVAD"),
    "pyannote": ("vadcmp.models.pyannote_vad", "PyannoteVAD"),
    "firered": ("vadcmp.models.firered", "FireRedVAD"),
    "yamnet": ("vadcmp.models.yamnet", "YamnetVAD"),
    "precomputed": ("vadcmp.models.precomputed", "PrecomputedVAD"),
}


def create_model(cfg: dict, num_threads: int, base_dir: str):
    t = cfg["type"]
    if t not in REGISTRY:
        raise ValueError(f"Unknown model type {t!r}. Known: {', '.join(REGISTRY)}")
    mod, cls = REGISTRY[t]
    klass = getattr(importlib.import_module(mod), cls)
    return klass(cfg.get("name", t), cfg, num_threads=num_threads, base_dir=base_dir)
