from __future__ import annotations

import copy
import os

import yaml

DEFAULTS = {
    "data": {"manifest": "data/manifest.csv", "gt_bin": 0.5},
    "audio": {"sample_rate": 16000, "decoder": "librosa", "downmix": "mean", "res_type": "soxr_hq"},
    "output_dir": "out",
    "num_threads": 1,
    "models": [],
    # Applied to every model's binary decisions (same rules as the on-device lib)
    "postprocess": {
        "pad_before": 0.10, "pad_after": 0.12, "merge_gap": 0.50, "min_speech": 0.32,
        "order": ["pad", "merge", "drop"],
    },
    "scoring": {
        "bin_agg": "mean",          # model score per GT bin: mean | max
        "min_overlap": 0.5,         # segment -> bin is speech if it covers >= this fraction
        "boundary_tolerance": 1,    # bins on each side of a GT transition counted as boundary
    },
    "report": {"timelines": 12, "timeline_sort": "worst"},
}


def _merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = _merge(DEFAULTS, yaml.safe_load(f) or {})
    cfg["_base_dir"] = os.path.dirname(os.path.abspath(path))
    names = set()
    for m in cfg["models"]:
        m.setdefault("name", m["type"])
        m.setdefault("enabled", True)
        if "threshold" not in m:
            m["threshold"] = default_threshold(m["type"])
        if m["name"] in names:
            raise ValueError(f"Duplicate model name {m['name']!r}")
        names.add(m["name"])
    return cfg


def default_threshold(model_type: str) -> float:
    import importlib

    from vadcmp.models import REGISTRY

    if model_type not in REGISTRY:
        raise ValueError(f"Unknown model type {model_type!r}. Known: {', '.join(REGISTRY)}")
    mod, cls = REGISTRY[model_type]
    return float(getattr(importlib.import_module(mod), cls).default_threshold)


def resolve(cfg: dict, p: str) -> str:
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(cfg["_base_dir"], p))


def model_postprocess(cfg: dict, mcfg: dict) -> dict | None:
    """Model-level `postprocess:` overrides the global one; `postprocess: none` disables it."""
    pp = mcfg.get("postprocess", "global")
    if pp in (None, "none", False):
        return None
    if pp == "global":
        return cfg["postprocess"]
    return _merge(cfg["postprocess"], pp)
