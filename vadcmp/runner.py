"""Run every enabled model over every manifest item; cache frame scores + timings."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np

from vadcmp.audio import load_audio
from vadcmp.config import resolve
from vadcmp.dataset import read_manifest
from vadcmp.models import create_model
from vadcmp.timeline import FrameScores


def _cfg_hash(mcfg: dict) -> str:
    keep = {k: v for k, v in mcfg.items() if k not in ("enabled", "threshold", "postprocess", "smooth_s")}
    return hashlib.md5(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:10]


def cache_path(cfg: dict, model_name: str, item_id: str) -> str:
    return os.path.join(resolve(cfg, cfg["output_dir"]), "cache", model_name, item_id + ".npz")


def load_cached(cfg: dict, mcfg: dict, item: dict):
    p = cache_path(cfg, mcfg["name"], item["id"])
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=False)
    if str(z["cfg_hash"]) != _cfg_hash(mcfg):
        return None
    fs = FrameScores(z["start"], z["end"], z["prob"])
    return fs, {"duration": float(z["duration"]), "wall": float(z["wall"]), "cpu": float(z["cpu"])}


def _save(cfg, mcfg, item, fs: FrameScores, duration, wall, cpu):
    p = cache_path(cfg, mcfg["name"], item["id"])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez_compressed(p, start=fs.start, end=fs.end, prob=fs.prob, duration=duration,
                        wall=wall, cpu=cpu, cfg_hash=_cfg_hash(mcfg))


def selected_models(cfg: dict, only: list[str] | None = None):
    ms = [m for m in cfg["models"] if m.get("enabled", True)]
    if only:
        want = set(only)
        ms = [m for m in cfg["models"] if m["name"] in want]
        missing = want - {m["name"] for m in ms}
        if missing:
            raise SystemExit(f"Unknown model name(s): {', '.join(sorted(missing))}")
    return ms


def run_inference(cfg: dict, only: list[str] | None = None, force: bool = False,
                  limit: int | None = None, log=print):
    items = read_manifest(resolve(cfg, cfg["data"]["manifest"]))
    if limit:
        items = items[:limit]
    mcfgs = selected_models(cfg, only)
    acfg = cfg["audio"]

    # load models once (load time is reported separately)
    models, load_info = {}, {}
    for mc in mcfgs:
        t0 = time.perf_counter()
        try:
            m = create_model(mc, cfg["num_threads"], cfg["_base_dir"])
            m.load()
            models[mc["name"]] = (m, mc)
            load_info[mc["name"]] = {"load_s": time.perf_counter() - t0, "error": ""}
            log(f"[load] {mc['name']:<14} ok ({load_info[mc['name']]['load_s']:.2f}s)")
        except Exception as e:  # keep going with the other models
            load_info[mc["name"]] = {"load_s": float("nan"), "error": f"{type(e).__name__}: {e}"}
            log(f"[load] {mc['name']:<14} FAILED -> {type(e).__name__}: {e}")

    warmed = set()
    n_err = 0
    for k, item in enumerate(items, 1):
        todo = [(n, m, mc) for n, (m, mc) in models.items()
                if force or load_cached(cfg, mc, item) is None]
        if not todo:
            continue
        try:
            audio, info = load_audio(item["audio"], acfg["sample_rate"], acfg["decoder"],
                                     acfg["downmix"], acfg.get("res_type", "soxr_hq"))
        except Exception as e:
            log(f"[{k}/{len(items)}] decode FAILED {item['audio']}: {e}")
            n_err += 1
            continue
        for name, m, mc in todo:
            m.set_context(item)
            try:
                if name not in warmed and m.measures_speed:
                    m.process(audio[: 16000 * 2])  # warm-up (lazy init, caches)
                    warmed.add(name)
                c0, t0 = time.process_time(), time.perf_counter()
                fs = m.process(audio)
                wall, cpu = time.perf_counter() - t0, time.process_time() - c0
                if not m.measures_speed:
                    wall = cpu = float("nan")
                _save(cfg, mc, item, fs, info.duration, wall, cpu)
            except Exception as e:
                n_err += 1
                log(f"[{k}/{len(items)}] {name} FAILED on {item['audio']}: {type(e).__name__}: {e}")
                if os.environ.get("VADCMP_DEBUG"):
                    traceback.print_exc(file=sys.stderr)
        log(f"[{k}/{len(items)}] {os.path.basename(item['audio'])} ({info.duration:.1f}s, "
            f"{info.orig_sr} Hz, {info.channels} ch) -> {', '.join(n for n, _, _ in todo)}")

    out = resolve(cfg, cfg["output_dir"])
    os.makedirs(out, exist_ok=True)
    prev = {}
    lp = os.path.join(out, "load_times.json")
    if os.path.exists(lp):
        prev = json.load(open(lp))
    prev.update(load_info)
    json.dump(prev, open(lp, "w"), indent=2)
    return n_err
