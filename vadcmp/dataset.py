"""Manifest (audio <-> ground truth <-> category) loading and automatic pairing."""
from __future__ import annotations

import csv
import hashlib
import os
import re
from collections import defaultdict

from vadcmp.audio import AUDIO_EXTS

GT_EXTS = {".txt", ".lab", ".csv", ".tsv", ".label", ".labels"}


def _key(stem: str) -> str:
    s = stem.lower()
    s = re.sub(r"^(gt|groundtruth|label|labels)[_\-\s]*", "", s)
    s = re.sub(r"[_\-\s]*(gt|label|labels)$", "", s)
    return re.sub(r"[\s\-]+", "_", s)


def _category(root: str, path: str) -> str:
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    return parts[0] if len(parts) > 1 else ""


def file_id(audio_rel: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.splitext(audio_rel)[0])[-80:]
    return f"{stem}_{hashlib.md5(audio_rel.encode()).hexdigest()[:6]}"


def find_groundtruth(gt_root: str, gt_dir: str = "ground_truth"):
    """GT files = files whose path under `gt_root` passes through a folder named `gt_dir`
    (case-insensitive). Everything else is ignored, even with the same file name, e.g.

        Model/Asm/ground_truth/asm_1.txt   -> GT, category "Asm"
        Model/Asm/model_v1/asm_1.txt       -> ignored
        Model/Asm/lib_out/asm_1.txt        -> ignored

    Returns (gts, ignored) where gts = {key: [(category, path), ...]} and
    ignored = {folder (relative to gt_root): n_files}."""
    want = gt_dir.lower()
    gts, ignored = defaultdict(list), defaultdict(int)
    for dp, dirs, files in os.walk(gt_root):
        dirs.sort()
        rel_dir = os.path.relpath(dp, gt_root)
        parts = [] if rel_dir == "." else rel_dir.split(os.sep)
        inside = want in (q.lower() for q in parts)
        for fn in sorted(files):
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in GT_EXTS:
                continue
            if not inside:
                ignored[rel_dir] += 1
                continue
            i = [q.lower() for q in parts].index(want)
            category = parts[0] if i > 0 else ""          # Model/<category>/.../ground_truth/
            gts[_key(stem)].append((category, os.path.join(dp, fn)))
    return gts, dict(ignored)


def scan(audio_root: str, gt_root: str, gt_dir: str = "ground_truth"):
    """Pair audio files with GT files by normalised stem (+ category when ambiguous).

    GT: only files inside `gt_dir` folders, e.g. Model/<category>/ground_truth/<name>.txt.
    Audio: any supported file under audio_root; category = first folder under audio_root.
    e.g. Model/Asm/ground_truth/asm_1.txt  <->  Audio/Asm/asm_1.mp4  (category "Asm")"""
    gts, ignored = find_groundtruth(gt_root, gt_dir)
    if not gts:
        raise SystemExit(f"Không tìm thấy file GT nào trong thư mục '{gt_dir}/' dưới {gt_root}")
    rows, unmatched = [], []
    for dp, dirs, files in os.walk(audio_root):
        dirs.sort()
        for fn in sorted(files):
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in AUDIO_EXTS:
                continue
            ap = os.path.join(dp, fn)
            acat = _category(audio_root, ap)
            cands = gts.get(_key(stem), [])
            if len(cands) > 1:
                cands = [c for c in cands if c[0].lower() == acat.lower()] or cands
            if len(cands) != 1:
                unmatched.append((ap, len(cands)))
                continue
            gcat, gp = cands[0]
            rows.append({"audio": ap, "gt": gp, "category": gcat or acat})
    n_gt = sum(len(v) for v in gts.values())
    used = {r["gt"] for r in rows}
    unused_gt = [p for v in gts.values() for _, p in v if p not in used]
    return rows, unmatched, {"n_gt": n_gt, "ignored": ignored, "unused_gt": unused_gt}


def write_manifest(rows, path: str):
    base = os.path.dirname(os.path.abspath(path))
    os.makedirs(base, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["audio", "gt", "category"])
        w.writeheader()
        for r in rows:
            w.writerow({"audio": os.path.relpath(r["audio"], base),
                        "gt": os.path.relpath(r["gt"], base), "category": r["category"]})


def read_manifest(path: str):
    base = os.path.dirname(os.path.abspath(path))
    items = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            a = r["audio"].strip()
            g = r["gt"].strip()
            ap = a if os.path.isabs(a) else os.path.normpath(os.path.join(base, a))
            gp = g if os.path.isabs(g) else os.path.normpath(os.path.join(base, g))
            items.append({"id": r.get("id") or file_id(a), "audio": ap, "gt": gp,
                          "category": (r.get("category") or "").strip() or "all"})
    return items
