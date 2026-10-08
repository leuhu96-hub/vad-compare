"""Ground-truth parsing and the evaluation grid.

Format (one segment per line; separators: tab / space / comma / semicolon; '#' = comment):
    start end label [extra columns...]      e.g.  "0 0.5 Speech"   "0.5 1 Non-Speech 1"
Column 1 = start (s), column 2 = end (s), column 3 = label. Column 4+ is IGNORED.
Labels (case-insensitive): Speech / Non-Speech, nonspeech, non_speech, 1 / 0, sp / ns ...
A file with only a label per line is read as consecutive `bin_size` bins.
"""
from __future__ import annotations

import re

import numpy as np

SPEECH = {"speech", "sp", "s", "1", "true", "yes", "voice", "v", "vocal", "talk"}
NONSPEECH = {"non-speech", "nonspeech", "non_speech", "non speech", "ns", "n", "0", "false",
             "no", "silence", "sil", "noise", "music", "nospeech", "no-speech", "other"}
_SPLIT = re.compile(r"[,\t;]+|\s+")


def _label_to_int(tok: str) -> int | None:
    t = tok.strip().strip('"\'').lower()
    if t in SPEECH:
        return 1
    if t in NONSPEECH:
        return 0
    return None


def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def parse_groundtruth(path: str, bin_size: float = 0.5) -> np.ndarray:
    """Return array (N, 3): start, end, label (1 speech / 0 non-speech)."""
    rows, seq = [], []
    with open(path, encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            # keep "Non Speech" / "non_speech" as one token before splitting on whitespace
            line = re.sub(r"(?i)\bnon[\s_]+speech\b", "non-speech", line)
            toks = [t for t in _SPLIT.split(line) if t]
            if len(toks) == 1:                       # label-only line
                lab = _label_to_int(toks[0])
                if lab is None:
                    continue                         # header
                seq.append(lab)
                continue
            if len(toks) < 3 or not (_is_number(toks[0]) and _is_number(toks[1])):
                continue                             # header such as "start end label"
            lab = _label_to_int(toks[2])             # column 3; columns 4+ ignored
            if lab is None:
                raise ValueError(f"{path}:{lineno}: unknown label {toks[2]!r} in {line!r}")
            s, e = float(toks[0]), float(toks[1])
            if e <= s:
                raise ValueError(f"{path}:{lineno}: end <= start in {line!r}")
            rows.append((s, e, lab))
    if rows and seq:
        raise ValueError(f"{path}: mixes timed and label-only lines")
    if seq:
        return np.array([(i * bin_size, (i + 1) * bin_size, l) for i, l in enumerate(seq)], dtype=np.float64)
    if not rows:
        raise ValueError(f"{path}: no ground-truth lines found")
    gt = np.array(rows, dtype=np.float64)
    return gt[np.argsort(gt[:, 0])]


def gt_grid(gt: np.ndarray, bin_size: float, duration: float | None = None):
    """Fixed grid of `bin_size` bins over the labelled span.

    Returns (edges, labels) where labels is 1/0 by majority overlap, or -1 for bins that
    no GT segment covers (ignored in scoring)."""
    end = gt[:, 1].max()
    if duration is not None:
        end = min(end, duration + bin_size / 2)
    n = max(1, int(np.ceil(end / bin_size - 1e-6)))
    edges = np.arange(n + 1) * bin_size
    sp = np.zeros(n)
    ns = np.zeros(n)
    for s, e, lab in gt:
        a = max(0, int(np.floor(s / bin_size)))
        b = min(n, int(np.ceil(e / bin_size)))
        for i in range(a, b):
            ov = min(e, edges[i + 1]) - max(s, edges[i])
            if ov > 0:
                (sp if lab == 1 else ns)[i] += ov
    labels = np.where(sp + ns <= 1e-9, -1, (sp > ns).astype(int))
    return edges, labels.astype(np.int8)


def boundary_mask(labels: np.ndarray, tolerance: int = 1) -> np.ndarray:
    """True for bins within `tolerance` bins of a speech/non-speech transition."""
    mask = np.zeros(len(labels), dtype=bool)
    if tolerance <= 0:
        return mask
    for t in range(1, len(labels)):
        a, b = labels[t - 1], labels[t]
        if a >= 0 and b >= 0 and a != b:
            mask[max(0, t - tolerance):min(len(labels), t + tolerance)] = True
    return mask
