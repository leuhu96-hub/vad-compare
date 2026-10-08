"""Time-axis utilities shared by every model: frame scores, segments, post-processing,
and re-binning onto the ground-truth grid."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FrameScores:
    """Speech probability per (non-overlapping) frame interval, in seconds."""
    start: np.ndarray
    end: np.ndarray
    prob: np.ndarray

    def __post_init__(self):
        self.start = np.asarray(self.start, dtype=np.float64)
        self.end = np.asarray(self.end, dtype=np.float64)
        self.prob = np.asarray(self.prob, dtype=np.float64)

    @classmethod
    def from_hops(cls, prob, hop: float, offset: float = 0.0, duration: float | None = None):
        prob = np.asarray(prob, dtype=np.float64)
        start = offset + np.arange(len(prob)) * hop
        end = start + hop
        if duration is not None:
            keep = start < duration
            start, end, prob = start[keep], np.minimum(end[keep], duration), prob[keep]
        return cls(start, end, prob)

    @classmethod
    def from_centers(cls, centers, prob, step: float, duration: float | None = None):
        """Tile overlapping frames (e.g. pyannote) into intervals centred on each frame."""
        c = np.asarray(centers, dtype=np.float64)
        start = np.maximum(c - step / 2, 0.0)
        end = c + step / 2
        if duration is not None:
            keep = start < duration
            start, end, prob = start[keep], np.minimum(end[keep], duration), np.asarray(prob)[keep]
        return cls(start, end, prob)

    def tiled(self) -> "FrameScores":
        """Sort and clip so intervals never overlap (needed for exact integration)."""
        o = np.argsort(self.start, kind="stable")
        s, e, p = self.start[o], self.end[o].copy(), self.prob[o]
        if len(s) > 1:
            e[:-1] = np.minimum(e[:-1], s[1:])
        e = np.maximum(e, s)
        return FrameScores(s, e, p)

    def smoothed(self, window_s: float) -> "FrameScores":
        if window_s <= 0 or len(self.prob) < 2:
            return self
        hop = float(np.median(np.diff(self.start)))
        k = max(1, int(round(window_s / hop)))
        if k <= 1:
            return self
        kernel = np.ones(k) / k
        padded = np.pad(self.prob, (k // 2, k - 1 - k // 2), mode="edge")
        return FrameScores(self.start, self.end, np.convolve(padded, kernel, mode="valid"))


# ----------------------------------------------------------------------------- segments
def merge_segments(seg: np.ndarray, gap: float = 0.0) -> np.ndarray:
    """Union segments; also join ones separated by < `gap` seconds."""
    seg = np.asarray(seg, dtype=np.float64).reshape(-1, 2)
    if len(seg) == 0:
        return seg
    seg = seg[np.argsort(seg[:, 0])]
    out = [seg[0].copy()]
    for s, e in seg[1:]:
        if s - out[-1][1] < gap or s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append(np.array([s, e]))
    return np.array(out)


def binarize(fs: FrameScores, threshold: float) -> np.ndarray:
    """Frames with prob >= threshold -> contiguous speech segments (N, 2)."""
    fs = fs.tiled()
    on = fs.prob >= threshold
    if not on.any():
        return np.zeros((0, 2))
    return merge_segments(np.stack([fs.start[on], fs.end[on]], axis=1), gap=1e-6)


def postprocess(seg: np.ndarray, duration: float, pad_before: float = 0.0,
                pad_after: float = 0.0, merge_gap: float = 0.0, min_speech: float = 0.0,
                order=("pad", "merge", "drop")) -> np.ndarray:
    seg = np.asarray(seg, dtype=np.float64).reshape(-1, 2).copy()
    for step in order:
        if len(seg) == 0:
            break
        if step == "pad":
            seg[:, 0] = np.maximum(seg[:, 0] - pad_before, 0.0)
            seg[:, 1] = np.minimum(seg[:, 1] + pad_after, duration)
            seg = merge_segments(seg)
        elif step == "merge":
            seg = merge_segments(seg, gap=merge_gap)
        elif step == "drop":
            seg = seg[(seg[:, 1] - seg[:, 0]) >= min_speech]
        else:
            raise ValueError(f"Unknown post-process step {step!r}")
    return seg


# ----------------------------------------------------------------------------- binning
def _cumulative(starts, ends, values):
    """Breakpoints and running integral of a piecewise-constant function."""
    t = np.empty(2 * len(starts) + 1)
    F = np.empty_like(t)
    t[0], F[0] = (starts[0] if len(starts) else 0.0), 0.0
    acc = 0.0
    for i, (s, e, v) in enumerate(zip(starts, ends, values)):
        t[2 * i + 1], F[2 * i + 1] = s, acc
        acc += (e - s) * v
        t[2 * i + 2], F[2 * i + 2] = e, acc
    return t, F


def _integrate(starts, ends, values, edges):
    if len(starts) == 0:
        return np.zeros(len(edges) - 1)
    t, F = _cumulative(starts, ends, values)
    G = np.interp(edges, t, F, left=0.0, right=F[-1])
    return np.diff(G)


def bin_scores(fs: FrameScores, edges: np.ndarray, agg: str = "mean") -> np.ndarray:
    """Score per GT bin. `mean` = overlap-weighted mean, `max` = max over touching frames.
    Bins with no frame coverage get 0."""
    fs = fs.tiled()
    if agg == "mean":
        num = _integrate(fs.start, fs.end, fs.prob, edges)
        den = _integrate(fs.start, fs.end, np.ones_like(fs.prob), edges)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(den > 1e-9, num / np.maximum(den, 1e-12), 0.0)
        return out
    if agg == "max":
        out = np.zeros(len(edges) - 1)
        for i in range(len(edges) - 1):
            m = (fs.end > edges[i]) & (fs.start < edges[i + 1])
            out[i] = fs.prob[m].max() if m.any() else 0.0
        return out
    raise ValueError(f"Unknown bin aggregation {agg!r}")


def segments_to_bins(seg: np.ndarray, edges: np.ndarray, min_overlap: float = 0.5) -> np.ndarray:
    """1 if speech covers >= `min_overlap` of the bin, else 0."""
    seg = merge_segments(seg)
    cov = _integrate(seg[:, 0], seg[:, 1], np.ones(len(seg)), edges) if len(seg) else np.zeros(len(edges) - 1)
    width = np.diff(edges)
    return (cov >= min_overlap * width - 1e-9).astype(np.int8)


def window_scores_to_hops(win_scores: np.ndarray, hop: float, window: float,
                          n_hops: int, agg: str = "mean") -> np.ndarray:
    """Overlapping windows (start = i*hop, length = window) -> one score per hop
    ("rescore per hop"): every hop takes the mean/max of the windows that cover it."""
    win_scores = np.asarray(win_scores, dtype=np.float64)
    span = max(1, int(round(window / hop)))
    if agg == "mean":
        acc = np.zeros(n_hops)
        cnt = np.zeros(n_hops)
        for i, s in enumerate(win_scores):
            a, b = i, min(i + span, n_hops)
            acc[a:b] += s
            cnt[a:b] += 1
        return np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
    if agg == "max":
        out = np.zeros(n_hops)
        for i, s in enumerate(win_scores):
            a, b = i, min(i + span, n_hops)
            out[a:b] = np.maximum(out[a:b], s)
        return out
    raise ValueError(f"Unknown overlap aggregation {agg!r}")
