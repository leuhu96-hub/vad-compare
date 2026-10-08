"""Kaldi-compatible log-mel fbank + Kaldi CMVN reader (used by FireRedVAD).

Uses `kaldi_native_fbank` when installed (exact upstream parity); otherwise a numpy
re-implementation of Kaldi's fbank with the same defaults FireRedVAD uses:
25 ms / 10 ms, 80 bins, povey window, pre-emphasis 0.97, DC removal, snip_edges, no dither.
"""
from __future__ import annotations

import struct

import numpy as np

_EPS = float(np.finfo(np.float32).eps)


def _mel(f):
    return 1127.0 * np.log(1.0 + np.asarray(f) / 700.0)


def _mel_banks(num_bins: int, n_fft: int, sr: int, low: float = 20.0, high: float = 0.0):
    nyq = sr / 2.0
    high = nyq + high if high <= 0 else high
    n_bins_fft = n_fft // 2
    fft_freqs = np.arange(n_bins_fft) * (sr / n_fft)
    mel_f = _mel(fft_freqs)
    mlow, mhigh = _mel(low), _mel(high)
    delta = (mhigh - mlow) / (num_bins + 1)
    w = np.zeros((num_bins, n_bins_fft + 1))
    for b in range(num_bins):
        left, center, right = mlow + b * delta, mlow + (b + 1) * delta, mlow + (b + 2) * delta
        up = (mel_f - left) / (center - left)
        down = (right - mel_f) / (right - center)
        tri = np.maximum(0.0, np.minimum(up, down))
        tri[(mel_f <= left) | (mel_f >= right)] = 0.0
        w[b, :n_bins_fft] = tri
    return w


def fbank_numpy(wave_int16_scale: np.ndarray, sr: int = 16000, num_bins: int = 80,
                frame_ms: float = 25.0, shift_ms: float = 10.0) -> np.ndarray:
    x = np.asarray(wave_int16_scale, dtype=np.float64)
    flen = int(sr * frame_ms / 1000)
    fshift = int(sr * shift_ms / 1000)
    if len(x) < flen:
        return np.zeros((0, num_bins), dtype=np.float32)
    n = 1 + (len(x) - flen) // fshift
    idx = np.arange(flen)[None, :] + fshift * np.arange(n)[:, None]
    frames = x[idx]
    frames = frames - frames.mean(axis=1, keepdims=True)
    pre = frames.copy()
    pre[:, 1:] -= 0.97 * frames[:, :-1]
    pre[:, 0] -= 0.97 * frames[:, 0]
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(flen) / (flen - 1))) ** 0.85
    pre *= win
    n_fft = 1 << (flen - 1).bit_length()
    spec = np.abs(np.fft.rfft(pre, n=n_fft, axis=1)) ** 2
    mel = spec @ _mel_banks(num_bins, n_fft, sr).T
    return np.log(np.maximum(mel, _EPS)).astype(np.float32)


def fbank(wave_int16_scale: np.ndarray, sr: int = 16000, num_bins: int = 80) -> np.ndarray:
    try:
        import kaldi_native_fbank as knf
    except ImportError:
        return fbank_numpy(wave_int16_scale, sr, num_bins)
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = sr
    opts.frame_opts.frame_length_ms = 25
    opts.frame_opts.frame_shift_ms = 10
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = True
    opts.mel_opts.num_bins = num_bins
    opts.mel_opts.debug_mel = False
    fb = knf.OnlineFbank(opts)
    fb.accept_waveform(sr, np.asarray(wave_int16_scale, dtype=np.float32).tolist())
    if fb.num_frames_ready == 0:
        return np.zeros((0, num_bins), dtype=np.float32)
    return np.vstack([fb.get_frame(i) for i in range(fb.num_frames_ready)]).astype(np.float32)


def read_kaldi_matrix(path: str) -> np.ndarray:
    """Read a single Kaldi matrix (binary 'DM'/'FM' or text) - e.g. cmvn.ark."""
    data = open(path, "rb").read()
    i = data.find(b"\x00B")
    if i >= 0:
        p = i + 2
        tok_end = data.index(b" ", p)
        tok = data[p:tok_end].decode()
        p = tok_end + 1
        assert data[p] == 4
        rows = struct.unpack("<i", data[p + 1:p + 5])[0]
        p += 5
        assert data[p] == 4
        cols = struct.unpack("<i", data[p + 1:p + 5])[0]
        p += 5
        dtype = {"DM": "<f8", "FM": "<f4"}[tok]
        return np.frombuffer(data, dtype=dtype, count=rows * cols, offset=p).reshape(rows, cols).astype(np.float64)
    text = data.decode()
    body = text[text.index("[") + 1:text.index("]")]
    return np.array([[float(v) for v in ln.split()] for ln in body.strip().splitlines() if ln.strip()])


class KaldiCMVN:
    def __init__(self, path: str):
        stats = read_kaldi_matrix(path)
        dim = stats.shape[1] - 1
        count = stats[0, dim]
        mean = stats[0, :dim] / count
        var = np.maximum(stats[1, :dim] / count - mean ** 2, 1e-20)
        self.mean = mean
        self.istd = 1.0 / np.sqrt(var)

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        return ((feats - self.mean) * self.istd).astype(np.float32)
