"""Decode any audio/video file (mp3, mp4, m4a, wav, flac, ogg, ...) with any channel
count and sample rate into mono float32 at the target sample rate.

Default decoder = librosa: librosa.load(sr=None, mono=False) at the native rate/channels
-> librosa.to_mono (channel mean) -> librosa.resample (res_type, default soxr_hq).
librosa reads wav/flac/ogg/mp3 through soundfile; m4a/mp4/aac fall back to audioread,
which needs ffmpeg installed on the system.
Other decoders: "ffmpeg" (CLI + scipy resample_poly) and "pyav".
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

AUDIO_EXTS = {
    ".wav", ".mp3", ".mp4", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm",
    ".mkv", ".mov", ".amr", ".3gp", ".wma", ".aiff", ".aif", ".caf",
}


@dataclass
class AudioInfo:
    path: str
    orig_sr: int
    channels: int
    duration: float  # seconds, after resampling


def _probe_ffprobe(path: str) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels", "-of", "json", path],
        capture_output=True, text=True, check=True,
    ).stdout
    streams = json.loads(out).get("streams", [])
    if not streams:
        raise RuntimeError(f"No audio stream in {path}")
    return int(streams[0]["sample_rate"]), int(streams[0]["channels"])


def _decode_ffmpeg(path: str) -> tuple[np.ndarray, int]:
    sr, ch = _probe_ffprobe(path)
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn", "-map", "0:a:0",
         "-f", "f32le", "-acodec", "pcm_f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    x = np.frombuffer(raw, dtype=np.float32)
    n = (len(x) // ch) * ch
    return x[:n].reshape(-1, ch), sr


def _decode_pyav(path: str) -> tuple[np.ndarray, int]:
    import av  # optional dependency

    chunks, sr = [], None
    with av.open(path) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        sr = stream.codec_context.sample_rate
        for frame in container.decode(stream):
            arr = frame.to_ndarray()
            if arr.dtype.kind == "i":
                arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max + 1)
            else:
                arr = arr.astype(np.float32)
            nch = len(frame.layout.channels)
            if frame.format.is_planar:
                arr = arr.reshape(nch, -1).T            # (samples, ch)
            else:
                arr = arr.reshape(-1, nch)              # interleaved
            chunks.append(arr)
    if not chunks:
        raise RuntimeError(f"No audio decoded from {path}")
    return np.concatenate(chunks, axis=0), sr


def _decode_librosa(path: str) -> tuple[np.ndarray, int]:
    import warnings

    import librosa

    with warnings.catch_warnings():
        # audioread fallback for m4a/mp4 emits a FutureWarning in librosa >= 0.10
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message="PySoundFile failed")
        y, sr = librosa.load(path, sr=None, mono=False)
    y = np.atleast_2d(y)              # (channels, samples)
    return y.T, int(sr)               # -> (samples, channels)


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    frac = Fraction(sr_out, sr_in).limit_denominator(1000)
    return resample_poly(x, frac.numerator, frac.denominator).astype(np.float32)


def load_audio(path: str, sample_rate: int = 16000, decoder: str = "librosa",
               downmix: str = "mean", res_type: str = "soxr_hq") -> tuple[np.ndarray, AudioInfo]:
    """Return (mono float32 waveform in [-1, 1] at `sample_rate`, info)."""
    if decoder == "auto":
        try:
            import librosa  # noqa: F401
            decoder = "librosa"
        except ImportError:
            decoder = "ffmpeg" if shutil.which("ffmpeg") and shutil.which("ffprobe") else "pyav"
    if decoder == "librosa":
        x, sr = _decode_librosa(path)
    elif decoder == "ffmpeg":
        x, sr = _decode_ffmpeg(path)
    elif decoder == "pyav":
        x, sr = _decode_pyav(path)
    else:
        raise ValueError(f"Unknown decoder {decoder!r}")

    ch = x.shape[1]
    if downmix == "mean":
        mono = x.mean(axis=1)         # == librosa.to_mono
    elif downmix == "left":
        mono = x[:, 0]
    else:
        raise ValueError(f"Unknown downmix {downmix!r}")
    mono = mono.astype(np.float32)
    if sr != sample_rate:
        if decoder == "librosa":
            import librosa
            mono = librosa.resample(mono, orig_sr=sr, target_sr=sample_rate, res_type=res_type)
        else:
            mono = _resample(mono, sr, sample_rate)
    mono = np.clip(mono, -1.0, 1.0).astype(np.float32)
    return mono, AudioInfo(path, sr, ch, len(mono) / sample_rate)


def to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)
