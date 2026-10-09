"""FireRedVAD chạy trực tiếp bằng ONNX Runtime, KHÔNG dùng package `fireredvad`.

Mỗi bước là một hàm riêng, có thể gọi lẻ để xem input/output:

    step1_load_audio      file bất kỳ (sr, số kênh tuỳ ý)        -> x [C, N] float32 [-1, 1], sr
    step2_downmix         [C, N]                                  -> [N]
    step3_resample        [N] @ sr                                -> [N'] @ 16000 Hz
    step4_int16_scale     [-1, 1]                                 -> thang int16 (float32)
    step4b_dither         (tuỳ chọn) + nhiễu Gauss rất nhỏ, mặc định chỉ bật khi input < 16 kHz
    step5_fbank           [N'] @ 16 kHz                           -> [T, 80] log-mel (Kaldi fbank)
        5a frames         snip_edges: T = 1 + (N' - 400) // 160   -> [T, 400]
        5b remove DC      trừ trung bình mỗi frame
        5c pre-emphasis   y[n] = x[n] - 0.97 x[n-1]
        5d povey window   (0.5 - 0.5 cos(2πn/399))^0.85
        5e FFT 512        |X|^2                                   -> [T, 257]
        5f mel 80 band    20 Hz .. 8000 Hz (Kaldi mel)            -> [T, 80]
        5g log            log(max(mel, FLT_EPSILON))
    step6_cmvn            (feat - mean) * istd, mean/istd từ cmvn.ark -> [T, 80]
    step7_onnx_*          feat [1, T, 80] -> probs [T] (xác suất speech, mỗi frame 10 ms)
    step8_postprocess     probs -> decisions [T] (0/1) -> segments [(start_s, end_s)]

Chỉ cần: numpy, onnxruntime, librosa (bước 1 + 3). Không cần torch / kaldi-native-fbank / fireredvad.
"""
from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16000
FRAME_LEN = 400          # 25 ms @ 16 kHz
FRAME_SHIFT = 160        # 10 ms @ 16 kHz
FRAME_SHIFT_S = 0.010
FRAME_LEN_S = 0.025
N_FFT = 512              # Kaldi: round_to_power_of_two(400)
N_MELS = 80
PREEMPH = 0.97
LOW_FREQ, HIGH_FREQ = 20.0, 8000.0   # Kaldi mặc định: high_freq = 0 -> Nyquist
FLT_EPS = float(np.finfo(np.float32).eps)


# ============================================================================ step 1-4: audio
def step1_load_audio(path: str):
    """Đọc mọi định dạng librosa đọc được, giữ nguyên sample rate và số kênh.

    Returns: x [C, N] float32 trong [-1, 1], sr (int).
    wav/flac/ogg/mp3 qua soundfile; m4a/mp4/aac qua audioread (cần ffmpeg trên máy).
    """
    import warnings

    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message="PySoundFile failed")
        y, sr = librosa.load(path, sr=None, mono=False)
    y = np.atleast_2d(np.asarray(y, dtype=np.float32))     # mono -> [1, N]
    return y, int(sr)


def step2_downmix(x: np.ndarray, mode: str = "mean") -> np.ndarray:
    """[C, N] -> [N]. mean = trung bình các kênh (giống librosa.to_mono); hoặc chọn 1 kênh: '0', '1'..."""
    if x.ndim == 1:
        return x.astype(np.float32)
    if mode == "mean":
        return x.mean(axis=0).astype(np.float32)
    return x[int(mode)].astype(np.float32)


def step3_resample(x: np.ndarray, sr: int, target_sr: int = SAMPLE_RATE,
                   res_type: str = "soxr_hq") -> np.ndarray:
    """Đưa về 16 kHz. FireRedVAD chỉ nhận 16 kHz (upstream assert sample_rate == 16000)."""
    if sr == target_sr:
        return x.astype(np.float32)
    import librosa

    return librosa.resample(x, orig_sr=sr, target_sr=target_sr, res_type=res_type).astype(np.float32)


def step4_int16_scale(x: np.ndarray) -> np.ndarray:
    """[-1, 1] -> thang int16. Upstream đọc wav bằng soundfile dtype='int16' rồi đưa thẳng giá trị
    nguyên vào fbank, nên fbank/CMVN được train trên biên độ cỡ ±32768. Bỏ bước này thì log-mel
    lệch ~ log(32768²) ≈ 20.8 so với lúc train -> model hỏng."""
    return np.clip(x * 32768.0, -32768.0, 32767.0).astype(np.float32)


def step4b_dither(x: np.ndarray, amount: float, seed: int = 0) -> np.ndarray:
    """Cộng nhiễu Gauss N(0, amount²) (đơn vị int16) - giống Kaldi `dither`.

    Upstream suy luận với dither=0 vì input luôn là 16 kHz băng rộng. Với input gốc < 16 kHz
    (vd. 8 kHz điện thoại), sau upsample dải 4-8 kHz gần như bằng 0 -> log-mel các band trên rơi
    xuống log(FLT_EPSILON) ≈ -15.9, lệch xa phân bố lúc train -> model hỏng (thử 8k->16k bằng FFT:
    quyết định chỉ khớp 58% với bản 16k gốc). amount=1.0 đưa về ~94%."""
    if amount <= 0:
        return x
    rng = np.random.default_rng(seed)
    return (x + rng.standard_normal(len(x)).astype(np.float32) * np.float32(amount)).astype(np.float32)


# ============================================================================ step 5: fbank
def _mel_scale(f):
    return 1127.0 * np.log(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def kaldi_mel_banks(num_bins=N_MELS, n_fft=N_FFT, sr=SAMPLE_RATE, low=LOW_FREQ, high=HIGH_FREQ):
    """Ma trận mel [num_bins, n_fft//2 + 1] giống Kaldi MelBanks (tam giác trên thang mel,
    bỏ bin Nyquist)."""
    n_half = n_fft // 2
    fft_mel = _mel_scale(np.arange(n_half) * (sr / n_fft))
    mlo, mhi = _mel_scale(low), _mel_scale(high)
    delta = (mhi - mlo) / (num_bins + 1)
    W = np.zeros((num_bins, n_half + 1), dtype=np.float64)
    for b in range(num_bins):
        left, center, right = mlo + b * delta, mlo + (b + 1) * delta, mlo + (b + 2) * delta
        up = (fft_mel - left) / (center - left)
        down = (right - fft_mel) / (right - center)
        tri = np.maximum(0.0, np.minimum(up, down))
        tri[(fft_mel <= left) | (fft_mel >= right)] = 0.0
        W[b, :n_half] = tri
    return W


_MEL = kaldi_mel_banks()
_POVEY = ((0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FRAME_LEN) / (FRAME_LEN - 1))) ** 0.85)


def step5a_frames(x: np.ndarray) -> np.ndarray:
    """snip_edges=True: chỉ lấy frame nằm trọn trong tín hiệu. [N] -> [T, 400]."""
    if len(x) < FRAME_LEN:
        return np.zeros((0, FRAME_LEN), dtype=np.float64)
    T = 1 + (len(x) - FRAME_LEN) // FRAME_SHIFT
    idx = np.arange(FRAME_LEN)[None, :] + FRAME_SHIFT * np.arange(T)[:, None]
    return x.astype(np.float64)[idx]


def step5b_remove_dc(frames):
    return frames - frames.mean(axis=1, keepdims=True)


def step5c_preemphasis(frames, coeff=PREEMPH):
    out = frames.copy()
    out[:, 1:] -= coeff * frames[:, :-1]
    out[:, 0] -= coeff * frames[:, 0]          # Kaldi: x[0] -= c * x[0]
    return out


def step5d_window(frames):
    return frames * _POVEY


def step5e_power_spectrum(frames):
    """Zero-pad 400 -> 512, rFFT, |X|^2 -> [T, 257]."""
    return np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) ** 2


def step5f_mel(power):
    return power @ _MEL.T                      # [T, 257] x [257, 80] -> [T, 80]


def step5g_log(mel):
    return np.log(np.maximum(mel, FLT_EPS)).astype(np.float32)


def step5_fbank(x_int16_scale: np.ndarray, return_steps: bool = False):
    s = {}
    s["5a_frames"] = f = step5a_frames(x_int16_scale)
    s["5b_dc"] = f = step5b_remove_dc(f)
    s["5c_preemph"] = f = step5c_preemphasis(f)
    s["5d_window"] = f = step5d_window(f)
    s["5e_power"] = p = step5e_power_spectrum(f)
    s["5f_mel"] = m = step5f_mel(p)
    s["5g_logmel"] = feat = step5g_log(m)
    return (feat, s) if return_steps else feat


# ============================================================================ step 6: CMVN
def read_kaldi_matrix(path: str) -> np.ndarray:
    """Đọc 1 ma trận Kaldi (binary 'DM'/'FM' hoặc text). cmvn.ark là ma trận [2, 81]:
    hàng 0 = tổng x theo từng chiều + count ở cột cuối, hàng 1 = tổng x²."""
    data = open(path, "rb").read()
    i = data.find(b"\x00B")
    if i >= 0:
        p = i + 2
        end = data.index(b" ", p)
        tok = data[p:end].decode()
        p = end + 1
        assert data[p] == 4
        rows = struct.unpack("<i", data[p + 1:p + 5])[0]
        p += 5
        assert data[p] == 4
        cols = struct.unpack("<i", data[p + 1:p + 5])[0]
        p += 5
        dt = {"DM": "<f8", "FM": "<f4"}[tok]
        return np.frombuffer(data, dtype=dt, count=rows * cols, offset=p).reshape(rows, cols).astype(np.float64)
    text = data.decode()
    body = text[text.index("[") + 1:text.index("]")]
    return np.array([[float(v) for v in ln.split()] for ln in body.strip().splitlines() if ln.strip()])


def load_cmvn(path: str):
    """-> mean [80], istd [80] (giống fireredvad.core.audio_feat.CMVN)."""
    st = read_kaldi_matrix(path)
    dim = st.shape[1] - 1
    count = st[0, dim]
    mean = st[0, :dim] / count
    var = np.maximum(st[1, :dim] / count - mean ** 2, 1e-20)
    return mean, 1.0 / np.sqrt(var)


def step6_cmvn(feat: np.ndarray, mean: np.ndarray, istd: np.ndarray) -> np.ndarray:
    return ((feat - mean) * istd).astype(np.float32)


# ============================================================================ step 7: ONNX
def _session(path: str, threads: int = 1):
    import onnxruntime as ort

    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    o.log_severity_level = 3
    return ort.InferenceSession(path, sess_options=o, providers=["CPUExecutionProvider"])


def describe_session(sess) -> str:
    rows = [f"  IN  {i.name:<10} {i.shape} {i.type}" for i in sess.get_inputs()]
    rows += [f"  OUT {o.name:<10} {o.shape} {o.type}" for o in sess.get_outputs()]
    return "\n".join(rows)


def step7_onnx_nonstream(sess, feat: np.ndarray, chunk_max_frame: int = 30000) -> np.ndarray:
    """fireredvad_vad.onnx: feat [1, T, 80] -> probs [1, T, 1].
    Model nhìn cả quá khứ lẫn tương lai (lookahead) nên cần cả file; file > 300 s
    được cắt mỗi 30000 frame như upstream."""
    if len(feat) == 0:
        return np.zeros(0, np.float32)
    out = []
    for a in range(0, len(feat), chunk_max_frame):
        probs = sess.run(None, {"feat": feat[None, a:a + chunk_max_frame]})[0]
        out.append(probs.reshape(-1))
    return np.concatenate(out).astype(np.float32)


def step7_onnx_stream_full(sess, feat: np.ndarray, chunk_max_frame: int = 30000) -> np.ndarray:
    """fireredvad_stream_vad.onnx (causal, không có cache input): chạy cả chuỗi 1 lần.
    Outputs: probs [1, T, 1] + 8 cache [1, 128, 19] (bỏ qua)."""
    if len(feat) == 0:
        return np.zeros(0, np.float32)
    out = []
    for a in range(0, len(feat), chunk_max_frame):
        out.append(sess.run(["probs"], {"feat": feat[None, a:a + chunk_max_frame]})[0].reshape(-1))
    return np.concatenate(out).astype(np.float32)


def step7_onnx_stream_cached(sess, feat: np.ndarray, chunk_frames: int = 10):
    """fireredvad_stream_vad_with_cache.onnx: mô phỏng streaming thật.

    Mỗi lần gọi: feat [1, t, 80] (t = chunk_frames, chunk cuối có thể ngắn hơn),
                 caches_in [8, 1, 128, 19]  (8 khối DFSMN, 128 = projection, 19 = lookback (N1-1)*S1)
    Trả về:      probs [1, t, 1], caches_out [8, 1, 128, 19] -> đưa vào lần gọi sau.
    Cache bắt đầu bằng 0 cho mỗi file."""
    shape = [d if isinstance(d, int) else 1 for d in sess.get_inputs()[1].shape]
    caches = np.zeros(shape, dtype=np.float32)
    out = []
    for a in range(0, len(feat), chunk_frames):
        probs, caches = sess.run(None, {"feat": feat[None, a:a + chunk_frames], "caches_in": caches})
        out.append(probs.reshape(-1))
    return (np.concatenate(out) if out else np.zeros(0)).astype(np.float32)


# ============================================================================ step 8: post-process
@dataclass
class PostConfig:
    """Mặc định = FireRedVadConfig (non-stream) của upstream. Đơn vị: frame 10 ms."""
    smooth_window_size: int = 5
    speech_threshold: float = 0.4
    min_speech_frame: int = 20
    max_speech_frame: int = 2000
    min_silence_frame: int = 20
    merge_silence_frame: int = 0
    extend_speech_frame: int = 0


def _smooth(p, w):
    if w <= 1:
        return p.copy()
    s = np.convolve(p, np.ones(w) / w, mode="full")[:len(p)]
    for i in range(min(w - 1, len(p))):
        s[i] = p[:i + 1].mean()
    return s


def _state_machine(binary, min_speech, min_silence):
    if min_speech <= 0 and min_silence <= 0:
        return list(binary)
    SIL, PSP, SP, PSIL = 0, 1, 2, 3
    dec = [0] * len(binary)
    st, sp_start, sil_start = SIL, -1, -1
    for t, b in enumerate(binary):
        if st == SIL:
            if b:
                st, sp_start = PSP, t
        elif st == PSP:
            if b:
                if t - sp_start >= min_speech:
                    st = SP
                    dec[sp_start:t] = [1] * (t - sp_start)
            else:
                st, sp_start = SIL, -1
        elif st == SP:
            if not b:
                st, sil_start = PSIL, t
        elif st == PSIL:
            if not b:
                if t - sil_start >= min_silence:
                    st, sp_start = SIL, -1
            else:
                st, sil_start = SP, -1
        dec[t] = 1 if st in (SP, PSIL) else 0
    return dec


def _fix_smooth_start(dec, w):
    new = list(dec)
    for t in range(1, len(dec)):
        if dec[t - 1] == 0 and dec[t] == 1:
            a = max(0, t - w)
            new[a:t] = [1] * (t - a)
    return new


def _merge_short_silence(dec, n):
    if n <= 0:
        return dec
    new, s0 = list(dec), None
    for t in range(1, len(dec)):
        if dec[t - 1] == 1 and dec[t] == 0 and s0 is None:
            s0 = t
        elif dec[t - 1] == 0 and dec[t] == 1 and s0 is not None:
            if t - s0 < n:
                new[s0:t] = [1] * (t - s0)
            s0 = None
    return new


def _extend(dec, n):
    if n <= 0:
        return dec
    return (np.convolve(np.array(dec), np.ones(2 * n + 1), mode="same") > 0).astype(int).tolist()


def decisions_to_segments(dec, wav_dur=None):
    """Giống VadPostprocessor.decision_to_segment: frame t -> t*10 ms; đoạn cuối kéo tới hết file."""
    segs, start = [], None
    for t, d in enumerate(dec):
        if d == 1 and start is None:
            start = t
        elif d == 0 and start is not None:
            segs.append((start * FRAME_SHIFT_S, t * FRAME_SHIFT_S))
            start = None
    if start is not None:
        end = len(dec) * FRAME_SHIFT_S + FRAME_LEN_S
        segs.append((start * FRAME_SHIFT_S, min(end, wav_dur) if wav_dur else end))
    return [(round(s, 3), round(e, 3)) for s, e in segs]


def _split_long(dec, probs, max_frame):
    new = list(dec)
    for s, e in decisions_to_segments(dec):
        a, b = int(s / FRAME_SHIFT_S), int(e / FRAME_SHIFT_S)   # giống upstream (int, không round)
        if b - a <= max_frame:
            continue
        seg, start = probs[a:b], 0
        while len(seg) - start > max_frame:
            w0, w1 = int(start + max_frame / 2), int(start + max_frame)
            k = w0 + int(np.argmin(seg[w0:w1]))
            new[a + k] = 0
            start = k + 1
    return new


def step8_postprocess(probs: np.ndarray, cfg: PostConfig = PostConfig(), wav_dur=None):
    """Port 1-1 của fireredvad VadPostprocessor.process + decision_to_segment.
    Returns: smoothed [T], binary [T], decisions [T], segments [(start_s, end_s)]."""
    probs = np.asarray(probs, dtype=np.float64)
    if len(probs) == 0:
        return probs, np.zeros(0, int), np.zeros(0, int), []
    w = max(1, cfg.smooth_window_size)
    sm = _smooth(probs, w)
    binary = (sm >= cfg.speech_threshold).astype(int)
    dec = _state_machine(binary.tolist(), cfg.min_speech_frame, cfg.min_silence_frame)
    dec = _fix_smooth_start(dec, w)
    dec = _merge_short_silence(dec, cfg.merge_silence_frame)
    dec = _extend(dec, cfg.extend_speech_frame)
    dec = _split_long(dec, probs, cfg.max_speech_frame)
    return sm, binary, np.array(dec, dtype=int), decisions_to_segments(dec, wav_dur)


# ============================================================================ full pipeline
MODEL_FILES = {
    "nonstream": "fireredvad_vad.onnx",
    "stream": "fireredvad_stream_vad.onnx",
    "stream_cached": "fireredvad_stream_vad_with_cache.onnx",
}


def detect_mode(sess) -> str:
    """Nhận dạng loại model từ chữ ký ONNX (không phụ thuộc tên file):
    có input caches_in -> stream_cached; có output cache_out_* -> stream; còn lại -> nonstream."""
    ins = [i.name for i in sess.get_inputs()]
    outs = [o.name for o in sess.get_outputs()]
    if ins[0] != "feat" or sess.get_inputs()[0].shape[-1] != N_MELS:
        raise ValueError(f"Không giống FireRedVAD: input {ins}, cần 'feat' [.., T, 80]")
    if len(ins) > 1:
        return "stream_cached"
    if len(outs) > 1:
        return "stream"
    if sess.get_outputs()[0].shape[-1] not in (1, "1"):
        raise ValueError(f"Output {sess.get_outputs()[0].shape} có >1 lớp - có thể là model AED, không phải VAD")
    return "nonstream"


def resolve_model(model_dir=None, mode="auto", onnx_path=None, cmvn_path=None):
    """Tìm file ONNX + cmvn.ark. Thứ tự:
    1. onnx_path / cmvn_path chỉ định rõ (tên file tuỳ ý).
    2. model_dir chứa tên chuẩn (fireredvad_vad.onnx, ...).
    3. model_dir chứa .onnx tên khác -> chọn theo chữ ký (detect_mode).
    cmvn.ark: cạnh file ONNX, thư mục cha của nó, rồi mới tới model_dir."""
    if onnx_path is None:
        if not model_dir or not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Không thấy thư mục model: {model_dir}")
        std = os.path.join(model_dir, MODEL_FILES.get(mode, MODEL_FILES["nonstream"]))
        if os.path.exists(std):
            onnx_path = std
        else:
            cands = sorted(f for f in os.listdir(model_dir) if f.endswith(".onnx"))
            for f in cands:
                try:
                    m = detect_mode(_session(os.path.join(model_dir, f)))
                except Exception:
                    continue
                if mode in ("auto", m):
                    onnx_path = os.path.join(model_dir, f)
                    break
            if onnx_path is None:
                if os.path.exists(os.path.join(model_dir, "model.pth.tar")):
                    raise FileNotFoundError(
                        f"{model_dir} chứa checkpoint PyTorch (model.pth.tar), chưa phải ONNX. Export 1 lần "
                        "(cần torch): git clone https://github.com/FireRedTeam/FireRedVAD && cd FireRedVAD && "
                        f"pip install -e . onnx && python fireredvad/bin/export_onnx.py --task vad "
                        f"--model-dir {model_dir} --output-dir <thư mục ra>  (Stream-VAD: --task stream_vad)")
                raise FileNotFoundError(f"Không có file .onnx FireRedVAD hợp lệ ({mode}) trong {model_dir}: {cands}")
    if cmvn_path is None:
        here = os.path.dirname(os.path.abspath(onnx_path))
        dirs = [here, os.path.dirname(here)] + ([model_dir] if model_dir else [])
        cmvn_path = next((os.path.join(d, "cmvn.ark") for d in dirs
                          if os.path.exists(os.path.join(d, "cmvn.ark"))), None)
        if cmvn_path is None:
            raise FileNotFoundError(f"Không thấy cmvn.ark cạnh {onnx_path} - chỉ định bằng --cmvn")
    return onnx_path, cmvn_path


@dataclass
class Result:
    path: str
    orig_sr: int
    channels: int
    duration: float
    probs: np.ndarray                 # [T] mỗi 10 ms
    smoothed: np.ndarray
    decisions: np.ndarray
    segments: list
    timings: dict = field(default_factory=dict)
    steps: dict = field(default_factory=dict)

    def frame_times(self):
        t = np.arange(len(self.probs)) * FRAME_SHIFT_S
        return t, t + FRAME_LEN_S


class FireRedOnnx:
    def __init__(self, model_dir: str | None = None, mode: str = "auto", threads: int = 1,
                 chunk_frames: int = 10, post: PostConfig = PostConfig(),
                 downmix: str = "mean", res_type: str = "soxr_hq", dither="auto",
                 onnx_path: str | None = None, cmvn_path: str | None = None):
        """model_dir: thư mục chứa .onnx + cmvn.ark, hoặc onnx_path/cmvn_path chỉ định thẳng file.
        mode: auto (nhận từ chữ ký ONNX) | nonstream | stream | stream_cached."""
        if mode not in ("auto", *MODEL_FILES):
            raise ValueError(f"mode phải là auto hoặc một trong {list(MODEL_FILES)}")
        self.onnx_path, self.cmvn_path = resolve_model(
            model_dir, "nonstream" if (mode == "auto" and onnx_path is None) else mode, onnx_path, cmvn_path)
        self.sess = _session(self.onnx_path, threads)
        found = detect_mode(self.sess)
        if mode not in ("auto", found):
            raise ValueError(f"{self.onnx_path} là model '{found}', không phải '{mode}'")
        self.mode, self.chunk_frames, self.post = found, chunk_frames, post
        self.downmix, self.res_type, self.dither = downmix, res_type, dither
        self.mean, self.istd = load_cmvn(self.cmvn_path)

    def run_array(self, x: np.ndarray, sr: int, keep_steps: bool = False, path: str = "") -> Result:
        """x: [C, N] hoặc [N], float32 [-1, 1], sr bất kỳ."""
        tm, steps = {}, {}
        x = np.atleast_2d(x)
        ch = x.shape[0]
        t = time.perf_counter()
        mono = step2_downmix(x, self.downmix)
        tm["2_downmix"] = time.perf_counter() - t
        t = time.perf_counter()
        x16 = step3_resample(mono, sr, SAMPLE_RATE, self.res_type)
        tm["3_resample"] = time.perf_counter() - t
        xi0 = xi = step4_int16_scale(x16)
        dither = (1.0 if sr < SAMPLE_RATE else 0.0) if self.dither == "auto" else float(self.dither)
        tm["dither"] = dither
        xi = step4b_dither(xi, dither)
        t = time.perf_counter()
        if keep_steps:
            feat_raw, fsteps = step5_fbank(xi, return_steps=True)
            steps.update({k: v for k, v in fsteps.items() if k in ("5e_power", "5f_mel")})
        else:
            feat_raw = step5_fbank(xi)
        tm["5_fbank"] = time.perf_counter() - t
        t = time.perf_counter()
        feat = step6_cmvn(feat_raw, self.mean, self.istd)
        tm["6_cmvn"] = time.perf_counter() - t
        t = time.perf_counter()
        if self.mode == "nonstream":
            probs = step7_onnx_nonstream(self.sess, feat)
        elif self.mode == "stream":
            probs = step7_onnx_stream_full(self.sess, feat)
        else:
            probs = step7_onnx_stream_cached(self.sess, feat, self.chunk_frames)
        tm["7_onnx"] = time.perf_counter() - t
        dur = len(x16) / SAMPLE_RATE
        t = time.perf_counter()
        sm, binary, dec, segs = step8_postprocess(probs, self.post, dur)
        tm["8_post"] = time.perf_counter() - t
        tm.pop("dither")
        self.last_dither = dither
        if keep_steps:
            steps.update({"1_input": x, "2_mono": mono, "3_16k": x16, "4_int16": xi0,
                          "5g_logmel": feat_raw, "6_cmvn": feat, "7_probs": probs,
                          "8_smoothed": sm, "8_binary": binary, "8_decisions": dec})
            if dither > 0:
                steps["4b_dithered"] = xi
        return Result(path, sr, ch, dur, probs, sm, dec, segs, tm, steps)

    def run_file(self, path: str, keep_steps: bool = False) -> Result:
        t = time.perf_counter()
        x, sr = step1_load_audio(path)
        t1 = time.perf_counter() - t
        r = self.run_array(x, sr, keep_steps, path)
        r.timings = {"1_load": t1, **r.timings}
        return r
