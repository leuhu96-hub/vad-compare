"""YAMNet (transfer learning, 2 lớp speech / non-speech) chạy thẳng file .tflite, từng bước,
cùng cách làm với firered_vad/firered_onnx.py và silero_vad/silero_onnx.py. Không dùng vadcmp.

  1. đọc audio bất kỳ (sr, số kênh)        -> firered_onnx.step1_load_audio (librosa)
  2. trộn về 1 kênh (mean)                  -> step2_downmix
  3. resample về 16 kHz, float32 [-1, 1]    -> step3_resample (soxr_hq)
  4. cắt cửa sổ: dài = số mẫu input của model (YAMNet: 15 600 mẫu = 0.975 s), nhảy hop 0.08 s (1 280 mẫu).
     Phần cuối không đủ một cửa sổ: bỏ (tail=drop, giống lib) hoặc đệm 0 (tail=pad)
  5. mỗi cửa sổ -> interpreter TFLite -> output lớp -> xác suất speech
       * input: tự đọc shape (vd. [15600], [1, 15600]), lượng tử hoá int8/uint8 nếu có
       * output: chọn tensor có 2 lớp (hoặc --output-index); nếu là [frames, lớp] thì lấy trung bình;
         đã là xác suất (tổng = 1) thì giữ, không thì softmax; lấy lớp --speech-index
  6. gộp cửa sổ -> score mỗi hop             -> yamnet_post.rescore
  7. hậu xử lý -> đoạn speech               -> yamnet_post.y0/y1/y2_segments

TFLite runtime: ai-edge-litert (khuyên dùng), tflite-runtime, hoặc tensorflow (tf.lite.Interpreter); thử lần lượt,
cách nào lỗi (kể cả AttributeError của tensorflow mới) thì chuyển sang cách sau. File .onnx cùng kiểu input
cũng chạy được (onnxruntime).

    python yamnet_tflite.py audio.m4a --model yamnet_vad.tflite --threshold 0.6026
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "firered_vad"))
from firered_onnx import step1_load_audio, step2_downmix, step3_resample  # noqa: E402

SR = 16000
YAMNET_SAMPLES = 15600


# =========================================================================== backend
def _interpreter_classes():
    """Các cách lấy lớp Interpreter, theo thứ tự ưu tiên. Mỗi cách là (tên, hàm trả về lớp)."""
    def litert():
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter

    def tflite_runtime():
        from tflite_runtime.interpreter import Interpreter
        return Interpreter

    def tf_lite():                       # cách chuẩn với tensorflow: tf.lite.Interpreter
        import tensorflow as tf
        return tf.lite.Interpreter

    def tf_internal():                   # một số bản TF không cho import tensorflow.lite trực tiếp
        from tensorflow.lite.python.interpreter import Interpreter
        return Interpreter

    return [("ai-edge-litert", litert), ("tflite-runtime", tflite_runtime),
            ("tensorflow (tf.lite.Interpreter)", tf_lite), ("tensorflow (lite.python.interpreter)", tf_internal)]


def load_tflite(path, threads=1, prefer=None):
    """-> (interpreter, tên runtime). Thử lần lượt; lỗi nào (ImportError, AttributeError …) cũng chuyển sang
    cách tiếp theo và được ghi lại để báo nếu tất cả đều hỏng. prefer: chỉ thử runtime có tên chứa chuỗi này."""
    errors = []
    for name, get in _interpreter_classes():
        if prefer and prefer.lower() not in name.lower():
            continue
        try:
            cls = get()
        except Exception as e:                                   # noqa: BLE001
            errors.append(f"  {name}: {type(e).__name__}: {e}")
            continue
        try:
            return cls(model_path=path, num_threads=threads), name
        except TypeError:                                        # bản cũ không có num_threads
            return cls(model_path=path), name
    raise ImportError("Không mở được TFLite runtime nào. Cách dễ nhất: pip install ai-edge-litert\n"
                      "(Các bản tensorflow mới đã đánh dấu bỏ tf.lite.Interpreter, chuyển sang ai-edge-litert.) Chi tiết từng cách đã thử:\n"
                      + "\n".join(errors))


class _Tflite:
    """Bọc Interpreter: một input waveform, đọc mọi output."""

    def __init__(self, path, threads, prefer=None):
        self.it, self.runtime = load_tflite(path, threads, prefer)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.outs = self.it.get_output_details()

    @property
    def input_shape(self):
        return [int(d) for d in self.inp.get("shape_signature", self.inp["shape"])]

    def output_shapes(self):
        return [[int(d) for d in o.get("shape_signature", o["shape"])] for o in self.outs]

    def prepare(self, shape):
        if list(self.inp["shape"]) != list(shape):
            self.it.resize_tensor_input(self.inp["index"], shape)
            self.it.allocate_tensors()
            self.inp = self.it.get_input_details()[0]
            self.outs = self.it.get_output_details()

    def __call__(self, x, out_index):
        dt = self.inp["dtype"]
        if dt != np.float32:                                   # model lượng tử hoá
            scale, zero = self.inp.get("quantization", (0.0, 0))
            info = np.iinfo(dt)
            x = np.clip(np.round(x / (scale or 1.0) + zero), info.min, info.max).astype(dt)
        self.it.set_tensor(self.inp["index"], x)
        self.it.invoke()
        o = self.outs[out_index]
        y = self.it.get_tensor(o["index"])
        scale, zero = o.get("quantization", (0.0, 0))
        if y.dtype != np.float32 and scale:
            y = (y.astype(np.float32) - zero) * scale
        return np.asarray(y, np.float64)


class _Onnx:
    runtime = "onnxruntime"

    def __init__(self, path, threads):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0]
        self.outs = self.sess.get_outputs()
        self.shape = None

    @property
    def input_shape(self):
        return [d if isinstance(d, int) else -1 for d in self.inp.shape]

    def output_shapes(self):
        return [[d if isinstance(d, int) else -1 for d in o.shape] for o in self.outs]

    def prepare(self, shape):
        self.shape = shape

    def __call__(self, x, out_index):
        return np.asarray(self.sess.run(None, {self.inp.name: x.astype(np.float32)})[out_index], np.float64)


# =========================================================================== model
@dataclass
class Result:
    path: str
    orig_sr: int
    channels: int
    n_samples: int              # số mẫu 16 kHz
    starts: np.ndarray          # thời điểm bắt đầu mỗi cửa sổ (giây)
    scores: np.ndarray          # xác suất speech mỗi cửa sổ
    window_s: float
    timings: dict = field(default_factory=dict)

    @property
    def duration(self):
        return self.n_samples / SR


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class YamnetTflite:
    def __init__(self, model_path, threads=1, speech_index=1, output="auto", output_index=None,
                 input_samples=None, hop_s=0.08, tail="drop", downmix="mean", res_type="soxr_hq", runtime=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không thấy model: {model_path}")
        self.model_path = model_path
        self.backend = "onnx" if model_path.lower().endswith(".onnx") else "tflite"
        self.m = _Onnx(model_path, threads) if self.backend == "onnx" else _Tflite(model_path, threads, runtime)
        shape = self.m.input_shape
        fixed = [d for d in shape if d > 1]
        self.n_in = int(input_samples or (fixed[-1] if fixed else YAMNET_SAMPLES))
        # shape thật khi chạy: chiều cố định > 1 hoặc chiều động cuối cùng = n_in, còn lại = 1
        run = [1 if d in (-1, 0, 1) else d for d in shape] or [self.n_in]
        if self.n_in not in run:
            run[-1] = self.n_in
        self.run_shape = run
        self.m.prepare(run)
        self.out_index = self._pick_output(output_index)
        self.speech_index, self.output_mode = speech_index, output
        self.hop = int(round(hop_s * SR))
        self.tail, self.downmix, self.res_type = tail, downmix, res_type

    def _pick_output(self, output_index):
        shapes = self.m.output_shapes()
        if output_index is not None:
            return int(output_index)
        two = [i for i, s in enumerate(shapes) if s and s[-1] == 2]
        return two[0] if two else 0

    @property
    def window_s(self):
        return self.n_in / SR

    def describe(self):
        outs = self.m.output_shapes()
        return (f"{self.model_path} [{self.m.runtime}] input {self.m.input_shape} -> chạy {self.run_shape}, "
                f"cửa sổ {self.n_in} mẫu = {self.window_s:.3f} s, hop {self.hop} mẫu; output {outs}, "
                f"dùng output #{self.out_index}, lớp speech = {self.speech_index}")

    # ---- bước 4
    def step4_windows(self, x16):
        n = len(x16)
        n_win = 1 + (n - self.n_in) // self.hop if n >= self.n_in else 0
        x = x16
        if self.tail == "pad" and (n_win == 0 or (n_win - 1) * self.hop + self.n_in < n):
            n_win += 1
            x = np.pad(x16, (0, max(0, (n_win - 1) * self.hop + self.n_in - n)))
        return x, n_win

    # ---- bước 5
    def to_prob(self, y):
        y = np.asarray(y, np.float64)
        if y.ndim > 1 and y.shape[-1] > 1:                     # [frames, lớp] -> trung bình theo frame
            y = y.reshape(-1, y.shape[-1]).mean(axis=0)
        y = y.reshape(-1)
        if y.size == 1:
            v = float(y[0])
            return v if self.output_mode == "probs" or (self.output_mode == "auto" and 0 <= v <= 1) \
                else 1 / (1 + np.exp(-v))
        is_prob = self.output_mode == "probs" or (
            self.output_mode == "auto" and y.min() >= 0 and abs(y.sum() - 1) < 1e-3)
        p = y if is_prob else _softmax(y)
        return float(p[self.speech_index])

    def step5_window_prob(self, window):
        return self.to_prob(self.m(window.astype(np.float32).reshape(self.run_shape), self.out_index))

    def run_array(self, x, sr, path=""):
        tm = {}
        x = np.atleast_2d(x)
        t = time.perf_counter()
        x16 = step3_resample(step2_downmix(x, self.downmix), sr, SR, self.res_type)
        tm["2-3_mono_16k"] = time.perf_counter() - t
        t = time.perf_counter()
        xw, n_win = self.step4_windows(x16)
        scores = np.empty(n_win)
        for i in range(n_win):
            scores[i] = self.step5_window_prob(xw[i * self.hop:i * self.hop + self.n_in])
        tm["4-5_model"] = time.perf_counter() - t
        return Result(path, sr, x.shape[0], len(x16), np.arange(n_win) * self.hop / SR, scores, self.window_s, tm)

    def run_file(self, path):
        t = time.perf_counter()
        x, sr = step1_load_audio(path)
        t1 = time.perf_counter() - t
        r = self.run_array(x, sr, path)
        r.timings = {"1_load": t1, **r.timings}
        return r


if __name__ == "__main__":
    import argparse

    from yamnet_post import rescore, y1_segments

    ap = argparse.ArgumentParser(description="YAMNet .tflite cho 1 file: in score cửa sổ và đoạn speech (Y1)")
    ap.add_argument("audio")
    ap.add_argument("--model", required=True)
    ap.add_argument("--threshold", type=float, default=0.6026)
    ap.add_argument("--speech-index", type=int, default=1)
    ap.add_argument("--output", default="auto", choices=["auto", "probs", "logits"])
    ap.add_argument("--output-index", type=int)
    ap.add_argument("--tail", default="drop", choices=["drop", "pad"])
    ap.add_argument("--rescore", default="triangle", choices=["triangle", "mean", "max"])
    ap.add_argument("--smooth-hops", type=int, default=5)
    ap.add_argument("--runtime", help="ép dùng runtime: litert | tflite-runtime | tensorflow")
    a = ap.parse_args()
    m = YamnetTflite(a.model, speech_index=a.speech_index, output=a.output, output_index=a.output_index, tail=a.tail,
                     runtime=a.runtime)
    print("Model:", m.describe())
    r = m.run_file(a.audio)
    print(f"{a.audio}: {r.orig_sr} Hz, {r.channels} kênh, {r.duration:.2f} s, {len(r.scores)} cửa sổ; "
          + ", ".join(f"{k} {v:.2f} s" for k, v in r.timings.items()))
    hops = rescore(r.starts, r.window_s, r.scores, 0.08, r.duration, a.rescore, a.smooth_hops)
    for s, e in y1_segments(hops, a.threshold, 0.08, r.duration):
        print(f"{s:.3f} {e:.3f} Speech")
