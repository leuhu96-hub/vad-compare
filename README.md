# vad-compare

So sánh nhiều model VAD (speech / non-speech) trên **cùng một bộ audio và cùng ground truth 0.5 s**:
chất lượng (AUC, Precision/Recall/F1, **Miss rate, FAR**, lỗi biên vs lỗi thật, theo category),
tốc độ (RTF) và report HTML. Audio đọc bằng **librosa**.

| type          | Model                                   | Output gốc            | Cài                                   |
|---------------|-----------------------------------------|-----------------------|---------------------------------------|
| `yamnet`      | YAMNet transfer learning của bạn        | prob / cửa sổ 0.96 s, hop 0.08 s | `.tflite` (ai-edge-litert) hoặc `.onnx` |
| `silero`      | Silero VAD v5/v6 (ONNX, không cần torch) | prob / 32 ms          | `scripts/download_models.py`          |
| `webrtc`      | WebRTC VAD (GMM)                        | 0/1 / 10–30 ms        | `pip install webrtcvad-wheels`        |
| `ten`         | TEN VAD                                 | prob / 16 ms          | `pip install git+https://github.com/TEN-framework/ten-vad.git` |
| `firered`     | FireRedVAD (non-stream / stream, ONNX)  | prob / 10 ms          | `scripts/download_models.py`          |
| `pyannote`    | pyannote segmentation-3.0               | prob / ~17 ms         | `pip install pyannote.audio` + `HF_TOKEN` |
| `precomputed` | Kết quả có sẵn (vd. lib.so chạy trên điện thoại) | file `index,start,end,score` | –                    |

## Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py          # silero + firered -> models/
cp /path/to/your_model.tflite models/yamnet_vad.tflite
pip install ai-edge-litert                  # nếu model YAMNet là .tflite
```

Audio đọc bằng **librosa** (mp3, mp4, m4a, aac, flac, ogg, wav… **sample rate và số kênh bất kỳ**):
`librosa.load(sr=None, mono=False)` → `librosa.to_mono` (trung bình kênh) → `librosa.resample` về 16 kHz
(`res_type: soxr_hq`, đổi trong `config.yaml`). wav/flac/ogg/mp3 đi qua soundfile; **m4a/mp4/aac** librosa
phải dùng audioread nên máy cần có **ffmpeg** (`brew install ffmpeg` / `sudo apt install ffmpeg`).
Có thể đổi `audio.decoder` sang `ffmpeg` hoặc `pyav` nếu muốn.

TEN VAD trên Linux cần `sudo apt install libc++1`.

## Chạy thử nhanh (dữ liệu public)

```bash
git clone --depth 1 https://github.com/TEN-framework/ten-vad.git /tmp/ten-vad
python scripts/make_demo_dataset.py /tmp/ten-vad/testset data/demo   # 30 file, đổi sang wav/mp3/m4a
python -m vadcmp run -c config.demo.yaml
open out_demo/report.html
```

## Dùng với dữ liệu của bạn

### 1. Manifest (ghép audio ↔ GT ↔ category)

```bash
python -m vadcmp manifest --audio-root data/Audio --gt-root data/Model -o data/manifest.csv
```

Cấu trúc GT: `Model/<category>/ground_truth/<filename>.txt`. **Chỉ file nằm trong thư mục
`ground_truth/` được đọc**; các folder khác trong `<category>` (kể cả có file trùng tên) bị bỏ qua:

```
Model/Asm/ground_truth/asm_1.txt     ← GT, category "Asm"
Model/Asm/model_v1/asm_1.txt         ← bỏ qua
Model/Asm/lib_out/raw/asm_1.txt      ← bỏ qua
```

Lệnh in ra số file GT tìm được và số file bị bỏ qua theo từng folder. Đổi tên thư mục GT bằng
`--gt-dir`. Audio ghép với GT theo tên file (không phân biệt hoa thường, bỏ tiền tố `Gt_` nếu có):
`Audio/Asm/asm_1.mp4` ↔ `Model/Asm/ground_truth/asm_1.txt`. Muốn đưa output trong các folder khác
(vd. `model_v1/`) vào so sánh thì dùng model `precomputed` với
`path_pattern: data/Model/{category}/model_v1/{stem}.txt`.
Có thể viết tay `manifest.csv` với cột `audio,gt,category` (đường dẫn tương đối so với file manifest).

### 2. Ground truth

```
0 0.5 Speech
0.5 1 Non-Speech 1
```

Cột 1 = start (s), cột 2 = end (s), cột 3 = label; **cột 4 trở đi bị bỏ qua**. Phân tách bằng space /
tab / dấu phẩy. Label không phân biệt hoa thường: `Speech / Non-Speech / nonspeech / 1 / 0 …`.
Đoạn GT dài hơn 0.5 s được cắt thành bin 0.5 s theo đa số thời lượng.

### 3. Cấu hình `config.yaml`

Mỗi model là một mục trong `models:` — có thể khai báo nhiều biến thể cùng type
(vd. `webrtc` mode 1 và mode 3, `firered` stream và non-stream, YAMNet threshold khác nhau).

Model YAMNet của bạn:

```yaml
- name: yamnet
  type: yamnet
  model_path: models/yamnet_vad.tflite   # hoặc .onnx; input = waveform float32 16 kHz
  window_s: 0.96
  hop_s: 0.08
  speech_index: 1          # index class speech trong output 2 class
  output: auto             # auto | probs | logits (auto: softmax nếu output chưa là xác suất)
  overlap: mean            # rescore per hop: mỗi hop 0.08 s = mean (hoặc max) các cửa sổ phủ nó
  threshold: 0.6026
  # input_samples: 15600   # nếu model nhận số mẫu khác 0.96*16000 và shape input là dynamic
```

So sánh thêm output thật của `lib.so` (đã chạy trên điện thoại):

```yaml
- name: lib_so
  type: precomputed
  path_pattern: results/Model/{Category}_{index}.txt   # {stem} {id} {category} {Category} {index}
  threshold: 0.6026
```

### 4. Chạy

```bash
python -m vadcmp infer    -c config.yaml               # chạy model, cache vào out/cache/
python -m vadcmp evaluate -c config.yaml               # metric + report từ cache (chạy lại rất nhanh)
python -m vadcmp run      -c config.yaml               # cả hai
python -m vadcmp infer    -c config.yaml --models silero,ten --limit 20 --force
```

Đổi `threshold`, `postprocess`, `scoring` thì chỉ cần `evaluate` lại; đổi tham số model (`mode`,
`model_path`, `window_s`…) thì cache tự chạy lại cho model đó.

## Cách tính

```
audio ──decode/downmix/resample──► model ──► FrameScores (prob theo frame gốc của model)
                                             │
             ┌───────────────────────────────┴──────────────────────────────┐
   score mỗi bin 0.5 s = mean prob theo overlap          threshold → segment → post-processing chung
             │                                            (pad 0.10/0.12 s, merge gap < 0.5 s, bỏ < 0.32 s)
       AUC, AP, Youden thr                                → bin speech nếu phủ ≥ 50 % → P/R/F1/Spec
```

* **AUC / AP**: không phụ thuộc threshold — so sánh "khả năng tách" speech/non-speech giữa các model.
* **F1 (pp)**: sau post-processing chung → so sánh như khi dùng thật. **F1 raw**: không post-processing.
* **Miss rate** = FN / (TP + FN) = 1 − recall: tỉ lệ bin speech bị bỏ sót.
  **FAR** (false alarm rate) = FP / (FP + TN) = 1 − specificity: tỉ lệ bin non-speech bị báo nhầm là speech.
  Report có bản pp, raw, bỏ biên, tại Youden threshold, theo category và biểu đồ Miss rate vs FAR.
* **Lỗi biên**: bin nằm trong `boundary_tolerance` bin quanh điểm chuyển GT. **F1 bỏ biên** loại các bin đó;
  bảng lỗi tách *FN thật / FP thật / lỗi biên*.
* **Youden thr**: threshold tối ưu tính trên chính tập đánh giá (lạc quan) — dùng để chọn threshold, không để báo cáo.
* **RTF** = thời gian model xử lý / độ dài audio (không tính decode, có warm-up, `num_threads: 1`).
  Đo trên máy chạy script → chỉ so sánh tương đối, không thay cho đo trên điện thoại.
* Chỉ những file mà **mọi model** đều chạy được mới vào thống kê, để so sánh trên cùng tập.

`post-processing` mặc định áp cho mọi model để công bằng; ghi đè từng model bằng
`postprocess: none` hoặc `postprocess: {merge_gap: 0.3}`; `smooth_s: 0.1` làm mượt prob trước khi threshold.

## Kết quả (`out/`)

| File | Nội dung |
|------|----------|
| `report.html` | report tự chứa: bảng tổng, ROC, theo category, phân loại lỗi, tốc độ, timeline các file tệ nhất |
| `metrics_overall.csv`, `metrics_by_category.csv`, `metrics_per_file.csv`, `speed.csv` | số liệu |
| `roc_*.png`, `summary.png`, `errors.png`, `timelines/*.png` | hình |
| `cache/<model>/<file>.npz` | prob theo frame + thời gian chạy |

## Thêm model mới

Tạo `vadcmp/models/my_vad.py`:

```python
from vadcmp.models.base import VADModel
from vadcmp.timeline import FrameScores

class MyVAD(VADModel):
    type_name = "my_vad"
    default_threshold = 0.5
    def load(self):                       # gọi 1 lần, đo load time
        self.model = ...                  # self.cfg = dict trong config, self.path("model_path")
    def process(self, audio):             # float32 mono 16 kHz [-1, 1]; reset state mỗi file
        probs = ...                       # 1 prob mỗi hop
        return FrameScores.from_hops(probs, hop=0.016, duration=len(audio) / 16000)
```

rồi thêm `"my_vad": ("vadcmp.models.my_vad", "MyVAD")` vào `REGISTRY` trong `vadcmp/models/__init__.py`.

## Test

```bash
pip install pytest && pytest -q
```
