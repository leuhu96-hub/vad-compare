# Experiment: FireRedVAD chạy trực tiếp bằng ONNX

Chạy FireRedVAD cho cả một folder dataset (audio + ground truth), audio có **sample rate và số kênh
bất kỳ**. Code gọi thẳng file `.onnx` qua onnxruntime, không dùng package `fireredvad` và không cần torch.

| File | Vai trò |
|---|---|
| `dataset_reader.py` | Đọc và ghép audio ↔ ground truth. **Độc lập**, chỉ dùng thư viện chuẩn → copy sang project khác dùng được |
| `firered_onnx.py` | Pipeline FireRedVAD từng bước: audio → fbank → CMVN → ONNX → post-processing → segment |
| `run.py` | Chạy cả dataset (hoặc file lẻ), xuất segment, bin 0.5 s, metric so với GT |

## Cài đặt

```bash
pip install -r requirements.txt     # numpy, onnxruntime, librosa (+ matplotlib nếu --plot)
```

m4a/mp4/aac: librosa đọc qua audioread nên máy cần có `ffmpeg`.

Model: chỉ tới file có sẵn bằng `--onnx my_vad.onnx` (cmvn.ark được tìm cạnh file hoặc ở thư mục cha,
hoặc chỉ định bằng `--cmvn`). Cách khác: `--model-dir <thư mục>`, hoặc để vào `./models`.
Loại model (non-stream / stream / stream + cache) được nhận tự động từ input/output của file ONNX.

## Cấu trúc dữ liệu

```
phase_1/                              <- --audio-root
  <quality>/<filename>.mp4|mp3|m4a|wav|...
Groundtruth/                          <- --gt-root
  <category>/ground_truth/<filename>.txt    <- CHỈ đọc file trong ground_truth/
  <category>/<folder khác>/<filename>.txt   <- bỏ qua
```

- **Ghép theo tên file** (bỏ đuôi, không phân biệt hoa thường), không phụ thuộc thư mục.
  `quality` lấy từ đường dẫn audio, `category` lấy từ đường dẫn GT.
- **Cùng tên file ở nhiều `<quality>`**: mỗi bản audio là một item riêng, dùng chung một GT.
- **Cùng tên file ở nhiều `<category>`**: không biết dùng GT nào, nên bỏ qua và liệt kê trong báo cáo.
- **`ground_truth/` nằm sâu hơn** (vd. `<category>/batch2/ground_truth/`) vẫn được đọc.

File GT: `start end label`, cột 4 trở đi bị bỏ qua.

```
0 0.5 Speech
0.5 1 Non-Speech 1
```

## Chạy

```bash
# kiểm tra ghép dữ liệu (không chạy model)
python dataset_reader.py --audio-root phase_1 --gt-root Groundtruth --csv manifest.csv

# chạy FireRedVAD cho cả dataset
python run.py --audio-root phase_1 --gt-root Groundtruth --onnx models/fireredvad_vad.onnx -o out
python run.py --audio-root phase_1 --gt-root Groundtruth --onnx rt.onnx --plot --save-probs -o out

# file / folder lẻ, không có GT
python run.py song.m4a other_folder/ -o out
```

Output trong `-o`:

| Đường dẫn | Nội dung |
|---|---|
| `segments/<quality>/<filename>.txt` | `start end Speech`, segment sau post-processing |
| `bins/<quality>/<filename>.txt` | dự đoán trên lưới 0.5 s, cùng định dạng GT (`0 0.5 Speech`) |
| `results.csv` | mỗi file: quality, category, sr, số kênh, thời lượng, số segment, TP/FP/TN/FN, accuracy, precision, recall, F1, miss rate, FAR, RTF |
| `summary.csv` | metric gộp (micro): tất cả, theo quality, theo category, theo quality × category |
| `dataset_report.txt` | số file, file thiếu GT / thiếu audio / trùng tên, folder bị bỏ qua |
| `probs/…csv` | (`--save-probs`) prob mỗi frame 10 ms |
| `plots/…png` | (`--plot`) waveform, log-mel, CMVN, prob, GT, segment |
| `steps/…npz` | (`--dump`) tensor trung gian của các bước |

Metric so với GT: segment được quy về bin 0.5 s; một bin là speech nếu segment phủ ≥ 50 % bin.
Miss rate = FN / (TP + FN), FAR = FP / (FP + TN).

Tham số post-processing (đơn vị frame 10 ms): `--threshold 0.4 --smooth 5 --min-speech 20 --min-silence 20
--max-speech 2000 --merge-silence 0 --extend-speech 0` (mặc định giống upstream).

## Dùng `dataset_reader.py` trong project khác

```python
from dataset_reader import load_dataset, read_ground_truth, gt_to_bins, segments_to_bins, write_bins

ds = load_dataset("phase_1", "Groundtruth")           # gt_dir="ground_truth" mặc định
print(ds.report())
for it in ds.items:                                    # it.name, it.audio, it.gt, it.quality, it.category
    gt = read_ground_truth(it.gt)                      # [(start, end, 1|0), ...]
    gt_bins = gt_to_bins(gt, 0.5)                      # [1, 0, 1, ...] (-1 = không có nhãn)
    pred_bins = segments_to_bins(my_segments, duration, 0.5)
    write_bins(f"out/{it.name}.txt", pred_bins, 0.5, duration)
groups = ds.by("category")                             # hoặc ds.by("quality")
```

## Các bước của pipeline (`firered_onnx.py`)

| Bước | Input | Output |
|---|---|---|
| 1 load (`librosa.load(sr=None, mono=False)`) | file | `[C, N]` float32, sr gốc |
| 2 downmix | `[C, N]` | `[N]` (trung bình kênh) |
| 3 resample (`librosa.resample`) | sr bất kỳ | 16 kHz |
| 4 ×32768 (+ dither nếu sr gốc < 16 kHz) | [-1, 1] | thang int16 |
| 5 Kaldi fbank (25/10 ms, povey, pre-emph 0.97, FFT 512, mel 80, log) | `[N']` | `[T, 80]` |
| 6 CMVN từ `cmvn.ark` | `[T, 80]` | `[T, 80]` |
| 7 ONNX | `feat [1, T, 80]` (+ `caches_in [8, 1, 128, 19]` với bản stream có cache) | `probs [T]`, mỗi 10 ms |
| 8 post-processing (port 1-1 `VadPostprocessor`) | `probs` | segment |

Đã kiểm chứng: fbank khớp kaldi-native-fbank (sai lệch tối đa 1.1e-4); bản stream có cache khớp bản
stream chạy cả chuỗi; post-processing ra decision giống hệt upstream. Script kiểm chứng
(`check_steps.py`) đã được bỏ khỏi folder, nhưng vẫn còn trong lịch sử git ở commit `66c6ef6`.

## Lưu ý

- **Resample:** bộ lọc resample cắt bớt dải 7–8 kHz, làm prob lệch nhẹ (khoảng 99.7 % quyết định
  khớp bản 16 kHz gốc). `--res-type fft` cho kết quả giống hệt.
- **Input 8 kHz:** dải 4–8 kHz trống nên log-mel rơi xuống rất thấp và model sai nhiều.
  `--dither auto` (mặc định) cộng nhiễu σ = 1 khi sr gốc < 16 kHz, nâng tỉ lệ khớp từ ~58 % lên ~94–96 %.
- **cmvn.ark:** phải đúng của model đang dùng; sai cmvn thì kết quả sai mà không có lỗi nào.
