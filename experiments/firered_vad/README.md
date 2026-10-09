# Experiment: FireRedVAD chạy trực tiếp bằng ONNX

Chạy FireRedVAD trên audio có **sample rate và số kênh bất kỳ**. Code gọi thẳng file `.onnx`
qua onnxruntime, **không dùng package `fireredvad`** (cũng không cần torch hay kaldi-native-fbank).
Mỗi bước là một hàm riêng trong `firered_onnx.py`, gọi lẻ được và xem được tensor trung gian.

## Cài đặt

```bash
cd experiments/firered_vad
pip install -r requirements.txt     # numpy, scipy, onnxruntime, librosa, matplotlib
python download_model.py            # 3 file .onnx + cmvn.ark -> ./models
```

m4a/mp4/aac: librosa đọc qua audioread nên máy cần có `ffmpeg`.

### Nếu đã có sẵn model

Không cần chạy `download_model.py`. Có 3 cách chỉ tới model có sẵn:

```bash
# 1. Chỉ thẳng file .onnx (tên tuỳ ý). cmvn.ark được tìm cạnh file hoặc ở thư mục cha
python run.py song.m4a --onnx /path/my_vad.onnx
python run.py song.m4a --onnx /path/my_vad.onnx --cmvn /path/cmvn.ark

# 2. Chỉ thư mục chứa .onnx + cmvn.ark (tên file tuỳ ý, tự chọn theo --mode)
python run.py song.m4a --model-dir /path/onnx_models --mode stream_cached

# 3. Copy / symlink vào ./models
ln -s /path/onnx_models models
```

- **Không cần khai báo loại model.** Mặc định `--mode auto` đọc input/output của file ONNX:
  có `caches_in` → `stream_cached`; có output `cache_out_*` → `stream`; còn lại → `nonstream`.
- **Lỗi rõ ràng.** Nhầm file AED (output 3 lớp), hoặc `--mode` không khớp file, thì báo lỗi luôn.
- **Thư mục chỉ có `model.pth.tar` + `cmvn.ark`** (bản tải từ Hugging Face) là checkpoint PyTorch,
  chưa phải ONNX. Export 1 lần trên máy có torch, script in sẵn lệnh:
  `python fireredvad/bin/export_onnx.py --task vad --model-dir <VAD> --output-dir <ra>`
  (`--task stream_vad` cho Stream-VAD, ra 2 file stream và stream + cache).
- **cmvn.ark phải đi cùng model.** Dùng cmvn của model khác thì feature lệch phân bố, kết quả sai mà
  không báo lỗi.

## Chạy

```bash
python walkthrough.py song.m4a                     # in shape/dtype/giá trị từng bước
python run.py song.m4a --plot --dump               # kết quả + hình + tensor trung gian
python run.py data/Audio --mode stream_cached --chunk 10 -o out_stream
python check_steps.py ref_16k.wav --upstream-repo /path/FireRedVAD   # kiểm chứng
```

Output của `run.py` cho mỗi file:

| File | Nội dung |
|---|---|
| `<stem>_probs.csv` | `frame, start_s, end_s, prob, smoothed, decision`, mỗi frame 10 ms |
| `<stem>_segments.txt` | `start end Speech` sau post-processing |
| `<stem>_bins.txt` | `0 0.5 Speech` / `0.5 1 Non-Speech`, cùng định dạng ground truth |
| `<stem>_steps.npz` | (`--dump`) tensor của bước 1→8 |
| `<stem>.png` | (`--plot`) waveform, log-mel, CMVN, prob, segment |
| `summary.csv` | sr, số kênh, thời lượng, số segment, thời gian từng bước |

## Các bước

| Bước | Hàm | Input | Output |
|---|---|---|---|
| 1 | `step1_load_audio` | file (wav/mp3/m4a/ogg…) | `x [C, N]` float32 [-1, 1], `sr` gốc |
| 2 | `step2_downmix` | `[C, N]` | `[N]`, trung bình các kênh (hoặc chọn 1 kênh) |
| 3 | `step3_resample` | `[N]` @ sr | `[N']` @ 16 kHz (`librosa.resample`, `res_type`) |
| 4 | `step4_int16_scale` | [-1, 1] | ×32768 → thang int16 |
| 4b | `step4b_dither` | | + nhiễu Gauss σ=1 (int16), mặc định chỉ bật khi sr gốc < 16 kHz |
| 5a | `step5a_frames` | `[N']` | `[T, 400]`, T = 1 + (N' − 400) // 160 (25 ms / 10 ms, snip_edges) |
| 5b–5d | remove DC → pre-emphasis 0.97 → cửa sổ povey | `[T, 400]` | `[T, 400]` |
| 5e | `step5e_power_spectrum` | `[T, 400]` | `[T, 257]` = \|FFT 512\|² |
| 5f–5g | mel 80 band (20 Hz–8 kHz, kiểu Kaldi) → log | `[T, 257]` | `[T, 80]` log-mel |
| 6 | `step6_cmvn` | `[T, 80]` | `(feat − mean) · istd`, mean/istd tính từ `cmvn.ark` |
| 7 | `step7_onnx_*` | `feat [1, T, 80]` | `probs [T]`, xác suất speech mỗi 10 ms |
| 8 | `step8_postprocess` | `probs [T]` | làm mượt → threshold → state machine → `segments` |

### Bước 7: ba file ONNX

| `--mode` | File | Input | Output | Ghi chú |
|---|---|---|---|---|
| `nonstream` | `fireredvad_vad.onnx` | `feat [1, T, 80]` | `probs [1, T, 1]` | Nhìn cả trước và sau, cần cả file (cắt mỗi 300 s) |
| `stream` | `fireredvad_stream_vad.onnx` | `feat [1, T, 80]` | `probs [1, T, 1]` + 8 cache `[1, 128, 19]` | Model causal, chạy cả chuỗi 1 lần |
| `stream_cached` | `fireredvad_stream_vad_with_cache.onnx` | `feat [1, t, 80]`, `caches_in [8, 1, 128, 19]` | `probs [1, t, 1]`, `caches_out [8, 1, 128, 19]` | Streaming thật: gọi mỗi `--chunk` frame, cache = 0 khi bắt đầu file |

Kích thước cache `[8, 1, 128, 19]`: 8 khối DFSMN, 128 chiều projection, 19 frame lookback.

### Bước 8: post-processing (port 1-1 `VadPostprocessor` của upstream)

1. Làm mượt bằng trung bình trượt `--smooth` frame (mặc định 5).
2. Threshold `--threshold` (mặc định 0.4).
3. State machine: cần ≥ `--min-speech` frame (20) mới vào speech, ≥ `--min-silence` frame (20) mới thoát.
4. Bù độ trễ của cửa sổ làm mượt, gộp silence ngắn hơn `--merge-silence`, nới mỗi đầu `--extend-speech` frame.
5. Cắt đoạn dài hơn `--max-speech` frame (2000 = 20 s) tại điểm prob thấp nhất.

## Kiểm chứng (`check_steps.py`)

Kết quả trên một file speech 11.5 s:

| Kiểm tra | Kết quả |
|---|---|
| fbank numpy vs kaldi-native-fbank | max \|diff\| 1.1e-4 (log-mel có giá trị 5–25) |
| `stream_cached` (chunk 1 / 10 / 37) vs `stream` cả chuỗi | giống hệt (diff = 0) |
| post-processing vs `VadPostprocessor` của upstream | decisions và segments giống hệt (2 bộ tham số) |
| 16k ×2 kênh, 22.05k, 44.1k ×2, 48k ×6 kênh | lệch 0 mẫu, cùng số frame (1150) |

### Phát hiện 1: bộ resample ảnh hưởng tới kết quả

Thử với file gốc 16 kHz, nâng lên 44.1/48 kHz (giữ nguyên dải 0–8 kHz), rồi cho qua pipeline:

| `res_type` | mean \|Δprob\| | quyết định khớp bản 16 kHz gốc |
|---|---|---|
| `fft` (lọc brickwall) | 0.0000 | 100 % |
| polyphase Kaiser (lọc thông thường) | 0.011 | 99.7 % |

Bộ lọc chống aliasing cắt bớt dải 7–8 kHz, nên log-mel của vài band trên cùng lệch tới ~1. Model nhạy
với các band này ở những frame lưng chừng (prob 0.5–0.8). Hãy chạy `check_steps.py --res-types
soxr_hq,soxr_vhq,kaiser_best,fft` để chọn `res_type` cho dữ liệu của bạn.

### Phát hiện 2: input 8 kHz cần dither

File 8 kHz (vd. thu âm điện thoại) không có dải 4–8 kHz. Sau khi upsample, các band mel trên cùng gần
như bằng 0 → log rơi xuống log(FLT_EPSILON) ≈ −15.9, rất xa phân bố lúc train:

| Upsample 8 kHz → 16 kHz | dither = 0 | dither = 1 | dither = 8 |
|---|---|---|---|
| FFT (dải trên = 0 tuyệt đối) | 57.8 % | 94.0 % | 96.5 % |
| polyphase | 94.9 % | 95.5 % | 96.8 % |

(% quyết định khớp bản 16 kHz gốc, trung bình 10 file.) Vì vậy `--dither auto` cộng nhiễu σ = 1
(đơn vị int16, giống `dither` của Kaldi) khi sr gốc < 16 kHz. Với input ≥ 16 kHz thì không cộng,
giữ đúng như upstream.

## Ghi chú

- Bước 4 (×32768) là bắt buộc: upstream đọc wav dạng int16, nên `cmvn.ark` được tính trên biên độ
  cỡ ±32768. Bỏ bước này thì log-mel lệch khoảng 20.8 và model hỏng.
- Các số "polyphase" ở trên đo bằng `scipy.signal.resample_poly`, vì máy dựng experiment không cài
  được librosa. Hãy chạy lại `check_steps.py` trên máy bạn để có số đúng của `soxr_hq`.
