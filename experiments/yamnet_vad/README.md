# Báo cáo Excel cho YAMNet

Cùng quy trình và cùng bố cục Excel với [`../firered_vad`](../firered_vad) và [`../silero_vad`](../silero_vad)
(phần chung: [`../common/vad_excel.py`](../common/vad_excel.py)), để YAMNet, FireRed và Silero được chấm trên
đúng cùng tập file, cùng chia dev/test, cùng ngưỡng quét và cùng cách đổi sang bin 0.5 s.

| File | Vai trò |
|---|---|
| `yamnet_tflite.py` | Chạy `yamnet.tflite` từng bước (không dùng `vadcmp`): đọc audio (dùng chung bước 1–3 của `firered_onnx.py`), cắt cửa sổ, interpreter TFLite → xác suất speech mỗi cửa sổ. Chạy một file: `python yamnet_tflite.py audio.m4a --model yamnet.tflite` |
| `yamnet_post.py` | Gộp cửa sổ → score mỗi hop, hậu xử lý Y0 / Y1 / Y2, đọc file score có sẵn |
| `excel_report.py` | Báo cáo Excel, dùng phần chung `../common/vad_excel.py` |

## Các bước của `yamnet_tflite.py`

1. đọc audio bất kỳ (sr, số kênh) bằng librosa → 2. trộn về 1 kênh → 3. resample 16 kHz, float32 [-1, 1]
4. cắt cửa sổ: dài = số mẫu input của model (YAMNet: 15 600 mẫu = 0.975 s), nhảy 0.08 s (1 280 mẫu);
   phần cuối không đủ một cửa sổ bị bỏ (`--tail drop`, giống lib) hoặc đệm 0 (`--tail pad`)
5. mỗi cửa sổ → interpreter → xác suất speech:
   - **input**: tự đọc shape (`[15600]`, `[1, 15600]`, `[-1, 15600]` …), model lượng tử hoá int8 / uint8 được
     đổi scale tự động; `--input-samples` để ghi đè
   - **output**: tự chọn tensor có 2 lớp (model Model Maker thường có thêm output 521 lớp của YAMNet gốc),
     hoặc `--output-index`; `[frames, lớp]` thì lấy trung bình; đã là xác suất thì giữ, không thì softmax
     (`--output auto | probs | logits`); lớp speech `--speech-index 1`
6. gộp cửa sổ → score hop (`yamnet_post.rescore`) → 7. hậu xử lý (`yamnet_post.y0/y1/y2_segments`)

TFLite runtime: `pip install ai-edge-litert` (hoặc `tflite-runtime`, hoặc `tensorflow`). File `.onnx` cùng
kiểu input cũng chạy được bằng onnxruntime.

## Hai nguồn score cho báo cáo (chọn một)

```bash
pip install -r requirements.txt

# 1. Chạy model .tflite (hoặc .onnx) của bạn trên audio
python excel_report.py --audio-root "D:/data/phase_1" --gt-root "D:/data/Groundtruth" \
    --model "D:/models/yamnet_vad.tflite" -o out/yamnet_report.xlsx

# 2. Dùng kết quả có sẵn (vd. output của lib.so): mỗi file audio một file `index, start, end, score`
python excel_report.py --audio-root ... --gt-root ... --scores-dir "D:/results/lib_so"
#    tên file score khác tên audio: --scores-pattern "{Category}_{index}.txt"   ({stem} {category} {index})
```

- Với `--model`, score từng cửa sổ được cache, nên đổi cách gộp / hậu xử lý không phải chạy lại model.
- Với `--scores-dir`, không chạy model; audio chỉ được giải mã một lần để lấy độ dài (cũng có cache).

## Gộp cửa sổ → score mỗi hop

`--rescore triangle` (mặc định; trọng số tam giác theo khoảng cách tâm hop – tâm cửa sổ) | `mean` | `max`,
rồi `--smooth-hops 5` (trung bình trượt 5 hop, `0` = tắt). Mặc định khớp mô tả A1–A4 trong sheet Snapshot.


## Ba cấu hình (ô B7)

| | Ý nghĩa | Tham số |
|---|---|---|
| `Y0` | score hop ≥ ngưỡng, không hậu xử lý | |
| `Y1` (mặc định) | hậu xử lý hiện tại của lib = **A4**: pad 100 / 120 ms → drop 0 → merge < 500 ms | `--y1-pad-before/after`, `--y1-drop`, `--y1-merge`, `--y1-order` |
| `Y2` | máy trạng thái kiểu FireRed (`yamnet_post.y2_segments`): ngưỡng ra = ngưỡng − 0.1, đoạn > 0.1 s, lặng > 0.16 s mới cắt, hangover, co đuôi 0.08 s | `--y2-thr-off`, `--y2-min-speech`, `--y2-min-silence`, `--y2-hangover`, `--y2-pad-before/after` |

Các tham số chung (`--xlsx`, `--split-csv`, `--dev-ratio`, `--threshold`, `--rebin`, `--min-overlap`
(mặc định 20 %), `--dcf-weight`, `--criterion`, `--quality`, `--bootstrap`…) giống hệt báo cáo FireRed.
Ngưỡng cố định mặc định **0.6026**.

Với `--xlsx Tong_hop_danh_gia_VAD_test.xlsx`: dùng đúng chia dev/test trong file, và mục **D** đặt số YAMNet
mới tính cạnh số YAMNet đã có trong sheet *KQ du lieu that*. Muốn hai bên trùng nhau thì cấu hình cũng phải
giống (B7 = Y1, cùng ngưỡng, cùng quy tắc rebin): chênh lệch còn lại cho thấy cách tính hai nơi khác nhau ở đâu.

## Đã kiểm tra (trên dữ liệu demo)

- Gộp `mean` / `max`, hậu xử lý Y1 và Y2 cho kết quả trùng với code tương ứng trong `vadcmp` (570 trường hợp),
  dù thư mục này không dùng `vadcmp`; báo cáo ra trùng từng dòng với bản trước.
- Đường chạy `.tflite` được thử với một interpreter giả cùng API (model có 2 output 521 / 2 lớp, input
  `[-1, 15600]`, cả bản float lẫn int8 / uint8 lượng tử hoá). Chưa thử với TFLite runtime thật vì máy thử
  không cài được; nếu lỗi, chạy `python yamnet_tflite.py <file> --model <model>` và gửi dòng `Model: …`.
