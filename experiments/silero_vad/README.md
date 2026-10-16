# Experiment: Silero VAD chạy trực tiếp bằng ONNX + báo cáo Excel

Cùng cách làm với [`../firered_vad`](../firered_vad): gọi thẳng `silero_vad.onnx` qua onnxruntime
(không cần torch hay package `silero-vad`), audio có sample rate / số kênh bất kỳ, và xuất báo cáo Excel
cùng bố cục sheet **KQ du lieu that** (mục A, B, C, D) như FireRed và YAMNet.

| File | Vai trò |
|---|---|
| `silero_onnx.py` | Đọc audio (dùng chung bước 1–3 của `firered_onnx.py`: librosa, trộn kênh, resample 16 kHz) → khối 512 mẫu + 64 mẫu ngữ cảnh + state → prob mỗi 32 ms |
| `silero_utils.py` | Hậu xử lý gốc `get_speech_timestamps_from_probs`, **chép nguyên văn** từ silero-vad (MIT) |
| `excel_report.py` | Báo cáo Excel, dùng phần chung `../common/vad_excel.py` |

## Cài đặt

```bash
pip install -r requirements.txt
```

Model: `--model đường/dẫn/silero_vad.onnx` (v5/v6, input `input`/`state`/`sr`). Mặc định tìm ở
`./models/silero_vad.onnx` rồi `../../models/silero_vad.onnx`.

Chạy thử một file: `python silero_onnx.py audio.m4a --threshold 0.5`

## Báo cáo Excel

```bash
# chỉ có audio + ground truth + model: tự chia dev/test (30 % dev, theo file, phân tầng category, seed 3)
python excel_report.py --audio-root "D:/data/phase_1" --gt-root "D:/data/Groundtruth" \
    --model models/silero_vad.onnx -o out/silero_report.xlsx

# toàn bộ là test, ngưỡng cố định
python excel_report.py --audio-root ... --gt-root ... --dev-ratio 0 --threshold 0.5

# đúng chia dev/test của file YAMNet + mục D so sánh
python excel_report.py --xlsx "Tong_hop_danh_gia_VAD_test.xlsx" --audio-root ... --gt-root ...
```

Ba cấu hình (ô B7):

| | Ý nghĩa |
|---|---|
| `S0` | prob mỗi khối 32 ms ≥ ngưỡng, không hậu xử lý |
| `S1` (mặc định) | hậu xử lý gốc Silero: vào khi prob ≥ ngưỡng, ra khi < ngưỡng − 0.15 (`--neg-offset`), lặng ≥ 100 ms mới cắt (`--min-silence-ms`), bỏ đoạn ≤ 250 ms (`--min-speech-ms`), pad 30 ms mỗi phía (`--speech-pad-ms`) |
| `S2` | prob 32 ms ≥ ngưỡng + hậu xử lý A4 của YAMNet: pad 100/120 ms → drop → merge < 500 ms (`--s2-*`) |

Mọi tham số chung (`--rebin`, `--min-overlap` (mặc định 20 %), `--dcf-weight`, `--criterion`,
`--threshold`, `--dev-ratio`, `--split-csv`, `--xlsx`, `--quality`, `--bootstrap`…) và cách đọc file Excel
giống hệt báo cáo FireRed: xem [`../firered_vad/README.md`](../firered_vad/README.md). Ngưỡng mặc định khi
không chọn trên dev là **0.5** (mặc định của Silero).

Prob được cache ở `out/probs_cache/` nên chạy lại với tham số khác chỉ mất khoảng 2 phút cho 4–5 giờ audio.

## Đã kiểm tra

- Prob từng khối trùng tuyệt đối với `vadcmp` (model `silero` của project chính) trên dữ liệu demo.
- Chuyển đoạn → bin 0.5 s trùng `dataset_reader.segments_to_bins` (500 trường hợp ngẫu nhiên, gồm bin cuối lẻ).
- Hậu xử lý S1 là hàm gốc của Silero, không viết lại.
