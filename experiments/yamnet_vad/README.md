# Báo cáo Excel cho YAMNet

Cùng quy trình và cùng bố cục Excel với [`../firered_vad`](../firered_vad) và [`../silero_vad`](../silero_vad)
(phần chung: [`../common/vad_excel.py`](../common/vad_excel.py)), để YAMNet, FireRed và Silero được chấm trên
đúng cùng tập file, cùng chia dev/test, cùng ngưỡng quét và cùng cách đổi sang bin 0.5 s.

## Hai nguồn score (chọn một)

```bash
pip install -r requirements.txt

# 1. Chạy model .tflite / .onnx của bạn trên audio
python excel_report.py --audio-root "D:/data/phase_1" --gt-root "D:/data/Groundtruth" \
    --model "D:/models/yamnet_vad.tflite" -o out/yamnet_report.xlsx

# 2. Dùng kết quả có sẵn (vd. output của lib.so): mỗi file audio một file `index, start, end, score`
python excel_report.py --audio-root ... --gt-root ... --scores-dir "D:/results/lib_so"
#    tên file score khác tên audio: --scores-pattern "{Category}_{index}.txt"   ({stem} {category} {index})
```

- `--model`: dùng lại bộ đọc model của `vadcmp` (đọc input shape, softmax / logits, model lượng tử hoá).
  Số mẫu mỗi cửa sổ lấy từ model (vd. 15 600 mẫu = 0.975 s), hop `--hop-s 0.08`, lớp speech `--speech-index 1`.
  `--tail drop` (mặc định, giống lib: bỏ phần cuối không đủ một cửa sổ) hoặc `--tail pad`.
  Score từng cửa sổ được cache, nên đổi cách gộp / hậu xử lý không phải chạy lại model.
- `--scores-dir`: không chạy model; audio chỉ được giải mã một lần để lấy độ dài (cũng có cache).

## Gộp cửa sổ → score mỗi hop

`--rescore triangle` (mặc định; trọng số tam giác theo khoảng cách tâm hop – tâm cửa sổ) | `mean` | `max`,
rồi `--smooth-hops 5` (trung bình trượt 5 hop, `0` = tắt). Mặc định khớp mô tả A1–A4 trong sheet Snapshot.
Cách gộp `mean` / `max` khớp từng giá trị với `vadcmp.timeline.window_scores_to_hops`.

## Ba cấu hình (ô B7)

| | Ý nghĩa | Tham số |
|---|---|---|
| `Y0` | score hop ≥ ngưỡng, không hậu xử lý | |
| `Y1` (mặc định) | hậu xử lý hiện tại của lib = **A4**: pad 100 / 120 ms → drop 0 → merge < 500 ms | `--y1-pad-before/after`, `--y1-drop`, `--y1-merge`, `--y1-order` |
| `Y2` | máy trạng thái kiểu FireRed (`vadcmp/decide.py`): ngưỡng ra = ngưỡng − 0.1, đoạn > 0.1 s, lặng > 0.16 s mới cắt, hangover, co đuôi 0.08 s | `--y2-thr-off`, `--y2-min-speech`, `--y2-min-silence`, `--y2-hangover`, `--y2-pad-before/after` |

Các tham số chung (`--xlsx`, `--split-csv`, `--dev-ratio`, `--threshold`, `--rebin`, `--min-overlap`
(mặc định 20 %), `--dcf-weight`, `--criterion`, `--quality`, `--bootstrap`…) giống hệt báo cáo FireRed.
Ngưỡng cố định mặc định **0.6026**.

Với `--xlsx Tong_hop_danh_gia_VAD_test.xlsx`: dùng đúng chia dev/test trong file, và mục **D** đặt số YAMNet
mới tính cạnh số YAMNet đã có trong sheet *KQ du lieu that*. Muốn hai bên trùng nhau thì cấu hình cũng phải
giống (B7 = Y1, cùng ngưỡng, cùng quy tắc rebin): chênh lệch còn lại cho thấy cách tính hai nơi khác nhau ở đâu.
