# Plan: Phân loại âm thanh bằng lib.so và tùy chọn "Keep music" cho audio ducking

> Tài liệu này **mở rộng** `audio-ducking-pipeline.md` (Java, minSdk 34, MediaPlayer). Hãy đọc và giữ nguyên mọi quy ước của tài liệu đó (`VadSegment`, `SegmentPreprocessor`, `GainEnvelope`, `DuckingSession`). Phần dưới chỉ mô tả phần thêm mới và các chỗ cần sửa.

## 1. Mục tiêu

1. Gọi thư viện native `lib.so` (đã có JNI) từ Java để phân loại âm thanh của video. Kết quả là danh sách segment `(name, start, end)`, đơn vị **giây**.
2. Lưu toàn bộ kết quả phân loại, và lưu thêm **một danh sách riêng chỉ chứa các segment có label là music**.
3. Thêm một option **"Keep music"** trên Toolbar (dạng checkable).
   - **Tắt** (mặc định): nhạc nền chỉ bị ducking ở các vùng có giọng nói (từ model VAD), như hiện tại.
   - **Bật**: nhạc nền bị ducking ở các vùng có giọng nói **hoặc** có music (từ `lib.so`). Mục đích: video đã có nhạc riêng ở những đoạn đó, người dùng muốn giữ nhạc gốc của video nghe rõ, không bị nhạc nền đè lên.
4. Bật/tắt "Keep music" trong lúc đang phát phải áp dụng ngay, không cần phát lại.

## 2. Thông tin cần điền trước khi implement

ChatGPT: nếu mục nào chưa được điền, hãy đọc file JNI hiện có trong project để suy ra; nếu vẫn không rõ, hỏi lại thay vì tự đoán.

| Mục | Giá trị |
|---|---|
| Tên class Java chứa các hàm `native` | `______` (ví dụ `NativeLib`) |
| Chữ ký Java của 4 hàm native | `create()`: `______`; `scan(...)`: `______`; `process()`: `______`; `getClassification()`: `______` |
| `create()` có trả về handle (long) không? | `______` |
| Có hàm giải phóng native (`destroy`/`release`) không? | `______` |
| `scan` nhận toàn bộ audio một lần hay gọi nhiều lần theo từng chunk? Kích thước chunk? | `______` |
| Định dạng audio `scan` yêu cầu: sample rate, số kênh, dải giá trị | `______` Hz, `______` kênh, float trong [-1, 1]? |
| Kiểu dữ liệu `getClassification()` trả về qua JNI | `______` (ví dụ `Object[]`, `float[]`, `String` JSON...) |
| Danh sách label music có thể xuất hiện | `______` (mặc định: `"music"`) |
| Nơi lấy kết quả VAD hiện có (class/hàm) | `______` |

## 3. Luồng tổng thể

```
Video file
   │
   ├─► [A1] Decode audio → PCM float mono, đúng sample rate của lib.so
   │        │
   │        └─► [A2] NativeAudioClassifier: create → scan → process → getClassification
   │                 │
   │                 └─► [A3] Parse → List<ClassifiedSegment> (giây)
   │                          │
   │                          └─► [A4] AnalysisResult
   │                                   ├─ allSegments    (toàn bộ)
   │                                   └─ musicSegments  (chỉ label music)
   │
   └─► Model VAD (code hiện có) → voiceSegments (VadSegment[])

[B] DuckRegionBuilder.build(voiceSegments, musicSegments, keepMusic) → VadSegment[]
        │
        └─► DuckingSession.updateSegments(...) → SegmentPreprocessor → GainEnvelope
```

Phân tích (A1–A4) chạy trên background thread **trước khi phát**. Kết quả trả về main thread qua `Handler(Looper.getMainLooper())`.

## 4. Phần A — Gọi lib.so và lưu kết quả

### A1. Decode audio từ video (`AudioDecoder`)

- Dùng `MediaExtractor` chọn track audio đầu tiên (`mime` bắt đầu bằng `audio/`), `MediaCodec` để decode.
- Output decoder thường là PCM 16-bit. Đổi sang float: `sample / 32768f`.
- Nếu nhiều kênh: downmix về mono bằng trung bình các kênh.
- Nếu sample rate khác yêu cầu của `lib.so`: resample (linear interpolation là chấp nhận được cho phân loại).
- Đọc `KEY_SAMPLE_RATE`, `KEY_CHANNEL_COUNT` từ `getOutputFormat()` sau khi decoder báo `INFO_OUTPUT_FORMAT_CHANGED`, không lấy từ format của extractor.
- Video không có track audio: trả về mảng rỗng, không ném lỗi.
- Nếu `scan` hỗ trợ chunk: decode và đẩy từng chunk, không giữ toàn bộ audio trong RAM. Nếu chỉ nhận một lần: gom vào một `float[]` (10 phút ở 16 kHz mono khoảng 38 MB, cần lưu ý).

### A2. Wrapper native (`NativeAudioClassifier`)

Bọc các hàm JNI trong một class Java để đảm bảo đúng thứ tự gọi:

```
CREATED ──scan()──► SCANNED ──process()──► PROCESSED ──getClassification()──► DONE
```

- `create()` gọi trong constructor (hoặc `init()`).
- `scan()` chỉ hợp lệ sau `create()`. Nếu `scan` gọi theo chunk thì gọi nhiều lần ở trạng thái `CREATED/SCANNED`.
- `process()` chỉ hợp lệ sau ít nhất một lần `scan()`.
- `getClassification()` chỉ hợp lệ sau `process()`.
- Gọi sai thứ tự: ném `IllegalStateException`.
- Implement `AutoCloseable`; `close()` gọi hàm giải phóng native nếu có, gọi nhiều lần không crash.
- Không giả định thư viện native thread-safe: một instance chỉ dùng trên một thread, không gọi song song.
- Bắt mọi exception/lỗi từ JNI và trả ra ngoài dưới dạng exception Java có thông điệp rõ ràng.

### A3. Parse kết quả (`ClassificationParser`)

- Chuyển dữ liệu thô từ `getClassification()` thành `List<ClassifiedSegment>`.
- `ClassifiedSegment`: class `final` bất biến, field `String name`, `double start`, `double end` (giây). Có `VadSegment toVadSegment()`.
- Bỏ segment không hợp lệ: `name` null, `end <= start`, giá trị âm, `NaN`.
- Sắp xếp theo `start`.

### A4. Lưu kết quả (`AnalysisResult`)

```java
public final class AnalysisResult {
    public final List<ClassifiedSegment> allSegments;   // toàn bộ kết quả từ lib.so
    public final List<ClassifiedSegment> musicSegments; // chỉ label music
}
```

- Lọc music: `name.trim()` so sánh **không phân biệt hoa thường** với tập `musicLabels` (mặc định `{"music"}`, cấu hình được).
- Cả hai list là `Collections.unmodifiableList`.
- Giữ `AnalysisResult` trong Activity/ViewModel để bật/tắt "Keep music" không phải phân tích lại.

### A5. Điều phối (`AnalysisCoordinator`)

- Chạy trên `ExecutorService` một thread.
- Thứ tự: decode → classifier → parse → tạo `AnalysisResult`. Lấy kết quả VAD từ code hiện có.
- Callback: `onAnalysisDone(AnalysisResult result, VadSegment[] voiceSegments)`, `onAnalysisError(Exception e)`, gọi trên main thread.
- Hỗ trợ `cancel()`: khi Activity bị hủy giữa chừng thì dừng, đóng classifier, không gọi callback.

## 5. Phần B — Tổng hợp vùng ducking (`DuckRegionBuilder`, Java thuần)

```java
public static VadSegment[] build(VadSegment[] voice,
                                 List<ClassifiedSegment> music,
                                 boolean keepMusic)
```

- `keepMusic = false`: trả về `voice` (bản sao, `null` thành mảng rỗng).
- `keepMusic = true`: trả về `voice` + tất cả `music` (đổi qua `toVadSegment()`), tức **hợp (union)** hai tập.
- Không cần tự gộp đoạn chồng lấn ở đây: `SegmentPreprocessor` đã sắp xếp, nới, gộp và lọc.
- Logic quyết định vùng ducking chỉ nằm trong hàm này, để sau này đổi quy tắc (ví dụ chỉ lấy phần giao) chỉ cần sửa một chỗ.

## 6. Phần C — Option "Keep music" trên Toolbar

### C1. Menu

`res/menu/menu_player.xml`:

```xml
<item
    android:id="@+id/action_keep_music"
    android:title="Keep music"
    android:checkable="true"
    android:checked="false"
    app:showAsAction="never" />
```

### C2. Hành vi

- `onOptionsItemSelected`: đảo trạng thái `item.setChecked(!item.isChecked())`, cập nhật biến `keepMusic`, rồi gọi `applyDuckRegions()`.
- `applyDuckRegions()`: `session.updateSegments(DuckRegionBuilder.build(voice, result.musicSegments, keepMusic))`.
- Khi phân tích chưa xong: item bị disable (`setEnabled(false)` trong `onPrepareOptionsMenu`), bật lại khi có `AnalysisResult`.
- Khi ducking đang tắt: item vẫn bấm được, trạng thái được ghi nhớ và áp dụng khi ducking bật lại.
- Lưu `keepMusic` qua `onSaveInstanceState` (xoay màn hình) và `SharedPreferences` (lần mở sau).
- Nếu `musicSegments` rỗng: bật "Keep music" không thay đổi gì, có thể hiện `Toast` "Không phát hiện nhạc trong video".

## 7. Thay đổi cần có ở `DuckingSession`

Thêm API:

```java
public void updateSegments(VadSegment[] segments)
```

- Chạy lại `SegmentPreprocessor` với config hiện tại, thay danh sách segment trong `GainEnvelope`.
- Nếu đang phát: **không snap**, để gain ramp tự nhiên theo attack/release (tránh nhạc nền bị giật khi người dùng bấm option).
- Nếu đang pause: `snapTo(videoPos)` và set volume ngay.
- Gọi trên main thread.

## 8. Edge case

- Video không có audio: `allSegments` và `musicSegments` rỗng, ducking chỉ dựa vào VAD (cũng rỗng).
- `lib.so` lỗi hoặc ném exception: vẫn cho phát video, ducking chỉ theo VAD, disable "Keep music", báo lỗi qua callback.
- Segment music bao phủ gần hết video: nhạc nền bị hạ gần như toàn bộ thời gian, đây là hành vi đúng.
- Segment music và voice chồng lấn: được gộp thành một vùng liên tục.
- Bật/tắt "Keep music" liên tục nhiều lần: không crash, gain luôn tiến về đúng target.
- Activity bị hủy khi đang phân tích: hủy tác vụ, đóng native, không rò rỉ.

## 9. Test case bắt buộc (JUnit, không cần thiết bị)

### 9.1 Lọc music (`AnalysisResult` / bộ lọc)

Input: `("Speech", 0.0, 1.0)`, `(" Music ", 1.0, 4.0)`, `("MUSIC", 6.0, 8.0)`, `("Noise", 8.0, 9.0)`, `("music", 5.0, 5.0)`
Mong đợi: `allSegments` có 4 phần tử (phần tử cuối không hợp lệ bị bỏ), `musicSegments` có 2 phần tử `[1.0–4.0]`, `[6.0–8.0]`.

### 9.2 `DuckRegionBuilder` + `SegmentPreprocessor`

Cấu hình: `padMs = 0`, `mergeGapMs = 300`, `minSegmentMs = 200`.
Voice: `[1.2–3.5]`. Music: `[3.0–8.0]`, `[10.0–12.0]`.

| keepMusic | Vùng ducking mong đợi (ms) |
|---|---|
| false | `[1200–3500]` |
| true | `[1200–8000]`, `[10000–12000]` |

Thêm test: voice `null` và music rỗng trả về mảng rỗng; `keepMusic = true` với music rỗng cho kết quả giống `false`.

### 9.3 `NativeAudioClassifier` (dùng fake, không gọi native thật)

- Tách các lời gọi native qua một interface (ví dụ `NativeBridge`) để test bằng fake.
- Gọi `process()` trước `scan()` ném `IllegalStateException`.
- Gọi `getClassification()` trước `process()` ném `IllegalStateException`.
- Thứ tự đúng `create → scan → process → getClassification` chạy thành công.
- `close()` gọi hai lần không lỗi.

## 10. Tiêu chí hoàn thành và yêu cầu đầu ra

Tiêu chí:

- Toàn bộ test ở mục 9 và test của `audio-ducking-pipeline.md` pass.
- Bật "Keep music" khi đang phát: nhạc nền hạ mượt ở các đoạn video có nhạc, tắt thì trở lại như cũ.
- Phân tích không chặn main thread (không ANR).

Đầu ra:

1. Mã nguồn Java: `AudioDecoder`, `NativeBridge`, `NativeAudioClassifier`, `ClassifiedSegment`, `ClassificationParser`, `AnalysisResult`, `AnalysisCoordinator`, `DuckRegionBuilder`, và phần sửa `DuckingSession` (`updateSegments`).
2. `menu_player.xml` và phần sửa Activity (menu, lưu trạng thái, gọi phân tích trước khi phát).
3. Unit test JUnit cho mục 9.
4. Liệt kê các giả định đã dùng cho những mục còn trống ở bảng mục 2.
