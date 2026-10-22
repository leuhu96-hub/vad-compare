# Plan: Phát hiện đoạn có âm thanh bằng RMS để áp dụng ducking

> Tài liệu này mở rộng `audio-ducking-pipeline.md` và `keep-music-plan.md`. Giữ nguyên mọi quy ước đã có: Java, minSdk 34, `VadSegment` (giây), `SegmentPreprocessor`, `GainEnvelope`, `DuckingSession`, `AudioDecoder`, `AnalysisCoordinator`, `DuckRegionBuilder`. Tài liệu này chỉ mô tả phần thêm mới và các chỗ cần sửa.

## 1. Mục tiêu

1. Phân tích **track âm thanh của video** (không phải nhạc nền), tính RMS theo từng khung thời gian ngắn.
2. Đánh dấu các khung có âm thanh (vượt ngưỡng), gộp thành danh sách segment `VadSegment[]` (giây).
3. Dùng danh sách này làm **một nguồn kích hoạt ducking mới**, bên cạnh VAD: người dùng chọn ducking theo "Giọng nói" (VAD) hoặc "Mọi âm thanh" (RMS).
4. Lưu lại mảng độ lớn theo khung (dBFS) để giao diện có thể vẽ làn "Âm thanh video".

## 2. Lưu ý quan trọng trước khi làm

RMS **không phân biệt** giọng nói, nhạc, tiếng ồn hay tiếng gió. Nó chỉ đo "có âm thanh đủ lớn hay không". Vì vậy:

- Hợp với video có nền yên tĩnh: quay trong phòng, voice-over, screencast, video sản phẩm.
- Không hợp với video ồn liên tục (đường phố, quán cà phê, có nhạc gốc suốt): gần như cả video vượt ngưỡng, nhạc nền bị hạ gần như toàn bộ thời gian. Đây là hành vi đúng của thuật toán, không phải lỗi.
- Ở chế độ RMS, option "Keep music" gần như dư thừa vì nhạc gốc của video đã được RMS tính là "có âm thanh". Vẫn giữ logic hợp như cũ, không cần xử lý riêng.

Do đó RMS là **chế độ tùy chọn**, VAD vẫn là mặc định.

## 3. Ràng buộc kỹ thuật

- Java, minSdk 34. Không thêm thư viện bên thứ ba.
- Phần tính toán (RMS, ngưỡng, hysteresis, tạo segment) là **Java thuần**, không phụ thuộc Android, để test bằng JUnit.
- Phân tích chạy trên background thread, trong cùng lượt decode với `lib.so` (decode một lần, đẩy PCM cho cả hai bộ phân tích).
- Không giữ toàn bộ PCM trong RAM. Chỉ giữ mảng dB theo khung (10 phút ở khung 20 ms là 30.000 float, khoảng 120 KB).

## 4. Tham số cấu hình (`RmsConfig`, bất biến, tạo qua Builder, có `validate()`)

| Tham số | Kiểu | Mặc định | Ý nghĩa |
|---|---|---|---|
| `frameMs` | int | 20 | Độ dài một khung tính RMS |
| `thresholdMode` | enum | `ADAPTIVE` | `ADAPTIVE` (tự tính theo nền ồn) hoặc `FIXED` |
| `fixedThresholdDb` | float | -40 | Ngưỡng bật khi `FIXED` (dBFS) |
| `noiseFloorPercentile` | float | 10 | Phân vị dùng để ước lượng nền ồn (%) |
| `marginDb` | float | 12 | Ngưỡng bật = nền ồn + margin (khi `ADAPTIVE`) |
| `minOnThresholdDb` | float | -55 | Ngưỡng bật không bao giờ thấp hơn mức này |
| `hysteresisDb` | float | 3 | Ngưỡng tắt = ngưỡng bật − hysteresis |
| `silenceFloorDb` | float | -120 | Giá trị dB thấp nhất (khung im lặng tuyệt đối) |

Validate: `frameMs` trong [5, 100]; `noiseFloorPercentile` trong [0, 100]; `marginDb ≥ 0`; `hysteresisDb ≥ 0`; sai thì ném `IllegalArgumentException`.

Các tham số `mergeGapMs`, `padMs`, `minSegmentMs` **không** lặp lại ở đây. Chúng thuộc `SegmentPreprocessor` và áp dụng cho segment RMS giống hệt segment VAD.

## 5. Pipeline xử lý

```
Video file
  └─► AudioDecoder (MediaExtractor + MediaCodec, đã có)
        └─► PCM chunk (interleaved, nhiều kênh)
              ├─► lib.so classifier (đã có)
              └─► [R1] FrameRmsAccumulator → float[] frameDb
                        └─► [R2] ThresholdEstimator → onDb, offDb
                              └─► [R3] HysteresisSegmenter → VadSegment[] (giây)
                                    └─► [R4] LoudnessResult
                                          └─► DuckRegionBuilder (chọn nguồn VAD/RMS)
                                                └─► SegmentPreprocessor → GainEnvelope
```

### R1. Tính RMS theo khung (`FrameRmsAccumulator`, Java thuần)

Đầu vào mỗi lần gọi: mảng `float[]` interleaved trong [-1, 1], `channelCount`, `sampleRate`.

1. **Chuẩn hóa PCM** (làm trong `AudioDecoder` hoặc adapter trước khi đưa vào):
   - Đọc `KEY_PCM_ENCODING` từ `getOutputFormat()`. Không có key thì coi là PCM 16-bit.
   - `ENCODING_PCM_16BIT`: `sample / 32768f`.
   - `ENCODING_PCM_FLOAT`: dùng trực tiếp.
   - Đọc `KEY_SAMPLE_RATE` và `KEY_CHANNEL_COUNT` từ output format sau `INFO_OUTPUT_FORMAT_CHANGED`.
2. **Độ dài khung theo mẫu:** `frameSamples = Math.round(sampleRate * frameMs / 1000.0)` (mỗi kênh). Ví dụ 16 kHz, 20 ms → 320; 48 kHz → 960. Không cần resample.
3. **Cộng năng lượng trên tất cả các kênh, không downmix trước.** Với mỗi khung: `sumSq += s * s` cho mọi mẫu của mọi kênh, đếm `n = frameSamples × channelCount`.
   - Lý do: nếu downmix về mono trước, hai kênh lệch pha (L = s, R = −s) sẽ triệt tiêu thành 0 và khung bị coi là im lặng.
4. **RMS và dB:**
   ```
   rms = sqrt(sumSq / n)
   db  = rms > 0 ? 20 * log10(rms) : silenceFloorDb
   db  = max(db, silenceFloorDb)
   ```
5. **Khung chạy qua ranh giới chunk:** giữ `sumSq` và số mẫu đã cộng giữa các lần gọi, chỉ chốt khung khi đủ `frameSamples`.
6. **Khung cuối không đủ mẫu:** nếu có ít nhất một nửa `frameSamples` thì tính RMS trên số mẫu hiện có; ít hơn thì bỏ.
7. **Mốc thời gian:** `startOffsetMs` = `presentationTimeUs / 1000` của output buffer đầu tiên có dữ liệu (thường là 0, nhưng một số file AAC có độ trễ đầu). Khung thứ `i` bắt đầu tại `startOffsetMs + i × frameMs`.
8. Đầu ra: `float[] frameDb` (dùng `FloatArrayList` tự viết hoặc mảng tăng dần), `int frameMs`, `long startOffsetMs`.

### R2. Ước lượng ngưỡng (`ThresholdEstimator`, Java thuần)

Chạy **sau** khi có toàn bộ `frameDb` (phân tích offline nên biết trước cả file).

- `FIXED`: `onDb = fixedThresholdDb`.
- `ADAPTIVE`:
  1. `noiseFloorDb` = phân vị `noiseFloorPercentile` của `frameDb`, theo cách nearest-rank: sắp xếp tăng dần một **bản sao**, lấy phần tử tại chỉ số `ceil(p / 100 × N) − 1` (chặn trong [0, N − 1]).
  2. `onDb = max(noiseFloorDb + marginDb, minOnThresholdDb)`.
- `offDb = onDb − hysteresisDb`.
- `frameDb` rỗng: `noiseFloorDb = silenceFloorDb`, `onDb` tính như trên, segmenter trả về rỗng.

Ý nghĩa của `minOnThresholdDb`: video có các đoạn im lặng tuyệt đối (−120 dB) làm nền ồn rất thấp. Nếu không chặn, ngưỡng sẽ thấp tới mức tiếng xì nhỏ nhất cũng kích hoạt ducking.

### R3. Tạo segment bằng hysteresis (`HysteresisSegmenter`, Java thuần)

Máy trạng thái hai trạng thái:

```
active = false
for i in 0..N-1:
    if !active && frameDb[i] >= onDb:  active = true;  startFrame = i
    if  active && frameDb[i] <  offDb: active = false; emit(startFrame, i)   // [start, i)
if active: emit(startFrame, N)
```

- Segment `[startFrame, endFrame)` đổi sang giây: `start = (startOffsetMs + startFrame × frameMs) / 1000.0`, `end = (startOffsetMs + endFrame × frameMs) / 1000.0`.
- Hysteresis giúp một âm thanh dao động quanh ngưỡng không bị cắt thành nhiều đoạn vụn. Không cần làm mượt thêm: các đoạn ngắn và khoảng nghỉ ngắn đã được `SegmentPreprocessor` xử lý (`minSegmentMs`, `mergeGapMs`).
- Chỉ dùng các phép so sánh đúng như trên (`>=` khi bật, `<` khi tắt) để test ở mục 9 ra đúng kết quả.

### R4. Kết quả (`LoudnessResult`)

```java
public final class LoudnessResult {
    public final float[] frameDb;        // bản sao, không cho sửa
    public final int frameMs;
    public final long startOffsetMs;
    public final float noiseFloorDb;
    public final float onDb;
    public final float offDb;
    public final VadSegment[] activeSegments;  // giây, đã sắp xếp
}
```

Giữ `LoudnessResult` trong `AnalysisResult` (thêm field `loudness`, có thể `null` nếu phân tích lỗi) để đổi chế độ hoặc đổi ngưỡng không phải decode lại. Đổi `thresholdMode`, `marginDb`, `fixedThresholdDb` hay `hysteresisDb` chỉ cần chạy lại R2 và R3 trên `frameDb` có sẵn (vài ms).

## 6. Tích hợp vào pipeline có sẵn

### 6.1 `AnalysisCoordinator`

- Trong vòng decode, mỗi chunk PCM được đẩy cho cả `NativeAudioClassifier` (theo định dạng nó yêu cầu, có thể là mono đã resample) và `FrameRmsAccumulator` (PCM gốc, interleaved, sample rate gốc).
- Sau khi decode xong: chạy R2, R3, tạo `LoudnessResult`.
- Lỗi ở phần RMS không được làm hỏng kết quả `lib.so` và ngược lại. Mỗi phần bọc try/catch riêng.
- Hỗ trợ `cancel()` như cũ.
- Callback tiến độ: `onProgress(float fraction)` theo `presentationTimeUs / durationUs`, gọi trên main thread, tối đa khoảng 10 lần/giây.

### 6.2 `DuckRegionBuilder`

Thêm enum và đổi chữ ký:

```java
public enum DuckSource { VOICE, ANY_SOUND }

public static VadSegment[] build(VadSegment[] voice,
                                 VadSegment[] sound,
                                 List<ClassifiedSegment> music,
                                 DuckSource source,
                                 boolean keepMusic)
```

- `VOICE`: dùng `voice` (như cũ).
- `ANY_SOUND`: dùng `sound` (kết quả RMS).
- `keepMusic = true`: hợp thêm `music` như cũ, ở cả hai chế độ.
- `sound == null` (RMS lỗi) và chọn `ANY_SOUND`: rơi về `voice`, báo cho UI hiển thị thông báo.
- Hàm cũ (không có tham số `source`) giữ lại và gọi sang hàm mới với `VOICE`, để không làm vỡ code đang dùng.

### 6.3 `DuckingSession`

Không cần đổi. Vẫn nhận `VadSegment[]` qua `updateSegments(...)`, đi qua `SegmentPreprocessor` và `GainEnvelope` như cũ.

## 7. Giao diện (tối thiểu)

- Thêm vào menu overflow (cạnh "Keep music") một mục chọn một trong hai: **"Hạ nhạc khi: Có giọng nói / Có âm thanh"**. Mặc định "Có giọng nói".
- Chế độ "Có âm thanh": thay làn "Giọng nói" trên timeline bằng làn **"Âm thanh"**, vẽ từ `frameDb` (cột cao theo dB, ánh xạ tuyến tính từ `silenceFloorDb`… 0 dB, chặn dưới ở −60 dB để nhìn rõ) và tô các `activeSegments`. Có thể vẽ một vạch ngang mờ ở mức `onDb`.
- Phần nâng cao (tùy chọn, có thể làm sau): slider **"Độ nhạy"** ánh xạ ngược với `marginDb` (độ nhạy cao = margin thấp), khoảng 3–24 dB, hiển thị số dB.
- Lưu lựa chọn chế độ vào `SharedPreferences`.

## 8. Edge case

- Video không có track audio: `frameDb` rỗng, `activeSegments` rỗng, nhạc nền không bị hạ.
- Video toàn im lặng tuyệt đối: mọi khung = `silenceFloorDb`, ngưỡng bị chặn ở `minOnThresholdDb`, không có segment.
- Video ồn liên tục: gần như một segment dài cả video. Đúng hành vi; UI có thể hiện gợi ý "Video có âm thanh liên tục, nên dùng chế độ Giọng nói".
- File 5.1/7.1: công thức cộng năng lượng mọi kênh vẫn đúng.
- Output encoding không phải 16-bit hay float (hiếm): báo lỗi rõ ràng cho phần RMS, không crash, phần `lib.so` vẫn chạy nếu được.
- Hủy giữa chừng: không gọi callback, giải phóng `MediaCodec`/`MediaExtractor`.

## 9. Test bắt buộc (JUnit, không cần thiết bị)

Dùng sine tần số **400 Hz ở 16 kHz**: khung 20 ms chứa đúng 8 chu kỳ nên RMS chính xác. Sai số so sánh 1e-3 dB trừ khi ghi khác.

### 9.1 `FrameRmsAccumulator`

| Input (mono, 16 kHz, khung 20 ms) | dB mong đợi mỗi khung |
|---|---|
| Sine biên độ 1.0 | −3.0103 |
| Sine biên độ 0.5 | −9.0309 |
| Sine biên độ 0.1 | −23.0103 |
| Toàn số 0 | −120 (`silenceFloorDb`) |

Thêm:
- **Stereo lệch pha:** L = sine biên độ 1.0, R = −L. Kết quả phải là −3.0103 dB mỗi khung, **không** phải −120.
- **Chunk cắt ngang khung:** đưa 1 giây sine theo chunk 100 mẫu và theo một chunk duy nhất, kết quả `frameDb` phải giống hệt nhau (50 khung).
- **Khung cuối:** 330 mẫu (1 khung + 10 mẫu) → 1 khung; 480 mẫu (1 khung + 160 mẫu, đúng một nửa) → 2 khung.
- **PCM16:** giá trị 16384 sau chuẩn hóa = 0.5.

### 9.2 `ThresholdEstimator`

- 100 khung: 80 khung −60 dB, 20 khung −20 dB; `ADAPTIVE`, p = 10, `marginDb` = 15, `minOnThresholdDb` = −55, `hysteresisDb` = 3 → `noiseFloorDb` = −60, `onDb` = −45, `offDb` = −48.
- 100 khung toàn −80 dB, cùng cấu hình → `noiseFloorDb` = −80, `onDb` = −55 (bị chặn), `offDb` = −58.
- `FIXED` với `fixedThresholdDb` = −40, `hysteresisDb` = 3 → `onDb` = −40, `offDb` = −43.

### 9.3 `HysteresisSegmenter`

Khung 20 ms, `startOffsetMs` = 0, `frameDb` = `[−60, −30, −36, −30, −60]`:

| onDb | offDb | Segment mong đợi (ms) |
|---|---|---|
| −35 | −38 | `[20–80]` |
| −35 | −35 (không hysteresis) | `[20–40]`, `[60–80]` |

Thêm: khung cuối vẫn active → segment kết thúc tại `N × frameMs`; `startOffsetMs` = 23 dịch toàn bộ segment thêm 23 ms.

### 9.4 Tích hợp end-to-end (Java thuần)

Mono 16 kHz, 3 giây: 0–1 s toàn 0, 1–2 s sine 400 Hz biên độ 0.5, 2–3 s toàn 0. Cấu hình mặc định (`ADAPTIVE`, margin 12, `minOnThresholdDb` −55).
Mong đợi: `noiseFloorDb` = −120, `onDb` = −55, `offDb` = −58, `activeSegments` = `[1.000–2.000]` (giây).

### 9.5 `DuckRegionBuilder`

Voice `[1.2–3.5]`, sound `[0.5–4.0]`, `[6.0–7.0]`, music rỗng, `padMs` = 0, `mergeGapMs` = 300, `minSegmentMs` = 200:

| source | keepMusic | Vùng ducking (ms) |
|---|---|---|
| `VOICE` | false | `[1200–3500]` |
| `ANY_SOUND` | false | `[500–4000]`, `[6000–7000]` |
| `ANY_SOUND`, sound = `null` | false | `[1200–3500]` (rơi về voice) |

## 10. Danh sách file

1. Java thuần: `RmsConfig`, `FrameRmsAccumulator`, `ThresholdEstimator`, `HysteresisSegmenter`, `LoudnessResult`, `DuckSource`.
2. Sửa: `AudioDecoder` (chuẩn hóa PCM16/float, đưa ra PCM interleaved kèm `channelCount`, `sampleRate`, `presentationTimeUs`), `AnalysisCoordinator`, `AnalysisResult`, `DuckRegionBuilder`.
3. UI: mục chọn chế độ trong `menu_player.xml`, làn "Âm thanh" (`LoudnessLaneView`, custom `View` vẽ từ `frameDb`), lưu lựa chọn.
4. Unit test JUnit cho toàn bộ mục 9.

## 11. Tiêu chí hoàn thành

- Toàn bộ test mục 9 và test của các tài liệu trước đều pass.
- Phân tích video 10 phút không giữ PCM trong RAM, không chặn main thread.
- Đổi chế độ VAD/RMS khi đang phát: áp dụng ngay, gain ramp mượt (dùng `updateSegments`).
- Đổi ngưỡng/độ nhạy không decode lại file.
- Liệt kê các chỗ đã làm khác tài liệu này và lý do.
