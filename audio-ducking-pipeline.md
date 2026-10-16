# Yêu cầu implement: Audio ducking cho video + nhạc nền trên Android (MediaPlayer)

## 1. Mục tiêu

Phát một file video và một file nhạc nền cùng lúc bằng hai `MediaPlayer`. Khi bật ducking, âm lượng nhạc nền tự động hạ xuống trong các đoạn có giọng nói (do VAD phát hiện trước), rồi tăng lại khi hết giọng nói. Âm thanh của video giữ nguyên.

- Ducking tắt: cả video và nhạc nền phát bình thường, nhạc nền ở âm lượng gốc.
- Ducking bật: âm lượng nhạc nền thay đổi theo danh sách segment.

## 2. Ràng buộc kỹ thuật

- Ngôn ngữ: **Java** (không dùng Kotlin). Nền tảng: Android, **minSdk 34**.
- Dùng `android.media.MediaPlayer` cho cả hai nguồn. Không dùng ExoPlayer, không dùng thư viện bên thứ ba.
- Mọi lời gọi tới `MediaPlayer` chạy trên main thread (`Handler` gắn `Looper.getMainLooper()`).
- Video là đồng hồ chuẩn (master clock). Gain của nhạc nền luôn tính từ vị trí của video.
- Phần logic tính toán (tiền xử lý segment, tính gain) phải là code Java thuần, không phụ thuộc Android, để viết unit test bằng JUnit.
- Vì minSdk 34, dùng trực tiếp `AudioFocusRequest`, `AudioAttributes`, `MediaPlayer.SEEK_CLOSEST`, không cần kiểm tra phiên bản API.

## 3. Input

1. `Uri` của file video.
2. `Uri` của file nhạc nền.
3. Danh sách segment giọng nói từ VAD, dạng **mảng** `VadSegment[]`. Mỗi phần tử có `start` và `end` tính bằng **giây**:

```java
public final class VadSegment {
    public final double start; // giây
    public final double end;   // giây

    public VadSegment(double start, double end) {
        this.start = start;
        this.end = end;
    }
}

VadSegment[] segments = {
    new VadSegment(1.20, 2.10),
    new VadSegment(2.30, 3.50),
    new VadSegment(6.00, 6.15)
};
```

## 4. Tham số cấu hình (`DuckingConfig`)

| Tham số | Kiểu | Mặc định | Ý nghĩa |
|---|---|---|---|
| `enabled` | boolean | true | Bật/tắt ducking |
| `duckAmountPercent` | float | 70 | Mức hạ âm lượng nhạc nền (%). 70 nghĩa là còn 30% |
| `backgroundBaseVolume` | float | 1.0 | Âm lượng gốc của nhạc nền (0..1) |
| `attackMs` | long | 80 | Thời gian hạ từ 1.0 xuống mức duck |
| `releaseMs` | long | 400 | Thời gian tăng từ mức duck lên 1.0 |
| `lookaheadMs` | long | 100 | Hạ nhạc sớm hơn thời điểm bắt đầu giọng nói |
| `tickMs` | long | 20 | Chu kỳ cập nhật gain |
| `mergeGapMs` | long | 300 | Gộp hai segment nếu khoảng cách ≤ giá trị này |
| `padMs` | long | 50 | Nới mỗi segment ra hai phía |
| `minSegmentMs` | long | 200 | Bỏ segment ngắn hơn giá trị này (sau khi gộp) |
| `driftThresholdMs` | long | 150 | Ngưỡng lệch để seek lại nhạc nền |
| `syncIntervalMs` | long | 1000 | Chu kỳ kiểm tra lệch đồng bộ |

`DuckingConfig` là class bất biến, tạo qua `Builder`, có phương thức `validate()`.

Công thức dẫn xuất:

- `duckGain = 1 − duckAmountPercent / 100` (ví dụ 70% → 0.3).
- Âm lượng thực tế đưa vào `setVolume` = `backgroundBaseVolume × gain`.
- Validate: `duckAmountPercent` trong [0, 100]; `backgroundBaseVolume` trong [0, 1]; `attackMs`, `releaseMs`, `tickMs` > 0. Sai thì ném `IllegalArgumentException`.

## 5. Pipeline xử lý

### Stage 1 — Tiền xử lý segment (`SegmentPreprocessor`, Java thuần)

Thứ tự xử lý bắt buộc:

1. Nhận mảng `VadSegment[]`. Mảng `null` hoặc phần tử `null` được coi như rỗng/bỏ qua.
2. Đổi giây sang ms: `Math.round(sec * 1000)`.
3. Bỏ segment không hợp lệ (`end <= start`, giá trị âm, `NaN`).
4. Sắp xếp theo `startMs`.
5. Nới: `start = Math.max(0, start − padMs)`, `end = end + padMs`. Nếu biết `videoDurationMs` thì clamp `end` về tối đa bằng nó.
6. Gộp: nếu `next.start − current.end ≤ mergeGapMs` thì gộp thành một đoạn (`end = Math.max(current.end, next.end)`). Cũng gộp các đoạn chồng lấn.
7. Bỏ segment có độ dài < `minSegmentMs`.

Output: `List<Segment>` (không sửa được, ví dụ `Collections.unmodifiableList`) đã sắp xếp, không chồng lấn. `Segment` là class `final` bất biến với hai field `long startMs`, `long endMs`, biểu diễn khoảng đóng `[startMs, endMs]`.

### Stage 2 — Khởi tạo và prepare hai player (`DuckingSession`)

- `videoPlayer`: `setDataSource(context, videoUri)`, `setSurface(surface)` từ `SurfaceView`/`TextureView`, `setAudioAttributes(USAGE_MEDIA, CONTENT_TYPE_MOVIE)`.
- `bgPlayer`: `setDataSource(context, bgUri)`, `setLooping(true)`, `setVolume(base, base)`, `setAudioAttributes(USAGE_MEDIA, CONTENT_TYPE_MUSIC)`.
- Gọi `prepareAsync()` cho cả hai. Chỉ chuyển sang trạng thái sẵn sàng khi **cả hai** đã gọi `onPrepared`.
- Đăng ký `setOnErrorListener` cho cả hai, báo lỗi ra ngoài qua `Listener`.

### Stage 3 — Start đồng thời

- Gọi liên tiếp trên main thread: `videoPlayer.start()`, `bgPlayer.start()`, rồi bắt đầu vòng lặp tick.
- Nếu `enabled = false`: vẫn chạy cả hai player nhưng không cập nhật gain (âm lượng nhạc nền cố định = `backgroundBaseVolume`). Vẫn chạy đồng bộ chống trôi.

### Stage 4 — Vòng lặp cập nhật gain (`GainEnvelope` Java thuần + tick trong `DuckingSession`)

Mỗi tick (dùng `Handler.postDelayed(runnable, tickMs)`):

1. `long pos = videoPlayer.getCurrentPosition();`
2. `dt = now − lastTickTime` (dùng `SystemClock.uptimeMillis()`), clamp `dt` trong [1, 100] ms. Không giả định dt luôn bằng `tickMs`.
3. `float gain = envelope.update(pos, dt);`
4. `bgPlayer.setVolume(base * gain, base * gain);`

Logic `GainEnvelope`:

```
isSpeaking(t) = tồn tại segment với startMs ≤ t ≤ endMs   // dùng binary search
speaking     = isSpeaking(pos) || isSpeaking(pos + lookaheadMs)
target       = speaking ? duckGain : 1.0
range        = 1.0 − duckGain
if range == 0: gain = 1.0, return

if gain > target:  gain = max(target, gain − range * dt / attackMs)
if gain < target:  gain = min(target, gain + range * dt / releaseMs)
return gain
```

- Phải kiểm tra cả `pos` và `pos + lookaheadMs`. Nếu chỉ kiểm tra `pos + lookaheadMs`, nhạc sẽ tăng lại sớm và đè lên cuối câu.
- `isSpeaking` dùng binary search trên danh sách đã sắp xếp (không duyệt tuyến tính), để seek ngẫu nhiên vẫn đúng.
- Cung cấp hàm `snapTo(long pos)`: đặt ngay `gain = target(pos)` không ramp và trả về gain đó. Dùng khi seek, resume hoặc bắt đầu phát.

### Stage 5 — Đồng bộ chống trôi

Mỗi `syncIntervalMs`:

```
bgDur = bgPlayer.getDuration()
if bgDur <= 0: bỏ qua
expected = videoPos % bgDur
if |bgPlayer.getCurrentPosition() − expected| > driftThresholdMs:
    bgPlayer.seekTo(expected, MediaPlayer.SEEK_CLOSEST)
```

### Stage 6 — Xử lý sự kiện

| Sự kiện | Hành vi |
|---|---|
| `pause()` | Pause cả hai player, dừng tick |
| `play()` / resume | Đồng bộ nhạc nền, `envelope.snapTo(videoPos)`, set volume, start cả hai, chạy lại tick |
| `seekTo(ms)` | `videoPlayer.seekTo(ms, SEEK_CLOSEST)`; trong `onSeekComplete` của video: seek nhạc nền tới `ms % bgDur`, `envelope.snapTo(ms)`, set volume ngay |
| `setDuckingEnabled(false)` | Dừng cập nhật gain, set volume nhạc nền = base |
| `setDuckingEnabled(true)` | `snapTo(videoPos)`, cập nhật gain lại |
| `updateConfig(config)` | Áp dụng config mới, tiền xử lý lại segment nếu tham số segment thay đổi |
| Video `onCompletion` | Dừng tick, pause nhạc nền, gọi `Listener.onCompleted()` |
| Activity `onPause` / `onStop` | Gọi `pause()` |
| Mất audio focus | Pause cả hai. Chỉ xin audio focus một lần cho cả phiên bằng `AudioFocusRequest` (`AUDIOFOCUS_GAIN`, `USAGE_MEDIA`) |

### Stage 7 — Giải phóng

`release()`: dừng tick, `handler.removeCallbacksAndMessages(null)`, stop và release cả hai player, `abandonAudioFocusRequest`. Gọi được nhiều lần mà không crash.

## 6. Kiến trúc class đề xuất

```
DuckingConfig            class bất biến, tạo qua Builder, có validate()
VadSegment               input từ VAD (giây)
Segment                  segment nội bộ (ms), bất biến
SegmentPreprocessor      VadSegment[] -> List<Segment>: đổi đơn vị, pad, merge, filter (Java thuần)
GainEnvelope             isSpeaking, update(pos, dt), snapTo(pos) (Java thuần)
DuckingSession           quản lý 2 MediaPlayer, tick, sync, sự kiện, audio focus
DuckingSession.Listener  interface: onReady(), onCompleted(), onError(Exception e)
```

API công khai của `DuckingSession`:

```java
public DuckingSession(Context context, DuckingConfig config, Listener listener)

public void prepare(Uri videoUri, Uri bgUri, Surface surface, VadSegment[] segments)
public void play()
public void pause()
public void seekTo(long ms)
public void setDuckingEnabled(boolean enabled)
public void updateConfig(DuckingConfig config)
public void release()
```

## 7. Edge case phải xử lý

- Mảng segment rỗng hoặc `null`: nhạc nền luôn ở gain 1.0.
- `duckAmountPercent = 0`: không ducking. `= 100`: tắt hẳn nhạc khi có giọng nói.
- Nhạc nền ngắn hơn video: lặp lại. Dài hơn video: dừng khi video kết thúc.
- Segment vượt quá độ dài video: clamp.
- Gọi `play()` trước khi cả hai player prepared: ghi nhớ yêu cầu và play khi sẵn sàng.
- Seek vào giữa một segment: gain phải bằng `duckGain` ngay, không ramp.
- `release()` được gọi trong lúc đang prepare: không crash, không rò rỉ.

## 8. Test case bắt buộc (JUnit, không cần thiết bị)

### 8.1 SegmentPreprocessor

Cấu hình: `padMs = 0`, `mergeGapMs = 300`, `minSegmentMs = 200`.
Input: `{ new VadSegment(1.20, 2.10), new VadSegment(2.30, 3.50), new VadSegment(6.00, 6.15) }`
Output mong đợi (ms): `[1200–3500]`
(Hai đoạn đầu cách nhau 200ms nên gộp; đoạn cuối dài 150ms nên bị bỏ.)

Thêm test: segment chồng lấn được gộp; input chưa sắp xếp vẫn ra đúng; `padMs = 50` không làm start âm; mảng `null` trả về danh sách rỗng.

### 8.2 GainEnvelope

Cấu hình: `duckAmountPercent = 70`, `attackMs = 80`, `releaseMs = 400`, `lookaheadMs = 100`, segment `[1200–3500]`, `dt = 20` mỗi bước, gain ban đầu 1.0. Chạy `update` tuần tự với pos tăng 20ms từ 1000 đến 4000.

| pos (ms) | gain mong đợi |
|---|---|
| 1080 | 1.000 |
| 1100 | 0.825 |
| 1120 | 0.650 |
| 1140 | 0.475 |
| 1160 | 0.300 |
| 3500 | 0.300 |
| 3520 | 0.335 |
| 3600 | 0.475 |
| 3900 | 1.000 |

So sánh với sai số 1e-3.

Thêm test: `snapTo(2000)` trả về 0.3; `snapTo(500)` trả về 1.0; segment rỗng luôn trả về 1.0; `duckAmountPercent = 0` luôn trả về 1.0.

## 9. Tiêu chí hoàn thành

- Toàn bộ unit test ở mục 8 pass.
- Khi phát thực tế: nhạc nền hạ trước khi giọng nói bắt đầu, không bật lên giữa các câu ngắt nghỉ ngắn, tăng lại mượt sau khi hết câu.
- Pause/resume/seek nhiều lần liên tục không làm lệch đồng bộ quá `driftThresholdMs` và không crash.
- Không rò rỉ `Handler`/`MediaPlayer` sau `release()`.

## 10. Yêu cầu đầu ra

1. Mã nguồn Java đầy đủ cho các class ở mục 6, có Javadoc ngắn.
2. Unit test JUnit cho `SegmentPreprocessor` và `GainEnvelope`.
3. Một `Activity` mẫu (Java): `TextureView` hoặc `SurfaceView`, nút Play/Pause, `SeekBar` để seek, `Switch` bật/tắt ducking, `SeekBar` thứ hai chỉnh `duckAmountPercent` (0–100).
4. Giải thích ngắn các quyết định thiết kế nếu khác với tài liệu này.
