# Plan: Implement giao diện màn hình chính (phương án B) cho app Audio Ducking

> Tài liệu này dùng cùng với `audio-ducking-pipeline.md` và `keep-music-plan.md`. Giữ nguyên mọi quy ước đã có ở hai tài liệu đó (Java, minSdk 34, `MediaPlayer`, `DuckingSession`, `VadSegment`, `ClassifiedSegment`, `AnalysisResult`, `DuckRegionBuilder`). Tài liệu này chỉ mô tả phần giao diện và cách nối giao diện với logic đã có.

## 1. Mục tiêu

Xây một màn hình duy nhất (`MainActivity`) để:

1. Chọn video và chọn nhạc nền.
2. Hiển thị video trên surface.
3. Hiển thị các đoạn giọng nói (VAD) và nhạc (từ `lib.so`) trên timeline, kèm playhead và thanh seek.
4. Nhạc nền hiển thị như **một làn (lane) trong timeline**; bấm vào làn này để đổi file nhạc nền.
5. Điều chỉnh âm lượng video, âm lượng nhạc nền, mức ducking bằng **nút**. Bấm nút mới hiện seekbar để chỉnh; bấm lại thì ẩn.
6. Bật/tắt ducking bằng một nút. Option "Keep music" nằm trong menu overflow của Toolbar.

## 2. Ràng buộc kỹ thuật

- Java, minSdk 34, Android View + XML layout (không dùng Compose).
- Được dùng AndroidX AppCompat, ConstraintLayout và Material Components. Không dùng thư viện bên thứ ba khác.
- Theme tối cố định (`Theme.Material3.Dark.NoActionBar`), không cần hỗ trợ theme sáng.
- Mọi chuỗi hiển thị đặt trong `res/values/strings.xml` (tiếng Việt). Mọi màu đặt trong `colors.xml`, mọi kích thước trong `dimens.xml`. Không hard-code trong layout hay Java.
- Màn hình dọc là chính. Khi xoay ngang, layout không được vỡ (cho phép cuộn bằng `NestedScrollView`).

## 3. Design tokens

### 3.1 Màu (`colors.xml`)

| Tên | Giá trị | Dùng cho |
|---|---|---|
| `bg` | `#0F1114` | Nền màn hình |
| `surface` | `#181B20` | Nền nút điều khiển, panel |
| `surface_raised` | `#22262D` | Nền menu popup |
| `lane_track` | `#1D2127` | Nền làn timeline |
| `border` | `#2C313A` | Viền nút |
| `border_menu` | `#353B45` | Viền menu |
| `video_surface` | `#05070A` | Nền khung video |
| `text_primary` | `#ECEEF1` | Chữ chính, playhead |
| `text_secondary` | `#A3AAB5` | Nhãn, giá trị phụ |
| `speech` | `#5B9BFF` | Đoạn giọng nói, màu nhấn chính |
| `music` | `#F5A04A` | Đoạn nhạc, chip "Keep music" |
| `bg_lane_fill` | `#2A2219` | Nền làn nhạc nền |
| `bg_lane_stroke` | `#6B4A26` | Viền làn nhạc nền |
| `bg_lane_text` | `#F5C089` | Chữ và icon trong làn nhạc nền |
| `control_active_bg` | `#1D2633` | Nền nút điều chỉnh đang mở |
| `status_idle` | `#6B7482` | Chấm trạng thái khi không ducking |
| `overlay_scrim` | `#B3000000` | Nền pill thời gian trên video |
| `play_button_bg` | `#24ECEEF1` | Nền nút phát (trắng 14%) |

### 3.2 Kích thước (`dimens.xml`)

| Tên | Giá trị |
|---|---|
| `screen_padding_h` | 16dp |
| `section_gap` | 12dp |
| `toolbar_height` | 56dp |
| `video_corner` | 12dp |
| `lane_label_width` | 60dp |
| `lane_label_gap` | 10dp |
| `lane_height` | 16dp |
| `lane_gap` | 8dp |
| `lane_corner` | 4dp |
| `bg_lane_height` | 40dp |
| `bg_lane_corner` | 8dp |
| `playhead_width` | 2dp |
| `playhead_overhang` | 4dp (thò ra trên và dưới vùng làn) |
| `control_height` | 72dp |
| `control_corner` | 12dp |
| `control_gap` | 8dp |
| `play_button_size` | 56dp |
| `icon_button_size` | 44dp |
| `status_dot` | 8dp |

### 3.3 Chữ

- Font: IBM Plex Sans (chữ thường) và IBM Plex Mono (số, thời gian). Dùng Downloadable Fonts qua Google Fonts provider của AndroidX, hoặc đặt file `.ttf` trong `res/font/`. Nếu không tải được thì fallback `sans-serif` / `monospace`.
- Cỡ chữ: tiêu đề Toolbar 18sp (semibold); tên file 13–14sp; nhãn làn và nhãn nút 12sp; giá trị trên nút 11sp mono; dòng trạng thái 13sp; nhãn panel 13sp; giá trị panel 13sp mono.

## 4. Cấu trúc màn hình (từ trên xuống)

```
CoordinatorLayout / ConstraintLayout (bg)
├── MaterialToolbar               "Audio Ducking" + overflow menu (Keep music)
└── NestedScrollView (fillViewport=true)
    └── LinearLayout vertical (padding 0 16dp 16dp, gap 12dp)
        ├── [1] VideoContainer     FrameLayout 16:9
        ├── [2] TimelineSection    nhãn làn + làn + playhead + seekbar + trạng thái
        ├── (Space weight=1)       đẩy phần điều khiển xuống đáy
        ├── [3] AdjustPanel        ẩn mặc định
        └── [4] ControlRow         4 nút bằng nhau
```

### [1] VideoContainer

- `FrameLayout`, chiều rộng `match_parent`, tỉ lệ 16:9 (dùng `ConstraintLayout` với `app:layout_constraintDimensionRatio="16:9"` hoặc tự đo trong `onMeasure`). Bo góc 12dp (`outlineProvider` + `clipToOutline=true`). Nền `video_surface`.
- Bên trong:
  - `TextureView` (`@+id/videoSurface`), `match_parent`. Giữ đúng tỉ lệ video thật bằng cách scale bằng `setTransform` (letterbox), không kéo méo hình.
  - `ImageButton` (`@+id/btnPlayPause`) ở giữa, 56dp, nền tròn `play_button_bg`, icon play/pause trắng.
  - `TextView` (`@+id/txtTime`) góc dưới trái, margin 10dp, nền pill `overlay_scrim` bo 6dp, padding 3dp/8dp, mono 12sp, dạng `00:14 / 01:00`.
  - `ImageButton` (`@+id/btnChangeVideo`) góc trên phải, margin 6dp, 44dp, nền tròn `#CC0F1114`, icon video. `contentDescription` = "Đổi video".
- **Trạng thái chưa có video:** ẩn `btnPlayPause`, `txtTime`, `btnChangeVideo`. Hiện một nút lớn ở giữa (`@+id/btnPickVideoEmpty`, `MaterialButton` dạng outlined) với icon video và chữ "Chọn video". Toàn bộ khung video cũng bấm được để chọn video.

### [2] TimelineSection

```
LinearLayout vertical (gap 8dp)
├── LinearLayout horizontal (gap 10dp)
│   ├── Cột nhãn (width 60dp, vertical, gap 8dp, 12sp, text_secondary)
│   │     "Giọng nói" (cao 16dp) · "Nhạc" (cao 16dp) · "Nhạc nền" (cao 40dp, căn giữa dọc)
│   └── FrameLayout laneArea (weight 1)
│       ├── LinearLayout vertical (gap 8dp)
│       │   ├── SegmentLaneView @+id/laneSpeech  (cao 16dp, màu speech)
│       │   ├── SegmentLaneView @+id/laneMusic   (cao 16dp, màu music)
│       │   └── BackgroundLaneButton @+id/laneBackground (cao 40dp)
│       └── PlayheadView @+id/playhead (match_parent, không nhận touch)
├── SeekBar @+id/seekPosition   (cao 24dp, max = độ dài video ms)
└── Dòng trạng thái (horizontal, gap 8dp, căn giữa dọc)
      ● chấm 8dp · TextView @+id/txtStatus (weight 1, 13sp) · Chip "Keep music" @+id/chipKeepMusic
```

**SegmentLaneView** (custom `View`, tự vẽ bằng `Canvas`):

- Thuộc tính: `durationMs`, `List<Segment>` (ms), `segmentColor`, `trackColor`.
- Vẽ nền bo góc 4dp màu `lane_track`, rồi vẽ từng segment thành hình chữ nhật bo góc 3dp. Toạ độ x = `startMs / durationMs × width`, độ rộng = `(endMs − startMs) / durationMs × width`, tối thiểu 2dp để đoạn rất ngắn vẫn nhìn thấy.
- Clamp segment vượt ngoài `[0, durationMs]`. `durationMs <= 0` thì chỉ vẽ nền.
- `setSegments(...)` gọi `invalidate()`. Không tạo object mới trong `onDraw` (dùng `Paint`, `RectF` tạo sẵn).
- Bấm (tap) vào làn: gọi callback `onSeekRequested(ms)` theo vị trí x. Kéo không làm gì.
- `contentDescription` động, ví dụ "Giọng nói: 4 đoạn".

**BackgroundLaneButton** (`MaterialButton` hoặc `LinearLayout` clickable, phải là view bấm được thật để có focus và TalkBack):

- Cao 40dp, nền `bg_lane_fill`, viền 1dp `bg_lane_stroke`, bo góc 8dp, padding trái 10dp, phải 6dp.
- Nội dung ngang: icon nốt nhạc 16dp → tên file (13sp, `bg_lane_text`, 1 dòng, ellipsize end, weight 1) → icon "đổi" (hai mũi tên) 16dp.
- Ripple khi bấm. `contentDescription` = "Đổi nhạc nền: <tên file>".
- **Chưa có nhạc nền:** nền trong suốt, viền nét đứt `bg_lane_stroke`, icon "+" và chữ "Chọn nhạc nền". `contentDescription` = "Chọn nhạc nền".
- Làn chỉ hiển thị file, **không** vẽ envelope gain (giữ đơn giản).

**PlayheadView** (custom `View`):

- Vẽ một vạch dọc 2dp màu `text_primary`, bo đầu, tại `x = positionMs / durationMs × width`.
- Phủ toàn bộ `laneArea`, thò ra 4dp ở trên và dưới (đặt `clipChildren=false` ở cha, hoặc tự cộng padding).
- `setClickable(false)`, `setImportantForAccessibility(IMPORTANT_FOR_ACCESSIBILITY_NO)` để không chặn touch của các làn và nút bên dưới.

**SeekBar vị trí:** `progressTint`/`thumbTint` = `text_primary`, `progressBackgroundTint` = `border`. Chỉ seek khi `fromUser = true`. Khi đang kéo, ngừng cập nhật progress từ vòng lặp phát để thumb không giật.

**Dòng trạng thái:**

| Điều kiện | Chấm | Chữ |
|---|---|---|
| Chưa chọn video | `status_idle` | "Chọn video để bắt đầu" |
| Đang phân tích | `status_idle` | "Đang phân tích âm thanh…" |
| Ducking tắt | `status_idle` | "Ducking đang tắt" |
| Ducking bật, vị trí hiện tại nằm trong vùng ducking | `speech` | "Đang hạ nhạc nền còn X%" |
| Ducking bật, ngoài vùng ducking | `speech` | "Nhạc nền ở mức X%" |

X là phần trăm âm lượng thực tế của nhạc nền (`backgroundBaseVolume × gain × 100`, làm tròn). Lấy gain hiện tại từ `DuckingSession` (xem mục 7). Cập nhật cùng nhịp với `txtTime`, tối đa khoảng 10 lần/giây, và chỉ `setText` khi chuỗi thay đổi.

Chip "Keep music": viền 1dp `music`, chữ `music` 11sp, nền trong suốt, không bấm được. Chỉ hiện khi "Keep music" đang bật.

**Trạng thái của timeline:**

- Chưa có video: hai làn speech/music chỉ vẽ nền; seekbar disabled.
- Đang phân tích: hai làn speech/music hiện chữ "Đang phân tích…" (11sp, `text_secondary`) căn giữa hoặc một `LinearProgressIndicator` indeterminate cao 2dp thay cho làn. Làn nhạc nền vẫn bấm được.
- `lib.so` lỗi: làn "Nhạc" hiện chữ "Không phân tích được nhạc"; ducking vẫn chạy theo VAD.

### [3] AdjustPanel

- `MaterialCardView` (`@+id/panelAdjust`), nền `surface`, viền 1dp `border`, bo góc 12dp, padding 12dp/14dp. Mặc định `GONE`.
- Nội dung: hàng trên gồm nhãn (`@+id/txtPanelLabel`, 13sp) bên trái và giá trị (`@+id/txtPanelValue`, mono 13sp, `text_secondary`) bên phải; bên dưới là `Slider` của Material (`@+id/sliderAdjust`, `valueFrom=0`, `valueTo=100`, `stepSize=1`, cao tối thiểu 44dp).
- Panel dùng chung cho cả ba loại điều chỉnh. Khi đổi loại thì đổi nhãn, giá trị, màu track:

| Loại | Nhãn | Giá trị hiển thị | Màu track |
|---|---|---|---|
| `VIDEO` | "Âm lượng video" | `100%` | `speech` |
| `BACKGROUND` | "Âm lượng nhạc nền" | `80%` | `music` |
| `DUCK_AMOUNT` | "Mức ducking" | `70% · −10.5 dB` | `speech` |

- Hiện/ẩn có animation ngắn (150–200ms): dùng `TransitionManager.beginDelayedTransition(parent, new AutoTransition())` trước khi đổi visibility.
- Khi kéo slider: cập nhật chữ giá trị và áp dụng ngay vào phát (mục 7). Không cần nút "OK".

### [4] ControlRow

- `LinearLayout` ngang hoặc `GridLayout` 4 cột bằng nhau, gap 8dp, cao 72dp.
- Mỗi nút là một view bấm được (`MaterialButton` tuỳ biến hoặc `LinearLayout` clickable + ripple), bo góc 12dp, viền 1dp, nội dung dọc căn giữa: icon 20dp → nhãn 12sp → giá trị 11sp mono `text_secondary`.

| ID | Icon | Nhãn | Giá trị | Bấm |
|---|---|---|---|---|
| `btnVideoVolume` | video | "Video" | `100%` | Mở/đóng panel `VIDEO` |
| `btnBgVolume` | nốt nhạc | "Nhạc nền" | `80%` | Mở/đóng panel `BACKGROUND` |
| `btnDuckAmount` | loa gạch chéo | "Mức duck" | `70%` | Mở/đóng panel `DUCK_AMOUNT` |
| `btnDuckToggle` | nút nguồn | "Ducking" | "Bật"/"Tắt" | Bật/tắt ducking |

Trạng thái giao diện của nút:

- Nút điều chỉnh bình thường: nền `surface`, viền `border`.
- Nút điều chỉnh đang mở panel: nền `control_active_bg`, viền `speech`. Chỉ một nút được mở tại một thời điểm. Bấm nút khác thì chuyển panel sang loại đó. Bấm lại chính nút đang mở thì đóng panel.
- `btnDuckToggle` bật: nền `speech`, chữ và icon `bg`. Tắt: nền `surface`, viền `border`, chữ `text_primary`.
- Khi ducking tắt: `btnDuckAmount` có `alpha = 0.45` và vẫn bấm được (người dùng có thể chỉnh trước rồi bật sau).
- Dùng `setSelected`/`setActivated` + `ColorStateList`/selector drawable cho các trạng thái trên, không đổi màu thủ công rải rác.
- Accessibility: các nút điều chỉnh đặt `stateDescription` "Đang mở"/"Đã đóng"; `btnDuckToggle` dùng `setStateDescription("Bật"/"Tắt")`.

### Toolbar và menu

- `MaterialToolbar` cao 56dp, nền `bg`, tiêu đề "Audio Ducking".
- `res/menu/menu_player.xml` có item `action_keep_music` (checkable, `showAsAction="never"`) như trong `keep-music-plan.md`.
- Popup menu: nền `surface_raised`, viền `border_menu`, bo góc 12dp (cấu hình qua `popupMenuStyle`/`popupTheme` trong theme).
- Khi bật/tắt "Keep music": cập nhật chip, gọi `applyDuckRegions()` như `keep-music-plan.md` mục 6.

## 5. Chọn file

- Dùng `registerForActivityResult(new ActivityResultContracts.OpenDocument(), ...)`:
  - Video: `launch(new String[]{"video/*"})`.
  - Nhạc nền: `launch(new String[]{"audio/*"})`.
- Sau khi chọn: `getContentResolver().takePersistableUriPermission(uri, FLAG_GRANT_READ_URI_PERMISSION)`.
- Lấy tên hiển thị qua `OpenableColumns.DISPLAY_NAME`, độ dài qua `MediaMetadataRetriever` (`METADATA_KEY_DURATION`), chạy trên background thread.
- **Đổi video:** dừng phát, `release()` session cũ, xoá kết quả phân tích cũ, chạy lại phân tích (VAD + `lib.so`) cho video mới, rồi prepare session mới.
- **Đổi nhạc nền:** giữ nguyên kết quả phân tích; prepare lại session với nhạc nền mới, tiếp tục từ vị trí video hiện tại, giữ trạng thái play/pause.
- Lưu URI video và nhạc nền vào `SharedPreferences` để mở lại app thì nạp lại (nếu quyền đọc còn hiệu lực; mất quyền thì quay về trạng thái chưa chọn).

## 6. Trạng thái màn hình (`ScreenState`)

| Trạng thái | Video | Timeline | Nút điều khiển | Keep music |
|---|---|---|---|---|
| `EMPTY` (chưa có video) | Nút "Chọn video" ở giữa | Làn trống, seek disabled | Disabled (alpha 0.45, không bấm được) | Disabled |
| `ANALYZING` | Hiện frame đầu, nút phát disabled | Chữ "Đang phân tích…" | Bấm được, chỉnh trước | Disabled |
| `READY` | Phát được | Đầy đủ segment | Bấm được | Bấm được |
| `ERROR` | Hiện Snackbar lỗi, có nút "Thử lại" | Như `READY` nếu chỉ `lib.so` lỗi | Bấm được | Disabled nếu `lib.so` lỗi |

Toàn bộ cập nhật giao diện đi qua một hàm `render(UiState state)` duy nhất trong Activity, để tránh mỗi chỗ tự đổi visibility.

## 7. Nối giao diện với logic phát

`UiState` (class bất biến, Java thuần) chứa: `screenState`, `videoName`, `bgName`, `durationMs`, `positionMs`, `playing`, `videoVolumePercent`, `bgVolumePercent`, `duckAmountPercent`, `duckingEnabled`, `keepMusic`, `openPanel` (`NONE`/`VIDEO`/`BACKGROUND`/`DUCK_AMOUNT`), `speechSegments`, `musicSegments`, `currentBgGain`.

Giữ `UiState` trong một `ViewModel` (`AndroidViewModel`) để xoay màn hình không mất trạng thái. `DuckingSession` cũng sống trong ViewModel; `Surface` được gắn lại khi `TextureView` tạo surface mới.

Cần bổ sung vào `DuckingSession` (nếu chưa có):

```java
public void setVideoVolume(float volume)        // 0..1, gọi videoPlayer.setVolume(v, v)
public void setBackgroundBaseVolume(float vol)  // 0..1, cập nhật config, áp dụng ngay
public void setDuckAmountPercent(float percent) // 0..100, cập nhật config, gain ramp tự nhiên
public float getCurrentBackgroundGain()          // gain envelope hiện tại (0..1)
public long getCurrentPositionMs()
public long getDurationMs()
public void setBackgroundUri(Uri bgUri)          // đổi nhạc nền giữ vị trí và trạng thái phát
```

Vòng lặp cập nhật giao diện: `Handler` main thread, mỗi 100ms đọc vị trí, cập nhật `txtTime`, `PlayheadView`, `seekPosition` (nếu không đang kéo), dòng trạng thái. Dừng vòng lặp trong `onStop`, chạy lại trong `onStart`.

Định dạng (đặt trong class Java thuần `Formatters`, có unit test):

- Thời gian: `mm:ss` (ví dụ `00:14 / 01:00`); video dài từ 1 giờ trở lên thì `h:mm:ss`.
- dB của mức ducking: `amount = 0` → `0 dB`; `amount = 100` → `−∞ dB`; còn lại `−` + `|20·log10(1 − amount/100)|` làm tròn 1 chữ số thập phân. Dùng ký tự trừ `−` (U+2212). Ví dụ 70 → `−10.5 dB`, 50 → `−6.0 dB`.

## 8. Danh sách file cần tạo/sửa

1. `res/layout/activity_main.xml`
2. `res/layout/view_control_button.xml` (layout dùng chung cho 4 nút) và `view_background_lane.xml`
3. `res/menu/menu_player.xml`
4. `res/values/colors.xml`, `dimens.xml`, `strings.xml`, `themes.xml`
5. `res/drawable/`: icon vector (play, pause, video, music, swap, plus, duck, power, more), nền nút và selector trạng thái, nền làn nhạc nền (thường và nét đứt)
6. Java: `MainActivity`, `MainViewModel`, `UiState`, `ScreenState`, `PanelType`, `SegmentLaneView`, `PlayheadView`, `BackgroundLaneButton` (nếu tách class), `ControlButton` (nếu tách class), `Formatters`
7. Phần sửa `DuckingSession` theo mục 7
8. Unit test JUnit: `FormattersTest`, `UiStateTest` (logic mở/đóng panel)

## 9. Test bắt buộc

### 9.1 `Formatters`

| Input | Output |
|---|---|
| `formatTime(14_000)` | `00:14` |
| `formatTime(60_000)` | `01:00` |
| `formatTime(3_725_000)` | `1:02:05` |
| `formatDuckDb(0)` | `0 dB` |
| `formatDuckDb(70)` | `−10.5 dB` |
| `formatDuckDb(50)` | `−6.0 dB` |
| `formatDuckDb(100)` | `−∞ dB` |

### 9.2 Logic panel (`UiState.togglePanel(PanelType)`)

- `NONE` → bấm `VIDEO` → `VIDEO`.
- `VIDEO` → bấm `VIDEO` → `NONE`.
- `VIDEO` → bấm `DUCK_AMOUNT` → `DUCK_AMOUNT`.

### 9.3 `SegmentLaneView` (tách phép tính toạ độ ra hàm thuần để test)

- `durationMs = 60_000`, `width = 300px`, segment `[12_000–21_000]` → x = 60px, rộng 45px.
- Segment `[59_000–70_000]` bị clamp → x = 295px, rộng 5px.
- Segment dài 10ms → rộng tối thiểu 2dp.

## 10. Tiêu chí hoàn thành

- Màn hình khớp mô tả ở mục 3–4: một hàng chọn file đã được bỏ, nhạc nền nằm trong timeline, các nút điều khiển ở đáy, panel seekbar chỉ hiện khi bấm nút.
- Bấm nút điều chỉnh: panel hiện ra mượt, kéo slider thay đổi âm lượng ngay khi đang phát; bấm lại thì ẩn.
- Playhead chạy đúng theo video; bấm vào làn hoặc kéo seekbar đều seek được.
- Bấm làn nhạc nền mở trình chọn audio; đổi nhạc không làm mất vị trí phát.
- Xoay màn hình không mất trạng thái và không phát lại từ đầu.
- Mọi phần tử bấm được có vùng chạm tối thiểu 44dp và có `contentDescription`/`stateDescription` phù hợp.
- Test ở mục 9 và test của hai tài liệu trước đều pass.

## 11. Yêu cầu đầu ra

1. Toàn bộ file ở mục 8, code Java có Javadoc ngắn cho các class public.
2. Không thay đổi hành vi ducking đã mô tả ở hai tài liệu trước, ngoài các API bổ sung ở mục 7.
3. Liệt kê ngắn các chỗ đã làm khác tài liệu này và lý do.
