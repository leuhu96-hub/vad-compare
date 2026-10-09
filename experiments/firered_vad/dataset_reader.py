"""Đọc dataset audio + ground truth, ghép theo tên file. Một file duy nhất, chỉ dùng thư viện
chuẩn của Python -> copy sang project khác là dùng được.

Cấu trúc:
    <audio_root>/<quality>/<filename>.mp4|mp3|m4a|wav|...       vd. phase_1/high/abc_01.mp4
    <gt_root>/<category>/ground_truth/<filename>.txt            vd. Groundtruth/Asm/ground_truth/abc_01.txt

- GT chỉ đọc trong thư mục `ground_truth/`. Các folder khác trong <category> (kể cả có file trùng
  tên) bị bỏ qua.
- Audio ghép với GT theo tên file (bỏ đuôi, không phân biệt hoa thường), không phụ thuộc thư mục.
  quality lấy từ audio, category lấy từ GT.
- Cùng một filename ở nhiều <quality> -> mỗi bản audio là 1 item, dùng chung 1 GT.
- Cùng một filename ở nhiều <category> -> không biết chọn GT nào -> bỏ qua và báo "ambiguous".

File GT: mỗi dòng `start end label [cột 4+ bị bỏ qua]`, vd.
    0 0.5 Speech
    0.5 1 Non-Speech 1

Dùng trong code:
    from dataset_reader import load_dataset, read_ground_truth, gt_to_bins
    ds = load_dataset("phase_1", "Groundtruth")
    print(ds.report())
    for item in ds.items:            # item.audio, item.gt, item.name, item.quality, item.category
        segs = read_ground_truth(item.gt)        # [(start, end, 1|0), ...]

Dùng từ dòng lệnh (kiểm tra dữ liệu, xuất manifest):
    python dataset_reader.py --audio-root phase_1 --gt-root Groundtruth --csv manifest.csv
"""
from __future__ import annotations

import csv
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field

AUDIO_EXTS = (".mp4", ".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg", ".opus", ".webm",
              ".mkv", ".mov", ".amr", ".3gp", ".wma", ".aiff", ".aif", ".caf")
GT_EXTS = (".txt",)
GT_DIR = "ground_truth"

SPEECH_LABELS = {"speech", "sp", "s", "1", "true", "voice"}
NONSPEECH_LABELS = {"non-speech", "nonspeech", "non_speech", "ns", "n", "0", "false", "silence", "noise"}


# ============================================================================ data classes
@dataclass(frozen=True)
class Item:
    name: str          # tên file không đuôi (giữ nguyên hoa/thường của file audio)
    audio: str         # đường dẫn audio
    gt: str            # đường dẫn ground truth
    quality: str       # thư mục ngay dưới audio_root ("" nếu file nằm thẳng trong root)
    category: str      # thư mục ngay dưới gt_root


@dataclass
class Dataset:
    audio_root: str
    gt_root: str
    items: list = field(default_factory=list)            # list[Item]
    audio_without_gt: list = field(default_factory=list)  # audio không có GT cùng tên
    gt_without_audio: list = field(default_factory=list)  # GT không có audio cùng tên
    ambiguous: dict = field(default_factory=dict)         # key -> [GT paths] (trùng tên ở nhiều category)
    ignored_gt: dict = field(default_factory=dict)        # folder (tương đối) -> số file .txt ngoài ground_truth/
    n_audio: int = 0
    n_gt: int = 0

    def by(self, attr: str) -> dict:
        """Nhóm item theo 'quality' hoặc 'category'."""
        out = defaultdict(list)
        for it in self.items:
            out[getattr(it, attr)].append(it)
        return dict(sorted(out.items()))

    def report(self) -> str:
        lines = [f"Audio: {self.n_audio} file trong {self.audio_root}",
                 f"GT   : {self.n_gt} file trong */{GT_DIR}/ của {self.gt_root}"]
        if self.ignored_gt:
            n = sum(self.ignored_gt.values())
            ex = ", ".join(sorted(self.ignored_gt)[:4]) + (" ..." if len(self.ignored_gt) > 4 else "")
            lines.append(f"       bỏ qua {n} file .txt ở {len(self.ignored_gt)} folder khác ({ex})")
        lines.append(f"Ghép : {len(self.items)} item")
        for attr in ("quality", "category"):
            g = self.by(attr)
            lines.append(f"       theo {attr}: " + ", ".join(f"{k or '(root)'}={len(v)}" for k, v in g.items()))
        if self.audio_without_gt:
            lines.append(f"Audio không có GT ({len(self.audio_without_gt)}): "
                         + ", ".join(os.path.relpath(p, self.audio_root) for p in self.audio_without_gt[:5])
                         + (" ..." if len(self.audio_without_gt) > 5 else ""))
        if self.gt_without_audio:
            lines.append(f"GT không có audio ({len(self.gt_without_audio)}): "
                         + ", ".join(os.path.relpath(p, self.gt_root) for p in self.gt_without_audio[:5])
                         + (" ..." if len(self.gt_without_audio) > 5 else ""))
        if self.ambiguous:
            lines.append(f"GT trùng tên ở nhiều category, đã bỏ qua ({len(self.ambiguous)}):")
            for k, ps in list(self.ambiguous.items())[:5]:
                lines.append(f"       {k}: " + " | ".join(os.path.relpath(p, self.gt_root) for p in ps))
        return "\n".join(lines)

    def to_csv(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "quality", "category", "audio", "gt"])
            for it in self.items:
                w.writerow([it.name, it.quality, it.category, it.audio, it.gt])


# ============================================================================ scanning
def file_key(path: str) -> str:
    """Khoá để ghép audio <-> GT: tên file bỏ đuôi, chữ thường, bỏ khoảng trắng 2 đầu."""
    return os.path.splitext(os.path.basename(path))[0].strip().lower()


def _walk(root: str):
    for dp, dirs, files in os.walk(root):
        dirs.sort()
        yield dp, sorted(files)


def find_audio(audio_root: str, exts=AUDIO_EXTS):
    """-> list[(path, quality)]. quality = thư mục đầu tiên dưới audio_root."""
    exts = tuple(e.lower() for e in exts)
    out = []
    for dp, files in _walk(audio_root):
        rel = os.path.relpath(dp, audio_root)
        quality = "" if rel == "." else rel.split(os.sep)[0]
        out += [(os.path.join(dp, f), quality) for f in files if f.lower().endswith(exts)]
    return out


def find_ground_truth(gt_root: str, gt_dir: str = GT_DIR, exts=GT_EXTS):
    """-> (list[(path, category)], ignored {folder: n}).
    Chỉ nhận file nằm trong thư mục tên `gt_dir` (không phân biệt hoa thường), ở bất kỳ độ sâu nào
    bên dưới <category>. category = thư mục đầu tiên dưới gt_root."""
    exts = tuple(e.lower() for e in exts)
    want = gt_dir.lower()
    found, ignored = [], defaultdict(int)
    for dp, files in _walk(gt_root):
        rel = os.path.relpath(dp, gt_root)
        parts = [] if rel == "." else rel.split(os.sep)
        hits = [f for f in files if f.lower().endswith(exts)]
        if not hits:
            continue
        if want not in (p.lower() for p in parts):
            ignored[rel] += len(hits)
            continue
        category = parts[0] if parts[0].lower() != want else ""
        found += [(os.path.join(dp, f), category) for f in hits]
    return found, dict(ignored)


def load_dataset(audio_root: str, gt_root: str, gt_dir: str = GT_DIR,
                 audio_exts=AUDIO_EXTS, gt_exts=GT_EXTS, key=file_key) -> Dataset:
    if not os.path.isdir(audio_root):
        raise FileNotFoundError(f"Không thấy thư mục audio: {audio_root}")
    if not os.path.isdir(gt_root):
        raise FileNotFoundError(f"Không thấy thư mục ground truth: {gt_root}")
    audios = find_audio(audio_root, audio_exts)
    gts, ignored = find_ground_truth(gt_root, gt_dir, gt_exts)

    gt_index = defaultdict(list)
    for p, cat in gts:
        gt_index[key(p)].append((p, cat))
    ambiguous = {k: [p for p, _ in v] for k, v in gt_index.items() if len(v) > 1}

    ds = Dataset(audio_root, gt_root, ignored_gt=ignored, ambiguous=ambiguous,
                 n_audio=len(audios), n_gt=len(gts))
    used = set()
    for ap, quality in audios:
        k = key(ap)
        if k in ambiguous:
            continue
        if k not in gt_index:
            ds.audio_without_gt.append(ap)
            continue
        gp, cat = gt_index[k][0]
        used.add(gp)
        ds.items.append(Item(os.path.splitext(os.path.basename(ap))[0], ap, gp, quality, cat))
    ds.gt_without_audio = [p for p, _ in gts if p not in used and key(p) not in ambiguous]
    return ds


# ============================================================================ ground truth
def _label(tok: str):
    t = tok.strip().strip("\"'").lower()
    if t in SPEECH_LABELS:
        return 1
    if t in NONSPEECH_LABELS:
        return 0
    return None


def read_ground_truth(path: str):
    """-> list[(start, end, label)] label 1 = speech, 0 = non-speech, sắp theo start.
    Cột 1 = start, cột 2 = end, cột 3 = label; cột 4+ bị bỏ qua. Phân tách: space/tab/dấu phẩy/;.
    Dòng trống, dòng '#', dòng header bị bỏ qua."""
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for n, line in enumerate(f, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            line = re.sub(r"(?i)\bnon[\s_]+speech\b", "non-speech", line)
            toks = [t for t in re.split(r"[\s,;]+", line) if t]
            if len(toks) < 3:
                continue
            try:
                s, e = float(toks[0]), float(toks[1])
            except ValueError:
                continue                       # header
            lab = _label(toks[2])
            if lab is None:
                raise ValueError(f"{path}:{n}: không hiểu label {toks[2]!r}")
            if e <= s:
                raise ValueError(f"{path}:{n}: end <= start ({line!r})")
            rows.append((s, e, lab))
    return sorted(rows)


def gt_to_bins(segments, bin_s: float = 0.5, duration: float | None = None):
    """Đưa GT (đoạn dài ngắn bất kỳ) về lưới bin cố định.
    -> list[label] với 1/0 theo đa số thời lượng trong bin, -1 nếu bin không có nhãn."""
    if not segments:
        return []
    end = max(e for _, e, _ in segments)
    if duration is not None:
        end = min(end, duration)
    n = max(0, math.ceil(end / bin_s - 1e-9))
    sp, ns = [0.0] * n, [0.0] * n
    for s, e, lab in segments:
        for i in range(int(s // bin_s), min(n, math.ceil(e / bin_s))):
            ov = min(e, (i + 1) * bin_s) - max(s, i * bin_s)
            if ov > 0:
                (sp if lab == 1 else ns)[i] += ov
    return [-1 if sp[i] + ns[i] <= 1e-9 else int(sp[i] > ns[i]) for i in range(n)]


def segments_to_bins(segments, duration: float, bin_s: float = 0.5, min_overlap: float = 0.5):
    """Segment speech [(start, end)] -> list[0/1] mỗi bin; 1 nếu speech phủ >= min_overlap bin."""
    n = max(0, math.ceil(duration / bin_s - 1e-9))
    cov = [0.0] * n
    for s, e in segments:
        for i in range(max(0, int(s // bin_s)), min(n, math.ceil(e / bin_s))):
            cov[i] += max(0.0, min(e, (i + 1) * bin_s) - max(s, i * bin_s))
    return [int(cov[i] >= min_overlap * (min((i + 1) * bin_s, duration) - i * bin_s) - 1e-9) for i in range(n)]


def write_bins(path: str, labels, bin_s: float = 0.5, duration: float | None = None):
    """Ghi lưới bin theo đúng định dạng GT: `0 0.5 Speech` / `0.5 1 Non-Speech`."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, lab in enumerate(labels):
            if lab < 0:
                continue
            e = (i + 1) * bin_s if duration is None else min((i + 1) * bin_s, duration)
            f.write(f"{i * bin_s:g} {round(e, 3):g} {'Speech' if lab == 1 else 'Non-Speech'}\n")


# ============================================================================ CLI
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Kiểm tra / liệt kê dataset audio + ground truth")
    ap.add_argument("--audio-root", required=True, help="vd. phase_1")
    ap.add_argument("--gt-root", required=True, help="vd. Groundtruth")
    ap.add_argument("--gt-dir", default=GT_DIR, help="tên thư mục chứa GT (mặc định ground_truth)")
    ap.add_argument("--csv", help="ghi danh sách item ra file csv")
    a = ap.parse_args()
    d = load_dataset(a.audio_root, a.gt_root, a.gt_dir)
    print(d.report())
    if a.csv:
        d.to_csv(a.csv)
        print(f"-> {a.csv}")
