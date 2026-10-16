"""Báo cáo Excel cho YAMNet (model transfer learning của bạn), cùng bố cục với báo cáo FireRed / Silero
và sheet "KQ du lieu that" (mục A, B, C, D), để ba model được chấm trên đúng cùng một quy trình.

    # chạy model .tflite / .onnx trên audio
    python excel_report.py --audio-root "D:/data/phase_1" --gt-root "D:/data/Groundtruth" \
        --model models/yamnet_vad.tflite -o yamnet_report.xlsx

    # dùng kết quả có sẵn (vd. output của lib.so: mỗi file một .txt `index, start, end, score`)
    python excel_report.py --audio-root ... --gt-root ... --scores-dir "D:/results/lib_so"

    # đúng chia dev/test của file tổng hợp + mục D so với số YAMNet đã có trong file đó
    python excel_report.py --xlsx Tong_hop_danh_gia_VAD_test.xlsx --audio-root ... --gt-root ... --model ...

Ba cấu hình (sau khi gộp cửa sổ thành score mỗi hop 0.08 s: --rescore triangle + --smooth-hops 5):
  Y0  score hop ≥ ngưỡng, không hậu xử lý
  Y1  hậu xử lý hiện tại của lib: pad 100 / 120 ms → drop 0 → merge < 500 ms   (= A4 trong file tổng hợp)
  Y2  hậu xử lý kiểu FireRed (vadcmp/decide.py, method fsm): ngưỡng ra = ngưỡng − 0.1, đoạn thô phải dài hơn
      0.1 s, lặng dài hơn 0.16 s mới cắt, giữ đuôi (hangover), co đuôi 0.08 s

Tham số chung (--rebin, --min-overlap, --dcf-weight, --criterion, --threshold, --dev-ratio …) giống hệt
báo cáo FireRed / Silero: xem experiments/common/vad_excel.py. Ngưỡng cố định mặc định 0.6026.
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, ROOT)
import vad_excel  # noqa: E402
from dataset_reader import file_key  # noqa: E402
from yamnet_scores import SR, YamnetModel, find_score_file, index_score_files, read_window_file, rescore  # noqa: E402

THRESHOLDS = vad_excel.THRESHOLDS


def _audio_16k(path):
    sys.path.insert(0, os.path.join(HERE, "..", "firered_vad"))
    from firered_onnx import step1_load_audio, step2_downmix, step3_resample

    x, sr = step1_load_audio(path)
    return step3_resample(step2_downmix(x), sr, SR)


def _duration(path):
    """Độ dài sau khi giải mã (giống chế độ --model), không dùng metadata của container."""
    sys.path.insert(0, os.path.join(HERE, "..", "firered_vad"))
    from firered_onnx import step1_load_audio

    x, sr = step1_load_audio(path)
    return x.shape[1] / sr


def get_scores(items, args, log=print):
    os.makedirs(args.cache, exist_ok=True)
    if args.scores_dir:
        idx = None if args.scores_pattern else index_score_files(args.scores_dir)
        missing = []
        for it in items:
            p = find_score_file(it, args.scores_dir, args.scores_pattern, idx)
            if not p:
                missing.append(it)
                continue
            s, e, sc = read_window_file(p)
            it["starts"], it["scores"] = s, sc
            it["window_s"] = float(np.median(e - s)) if len(s) else args.hop_s
            it["score_file"] = p
            h = hashlib.md5(os.path.abspath(it["audio"]).encode()).hexdigest()[:8]
            cp = os.path.join(args.cache, f"{file_key(it['audio'])}_dur_{h}.npy")
            if os.path.exists(cp) and not args.force:
                it["dur"] = float(np.load(cp))
            else:
                it["dur"] = _duration(it["audio"])
                np.save(cp, it["dur"])
        if missing:
            log(f"[!] {len(missing)} file không tìm thấy file score, bỏ qua: "
                + ", ".join(it["file"] for it in missing[:10]) + (" …" if len(missing) > 10 else ""))
            for it in missing:
                items.remove(it)
        if not items:
            raise SystemExit("Không file nào có score (kiểm tra --scores-dir / --scores-pattern)")
        args.model_desc = f"score có sẵn: {args.scores_dir}" + (f" ({args.scores_pattern})" if args.scores_pattern else "")
        return
    if not args.model:
        raise SystemExit("Cần --model (file .tflite / .onnx) hoặc --scores-dir (kết quả có sẵn)")
    model = None
    for i, it in enumerate(items, 1):
        key = f"{os.path.abspath(it['audio'])}|{os.path.abspath(args.model)}|{args.hop_s}|{args.speech_index}|" \
              f"{args.output}|{args.input_samples}|{args.tail}"
        cp = os.path.join(args.cache, f"{file_key(it['audio'])}_yamnet_{hashlib.md5(key.encode()).hexdigest()[:8]}.npz")
        if os.path.exists(cp) and not args.force:
            z = np.load(cp)
            it["starts"], it["scores"] = z["starts"], z["scores"]
            it["window_s"], it["dur"] = float(z["window_s"]), float(z["duration"])
            continue
        if model is None:
            model = YamnetModel(args.model, args.hop_s, args.speech_index, args.output, args.output_index,
                                args.input_samples, args.threads, args.tail)
            args.model_desc = model.describe()
            log(f"[model] {args.model_desc}")
        x16 = _audio_16k(it["audio"])
        s, sc = model.window_scores(x16)
        it["starts"], it["scores"], it["window_s"], it["dur"] = s, sc, model.window_s, len(x16) / SR
        np.savez_compressed(cp, starts=s, scores=sc, window_s=model.window_s, duration=it["dur"])
        log(f"[{i}/{len(items)}] {os.path.basename(it['audio'])} ({it['dur']:.1f} s, {len(sc)} cửa sổ)")


def hop_segments(on, hop_s, dur):
    d = np.diff(np.r_[0, on.astype(np.int8), 0])
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(a * hop_s, min(b * hop_s, dur)) for a, b in zip(s, e)]


class YamnetSpec:
    name, short, abbr, sheet = "YAMNet VAD", "YAMNet", "YN", "KQ YAMNet"
    configs = ["Y0", "Y1", "Y2"]
    default_config, default_thr = "Y1", 0.6026
    frame_desc = "hop 0.08 s"
    here, out_name = HERE, "yamnet_report.xlsx"
    doc = __doc__

    @staticmethod
    def add_args(ap):
        g = ap.add_argument_group("nguồn score YAMNet (chọn một)")
        g.add_argument("--model", help="file .tflite hoặc .onnx của bạn")
        g.add_argument("--scores-dir", help="thư mục kết quả có sẵn, mỗi file audio một file `index,start,end,score`")
        g.add_argument("--scores-pattern", help="tên file score theo audio, vd. '{Category}_{index}.txt' "
                                                "(mặc định: cùng tên file audio, tìm khắp --scores-dir)")
        m = ap.add_argument_group("model")
        m.add_argument("--hop-s", type=float, default=0.08)
        m.add_argument("--speech-index", type=int, default=1, help="vị trí lớp speech trong output 2 lớp")
        m.add_argument("--output", default="auto", choices=["auto", "probs", "logits"])
        m.add_argument("--output-index", type=int, default=0)
        m.add_argument("--input-samples", type=int, help="ghi đè số mẫu mỗi cửa sổ (mặc định đọc từ model)")
        m.add_argument("--tail", choices=["drop", "pad"], default="drop",
                       help="đoạn cuối không đủ một cửa sổ: drop = bỏ (giống lib), pad = thêm cửa sổ đệm 0")
        r = ap.add_argument_group("gộp cửa sổ -> hop")
        r.add_argument("--rescore", choices=["triangle", "mean", "max"], default="triangle")
        r.add_argument("--smooth-hops", type=int, default=5, help="trung bình trượt N hop sau khi gộp (0/1 = tắt)")
        y1 = ap.add_argument_group("Y1 = hậu xử lý hiện tại của lib")
        y1.add_argument("--y1-pad-before", type=float, default=0.10)
        y1.add_argument("--y1-pad-after", type=float, default=0.12)
        y1.add_argument("--y1-drop", type=float, default=0.0)
        y1.add_argument("--y1-merge", type=float, default=0.5)
        y1.add_argument("--y1-order", default="pad,drop,merge")
        y2 = ap.add_argument_group("Y2 = máy trạng thái kiểu FireRed")
        y2.add_argument("--y2-thr-off", type=float, default=0.1, help="ngưỡng ra = ngưỡng vào − giá trị này (0 = tắt)")
        y2.add_argument("--y2-min-speech", type=float, default=0.1)
        y2.add_argument("--y2-min-silence", type=float, default=0.16)
        y2.add_argument("--y2-hangover", type=int, choices=[0, 1], default=1)
        y2.add_argument("--y2-pad-before", type=float, default=0.0)
        y2.add_argument("--y2-pad-after", type=float, default=-0.08)

    @staticmethod
    def prepare(args):
        args.y1_order = [s.strip() for s in args.y1_order.split(",") if s.strip()]
        if bool(args.model) == bool(args.scores_dir):
            raise SystemExit("Chọn đúng một nguồn: --model <file.tflite|.onnx> hoặc --scores-dir <thư mục kết quả>")

    @staticmethod
    def model_label(args):
        return getattr(args, "model_desc", args.model or args.scores_dir)

    @staticmethod
    def descriptions(args):
        rs = f"gộp cửa sổ {args.rescore}" + (f" + làm mượt {args.smooth_hops} hop" if args.smooth_hops > 1 else "")
        return {
            "Y0": f"Score hop ({rs}) ≥ ngưỡng, không hậu xử lý",
            "Y1": f"Hậu xử lý hiện tại của lib: pad {args.y1_pad_before * 1000:.0f}/{args.y1_pad_after * 1000:.0f} ms, "
                  f"drop {args.y1_drop * 1000:.0f} ms, merge < {args.y1_merge * 1000:.0f} ms "
                  f"({' → '.join(args.y1_order)})",
            "Y2": f"Máy trạng thái kiểu FireRed: ngưỡng ra = ngưỡng − {args.y2_thr_off:g}, đoạn > {args.y2_min_speech:g} s, "
                  f"lặng > {args.y2_min_silence:g} s, hangover {'bật' if args.y2_hangover else 'tắt'}, "
                  f"pad {args.y2_pad_before:+g}/{args.y2_pad_after:+g} s",
        }

    @staticmethod
    def pp_note(args):
        return (f"Score hop: gộp cửa sổ {args.rescore}, làm mượt {args.smooth_hops} hop, hop {args.hop_s:g} s, "
                f"cuối file {args.tail}. Y1 = pad {args.y1_pad_before * 1000:.0f}/{args.y1_pad_after * 1000:.0f} ms, "
                f"drop {args.y1_drop * 1000:.0f} ms, merge < {args.y1_merge * 1000:.0f} ms.")

    @staticmethod
    def load_probs(items, args):
        get_scores(items, args)

    @staticmethod
    def decide(it, args, thr, n):
        from vadcmp.decide import decide_fsm
        from vadcmp.timeline import FrameScores

        dur, hop = it["dur"], args.hop_s
        if "hops" not in it:
            it["hops"] = rescore(it["starts"], it["window_s"], it["scores"], hop, dur, args.rescore, args.smooth_hops)
        hops = it["hops"]
        fs = FrameScores.from_hops(hops, hop, duration=dur)
        bins = lambda segs: vad_excel.seg_bins(segs, n, dur, args.bin, args.rebin, args.min_overlap)  # noqa: E731
        D0, D1, D2 = [], [], []
        for t in thr:
            raw = hop_segments(hops >= t, hop, dur)
            D0.append(bins(raw))
            D1.append(bins(vad_excel.rules_segs(raw, dur, args.y1_pad_before, args.y1_pad_after, args.y1_drop,
                                                args.y1_merge, args.y1_order)))
            pp = {"method": "fsm", "thr_off": (float(t) - args.y2_thr_off) if args.y2_thr_off > 0 else None,
                  "min_speech": args.y2_min_speech, "min_silence": args.y2_min_silence,
                  "hangover": bool(args.y2_hangover), "pad_before": args.y2_pad_before,
                  "pad_after": args.y2_pad_after}
            seg = decide_fsm(fs, dur, float(t), pp)
            D2.append(bins(vad_excel.merge_segs([tuple(x) for x in np.asarray(seg).reshape(-1, 2)])))
        return {"Y0": np.stack(D0), "Y1": np.stack(D1), "Y2": np.stack(D2)}


SPEC = YamnetSpec()


def main(argv=None):
    return vad_excel.main(SPEC, argv)


if __name__ == "__main__":
    main()
