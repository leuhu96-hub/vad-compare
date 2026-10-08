import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vadcmp.groundtruth import boundary_mask, gt_grid, parse_groundtruth  # noqa: E402
from vadcmp.metrics import binary_metrics  # noqa: E402
from vadcmp.timeline import (FrameScores, bin_scores, binarize, postprocess,  # noqa: E402
                             segments_to_bins, window_scores_to_hops)


def test_bin_scores_mean_weighted():
    fs = FrameScores.from_hops([0, 1, 1, 0], 0.25)       # 0..1 s
    edges = np.array([0, 0.5, 1.0])
    assert np.allclose(bin_scores(fs, edges), [0.5, 0.5])
    assert np.allclose(bin_scores(fs, edges, "max"), [1, 1])


def test_binarize_postprocess():
    fs = FrameScores.from_hops([0, 1, 1, 0, 0, 0, 0, 1, 0, 0], 0.08)
    seg = binarize(fs, 0.5)
    assert np.allclose(seg, [[0.08, 0.24], [0.56, 0.64]])
    pp = postprocess(seg, 0.8, 0.1, 0.12, 0.5, 0.32)
    assert len(pp) == 1 and np.isclose(pp[0, 0], 0.0) and np.isclose(pp[0, 1], 0.76)
    # isolated 80 ms blip: padded to 0.30 s < 0.32 s -> dropped
    blip = postprocess(np.array([[2.0, 2.08]]), 5, 0.1, 0.12, 0.5, 0.32)
    assert len(blip) == 0


def test_segments_to_bins():
    edges = np.arange(5) * 0.5
    b = segments_to_bins(np.array([[0.2, 1.3]]), edges, 0.5)
    assert b.tolist() == [1, 1, 1, 0]   # 60%, 100%, 60%, 0%


def test_window_rescore():
    hops = window_scores_to_hops([1.0, 0.0], 0.08, 0.16, 3)
    assert np.allclose(hops, [1.0, 0.5, 0.0])


def test_gt_formats(tmp_path):
    a = tmp_path / "a.txt"          # your format: 4th column must be ignored
    a.write_text("0 0.5 Speech\n0.5 1 Non-Speech 1\n1 1.5 Speech 0\n1.5 2 Speech\n")
    b = tmp_path / "b.txt"
    b.write_text("start,end,label\n0.0,0.5,1\n0.5,1.0,0,1\n1.0,2.0,1\n")
    c = tmp_path / "c.txt"
    c.write_text("speech\nnon speech\nspeech\nspeech\n")
    d = tmp_path / "d.txt"          # longer segments are split into 0.5 s bins
    d.write_text("0 0.5 speech\n0.5 1 Non Speech 7\n1 2 SPEECH\n")
    for p in (a, b, c, d):
        e, lab = gt_grid(parse_groundtruth(str(p)), 0.5)
        assert lab.tolist() == [1, 0, 1, 1], p.name


def test_miss_far():
    lab = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])
    dec = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0])
    r = binary_metrics(lab, dec)
    assert np.isclose(r["miss_rate"], 0.25) and np.isclose(r["far"], 0.2)
    assert np.isclose(r["miss_rate"], 1 - r["recall"]) and np.isclose(r["far"], 1 - r["specificity"])


def test_boundary_and_metrics():
    lab = np.array([0, 0, 1, 1, 1, 0])
    m = boundary_mask(lab, 1)
    assert m.tolist() == [False, True, True, False, True, True]
    r = binary_metrics(lab, np.array([0, 1, 1, 1, 1, 1]), m)
    assert r["err_boundary"] == 2 and r["err_fp_true"] == 0 and r["f1_excl_boundary"] == 1.0


def test_fbank_numpy_matches_knf():
    knf = pytest.importorskip("kaldi_native_fbank")
    from vadcmp import kaldi_fbank as kf

    rng = np.random.default_rng(0)
    x = (rng.standard_normal(16000) * 3000).astype(np.float32)
    ref = kf.fbank(x)
    got = kf.fbank_numpy(x)
    assert ref.shape == got.shape
    assert np.abs(ref - got).max() < 1e-2


def test_scan_reads_only_ground_truth(tmp_path):
    from vadcmp.dataset import scan

    for cat in ("Asm", "Music"):
        for sub, content in (("ground_truth", "0 0.5 Speech\n0.5 1 Non-Speech 1\n"),
                             ("model_v1", "0, 0.0, 0.96, 0.8\n"),
                             ("lib_out/raw", "0, 0.0, 0.96, 0.3\n")):
            d = tmp_path / "Model" / cat / sub
            d.mkdir(parents=True)
            (d / f"{cat.lower()}_1.txt").write_text(content)
        a = tmp_path / "Audio" / cat
        a.mkdir(parents=True)
        (a / f"{cat.lower()}_1.mp4").write_bytes(b"")
    rows, unmatched, info = scan(str(tmp_path / "Audio"), str(tmp_path / "Model"))
    assert not unmatched and len(rows) == 2 and info["n_gt"] == 2
    assert all(os.sep + "ground_truth" + os.sep in r["gt"] for r in rows)
    assert sorted(r["category"] for r in rows) == ["Asm", "Music"]
    assert sum(info["ignored"].values()) == 4
