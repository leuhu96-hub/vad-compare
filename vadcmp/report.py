"""CSV tables + PNG charts + a single self-contained report.html."""
from __future__ import annotations

import base64
import html
import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vadcmp.config import model_postprocess, resolve  # noqa: E402
from vadcmp.groundtruth import gt_grid, parse_groundtruth  # noqa: E402
from vadcmp.metrics import roc_points  # noqa: E402
from vadcmp.runner import load_cached  # noqa: E402
from vadcmp.timeline import binarize, postprocess  # noqa: E402

# Categorical slots in fixed order (validated palette); colour follows the model, by config order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
GT_FILL = "#e3e1db"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2.0,
})


def colors_for(order):
    return {m: SERIES[i % len(SERIES)] for i, m in enumerate(order)}


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _save(fig, out_dir, name) -> str:
    fig.savefig(os.path.join(out_dir, name), dpi=150, bbox_inches="tight")
    return _png(fig)


# ----------------------------------------------------------------------------- charts
def chart_roc(bins, order, col, out_dir):
    pts = roc_points(bins, order)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED)
    for m in order:
        if m in pts:
            fpr, tpr, auc = pts[m]
            ax.plot(fpr, tpr, color=col[m], label=f"{m}  (AUC {auc:.4f})")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate", ylabel="True positive rate",
           title="ROC - tất cả file (bin 0.5 s)")
    ax.legend(loc="lower right")
    return _save(fig, out_dir, "roc_overall.png")


def chart_roc_by_category(bins, order, col, out_dir):
    cats = list(dict.fromkeys(bins["category"]))
    if len(cats) <= 1:
        return None
    cats = cats[:12]
    nc = min(3, len(cats))
    nr = int(np.ceil(len(cats) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.6 * nc, 3.2 * nr), squeeze=False, sharex=True, sharey=True)
    for ax, c in zip(axes.flat, cats):
        pts = roc_points(bins, order, c)
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED)
        for m in order:
            if m in pts:
                ax.plot(pts[m][0], pts[m][1], color=col[m], lw=1.6)
        ax.set_title(c, fontsize=10)
        ax.set(xlim=(0, 1), ylim=(0, 1.01))
    for ax in list(axes.flat)[len(cats):]:
        ax.set_visible(False)
    handles = [plt.Line2D([], [], color=col[m], lw=2) for m in order]
    fig.legend(handles, order, loc="lower center", ncol=min(len(order), 6), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("ROC theo category", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return _save(fig, out_dir, "roc_by_category.png")


def _hbar(ax, order, vals, col, fmt, title, xlabel, log=False):
    y = np.arange(len(order))[::-1]
    v = np.array([vals.get(m, np.nan) for m in order], dtype=float)
    ax.barh(y, np.nan_to_num(v, nan=0), height=0.62, color=[col[m] for m in order],
            edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(y, order)
    ax.grid(axis="y", visible=False)
    if log:
        ax.set_xscale("log")
    finite = v[np.isfinite(v) & (v > 0)] if log else v[np.isfinite(v)]
    if len(finite):
        hi = finite.max()
        ax.set_xlim(right=hi * (3 if log else 1.18))
    for yi, vi in zip(y, v):
        txt = "n/a" if not np.isfinite(vi) else fmt(vi)
        xpos = vi if np.isfinite(vi) and vi > 0 else (ax.get_xlim()[0] * 1.2 if log else 0)
        ax.text(xpos * (1.08 if log else 1) + (0 if log else 0.01 * ax.get_xlim()[1]), yi, txt,
                va="center", ha="left", color=INK, fontsize=9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)


def _unit_axis(ax, vals):
    """Metric in [0, 1]: start a bit below the worst model, keep ticks <= 1, room for labels."""
    lo = max(0.0, float(np.floor((np.nanmin(vals) - 0.05) * 10) / 10))
    ax.set_xlim(lo, 1.0 + (1.0 - lo) * 0.16)
    ticks = [t for t in np.linspace(lo, 1.0, 6) if t <= 1.0 + 1e-9]
    ax.set_xticks(ticks, [f"{t:.2f}".rstrip("0").rstrip(".") if t else "0" for t in ticks])


def chart_summary(overall, speed_sum, order, col, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.55 * len(order) + 1.6))
    o = overall.set_index("model")
    _hbar(axes[0], order, o["auc"].to_dict(), col, lambda v: f"{v:.4f}", "AUC", "cao hơn = tốt hơn")
    _unit_axis(axes[0], o["auc"])
    _hbar(axes[1], order, o["f1"].to_dict(), col, lambda v: f"{v:.4f}",
          "F1 (sau post-processing)", "cao hơn = tốt hơn")
    _unit_axis(axes[1], o["f1"])
    s = speed_sum.set_index("model")
    _hbar(axes[2], order, s["rtf"].to_dict(), col, lambda v: f"{v:.4f}", "Real-time factor",
          "thời gian xử lý / độ dài audio (log, thấp hơn = nhanh hơn)", log=True)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.tight_layout()
    return _save(fig, out_dir, "summary.png")


def chart_errors(overall, order, out_dir):
    o = overall.set_index("model").loc[order]
    parts = [("err_fn_true", "Bỏ sót speech (FN thật)", "#3d3c39"),
             ("err_fp_true", "Báo nhầm speech (FP thật)", "#8a8984"),
             ("err_boundary", "Lỗi ở biên chuyển tiếp", "#cfcdc7")]
    tot = o["n_bins"].astype(float)
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(order) + 1.4))
    y = np.arange(len(order))[::-1]
    left = np.zeros(len(order))
    for key, lab, c in parts:
        v = 100 * o[key].to_numpy(float) / tot.to_numpy()
        ax.barh(y, v, left=left, height=0.6, color=c, edgecolor=SURFACE, linewidth=2, label=lab)
        left += v
    for yi, l in zip(y, left):
        ax.text(l + 0.2, yi, f"{l:.1f}%", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, order)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=max(left.max() * 1.15, 1))
    ax.set_xlabel("% số bin 0.5 s bị sai")
    ax.set_title("Phân loại lỗi: lỗi thật vs lỗi ở biên")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    fig.tight_layout()
    return _save(fig, out_dir, "errors.png")


def chart_miss_far(overall, order, col, out_dir):
    """Operating point of every model: FAR (x) vs miss rate (y). Filled = after post-processing,
    hollow = raw threshold; lower-left is better."""
    o = overall.set_index("model")
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    xs, ys = [], []
    for m in order:
        r = o.loc[m]
        x0, y0, x1, y1 = 100 * r["far_raw"], 100 * r["miss_rate_raw"], 100 * r["far"], 100 * r["miss_rate"]
        if np.isfinite([x0, y0, x1, y1]).all():
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1, shrinkA=6, shrinkB=7))
        ax.scatter([x0], [y0], s=70, facecolor=SURFACE, edgecolor=col[m], linewidth=2, zorder=3)
        ax.scatter([x1], [y1], s=80, color=col[m], edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.annotate(m, (x1, y1), xytext=(7, 5), textcoords="offset points", fontsize=9, color=INK)
        xs += [x0, x1]
        ys += [y0, y1]
    xs, ys = np.array(xs, float), np.array(ys, float)
    ax.set_xlim(0, max(1.0, np.nanmax(xs) * 1.2))
    ax.set_ylim(0, max(1.0, np.nanmax(ys) * 1.2))
    ax.set_xlabel("FAR - % bin non-speech báo nhầm là speech")
    ax.set_ylabel("Miss rate - % bin speech bị bỏ sót")
    ax.set_title("Miss rate vs FAR (góc dưới trái = tốt hơn)")
    handles = [plt.Line2D([], [], ls="", marker="o", ms=8, color=col[m]) for m in order]
    handles += [plt.Line2D([], [], ls="", marker="o", ms=8, mfc=SURFACE, mec=INK2, mew=1.5),
                plt.Line2D([], [], ls="", marker="o", ms=8, color=INK2)]
    ax.legend(handles, list(order) + ["raw threshold", "sau post-processing"], loc="upper left",
              bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return _save(fig, out_dir, "miss_far.png")


def chart_timeline(cfg, item, order, mcfgs, col, out_dir, title_extra=""):
    bin_size = float(cfg["data"]["gt_bin"])
    gt = parse_groundtruth(item["gt"], bin_size)
    fig, axes = plt.subplots(len(order), 1, figsize=(11, 0.95 * len(order) + 0.9), sharex=True, squeeze=False)
    axes = axes[:, 0]
    dur = None
    for ax, m in zip(axes, order):
        mc = mcfgs[m]
        r = load_cached(cfg, mc, item)
        if r is None:
            continue
        fs, timing = r
        dur = timing["duration"]
        edges, labels = gt_grid(gt, bin_size, dur)
        for a, b, lab in zip(edges[:-1], edges[1:], labels):
            if lab == 1:
                ax.axvspan(a, b, color=GT_FILL, lw=0)
        fs_s = fs.smoothed(float(mc.get("smooth_s", 0.0)))
        thr = float(mc["threshold"])
        ax.step(np.r_[fs_s.start, fs_s.end[-1:]], np.r_[fs_s.prob, fs_s.prob[-1:]], where="post",
                color=col[m], lw=1.3)
        ax.axhline(thr, color=MUTED, lw=0.8, ls="--")
        seg = binarize(fs_s, thr)
        pp = model_postprocess(cfg, mc)
        if pp:
            seg = postprocess(seg, dur, pp["pad_before"], pp["pad_after"], pp["merge_gap"],
                              pp["min_speech"], pp.get("order", ("pad", "merge", "drop")))
        for s, e in seg:
            ax.plot([s, e], [-0.12, -0.12], color=INK, lw=3, solid_capstyle="butt")
        ax.set_ylim(-0.22, 1.05)
        ax.set_yticks([0, 1])
        ax.grid(axis="x", visible=False)
        ax.text(0.004, 0.96, m, transform=ax.transAxes, va="top", fontsize=9, color=INK, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none", alpha=0.85))
    axes[-1].set_xlabel("giây  (nền xám = GT speech, vạch đen = segment sau post-processing, nét đứt = threshold)")
    if dur:
        axes[-1].set_xlim(0, dur)
    fig.suptitle(f"{os.path.basename(item['audio'])}  [{item['category']}]{title_extra}", x=0.01, ha="left",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.join(out_dir, "timelines"), exist_ok=True)
    return _save(fig, out_dir, os.path.join("timelines", f"{item['id']}.png"))


# ----------------------------------------------------------------------------- html
def _fmt(v, nd=4):
    if isinstance(v, (float, np.floating)):
        return "n/a" if not np.isfinite(v) else f"{v:.{nd}f}"
    return html.escape(str(v))


def _table(df, cols, labels, best=None, nd=None):
    """best: {col: 'max'|'min'} -> bold best value per column."""
    nd = nd or {}
    best = best or {}
    marks = {}
    for c, how in best.items():
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            marks[c] = s.max() if how == "max" else s.min()
    h = ["<table><thead><tr>"] + [f"<th>{html.escape(l)}</th>" for l in labels] + ["</tr></thead><tbody>"]
    for _, r in df.iterrows():
        h.append("<tr>")
        for c in cols:
            v = r[c]
            cell = _fmt(v, nd.get(c, 4))
            if c in marks and isinstance(v, (float, int, np.floating, np.integer)) and np.isfinite(v) \
                    and abs(float(v) - float(marks[c])) < 1e-12:
                cell = f"<b>{cell}</b>"
            cls = ' class="num"' if isinstance(v, (float, int, np.floating, np.integer)) else ""
            h.append(f"<td{cls}>{cell}</td>")
        h.append("</tr>")
    h.append("</tbody></table>")
    return "".join(h)


def _pivot_table(bycat, metric, order, best="max"):
    p = bycat.pivot(index="category", columns="model", values=metric)
    p = p[[m for m in order if m in p.columns]]
    h = ["<table><thead><tr><th>category</th>"] + [f"<th>{html.escape(m)}</th>" for m in p.columns] + ["</tr></thead><tbody>"]
    for cat, row in p.iterrows():
        mx = row.max() if best == "max" else row.min()
        h.append(f"<tr><td>{html.escape(str(cat))}</td>")
        for v in row:
            cell = _fmt(v)
            if np.isfinite(v) and abs(v - mx) < 1e-12:
                cell = f"<b>{cell}</b>"
            h.append(f'<td class="num">{cell}</td>')
        h.append("</tr>")
    h.append("</tbody></table>")
    return "".join(h)


CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e6e5e1;--soft:#f3f2ef}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--ink);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 16px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 10px;padding-top:8px;border-top:1px solid var(--line)}
.sub{color:var(--ink2);margin:0 0 18px}.note{color:var(--ink2);font-size:13px;max-width:900px}
.wrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;margin:8px 0 6px;font-size:13px;white-space:nowrap}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--ink2);font-weight:600;background:var(--soft)}td.num{text-align:right;font-variant-numeric:tabular-nums}
img{max-width:100%;height:auto;display:block;margin:8px 0}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:2px 16px;font-size:13px;color:var(--ink2)}
code{background:var(--soft);padding:1px 5px;border-radius:4px}
details{margin:6px 0}summary{cursor:pointer;color:var(--ink2)}
"""


def write_report(cfg, bins, overall, bycat, perfile, speed_sum, order, items, log=print):
    out_dir = resolve(cfg, cfg["output_dir"])
    os.makedirs(out_dir, exist_ok=True)
    overall.to_csv(os.path.join(out_dir, "metrics_overall.csv"), index=False)
    bycat.to_csv(os.path.join(out_dir, "metrics_by_category.csv"), index=False)
    perfile.to_csv(os.path.join(out_dir, "metrics_per_file.csv"), index=False)
    speed_sum.to_csv(os.path.join(out_dir, "speed.csv"), index=False)

    col = colors_for(order)
    mcfgs = {m["name"]: m for m in cfg["models"]}
    img_roc = chart_roc(bins, order, col, out_dir)
    img_rocc = chart_roc_by_category(bins, order, col, out_dir)
    img_sum = chart_summary(overall, speed_sum, order, col, out_dir)
    img_err = chart_errors(overall, order, out_dir)
    img_mf = chart_miss_far(overall, order, col, out_dir)

    # timelines: worst files by mean F1 across models (or first N)
    n_tl = int(cfg["report"].get("timelines", 12))
    by_id = {it["id"]: it for it in items}
    pf = perfile.groupby("id")["f1"].mean().fillna(0)
    ids = list(pf.sort_values().index) if cfg["report"].get("timeline_sort", "worst") == "worst" else list(pf.index)
    tl_imgs = []
    for i in ids[:n_tl]:
        try:
            tl_imgs.append((i, chart_timeline(cfg, by_id[i], order, mcfgs, col, out_dir,
                                              f"   mean F1 = {pf[i]:.3f}")))
        except Exception as e:
            log(f"[report] timeline {i} failed: {e}")

    pp = cfg["postprocess"]
    ov = overall.copy()
    ov = ov.sort_values("auc", ascending=False)
    sec = []
    sec.append("<h2>Tổng quan</h2>")
    sec.append(f'<img alt="AUC, F1 and real-time factor per model" src="data:image/png;base64,{img_sum}">')
    sec.append('<div class="wrap">' + _table(
        ov, ["model", "n_files", "auc", "ap", "f1", "f1_excl_boundary", "f1_raw", "precision", "recall",
             "specificity", "miss_rate", "far", "youden_thr", "f1_at_youden", "rtf", "x_realtime"],
        ["model", "files", "AUC", "AP", "F1 (pp)", "F1 bỏ biên", "F1 raw", "Precision", "Recall",
         "Specificity", "Miss rate", "FAR", "Youden thr", "F1 @Youden", "RTF", "× real-time"],
        best={"auc": "max", "ap": "max", "f1": "max", "f1_excl_boundary": "max", "f1_raw": "max",
              "precision": "max", "recall": "max", "specificity": "max", "f1_at_youden": "max",
              "miss_rate": "min", "far": "min",
              "rtf": "min", "x_realtime": "max"},
        nd={"n_files": 0, "rtf": 5, "x_realtime": 1}) + "</div>")
    sec.append(
        '<p class="note"><b>AUC/AP</b>: tính trên score trung bình mỗi bin 0.5 s (không phụ thuộc threshold). '
        '<b>F1 (pp)</b>: threshold → segment → post-processing chung → bin (≥ '
        f'{cfg["scoring"]["min_overlap"]:.0%} bin là speech). <b>F1 raw</b>: score bin ≥ threshold, không post-processing. '
        '<b>F1 bỏ biên</b>: loại các bin sát điểm chuyển speech/non-speech. '
        '<b>Miss rate</b> = FN / (TP + FN): tỉ lệ bin speech bị bỏ sót. '
        '<b>FAR</b> (false alarm rate) = FP / (FP + TN): tỉ lệ bin non-speech bị báo nhầm là speech. '
        '<b>Youden thr</b>: threshold tối ưu tìm trên chính tập này (lạc quan, chỉ để tham khảo).</p>')
    sec.append("<h2>Miss rate &amp; FAR</h2>")
    sec.append(f'<img alt="Miss rate versus false alarm rate per model" src="data:image/png;base64,{img_mf}">')
    sec.append('<div class="wrap">' + _table(
        ov, ["model", "miss_rate", "far", "miss_rate_raw", "far_raw", "miss_rate_excl_boundary",
             "far_excl_boundary", "youden_thr", "miss_rate_at_youden", "far_at_youden"],
        ["model", "Miss (pp)", "FAR (pp)", "Miss (raw)", "FAR (raw)", "Miss bỏ biên", "FAR bỏ biên",
         "Youden thr", "Miss @Youden", "FAR @Youden"],
        best={k: "min" for k in ("miss_rate", "far", "miss_rate_raw", "far_raw", "miss_rate_excl_boundary",
                                 "far_excl_boundary", "miss_rate_at_youden", "far_at_youden")}) + "</div>")
    sec.append('<p class="note">pp = sau post-processing chung (pad/merge kéo dài speech → miss giảm, FAR tăng); '
               'raw = score bin ≥ threshold của model, không post-processing. Thấp hơn = tốt hơn.</p>')
    sec.append("<h2>ROC</h2>")
    sec.append(f'<img alt="ROC curves" src="data:image/png;base64,{img_roc}">')
    if img_rocc:
        sec.append(f'<img alt="ROC curves per category" src="data:image/png;base64,{img_rocc}">')
    if bycat["category"].nunique() > 1:
        sec.append("<h2>Theo category</h2><h3>AUC</h3><div class='wrap'>" + _pivot_table(bycat, "auc", order) + "</div>")
        sec.append("<h3>F1 (sau post-processing)</h3><div class='wrap'>" + _pivot_table(bycat, "f1", order) + "</div>")
        sec.append("<h3>F1 bỏ biên</h3><div class='wrap'>" + _pivot_table(bycat, "f1_excl_boundary", order) + "</div>")
        sec.append("<h3>Miss rate (sau post-processing, thấp hơn = tốt hơn)</h3><div class='wrap'>"
                   + _pivot_table(bycat, "miss_rate", order, best="min") + "</div>")
        sec.append("<h3>FAR (sau post-processing, thấp hơn = tốt hơn)</h3><div class='wrap'>"
                   + _pivot_table(bycat, "far", order, best="min") + "</div>")
    sec.append("<h2>Phân loại lỗi</h2>")
    sec.append(f'<img alt="Error breakdown per model" src="data:image/png;base64,{img_err}">')
    sec.append('<div class="wrap">' + _table(
        overall, ["model", "n_bins", "err_fn_true", "err_fp_true", "err_boundary", "tp", "fp", "tn", "fn"],
        ["model", "bins", "FN thật", "FP thật", "lỗi biên", "TP", "FP", "TN", "FN"],
        best={"err_fn_true": "min", "err_fp_true": "min", "err_boundary": "min"},
        nd={k: 0 for k in ("n_bins", "err_fn_true", "err_fp_true", "err_boundary", "tp", "fp", "tn", "fn")}) + "</div>")
    sec.append("<h2>Tốc độ</h2>")
    sec.append('<div class="wrap">' + _table(
        speed_sum, ["model", "audio_s", "rtf", "rtf_median", "rtf_p95", "cpu_rtf", "x_realtime", "load_s"],
        ["model", "audio (s)", "RTF", "RTF median", "RTF p95", "CPU-time / audio", "× real-time", "load (s)"],
        best={"rtf": "min", "rtf_median": "min", "x_realtime": "max"},
        nd={"audio_s": 1, "rtf": 5, "rtf_median": 5, "rtf_p95": 5, "cpu_rtf": 5, "x_realtime": 1, "load_s": 2}) + "</div>")
    sec.append(f'<p class="note">Đo trên máy chạy script, {cfg["num_threads"]} thread/model, không tính thời gian decode. '
               'RTF trên desktop chỉ để so sánh tương đối giữa các model, không thay cho đo trên điện thoại.</p>')
    if tl_imgs:
        sec.append(f"<h2>Timeline ({len(tl_imgs)} file F1 thấp nhất)</h2>")
        for i, b64 in tl_imgs:
            sec.append(f'<img alt="timeline {html.escape(i)}" src="data:image/png;base64,{b64}" loading="lazy">')
    worst = perfile.sort_values("f1").head(40)
    sec.append("<h2>Chi tiết từng file</h2><details><summary>40 cặp (model, file) F1 thấp nhất — đầy đủ trong "
               "<code>metrics_per_file.csv</code></summary><div class='wrap'>" + _table(
                   worst, ["model", "id", "category", "f1", "precision", "recall", "miss_rate", "far", "auc",
                           "err_boundary",
                           "err_fp_true", "err_fn_true"],
                   ["model", "file", "category", "F1", "P", "R", "Miss", "FAR", "AUC", "lỗi biên", "FP thật", "FN thật"],
                   nd={"err_boundary": 0, "err_fp_true": 0, "err_fn_true": 0}) + "</div></details>")

    cfg_rows = "".join(
        f"<div>{html.escape(m['name'])}</div><div><code>{html.escape(m['type'])}</code> "
        f"threshold={_fmt(float(m['threshold']), 4)}"
        f"{' · post-processing: ' + html.escape(str(m['postprocess'])) if 'postprocess' in m else ''}</div>"
        for m in cfg["models"] if m["name"] in order)
    head = (f"<h1>So sánh VAD</h1><p class='sub'>{len(items)} file · {bins.groupby('model').size().iloc[0]} bin "
            f"{cfg['data']['gt_bin']} s · {len(order)} model</p>"
            f"<div class='kv'><div>Post-processing chung</div><div>pad trước {pp['pad_before']} s, pad sau "
            f"{pp['pad_after']} s, merge gap &lt; {pp['merge_gap']} s, bỏ segment &lt; {pp['min_speech']} s "
            f"(thứ tự: {' → '.join(pp.get('order', []))})</div>"
            f"<div>Boundary tolerance</div><div>{cfg['scoring']['boundary_tolerance']} bin mỗi phía</div>{cfg_rows}</div>")
    page = (f"<!doctype html><html lang='vi'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>VAD Comparison</title><style>{CSS}</style>"
            f"</head><body><main>{head}{''.join(sec)}</main></body></html>")
    path = os.path.join(out_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path
