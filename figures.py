"""Step 3: Fig. 2 and Fig. 3 of the poster.

    python figures.py --data data --marks results/marks --out results/figures

fig2_a.png, fig2_b.png, fig2_c.png  detector marks on short windows (B baseline method, X XQRS with fix,
                                     NK NeuroKit), 0.5-40 Hz signal
fig3.png                             beat-shape groups of recording 1652786, baseline method vs the vote
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402

from evaluate import vote_at_extremum  # noqa: E402
from rpeak.data import load_records, review_copy  # noqa: E402
from rpeak.detectors import bandpass  # noqa: E402

FS = 200
RED, PURPLE, BLUE, AQUA, GRID, INK, MUTED = "#e53935", "#7b1fa2", "#2a78d6", "#1baf7a", "#e3e3e3", "#222222", "#555555"
PANELS = {  # name: (record, first sample, length, show NeuroKit)
    "a": ("1704502.531", 121850, 800, False),
    "b": ("1640106.330", 28550, 800, False),
    "c": ("1716089.465", 19480, 800, True),
}
GROUP_RECORD = "1652786.026"
minus = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−"))


def style(ax, fs=11):
    ax.set_facecolor("white"); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=fs, length=0)


def flag(ax, x, y, tag, col, side, dy, dx, ms=17, fs=10):
    """A line from the mark to a circle with the method's letter."""
    cx, cy = x + (-dx if side == "left" else dx), y + dy
    ax.plot([x, cx], [y, cy], color=col, lw=1.8, zorder=4)
    ax.plot(cx, cy, "o", ms=ms, mfc=col, mec="white", mew=1.2, zorder=5)
    ax.text(cx, cy, tag, ha="center", va="center", fontsize=fs - 1.5 if len(tag) > 1 else fs,
            color="white", zorder=6, fontweight="bold")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--marks", type=Path, default=Path("results/marks"))
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    recs = load_records(args.data)
    need = {r for r, *_ in PANELS.values()} | {GROUP_RECORD}
    x = {k: bandpass(review_copy(recs[k]["sig"]), FS, 0.5, 40.0) for k in need if k in recs}
    marks = {k: json.loads((args.marks / f"{k}.json").read_text(encoding="utf-8")) for k in x}

    # ---------------------------------------------------------------- Fig. 2
    for name, (rec, a, n, show_nk) in PANELS.items():
        if rec not in x:
            continue
        sig, b = x[rec], a + n
        seg = sig[a:b]; lo, hi = seg.min(), seg.max(); span = hi - lo
        fig, ax = plt.subplots(figsize=(679 / 96, 205 / 96))
        style(ax)
        ax.plot(np.arange(a, b), seg, color=RED, lw=1.3, zorder=3)
        dy, dx = 0.2 * span, 14
        for p in marks[rec]["baseline"]:
            if a + 12 <= p < b - 12:
                flag(ax, p, sig[p], "B", PURPLE, "left", dy, dx)
        for p in marks[rec]["xqrs+fix"]:
            if a + 12 <= p < b - 12:
                flag(ax, p, sig[p], "X", BLUE, "right", dy, dx)
        if show_nk:
            for p in marks[rec]["neurokit"]:
                if a + 12 <= p < b - 12:
                    flag(ax, p, sig[p], "NK", AQUA, "right", -dy * 1.1, dx)
        ax.set_xlim(a, b); ax.set_ylim(lo - 0.32 * span, hi + 0.32 * span)
        ax.set_ylabel("Voltage, mV", fontsize=11, color=MUTED)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(200))
        ax.yaxis.set_major_formatter(minus)
        fig.tight_layout(pad=0.3)
        fig.savefig(args.out / f"fig2_{name}.png", dpi=300)
        plt.close(fig)

    # ---------------------------------------------------------------- Fig. 3
    if GROUP_RECORD in x:
        sig = x[GROUP_RECORD]; pre, post = int(0.2 * FS), int(0.4 * FS)
        tt = np.arange(-pre, post) * 1000 / FS

        def groups(p):
            p = np.sort(np.asarray(p)); p = p[(p - pre >= 0) & (p + post < len(sig))]
            W = np.stack([sig[q - pre:q + post] for q in p])
            Wn = (W - W.mean(1, keepdims=True)) / (W.std(1, keepdims=True) + 1e-9)
            lab = fcluster(linkage(Wn, "average", metric="correlation"), t=0.2, criterion="distance")
            return W, lab == np.argmax(np.bincount(lab))

        vote = vote_at_extremum(sig, marks[GROUP_RECORD]["vote"])
        fig, axes = plt.subplots(2, 1, figsize=(663 / 96, 430 / 96), sharex=True, sharey=True)
        for ax, (title, p) in zip(axes, [("Baseline method", marks[GROUP_RECORD]["baseline"]),
                                         ("Vote of 4 detectors", vote)]):
            W, is_main = groups(p); style(ax)
            for w in W[is_main]:
                ax.plot(tt, w, color=RED, lw=0.6, alpha=0.2)
            for w in W[~is_main]:
                ax.plot(tt, w, color=PURPLE, lw=0.9, alpha=0.7)
            n_out = int((~is_main).sum())
            ax.axvline(0, color=INK, lw=0.9, ls=":")
            ax.set_title(f"{title}: {len(W)} marks, {n_out} in other groups ({100 * n_out / len(W):.1f} %)",
                         loc="left", fontsize=12.5, color=INK)
            ax.set_ylabel("mV", fontsize=11, color=MUTED)
            ax.yaxis.set_major_formatter(minus); ax.xaxis.set_major_formatter(minus)
        axes[1].set_xlabel("ms from the mark", fontsize=11, color=MUTED)
        handles = [plt.Line2D([], [], color=RED, lw=2, label="main beat-shape group"),
                   plt.Line2D([], [], color=PURPLE, lw=2, label="other groups")]
        fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout(pad=0.3, h_pad=0.6, rect=(0, 0, 1, 0.94))
        fig.savefig(args.out / "fig3.png", dpi=300)
        plt.close(fig)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
