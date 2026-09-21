"""Figure C.2 -- precision among flagged records against the share of flagged records
auto-accepted, measured on the human reference set rather than on coded fields.

This is the human-referenced companion of F5(b). Nothing is fitted: for each variable the
flagged records (p > 0.5) are ranked by probability and the Horvitz--Thompson weighted
precision of the top share is traced, using the same `precision_coverage` function that
produces the review budgets reported in the text. The pooled curve is drawn over all
variables together. The 0.90 and 0.95 precision targets are marked.

Input:  data/gold/gold_labels.csv
Output: outputs/figures/FC2_gold_precision.pdf / .png
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1]))
import style as S
import s07_metrics as M

DATA = Path(__file__).resolve().parents[2] / "data"
MIN_N = 40


def main() -> None:
    S.setup()
    g = pd.read_csv(DATA / "gold" / "gold_labels.csv")
    g = g[g.y.notna() & g.p.notna()]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(S.W2, 3.25))
    for ax in (ax1, ax2):
        ax.axhspan(90, 95, color=S.ORANGE, alpha=0.12, lw=0)
        ax.axhline(90, color=S.ORANGE, ls="--", lw=0.8)
        ax.axhline(95, color=S.ORANGE, ls=":", lw=0.8)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 102)
        ax.set_xlabel("share of flagged records auto-accepted")
        ax.set_ylabel("precision among accepted records (%)")
    # (a) per variable, thin lines
    n_trajectories = 0
    for v, sub in g.groupby("variable"):
        if len(sub) < MIN_N:
            continue
        cov, fdr, _ = M.precision_coverage(sub.p.to_numpy(float), sub.y.to_numpy(float),
                                           sub.pair_weight.to_numpy(float))
        if cov.size == 0:
            continue
        ax1.step(cov, 100 * (1 - fdr), where="pre", lw=0.9, color=S.BLUE, alpha=0.55)
        n_trajectories += 1
    ax1.set_title("(a) variable-level trajectories", loc="left", fontweight="bold")
    # (b) pooled
    cov, fdr, _ = M.precision_coverage(g.p.to_numpy(float), g.y.to_numpy(float),
                                       g.pair_weight.to_numpy(float))
    pooled_precision = 100 * (1 - fdr)
    ax2.step(cov, pooled_precision, where="pre", lw=1.8, color=S.BLUE, zorder=4)
    ax2.scatter([cov[0], cov[-1]], [pooled_precision[0], pooled_precision[-1]],
                s=25, color=S.BLUE, edgecolor="white", linewidth=0.75, zorder=5)
    ax2.set_title("(b) pooled precision trajectory", loc="left", fontweight="bold")
    # Use the unused lower ranges as local visual keys.  This keeps the figure
    # self-explanatory without making a separate legend row above the panels.
    ax1.legend(handles=[
                   Line2D([], [], color=S.BLUE, lw=1.0, alpha=0.55,
                          label=f"{n_trajectories} variable trajectories\n(one step curve per variable)"),
               ], loc="lower left", bbox_to_anchor=(0.015, 0.035),
               fontsize=6.0, handlelength=1.8, handletextpad=0.55,
               borderaxespad=0.0, frameon=False)
    ax2.legend(handles=[
                   Line2D([], [], color=S.BLUE, lw=1.8, marker="o", markeredgecolor="white",
                          markeredgewidth=0.7, markersize=4.3, label="pooled trajectory"),
                   Patch(facecolor=S.ORANGE, edgecolor="none", alpha=0.12, label="90–95% policy band"),
                   Line2D([], [], color=S.ORANGE, lw=0.9, ls="--", label="90% minimum"),
                   Line2D([], [], color=S.ORANGE, lw=0.9, ls=":", label="95% preferred"),
               ], loc="lower left", bbox_to_anchor=(0.015, 0.035), ncol=2,
               fontsize=6.0, handlelength=1.55, handletextpad=0.42,
               columnspacing=0.9, labelspacing=0.55, borderaxespad=0.0, frameon=False)
    S.finish(fig, grid="y")
    fig.tight_layout(w_pad=2.0)
    S.save(fig, "FC2_gold_precision")


if __name__ == "__main__":
    main()
