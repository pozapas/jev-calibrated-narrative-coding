"""F3 -- Probability landscape (§3, §5.2).

Small multiples, one per presence Noul: log-scaled histogram of p on the random stratum,
with the uncertain band [0.3, 0.7] shaded and its share annotated, and a right-hand strip
showing two prevalence estimates that a calibrated model would agree on:

  - the thresholded rate   P(p > 0.5)        (what a hard-label pipeline reports)
  - the expected rate      E[p]              (the calibrated estimate)

The gap between them is the figure's point: probability mass below 0.5 is not nothing, and
whether E[p] is the better prevalence estimate is exactly what the calibration audit decides.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
import style as S

DATA = Path(__file__).resolve().parents[2] / "data"
RUN = "stage2"          # falls back to stage1 if stage2 has not finished


def load() -> tuple[pd.DataFrame, str]:
    f = DATA / RUN / f"{RUN}_flat.parquet"
    if f.exists():
        d = pd.read_parquet(f)
        if "stratum" in d.columns:
            d = d[d["stratum"] == "random"]
        return d, RUN
    d = pd.read_parquet(DATA / "stage1" / "stage1_flat.parquet")
    return d, "stage1"


def main() -> None:
    d, run = load()
    pcols = sorted(c for c in d.columns if c.startswith("p_"))
    n = len(pcols)
    ncol = 4
    nrow = int(np.ceil(n / ncol))

    S.setup()
    fig, axes = plt.subplots(nrow, ncol, figsize=(S.W2, 1.57 * nrow + 0.55),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()
    bins = np.linspace(0, 1, 51)

    for i, c in enumerate(pcols):
        ax = axes[i]
        p = d[c].dropna().to_numpy(dtype=float)
        v = c[2:]
        ax.axvspan(0.3, 0.7, color=S.ORANGE, alpha=0.11, zorder=0, lw=0)
        ax.hist(p, bins=bins, color=S.BLUE, alpha=0.9, edgecolor="none", zorder=2)
        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_title(v, loc="left", fontsize=6.8, pad=2)

        thr = float((p > 0.5).mean())
        exp = float(p.mean())
        band = float(((p >= 0.3) & (p <= 0.7)).mean())
        ax.axvline(0.5, color="#888888", lw=0.5, ls=(0, (2, 2)), zorder=3)

        ax.text(0.5, 0.965, f"band {100*band:.2f}%", transform=ax.transAxes,
                ha="center", va="top", fontsize=5.2, color="#8A5A00")
        ax.text(0.98, 0.80, f"P(p>0.5) {100*thr:.2f}%\nE[p]      {100*exp:.2f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=5.2,
                family="monospace", color="#333333", linespacing=1.5)
        if exp > 0 and thr > 0:
            ax.text(0.98, 0.50, f"ratio {exp/thr:.1f}×", transform=ax.transAxes,
                    ha="right", va="top", fontsize=5.2, color=S.RED)
        if i % ncol == 0:
            ax.set_ylabel("narratives", fontsize=6.5)
        if i >= n - ncol:
            ax.set_xlabel("predicted probability $p$", fontsize=6.5)
        ax.tick_params(labelsize=5.8)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    # A visual legend states the two shared encodings without a figure-level
    # sentence or repeated prose above the small multiples.
    fig.legend(handles=[
                   Patch(facecolor=S.ORANGE, edgecolor="none", alpha=0.18,
                         label=r"uncertain band: $0.30 \leq p \leq 0.70$"),
                   Line2D([], [], color="#888888", lw=0.8, ls=(0, (2, 2)),
                          label=r"decision threshold: $p = 0.50$"),
               ],
               loc="upper center", bbox_to_anchor=(0.5, 0.996), ncol=2,
               fontsize=6.8, handlelength=1.55, handletextpad=0.5,
               columnspacing=1.45, borderaxespad=0.0, frameon=False)
    S.finish(fig)
    fig.tight_layout(pad=0.75, w_pad=1.15, h_pad=1.20, rect=(0, 0, 1, 0.970))
    S.save(fig, "F3_probability_landscape")


if __name__ == "__main__":
    main()
