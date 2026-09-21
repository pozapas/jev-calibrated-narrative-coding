"""F4 -- Calibration audit (hero figure, §5.3).

Reliability is plotted on the model's OWN OUTPUT GRID, not in equal-mass bins.

Why: Jev returns probabilities on a fixed two-decimal grid (99 attainable values, 0.01 to
0.99, never 0 or 1). On a rare variable almost all of the mass lands on two or three grid
points, so equal-mass binning yields 2-6 bins against 72-98 occupied grid values and simply
erases the low-probability region -- which is the region that decides a rare variable's
review budget. Binning by grid value needs no binning parameter and is exact.

Axes are log-log so the rare region is visible. Grid values whose observed rate is exactly
zero cannot be drawn on a log axis and are marked as open triangles on the floor.

Two vertical/horizontal guides carry the structural finding:
  - the GRID FLOOR at p = 0.01, the smallest probability the model can express
  - the coded base rate; when it falls BELOW the grid floor, no assignment of grid values
    can be calibrated for that variable, so E[p] must overestimate its prevalence

Two references are drawn, and the distinction is the point of the figure:

  grey, on the model's output grid -- the CODED CRIS field. Large n, but a noisy reference:
        a narrative that mentions a factor the coded field omits is what this study looks
        for, not necessarily a model error. Agreement, not accuracy.
  blue, on the §4.4 strata -- the HUMAN gold set. Small n, but it is the accuracy reference.
        Where blue sits above grey, the coded field was missing what the narrative said.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1]))
import style as S
import s07_metrics as M
from s10_analysis import CODED_MAP, PROXY_ONLY

DATA = Path(__file__).resolve().parents[2] / "data"
FLOOR = 1.5e-3          # bottom of the log axis; zero-rate grid points sit here
MIN_COUNT = 20          # only draw grid points with enough support to mean anything


GOLD_BINS = [0.0, 0.05, 0.30, 0.70, 0.95, 1.0000001]   # the §4.4 stratification


def load() -> tuple[pd.DataFrame, str]:
    f, run = DATA / "stage2" / "stage2_flat.parquet", "stage2"
    if not f.exists():
        f, run = DATA / "stage1" / "stage1_flat.parquet", "stage1"
    d = pd.read_parquet(f)
    if "stratum" in d.columns:
        d = d[d["stratum"] == "random"]
    d = d.merge(pd.read_csv(DATA / "coded_join.csv"), on="Crash_ID", how="inner")
    print(f"{run}: {len(d):,} rows joined to coded fields")
    return d, run


def load_gold() -> pd.DataFrame | None:
    """Human gold labels, or None before the coding is in."""
    f = DATA / "gold" / "gold_labels.csv"
    if not f.exists():
        return None
    g = pd.read_csv(f)
    g = g[g.y.notna() & g.p.notna()]
    print(f"gold set: {len(g):,} human-labelled judgments over "
          f"{g.Crash_ID.nunique()} narratives")
    return g


def gold_curve(sub: pd.DataFrame):
    """Reliability for the human labels, binned on the §4.4 strata rather than the model's
    output grid.

    The coded-field curve has 150k rows and can afford one point per grid value; the gold
    set has ~150 per variable, so per-grid-value rates would be noise. The frame was
    stratified on exactly these bins, so they are the supported resolution. Rates are
    Horvitz-Thompson weighted and the interval uses the effective sample size, because the
    frame deliberately over-samples the high-probability bins.
    """
    p = sub.p.to_numpy(float); y = sub.y.to_numpy(float); w = sub.pair_weight.to_numpy(float)
    b = np.digitize(p, GOLD_BINS[1:-1], right=False)
    xs, ys, los, his, ns = [], [], [], [], []
    for k in range(len(GOLD_BINS) - 1):
        m = b == k
        if m.sum() < 8:
            continue
        ww = w[m]
        rate = float(np.average(y[m], weights=ww))
        n_eff = float(ww.sum() ** 2 / np.sum(ww ** 2))      # Kish effective n
        lo, hi = M.clopper_pearson(int(round(rate * n_eff)), int(round(n_eff)))
        xs.append(float(np.average(p[m], weights=ww)))
        ys.append(rate); los.append(lo); his.append(hi); ns.append(int(m.sum()))
    return (np.array(xs), np.array(ys), np.array(los), np.array(his), np.array(ns))


def main() -> None:
    d, run = load()
    gold = load_gold()
    items = [(v, c) for v, c in CODED_MAP.items()
             if f"p_{v}" in d.columns and c in d.columns]
    # Nine variables form a balanced 3x3 audit plate.  This makes each
    # reliability curve materially wider than the old 4-column layout.
    n, ncol = len(items), 3
    nrow = int(np.ceil(n / ncol))

    S.setup()
    fig = plt.figure(figsize=(S.W2, 2.35 * nrow + 0.45))
    gs = fig.add_gridspec(nrow * 2, ncol, height_ratios=[3.1, 1.0] * nrow,
                          hspace=0.58, wspace=0.24,
                          left=0.075, right=0.985, top=0.930, bottom=0.075)

    for i, (v, col) in enumerate(items):
        r, c = divmod(i, ncol)
        ax = fig.add_subplot(gs[2 * r, c])
        axh = fig.add_subplot(gs[2 * r + 1, c])

        sub = d[[f"p_{v}", col]].dropna()
        p = sub[f"p_{v}"].to_numpy(dtype=float)
        y = sub[col].astype(bool).to_numpy().astype(float)

        gp, gy, gw = M.discrete_reliability(p, y)
        keep = gw >= MIN_COUNT
        gp, gy, gw = gp[keep], gy[keep], gw[keep]
        lo = np.array([M.clopper_pearson(int(round(a * b)), int(round(b)))[0]
                       for a, b in zip(gy, gw)])
        hi = np.array([M.clopper_pearson(int(round(a * b)), int(round(b)))[1]
                       for a, b in zip(gy, gw)])

        ax.plot([FLOOR, 1], [FLOOR, 1], ls=(0, (2, 2)), lw=0.7, color="#999999", zorder=1)
        base = float(y.mean())
        ax.axhline(base, color=S.RED, lw=0.6, ls=(0, (3, 1.6)), zorder=2)
        ax.axvline(M.GRID_STEP, color=S.ORANGE, lw=0.6, ls=(0, (1, 1.4)), zorder=2)

        pos = gy > 0
        sz = 2 + 9 * np.sqrt(gw / gw.max())
        ax.vlines(gp[pos], np.maximum(lo[pos], FLOOR), hi[pos], color=S.GREY,
                  lw=0.5, alpha=0.55, zorder=3)
        ax.scatter(gp[pos], gy[pos], s=sz[pos], color=S.GREY, zorder=4, lw=0,
                   label="coded field (reference)")
        if (~pos).any():
            ax.scatter(gp[~pos], np.full((~pos).sum(), FLOOR), s=sz[~pos] * 0.7,
                       facecolors="none", edgecolors=S.GREY, lw=0.5, marker="v", zorder=4)
        # human gold set, on the §4.4 strata (see gold_curve for why not the model's grid)
        gold_ece = None
        if gold is not None:
            gs_ = gold[gold.variable == v]
            if len(gs_) >= 40:
                hx, hy, hlo, hhi, hn = gold_curve(gs_)
                gold_ece = M.ece_discrete(gs_.p.to_numpy(float), gs_.y.to_numpy(float),
                                          gs_.pair_weight.to_numpy(float))
                if hx.size:
                    hpos = hy > 0
                    # A bin in which no gold narrative was positive has an observed rate of
                    # zero, which a log axis cannot show. Clamping it to the floor and then
                    # running the line through it would assert a value we did not measure,
                    # so zero bins are drawn as open markers and left out of the line --
                    # the same convention already used for the coded-field points.
                    ax.vlines(hx[hpos], np.maximum(hlo[hpos], FLOOR), hhi[hpos],
                              color=S.BLUE, lw=0.7, alpha=0.65, zorder=5)
                    ax.plot(hx[hpos], hy[hpos], "-o", color=S.BLUE, ms=3.4, lw=1.0,
                            zorder=6, label="human gold set (gold-set strata)")
                    if (~hpos).any():
                        ax.scatter(hx[~hpos], np.full((~hpos).sum(), FLOOR), s=14,
                                   facecolors="none", edgecolors=S.BLUE, lw=0.8,
                                   marker="o", zorder=6)
        else:
            ax.scatter([], [], s=12, color=S.BLUE, label="human gold set (pending)")

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(0.7 * M.GRID_STEP, 1.4); ax.set_ylim(FLOOR * 0.7, 1.4)
        ax.set_title(v, loc="left", fontsize=7.4, pad=4)
        ax.tick_params(labelsize=6.0)
        if c == 0:
            ax.set_ylabel("observed rate", fontsize=6.7)

        rf = M.resolution_floor_report(p, y)
        bd = M.brier_decomposition(p, y)
        sl = M.calibration_slope(p, y)
        # Both ECEs, because the gap between them IS the reference-noise result: the coded
        # field makes the model look better calibrated than the human labels say it is.
        gold_line = f"\nECE|gold {gold_ece:.3f}" if gold_ece is not None else ""
        ax.text(0.035, 0.97,
                f"ECE  {M.ece_discrete(p, y):.3f}\nBrier {bd['brier']:.3f}\n"
                f"slope {sl['slope']:.2f}\nn    {p.size:,}{gold_line}",
                transform=ax.transAxes, fontsize=5.2, va="top", ha="left",
                family="monospace", color="#333333", linespacing=1.5)
        note = []
        if rf["base_rate_below_floor"]:
            note.append("base rate < grid floor")
        if v in PROXY_ONLY:
            note.append("proxy reference")
        if note:
            ax.text(0.97, 0.05, "\n".join(note), transform=ax.transAxes, fontsize=4.8,
                    ha="right", va="bottom", color=S.RED, linespacing=1.4)
        ax.text(0.97, 0.22, f"E[p]/base {rf['expected_p_over_base_rate']:.1f}×",
                transform=ax.transAxes, fontsize=5.1, ha="right", va="bottom",
                color="#555555")

        axh.hist(p, bins=np.logspace(np.log10(0.7 * M.GRID_STEP), 0, 45),
                 color=S.BLUE, alpha=0.85, edgecolor="none")
        axh.set_xscale("log"); axh.set_yscale("log")
        axh.set_xlim(0.7 * M.GRID_STEP, 1.4)
        axh.tick_params(labelsize=5.8)
        # Outer labels retain a clean reading path through the paired panels.
        # Repeating this label below every histogram compresses the next row.
        if r == nrow - 1:
            axh.set_xlabel("predicted $p$", fontsize=6.7)
        if c == 0:
            axh.set_ylabel("count", fontsize=6.4)

    h, l = fig.axes[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color=S.ORANGE, ls=(0, (1, 1.4)), lw=0.9),
          plt.Line2D([], [], color=S.RED, ls=(0, (3, 1.6)), lw=0.9)]
    l += ["grid floor $p=0.01$", "coded base rate"]
    # As in Figure 3, the shared encoding is a visual legend, not a prose title.
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=6.6,
               bbox_to_anchor=(0.5, 0.995), frameon=False, handlelength=1.45,
               handletextpad=0.5, columnspacing=1.25, borderaxespad=0.0)
    S.finish(fig)
    S.save(fig, "F4_calibration")


if __name__ == "__main__":
    main()
