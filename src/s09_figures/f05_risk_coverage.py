"""F5 -- Selective prediction and the human-review budget (§5.4).

(a) accuracy-based risk-coverage -- and why it is uninformative here. With base rates of
    0.5-3.6%, full-coverage accuracy risk already sits below both the 1% and 5% targets, so
    H(r) = 0 for every variable. The panel shows the targets sitting ABOVE the curves.
(b) precision-coverage over the flagged positives: the operational curve.
(c) review budget at target precision 0.90 and 0.95, as a share of flagged positives.
(d) absolute narratives per year Texas must review, at target precision 0.90.

The caveat travels with the figure: precision is measured against a NOISY coded field, so a
"false positive" may be a narrative reporting something the coded field never recorded --
this paper's premise. The human gold set (v2) replaces the reference; the machinery is
identical.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1]))
import style as S
import s07_metrics as M
from s10_analysis import CODED_MAP

DATA = Path(__file__).resolve().parents[2] / "data"
HIGHLIGHT = ["alcohol_involved", "fatigue", "wrong_way"]
HCOLOR = {"alcohol_involved": S.BLUE, "fatigue": S.GREEN, "wrong_way": S.PURPLE}
TARGETS = (0.90, 0.95)


def load():
    f = DATA / "stage2" / "stage2_flat.parquet"
    if not f.exists():
        f = DATA / "stage1" / "stage1_flat.parquet"
    d = pd.read_parquet(f)
    if "stratum" in d.columns:
        d = d[d["stratum"] == "random"]
    d = d.merge(pd.read_csv(DATA / "coded_join.csv"), on="Crash_ID", how="inner")
    return d, json.loads((DATA / "corpus_stats.json").read_text())


def main() -> None:
    d, stats = load()
    per_year = stats["n_kept_gt_40"] / 9.0

    series = {}
    for v, col in CODED_MAP.items():
        if f"p_{v}" not in d.columns or col not in d.columns:
            continue
        sub = d[[f"p_{v}", col]].dropna()
        series[v] = (sub[f"p_{v}"].to_numpy(dtype=float),
                     sub[col].astype(bool).to_numpy().astype(float))

    S.setup()
    # Use the same deliberate four-panel plate as Figure 2.  The header is
    # reserved for the shared variable key rather than a prose caveat.
    fig = plt.figure(figsize=(S.W2, 5.45))
    gs = fig.add_gridspec(2, 2, left=0.105, right=0.975, bottom=0.095, top=0.885,
                          wspace=0.42, hspace=0.54)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    c, e = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # ------------------------------------------------- (a) target-screen ranking
    # A full curve is not useful here: the result is that every full-coverage
    # risk is below the conventional targets.  A ranked target screen states
    # that finding directly and leaves no empty plotting space.
    risk_rows = []
    for v, (p, y) in series.items():
        cor = ((p > 0.5).astype(float) == y).astype(float)
        cov, risk, _ = M.risk_coverage(M.noul_confidence(p), cor)
        risk_rows.append((v, 100 * risk[-1]))
    risk_rows.sort(key=lambda row: row[1], reverse=True)
    risk_names = [row[0] for row in risk_rows]
    risk_values = np.array([row[1] for row in risk_rows])
    risk_y = np.arange(len(risk_rows))
    a.hlines(risk_y, 0, risk_values, color="#CBD5DA", lw=1.7, zorder=1)
    a.scatter(risk_values, risk_y, s=48,
              color=[HCOLOR.get(v, S.GREY) for v in risk_names],
              edgecolor="white", linewidth=0.95, zorder=3)
    for target, label, ls in ((1, "1% target", (0, (3, 1.8))),
                              (5, "5% target", (0, (1, 1.6)))):
        a.axvline(target, color=S.ORANGE, lw=1.0, ls=ls, zorder=2)
        # Keep these clear of the x-axis and of the final y-category.
        a.text(target + 0.08, len(risk_rows) - 2.0, label, fontsize=5.9,
               color=S.ORANGE, rotation=90, va="bottom", ha="left")
    a.set_yticks(risk_y); a.set_yticklabels(risk_names, fontsize=5.9)
    a.invert_yaxis()
    a.set_xlabel("Accuracy risk at full coverage (%)")
    a.set_title("(a) accuracy-risk target screen", loc="left", fontweight="bold", y=1.08)
    a.set_xlim(0, 6.1)
    n_over_1 = int((risk_values > 1.0).sum())
    a.text(0.02, 1.005,
           f"All variables are below 5%; {n_over_1} exceed the stricter 1% target.",
           transform=a.transAxes, fontsize=5.9, color=S.MUTED, va="bottom")

    # ------------------------------------------------- (b) precision trajectories
    # The previous version showed all nine series.  Most of these grey curves
    # carried no readable decision information.  This focused trajectory view
    # keeps the three paper variables and directly labels the curve endpoints.
    b.axhspan(90, 95, color=S.ORANGE, alpha=0.11, zorder=0)
    b.axhline(90, color=S.ORANGE, lw=0.9, ls=(0, (3, 1.8)), zorder=2)
    b.axhline(95, color=S.ORANGE, lw=0.9, ls=(0, (1, 1.6)), zorder=2)
    for v in HIGHLIGHT:
        p, y = series[v]
        cov, fdr, _ = M.precision_coverage(p, y)
        if cov.size == 0:
            continue
        precision = 100 * (1 - fdr)
        color = HCOLOR[v]
        b.plot(cov, precision, lw=2.15, color=color, zorder=4)
        b.scatter([cov[0], cov[-1]], [precision[0], precision[-1]], s=28,
                  color=color, edgecolor="white", linewidth=0.8, zorder=5)
    b.text(0.98, 99.35, "90–95% policy band", fontsize=5.8,
           color="#9A6700", ha="right", va="top")
    b.set_xlabel("Share of flagged positives auto-accepted")
    b.set_ylabel("Precision vs coded field (%)")
    b.set_title("(b) precision trajectories", loc="left", fontweight="bold",
                y=1.08)
    b.set_xlim(0, 1); b.set_ylim(20, 101)

    # ------------------------------------------------- (c) policy frontier
    rows = []
    for v, (p, y) in series.items():
        r90 = M.positive_review_budget(p, y, TARGETS[0])
        r95 = M.positive_review_budget(p, y, TARGETS[1])
        rows.append((v, r90, r95))
    # Highest annual workload first.  This order is shared by (c) and (d).
    rows.sort(key=lambda t: t[1]["review_share_of_corpus"], reverse=True)
    names = [r[0] for r in rows]
    yy = np.arange(len(rows))
    # Plot the complement of review burden: the fraction that can proceed
    # without manual review.  This creates a compact policy frontier rather
    # than a value table, and makes the single viable 90% path visible.
    accept90 = 100 * (1 - np.array([r[1]["review_budget_of_flagged"] for r in rows]))
    accept95 = 100 * (1 - np.array([r[2]["review_budget_of_flagged"] for r in rows]))
    c.axvline(0, color="#AAB7C2", lw=1.05, zorder=1)
    for i, (x90, x95, name) in enumerate(zip(accept90, accept95, names)):
        is_viable = max(x90, x95) > 1.0
        c.plot([x95, x90], [i + 0.13, i - 0.13], color="#C9D2D8" if not is_viable else S.SKY,
               lw=1.2 if not is_viable else 2.0, zorder=2)
        c.scatter(x90, i - 0.13, s=36 if is_viable else 22, color=S.BLUE,
                  edgecolor="white", linewidth=0.8, zorder=4)
        c.scatter(x95, i + 0.13, s=31 if is_viable else 19, marker="D", color=S.ORANGE,
                  edgecolor="white", linewidth=0.75, zorder=4)
        if is_viable:
            c.text(x90 + 2.2, i - 0.13, f"{x90:.0f}% at 90% target", fontsize=6.1,
                   color=S.BLUE, va="center", ha="left", fontweight="bold")
    c.set_yticks(yy); c.set_yticklabels(names, fontsize=6.0)
    c.invert_yaxis()
    c.set_xlabel("Flagged positives eligible for auto-acceptance (%)")
    c.set_title("(c) policy-compliant automation frontier", loc="left", fontweight="bold", y=1.08)
    c.set_xlim(-2, 112)
    c.text(4.0, 2.4, "Eight variables have no policy-compliant\nauto-acceptance at either target.",
           fontsize=5.7, color=S.MUTED, ha="left", va="center", linespacing=1.22)
    c.legend(handles=[
                 Line2D([], [], marker="o", color="none", markerfacecolor=S.BLUE,
                        markeredgecolor="white", markersize=5.5, label="90% precision target"),
                 Line2D([], [], marker="D", color="none", markerfacecolor=S.ORANGE,
                        markeredgecolor="white", markersize=4.9, label="95% precision target"),
             ], loc="upper right", fontsize=5.5, frameon=False, handletextpad=0.35,
             borderaxespad=0.3)

    # ------------------------------------------------- (d) absolute load
    flagged = np.array([r[1]["flagged_share_of_corpus"] for r in rows]) * per_year
    review = np.array([r[1]["review_share_of_corpus"] for r in rows]) * per_year
    e.barh(yy, flagged / 1e3, color="#D9D9E2", height=0.66, label="flagged by the model")
    e.barh(yy, review / 1e3, color=S.ORANGE, height=0.66, label=f"to review at {TARGETS[0]:.0%}")
    e.set_yticks(yy); e.set_yticklabels(names, fontsize=6.0)
    e.invert_yaxis()
    e.set_xlabel("narratives per year in Texas (thousands)")
    e.set_title("(d) annual review load", loc="left", fontweight="bold", y=1.08)
    e.legend(loc="lower right", fontsize=6.0, frameon=False)
    for i, (f_, r_) in enumerate(zip(flagged, review)):
        e.text(f_ / 1e3 + max(flagged) / 1e3 * 0.02, i, f"{r_/1e3:,.1f}k of {f_/1e3:,.1f}k",
               va="center", fontsize=5.0, color="#444444")
    e.set_xlim(0, max(flagged) / 1e3 * 1.42)
    e.text(1.0, 1.015, f"Texas: {per_year:,.0f} narratives/year",
           transform=e.transAxes, fontsize=5.6, ha="right", va="bottom", color="#777777")

    # One visual key replaces the duplicate legends in the top panels.
    fig.legend(handles=[
                   Line2D([], [], color=S.BLUE, lw=2.0, label="alcohol involved"),
                   Line2D([], [], color=S.GREEN, lw=2.0, label="fatigue"),
                   Line2D([], [], color=S.PURPLE, lw=2.0, label="wrong way"),
               ],
               loc="upper center", bbox_to_anchor=(0.5, 0.996), ncol=3,
               fontsize=6.8, handlelength=1.65, handletextpad=0.5,
               columnspacing=1.45, borderaxespad=0.0, frameon=False)
    S.finish(fig)
    c.grid(False)
    c.grid(which="minor", color="white", lw=2.2)
    S.save(fig, "F5_risk_coverage")


if __name__ == "__main__":
    main()
