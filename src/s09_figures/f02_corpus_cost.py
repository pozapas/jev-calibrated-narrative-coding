"""F2 -- Corpus and cost (§3).

(a) corpus pulse by year with the empty-narrative share
(b) narrative length distribution (log x) with a token axis
(c) casing composition as a direct-label dumbbell timeline
(d) measured cost per 1,000 narratives vs number of questions, with the fitted token model
    and a generative-LLM reference band

Panel (d) carries the paper's cost finding: input tokens = alpha + beta*k + b*nchar, so
cost scales with schema size, not narrative length.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1]))
import style as S
import pricing as P

DATA = Path(__file__).resolve().parents[2] / "data"


def main() -> None:
    stats = json.loads((DATA / "corpus_stats.json").read_text())
    cost = json.loads((DATA / "cost_model.json").read_text())
    lengths = np.load(DATA / "_length_reservoir.npy")

    S.setup()
    # A deliberate grid, not a default 2x2.  The added vertical room gives the
    # two top panels the same title baseline, even though (b) has a token axis.
    fig = plt.figure(figsize=(S.W2, 5.75))
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.970, bottom=0.095, top=0.820,
                          wspace=0.40, hspace=0.50)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    c, d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # ---------------------------------------------------------------- (a) per year
    yrs = sorted(k for k in stats["year_kept"] if k.isdigit())
    kept = [stats["year_kept"][y] for y in yrs]
    empty = [stats["year_empty"].get(y, 0) for y in yrs]
    tot = [stats["year_total"].get(y, 0) for y in yrs]
    x = np.arange(len(yrs))
    kept_k = np.array(kept) / 1e3
    total_k = np.array(tot) / 1e3
    # The corpus pulse keeps the annual volume readable without turning the
    # time series into a second bar chart.
    a.fill_between(x, kept_k, 0, color=S.BLUE, alpha=0.10, zorder=1)
    a.vlines(x, 0, kept_k, color=S.BLUE, alpha=0.22, lw=2.4, zorder=2)
    a.plot(x, kept_k, color=S.BLUE, lw=2.25, zorder=3)
    a.scatter(x, kept_k, s=42, color=S.BLUE, edgecolor="white", linewidth=1.15,
              zorder=4)
    a.scatter(x, total_k, s=34, facecolor="white", edgecolor=S.MUTED,
              linewidth=0.9, zorder=3)
    a.set_xticks(x); a.set_xticklabels([y[-2:] for y in yrs])
    a.set_ylabel("narratives (thousands)")
    a.set_xlabel("year")
    a.set_title("(a) corpus pulse", loc="left", fontweight="bold", y=1.22)
    a.set_ylim(0, max(total_k) * 1.14)
    a2 = a.twinx()
    share = 100 * np.array(empty) / np.maximum(np.array(tot), 1)
    a2.fill_between(x, share, 0, color=S.ORANGE, alpha=0.08, zorder=0)
    a2.plot(x, share, "-o", color=S.ORANGE, lw=1.55, ms=4.1, zorder=5)
    a2.set_ylabel("empty narrative (%)", color=S.ORANGE)
    a2.tick_params(axis="y", colors=S.ORANGE)
    a2.set_ylim(0, max(share.max() * 1.32, 1))
    a2.spines["right"].set_visible(True)
    a2.spines["right"].set_color(S.ORANGE)
    # The header uses the intentional gap below the panel title.  It keeps the
    # key corpus total and the orange finding out of the plotted data.
    a.text(0.00, 1.085, f"{stats['n_kept_gt_40']/1e6:.2f} M usable narratives",
           transform=a.transAxes, fontsize=6.6, va="bottom", color=S.INK,
           fontweight="bold")
    a2.text(1.00, 1.085, f"Empty share: {share[0]:.1f}% → {share[-1]:.1f}%",
            transform=a2.transAxes, fontsize=6.4, va="bottom", ha="right",
            color=S.ORANGE, fontweight="bold")
    a.legend(handles=[
                 Line2D([], [], color=S.BLUE, lw=2.0, marker="o", ms=4.2,
                        markeredgecolor="white", label="usable"),
                 Line2D([], [], color=S.MUTED, lw=0, marker="o", ms=4.2,
                        markerfacecolor="white", label="all records"),
                 Line2D([], [], color=S.ORANGE, lw=1.4, marker="o", ms=3.6,
                        label="empty share"),
             ],
             loc="lower left", bbox_to_anchor=(0.00, 0.990), ncol=3,
             fontsize=5.7, handlelength=1.25, columnspacing=0.85,
             handletextpad=0.35, borderaxespad=0.0)
    a.annotate("all crash records", xy=(x[-1], total_k[-1]), xytext=(x[-1] - 2.1, 638),
               fontsize=6.1, color=S.MUTED, ha="left",
               arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=0.65))

    # ---------------------------------------------------------------- (b) length
    lg = lengths[lengths > 0]
    bins = np.logspace(np.log10(41), np.log10(max(lg.max(), 100)), 60)
    b.hist(lg, bins=bins, color=S.BLUE, alpha=0.85, edgecolor="none")
    b.set_xscale("log")
    b.set_xlabel("narrative length (characters, log scale)")
    b.set_ylabel("narratives in sample")
    b.set_title("(b) narrative length", loc="left", fontweight="bold", y=1.22)
    q = stats["length_quantiles_chars"]
    med = q["0.5"]
    b.axvline(med, color=S.ORANGE, lw=0.9, ls=(0, (3, 2)))
    b.text(med * 1.26, b.get_ylim()[1] * 0.92, f"Median {med:,.0f} characters", fontsize=6.2,
           color=S.ORANGE, va="top")
    b.text(0.97, 0.72,
           f"95th percentile: {q['0.95']:,.0f} characters\n"
           f"99th percentile: {q['0.99']:,.0f} characters",
           transform=b.transAxes, fontsize=6.0, ha="right", va="top", color="#444444",
           linespacing=1.45)
    bt = b.secondary_xaxis("top", functions=(lambda v: v * cost["per_question_set"]["27"]["b_tokens_per_char"],
                                             lambda v: v / cost["per_question_set"]["27"]["b_tokens_per_char"]))
    bt.set_xlabel("Input tokens contributed by the narrative", fontsize=6.5, labelpad=7)
    bt.tick_params(labelsize=6)

    # ---------------------------------------------------------------- (c) casing
    cb = stats["casing_by_year"]
    up = np.array([100 * cb.get(y, {}).get("upper", 0)
                   / max(sum(cb.get(y, {}).values()), 1) for y in yrs])
    mixed = 100 - up
    # A linked pair per year encodes composition as a continuous allocation,
    # with direct labels instead of a detached legend or stacked bars.
    c.vlines(x, up, mixed, color="#CCD6DC", lw=4.2, zorder=1, capstyle="round")
    c.plot(x, up, color=S.GREY, lw=1.15, alpha=0.75, zorder=2)
    c.plot(x, mixed, color=S.SKY, lw=1.15, alpha=0.90, zorder=2)
    c.scatter(x, up, s=54, color=S.GREY, edgecolor="white", linewidth=0.9, zorder=3)
    c.scatter(x, mixed, s=54, color=S.SKY, edgecolor="white", linewidth=0.9, zorder=3)
    c.set_xticks(x); c.set_xticklabels([y[-2:] for y in yrs])
    c.set_ylabel("share of narratives (%)")
    c.set_xlabel("year")
    c.set_ylim(0, 100)
    c.set_title("(c) casing composition", loc="left", fontweight="bold", y=1.07)
    c.axhline(50, color="#BFC8CE", lw=0.6, ls=(0, (2, 2)), zorder=0)
    c.text(0.02, 0.995, "Dashed guide: equal share (50%)", transform=c.transAxes,
           fontsize=5.8, color=S.MUTED, va="bottom", ha="left",
           bbox=dict(facecolor="white", edgecolor="none", pad=0.45, alpha=0.95))
    c.annotate("mixed case", xy=(x[-1], mixed[-1]), xytext=(x[-1] - 1.85, 78),
               color=S.SKY, fontsize=6.6, fontweight="normal",
               arrowprops=dict(arrowstyle="-", color=S.SKY, lw=0.75))
    c.annotate("all caps", xy=(x[-1], up[-1]), xytext=(x[-1] - 1.85, 21),
               color=S.GREY, fontsize=6.6, fontweight="normal",
               arrowprops=dict(arrowstyle="-", color=S.GREY, lw=0.75))

    # ---------------------------------------------------------------- (d) cost
    ks = sorted(int(k) for k in cost["per_question_set"])
    meas = [cost["per_question_set"][str(k)]["cost_per_1k_narratives_usd"] for k in ks]
    d.scatter(ks, meas, s=42, color=S.BLUE, edgecolor="white", linewidth=1.15,
              zorder=5, label="Jev, measured")
    im = cost["intercept_model"]
    bch = cost["per_question_set"]["27"]["b_tokens_per_char"]
    kk = np.linspace(1, 30, 100)
    fit_tok = im["alpha_tokens"] + im["beta_tokens_per_question"] * kk + bch * med
    d.plot(kk, P.JEV_INPUT_PER_M * fit_tok * 1000 / 1e6, "-", color=S.BLUE, lw=1.55,
           label=f"fit: {im['alpha_tokens']:.0f} + {im['beta_tokens_per_question']:.0f}k "
                 f"+ {bch:.3f}·chars")
    llm_lo = P.llm_cost_per_1k(im["alpha_tokens"] + im["beta_tokens_per_question"] * 1 + bch * med)
    llm_hi = P.llm_cost_per_1k(im["alpha_tokens"] + im["beta_tokens_per_question"] * 30 + bch * med)
    d.axhspan(llm_lo, llm_hi, color=S.RED, alpha=0.12, zorder=0)
    d.text(29.5, llm_hi * 0.85, f"{P.LLM_BASELINE}\n(same input, JSON out)", fontsize=5.8,
           color=S.RED, ha="right", va="top", linespacing=1.4)
    d.set_yscale("log")
    d.set_xlabel("questions per call")
    d.set_ylabel("USD per 1,000 narratives")
    d.set_title("(d) cost scales with schema, not narrative", loc="left", fontweight="bold",
                y=1.07)
    d.set_xlim(0, 31)
    d.legend(loc="lower right", fontsize=5.8)
    # the same measured token count every other cost in the paper uses, not a literal
    r = P.llm_cost_per_1k(P.run_input_tokens()) / P.jev_cost_per_1k(P.run_input_tokens())
    d.annotate(f"{r:.0f}×", xy=(27, meas[-1]), xytext=(21, llm_lo * 0.35),
               fontsize=7, color="#444444",
               arrowprops=dict(arrowstyle="<->", lw=0.6, color="#888888"))

    S.finish(fig)
    # Secondary axes should not add duplicate reference grids to the data plate.
    a2.grid(False)
    bt.grid(False)
    S.save(fig, "F2_corpus_cost")


if __name__ == "__main__":
    main()
