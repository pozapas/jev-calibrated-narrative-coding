"""F9 -- Cost-latency-accuracy frontier (§5.7).

Scatter of methods in cost-per-1,000-narratives (log x) against macro-F1 vs the coded
fields (y), with latency as marker size, plus an inset giving the cost to code all Texas
2017-2025 narratives.

This panel plots the two arms scored on THIS axis: keyword rules (free) and Jev (measured),
both against coded CRIS fields over the same nine variables. The frontier models appear as a
price reference only. Their accuracy was measured (§5.7) but against the human gold set, a
different reference, so plotting it here would invite a comparison the axis does not support.
Their cost axis is real: published list price times the same measured input tokens.
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
import pricing as P
from s10_analysis import CODED_MAP, KW_MAP_ALL as KW_MAP

DATA = Path(__file__).resolve().parents[2] / "data"


def common_variables(d) -> list[str]:
    """Variables that BOTH methods can be scored on -- a macro-F1 over different variable
    sets would not be a comparison."""
    inv = {v: k for k, v in KW_MAP.items()}
    out = []
    for v, col in CODED_MAP.items():
        fam = inv.get(v)
        if (f"p_{v}" in d.columns and col in d.columns
                and fam is not None and f"kw_{fam}" in d.columns):
            out.append(v)
    return out


def macro_f1(d, use_keyword: bool, variables: list[str]) -> tuple[float, int]:
    inv = {v: k for k, v in KW_MAP.items()}
    f1s = []
    for v in variables:
        col, fam = CODED_MAP[v], inv.get(v)
        cols = [col] + ([f"kw_{fam}"] if use_keyword else [f"p_{v}"])
        sub = d[cols].dropna()
        y = sub[col].astype(bool).to_numpy()
        pr = (sub[f"kw_{fam}"].astype(bool).to_numpy() if use_keyword
              else sub[f"p_{v}"].to_numpy(dtype=float) > 0.5)
        tp = float((y & pr).sum())
        prec = tp / max(pr.sum(), 1)
        rec = tp / max(y.sum(), 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-9))
    return float(np.mean(f1s)), len(f1s)


def main() -> None:
    stats = json.loads((DATA / "corpus_stats.json").read_text())
    cost = json.loads((DATA / "cost_model.json").read_text())
    f = DATA / "stage2" / "stage2_flat.parquet"
    run = "stage2"
    if not f.exists():
        f, run = DATA / "stage1" / "stage1_flat.parquet", "stage1"
    d = pd.read_parquet(f)
    if "stratum" in d.columns:
        d = d[d["stratum"] == "random"]
    d = d.merge(pd.read_csv(DATA / "coded_join.csv"), on="Crash_ID", how="inner")
    d = d.merge(pd.read_parquet(DATA / "keyword_flags.parquet"), on="Crash_ID", how="left")
    d = d.merge(pd.read_parquet(DATA / "keyword_flags_extra.parquet"), on="Crash_ID", how="left")

    common = common_variables(d)
    f1_jev, n_jev = macro_f1(d, False, common)
    f1_kw, n_kw = macro_f1(d, True, common)
    assert n_jev == n_kw, "macro-F1 must be computed over the same variables"
    # The run's measured mean, not this stratum's. Every cost in the paper derives from that
    # one number (see pricing.run_input_tokens); pricing the point off the random stratum
    # instead put this panel at $0.1539 per thousand against the $0.1543 the prose quotes,
    # and printed 128x here where Table 2 implies 129x.
    tok = P.run_input_tokens()
    lat_jev = float(d["latency_s"].median())
    c_jev = P.jev_cost_per_1k(tok)
    c_llm = P.llm_cost_per_1k(tok)
    n_corpus = stats["n_kept_gt_40"]

    S.setup()
    fig = plt.figure(figsize=(S.W15, 3.15))
    ax = fig.add_axes([0.10, 0.18, 0.875, 0.69])

    C_KW = 0.0012      # regex over 5 M narratives is about one CPU-hour
    PARETO = "#9BA8B1"
    ax.scatter([C_KW], [f1_kw], s=44, color=S.GREEN, edgecolor="white", linewidth=0.8, zorder=5)
    ax.scatter([c_jev], [f1_jev], s=44 + 260 * lat_jev, color=S.BLUE,
               edgecolor="white", linewidth=0.8, zorder=5)
    ax.axvspan(c_llm * 0.92, c_llm * 1.08, color=S.ORANGE, alpha=0.12, zorder=0)
    ax.axvline(c_llm, color=S.ORANGE, lw=1.0, ls=(0, (3, 2)), zorder=2)

    # Pareto staircase over methods with observed cost and F1.
    pts = sorted([(C_KW, f1_kw), (c_jev, f1_jev)])
    ax.step([p[0] for p in pts] + [c_llm * 1.6],
            [p[1] for p in pts] + [max(f1_kw, f1_jev)], where="post",
            color=PARETO, lw=1.05, ls=(0, (4, 2)), zorder=1)

    ax.set_xscale("log")
    ax.set_xlabel("USD per 1,000 narratives (log scale)")
    ax.set_ylabel(f"Macro-F1 vs coded fields ({n_jev} variables)")
    ax.set_xlim(C_KW * 0.35, c_llm * 4.0)
    ax.set_ylim(max(0, min(f1_kw, f1_jev) - 0.14), max(f1_kw, f1_jev) + 0.23)

    # One compact comparison bracket replaces the former diagonal arrow and
    # multi-line callout in the middle of the data area.
    comparison_y = min(f1_kw, f1_jev) - 0.050
    ax.annotate("", xy=(c_jev, comparison_y), xytext=(C_KW, comparison_y),
                arrowprops=dict(arrowstyle="<->", lw=0.75, color=PARETO))
    ax.text(np.sqrt(C_KW * c_jev), comparison_y - 0.014,
            f"+{100*(f1_jev-f1_kw):.1f} macro-F1\n{c_jev/C_KW:.0f}× cost",
            fontsize=5.8, ha="center", va="top", color=S.MUTED, linespacing=1.35)

    # The frontier F1 exists but lives on the human-label reference, not this axis, so the
    # label says where it is rather than implying the number was never measured.
    ax.text(c_llm * 0.85, max(f1_kw, f1_jev) + 0.025,
            f"{P.LLM_BASELINE.replace('claude-', 'Claude ')}\nlist price; accuracy not plotted",
            fontsize=5.6, color=S.ORANGE, rotation=90, rotation_mode="anchor",
            ha="left", va="bottom", linespacing=1.25)
    ax.text(0.015, 0.965, f"{run} random stratum · n = {len(d):,}",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.2, color=S.MUTED)

    # Whole-corpus inset: shifted left and up so it cannot cover the Pareto or
    # LLM price-reference guides.
    ins = ax.inset_axes([0.40, 0.64, 0.34, 0.28])
    labels = ["keywords", "Jev", P.LLM_BASELINE.replace("claude-", "")]
    vals = [C_KW * n_corpus / 1e3, c_jev * n_corpus / 1e3, c_llm * n_corpus / 1e3]
    ins.barh([0, 1, 2], vals, color=[S.GREEN, S.BLUE, S.ORANGE], height=0.58)
    ins.set_xscale("log")
    ins.set_yticks([0, 1, 2]); ins.set_yticklabels(labels, fontsize=5.0)
    # WP7. The corpus was never coded end to end, so the inset is a projection at the
    # measured per-narrative rate and says so.
    ins.set_title("Projected cost to code all 5.02 M narratives", loc="left", fontsize=5.6,
                  pad=2.4)
    ins.tick_params(labelsize=4.8)
    for i, v in enumerate(vals):
        ins.text(v * 1.22, i, f"${v:,.0f}", va="center", fontsize=4.8, color=S.INK)
    ins.set_xlim(min(vals) * 0.4, max(vals) * 8)
    for sp in ins.spines.values():
        sp.set_linewidth(0.5)

    # Shared visual key: all method and guide meanings are outside the data area.
    fig.legend(handles=[
                   Line2D([], [], marker="o", lw=0, markerfacecolor=S.GREEN, markeredgecolor="white",
                          markersize=5.8, label="keyword rules"),
                   Line2D([], [], marker="o", lw=0, markerfacecolor=S.BLUE, markeredgecolor="white",
                          markersize=5.8, label=f"Jev ({P.JEV_MODEL})"),
                   Line2D([], [], color=PARETO, lw=1.05, ls=(0, (4, 2)), label="observed Pareto frontier"),
                   Line2D([], [], color=S.ORANGE, lw=1.0, ls=(0, (3, 2)), label="LLM price reference"),
                   Line2D([], [], marker="o", lw=0, color=S.INK, markerfacecolor=S.INK,
                          markersize=3.8, label="circle area = p50 latency"),
               ], loc="upper center", bbox_to_anchor=(0.56, 0.945), ncol=5,
               fontsize=5.45, handlelength=1.15, handletextpad=0.35,
               columnspacing=0.75, borderaxespad=0.0, frameon=False)

    S.finish(fig)
    S.save(fig, "F9_frontier")
    print(f"  keyword macro-F1 {f1_kw:.3f} ({n_kw} vars) | Jev macro-F1 {f1_jev:.3f} ({n_jev} vars)")
    print(f"  Jev ${c_jev:.4f}/1k | {P.LLM_BASELINE} ${c_llm:.4f}/1k | whole corpus: "
          f"Jev ${c_jev*n_corpus/1e3:,.0f} vs LLM ${c_llm*n_corpus/1e3:,.0f}")


if __name__ == "__main__":
    main()
