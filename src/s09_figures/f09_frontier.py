"""F9 -- Cost-latency-accuracy frontier (§5.7).

Scatter of methods in cost-per-1,000-narratives (log x) against macro-F1 vs the coded
fields (y), with latency as marker size, plus an inset giving the cost to code all Texas
2017-2025 narratives.

v1 plots the two methods that have been RUN: keyword rules (free) and Jev (measured). The
frontier LLM is shown as a cost-only reference band, because §6's 48-hour scope does not
include the LLM baseline run -- its F1 is a v2 number and the panel says so rather than
inventing a point. Its cost axis is real: published list price times the same input tokens.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    tok = float(d["input_tokens"].mean())
    lat_jev = float(d["latency_s"].median())
    c_jev = P.jev_cost_per_1k(tok)
    c_llm = P.llm_cost_per_1k(tok)
    n_corpus = stats["n_kept_gt_40"]

    S.setup()
    fig, ax = plt.subplots(figsize=(S.W15, 2.95))

    C_KW = 0.0012      # not free in practice: regex over 5 M narratives is ~1 CPU-hour
    ax.scatter([C_KW], [f1_kw], s=40, color=S.GREEN, zorder=5, label="keyword rules")
    ax.scatter([c_jev], [f1_jev], s=40 + 260 * lat_jev, color=S.BLUE, zorder=5,
               label=f"Jev ({P.JEV_MODEL})")
    ax.axvspan(c_llm * 0.92, c_llm * 1.08, color=S.RED, alpha=0.15, zorder=0)
    ax.axvline(c_llm, color=S.RED, lw=0.9, ls=(0, (3, 2)), zorder=2)
    ax.annotate(f"{P.LLM_BASELINE}\ncost measured, F1 in v2",
                xy=(c_llm, 0.30), xytext=(c_llm * 0.62, 0.20), fontsize=6.0, color=S.RED,
                ha="right", linespacing=1.45)

    # Pareto staircase over the methods that have both axes
    pts = sorted([(C_KW, f1_kw), (c_jev, f1_jev)])
    best = -1.0
    xs, ys = [], []
    for cx, fy in pts:
        best = max(best, fy)
        xs += [cx, cx]; ys += [best if len(ys) == 0 else ys[-1], best]
    ax.step([p[0] for p in pts] + [c_llm * 1.6],
            [p[1] for p in pts] + [max(f1_kw, f1_jev)], where="post",
            color="#BBBBC4", lw=0.9, ls=(0, (4, 2)), zorder=1, label="Pareto frontier")

    ax.annotate("", xy=(c_jev, f1_jev - 0.035), xytext=(C_KW, f1_kw - 0.035),
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#999999"))
    ax.text(np.sqrt(C_KW * c_jev), min(f1_kw, f1_jev) - 0.075,
            f"+{100*(f1_jev-f1_kw):.1f} F1 points\nfor {c_jev/C_KW:.0f}× the cost",
            fontsize=6.0, ha="center", va="top", color="#666666", linespacing=1.45)

    ax.set_xscale("log")
    ax.set_xlabel("USD per 1,000 narratives (log scale)")
    ax.set_ylabel(f"macro-F1 vs coded fields ({n_jev} variables)")
    ax.set_xlim(C_KW * 0.35, c_llm * 4.0)
    ax.set_ylim(max(0, min(f1_kw, f1_jev) - 0.14), max(f1_kw, f1_jev) + 0.23)
    ax.legend(loc="upper left", fontsize=6.2)
    ax.set_title("Cost-accuracy frontier; marker area $\\propto$ median latency",
                 loc="left", fontweight="bold", fontsize=7.4)
    ax.text(0.99, 0.06,
            f"{run}, n = {len(d):,}\nJev {tok:,.0f} input tok, p50 {lat_jev:.2f}s\n"
            f"agreement with a noisy reference, not accuracy",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6,
            color="#777777", linespacing=1.5)

    # inset: whole-corpus cost
    ins = ax.inset_axes([0.60, 0.545, 0.375, 0.335])
    labels = ["keyword", "Jev", P.LLM_BASELINE.replace("claude-", "")]
    vals = [C_KW * n_corpus / 1e3, c_jev * n_corpus / 1e3, c_llm * n_corpus / 1e3]
    ins.barh([0, 1, 2], vals, color=[S.GREEN, S.BLUE, S.RED], height=0.6)
    ins.set_xscale("log")
    ins.set_yticks([0, 1, 2]); ins.set_yticklabels(labels, fontsize=5.2)
    ins.set_xlabel("USD to code all 5.02 M", fontsize=5.4, labelpad=1.5)
    ins.tick_params(labelsize=5.0)
    for i, v in enumerate(vals):
        ins.text(v * 1.25, i, f"${v:,.0f}", va="center", fontsize=5.0, color="#444444")
    ins.set_xlim(min(vals) * 0.4, max(vals) * 9)
    for sp in ins.spines.values():
        sp.set_linewidth(0.5)

    S.finish(fig)
    fig.tight_layout(pad=0.85)
    S.save(fig, "F9_frontier")
    print(f"  keyword macro-F1 {f1_kw:.3f} ({n_kw} vars) | Jev macro-F1 {f1_jev:.3f} ({n_jev} vars)")
    print(f"  Jev ${c_jev:.4f}/1k | {P.LLM_BASELINE} ${c_llm:.4f}/1k | whole corpus: "
          f"Jev ${c_jev*n_corpus/1e3:,.0f} vs LLM ${c_llm*n_corpus/1e3:,.0f}")


if __name__ == "__main__":
    main()
