"""F7 -- Agreement with coded CRIS fields (§5.5, §4.7).

(a) discrepancy taxonomy per variable: narrative-only / code-only / both / neither, as
    counts, which is the §4.7 deliverable
(b) where the narrative adds and where it contradicts: narrative-only vs code-only rates,
    on a log scale, with the diagonal marking "equally often"
(c) keyword vs Jev against the same coded reference: precision and recall, paired

Panel (b) is the "value added" panel and it must be read carefully: narrative-only is NOT
an error rate. A crash whose narrative says the driver fell asleep but whose Contributing
Factor field says otherwise is exactly the case this paper exists to find. The adjudicated
attribution of those discrepancies (§4.7, 200 cases) is v2.
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
from s10_analysis import CODED_MAP, KW_MAP_ALL as KW_MAP

DATA = Path(__file__).resolve().parents[2] / "data"


def load():
    f = DATA / "stage2" / "stage2_flat.parquet"
    if not f.exists():
        f = DATA / "stage1" / "stage1_flat.parquet"
    d = pd.read_parquet(f)
    if "stratum" in d.columns:
        d = d[d["stratum"] == "random"]
    d = d.merge(pd.read_csv(DATA / "coded_join.csv"), on="Crash_ID", how="inner")
    d = d.merge(pd.read_parquet(DATA / "keyword_flags.parquet"), on="Crash_ID", how="left")
    d = d.merge(pd.read_parquet(DATA / "keyword_flags_extra.parquet"), on="Crash_ID", how="left")
    return d


def main() -> None:
    d = load()
    rows = []
    for v, col in CODED_MAP.items():
        if f"p_{v}" not in d.columns or col not in d.columns:
            continue
        sub = d[[f"p_{v}", col]].dropna()
        pred = (sub[f"p_{v}"].to_numpy(dtype=float) > 0.5)
        y = sub[col].astype(bool).to_numpy()
        rows.append({
            "v": v, "n": len(sub),
            "both": int((pred & y).sum()),
            "narr_only": int((pred & ~y).sum()),
            "code_only": int((~pred & y).sum()),
            "neither": int((~pred & ~y).sum()),
            "kappa": M.cohen_kappa(pred.astype(int), y.astype(int)),
        })
    rows.sort(key=lambda r: -(r["narr_only"] + r["code_only"]))
    names = [r["v"] for r in rows]
    yy = np.arange(len(rows))

    S.setup()
    # Read left to right as (a) taxonomy, (b) metric shifts, and (c) phase
    # diagram.  The compact matrix receives the smallest panel width.
    fig = plt.figure(figsize=(S.W2, 3.55))
    # Group the shared-row panels tightly.  Keep a larger separation before
    # panel (c), which uses an independent two-axis coordinate system.
    outer = fig.add_gridspec(1, 2, width_ratios=[2.40, 1.52],
                             left=0.09, right=0.985, bottom=0.18, top=0.84, wspace=0.23)
    paired = outer[0].subgridspec(1, 2, width_ratios=[1.50, 0.88], wspace=0.07)
    a = fig.add_subplot(paired[0, 0])
    b = fig.add_subplot(paired[0, 1], sharey=a)
    c = fig.add_subplot(outer[0, 1])

    # --------------------------------------------------- (a) taxonomy: largest load first
    both = np.array([r["both"] for r in rows], float)
    no = np.array([r["narr_only"] for r in rows], float)
    co = np.array([r["code_only"] for r in rows], float)
    a.barh(yy, both, color=S.BLUE, height=0.66)
    a.barh(yy, no, left=both, color=S.ORANGE, height=0.66)
    a.barh(yy, co, left=both + no, color=S.GREY, height=0.66)
    a.set_yticks(yy); a.set_yticklabels(names, fontsize=6.0)
    a.invert_yaxis()
    a.set_xlabel("Crashes (neither omitted)")
    a.set_title("(a) discrepancy taxonomy", loc="left", fontweight="bold", y=1.08)
    for i, r in enumerate(rows):
        a.text((both + no + co)[i] * 1.02, i, f"$\\kappa$={r['kappa']:.2f}", va="center",
               fontsize=5.2, color="#7A8790")
    a.set_xlim(0, (both + no + co).max() * 1.22)

    # --------------------------------------------------- (c) discrepancy phase diagram
    # Position records where the narrative and coded field disagree; circle area
    # represents the total discrepancy count.  The diagonal is exact agreement.
    n = np.array([r["n"] for r in rows], float)
    narr_rate = 100 * no / n
    code_rate = 100 * co / n
    point_area = 26 + 76 * (no + co) / (no + co).max()
    c.plot([0.015, 2.5], [0.015, 2.5], ls=(0, (2, 2)), lw=0.85, color="#96A4AE", zorder=1)
    c.scatter(narr_rate, code_rate, s=point_area, color=S.BLUE, edgecolor="white",
              linewidth=0.9, zorder=4)
    c_labels = {
        "alcohol_involved": (1.34, 1.30), "unbelted": (0.46, 2.12),
        "wrong_way": (1.43, 0.20), "hydroplane": (0.86, 0.55),
        "phone_use": (0.86, 0.065), "fatigue": (0.57, 0.18),
        "medical_episode": (0.87, 0.118), "drug_involved": (0.21, 0.38),
        "animal_involved": (0.45, 0.045),
    }
    for i, name in enumerate(names):
        tx, ty = c_labels[name]
        c.annotate(name.replace("_", " "), xy=(narr_rate[i], code_rate[i]), xytext=(tx, ty),
                   textcoords="data", fontsize=5.15, color="#48535C", zorder=6,
                   ha="left", va="center",
                   bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.96),
                   arrowprops=dict(arrowstyle="-", color="#9BA8B1", lw=0.6,
                                   shrinkA=2.0, shrinkB=3.0))
    c.set_xscale("log"); c.set_yscale("log")
    import matplotlib.ticker as mt
    for axis in (c.xaxis, c.yaxis):
        axis.set_major_locator(mt.LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12))
        axis.set_minor_locator(mt.NullLocator())
        axis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}" if v >= 0.1 else f"{v:.2f}"))
    c.set_xlim(0.18, 2.45); c.set_ylim(0.015, 2.45)
    c.set_xlabel("Narrative-only rate (% of crashes; log scale)")
    c.set_ylabel("Code-only rate (% of crashes; log scale)")
    c.set_title("(c) discrepancy phase diagram", loc="left", fontweight="bold", y=1.08)
    c.text(0.98, 0.04, "Below diagonal: narrative-only exceeds code-only", transform=c.transAxes,
           fontsize=5.45, color=S.MUTED, ha="right", va="bottom",
           bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.93))
    c.text(0.98, 1.01, "Circle area = total mismatch count", transform=c.transAxes,
           fontsize=5.45, color=S.MUTED, ha="right", va="bottom")

    # --------------------------------------------------- (b) Hinton-style metric-shift matrix
    inv = {v: k for k, v in KW_MAP.items()}
    perf = []
    for r in rows:
        v = r["v"]
        col, fam = CODED_MAP[v], inv.get(v)
        if fam is None or f"kw_{fam}" not in d.columns:
            continue
        sub = d[[f"p_{v}", col, f"kw_{fam}"]].dropna()
        truth = sub[col].astype(bool).to_numpy()
        jev = sub[f"p_{v}"].to_numpy(dtype=float) > 0.5
        keyword = sub[f"kw_{fam}"].astype(bool).to_numpy()
        perf.append({
            "v": v,
            "jev_precision": 100 * (truth & jev).sum() / max(jev.sum(), 1),
            "jev_recall": 100 * (truth & jev).sum() / max(truth.sum(), 1),
            "keyword_precision": 100 * (truth & keyword).sum() / max(keyword.sum(), 1),
            "keyword_recall": 100 * (truth & keyword).sum() / max(truth.sum(), 1),
        })
    for record in perf:
        record["delta_recall"] = record["jev_recall"] - record["keyword_recall"]
        record["delta_precision"] = record["jev_precision"] - record["keyword_precision"]
        record["gain_total"] = record["delta_recall"] + record["delta_precision"]
    # Use the exact discrepancy order in panel (a).  This makes the aligned
    # bars and metric shifts readable as one two-part comparison.
    perf_by_variable = {r["v"]: r for r in perf}
    perf = [perf_by_variable[v] for v in names if v in perf_by_variable]
    matrix = np.array([[r["delta_recall"], r["delta_precision"]] for r in perf])
    cy = np.arange(len(perf))
    max_shift = float(np.abs(matrix).max())
    b.axvspan(-0.5, 1.5, color="#F8FAFB", zorder=0)
    for i, (recall_shift, precision_shift) in enumerate(matrix):
        for j, value in enumerate((recall_shift, precision_shift)):
            # Hinton encoding: area = magnitude, blue = gain, orange = loss.
            size = 74 + 360 * abs(value) / max_shift
            color = S.BLUE if value >= 0 else S.ORANGE
            b.scatter(j, i, marker="s", s=size, color=color, edgecolor="white",
                      linewidth=0.9, zorder=3)
            b.text(j, i, f"{value:+.0f}%", ha="center", va="center", fontsize=4.8,
                   color="white",
                   fontweight="bold", zorder=4)
    b.set_xticks([0, 1]); b.set_xticklabels(["Δ Recall", "Δ Precision"], fontsize=6.1)
    b.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    b.set_yticks(cy)
    # Do not set blank tick labels here: the shared formatter also controls
    # panel (a).  Hide only this panel's rendered labels and tick marks.
    b.tick_params(axis="y", left=False, labelleft=False, length=0)
    b.spines["left"].set_visible(False)
    b.set_xlim(-0.62, 1.62)
    b.set_title("(b) keyword-to-Jev shifts", loc="left", fontweight="bold", y=1.08)
    # The side grows with the square root of the change on top of a minimum size, so the area
    # is not proportional to it; the printed percentage is what carries the value.
    b.text(0.5, -0.13, "Square size grows with |change|",
           transform=b.transAxes, fontsize=6.4, color=S.MUTED, ha="center", va="top")

    # A shared visual key replaces legends inside the data areas and the former
    # prose sentence above the figure.
    fig.legend(handles=[
                   Patch(facecolor=S.BLUE, label="both"),
                   Patch(facecolor=S.ORANGE, label="narrative only"),
                   Patch(facecolor=S.GREY, label="code only"),
                   Line2D([], [], marker="s", lw=0, markerfacecolor=S.BLUE, markeredgecolor="white",
                          markersize=5.6, label="Jev gain"),
                   Line2D([], [], marker="s", lw=0, markerfacecolor=S.ORANGE, markeredgecolor="white",
                          markersize=5.6, label="Jev loss"),
               ], loc="upper center", bbox_to_anchor=(0.5, 0.996), ncol=5,
               fontsize=6.2, handlelength=1.25, handletextpad=0.4,
               columnspacing=1.1, borderaxespad=0.0, frameon=False)
    S.finish(fig)
    S.save(fig, "F7_coded_agreement")


if __name__ == "__main__":
    main()
