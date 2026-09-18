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
    fig = plt.figure(figsize=(S.W2, 2.90))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.42,
                          left=0.115, right=0.985, top=0.84, bottom=0.18)
    a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))

    # --------------------------------------------------- (a) taxonomy
    both = np.array([r["both"] for r in rows], float)
    no = np.array([r["narr_only"] for r in rows], float)
    co = np.array([r["code_only"] for r in rows], float)
    a.barh(yy, both, color=S.BLUE, height=0.66, label="both")
    a.barh(yy, no, left=both, color=S.ORANGE, height=0.66, label="narrative only")
    a.barh(yy, co, left=both + no, color=S.GREY, height=0.66, label="code only")
    a.set_yticks(yy); a.set_yticklabels(names, fontsize=6.0)
    a.set_xlabel("crashes (neither omitted)")
    a.set_title("(a) discrepancy taxonomy", loc="left", fontweight="bold")
    a.legend(loc="upper right", fontsize=5.8)
    for i, r in enumerate(rows):
        a.text((both + no + co)[i] * 1.02, i, f"$\\kappa$={r['kappa']:.2f}", va="center",
               fontsize=5.2, color="#888888")
    a.set_xlim(0, (both + no + co).max() * 1.22)

    # --------------------------------------------------- (b) added vs missed
    n = np.array([r["n"] for r in rows], float)
    x = 100 * no / n
    yv = 100 * co / n
    lim = max(x.max(), yv.max()) * 1.6
    b.plot([1e-3, lim], [1e-3, lim], ls=(0, (2, 2)), lw=0.7, color="#999999")
    b.scatter(x, yv, s=22, color=S.BLUE, zorder=3)
    for i, nm in enumerate(names):
        b.annotate(nm, (x[i], yv[i]), fontsize=4.8, xytext=(2.5, 2.5),
                   textcoords="offset points", color="#444444")
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlim(max(1e-3, x.min() * 0.45), lim); b.set_ylim(max(1e-3, yv.min() * 0.45), lim)
    # the default log locator crowds this narrow range into unreadable minor labels
    import matplotlib.ticker as mt
    for axis in (b.xaxis, b.yaxis):
        axis.set_major_locator(mt.LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12))
        axis.set_minor_locator(mt.NullLocator())
        axis.set_major_formatter(mt.FuncFormatter(
            lambda v, _: f"{v:g}" if v >= 0.1 else f"{v:.2f}"))
    b.set_xlabel("narrative only (% of crashes)")
    b.set_ylabel("code only (% of crashes)")
    b.set_title("(b) what each source adds", loc="left", fontweight="bold")
    b.text(0.96, 0.06, "below the line:\nnarrative flags more", transform=b.transAxes,
           fontsize=5.4, ha="right", color="#777777", linespacing=1.4)

    # --------------------------------------------------- (c) keyword vs Jev
    inv = {v: k for k, v in KW_MAP.items()}
    jp, jr, kp, kr, lab = [], [], [], [], []
    for r in rows:
        v = r["v"]
        col, fam = CODED_MAP[v], inv.get(v)
        if fam is None or f"kw_{fam}" not in d.columns:
            continue
        sub = d[[f"p_{v}", col, f"kw_{fam}"]].dropna()
        y = sub[col].astype(bool).to_numpy()
        pj = (sub[f"p_{v}"].to_numpy(dtype=float) > 0.5)
        pk = sub[f"kw_{fam}"].astype(bool).to_numpy()
        for pr, P, R in ((pj, jp, jr), (pk, kp, kr)):
            P.append(100 * (y & pr).sum() / max(pr.sum(), 1))
            R.append(100 * (y & pr).sum() / max(y.sum(), 1))
        lab.append(v)
    ii = np.arange(len(lab))
    for i in ii:
        c.plot([kr[i], jr[i]], [kp[i], jp[i]], color="#D5D5DD", lw=0.8, zorder=1)
    c.scatter(kr, kp, s=20, color=S.GREEN, zorder=3, label="keyword")
    c.scatter(jr, jp, s=20, color=S.BLUE, zorder=3, label="Jev")
    for i, nm in enumerate(lab):
        c.annotate(nm, (jr[i], jp[i]), fontsize=4.8, xytext=(2.5, 2.5),
                   textcoords="offset points", color="#444444")
    c.set_xlabel("recall vs coded field (%)")
    c.set_ylabel("precision vs coded field (%)")
    c.set_title("(c) keyword vs Jev", loc="left", fontweight="bold")
    c.set_xlim(0, 102); c.set_ylim(0, 102)
    c.legend(loc="upper right", fontsize=5.8)

    fig.suptitle("\"Narrative only\" is not an error: a factor the coded field omits is what "
                 "this paper looks for (adjudication: v2)", fontsize=6.8, y=0.985,
                 color="#555555")
    S.finish(fig)
    S.save(fig, "F7_coded_agreement")


if __name__ == "__main__":
    main()
