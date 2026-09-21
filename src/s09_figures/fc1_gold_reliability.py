"""Figure C.1 -- reliability against the human reference set for the presence variables that
have no coded CRIS reference and therefore do not appear in F4.

Nothing is fitted or recomputed here. Each panel draws, for one variable, the observed
positive rate among the human labels within each of the five probability strata used to
draw the reference set, Horvitz--Thompson weighted, with a Clopper--Pearson interval on the
Kish effective sample size. This is the same construction as the blue series of F4, applied
to the seven variables F4 omits, so a reader can see every variable's reliability against
human judgment. Axes are logarithmic so the rare region is visible.

Input:  data/gold/gold_labels.csv (majority human labels with the WP1 calibration weights and the model's p)
Output: outputs/figures/FC1_gold_reliability.pdf / .png
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

DATA = Path(__file__).resolve().parents[2] / "data"
BINS = [0.0, 0.05, 0.30, 0.70, 0.95, 1.0000001]
FLOOR = 1.5e-3
# the presence variables that have a coded reference and are therefore already in F4
IN_F4 = {"alcohol_involved", "drug_involved", "fatigue", "animal_involved", "phone_use",
         "unbelted", "hydroplane", "wrong_way", "medical_episode"}


def strata_curve(sub: pd.DataFrame):
    p = sub.p.to_numpy(float); y = sub.y.to_numpy(float); w = sub.pair_weight.to_numpy(float)
    b = np.digitize(p, BINS[1:-1], right=False)
    xs, ys, lo, hi, empty = [], [], [], [], []
    for k in range(5):
        m = b == k
        if not m.any():
            continue
        wk = w[m]
        rate = float(np.average(y[m], weights=wk))
        n_eff = float(wk.sum() ** 2 / np.sum(wk ** 2))
        l, h = M.clopper_pearson(int(round(rate * n_eff)), int(round(n_eff)))
        x = float(np.average(p[m], weights=wk))
        if rate <= 0:
            empty.append(x)
        else:
            xs.append(x); ys.append(rate); lo.append(l); hi.append(h)
    return np.array(xs), np.array(ys), np.array(lo), np.array(hi), np.array(empty)


def main() -> None:
    S.setup()
    g = pd.read_csv(DATA / "gold" / "gold_labels.csv")
    g = g[g.y.notna() & g.p.notna()]
    vars_ = sorted(v for v in g.variable.unique() if v not in IN_F4)
    ncol = 4
    nrow = int(np.ceil(len(vars_) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(S.W2, 2.1 * nrow + 0.5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, v in zip(axes, vars_):
        sub = g[g.variable == v]
        xs, ys, lo, hi, empty = strata_curve(sub)
        ax.plot([FLOOR, 1], [FLOOR, 1], ls=":", color=S.GREY, lw=0.8, zorder=1)
        if xs.size:
            ax.errorbar(xs, ys, yerr=[np.maximum(ys - lo, 0), np.maximum(hi - ys, 0)],
                        fmt="o-", color=S.BLUE, ms=4, lw=1.1, capsize=2, zorder=5)
        if empty.size:
            ax.plot(empty, np.full(empty.size, FLOOR), "o", mfc="none", mec=S.BLUE, ms=5, zorder=5)
        ax.axvline(0.01, color=S.ORANGE, ls=":", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(7e-3, 1.15); ax.set_ylim(FLOOR * 0.8, 1.3)
        ax.set_title(v, loc="left", fontsize=8.5, pad=3)
        ax.text(0.04, 0.96, f"n = {len(sub)}", transform=ax.transAxes, va="top", fontsize=7.5)
    for ax in axes[len(vars_):]:
        ax.set_visible(False)
    for ax in axes[:len(vars_)]:
        ax.set_xlabel("predicted $p$")
    for i in range(0, len(vars_), ncol):
        axes[i].set_ylabel("observed rate")
    S.finish(fig, grid="both")
    fig.tight_layout()
    S.save(fig, "FC1_gold_reliability")


if __name__ == "__main__":
    main()
