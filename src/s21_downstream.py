"""WP14 -- what changes in a safety diagnosis when calibrated narrative variables are added.

The coded contributing-factor fields are what a state agency counts when it ranks the factors
behind its injury and fatal crashes. A narrative variable can only change that ranking if it
finds crashes the coded field misses, and the size of that change is a transportation quantity
rather than a machine-learning one. This script measures it on the Stage-2 random stratum,
using only data already in the repository and no new API calls.

The estimate has three parts. The coded count is the number of injury or fatal crashes whose
coded field records the factor. The added count is the expected number of crashes that the
coded field records as negative and whose narrative states the factor, which is the sum of the
recalibrated probability over those crashes. The recalibration map is the WP3 Platt map, fitted
on the human reference set per variable where a variable carries enough positives and on the
pooled map otherwise, so the probabilities that are summed are the corrected ones rather than
the model's raw output.

Summing a calibrated probability is what makes the added count an estimate of a population
total rather than a count of model flags, and the gold set supplies the independent check: the
share of narrative-only flags that a human confirms. That confirmation rate is reported beside
the calibrated estimate, and the two are compared in the manuscript.

The uncertainty that matters is the uncertainty of the map, which is fitted on 400 narratives,
so the interval comes from resampling those narratives and refitting. The sampling error of the
150,000-narrative stratum is an order of magnitude smaller and is included in the same pass.

Output: data/downstream_severity.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

import s07_metrics as M
from s10_analysis import CODED_MAP, PROXY_ONLY, TEXAS_PER_YEAR
from s17_gold_analysis import _fit_platt, _apply_platt, STRATUM, WCOL

ROOT = Path(__file__).resolve().parents[2]   # repository root, resolved from this file
DATA = ROOT / "paper1" / "data"
GOLD = DATA / "gold"
B_BOOT = 300
MIN_POSITIVES = 20          # the WP3 rule for a variable that can carry its own map


def severity_masks(d: pd.DataFrame) -> dict:
    """Fatal and injury crashes, from the person-level coded counts.

    `coded_killed` and `coded_injured` are the crash-level indicators built in s04; the raw
    `death_cnt` and `tot_injry` totals are used as the fallback where the indicator is absent,
    which keeps a crash with a recorded fatality in the fatal set even when the indicator did
    not resolve.
    """
    killed = (d.get("coded_killed").fillna(False).astype(bool)
              | (d.get("death_cnt").fillna(0) > 0))
    injured = (d.get("coded_injured").fillna(False).astype(bool)
               | (d.get("tot_injry").fillna(0) > 0))
    return {"fatal": killed.to_numpy(),
            "injury_or_fatal": (injured | killed).to_numpy()}


def fit_maps(g: pd.DataFrame) -> tuple:
    """The pooled map and the per-variable maps, fitted on the human reference set."""
    pooled = _fit_platt(g.p.to_numpy(float), g.y.to_numpy(float), g[WCOL].to_numpy(float))
    per = {}
    for v, s in g.groupby("variable"):
        if int((s.y == 1).sum()) >= MIN_POSITIVES and len(s) >= 40:
            try:
                per[v] = _fit_platt(s.p.to_numpy(float), s.y.to_numpy(float),
                                    s[WCOL].to_numpy(float))
            except Exception:
                pass
    return pooled, per


def counts_for(v: str, col: str, rand: pd.DataFrame, masks: dict, maps: tuple) -> dict:
    """Coded and narrative-augmented counts of a factor, inside each severity class."""
    pooled, per = maps
    p = rand[f"p_{v}"].to_numpy(float)
    q = _apply_platt(per.get(v, pooled), p)
    coded = rand[col].fillna(False).astype(bool).to_numpy()
    out = {}
    for name, m in masks.items():
        n_sev = int(m.sum())
        c = float(coded[m].sum())
        added = float(q[m & ~coded].sum())
        out[name] = {"n_severity": n_sev, "coded_count": c, "added_count": added,
                     "total_count": c + added,
                     "coded_share": c / n_sev if n_sev else float("nan"),
                     "augmented_share": (c + added) / n_sev if n_sev else float("nan"),
                     "relative_increase": added / c if c > 0 else float("nan")}
    return out


def main() -> None:
    d = pd.read_parquet(DATA / "stage2" / "stage2_flat.parquet")
    d = d[d.stratum == "random"].reset_index(drop=True)
    cj = pd.read_csv(DATA / "coded_join.csv")
    rand = d.merge(cj, on="Crash_ID", how="left")
    masks = severity_masks(rand)
    n_rand = len(rand)
    # one narrative in the random stratum stands for this many in the corpus, per year
    scale = (5_018_080 / n_rand) / 9.0
    print(f"random stratum {n_rand:,}; fatal {masks['fatal'].sum():,}; "
          f"injury or fatal {masks['injury_or_fatal'].sum():,}")

    g = pd.read_csv(GOLD / "gold_labels.csv")
    g = g[g.y.notna() & g.p.notna()].copy()
    gold = json.loads((GOLD / "gold_analysis.json").read_text())
    conf = gold.get("narrative_only", {}).get("per_variable", {})

    maps = fit_maps(g)
    point = {v: counts_for(v, col, rand, masks, maps)
             for v, col in CODED_MAP.items()
             if f"p_{v}" in rand.columns and col in rand.columns}

    # Interval. Resampling the reference narratives refits the map, which is where nearly all
    # of the uncertainty in the added count sits.
    ids = g.Crash_ID.to_numpy()
    uniq, inv = np.unique(ids, return_inverse=True)
    rows_of = [np.flatnonzero(inv == k) for k in range(uniq.size)]
    scol = g[STRATUM].to_numpy()
    sv = np.array([scol[rows_of[k]][0] for k in range(uniq.size)])
    groups = [np.flatnonzero(sv == u) for u in pd.unique(pd.Series(sv))]
    rng = np.random.default_rng(23)

    draws = {v: {s: [] for s in masks} for v in point}
    for b in range(B_BOOT):
        pick = np.concatenate([rng.choice(gr, size=gr.size, replace=True) for gr in groups])
        take = np.concatenate([rows_of[k] for k in pick])
        try:
            mb = fit_maps(g.iloc[take])
        except Exception:
            continue
        for v, col in CODED_MAP.items():
            if v not in point:
                continue
            c = counts_for(v, col, rand, masks, mb)
            for s in masks:
                draws[v][s].append(c[s]["added_count"])

    out = {
        "question": ("what changes in a safety diagnosis when calibrated narrative variables "
                     "are added to the coded fields"),
        "population": "Stage-2 random stratum",
        "n_random_stratum": n_rand,
        "per_year_scale": scale,
        "severity_definitions": {
            "fatal": "coded_killed, or death_cnt above zero",
            "injury_or_fatal": "coded_injured or coded_killed, or tot_injry or death_cnt "
                               "above zero"},
        "recalibration": ("WP3 Platt map, per variable where the reference set carries at "
                          f"least {MIN_POSITIVES} positives and pooled otherwise"),
        "B_bootstrap": B_BOOT,
        "bootstrap_unit": "reference-set narrative, stratified by draw route",
        "n_severity": {s: int(m.sum()) for s, m in masks.items()},
        "per_variable": {},
    }
    for v, r in point.items():
        row = {"coded_field": CODED_MAP[v], "proxy_only": v in PROXY_ONLY,
               "narrative_only_confirmed_rate":
                   conf.get(v, {}).get("confirmed_rate"),
               "narrative_only_confirmed_ci95": conf.get(v, {}).get("ci95")}
        for s in masks:
            a = np.array(draws[v][s], dtype=float)
            a = a[np.isfinite(a)]
            lo, hi = (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) \
                if a.size > 2 else (float("nan"), float("nan"))
            row[s] = {**r[s],
                      "added_count_ci95": [lo, hi],
                      "coded_per_year": r[s]["coded_count"] * scale,
                      "added_per_year": r[s]["added_count"] * scale,
                      "added_per_year_ci95": [lo * scale, hi * scale],
                      "total_per_year": r[s]["total_count"] * scale}
        out["per_variable"][v] = row

    # the ranking a state agency would read off each source, which is the diagnosis itself
    for s in masks:
        ranked_c = sorted(point, key=lambda v: -point[v][s]["coded_count"])
        ranked_a = sorted(point, key=lambda v: -point[v][s]["total_count"])
        moved = [v for v in ranked_c if ranked_c.index(v) != ranked_a.index(v)]
        out.setdefault("ranking", {})[s] = {
            "coded_only": ranked_c, "augmented": ranked_a,
            "n_variables_that_move": len(moved), "variables_that_move": moved,
            "largest_rank_change": max((abs(ranked_c.index(v) - ranked_a.index(v))
                                        for v in ranked_c), default=0)}

    (DATA / "downstream_severity.json").write_text(json.dumps(out, indent=1, default=float))

    print(f"\ninjury or fatal crashes, per year in Texas (scale {scale:.2f})")
    print(f"{'variable':18s}{'coded/yr':>11s}{'added/yr':>11s}{'95% CI':>22s}{'+%':>8s}"
          f"{'confirm':>9s}")
    S = "injury_or_fatal"
    for v, r in sorted(out["per_variable"].items(), key=lambda kv: -kv[1][S]["added_count"]):
        a = r[S]
        cr = r["narrative_only_confirmed_rate"]
        lo, hi = a["added_per_year_ci95"]
        ci = f"[{lo:,.0f}, {hi:,.0f}]"
        conf_s = f"{cr:.2f}" if cr is not None else "-"
        print(f"{v:18s}{a['coded_per_year']:>11,.0f}{a['added_per_year']:>11,.0f}"
              f"{ci:>22s}{100*a['relative_increase']:>7.0f}%{conf_s:>9s}")
    rk = out["ranking"][S]
    print(f"\nranking of factors by count in injury or fatal crashes:")
    print(f"  coded fields only : {', '.join(rk['coded_only'])}")
    print(f"  with narratives   : {', '.join(rk['augmented'])}")
    print(f"  {rk['n_variables_that_move']} of {len(point)} variables change rank, "
          f"largest move {rk['largest_rank_change']}")
    print(f"\nwrote {DATA/'downstream_severity.json'}")


if __name__ == "__main__":
    main()
