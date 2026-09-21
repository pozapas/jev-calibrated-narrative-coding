"""WP1 -- calibration weights for the human reference set under a stated design.

The frame that `s05_gold_sample.py` drew is not a probability sample with known inclusion
probabilities. The draw is a greedy sequential cell-filling rule over (variable, bin) cells,
`s15_gold_export.py` then withheld 137 of the 640 candidates on the identifier screen and
labelled the first 400 that cleared it, and the realised `incl_prob` recorded by `s05` was
never recomputed after either step. Treating `1/incl_prob` as a design weight therefore
asserts a design that was not executed, and it understates the variance badly: the recorded
weights span a factor of 3.4 and give a Kish design effect of 1.05, which is the design
effect of a self-weighting sample.

This script replaces those weights with calibration weights in the sense of Deville and
Sarndal (1992). The analysis population is the Stage-2 random stratum, 150,000 narratives,
which carries a probability for every variable. The calibration totals are the population
counts of the (variable, bin) cells that the draw targeted, with the bin edges of `s05`.

Two routes put a pair into the sample and they are separated exactly by the bin. A variable
whose probability is at or above 0.05 is always labelled, so every pair in bins 1 to 4 is an
active pair with assignment probability 1. A variable below 0.05 is labelled only when it is
one of the three negative controls drawn from that narrative's low-probability variables, so
every pair in bin 0 is a control pair with assignment probability 3 / n_low(i). No pair
crosses the boundary, which is asserted below.

Because the routes do not cross cells, calibrating within (variable, bin) reproduces the
population cell counts exactly for every variable and every bin while leaving the two design
factors visible on each pair. Calibrating a single narrative weight across all sixteen
variables at once was tried first and rejected: raking 400 narratives to sixteen margins
converges, but it puts a Kish design effect of 15.3 on the sample and leaves 26 effective
observations out of 400, because the greedy draw over-samples the upper bins of every
variable at once and no single narrative weight can undo sixteen such distortions together.
The per-variable calibration keeps the same estimand and the same calibration totals with a
design effect of 1.95.

Output: data/gold/gold_design_weights.csv and data/gold/gold_design_report.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]   # repository root, resolved from this file
DATA = ROOT / "paper1" / "data"
GOLD = DATA / "gold"

BIN_EDGES = [0.05, 0.30, 0.70, 0.95]          # interior edges of the s05 bins
BIN_NAMES = ["[0,0.05)", "[0.05,0.3)", "[0.3,0.7)", "[0.7,0.95)", "[0.95,1]"]
ACTIVE_THRESHOLD = 0.05
N_NEG_CONTROL = 3
MIN_CELL = 5                                   # smallest labelled cell that is kept on its own


def bin_of(p) -> np.ndarray:
    return np.digitize(np.asarray(p, dtype=float), BIN_EDGES, right=False)


def collapse_bins(counts: dict[int, int], min_cell: int = MIN_CELL) -> dict[int, int]:
    """Merge adjacent active bins until every group holds at least `min_cell` labelled pairs.

    Bin 0 is never merged with an active bin because the two carry different assignment
    probabilities, and it never needs to be: the smallest bin-0 cell holds 58 pairs. The rule
    repeatedly takes the smallest short group and merges it with its smaller neighbour, with
    ties broken toward the lower bin, so the result does not depend on iteration order.
    """
    groups = [1, 2, 3, 4]
    members = {k: [k] for k in groups}
    c = {k: int(counts.get(k, 0)) for k in groups}
    while len(groups) > 1:
        short = [k for k in groups if c[k] < min_cell]
        if not short:
            break
        k = min(short, key=lambda k: (c[k], k))
        i = groups.index(k)
        neigh = [groups[j] for j in (i - 1, i + 1) if 0 <= j < len(groups)]
        j = min(neigh, key=lambda j: (c[j], j))
        lo, hi = min(k, j), max(k, j)
        members[lo] = members[lo] + members[hi]
        c[lo] += c[hi]
        groups.remove(hi)
        del members[hi], c[hi]
    out = {0: 0}
    for gi, ks in members.items():
        for k in ks:
            out[k] = gi
    return out


def population_cells(pop: pd.DataFrame, variables: list[str]) -> dict[str, dict[int, int]]:
    """Population count of every (variable, bin) cell over the analysis population."""
    return {v: {k: int((bin_of(pop[f"p_{v}"]) == k).sum()) for k in range(5)}
            for v in variables}


def design_weights(labels: pd.DataFrame, frame: pd.DataFrame, pop: pd.DataFrame,
                   variables: list[str]) -> tuple[pd.DataFrame, dict]:
    """Calibration weights for every labelled pair, with both design factors recorded."""
    npop = population_cells(pop, variables)
    pv = {int(r.Crash_ID): {v: float(r[f"p_{v}"]) for v in variables}
          for _, r in frame.iterrows()}
    pi = {int(r.Crash_ID): float(r.incl_prob) for _, r in frame.iterrows()}
    n_low = {c: sum(1 for v in variables if pv[c][v] < ACTIVE_THRESHOLD) for c in pv}

    # The draw route is the cell the greedy rule drew the narrative to fill, which is the
    # stratum the cluster bootstrap of WP2 resamples within. A narrative that the top-up step
    # added rather than the cell-filling step carries no targeted cell and forms its own
    # stratum.
    route_of = {}
    for _, r in frame.iterrows():
        pb = r.primary_bin
        route_of[int(r.Crash_ID)] = (pb.split("|")[0].split(":")[1]
                                     if isinstance(pb, str) and pb else "top-up")

    d = labels.copy()
    d["draw_stratum"] = [route_of[int(c)] for c in d.Crash_ID]
    d["p_variable"] = [pv[int(r.Crash_ID)][r.variable] for _, r in d.iterrows()]
    d["bin"] = bin_of(d.p_variable)
    d["route"] = np.where(d.bin == 0, "control", "active")

    # the two routes must not cross the bin boundary, or the assignment probability below is
    # not the one that put the pair into the sample
    assert not ((d.route == "control") & (d.p_variable >= ACTIVE_THRESHOLD)).any()
    assert not ((d.route == "active") & (d.p_variable < ACTIVE_THRESHOLD)).any()

    d["assign_prob"] = [N_NEG_CONTROL / n_low[int(c)] if b == 0 else 1.0
                        for c, b in zip(d.Crash_ID, d.bin)]
    d["narrative_base_weight"] = [1.0 / pi[int(c)] for c in d.Crash_ID]
    d["base_pair_weight"] = d.narrative_base_weight / d.assign_prob

    parts, collapses = [], {}
    for v, sub in d.groupby("variable"):
        counts = {k: int((sub.bin == k).sum()) for k in range(5)}
        cmap = collapse_bins(counts)
        if len(set(cmap.values())) < 5:
            collapses[v] = [[k for k in range(5) if cmap[k] == gi]
                            for gi in sorted(set(cmap.values()))]
        sub = sub.copy()
        sub["cell"] = [cmap[b] for b in sub.bin]
        for cell, s in sub.groupby("cell"):
            target = sum(npop[v][k] for k in range(5) if cmap[k] == cell)
            factor = target / s.base_pair_weight.sum()
            parts.append(s.assign(cell_target=target, calibration_factor=factor,
                                  pair_weight=s.base_pair_weight * factor))
    out = pd.concat(parts).sort_values(["variable", "Crash_ID"]).reset_index(drop=True)

    # the calibration total of every variable is the whole analysis population
    n_pop = len(pop)
    for v, s in out.groupby("variable"):
        assert abs(s.pair_weight.sum() - n_pop) < 1e-6, v

    w = out.pair_weight.to_numpy(float)
    nw = out.groupby("Crash_ID").pair_weight.sum().to_numpy(float)
    report = {
        "population": "Stage-2 random stratum",
        "n_population": n_pop,
        "n_variables": len(variables),
        "n_pairs": int(len(out)),
        "n_narratives": int(out.Crash_ID.nunique()),
        "bin_edges": [0.0] + BIN_EDGES + [1.0],
        "bin_names": BIN_NAMES,
        "min_cell": MIN_CELL,
        "collapsed_cells": collapses,
        "n_active_pairs": int((out.route == "active").sum()),
        "n_control_pairs": int((out.route == "control").sum()),
        "weight_min": float(w.min()), "weight_max": float(w.max()),
        "weight_ratio": float(w.max() / w.min()),
        "kish_deff": float(len(w) * np.sum(w ** 2) / w.sum() ** 2),
        "n_effective": float(w.sum() ** 2 / np.sum(w ** 2)),
        "narrative_weight_ratio": float(nw.max() / nw.min()),
        "draw_strata": {str(k): int(v) for k, v in
                        out.drop_duplicates("Crash_ID").draw_stratum.value_counts().items()},
        "per_variable_n_effective": {
            v: float(s.pair_weight.sum() ** 2 / np.sum(s.pair_weight.to_numpy(float) ** 2))
            for v, s in out.groupby("variable")},
    }
    return out, report


def main() -> None:
    pop = pd.read_parquet(DATA / "stage2" / "stage2_flat.parquet")
    pop = pop[pop.stratum == "random"].reset_index(drop=True)
    variables = [c[2:] for c in pop.columns if c.startswith("p_")]

    fp = GOLD / ("gold_frame_oversampled.csv"
                 if (GOLD / "gold_frame_oversampled.csv").exists() else "gold_frame.csv")
    frame = pd.read_csv(fp)
    labels = pd.read_csv(GOLD / "gold_labels.csv")
    frame = frame[frame.Crash_ID.isin(set(labels.Crash_ID))].reset_index(drop=True)

    # every labelled narrative has to come from the analysis population, or the calibration
    # totals do not describe the frame the weights are applied to
    missing = set(labels.Crash_ID) - set(pop.Crash_ID)
    assert not missing, f"{len(missing)} labelled narratives outside the analysis population"

    out, report = design_weights(labels, frame, pop, variables)
    cols = ["Crash_ID", "variable", "p_variable", "bin", "cell", "route", "draw_stratum",
            "narrative_base_weight", "assign_prob", "base_pair_weight",
            "cell_target", "calibration_factor", "pair_weight"]
    out[cols].to_csv(GOLD / "gold_design_weights.csv", index=False)
    (GOLD / "gold_design_report.json").write_text(json.dumps(report, indent=1, default=float))

    print(f"{report['n_pairs']:,} pairs over {report['n_narratives']} narratives, "
          f"{report['n_active_pairs']:,} active and {report['n_control_pairs']:,} control")
    print(f"calibration totals: {report['n_variables']} variables x "
          f"{report['n_population']:,} narratives")
    if report["collapsed_cells"]:
        print("collapsed cells (fewer than "
              f"{MIN_CELL} labelled pairs):")
        for v, gs in report["collapsed_cells"].items():
            print(f"  {v:18s} {gs}")
    print(f"pair weights {report['weight_min']:.1f} to {report['weight_max']:.1f} "
          f"(ratio {report['weight_ratio']:.0f})")
    print(f"Kish design effect {report['kish_deff']:.3f}, "
          f"effective sample {report['n_effective']:.0f} of {report['n_pairs']:,}")
    print(f"wrote {GOLD / 'gold_design_weights.csv'}")


if __name__ == "__main__":
    main()
