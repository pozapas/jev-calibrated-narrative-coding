"""Step 5 (v1 part) -- probability-stratified gold-set frame (§4.4).

v1 scope emits the FRAME only: which narratives to label, with what inclusion probability,
and which variables are "active" for each. The LLM pre-annotation and the human adjudication
are v2 (7-day) items; the 48-hour v1 reports the design fully specified and the gold set as
in progress, per §6 "48-hour arXiv v1 scope".

Design (§4.4):
  bins       [0,0.05) [0.05,0.3) [0.3,0.7) [0.7,0.95) [0.95,1]
  targets       30        40         40        40        30      per Noul
  frame      Stage-2 RANDOM rows only (never the enriched strata)
  active     variables with p >= 0.05, plus 3 random low-p variables as negative controls
  weights    w_i = 1/pi_i, pi_i = P(narrative i is drawn), recorded per narrative

Because one narrative serves several variables at once, the draw is sequential and greedy:
bins that are still short pull narratives that also fill other short bins. pi_i is then the
realised marginal inclusion probability within the bin the narrative was drawn from, which
is what the Horvitz-Thompson estimator in s07_metrics needs.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:/OneDrive - Texas State University/AIT/Papers/Jev")
DATA = ROOT / "paper1" / "data"
OUT = DATA / "gold"
SEED = 42

BIN_EDGES = [0.0, 0.05, 0.30, 0.70, 0.95, 1.0000001]
BIN_NAMES = ["[0,0.05)", "[0.05,0.3)", "[0.3,0.7)", "[0.7,0.95)", "[0.95,1]"]
BIN_TARGET = [30, 40, 40, 40, 30]
N_TARGET = 400
N_NEG_CONTROL = 3
ACTIVE_THRESHOLD = 0.05


def load_random(source: str = "stage2") -> pd.DataFrame:
    """Flattened Noul probabilities, random stratum only.

    `source` exists so the frame builder can be exercised against Stage 1 before Stage 2
    finishes; the released frame is always built from stage2.
    """
    f = DATA / source / f"{source}_flat.parquet"
    if not f.exists():
        raise SystemExit(f"missing {f} -- run the {source} runner then s06_flatten.py first")
    d = pd.read_parquet(f)
    if "stratum" in d.columns:
        d = d[d["stratum"] == "random"]
    d = d.reset_index(drop=True)
    print(f"{source} random rows: {len(d):,}")
    return d


def noul_columns(d: pd.DataFrame) -> list[str]:
    return [c[2:] for c in d.columns if c.startswith("p_")]


def build_frame(d: pd.DataFrame, n_target: int = N_TARGET,
                exclude: set | None = None) -> pd.DataFrame:
    """`n_target` allows oversampling so that narratives withheld by the PII screen
    can be replaced without shrinking the gold set below its design size;
    `exclude` drops Crash_IDs already used or already withheld."""
    if exclude:
        d = d[~d['Crash_ID'].isin(exclude)].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    nouls = noul_columns(d)
    P = {v: d[f"p_{v}"].to_numpy(dtype=float) for v in nouls}
    n = len(d)

    # bin index per (variable, narrative)
    B = {v: np.digitize(P[v], BIN_EDGES[1:-1], right=False) for v in nouls}

    need = {(v, b): BIN_TARGET[b] for v in nouls for b in range(5)}
    pool_size = {(v, b): int(np.count_nonzero(B[v] == b)) for v in nouls for b in range(5)}
    for k, t in need.items():
        need[k] = min(t, pool_size[k])          # cannot draw more than exist

    chosen: dict[int, dict] = {}
    order = rng.permutation(n)

    def gain(i: int) -> int:
        return sum(1 for v in nouls if need[(v, B[v][i])] > 0)

    # greedy sequential draw: repeatedly take a random candidate that fills the most
    # still-short bins, so 400 narratives cover ~2,000 judgments
    remaining = list(order)
    while any(t > 0 for t in need.values()) and len(chosen) < n_target and remaining:
        batch = remaining[:4000]
        gains = np.array([gain(i) for i in batch])
        if gains.max() == 0:
            break
        best = [b for b, g in zip(batch, gains) if g == gains.max()]
        i = int(rng.choice(best))
        remaining.remove(i)
        filled = [v for v in nouls if need[(v, B[v][i])] > 0]
        for v in filled:
            need[(v, B[v][i])] -= 1
        chosen[i] = {"drawn_for": filled}

    # top up to N_TARGET with a simple random sample so the frame is never short
    if len(chosen) < n_target:
        extra = [i for i in remaining][: n_target - len(chosen)]
        for i in extra:
            chosen[i] = {"drawn_for": []}

    # realised inclusion probability: per (variable, bin) cell, n_drawn / pool_size.
    # A narrative's pi_i is 1 - prod(1 - pi_cell) over the cells it could have been drawn from,
    # which for these small rates is dominated by the largest cell rate.
    drawn_in_cell: dict[tuple[str, int], int] = {}
    for i in chosen:
        for v in nouls:
            k = (v, int(B[v][i]))
            drawn_in_cell[k] = drawn_in_cell.get(k, 0) + 1

    rows = []
    for i, meta in chosen.items():
        cell_rates = []
        for v in nouls:
            k = (v, int(B[v][i]))
            ps = pool_size.get(k, 0)
            if ps:
                cell_rates.append(drawn_in_cell.get(k, 0) / ps)
        pi = float(1.0 - np.prod([1.0 - r for r in cell_rates])) if cell_rates else 1.0 / n
        pi = min(max(pi, 1e-9), 1.0)
        active = [v for v in nouls if P[v][i] >= ACTIVE_THRESHOLD]
        low = [v for v in nouls if P[v][i] < ACTIVE_THRESHOLD]
        controls = list(rng.choice(low, size=min(N_NEG_CONTROL, len(low)), replace=False)) if low else []
        rows.append({
            "Crash_ID": int(d["Crash_ID"].iloc[i]),
            "incl_prob": pi,
            "ht_weight": 1.0 / pi,
            "n_active": len(active),
            "active_vars": "|".join(active),
            "control_vars": "|".join(controls),
            "drawn_for": "|".join(meta["drawn_for"]),
            "primary_bin": "|".join(f"{v}:{BIN_NAMES[int(B[v][i])]}" for v in meta["drawn_for"]),
            **{f"p_{v}": float(P[v][i]) for v in nouls},
        })

    g = pd.DataFrame(rows)
    n_judgments = int(g["n_active"].sum() + g["control_vars"].str.count(r"\|").add(1).where(
        g["control_vars"] != "", 0).sum())
    print(f"gold frame: {len(g):,} narratives, ~{n_judgments:,} judgments "
          f"(target {n_target} / ~2,000)")

    shortfall = {f"{v} {BIN_NAMES[b]}": t for (v, b), t in need.items() if t > 0}
    cov = pd.DataFrame(
        [{"variable": v, **{BIN_NAMES[b]: drawn_in_cell.get((v, b), 0) for b in range(5)},
          **{f"pool_{BIN_NAMES[b]}": pool_size[(v, b)] for b in range(5)}} for v in nouls])

    OUT.mkdir(parents=True, exist_ok=True)
    g.to_csv(OUT / "gold_frame.csv", index=False)
    cov.to_csv(OUT / "gold_bin_coverage.csv", index=False)
    (OUT / "gold_design.json").write_text(json.dumps({
        "bins": BIN_NAMES, "bin_targets": BIN_TARGET, "n_target": n_target,
        "active_threshold": ACTIVE_THRESHOLD, "n_negative_controls": N_NEG_CONTROL,
        "seed": SEED, "frame": "stage2 random stratum only",
        "n_drawn": len(g), "n_judgments_planned": n_judgments,
        "bins_short_of_target": shortfall,
        "estimator": "Horvitz-Thompson, w_i = 1/pi_i; stratified bootstrap B=1000 within bin",
        "status_v1": "frame only; LLM pre-annotation and human adjudication are v2",
    }, indent=1))
    print(f"wrote {OUT/'gold_frame.csv'}, gold_bin_coverage.csv, gold_design.json")
    if shortfall:
        print(f"NOTE: {len(shortfall)} (variable, bin) cells short of target "
              f"(rare variables simply have too few narratives in the upper bins)")
    return g


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="stage2", choices=["stage1", "stage2"])
    ap.add_argument("--dry-run", action="store_true",
                    help="build and report but write nothing (for testing before Stage 2)")
    a = ap.parse_args()
    if a.dry_run:
        OUT = DATA / "_gold_dryrun"
    build_frame(load_random(a.source))
