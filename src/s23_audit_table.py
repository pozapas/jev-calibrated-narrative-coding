"""WP13 -- the de-identified pair-level audit table that the release can actually be checked on.

Everything measured against the human reference set can be recomputed from one table, provided
that table carries the weights and the fold structure as well as the labels. This script builds
it. The crash identifier is replaced by a salted hash truncated to twelve hexadecimal
characters, no narrative text is carried, and the salt is written to a file that is not
released, so the mapping back to a crash cannot be reconstructed from the table alone.

What the table does NOT cover is the population-scale work, because that needs the Stage-2
outputs for 150,000 narratives joined to coded fields at crash level. Those are not
redistributable, and the Data availability statement says so rather than implying otherwise.

Output: data/release/gold_pairs_audit.csv and data/release/gold_pairs_audit_README.md
"""
from __future__ import annotations
import hashlib
import json
import os
import secrets
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]   # repository root, resolved from this file
DATA = ROOT / "paper1" / "data"
GOLD = DATA / "gold"
FRONT = DATA / "frontier"
OUT = DATA / "release"
SALT_FILE = DATA / "release_salt.txt"          # never released; listed in the manifest as private


def salt() -> str:
    """A stable salt, generated once and reused so the identifiers persist across rebuilds."""
    env = os.environ.get("JEV_RELEASE_SALT")
    if env:
        return env
    if SALT_FILE.exists():
        return SALT_FILE.read_text(encoding="utf-8").strip()
    s = secrets.token_hex(16)
    SALT_FILE.write_text(s, encoding="utf-8")
    print(f"generated a new release salt at {SALT_FILE}; keep it out of the release")
    return s


def anon(crash_ids: pd.Series, s: str) -> pd.Series:
    return crash_ids.astype(str).map(
        lambda c: hashlib.sha256((s + c).encode()).hexdigest()[:12])


def frontier_predictions() -> pd.DataFrame | None:
    """Both benchmarked arms' probabilities, keyed by crash and variable."""
    import s18_frontier_analysis as FA
    parts = []
    for prefix, label in (("fable", "claude-fable-5-1"), ("gpt", "gpt-5.6-sol")):
        arm = FA.load_arm(prefix)
        if arm is None:
            continue
        parts.append(arm[["Crash_ID", "variable", "p"]].rename(columns={"p": f"p_{prefix}"}))
    if not parts:
        return None
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on=["Crash_ID", "variable"], how="outer")
    return out


def main() -> None:
    g = pd.read_csv(GOLD / "gold_labels.csv")
    if "pair_weight" not in g.columns:
        raise SystemExit("gold_labels.csv carries no pair_weight; run s05b then s16 first")

    fr = frontier_predictions()
    if fr is not None:
        fr["Crash_ID"] = fr["Crash_ID"].astype(str)
        g = g.merge(fr.assign(Crash_ID=fr.Crash_ID.astype(str)),
                    left_on=[g.Crash_ID.astype(str), "variable"],
                    right_on=["Crash_ID", "variable"], how="left",
                    suffixes=("", "_fr")).drop(columns=[c for c in ("Crash_ID_fr", "key_0")
                                                       if c in g.columns], errors="ignore")

    s = salt()
    out = pd.DataFrame({
        "narrative_id": anon(g["Crash_ID"], s),
        "variable": g["variable"],
        "p_model": g["p"],
        "consensus_label": g["label"],
        "y": g["y"],
        "n_raters": g["n_raters"],
        "rater_agreement": g["agreement"],
        "dispute_status": g["label"].map(
            lambda x: "disputed" if x == "DISPUTED" else ("unclear" if x == "unclear"
                                                          else "resolved")),
        "bin": g["bin"],
        "calibration_cell": g["cell"],
        "route": g["route"],
        "draw_stratum": g["draw_stratum"],
        "narrative_base_weight": g["narrative_base_weight"],
        "assign_prob": g["assign_prob"],
        "calibration_factor": g["calibration_factor"],
        "pair_weight": g["pair_weight"],
    })
    for c in ("p_fable", "p_gpt"):
        if c in g.columns:
            out[c] = g[c]

    assert "Crash_ID" not in out.columns
    assert out.narrative_id.nunique() == g.Crash_ID.nunique(), "hash collision in the identifier"

    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "gold_pairs_audit.csv", index=False)

    design = json.loads((GOLD / "gold_design_report.json").read_text())
    readme = f"""# De-identified pair-level audit table

{len(out):,} rows, one per (narrative, variable) pair of the human reference set, over
{out.narrative_id.nunique()} narratives and {out.variable.nunique()} variables.

The crash identifier is replaced by a salted SHA-256 hash truncated to twelve characters. The
salt is not released, so the table cannot be linked back to a crash record. No narrative text
is carried.

## Columns

| column | meaning |
|---|---|
| `narrative_id` | salted hash of the crash identifier, stable across rebuilds |
| `variable` | the presence variable the pair concerns |
| `p_model` | the typed model's returned probability |
| `consensus_label` | majority label over the coders, or DISPUTED or unclear |
| `y` | 1 for yes, 0 for no, empty where the pair was not resolved |
| `n_raters`, `rater_agreement` | how many coders saw the pair and the majority's share |
| `dispute_status` | resolved, disputed or unclear |
| `bin`, `calibration_cell` | the probability bin and the cell the weight was calibrated in |
| `route` | active or control, which fixes the assignment probability |
| `draw_stratum` | the cell the greedy draw took the narrative to fill; the bootstrap stratum |
| `narrative_base_weight` | the reciprocal of the frame's realized inclusion probability |
| `assign_prob` | probability that this variable was assigned to this narrative |
| `calibration_factor` | the within-cell scaling that matches the population count |
| `pair_weight` | the analysis weight, the product of the three columns above |
| `p_fable`, `p_gpt` | the two benchmarked frontier arms' probabilities on the same pair |

## What can be recomputed from this file

Every human-reference result in the paper. The accuracy and calibration statistics, the
recalibration maps and their held-out errors, the cross-fitted review budgets, the exclusion
sensitivity and the frontier comparison all read only these columns. The design totals are
in `gold_design_report.json`, which is released beside it.

## What cannot

Anything that needs the 150,000-narrative Stage-2 stratum or the coded fields at crash level,
which is the population coding of Section 4.1 and the downstream counts of Section 4.5. Those
inputs are not redistributable under the data agreement. The aggregated result files behind
them are released instead.

## Design summary

Population {design['n_population']:,} narratives; {design['n_active_pairs']:,} active pairs and
{design['n_control_pairs']:,} control pairs; weights spanning a factor of
{design['weight_ratio']:.0f}; Kish design effect {design['kish_deff']:.2f} for an effective
sample of {design['n_effective']:.0f}.
"""
    (OUT / "gold_pairs_audit_README.md").write_text(readme, encoding="utf-8")
    print(f"wrote {OUT / 'gold_pairs_audit.csv'} ({len(out):,} rows, {out.shape[1]} columns)")
    print(f"wrote {OUT / 'gold_pairs_audit_README.md'}")


if __name__ == "__main__":
    main()
