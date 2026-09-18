"""Score the frontier-LLM baseline arm against the same human gold labels as Jev (§5.7).

This is the comparison a reader of §5.4 immediately wants: Jev is cheap, but is it cheap
*enough* to justify not using a frontier model? The arm is built so that the only difference
between the two columns is the model:

  - same 400 narratives, same per-narrative variable assignment;
  - same criteria text, verbatim;
  - same human labels as the reference, same Horvitz-Thompson weights;
  - same metric code -- s17.block is imported rather than reimplemented, so a change to the
    definition of ECE or the calibration slope cannot move one arm and not the other.

Two things are deliberately NOT matched, and both are reported rather than hidden.

  1. Jev returns a probability as part of its interface; a generative model has to be asked
     for one. Whether an elicited probability is calibrated is a question about the interface,
     not a defect in the comparison, and it is the reason this arm reports ECE and not only
     F1.
  2. Cost. Accuracy here is measured by actually running the models; cost is NOT, because the
     harness used does not meter tokens at list price. The dollar figures are computed
     analytically from the measured token counts in cost_model.json times published list
     prices, exactly as F9 already does. The manuscript must state that split.

Output: data/frontier/frontier_analysis.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

import pricing as P
from s17_gold_analysis import block, MIN_N

ROOT = Path(r"D:/OneDrive - Texas State University/AIT/Papers/Jev")
DATA = ROOT / "paper1" / "data"
FRONT = DATA / "frontier"
GOLD = DATA / "gold"


def load_arm(prefix: str) -> pd.DataFrame | None:
    """Read one model's batch outputs into (Crash_ID, variable, p, answer) rows."""
    files = sorted((FRONT / "raw").glob(f"{prefix}_batch_*.json"))
    if not files:
        return None
    rows = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        for r in d["results"]:
            for var, a in r["answers"].items():
                rows.append({"Crash_ID": str(r["id"]), "variable": var,
                             "p": float(a["p"]), "answer": bool(a["answer"])})
    out = pd.DataFrame(rows)
    print(f"{prefix}: {len(files)} batch files, {len(out)} answers, "
          f"{out.Crash_ID.nunique()} narratives")
    return out


def join_gold(arm: pd.DataFrame) -> pd.DataFrame:
    """Join to the human labels.

    Crash_ID is an int64 in gold_labels.csv and a JSON string in the model output. Merging
    across that dtype boundary silently yields zero rows rather than raising, so both sides
    are cast and the join is asserted rather than trusted.
    """
    g = pd.read_csv(GOLD / "gold_labels.csv")
    g["Crash_ID"] = g["Crash_ID"].astype(str)
    g = g[g.y.notna()]
    m = g.merge(arm, on=["Crash_ID", "variable"], how="left", suffixes=("_jev", ""))
    missing = int(m.p.isna().sum())
    if missing:
        print(f"  WARNING: {missing} human-labelled pairs have no frontier answer; "
              "these are dropped and the coverage is reported")
    m = m[m.p.notna()].copy()
    assert len(m) > 0, "join produced no rows -- check Crash_ID dtypes"
    m["coverage_of_gold"] = len(m) / len(g)
    return m


def arm_metrics(m: pd.DataFrame) -> dict:
    d = m.rename(columns={"p": "p"})[["p", "y", "ht_weight", "variable"]].copy()
    per = {v: block(s) for v, s in d.groupby("variable") if len(s) >= MIN_N}
    pooled = block(d)
    pooled["n_variables"] = len(per)
    # A generative model states a label AND a probability, and they need not agree. Jev
    # cannot disagree with itself this way, so the rate is reported as an interface property.
    disagree = float((m.answer != (m.p > 0.5)).mean())
    return {"pooled": pooled, "per_variable": per,
            "label_probability_disagreement": disagree,
            "n_scored": int(len(m)),
            "coverage_of_gold_labels": float(m.coverage_of_gold.iloc[0]),
            "distinct_p_values": int(m.p.nunique()),
            "p_on_two_decimal_grid": float((np.round(m.p, 2) == m.p).mean())}


def paired_comparison(arms_raw: dict, seed: int = 7, B: int = 2000) -> dict:
    """Paired bootstrap of the F1 difference between arms on the SAME judgments.

    Comparing two independent confidence intervals is the wrong test: the arms answer the
    identical 2,416 pairs, so the comparison is paired and an unpaired reading would be
    needlessly conservative. Resampling judgments (not models) preserves that pairing.

    This exists because the headline is a difference, not a level. "Model A scores 0.959 and
    model B 0.891" invites the reader to supply their own significance test; reporting the
    difference with its interval does not.
    """
    g = pd.read_csv(GOLD / "gold_labels.csv")
    g["Crash_ID"] = g["Crash_ID"].astype(str)
    g = g[g.y.notna()]
    m = g.rename(columns={"p": "Jev"})
    for label, arm in arms_raw.items():
        m = m.merge(arm.rename(columns={"p": label})[["Crash_ID", "variable", label]],
                    on=["Crash_ID", "variable"], how="inner")
    names = ["Jev"] + list(arms_raw)
    y = m.y.to_numpy(float)
    w = m.ht_weight.to_numpy(float)
    cols = {k: m[k].to_numpy(float) for k in names}

    def f1(p, y, w) -> float:
        pr = p > 0.5
        tp = w[pr & (y == 1)].sum()
        fp = w[pr & (y == 0)].sum()
        fn = w[~pr & (y == 1)].sum()
        return 0.0 if tp == 0 else float(2 * tp / (2 * tp + fp + fn))

    rng = np.random.default_rng(seed)
    idx = np.arange(len(m))
    point = {k: f1(v, y, w) for k, v in cols.items()}
    draws = {k: [] for k in names}
    for _ in range(B):
        s_ = rng.choice(idx, len(idx), replace=True)
        for k, v in cols.items():
            draws[k].append(f1(v[s_], y[s_], w[s_]))
    draws = {k: np.array(v) for k, v in draws.items()}

    out = {"n_paired": int(len(m)), "B": B,
           "f1": {k: {"point": point[k],
                      "ci95": [float(np.percentile(draws[k], 2.5)),
                               float(np.percentile(draws[k], 97.5))]} for k in names},
           "differences": {}}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = draws[a] - draws[b]
            out["differences"][f"{a} - {b}"] = {
                "point": point[a] - point[b],
                "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "p_a_better": float(np.mean(d > 0)),
                # The interval straddling zero is the claim worth making when it happens,
                # so it is recorded rather than left for the reader to infer from the bounds.
                "indistinguishable": bool(np.percentile(d, 2.5) < 0 < np.percentile(d, 97.5)),
            }
    return out


def costs() -> dict:
    """Analytic cost per 1,000 narratives. NOT measured by the accuracy harness."""
    cm = json.loads((DATA / "cost_model.json").read_text())
    tok = cm["per_question_set"]["27"]["mean_tokens"]
    out = {"basis": ("measured input tokens for the 27-question schema times published list "
                     "prices; output tokens assumed at 60 per narrative for the generative "
                     "arm, which must emit JSON"),
           "input_tokens_per_narrative": tok,
           "jev_per_1k_usd": P.jev_cost_per_1k(tok),
           "jev_price_accessed": "2026-09-17"}
    for name in P.CLAUDE:
        out[f"{name}_per_1k_usd"] = P.llm_cost_per_1k(tok, model=name)
    return out


def main() -> None:
    gold = json.loads((GOLD / "gold_analysis.json").read_text())
    arms: dict[str, dict] = {}
    raw: dict[str, pd.DataFrame] = {}
    for prefix, label in (("fable", "claude-fable-5-1"), ("gpt", "gpt-5.6-sol")):
        a = load_arm(prefix)
        if a is None:
            print(f"{prefix}: no output found, arm skipped")
            continue
        raw[label] = a
        arms[label] = arm_metrics(join_gold(a))

    out = {
        "reference": "human gold set (majority of 2-3 independent coders), §4.4",
        "note": ("Accuracy is measured by running each model on the same narratives, criteria "
                 "and variable assignment as Jev. Cost is NOT measured here: it is computed "
                 "analytically from measured token counts times published list prices."),
        "scored_on": ("the gold set, which is also the set the Table 11 recalibration maps "
                      "were fitted on; neither arm is held out from it"),
        "jev": {"pooled": gold["pooled"]},
        "arms": arms,
        "cost": costs(),
        "paired": paired_comparison(raw) if raw else {},
    }
    (FRONT / "frontier_analysis.json").write_text(json.dumps(out, indent=1, default=float))

    J = gold["pooled"]
    print(f"\n{'model':22s}{'n':>6s}{'prec':>7s}{'rec':>7s}{'F1':>7s}{'kappa':>8s}"
          f"{'ECE':>8s}{'Brier':>8s}")
    print(f"{'Jev 1.13':22s}{J['n']:>6d}{J['precision']:>7.3f}{J['recall']:>7.3f}"
          f"{J['f1']:>7.3f}{J['cohens_kappa']:>8.3f}{J['ece']:>8.4f}{J['brier']:>8.4f}")
    for label, a in arms.items():
        p = a["pooled"]
        print(f"{label:22s}{p['n']:>6d}{p['precision']:>7.3f}{p['recall']:>7.3f}"
              f"{p['f1']:>7.3f}{p['cohens_kappa']:>8.3f}{p['ece']:>8.4f}{p['brier']:>8.4f}")
        print(f"{'':22s}coverage {100*a['coverage_of_gold_labels']:.1f}% | "
              f"{a['distinct_p_values']} distinct p | label/p disagreement "
              f"{100*a['label_probability_disagreement']:.1f}%")
    if out["paired"]:
        print(f"\npaired F1 differences (n={out['paired']['n_paired']:,}, "
              f"B={out['paired']['B']}):")
        for k, v in out["paired"]["differences"].items():
            flag = "  INDISTINGUISHABLE" if v["indistinguishable"] else ""
            print(f"  {k:28s} {v['point']:+.3f}  "
                  f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]{flag}")
    c = out["cost"]
    print(f"\ncost per 1k narratives: Jev ${c['jev_per_1k_usd']:.4f}", end="")
    for name in P.CLAUDE:
        print(f" | {name} ${c[f'{name}_per_1k_usd']:.4f}", end="")
    print(f"\n\nwrote {FRONT/'frontier_analysis.json'}")


if __name__ == "__main__":
    main()
