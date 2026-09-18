"""§4.5 metrics against the HUMAN gold set — the accuracy reference the paper promised.

Everything in s10_analysis is measured against coded CRIS fields and is therefore AGREEMENT.
This script uses the human labels instead, so these are accuracy statistics.

Two things matter here and are easy to get wrong:

  1. The gold frame is probability-stratified — high-probability narratives are deliberately
     over-sampled so the middle bins are estimable. Every rate therefore has to be
     Horvitz-Thompson weighted. The unweighted precision of a variable in this sample is a
     property of the sample, not of Texas, and quoting it would overstate performance on
     common variables and understate it on rare ones.
  2. Ties and disputes are NOT evidence. Pairs the coders left `unclear` or disagreed on are
     dropped from the rate calculations and reported separately, because scoring the model
     against a label the humans could not agree on measures nothing.

Output: data/gold/gold_analysis.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

import s07_metrics as M

ROOT = Path(r"D:/OneDrive - Texas State University/AIT/Papers/Jev")
DATA = ROOT / "paper1" / "data"
GOLD = DATA / "gold"
B_BOOT = 1000
MIN_N = 40


def wrate(mask: np.ndarray, w: np.ndarray) -> float:
    """Horvitz-Thompson (Hajek) rate: the share of the POPULATION the sample represents."""
    return float(w[mask].sum() / w.sum()) if w.sum() > 0 else float("nan")


def block(d: pd.DataFrame) -> dict:
    p = d.p.to_numpy(float)
    y = d.y.to_numpy(float)
    w = d.ht_weight.to_numpy(float)
    pred = (p > 0.5)
    yb = y == 1

    # weighted confusion counts
    tp, fp = w[pred & yb].sum(), w[pred & ~yb].sum()
    fn, tn = w[~pred & yb].sum(), w[~pred & ~yb].sum()
    prec = float(tp / (tp + fp)) if tp + fp > 0 else float("nan")
    rec = float(tp / (tp + fn)) if tp + fn > 0 else float("nan")
    f1 = float(2 * prec * rec / (prec + rec)) if prec + rec > 0 else float("nan")

    sl = M.calibration_slope(p, y, w)
    strata = np.clip((p * 5).astype(int), 0, 4)
    ece_ci = M.weighted_bootstrap(lambda a, b, w=None: M.ece_discrete(a, b, w), p, y,
                                  w=w, strata=strata, B=B_BOOT, seed=42)
    rf = M.resolution_floor_report(p, y, w)
    kap = M.noul_confidence(p)
    correct = (pred == yb).astype(float)
    return {
        "n": int(len(d)),
        "base_rate_weighted": wrate(yb, w),
        "base_rate_unweighted": float(yb.mean()),
        "flagged_rate_weighted": wrate(pred, w),
        "precision": prec, "recall": rec, "f1": f1,
        "accuracy_weighted": float(w[pred == yb].sum() / w.sum()),
        "cohens_kappa": M.cohen_kappa(pred.astype(int), yb.astype(int)),
        "ece": M.ece_discrete(p, y, w),
        "ece_ci95": [ece_ci["lo"], ece_ci["hi"]],
        "brier": M.brier(p, y, w),
        "brier_skill_score": M.brier_decomposition(p, y, w)["brier_skill_score"],
        "calibration_slope": sl["slope_reportable"],
        "calibration_slope_raw": sl["slope"],
        "separated": sl["separated"],
        "calibration_intercept": sl["intercept"],
        "spiegelhalter_z": M.spiegelhalter_z(p, y, w)["z"],
        "expected_p_weighted": float(np.average(p, weights=w)),
        "base_rate_below_grid_floor": rf["base_rate_below_floor"],
        "aurc": M.aurc(kap, correct, w),
        "excess_aurc": M.excess_aurc(kap, correct, w),
        "review_budget_90": M.positive_review_budget(p, y, 0.90, w)["review_budget_of_flagged"],
        "review_budget_95": M.positive_review_budget(p, y, 0.95, w)["review_budget_of_flagged"],
    }


def recalibration(d: pd.DataFrame, seed: int = 42) -> dict:
    """§4.5 post-hoc recalibration, split-half validated.

    Fitted on one half of the gold set and scored on the other, both ways, so the reported
    numbers are out-of-fold. Reporting an in-sample recalibration would be circular: any
    flexible map can drive training-set ECE to zero.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    p_ = d.p.to_numpy(float); y = d.y.to_numpy(float); w = d.ht_weight.to_numpy(float)
    idx = np.random.default_rng(seed).permutation(len(d))
    folds = (idx[: len(d) // 2], idx[len(d) // 2:])

    def evaluate(fit, apply) -> dict:
        e, b = [], []
        for tr, te in (folds, folds[::-1]):
            m = fit(p_[tr], y[tr], w[tr])
            q = apply(m, p_[te])
            e.append(M.ece_discrete(q, y[te], w[te]))
            b.append(M.brier(q, y[te], w[te]))
        return {"ece": float(np.mean(e)), "brier": float(np.mean(b))}

    return {
        "method": "split-half, fitted on one half and scored on the other, both directions",
        "none": evaluate(lambda *_: None, lambda m, x: x),
        "platt": evaluate(
            lambda P, Y, W: LogisticRegression().fit(M._logit(P).reshape(-1, 1), Y,
                                                     sample_weight=W),
            lambda m, x: m.predict_proba(M._logit(x).reshape(-1, 1))[:, 1]),
        "isotonic": evaluate(
            lambda P, Y, W: IsotonicRegression(out_of_bounds="clip").fit(P, Y,
                                                                         sample_weight=W),
            lambda m, x: m.predict(x)),
    }


def narrative_only(d: pd.DataFrame) -> dict:
    """How often a NARRATIVE-ONLY flag is confirmed by a human.

    Section 5.6 shows the model flags these factors more often than the coded fields record
    them, and asks whether that gap is value added or model error. Pooled precision cannot
    answer it: false positives are concentrated in exactly the cell where the coded field
    disagrees, because coded-field agreement is itself evidence of correctness. So the
    quantity has to be measured in that cell -- model-flagged, coded field negative -- rather
    than inferred from the overall rate.
    """
    from s10_analysis import CODED_MAP

    c = pd.read_csv(DATA / "coded_join.csv")
    parts = []
    for v, col in CODED_MAP.items():
        if col not in c.columns:
            continue
        sub = d[d.variable == v].merge(c[["Crash_ID", col]], on="Crash_ID", how="inner")
        sub = sub[sub[col].notna()]
        if not sub.empty:
            parts.append(sub.assign(variable=v, coded=sub[col].astype(bool)))
    if not parts:
        return {}
    m = pd.concat(parts)

    def cell(sub: pd.DataFrame) -> dict:
        if sub.empty:
            return {}
        w = sub.ht_weight.to_numpy(float)
        y = (sub.y == 1).to_numpy()
        rate = float(w[y].sum() / w.sum())
        # The frame is weighted, so a Clopper-Pearson interval on the raw count would be too
        # narrow; Kish's effective n is the honest denominator here.
        n_eff = float(w.sum() ** 2 / np.sum(w ** 2))
        lo, hi = M.clopper_pearson(int(round(rate * n_eff)), int(round(n_eff)))
        return {"n": int(len(sub)), "n_eff": n_eff, "confirmed_rate": rate,
                "ci95": [lo, hi]}

    flagged = m[m.p > 0.5]
    no = flagged[~flagged.coded]
    weak = {v for v, r in per_variable_f1(d).items() if r < 0.70}
    return {
        "definition": "model flags (p>0.5) that the coded field records as negative",
        "n_variables": int(m.variable.nunique()),
        "narrative_only_share_of_flags":
            float(no.ht_weight.sum() / flagged.ht_weight.sum()) if len(flagged) else None,
        "all_flags": cell(flagged),
        "narrative_only": cell(no),
        "coded_confirmed_flags": cell(flagged[flagged.coded]),
        "narrative_only_strong_vars": cell(no[~no.variable.isin(weak)]),
        "narrative_only_weak_vars": cell(no[no.variable.isin(weak)]),
        "weak_vars": sorted(weak),
        "per_variable": {v: cell(s) for v, s in no.groupby("variable")},
    }


def per_variable_f1(d: pd.DataFrame) -> dict:
    return {v: block(s)["f1"] for v, s in d.groupby("variable") if len(s) >= MIN_N}


THRESH_GRID = np.arange(0.05, 0.96, 0.01)
THRESH_REPEATS = 200


def _wf1(p, y, w, t) -> tuple:
    pred = p > t
    tp = w[pred & (y == 1)].sum()
    fp = w[pred & (y == 0)].sum()
    fn = w[~pred & (y == 1)].sum()
    if tp == 0:
        return 0.0, 0.0, 0.0
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return float(2 * prec * rec / (prec + rec)), float(prec), float(rec)


def threshold_tuning(d: pd.DataFrame, seed: int = 0) -> dict:
    """Does the tau=0.5 default cost the weak variables their F1?

    Everything else in this paper thresholds at 0.5 because that is the decision rule an
    analyst applies without a gold set. But Section 5.4 finds the probabilities sit too high
    overall, which makes 0.5 too permissive for rare variables -- so some of what looks like
    a model failure may be a threshold failure. The two are distinguishable: a threshold
    failure leaves the RANKING intact and is fixed for free, a model failure does not.

    Tuning a threshold on the same labels used to score it inflates F1, badly so when the
    positives are few. Every number here is therefore out of fold: the threshold is chosen on
    one random half and scored on the other, both directions, repeated THRESH_REPEATS times.
    The spread across repeats is reported because it is the honest warning about when this
    procedure should not be trusted -- a variable with a handful of positives shows a wide
    spread and a tuned threshold that will not transport.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for v, sub in d.groupby("variable"):
        if len(sub) < MIN_N:
            continue
        p = sub.p.to_numpy(float)
        y = sub.y.to_numpy(float)
        w = sub.ht_weight.to_numpy(float)
        n_pos = int((y == 1).sum())
        base = _wf1(p, y, w, 0.5)

        scores = []
        for _ in range(THRESH_REPEATS):
            idx = rng.permutation(len(sub))
            halves = (idx[: len(sub) // 2], idx[len(sub) // 2:])
            for tr, te in (halves, halves[::-1]):
                if (y[tr] == 1).sum() == 0 or (y[te] == 1).sum() == 0:
                    continue
                t = max(THRESH_GRID, key=lambda t: _wf1(p[tr], y[tr], w[tr], t)[0])
                scores.append(_wf1(p[te], y[te], w[te], t))
        if not scores:
            continue
        f1s = np.array([x[0] for x in scores])
        # the threshold we would actually ship: fitted on all the gold data we have
        t_star = float(max(THRESH_GRID, key=lambda t: _wf1(p, y, w, t)[0]))
        out[v] = {
            "n_positive": n_pos,
            "f1_at_half": base[0], "precision_at_half": base[1], "recall_at_half": base[2],
            "tuned_threshold": t_star,
            "f1_tuned_oof": float(f1s.mean()),
            "f1_tuned_oof_sd": float(f1s.std()),
            "f1_tuned_oof_p10": float(np.percentile(f1s, 10)),
            "gain": float(f1s.mean() - base[0]),
            # A tuned threshold that is unstable across resamples is not a fix, it is an
            # overfit. Flagged rather than silently averaged away.
            "unstable": bool(f1s.std() > 0.10 or n_pos < 20),
        }
    return out


def main() -> None:
    g = pd.read_csv(GOLD / "gold_labels.csv")
    n_all = len(g)
    n_unclear = int((g.label == "unclear").sum())
    n_disp = int((g.label == "DISPUTED").sum())
    d = g[g.y.notna() & g.p.notna()].copy()
    print(f"{n_all:,} labelled pairs: {len(d):,} usable, {n_unclear} unclear, "
          f"{n_disp} disputed (both excluded from rates)")

    per = {}
    for v, sub in d.groupby("variable"):
        if len(sub) >= MIN_N:
            per[v] = block(sub)

    pooled = block(d)
    pooled["n_variables"] = len(per)

    out = {
        "reference": "human gold set (majority of 2-3 independent coders), §4.4",
        "note": ("All rates are Horvitz-Thompson weighted to the population; the gold frame "
                 "over-samples high-probability narratives by design, so unweighted rates "
                 "describe the sample, not Texas. Unclear and disputed pairs are excluded."),
        "n_pairs_labelled": n_all, "n_usable": int(len(d)),
        "n_unclear": n_unclear, "n_disputed": n_disp,
        "pooled": pooled, "per_variable": per,
        "recalibration": recalibration(d),
        "narrative_only": narrative_only(d),
        "threshold_tuning": threshold_tuning(d),
    }
    (GOLD / "gold_analysis.json").write_text(json.dumps(out, indent=1, default=float))

    print(f"\n{'variable':18s}{'n':>5s}{'base%':>7s}{'prec':>6s}{'rec':>6s}{'F1':>6s}"
          f"{'kappa':>7s}{'ECE':>7s}{'slope':>8s}")
    for v, r in sorted(per.items(), key=lambda kv: -(kv[1]["f1"] or 0)):
        sl = "sep." if r["separated"] else f"{r['calibration_slope']:.2f}"
        print(f"{v:18s}{r['n']:>5d}{100*r['base_rate_weighted']:>7.2f}"
              f"{r['precision']:>6.2f}{r['recall']:>6.2f}{r['f1']:>6.2f}"
              f"{r['cohens_kappa']:>7.3f}{r['ece']:>7.4f}{sl:>8s}")
    P = pooled
    print(f"\nPOOLED  n={P['n']:,}  precision {P['precision']:.3f}  recall {P['recall']:.3f}  "
          f"F1 {P['f1']:.3f}  kappa {P['cohens_kappa']:.3f}")
    print(f"        ECE {P['ece']:.4f} [{P['ece_ci95'][0]:.4f}, {P['ece_ci95'][1]:.4f}]  "
          f"Brier {P['brier']:.4f}  BSS {P['brier_skill_score']:.3f}")
    print(f"        slope {P['calibration_slope']:.2f}  intercept {P['calibration_intercept']:.2f}"
          f"  Spiegelhalter z {P['spiegelhalter_z']:.1f}")
    print(f"        E[p] {100*P['expected_p_weighted']:.2f}%  vs true base rate "
          f"{100*P['base_rate_weighted']:.2f}%")
    print(f"\nwrote {GOLD/'gold_analysis.json'}")


if __name__ == "__main__":
    main()
