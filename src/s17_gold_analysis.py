"""§4.5 metrics against the HUMAN reference set -- the accuracy reference the paper promised.

Everything in s10_analysis is measured against coded CRIS fields and is therefore AGREEMENT.
This script uses the human labels instead, so these are accuracy statistics.

Four things matter here and are easy to get wrong.

  1. The reference set is a calibrated sample, not a self-weighting one. The frame was drawn
     by a greedy cell-filling rule over (variable, bin) cells, so high-probability narratives
     are deliberately over-represented. Every rate is weighted by the calibration weight that
     `s05b_design_weights.py` computes, which carries both the narrative's base weight and the
     probability that the variable was assigned to that narrative. The unweighted precision of
     a variable in this sample is a property of the sample, not of Texas.
  2. The resampling unit is the narrative, not the pair. One narrative supplies up to sixteen
     judgments and they move together, so a row bootstrap reports an interval that is too
     narrow by the amount of that dependence.
  3. Recalibration and threshold choice are evaluated out of fold, and the fold boundary is
     the narrative. Splitting pairs would put the same narrative on both sides of the split
     and leak.
  4. Ties and disputes are NOT evidence. Pairs the coders left `unclear` or disagreed on are
     dropped from the rate calculations and reported separately, because scoring the model
     against a label the humans could not agree on measures nothing. The sensitivity of every
     headline number to that exclusion is computed rather than assumed away.

Output: data/gold/gold_analysis.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

import s07_metrics as M

ROOT = Path(__file__).resolve().parents[2]   # project root, relative to this file
DATA = ROOT / "paper1" / "data"
GOLD = DATA / "gold"
B_BOOT = 1000
B_BOOT_SMALL = 400          # for the per-variable blocks, where 16 variables share the budget
MIN_N = 40
STRATUM = "draw_stratum"    # the cell the greedy rule drew the narrative to fill
WCOL = "pair_weight"        # the calibration weight of WP1


def wcol(d: pd.DataFrame) -> np.ndarray:
    return d[WCOL].to_numpy(float)


def wrate(mask: np.ndarray, w: np.ndarray) -> float:
    """Hajek ratio: the share of the POPULATION the sample represents."""
    return float(w[mask].sum() / w.sum()) if w.sum() > 0 else float("nan")


def _core(d: pd.DataFrame) -> dict:
    """The statistics that the cluster bootstrap has to recompute on every replicate."""
    p = d.p.to_numpy(float)
    y = d.y.to_numpy(float)
    w = wcol(d)
    pred = p > 0.5
    yb = y == 1
    tp, fp = w[pred & yb].sum(), w[pred & ~yb].sum()
    fn = w[~pred & yb].sum()
    prec = float(tp / (tp + fp)) if tp + fp > 0 else float("nan")
    rec = float(tp / (tp + fn)) if tp + fn > 0 else float("nan")
    return {
        "base_rate_weighted": wrate(yb, w),
        "flagged_rate_weighted": wrate(pred, w),
        "precision": prec,
        "recall": rec,
        "f1": float(2 * prec * rec / (prec + rec)) if prec + rec > 0 else float("nan"),
        "accuracy_weighted": float(w[pred == yb].sum() / w.sum()),
        "cohens_kappa_weighted": M.cohen_kappa(pred.astype(int), yb.astype(int), w),
        "ece": M.ece_discrete(p, y, w),
        "brier": M.brier(p, y, w),
        "expected_p_weighted": float(np.average(p, weights=w)),
    }


def block(d: pd.DataFrame, B: int = B_BOOT) -> dict:
    p = d.p.to_numpy(float)
    y = d.y.to_numpy(float)
    w = wcol(d)
    pred = p > 0.5
    yb = y == 1

    sl = M.calibration_slope(p, y, w)
    rf = M.resolution_floor_report(p, y, w)
    kap = M.noul_confidence(p)
    correct = (pred == yb).astype(float)

    ci = M.cluster_bootstrap(_core, d, strata=STRATUM, B=B, seed=42)
    point = _core(d)
    out = {
        "n": int(len(d)),
        "n_narratives": int(d.Crash_ID.nunique()),
        "n_effective": float(w.sum() ** 2 / np.sum(w ** 2)),
        **point,
        "ci95": {k: [ci[k]["lo"], ci[k]["hi"]] for k in point},
        "ci95_raw": {k: [ci[k].get("lo_raw"), ci[k].get("hi_raw")] for k in point},
        # kept under its own key so the change of estimator is visible in the audit table
        "cohens_kappa_unweighted": M.cohen_kappa(pred.astype(int), yb.astype(int)),
        "brier_skill_score": M.brier_decomposition(p, y, w)["brier_skill_score"],
        "calibration_slope": sl["slope_reportable"],
        "calibration_slope_raw": sl["slope"],
        "separated": sl["separated"],
        "calibration_intercept": sl["intercept"],
        "spiegelhalter_z": M.spiegelhalter_z(p, y, w)["z"],
        "mean_p_minus_prevalence": float(np.average(p, weights=w) - wrate(yb, w)),
        "base_rate_below_grid_floor": rf["base_rate_below_floor"],
        "aurc": M.aurc(kap, correct, w),
        "excess_aurc": M.excess_aurc(kap, correct, w),
        "review_budget_90_insample": M.positive_review_budget(p, y, 0.90, w)[
            "review_budget_of_flagged"],
        "review_budget_95_insample": M.positive_review_budget(p, y, 0.95, w)[
            "review_budget_of_flagged"],
    }
    # Aliases, so callers written against the old key names keep working. `cohens_kappa` is
    # now the WEIGHTED statistic, which is the one the tables and the text must label as
    # such; the unweighted value stays available for the audit record.
    out["ece_ci95"] = out["ci95"]["ece"]
    out["cohens_kappa"] = out["cohens_kappa_weighted"]
    return out


# --------------------------------------------------------------- WP2, Spiegelhalter p-value
def spiegelhalter_pvalue(d: pd.DataFrame, B: int = B_BOOT, seed: int = 7) -> dict:
    """A cluster-bootstrap p-value for the Spiegelhalter statistic.

    The statistic's reference distribution is standard normal only for independent
    observations with known probabilities. The reference set is weighted and clustered, so
    the null is obtained by resampling narratives instead. The sign is not interpreted;
    the mean probability minus the prevalence is reported beside it, which is the comparison
    Appendix D justifies.
    """
    def stat(x):
        return M.spiegelhalter_z(x.p.to_numpy(float), x.y.to_numpy(float), wcol(x))["z"]

    r = M.cluster_bootstrap(stat, d, strata=STRATUM, B=B, seed=seed)
    z0 = r["point"]
    se = r.get("se", float("nan"))
    # a two-sided p-value from the bootstrap spread, centred on the point estimate
    p = float(2 * (1 - M.stats.norm.cdf(abs(z0) / se))) if se and se > 0 else float("nan")
    return {"z": z0, "bootstrap_se": se, "p_value_cluster": p,
            "ci95": [r["lo"], r["hi"]], "n_clusters": r["n_clusters"]}


# --------------------------------------------------------------- WP3, recalibration
RECAL_REPEATS = 50
RECAL_SEEDS = list(range(RECAL_REPEATS))


def _fit_platt(P, Y, W):
    from sklearn.linear_model import LogisticRegression
    # Two parameters, intercept and slope, and no penalty. The default L2 penalty makes the
    # map a regularised fit whose slope is shrunk toward zero, which is not the Platt map the
    # manuscript names and not the correction the deployment argument needs. `C=inf` is the
    # spelling scikit-learn 1.8 and later ask for; `penalty=None` is the older one and is
    # kept as the fallback so the script runs on either.
    X, W = M._logit(P).reshape(-1, 1), W
    try:
        return LogisticRegression(C=np.inf).fit(X, Y, sample_weight=W)
    except (TypeError, ValueError):
        return LogisticRegression(penalty=None).fit(X, Y, sample_weight=W)


def _apply_platt(m, x):
    return m.predict_proba(M._logit(x).reshape(-1, 1))[:, 1]


def _fit_isotonic(P, Y, W):
    from sklearn.isotonic import IsotonicRegression
    return IsotonicRegression(out_of_bounds="clip").fit(P, Y, sample_weight=W)


METHODS = {
    "none": (lambda *_: None, lambda m, x: x),
    "platt": (_fit_platt, _apply_platt),
    "isotonic": (_fit_isotonic, lambda m, x: m.predict(x)),
}


def _held_out_maps(d: pd.DataFrame, seeds=RECAL_SEEDS) -> dict:
    """Out-of-fold recalibrated probabilities, averaged over repeated narrative splits.

    The split is over narratives, so no narrative contributes to both the map and the score.
    Each repeat splits the narratives in half, fits on one half, scores the other, and then
    swaps, so every pair is scored out of fold in every repeat.
    """
    ids = d.Crash_ID.unique()
    p = d.p.to_numpy(float)
    y = d.y.to_numpy(float)
    w = wcol(d)
    acc = {k: np.zeros(len(d)) for k in METHODS}
    n_ok = {k: np.zeros(len(d)) for k in METHODS}
    for s in seeds:
        perm = np.random.default_rng(s).permutation(ids)
        half = set(perm[: len(perm) // 2])
        a = d.Crash_ID.isin(half).to_numpy()
        for tr, te in ((a, ~a), (~a, a)):
            if tr.sum() < 10 or te.sum() < 1 or len(np.unique(y[tr])) < 2:
                continue
            for k, (fit, apply) in METHODS.items():
                try:
                    m = fit(p[tr], y[tr], w[tr])
                    acc[k][te] += apply(m, p[te])
                    n_ok[k][te] += 1
                except Exception:
                    pass
    return {k: np.divide(acc[k], n_ok[k], out=np.full(len(d), np.nan),
                         where=n_ok[k] > 0) for k in METHODS}


def _recal_scores(d: pd.DataFrame) -> dict:
    """Held-out calibration error, Brier score, intercept and slope for each map."""
    q = _held_out_maps(d)
    y = d.y.to_numpy(float)
    w = wcol(d)
    out = {}
    for k, qq in q.items():
        ok = np.isfinite(qq)
        if ok.sum() < 10:
            continue
        sl = M.calibration_slope(qq[ok], y[ok], w[ok])
        out[f"{k}_ece"] = M.ece_discrete(qq[ok], y[ok], w[ok])
        out[f"{k}_brier"] = M.brier(qq[ok], y[ok], w[ok])
        out[f"{k}_slope"] = sl["slope_reportable"]
        out[f"{k}_intercept"] = sl["intercept"]
    return out


def recalibration(d: pd.DataFrame, B: int = 200) -> dict:
    """WP3. Pooled and per-variable recalibration, held out over narrative splits.

    The pooled map is one map over all variables and is a proof of concept: it shows that a
    few thousand judgments carry enough signal to move the aggregate calibration error. The
    per-variable maps are the deployment evidence, because an analyst applies a map to one
    variable at a time. Variables with fewer than MIN_POSITIVES positives cannot support their
    own map and are reported as pooled-only.
    """
    MIN_POSITIVES = 20
    pooled_point = _recal_scores(d)
    pooled_ci = M.cluster_bootstrap(_recal_scores, d, strata=STRATUM, B=B, seed=11)

    per, pooled_only = {}, []
    for v, sub in d.groupby("variable"):
        n_pos = int((sub.y == 1).sum())
        if len(sub) < MIN_N:
            continue
        if n_pos < MIN_POSITIVES:
            pooled_only.append(v)
            continue
        sc = _recal_scores(sub)
        if sc:
            per[v] = {"n": int(len(sub)), "n_positive": n_pos, **sc}

    return {
        "method": ("repeated half-split over NARRATIVES, fitted on one half and scored on the "
                   "other, both directions, averaged over 50 seeds"),
        "n_repeats": RECAL_REPEATS,
        "platt_penalty": "none (two parameters, intercept and slope)",
        "isotonic_out_of_bounds": "clip",
        "min_positives_for_own_map": MIN_POSITIVES,
        "pooled": pooled_point,
        "pooled_ci95": {k: [pooled_ci[k]["lo"], pooled_ci[k]["hi"]]
                        for k in pooled_point if k in pooled_ci},
        "per_variable": per,
        "pooled_only_variables": sorted(pooled_only),
        # the shape the old JSON had, so the tables and macros keep a stable path
        "none": {"ece": pooled_point.get("none_ece"), "brier": pooled_point.get("none_brier")},
        "platt": {"ece": pooled_point.get("platt_ece"),
                  "brier": pooled_point.get("platt_brier")},
        "isotonic": {"ece": pooled_point.get("isotonic_ece"),
                     "brier": pooled_point.get("isotonic_brier")},
    }


# --------------------------------------------------------------- WP4, held-out review budget
BUDGET_GRID = np.round(np.arange(0.50, 1.0001, 0.01), 4)
BUDGET_REPEATS = 50


def _budget_on(p, y, w, target: float) -> tuple:
    """Largest accepted set on the grid that meets `target` precision, and its threshold.

    Convention, stated in Eq. (budget): an accepted set is the flagged positives at or above a
    grid threshold. If no threshold meets the target the accepted set is empty, coverage is
    zero and the review budget H is 1. Ties on the grid are not split, because a threshold
    cannot separate two records that carry the same probability.
    """
    m = p > 0.5
    if not m.any():
        return 0.0, float("nan"), float("nan"), 0.0
    pp, yy, ww = p[m], y[m], w[m]
    tot = ww.sum()
    best = (0.0, float("nan"), float("nan"), 0.0)
    for t in BUDGET_GRID:
        k = pp >= t
        if not k.any():
            continue
        acc = ww[k].sum()
        prec = float(ww[k & (yy == 1)].sum() / acc)
        if prec >= target and acc / tot > best[0]:
            best = (float(acc / tot), float(t), prec, float(acc))
    return best


def held_out_budget(d: pd.DataFrame, target: float = 0.90,
                    repeats: int = BUDGET_REPEATS) -> dict:
    """WP4. Cross-fitted review budget with a held-out precision and a lower bound.

    The threshold is chosen on one half of the narratives and both the coverage it buys and
    the precision it actually delivers are measured on the other half, in both directions and
    over repeated splits. Selecting and certifying on the same labels reports the best of many
    thresholds as if it were the value of one, which is the finding this answers.
    """
    ids = d.Crash_ID.unique()
    p = d.p.to_numpy(float)
    y = d.y.to_numpy(float)
    w = wcol(d)
    covs, precs, ths, n_acc, n_inf = [], [], [], [], 0
    for s in range(repeats):
        perm = np.random.default_rng(1000 + s).permutation(ids)
        half = set(perm[: len(perm) // 2])
        a = d.Crash_ID.isin(half).to_numpy()
        for tr, te in ((a, ~a), (~a, a)):
            if tr.sum() < 10 or te.sum() < 10:
                continue
            _, t, _, _ = _budget_on(p[tr], y[tr], w[tr], target)
            if not np.isfinite(t):
                covs.append(0.0); precs.append(np.nan); n_inf += 1
                continue
            m = p[te] > 0.5
            if not m.any():
                continue
            pp, yy, ww = p[te][m], y[te][m], w[te][m]
            k = pp >= t
            acc = ww[k].sum()
            covs.append(float(acc / ww.sum()) if ww.sum() > 0 else 0.0)
            precs.append(float(ww[k & (yy == 1)].sum() / acc) if acc > 0 else np.nan)
            ths.append(float(t))
            n_acc.append(float(ww[k].sum() ** 2 / np.sum(ww[k] ** 2)) if acc > 0 else 0.0)

    cov = np.array(covs, dtype=float)
    pr = np.array(precs, dtype=float)
    prf = pr[np.isfinite(pr)]
    ins_cov, ins_t, ins_prec, _ = _budget_on(p, y, w, target)

    # a lower confidence bound on the delivered precision, from the Clopper-Pearson interval
    # at Kish's effective accepted count. It is approximate because the effective count is
    # itself estimated from the weights.
    lo = float("nan")
    if prf.size and n_acc:
        n_eff = float(np.mean([x for x in n_acc if x > 0])) if any(n_acc) else 0.0
        mean_prec = float(prf.mean())
        if n_eff >= 1:
            lo = M.clopper_pearson(int(round(mean_prec * n_eff)), int(round(n_eff)))[0]
    return {
        "target_precision": target,
        "n_splits": int(len(cov)),
        "n_splits_with_no_feasible_threshold": int(n_inf),
        "coverage_heldout_mean": float(cov.mean()) if cov.size else float("nan"),
        "coverage_heldout_sd": float(cov.std()) if cov.size else float("nan"),
        "review_budget_heldout_mean": float(1 - cov.mean()) if cov.size else float("nan"),
        "review_budget_heldout_p90": float(1 - np.percentile(cov, 10)) if cov.size else float("nan"),
        "precision_heldout_mean": float(prf.mean()) if prf.size else float("nan"),
        "precision_heldout_lcb95": lo,
        "precision_lcb_note": ("Clopper-Pearson at Kish's effective accepted count, "
                               "approximate because that count is estimated"),
        "threshold_median": float(np.median(ths)) if ths else float("nan"),
        "threshold_iqr": [float(np.percentile(ths, 25)), float(np.percentile(ths, 75))]
                         if ths else [float("nan"), float("nan")],
        "coverage_insample": ins_cov,
        "review_budget_insample": float(1 - ins_cov),
        "precision_insample": ins_prec,
        "threshold_insample": ins_t,
        "optimism": float(ins_cov - cov.mean()) if cov.size else float("nan"),
    }


# --------------------------------------------------------------- WP5, exclusion sensitivity
def exclusion_sensitivity(g: pd.DataFrame, usable: pd.DataFrame) -> dict:
    """WP5. What the headline numbers would be under three treatments of the excluded pairs.

    The 48 disputed and 1 unclear pairs are excluded from every rate, so every accuracy
    statement in the paper is conditional on pairs the coders found clear. The three
    treatments bound what that conditioning can be worth: counting every excluded pair as a
    model error, counting every one as model-correct, and routing all of them to review
    whatever the probability says.
    """
    ex = g[g.y.isna() & g.p.notna()].copy()
    out = {"n_excluded": int(len(ex)),
           "n_disputed": int((g.label == "DISPUTED").sum()),
           "n_unclear": int((g.label == "unclear").sum()),
           "by_variable": {k: int(v) for k, v in ex.variable.value_counts().items()},
           "by_bin": {str(k): int(v) for k, v in ex.bin.value_counts().sort_index().items()}}

    base = _core(usable)
    out["excluded"] = {"f1": base["f1"], "ece": base["ece"],
                       "review_budget_90": M.positive_review_budget(
                           usable.p.to_numpy(float), usable.y.to_numpy(float), 0.90,
                           wcol(usable))["review_budget_of_flagged"]}

    for name, rule in (("as_model_error", "error"), ("as_model_correct", "correct")):
        e = ex.copy()
        pred = (e.p > 0.5).astype(float)
        e["y"] = (1 - pred) if rule == "error" else pred
        d2 = pd.concat([usable, e], ignore_index=True)
        c = _core(d2)
        out[name] = {"f1": c["f1"], "ece": c["ece"],
                     "review_budget_90": M.positive_review_budget(
                         d2.p.to_numpy(float), d2.y.to_numpy(float), 0.90,
                         wcol(d2))["review_budget_of_flagged"]}

    # routed to review: the excluded pairs never enter the accepted set, so they enlarge the
    # review budget by their weighted share of the flagged positives and leave F1 undefined
    fl = usable[usable.p > 0.5]
    exf = ex[ex.p > 0.5]
    share = float(exf[WCOL].sum() / (fl[WCOL].sum() + exf[WCOL].sum())) if len(fl) else float("nan")
    b90 = out["excluded"]["review_budget_90"]
    out["as_routed_to_review"] = {
        "f1": None, "ece": None,
        "excluded_share_of_flagged": share,
        "review_budget_90": float(b90 * (1 - share) + share)}
    return out


# --------------------------------------------------------------- narrative-only confirmation
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
        w = wcol(sub)
        y = (sub.y == 1).to_numpy()
        rate = float(w[y].sum() / w.sum())
        # The frame is weighted and clustered, so a Clopper-Pearson interval on the raw count
        # would be too narrow. Kish's effective n is the denominator, and the interval is
        # labelled approximate for that reason.
        n_eff = float(w.sum() ** 2 / np.sum(w ** 2))
        lo, hi = M.clopper_pearson(int(round(rate * n_eff)), int(round(n_eff)))
        return {"n": int(len(sub)), "n_narratives": int(sub.Crash_ID.nunique()),
                "n_eff": n_eff, "confirmed_rate": rate, "ci95": [lo, hi],
                "ci95_note": "Clopper-Pearson at Kish's effective n, approximate"}

    flagged = m[m.p > 0.5]
    no = flagged[~flagged.coded]
    weak = {v for v, r in per_variable_f1(d).items() if r < 0.70}
    return {
        "definition": "model flags (p>0.5) that the coded field records as negative",
        "n_variables": int(m.variable.nunique()),
        "narrative_only_share_of_flags":
            float(no[WCOL].sum() / flagged[WCOL].sum()) if len(flagged) else None,
        "all_flags": cell(flagged),
        "narrative_only": cell(no),
        "coded_confirmed_flags": cell(flagged[flagged.coded]),
        "narrative_only_strong_vars": cell(no[~no.variable.isin(weak)]),
        "narrative_only_weak_vars": cell(no[no.variable.isin(weak)]),
        "weak_vars": sorted(weak),
        "per_variable": {v: cell(s) for v, s in no.groupby("variable")},
    }


def per_variable_f1(d: pd.DataFrame) -> dict:
    return {v: _core(s)["f1"] for v, s in d.groupby("variable") if len(s) >= MIN_N}


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
    analyst applies without a reference set. But Section 5.4 finds the probabilities sit too
    high overall, which makes 0.5 too permissive for rare variables, so some of what looks
    like a model failure may be a threshold failure. The two are distinguishable: a threshold
    failure leaves the RANKING intact and is fixed for free, a model failure does not.

    Tuning a threshold on the same labels used to score it inflates F1, badly so when the
    positives are few. Every number here is therefore out of fold, and the fold boundary is
    the narrative rather than the pair, so a narrative cannot inform the threshold that scores
    it. The spread across repeats is reported because it is the warning about when this
    procedure should not be trusted: a variable with a handful of positives shows a wide
    spread and a tuned threshold that will not transport.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for v, sub in d.groupby("variable"):
        if len(sub) < MIN_N:
            continue
        p = sub.p.to_numpy(float)
        y = sub.y.to_numpy(float)
        w = wcol(sub)
        ids = sub.Crash_ID.unique()
        n_pos = int((y == 1).sum())
        base = _wf1(p, y, w, 0.5)

        scores, chosen = [], []
        for _ in range(THRESH_REPEATS):
            perm = rng.permutation(ids)
            half = set(perm[: len(perm) // 2])
            a = sub.Crash_ID.isin(half).to_numpy()
            for tr, te in ((a, ~a), (~a, a)):
                if (y[tr] == 1).sum() == 0 or (y[te] == 1).sum() == 0:
                    continue
                t = max(THRESH_GRID, key=lambda t: _wf1(p[tr], y[tr], w[tr], t)[0])
                scores.append(_wf1(p[te], y[te], w[te], t))
                chosen.append(float(t))
        if not scores:
            continue
        f1s = np.array([x[0] for x in scores])
        ch = np.array(chosen)
        # the threshold we would actually ship: fitted on all the reference data we have
        t_star = float(max(THRESH_GRID, key=lambda t: _wf1(p, y, w, t)[0]))
        out[v] = {
            "n_positive": n_pos,
            "f1_at_half": base[0], "precision_at_half": base[1], "recall_at_half": base[2],
            "tuned_threshold": t_star,
            "threshold_median": float(np.median(ch)),
            "threshold_iqr": [float(np.percentile(ch, 25)), float(np.percentile(ch, 75))],
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
    if WCOL not in g.columns:
        raise SystemExit(f"{WCOL} missing from gold_labels.csv -- run s05b_design_weights.py "
                         "then s16_gold_ingest.py")
    n_all = len(g)
    n_unclear = int((g.label == "unclear").sum())
    n_disp = int((g.label == "DISPUTED").sum())
    d = g[g.y.notna() & g.p.notna()].copy()
    print(f"{n_all:,} labelled pairs: {len(d):,} usable, {n_unclear} unclear, "
          f"{n_disp} disputed (both excluded from rates)")
    w = wcol(d)
    print(f"calibration weights: {w.min():.1f} to {w.max():.1f}, "
          f"effective n {w.sum()**2/np.sum(w**2):,.0f} over {d.Crash_ID.nunique()} narratives")

    per = {}
    for v, sub in d.groupby("variable"):
        if len(sub) >= MIN_N:
            per[v] = block(sub, B=B_BOOT_SMALL)
            per[v]["held_out_budget_90"] = held_out_budget(sub, 0.90)
            per[v]["held_out_budget_95"] = held_out_budget(sub, 0.95)

    pooled = block(d)
    pooled["n_variables"] = len(per)
    pooled["held_out_budget_90"] = held_out_budget(d, 0.90)
    pooled["held_out_budget_95"] = held_out_budget(d, 0.95)
    pooled["spiegelhalter"] = spiegelhalter_pvalue(d)

    design = json.loads((GOLD / "gold_design_report.json").read_text())
    out = {
        "reference": "human reference set (majority of 2-3 independent coders), §4.4",
        "note": ("All rates are calibration-weighted to the Stage-2 random stratum with the "
                 "weights of s05b_design_weights.py, which carry the narrative's base weight "
                 "and the variable-assignment probability. The estimator is a Hajek ratio. "
                 "Intervals resample NARRATIVES within the draw route, not rows. Unclear and "
                 "disputed pairs are excluded from the rates and their effect is reported "
                 "under exclusion_sensitivity."),
        "design": design,
        "n_pairs_labelled": n_all, "n_usable": int(len(d)),
        "n_unclear": n_unclear, "n_disputed": n_disp,
        "pooled": pooled, "per_variable": per,
        "recalibration": recalibration(d),
        "exclusion_sensitivity": exclusion_sensitivity(g, d),
        "narrative_only": narrative_only(d),
        "threshold_tuning": threshold_tuning(d),
    }
    (GOLD / "gold_analysis.json").write_text(json.dumps(out, indent=1, default=float))

    print(f"\n{'variable':18s}{'n':>5s}{'base%':>7s}{'prec':>6s}{'rec':>6s}{'F1':>6s}"
          f"{'kappa':>7s}{'ECE':>7s}{'slope':>8s}{'H90 oof':>9s}")
    for v, r in sorted(per.items(), key=lambda kv: -(kv[1]["f1"] or 0)):
        sl = "sep." if r["separated"] else f"{r['calibration_slope']:.2f}"
        print(f"{v:18s}{r['n']:>5d}{100*r['base_rate_weighted']:>7.2f}"
              f"{r['precision']:>6.2f}{r['recall']:>6.2f}{r['f1']:>6.2f}"
              f"{r['cohens_kappa_weighted']:>7.3f}{r['ece']:>7.4f}{sl:>8s}"
              f"{r['held_out_budget_90']['review_budget_heldout_mean']:>9.2f}")
    P = pooled
    ci = P["ci95"]
    print(f"\nPOOLED  n={P['n']:,}  precision {P['precision']:.3f}  recall {P['recall']:.3f}  "
          f"F1 {P['f1']:.3f} [{ci['f1'][0]:.3f}, {ci['f1'][1]:.3f}]  "
          f"kappa {P['cohens_kappa_weighted']:.3f}")
    print(f"        ECE {P['ece']:.4f} [{ci['ece'][0]:.4f}, {ci['ece'][1]:.4f}]  "
          f"Brier {P['brier']:.4f}  BSS {P['brier_skill_score']:.3f}")
    print(f"        slope {P['calibration_slope']:.2f}  intercept {P['calibration_intercept']:.2f}"
          f"  Spiegelhalter z {P['spiegelhalter_z']:.1f} "
          f"(cluster p {P['spiegelhalter']['p_value_cluster']:.3g})")
    print(f"        E[p] {100*P['expected_p_weighted']:.2f}%  vs base rate "
          f"{100*P['base_rate_weighted']:.2f}%")
    R = out["recalibration"]["pooled"]
    print(f"        recalibration held-out ECE: none {R.get('none_ece', float('nan')):.4f}  "
          f"platt {R.get('platt_ece', float('nan')):.4f}  "
          f"isotonic {R.get('isotonic_ece', float('nan')):.4f}")
    B = P["held_out_budget_90"]
    print(f"        review budget at 90% precision: held out {B['review_budget_heldout_mean']:.3f}"
          f"  in sample {B['review_budget_insample']:.3f}  optimism {B['optimism']:.3f}")
    print(f"\nwrote {GOLD/'gold_analysis.json'}")


if __name__ == "__main__":
    main()
