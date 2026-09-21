"""Unit tests for s07_metrics, per §6 Step 6: "Unit-test each on synthetic data
(perfectly calibrated Bernoulli draws must give ECE ~ 0, slope ~ 1)."

Run: python -m pytest paper1/src/tests/test_metrics.py -q
 or: python paper1/src/tests/test_metrics.py   (no pytest dependency needed)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import s07_metrics as M  # noqa: E402

RNG = np.random.default_rng(0)


def calibrated(n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p).astype(float)
    return p, y


# --------------------------------------------------------------- calibration
def test_ece_near_zero_when_calibrated():
    p, y = calibrated()
    e = M.ece(p, y, n_bins=10)
    assert e < 0.01, f"ECE {e} should be ~0 for perfectly calibrated draws"


def test_ece_large_when_miscalibrated():
    p, y = calibrated()
    p_bad = np.clip(p + 0.25, 0, 1)          # systematically over-confident
    assert M.ece(p_bad, y) > 0.15


def test_mce_near_zero_when_calibrated():
    p, y = calibrated()
    assert M.mce(p, y, n_bins=10) < 0.02


def test_slope_near_one_when_calibrated():
    p, y = calibrated()
    r = M.calibration_slope(p, y)
    assert r["converged"]
    assert abs(r["slope"] - 1.0) < 0.05, r
    assert abs(r["intercept"]) < 0.05, r


def test_slope_below_one_when_overconfident():
    """Over-confident probabilities (pushed toward 0/1) give slope < 1."""
    p, y = calibrated()
    p_over = 1 / (1 + np.exp(-2.0 * M._logit(p)))   # double the logit
    r = M.calibration_slope(p_over, y)
    assert r["slope"] < 0.7, r


def test_spiegelhalter_z_is_standard_normal_when_calibrated():
    """z is a test statistic, so a single calibrated draw can land in the tail by chance
    (seed 0 gives z = 2.69). The property to test is its sampling distribution."""
    zs = np.array([M.spiegelhalter_z(*calibrated(n=100_000, seed=s))["z"] for s in range(12)])
    assert abs(np.median(zs)) < 1.0, zs
    assert np.median(np.abs(zs)) < 1.5, zs
    assert np.mean(np.abs(zs) > 3.0) < 0.2, zs


def test_spiegelhalter_z_significant_when_miscalibrated():
    p, y = calibrated()
    r = M.spiegelhalter_z(np.clip(p + 0.2, 0, 1), y)
    assert r["p_value"] < 1e-6, r


def test_separation_is_flagged_not_reported():
    """Perfectly separable data has no finite MLE slope. The estimator must say so rather
    than return whatever value the optimiser stopped at."""
    p = np.concatenate([np.full(300, 0.05), np.full(300, 0.95)])
    y = np.concatenate([np.zeros(300), np.ones(300)])
    r = M.calibration_slope(p, y)
    assert r["separated"], r
    assert r["slope_reportable"] is None


def test_non_separated_fit_is_reportable():
    p, y = calibrated(n=20_000)
    r = M.calibration_slope(p, y)
    assert not r["separated"], r
    assert r["slope_reportable"] is not None
    assert abs(r["slope_reportable"] - 1.0) < 0.1


def test_slope_has_no_overflow_warning_on_separable_data():
    """The stable sigmoid must not emit a RuntimeWarning on the inputs that separate."""
    import warnings
    p = np.concatenate([np.full(200, 0.02), np.full(200, 0.98)])
    y = np.concatenate([np.zeros(200), np.ones(200)])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        M.calibration_slope(p, y)


# --------------------------------------------------------------- scoring rules
def test_brier_decomposition_identity():
    p, y = calibrated()
    d = M.brier_decomposition(p, y, n_bins=20)
    # binned Brier = reliability - resolution + uncertainty, exactly
    assert abs(d["binned_brier"] - (d["reliability"] - d["resolution"] + d["uncertainty"])) < 1e-9
    # and the binned Brier tracks the raw Brier closely at 20 bins
    assert abs(d["binned_brier"] - d["brier"]) < 0.01, d
    assert d["reliability"] < 0.001, d          # calibrated => reliability ~ 0
    # uniform p with y~Bernoulli(p): Brier = E[p(1-p)] = 1/6, uncertainty = 1/4 => BSS = 1/3
    assert abs(d["brier_skill_score"] - 1/3) < 0.02, d


def test_brier_of_constant_prediction_equals_uncertainty():
    y = (RNG.uniform(size=50_000) < 0.3).astype(float)
    p = np.full_like(y, y.mean())
    d = M.brier_decomposition(p, y)
    assert abs(d["brier"] - d["uncertainty"]) < 1e-3, d
    assert abs(d["brier_skill_score"]) < 1e-2, d


# --------------------------------------------------------------- weights
def test_constant_weights_match_unweighted():
    p, y = calibrated(n=20_000)
    w = np.full_like(p, 7.3)
    # ECE is exact only up to float tie-boundary placement in the equal-mass cut search
    assert abs(M.ece(p, y) - M.ece(p, y, w)) < 1e-3
    assert abs(M.brier(p, y) - M.brier(p, y, w)) < 1e-12
    assert abs(M.calibration_slope(p, y)["slope"] - M.calibration_slope(p, y, w)["slope"]) < 1e-6


def test_ht_weights_recover_population_prevalence():
    """A probability-stratified sample with HT weights must recover the population rate."""
    rng = np.random.default_rng(3)
    n = 400_000
    p = rng.beta(0.4, 4.0, n)                  # rare-event-like landscape
    y = (rng.uniform(size=n) < p).astype(float)
    pop = y.mean()
    # oversample the high-probability tail 20x
    hi = p > 0.3
    incl = np.where(hi, 0.20, 0.01)
    drawn = rng.uniform(size=n) < incl
    w = 1.0 / incl[drawn]
    est = np.average(y[drawn], weights=w)          # Hajek ratio form of the HT estimator
    naive = y[drawn].mean()
    se = np.sqrt(np.sum(w**2 * (y[drawn] - est)**2)) / w.sum()
    assert abs(est - pop) < 4 * se, (est, pop, se)
    assert naive > pop * 1.5, "the unweighted estimate should be badly biased upward"


# --------------------------------------------------------------- selective prediction
def test_risk_coverage_monotone_endpoints():
    p, y = calibrated(n=50_000)
    pred = (p >= 0.5).astype(float)
    correct = (pred == y).astype(float)
    cov, risk, thr = M.risk_coverage(M.noul_confidence(p), correct)
    assert abs(cov[-1] - 1.0) < 1e-9
    assert abs(risk[-1] - (1 - correct.mean())) < 1e-6      # full coverage == overall error
    assert risk[0] <= risk[-1], "risk at low coverage should not exceed full-coverage risk"
    assert np.all(np.diff(cov) > 0)


def test_confidence_beats_random_abstention():
    p, y = calibrated(n=50_000)
    correct = ((p >= 0.5).astype(float) == y).astype(float)
    a = M.aurc(M.noul_confidence(p), correct)
    rand = M.aurc(RNG.uniform(size=correct.size), correct)
    assert a < rand, (a, rand)


def test_oracle_aurc_is_minimal_and_excess_nonnegative():
    p, y = calibrated(n=20_000)
    correct = ((p >= 0.5).astype(float) == y).astype(float)
    k = M.noul_confidence(p)
    assert M.excess_aurc(k, correct) >= -1e-9
    assert M.oracle_aurc(correct) <= M.aurc(k, correct) + 1e-9
    # and no ranking can beat the oracle, including random ones
    rng = np.random.default_rng(7)
    for _ in range(20):
        assert M.oracle_aurc(correct) <= M.aurc(rng.uniform(size=correct.size), correct) + 1e-9


def test_aurc_is_bounded_and_matches_full_coverage_risk_when_uninformative():
    """A constant confidence score gives one tie-block: AURC == the overall error rate."""
    correct = (np.random.default_rng(5).uniform(size=5_000) < 0.8).astype(float)
    k = np.ones_like(correct)
    assert abs(M.aurc(k, correct) - (1 - correct.mean())) < 1e-9
    assert 0.0 <= M.aurc(k, correct) <= 1.0


def test_coverage_at_risk_and_review_budget_complement():
    p, y = calibrated(n=50_000)
    correct = ((p >= 0.5).astype(float) == y).astype(float)
    k = M.noul_confidence(p)
    for r in (0.01, 0.05):
        c = M.coverage_at_risk(k, correct, r)
        assert abs(M.review_budget(k, correct, r) - (1 - c)) < 1e-12
        assert 0.0 <= c <= 1.0


def test_coverage_at_risk_unattainable_returns_zero():
    correct = np.zeros(1000)                     # never right
    assert M.coverage_at_risk(RNG.uniform(size=1000), correct, 0.01) == 0.0


def test_perfect_separation_gives_full_coverage_at_zero_risk():
    correct = np.repeat([1.0, 0.0], 500)
    kappa = np.repeat([0.9, 0.1], 500)           # confidence ranks correct items first
    assert abs(M.coverage_at_risk(kappa, correct, 0.0) - 0.5) < 1e-9


# --------------------------------------------------------------- discrete output grid
def test_probability_grid_detects_two_decimal_model():
    rng = np.random.default_rng(1)
    p = np.round(rng.uniform(0.01, 0.99, 20_000), 2)
    g = M.probability_grid(p)
    assert g["is_two_decimal"]
    assert not g["hits_zero"] and not g["hits_one"]
    assert abs(g["min_gap"] - 0.01) < 1e-6


def test_ece_discrete_is_zero_when_calibrated_on_a_grid():
    """A calibrated model on a 2-dp grid must give ~0 discrete ECE."""
    rng = np.random.default_rng(2)
    p = np.round(rng.uniform(0.01, 0.99, 400_000), 2)
    y = (rng.uniform(size=p.size) < p).astype(float)
    assert M.ece_discrete(p, y) < 0.005


def test_equal_mass_binning_collapses_on_a_skewed_grid():
    """The reason F4 bins by grid value rather than by equal mass.

    Not a claim that one ECE exceeds the other -- which scheme reports the larger error is
    case-specific. The claim is about RESOLUTION: when most of the mass sits on a few grid
    values, equal-mass binning cannot place distinct cut points, so the reliability curve
    loses the low-probability region that matters for rare variables.
    """
    rng = np.random.default_rng(3)
    p = np.where(rng.uniform(size=200_000) < 0.97,
                 rng.choice([0.01, 0.02, 0.03], 200_000),
                 np.round(rng.uniform(0.5, 0.99, 200_000), 2))
    y = (rng.uniform(size=p.size) < p / 5).astype(float)
    n_equal_mass = np.unique(M.equal_mass_bins(p, np.ones_like(p), 10)).size
    n_discrete = M.discrete_reliability(p, y)[0].size
    assert n_equal_mass <= 5, n_equal_mass
    assert n_discrete > 10 * n_equal_mass, (n_discrete, n_equal_mass)


def test_ece_discrete_needs_no_binning_parameter():
    """ece() changes with n_bins; ece_discrete() is a fixed property of the data."""
    rng = np.random.default_rng(6)
    p = np.round(rng.uniform(0.01, 0.99, 50_000), 2)
    y = (rng.uniform(size=p.size) < p * 0.7).astype(float)
    assert M.ece(p, y, n_bins=5) != M.ece(p, y, n_bins=25)
    assert M.ece_discrete(p, y) == M.ece_discrete(p, y)


def test_discrete_reliability_returns_one_point_per_grid_value():
    p = np.repeat([0.01, 0.02, 0.50, 0.99], 500)
    y = (np.random.default_rng(4).uniform(size=p.size) < p).astype(float)
    gp, gy, gw = M.discrete_reliability(p, y)
    assert gp.size == 4
    assert np.allclose(gp, [0.01, 0.02, 0.50, 0.99])
    assert np.allclose(gw.sum(), p.size)


def test_resolution_floor_flags_sub_floor_base_rates():
    """A variable rarer than the grid floor cannot be represented calibratedly."""
    rng = np.random.default_rng(5)
    n = 100_000
    p = np.full(n, 0.01)                      # model's smallest expressible value
    y = (rng.uniform(size=n) < 0.002).astype(float)   # truth 5x rarer than the floor
    r = M.resolution_floor_report(p, y)
    assert r["base_rate_below_floor"]
    assert r["expected_p_over_base_rate"] > 3
    assert r["min_achievable_ece_from_floor"] > 0
    # and a base rate above the floor is not flagged
    y2 = (rng.uniform(size=n) < 0.05).astype(float)
    assert not M.resolution_floor_report(np.full(n, 0.05), y2)["base_rate_below_floor"]


# --------------------------------------------------------------- positive-class budget
def test_accuracy_budget_is_degenerate_for_rare_events():
    """The motivation for precision_coverage: with a rare positive class, always-no already
    beats a 5% risk target, so the accuracy review budget carries no information."""
    rng = np.random.default_rng(11)
    n = 50_000
    y = (rng.uniform(size=n) < 0.005).astype(float)
    p = np.where(y == 1, rng.uniform(0.3, 0.99, n), rng.uniform(0.01, 0.2, n))
    correct = ((p > 0.5).astype(float) == y).astype(float)
    assert 1 - correct.mean() < 0.05
    assert M.review_budget(M.noul_confidence(p), correct, 0.05) == 0.0


def test_precision_coverage_endpoints_and_monotone_coverage():
    rng = np.random.default_rng(12)
    n = 20_000
    p = np.round(rng.uniform(0.01, 0.99, n), 2)
    y = (rng.uniform(size=n) < p).astype(float)
    cov, fdr, thr = M.precision_coverage(p, y)
    assert abs(cov[-1] - 1.0) < 1e-9
    flagged = p > 0.5
    assert abs((1 - fdr[-1]) - y[flagged].mean()) < 1e-6   # full coverage == raw precision
    assert np.all(np.diff(cov) > 0)
    assert np.all(np.diff(thr) < 0)                        # thresholds descend


def test_coverage_at_precision_is_one_when_target_already_met():
    p = np.repeat([0.9], 1000)
    y = np.ones(1000)
    assert M.coverage_at_precision(p, y, 0.95) == 1.0
    r = M.positive_review_budget(p, y, 0.95)
    assert r["already_meets_target"] and r["review_budget_of_flagged"] == 0.0


def test_coverage_at_precision_shrinks_as_target_rises():
    rng = np.random.default_rng(13)
    n = 20_000
    p = np.round(rng.uniform(0.51, 0.99, n), 2)
    y = (rng.uniform(size=n) < p).astype(float)      # higher p really is more often right
    c50 = M.coverage_at_precision(p, y, 0.60)
    c90 = M.coverage_at_precision(p, y, 0.90)
    assert c90 <= c50
    assert M.positive_review_budget(p, y, 0.90)["review_budget_of_flagged"] >=            M.positive_review_budget(p, y, 0.60)["review_budget_of_flagged"]


def test_positive_budget_scales_to_corpus_share():
    rng = np.random.default_rng(14)
    n = 10_000
    p = np.where(rng.uniform(size=n) < 0.02, 0.9, 0.02)
    y = (rng.uniform(size=n) < np.where(p > 0.5, 0.5, 0.001)).astype(float)
    r = M.positive_review_budget(p, y, 0.99)
    assert 0.0 < r["flagged_share_of_corpus"] < 0.05
    assert r["review_share_of_corpus"] <= r["flagged_share_of_corpus"] + 1e-12


# --------------------------------------------------------------- uncertainty
def test_bootstrap_ci_covers_point_and_truth():
    p, y = calibrated(n=4_000)
    r = M.weighted_bootstrap(lambda a, b, w=None: M.brier(a, b, w), p, y, B=300, seed=1)
    assert r["lo"] < r["point"] < r["hi"]
    assert r["hi"] - r["lo"] < 0.05


def test_bias_corrected_interval_brackets_the_point_estimate():
    """ECE is an absolute deviation, so bootstrap resampling inflates it and the naive
    percentile interval can sit entirely above the point estimate. After bias correction the
    interval must bracket it."""
    rng = np.random.default_rng(21)
    p = np.round(rng.uniform(0.01, 0.99, 3_000), 2)
    y = (rng.uniform(size=p.size) < p).astype(float)
    r = M.weighted_bootstrap(lambda a, b, w=None: M.ece_discrete(a, b, w), p, y,
                             B=400, seed=5)
    assert r["bias"] > 0, "resampling should inflate an absolute-deviation statistic"
    assert r["lo"] <= r["point"] <= r["hi"], r
    assert r["lo_raw"] > r["lo"], "uncorrected bounds should sit higher"


def test_bootstrap_respects_strata():
    p, y = calibrated(n=4_000)
    strata = (p * 5).astype(int)
    r = M.weighted_bootstrap(lambda a, b, w=None: M.ece(a, b, w), p, y,
                             strata=strata, B=200, seed=2)
    assert np.isfinite(r["lo"]) and np.isfinite(r["hi"])


def test_clopper_pearson_bounds():
    lo, hi = M.clopper_pearson(5, 100)
    assert lo < 0.05 < hi
    assert M.clopper_pearson(0, 50)[0] == 0.0
    assert M.clopper_pearson(50, 50)[1] == 1.0


def test_cohen_kappa_extremes():
    a = RNG.integers(0, 3, 500)
    assert abs(M.cohen_kappa(a, a) - 1.0) < 1e-9
    b = np.where(RNG.uniform(size=500) < 0.5, a, RNG.integers(0, 3, 500))
    assert 0.0 < M.cohen_kappa(a, b) < 1.0


def test_mcnemar_detects_difference():
    a = np.ones(200, dtype=bool)
    b = np.zeros(200, dtype=bool)
    assert M.mcnemar(a, b)["p_value"] < 1e-10
    assert M.mcnemar(a, a)["p_value"] == 1.0


# --------------------------------------------------------------- WP1 and WP2 additions

def test_weighted_kappa_equals_unweighted_with_equal_weights():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 2, 500)
    b = np.where(rng.random(500) < 0.8, a, 1 - a)
    assert abs(M.cohen_kappa(a, b) - M.cohen_kappa(a, b, np.ones(500))) < 1e-12


def test_weighted_kappa_follows_the_weights():
    # two blocks that disagree at different rates; weighting toward the good block must raise
    # kappa above the unweighted value, and weighting toward the bad block must lower it
    a = np.array([0, 1] * 100)
    b = a.copy()
    b[:100] = 1 - b[:100]                       # first block disagrees completely
    w_good = np.r_[np.full(100, 0.01), np.full(100, 1.0)]
    w_bad = np.r_[np.full(100, 1.0), np.full(100, 0.01)]
    k0 = M.cohen_kappa(a, b)
    assert M.cohen_kappa(a, b, w_good) > k0 > M.cohen_kappa(a, b, w_bad)


def _clustered_frame(n_clusters=200, per=8, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_clusters):
        # a per-cluster offset is what makes the rows within a cluster dependent
        off = rng.normal(0, 1.2)
        for _ in range(per):
            p = float(np.clip(0.5 + 0.1 * rng.normal() + 0.05 * off, 0.01, 0.99))
            rows.append({"Crash_ID": c, "p": p,
                         "y": float(rng.random() < p), "w": 1.0,
                         "stratum": c % 4})
    return pd.DataFrame(rows)


def test_cluster_bootstrap_is_wider_than_a_row_bootstrap_on_clustered_data():
    d = _clustered_frame()
    fn = lambda x: float(np.average(x.y, weights=x.w))            # noqa: E731
    cl = M.cluster_bootstrap(fn, d, strata="stratum", B=300, seed=1)
    rows = M.weighted_bootstrap(lambda a, b, w=None: float(np.average(a, weights=w)),
                                d.y.to_numpy(float), d.y.to_numpy(float),
                                w=d.w.to_numpy(float), B=300, seed=1)
    assert (cl["hi"] - cl["lo"]) > (rows["hi"] - rows["lo"])
    assert cl["n_clusters"] == 200 and cl["unit"] == "Crash_ID"


def test_cluster_bootstrap_brackets_the_point_estimate():
    d = _clustered_frame(seed=5)
    fn = lambda x: float(np.average(x.y, weights=x.w))            # noqa: E731
    r = M.cluster_bootstrap(fn, d, strata="stratum", B=300, seed=2)
    assert r["lo"] <= r["point"] <= r["hi"]


def test_cluster_bootstrap_returns_one_entry_per_key_for_a_dict_statistic():
    d = _clustered_frame(seed=7)
    def stat(x):
        return {"mean_y": float(x.y.mean()), "mean_p": float(x.p.mean())}
    r = M.cluster_bootstrap(stat, d, strata="stratum", B=120, seed=3)
    for k in ("mean_y", "mean_p"):
        assert r[k]["lo"] <= r[k]["point"] <= r[k]["hi"]
    assert r["_meta"]["n_clusters"] == 200


def test_clopper_pearson_boundary_cases():
    assert M.clopper_pearson(0, 50)[0] == 0.0
    assert M.clopper_pearson(50, 50)[1] == 1.0
    lo, hi = M.clopper_pearson(0, 50)
    assert 0.0 < hi < 0.10


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
