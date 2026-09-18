"""Step 6 -- calibration and selective-prediction metrics (§4.5, §4.6).

Every estimator accepts Horvitz-Thompson weights `w` so the same code serves the
probability-stratified gold set (§4.4) and the unweighted coded-field audit (§4.7).
Weights are normalised to sum to n internally, so weighted and unweighted calls agree
when w is constant.

Unit-tested in tests/test_metrics.py against the properties §6 Step 6 names:
perfectly calibrated Bernoulli draws must give ECE ~ 0 and calibration slope ~ 1.
"""
from __future__ import annotations
import numpy as np
from scipy import stats
from scipy.optimize import minimize

EPS = 1e-12


def _prep(p, y, w=None):
    p = np.clip(np.asarray(p, dtype=float).ravel(), 0.0, 1.0)
    y = np.asarray(y, dtype=float).ravel()
    if w is None:
        w = np.ones_like(p)
    else:
        w = np.asarray(w, dtype=float).ravel()
    if not (p.size == y.size == w.size):
        raise ValueError(f"length mismatch: p={p.size} y={y.size} w={w.size}")
    if p.size == 0:
        raise ValueError("empty input")
    w = w * (p.size / w.sum())          # normalise to sum to n
    return p, y, w


# ------------------------------------------------------------------ binning
def equal_mass_bins(p: np.ndarray, w: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Bin index per item, using weighted quantiles so each bin holds ~equal weight.
    Ties in p are kept in the same bin, so fewer than n_bins may be returned."""
    order = np.argsort(p, kind="mergesort")
    cum = np.cumsum(w[order])
    total = cum[-1]
    edges_idx = np.searchsorted(cum, total * np.arange(1, n_bins) / n_bins)
    cuts = np.unique(p[order][np.clip(edges_idx, 0, p.size - 1)])
    cuts = cuts[(cuts > p.min()) & (cuts < p.max())]
    return np.searchsorted(cuts, p, side="right")


def equal_width_bins(p: np.ndarray, n_bins: int = 10) -> np.ndarray:
    return np.clip((p * n_bins).astype(int), 0, n_bins - 1)


def _bin_table(p, y, w, b):
    """Per-bin weighted (n_eff, mean_p, mean_y)."""
    rows = []
    for k in np.unique(b):
        m = b == k
        wk = w[m].sum()
        if wk <= 0:
            continue
        rows.append((wk, float(np.average(p[m], weights=w[m])), float(np.average(y[m], weights=w[m]))))
    return np.array(rows)  # (n_bins, 3)


# ------------------------------------------------------------------ calibration error
def ece(p, y, w=None, n_bins: int = 10, scheme: str = "equal-mass") -> float:
    """Expected calibration error: weighted mean |mean_p - mean_y| over bins."""
    p, y, w = _prep(p, y, w)
    b = equal_mass_bins(p, w, n_bins) if scheme == "equal-mass" else equal_width_bins(p, n_bins)
    t = _bin_table(p, y, w, b)
    if t.size == 0:
        return float("nan")
    return float(np.sum(t[:, 0] * np.abs(t[:, 1] - t[:, 2])) / t[:, 0].sum())


def adaptive_ece(p, y, w=None, n_bins: int = 10) -> float:
    """Adaptive ECE (Nixon et al. 2019): equal-mass binning. Kept as a named entry point
    because §4.5 reports ECE and adaptive ECE as separate columns of T5."""
    return ece(p, y, w, n_bins=n_bins, scheme="equal-mass")


def mce(p, y, w=None, n_bins: int = 10, scheme: str = "equal-mass") -> float:
    """Maximum calibration error: largest per-bin gap."""
    p, y, w = _prep(p, y, w)
    b = equal_mass_bins(p, w, n_bins) if scheme == "equal-mass" else equal_width_bins(p, n_bins)
    t = _bin_table(p, y, w, b)
    if t.size == 0:
        return float("nan")
    return float(np.max(np.abs(t[:, 1] - t[:, 2])))


def reliability_curve(p, y, w=None, n_bins: int = 10):
    """Returns (mean_p, mean_y, weight) per bin, for the F4 reliability diagrams."""
    p, y, w = _prep(p, y, w)
    t = _bin_table(p, y, w, equal_mass_bins(p, w, n_bins))
    return t[:, 1], t[:, 2], t[:, 0]


# ------------------------------------------------------------------ proper scoring
def brier(p, y, w=None) -> float:
    p, y, w = _prep(p, y, w)
    return float(np.average((p - y) ** 2, weights=w))


def brier_decomposition(p, y, w=None, n_bins: int = 10) -> dict:
    """Murphy (1973): Brier = reliability - resolution + uncertainty.

    Computed on the binned representation, so `reliability - resolution + uncertainty`
    reproduces the *binned* Brier score, not the raw one; the within-bin variance of p is
    the difference. Both are returned so the gap is visible rather than hidden.
    """
    p, y, w = _prep(p, y, w)
    b = equal_mass_bins(p, w, n_bins)
    t = _bin_table(p, y, w, b)
    W = t[:, 0].sum()
    ybar = float(np.average(y, weights=w))
    rel = float(np.sum(t[:, 0] * (t[:, 1] - t[:, 2]) ** 2) / W)
    res = float(np.sum(t[:, 0] * (t[:, 2] - ybar) ** 2) / W)
    unc = float(ybar * (1 - ybar))
    bs = brier(p, y, w)
    return {"brier": bs, "reliability": rel, "resolution": res, "uncertainty": unc,
            "binned_brier": rel - res + unc,
            "brier_skill_score": float(1 - bs / unc) if unc > 0 else float("nan"),
            "n_bins_used": int(t.shape[0])}


# ------------------------------------------------------------------ calibration slope
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_slope(p, y, w=None) -> dict:
    """Cox (1958) calibration: logistic regression of y on logit(p).
    Perfect calibration => intercept 0, slope 1. Weighted MLE via scipy (statsmodels is
    not installed on this machine and an unpenalised weighted fit is a two-parameter
    problem, so it is solved directly)."""
    p, y, w = _prep(p, y, w)
    x = _logit(p)

    def nll(theta):
        a, b = theta
        z = a + b * x
        # numerically stable log-likelihood
        return float(np.sum(w * (np.logaddexp(0.0, z) - y * z)))

    def _sigmoid(z):
        """Stable both ways: exp(-z) overflows for the large negative z that separation
        drives the optimiser towards, which is exactly when this is called."""
        z = np.clip(z, -500, 500)
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

    def grad(theta):
        a, b = theta
        r = w * (_sigmoid(a + b * x) - y)
        return np.array([r.sum(), float(np.dot(r, x))])

    res = minimize(nll, np.array([0.0, 1.0]), jac=grad, method="BFGS")
    a, b = res.x
    # observed-information standard errors, via a stable sigmoid: exp(-z) overflows for the
    # large |z| that separation drives the optimiser towards, and the resulting warning
    # hides a slope that is not a real estimate.
    mu = _sigmoid(a + b * x)
    v = w * mu * (1 - mu)
    H = np.array([[v.sum(), float(np.dot(v, x))],
                  [float(np.dot(v, x)), float(np.dot(v, x * x))]])
    try:
        se = np.sqrt(np.diag(np.linalg.inv(H)))
    except np.linalg.LinAlgError:
        se = np.array([np.nan, np.nan])
    # Quasi-complete separation: when logit(p) orders the outcomes almost perfectly the MLE
    # runs off to infinity and the optimiser stops at an arbitrarily large slope. Reporting
    # that number as a calibration slope would be meaningless, so it is flagged instead. The
    # practical test is a huge coefficient with a standard error of the same magnitude.
    separated = bool(abs(b) > 25 or not np.isfinite(se[1]) or se[1] > abs(b))
    return {"intercept": float(a), "slope": float(b),
            "intercept_se": float(se[0]), "slope_se": float(se[1]),
            "converged": bool(res.success),
            "separated": separated,
            "slope_reportable": None if separated else float(b)}


def spiegelhalter_z(p, y, w=None) -> dict:
    """Spiegelhalter (1986) calibration test. z ~ N(0,1) under perfect calibration."""
    p, y, w = _prep(p, y, w)
    num = float(np.sum(w * (y - p) * (1 - 2 * p)))
    den = float(np.sqrt(np.sum(w ** 2 * (1 - 2 * p) ** 2 * p * (1 - p))))
    if den <= EPS:
        return {"z": float("nan"), "p_value": float("nan")}
    z = num / den
    return {"z": z, "p_value": float(2 * stats.norm.sf(abs(z)))}


# ------------------------------------------------------------------ selective prediction
def risk_coverage(kappa, correct, w=None):
    """Risk-coverage curve. `kappa` is the confidence score, `correct` is 1 if the
    prediction at that item is right. Returns (coverage, risk, threshold) arrays, sorted
    by decreasing confidence, i.e. the curve traced by lowering the abstention threshold."""
    kappa = np.asarray(kappa, dtype=float).ravel()
    correct = np.asarray(correct, dtype=float).ravel()
    if w is None:
        w = np.ones_like(kappa)
    w = np.asarray(w, dtype=float).ravel()
    w = w * (kappa.size / w.sum())
    order = np.argsort(-kappa, kind="mergesort")
    k, c, ww = kappa[order], correct[order], w[order]
    cw = np.cumsum(ww)
    cerr = np.cumsum(ww * (1 - c))
    coverage = cw / cw[-1]
    risk = cerr / np.maximum(cw, EPS)
    # keep only the last index of each tied confidence value (a threshold cannot split ties)
    keep = np.append(np.diff(k) != 0, True)
    return coverage[keep], risk[keep], k[keep]


def aurc(kappa, correct, w=None) -> float:
    """Area under the risk-coverage curve over the full coverage range [0, 1] (lower is
    better).

    Integrated as a step function: each tie-group of equal confidence contributes its
    end-of-group selective risk across the coverage it spans. A threshold cannot split a
    tie, so charging the whole block the risk it reaches is the attainable (conservative)
    reading, and it keeps the oracle ranking minimal -- which a trapezoid rescaled to the
    observed coverage range does not.
    """
    cov, risk, _ = risk_coverage(kappa, correct, w)
    if cov.size == 0:
        return float("nan")
    width = np.diff(np.concatenate([[0.0], cov]))
    return float(np.sum(risk * width))


def oracle_aurc(correct, w=None) -> float:
    """AURC of an oracle that ranks all correct items above all incorrect ones."""
    correct = np.asarray(correct, dtype=float).ravel()
    return aurc(correct, correct, w)


def excess_aurc(kappa, correct, w=None) -> float:
    return aurc(kappa, correct, w) - oracle_aurc(correct, w)


def coverage_at_risk(kappa, correct, target_risk: float, w=None) -> float:
    """Largest coverage whose selective risk is <= target_risk. 0 if unattainable."""
    cov, risk, _ = risk_coverage(kappa, correct, w)
    ok = risk <= target_risk
    return float(cov[ok].max()) if ok.any() else 0.0


def review_budget(kappa, correct, target_risk: float, w=None) -> float:
    """H(r) = 1 - C*(r): the share of items a human must review to hit the target risk."""
    return 1.0 - coverage_at_risk(kappa, correct, target_risk, w)


def noul_confidence(p) -> np.ndarray:
    """kappa for a binary Noul: distance from the decision boundary (§4.6)."""
    p = np.asarray(p, dtype=float)
    return np.maximum(p, 1 - p)


# ------------------------------------------------------------------ uncertainty
def weighted_bootstrap(fn, *arrays, w=None, strata=None, B: int = 1000, seed: int = 42,
                       alpha: float = 0.05):
    """Stratified weighted bootstrap CI (§5.5: B = 1,000, resample within strata).

    `fn(*arrays_boot, w=w_boot)` must return a float. Resampling is within stratum, which
    preserves the probability-bin design of §4.4.
    """
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    strata = np.zeros(n, dtype=int) if strata is None else np.asarray(strata)
    idx_by_s = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    point = float(fn(*arrays, w=w))
    out = np.empty(B)
    for b in range(B):
        take = np.concatenate([rng.choice(ix, size=ix.size, replace=True) for ix in idx_by_s])
        try:
            out[b] = fn(*(a[take] for a in arrays), w=w[take])
        except Exception:
            out[b] = np.nan
    out = out[np.isfinite(out)]
    if out.size < 2:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n_boot": int(out.size)}
    lo, hi = np.quantile(out, [alpha / 2, 1 - alpha / 2])

    # Bias correction. Calibration error is an ABSOLUTE deviation, so the noise a bootstrap
    # resample adds to each bin's observed rate can only push |p - obs| up, never down. The
    # naive percentile interval is therefore shifted upward and can sit entirely above the
    # point estimate -- which is not a sampling statement about the estimand, it is an
    # artefact of the statistic. Subtracting the bootstrap bias re-centres the interval on
    # the point estimate. `lo_raw`/`hi_raw` keep the uncorrected values visible.
    bias = float(out.mean()) - point
    return {"point": point,
            "lo": float(lo - bias), "hi": float(hi - bias),
            "lo_raw": float(lo), "hi_raw": float(hi), "bias": bias,
            "se": float(out.std(ddof=1)), "n_boot": int(out.size)}


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    """Exact binomial CI (§5.8) for prevalences and confirmation rates."""
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def cohen_kappa(a, b) -> float:
    """Cohen (1960) kappa for two label vectors."""
    a, b = np.asarray(a), np.asarray(b)
    cats = np.unique(np.concatenate([a, b]))
    n = a.size
    po = float(np.mean(a == b))
    pe = float(sum((np.mean(a == c) * np.mean(b == c)) for c in cats))
    return float("nan") if pe >= 1 - EPS else (po - pe) / (1 - pe)


def mcnemar(correct_a, correct_b) -> dict:
    """Exact McNemar test for paired accuracy comparison (§5.9)."""
    a, b = np.asarray(correct_a, dtype=bool), np.asarray(correct_b, dtype=bool)
    b01 = int(np.sum(a & ~b))
    b10 = int(np.sum(~a & b))
    n = b01 + b10
    pv = 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(b01, b10), n, 0.5)))
    return {"a_only": b01, "b_only": b10, "p_value": pv}


# ------------------------------------------------------------------ discrete output grid
# Jev returns probabilities on a fixed two-decimal grid: exactly 99 attainable values,
# 0.01 to 0.99, never 0.00 or 1.00 (verified across all Nouls and across Choice/Score
# confidence). Three consequences the analysis has to respect:
#   1. Equal-mass binning cannot resolve the low-probability region -- for a rare variable
#      most of the mass sits on two or three grid points, so a 10-bin reliability diagram
#      collapses to 2-3 points. Bin by the grid value instead.
#   2. ECE needs no binning approximation at all: grouping by distinct value IS the exact
#      calibration error.
#   3. There is a RESOLUTION FLOOR at 0.01. A variable whose true prevalence is below 1%
#      cannot be represented calibratedly, which forces E[p] to overestimate its
#      prevalence no matter how good the model's ranking is.
GRID_STEP = 0.01


def probability_grid(p) -> dict:
    """Describe the attainable output grid of a probability vector."""
    p = np.asarray(p, dtype=float).ravel()
    u = np.unique(p[np.isfinite(p)])
    step = float(np.min(np.diff(u))) if u.size > 1 else float("nan")
    return {"n_distinct": int(u.size), "min": float(u.min()), "max": float(u.max()),
            "min_gap": step,
            "is_two_decimal": bool(np.allclose(u, np.round(u, 2))),
            "hits_zero": bool(u.min() <= 0.0), "hits_one": bool(u.max() >= 1.0)}


def discrete_reliability(p, y, w=None):
    """Reliability curve with one point per attainable grid value.

    Returns (grid_p, observed_rate, weight). This is the honest reliability diagram for a
    model with a discrete output grid: no binning choice is involved.
    """
    p, y, w = _prep(p, y, w)
    u = np.unique(p)
    gp, gy, gw = [], [], []
    for v in u:
        m = p == v
        ww = w[m].sum()
        if ww <= 0:
            continue
        gp.append(float(v)); gy.append(float(np.average(y[m], weights=w[m]))); gw.append(float(ww))
    return np.array(gp), np.array(gy), np.array(gw)


def ece_discrete(p, y, w=None) -> float:
    """Exact ECE for a discrete-output model: weighted mean |p - observed| over grid values."""
    gp, gy, gw = discrete_reliability(p, y, w)
    if gp.size == 0:
        return float("nan")
    return float(np.sum(gw * np.abs(gp - gy)) / gw.sum())


def resolution_floor_report(p, y, w=None, floor: float = GRID_STEP) -> dict:
    """How much of the calibration error is forced by the output grid's resolution floor.

    For a variable whose base rate is below `floor`, even a perfect ranker must assign at
    least `floor` to every narrative it does not rank at the bottom, so E[p] >= floor and
    the prevalence is structurally overestimated.
    """
    p, y, w = _prep(p, y, w)
    base = float(np.average(y, weights=w))
    below = base < floor
    return {"base_rate": base, "grid_floor": floor,
            "base_rate_below_floor": bool(below),
            "expected_p": float(np.average(p, weights=w)),
            "expected_p_over_base_rate": float(np.average(p, weights=w) / base) if base > 0 else float("nan"),
            "share_at_floor": float(w[p <= floor].sum() / w.sum()),
            "min_achievable_ece_from_floor": float(max(0.0, floor - base) if below else 0.0)}


# ------------------------------------------------------------------ positive-class budget
# Accuracy-based selective risk is degenerate for rare variables: with a 0.5-3.6% base rate,
# predicting "no" everywhere already scores below a 1% or 5% risk target, so the accuracy
# review budget H(r) is 0 for every variable and says nothing. Measured on the Texas data,
# full-coverage accuracy risk is 0.5-2.2% while precision among flagged positives is only
# 24-73% -- the positive class is where the review effort actually goes.
#
# So the operational quantity is the one §0 of the outline names: the review budget needed
# to reach a TARGET PRECISION. Rank the model's flagged positives by probability, auto-accept
# the most confident ones, and ask how many of the rest a human must check.


def precision_coverage(p, y, w=None, decide_threshold: float = 0.5):
    """Precision-coverage curve over the FLAGGED POSITIVES only.

    Returns (coverage, fdr, threshold) where coverage is the share of flagged positives
    auto-accepted at that probability threshold and fdr = 1 - precision among them.
    Coverage is relative to the flagged set, not to the corpus.
    """
    p, y, w = _prep(p, y, w)
    m = p > decide_threshold
    if not m.any():
        return np.array([]), np.array([]), np.array([])
    pp, yy, ww = p[m], y[m], w[m]
    order = np.argsort(-pp, kind="mergesort")
    pp, yy, ww = pp[order], yy[order], ww[order]
    cw = np.cumsum(ww)
    ctp = np.cumsum(ww * yy)
    coverage = cw / cw[-1]
    fdr = 1.0 - ctp / np.maximum(cw, EPS)
    keep = np.append(np.diff(pp) != 0, True)      # a threshold cannot split ties
    return coverage[keep], fdr[keep], pp[keep]


def coverage_at_precision(p, y, target_precision: float, w=None,
                          decide_threshold: float = 0.5) -> float:
    """Largest share of flagged positives that can be auto-accepted at >= target precision.
    FDR is not monotone in coverage, so this takes the maximum attainable coverage."""
    cov, fdr, _ = precision_coverage(p, y, w, decide_threshold)
    if cov.size == 0:
        return 0.0
    ok = fdr <= (1.0 - target_precision)
    return float(cov[ok].max()) if ok.any() else 0.0


def positive_review_budget(p, y, target_precision: float, w=None,
                           decide_threshold: float = 0.5) -> dict:
    """Share of FLAGGED POSITIVES a human must check to reach `target_precision`."""
    p2, y2, w2 = _prep(p, y, w)
    m = p2 > decide_threshold
    n_flagged = float(w2[m].sum())
    cov = coverage_at_precision(p, y, target_precision, w, decide_threshold)
    base_prec = float(np.average(y2[m], weights=w2[m])) if m.any() else float("nan")
    return {"target_precision": target_precision,
            "n_flagged_effective": n_flagged,
            "flagged_share_of_corpus": float(n_flagged / w2.sum()),
            "precision_at_full_coverage": base_prec,
            "coverage_at_target": cov,
            "review_budget_of_flagged": 1.0 - cov,
            "already_meets_target": bool(base_prec >= target_precision),
            "review_share_of_corpus": float((1.0 - cov) * n_flagged / w2.sum())}
