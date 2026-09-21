"""Main analysis: prevalence, coded-field agreement, calibration, selective prediction.

Produces `data/analysis.json`, which every v1 figure and table reads. Nothing here touches
the API; it is pure computation over Stage-2 outputs plus the coded-field join.

What it computes, by outline section:
  §4.3  prevalence on the RANDOM stratum only, with Clopper-Pearson CIs; keyword population
        prevalence; Jev confirmation rate among keyword hits
  §4.5  ECE / adaptive ECE / MCE / Brier decomposition / calibration slope / Spiegelhalter z,
        against coded fields (the v1 reference; the human gold set is v2)
  §4.6  risk-coverage, AURC, excess AURC, coverage at 1% and 5% risk, review budget H(r),
        and H(r) in narratives per year for Texas
  §4.7  agreement, Cohen's kappa, and the narrative-only / code-only / both / neither
        discrepancy taxonomy

IMPORTANT framing (§7 desk-reject checklist): comparisons against coded fields are
AGREEMENT, not accuracy. Coded fields are themselves noisy, so the "calibration" measured
here is bounded by reference noise. Every key carrying a coded-field result is named
`*_vs_coded` so no table can silently present it as accuracy.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

import s07_metrics as M

ROOT = Path(__file__).resolve().parents[2]   # repository root, resolved from this file
DATA = ROOT / "paper1" / "data"

# §4.7 mapping: narrative variable -> coded-field column produced by s04_join_fst.R
CODED_MAP = {
    "alcohol_involved": "coded_alcohol",
    "drug_involved": "coded_drug",
    "fatigue": "coded_fatigue",
    "animal_involved": "coded_animal",
    "phone_use": "coded_phone",
    "unbelted": "coded_unbelted",
    "hydroplane": "coded_hydro_proxy",
    # Contributing-factor wrong-way codes ("Wrong Way - One Way Road", "Wrong Side - Not
    # Passing/Approach") are a far more direct reference than the FHE opposite-direction
    # collision type the outline proposed; the FHE version is kept as coded_wrongway_proxy
    # for a sensitivity check.
    "wrong_way": "coded_wrongway",
    # Not in the outline's §4.7 table: CRIS codes a driver medical episode as the
    # contributing factor "Ill (Explain In Narrative)", which gives medical_episode a
    # coded reference it was assumed not to have.
    "medical_episode": "coded_medical",
}
# Weak references: agreement is bounded by how indirectly the coded field measures the
# narrative concept, so these are reported but never used as the headline.
PROXY_ONLY = {"hydroplane"}

# keyword family -> narrative variable, for the confirmation-rate column of T4
KW_MAP = {"preg": "preg_mentioned", "medical": "medical_episode", "drug": "drug_involved",
          "hydro": "hydroplane", "defect": "vehicle_defect", "rage": "aggression",
          "fatigue": "fatigue", "intent": "intentional", "wrongway": "wrong_way",
          "animal": "animal_involved"}
# Families added in s01b for variables that have a coded reference but no spike family, so
# the §4.8 baseline covers the same variable set as Jev. Scored on the working narratives
# only; the ten families above remain the ones reported at population scale in T4.
KW_MAP_EXTRA = {"alcohol": "alcohol_involved", "phone": "phone_use", "unbelted": "unbelted"}
KW_MAP_ALL = {**KW_MAP, **KW_MAP_EXTRA}

# 2017-2025 inclusive = 9 years of the analysed corpus
TEXAS_PER_YEAR = 5_018_080 / 9.0
B_BOOT = 1000


def noul_ids(d: pd.DataFrame) -> list[str]:
    return sorted(c[2:] for c in d.columns if c.startswith("p_"))


def prevalence_block(rand: pd.DataFrame, kwflags: pd.DataFrame, stats: dict) -> dict:
    """§4.3 -- prevalence on the random stratum, keyword population rates, confirmation."""
    out = {}
    n_corpus = stats["n_kept_gt_40"]
    for v in noul_ids(rand):
        p = rand[f"p_{v}"].to_numpy(dtype=float)
        p = p[np.isfinite(p)]
        n = p.size
        k = int((p > 0.5).sum())
        lo, hi = M.clopper_pearson(k, n)
        band = float(((p >= 0.3) & (p <= 0.7)).mean())
        out[v] = {
            "n_random": n,
            "jev_prevalence": k / n if n else None,
            "jev_prevalence_ci95": [lo, hi],
            # E[p] is the calibrated prevalence estimate ONLY where it is not pinned to the
            # model's 0.01 output floor. For a variable rarer than the floor, most narratives
            # receive exactly 0.01 and E[p] mostly measures that floor: for preg_mentioned,
            # 90.6% of narratives sit at 0.01 and the floor supplies 75% of E[p]. The two
            # diagnostic fields below let T4 flag that rather than present a floor artefact
            # as a prevalence.
            "expected_prevalence": float(p.mean()),
            "share_at_grid_floor": float((p <= M.GRID_STEP).mean()),
            "floor_contribution_to_expected": (
                float(M.GRID_STEP * (p <= M.GRID_STEP).mean() / p.mean()) if p.mean() > 0 else None),
            "expected_prevalence_above_floor": (
                float(p[p > M.GRID_STEP].mean()) if (p > M.GRID_STEP).any() else None),
            "uncertain_band_share": band,
            "n_texas_per_year_at_p50": (k / n) * (n_corpus / 9) if n else None,
        }
    for fam, v in KW_MAP.items():
        col = f"kw_{fam}"
        if col not in kwflags.columns or f"p_{v}" not in rand.columns:
            continue
        nhit = int(kwflags[col].sum())
        out.setdefault(v, {})["keyword_population_n"] = nhit
        out[v]["keyword_population_prevalence"] = nhit / n_corpus
        sub = rand[rand.get(col, pd.Series(False, index=rand.index)).astype(bool)] \
            if col in rand.columns else None
        if sub is not None and len(sub):
            conf = float((sub[f"p_{v}"] > 0.5).mean())
            clo, chi = M.clopper_pearson(int((sub[f"p_{v}"] > 0.5).sum()), len(sub))
            out[v]["jev_confirmation_among_keyword_hits"] = conf
            out[v]["jev_confirmation_ci95"] = [clo, chi]
            out[v]["n_keyword_hits_in_random"] = int(len(sub))
    return out


def coded_block(rand: pd.DataFrame) -> dict:
    """§4.5 + §4.6 + §4.7 against coded fields. AGREEMENT, not accuracy."""
    out = {}
    for v, col in CODED_MAP.items():
        if f"p_{v}" not in rand.columns or col not in rand.columns:
            continue
        sub = rand[[f"p_{v}", col]].dropna()
        if len(sub) < 200:
            continue
        p = sub[f"p_{v}"].to_numpy(dtype=float)
        y = sub[col].astype(bool).to_numpy().astype(float)
        pred = (p > 0.5).astype(float)
        correct = (pred == y).astype(float)
        # WP2. The resampling unit is the crash, and the random stratum is a simple random
        # sample of the corpus, so there is nothing to stratify on. The earlier version
        # resampled within 0.20-wide probability bins, which holds the distribution of the
        # model's own probabilities fixed across replicates and reports an interval narrower
        # than the sampling variation it claims to describe.

        bd = M.brier_decomposition(p, y)
        slope = M.calibration_slope(p, y)
        rc = {}
        kap = M.noul_confidence(p)
        for r in (0.01, 0.05):
            rc[f"coverage_at_{int(r*100)}pct_risk"] = M.coverage_at_risk(kap, correct, r)
            rc[f"review_budget_{int(r*100)}pct"] = M.review_budget(kap, correct, r)

        both = int(((pred == 1) & (y == 1)).sum())
        narr_only = int(((pred == 1) & (y == 0)).sum())
        code_only = int(((pred == 0) & (y == 1)).sum())
        neither = int(((pred == 0) & (y == 0)).sum())

        ece_ci = M.weighted_bootstrap(lambda a, b_, w=None: M.ece_discrete(a, b_, w), p, y,
                                      B=B_BOOT, seed=42)

        # §4.6 operational budget. Accuracy-based H(r) is degenerate for a rare positive
        # class (full-coverage accuracy risk already beats a 5% target), so the budget that
        # means something is the one over FLAGGED POSITIVES at a target precision.
        pb90 = M.positive_review_budget(p, y, 0.90)
        pb95 = M.positive_review_budget(p, y, 0.95)
        positive_budget = {
            "precision_at_full_coverage": pb90["precision_at_full_coverage"],
            "flagged_share_of_corpus": pb90["flagged_share_of_corpus"],
            "review_budget_of_flagged_90": pb90["review_budget_of_flagged"],
            "review_budget_of_flagged_95": pb95["review_budget_of_flagged"],
            "review_share_of_corpus_90": pb90["review_share_of_corpus"],
            "already_meets_90": pb90["already_meets_target"],
            "review_per_year_90": pb90["review_share_of_corpus"] * TEXAS_PER_YEAR,
            "flagged_per_year": pb90["flagged_share_of_corpus"] * TEXAS_PER_YEAR,
        }
        rf = M.resolution_floor_report(p, y)
        out[v] = {
            "coded_field": col,
            "proxy_only": v in PROXY_ONLY,
            "n": int(len(sub)),
            "jev_positive_rate": float(pred.mean()),
            "coded_positive_rate": float(y.mean()),
            # --- §4.7 agreement
            "agreement_vs_coded": float(correct.mean()),
            "cohens_kappa_vs_coded": M.cohen_kappa(pred.astype(int), y.astype(int)),
            "discrepancy_taxonomy": {"both": both, "narrative_only": narr_only,
                                     "code_only": code_only, "neither": neither},
            "precision_vs_coded": both / max(both + narr_only, 1),
            "recall_vs_coded": both / max(both + code_only, 1),
            # --- §4.5 calibration
            # WP6. The point estimate and its interval share one estimand, the exact
            # calibration error on the two-decimal output grid. The equal-mass binned value
            # is kept beside it so the two are comparable, but nothing reports the two
            # together: the human-reference ratio in Section 4.2 needs like against like, and
            # under the old pairing the hydroplaning point estimate fell outside its own
            # interval.
            "ece_vs_coded": M.ece_discrete(p, y),
            "ece_vs_coded_ci95": [ece_ci["lo"], ece_ci["hi"]],
            "ece_vs_coded_equal_mass": M.ece(p, y),
            "adaptive_ece_vs_coded": M.adaptive_ece(p, y),
            "mce_vs_coded": M.mce(p, y),
            "brier_vs_coded": bd["brier"],
            "reliability_vs_coded": bd["reliability"],
            "resolution_vs_coded": bd["resolution"],
            "uncertainty_vs_coded": bd["uncertainty"],
            "brier_skill_score_vs_coded": bd["brier_skill_score"],
            "calibration_slope_vs_coded": slope["slope"],
            "calibration_intercept_vs_coded": slope["intercept"],
            "calibration_slope_se": slope["slope_se"],
            "spiegelhalter_z_vs_coded": M.spiegelhalter_z(p, y)["z"],
            # --- §4.6 selective prediction
            "aurc_vs_coded": M.aurc(kap, correct),
            "excess_aurc_vs_coded": M.excess_aurc(kap, correct),
            "aurc_random_abstention": M.aurc(
                np.random.default_rng(0).uniform(size=correct.size), correct),
            # --- §4.6 operational budget over the flagged positives
            "positive_budget": positive_budget,
            # --- output-grid resolution (see s07_metrics: Jev emits a 2-decimal grid)
            "resolution_floor": rf,
            **rc,
        }
    return out


def keyword_baseline_block(rand: pd.DataFrame) -> dict:
    """§4.8 keyword baseline scored against the coded fields, same reference as Jev."""
    out = {}
    for fam, v in KW_MAP_ALL.items():
        col, kwcol = CODED_MAP.get(v), f"kw_{fam}"
        if (not col or col not in rand.columns or kwcol not in rand.columns
                or f"p_{v}" not in rand.columns):
            continue
        sub = rand[[kwcol, col, f"p_{v}"]].dropna()
        if len(sub) < 200:
            continue
        kwp = sub[kwcol].astype(bool).to_numpy().astype(float)
        y = sub[col].astype(bool).to_numpy().astype(float)
        jv = (sub[f"p_{v}"].to_numpy(dtype=float) > 0.5).astype(float)
        tp = float(((kwp == 1) & (y == 1)).sum())
        out[v] = {
            "n": int(len(sub)),
            "keyword_positive_rate": float(kwp.mean()),
            "keyword_agreement_vs_coded": float((kwp == y).mean()),
            "keyword_precision_vs_coded": tp / max(float((kwp == 1).sum()), 1),
            "keyword_recall_vs_coded": tp / max(float((y == 1).sum()), 1),
            "keyword_kappa_vs_coded": M.cohen_kappa(kwp.astype(int), y.astype(int)),
            "mcnemar_jev_vs_keyword": M.mcnemar((jv == y), (kwp == y)),
        }
    return out


def annual_workload(rand: pd.DataFrame) -> dict:
    """WP4, T13. The number of NARRATIVES an agency reads in a year, and the number of
    variable-level review decisions, which are different quantities.

    The per-variable review shares add up over nine variables, so their sum counts a narrative
    once for every variable that routes it to review. A reader opens a narrative once and
    answers every question it raises, so the staffing figure is the union of the flagged sets
    rather than the sum. Both are reported here and each is named where it is used.
    """
    per, flags = {}, None
    for v, col in CODED_MAP.items():
        if f"p_{v}" not in rand.columns or col not in rand.columns:
            continue
        sub = rand[[f"p_{v}", col]].dropna()
        if len(sub) < 200:
            continue
        p = sub[f"p_{v}"].to_numpy(dtype=float)
        y = sub[col].astype(bool).to_numpy().astype(float)
        pb = M.positive_review_budget(p, y, 0.90)
        # The records this variable sends to review are the flagged positives that the
        # auto-accept threshold does not clear, which are the lowest-probability ones. A
        # budget of 1, which is what most variables carry against the coded fields because no
        # threshold reaches 90 percent precision, routes every flagged record.
        m = p > 0.5
        budget = pb["review_budget_of_flagged"]
        routed = pd.Series(False, index=rand.index)
        if m.any():
            idx = sub.index[m]
            order = np.argsort(p[m], kind="mergesort")       # lowest probability first
            k = int(round(budget * len(idx)))
            routed.loc[idx[order[:k]]] = True
        per[v] = {"share_of_corpus": float(routed.sum() / len(rand)),
                  "per_year": float(routed.sum() / len(rand) * TEXAS_PER_YEAR)}
        flags = routed if flags is None else (flags | routed)

    union_share = float(flags.sum() / len(rand)) if flags is not None else float("nan")
    sum_share = float(sum(r["share_of_corpus"] for r in per.values()))
    return {
        "target_precision": 0.90,
        "n_variables": len(per),
        "per_variable": per,
        "union_share_of_corpus": union_share,
        "union_per_year": union_share * TEXAS_PER_YEAR,
        "sum_of_variable_decisions_share": sum_share,
        "sum_of_variable_decisions_per_year": sum_share * TEXAS_PER_YEAR,
        "note": ("the union counts narratives, the sum counts variable-level review "
                 "decisions; a narrative routed by more than one variable is read once"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()

    stats = json.loads((DATA / "corpus_stats.json").read_text())
    d = pd.read_parquet(DATA / "stage2" / "stage2_flat.parquet")
    kwflags = pd.read_parquet(DATA / "keyword_flags.parquet")

    # attach keyword flags and coded fields to the Stage-2 rows
    d = d.merge(kwflags, on="Crash_ID", how="left")
    extra = DATA / "keyword_flags_extra.parquet"
    if extra.exists():
        d = d.merge(pd.read_parquet(extra), on="Crash_ID", how="left")
    cj = DATA / "coded_join.csv"
    if cj.exists():
        d = d.merge(pd.read_csv(cj), on="Crash_ID", how="left")
        print(f"joined coded fields for {d['coded_alcohol'].notna().sum():,} of {len(d):,} rows")
    else:
        print("WARNING: coded_join.csv absent -- coded-field blocks will be empty")

    rand = d[d["stratum"] == "random"].copy()
    print(f"stage2 rows {len(d):,}; random stratum {len(rand):,}")

    leak = {}
    lf = DATA / "stage2" / "stage2_leakage.json"
    if lf.exists():
        leak = json.loads(lf.read_text())

    out = {
        "corpus": {k: stats[k] for k in
                   ("n_rows_total", "n_kept_gt_40", "n_narrative_empty", "empty_share",
                    "length_quantiles_chars", "keyword_counts", "keyword_prevalence",
                    "residual_pii_before_second_pass")},
        "run": {
            "n_stage2": int(len(d)),
            "n_stage2_random": int(len(rand)),
            "by_stratum": d["stratum"].value_counts().to_dict(),
            "mean_input_tokens": float(d["input_tokens"].mean()),
            "p50_latency_s": float(d["latency_s"].median()),
            "models_seen": d["model"].value_counts().to_dict(),
            "total_cost_usd": float(d["input_tokens"].sum() * 0.042 / 1e6),
        },
        "gate_leakage": leak,
        "output_grid": {v: M.probability_grid(rand[f"p_{v}"].dropna().to_numpy(dtype=float))
                        for v in noul_ids(rand)
                        if rand[f"p_{v}"].notna().any()},
        "prevalence": prevalence_block(rand, kwflags, stats),
        "vs_coded_fields": coded_block(rand),
        "annual_workload": annual_workload(rand),
        "keyword_baseline": keyword_baseline_block(rand),
        "notes": {
            "reference": "coded CRIS fields, which are themselves noisy; all comparisons "
                         "are AGREEMENT, not accuracy. The human gold set (§4.4) is v2.",
            "proxy_fields": sorted(PROXY_ONLY),
            "bootstrap_B": B_BOOT,
        },
    }
    (DATA / "analysis.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {DATA/'analysis.json'}")

    print("\nprevalence (random stratum, p>0.5):")
    for v, r in sorted(out["prevalence"].items(), key=lambda kv: -(kv[1].get("jev_prevalence") or 0)):
        if r.get("jev_prevalence") is None:
            continue
        print(f"  {v:18s} {100*r['jev_prevalence']:6.3f}%  "
              f"[{100*r['jev_prevalence_ci95'][0]:.3f}, {100*r['jev_prevalence_ci95'][1]:.3f}]  "
              f"uncertain-band {100*r['uncertain_band_share']:.2f}%")
    if out["vs_coded_fields"]:
        print("\nvs coded fields (AGREEMENT, not accuracy):")
        for v, r in out["vs_coded_fields"].items():
            print(f"  {v:18s} agree {100*r['agreement_vs_coded']:5.2f}%  "
                  f"kappa {r['cohens_kappa_vs_coded']:+.3f}  ECE {r['ece_vs_coded']:.4f}  "
                  f"slope {r['calibration_slope_vs_coded']:.3f}  "
                  f"H(5%) {100*r['review_budget_5pct']:5.1f}%"
                  + ("  [proxy]" if r["proxy_only"] else ""))


if __name__ == "__main__":
    main()
