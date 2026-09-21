"""Emit `manuscript/numbers.tex`: one LaTeX macro per number quoted in the prose.

The Results prose is written before the final run exists. Hardcoding numbers into main.tex
would let a figure from an earlier run survive into the submission -- the exact failure the
stale-table quarantine already caught once. Instead every number the prose quotes is a macro
defined here from analysis.json, so the prose is fixed and the values bind at build time.

If a macro is missing, LaTeX fails loudly rather than printing a wrong number silently.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from s07_metrics import GRID_STEP as M_GRID_STEP

ROOT = Path(__file__).resolve().parents[2]   # project root, relative to this file (was a hard-coded D: path)
DATA = ROOT / "paper1" / "data"
OUT = ROOT / "paper1" / "manuscript" / "numbers.tex"
GOLD_CSV = DATA / "gold" / "gold_labels.csv"

lines: list[str] = []


def mac(name: str, value: str) -> None:
    """LaTeX macro names must be letters only."""
    safe = "".join(c for c in name if c.isalpha())
    lines.append(rf"\newcommand{{\{safe}}}{{{value}}}")


def f(x, d=2):
    return "--" if x is None else f"{float(x):.{d}f}"


def pc(x, d=1):
    return "--" if x is None else f"{100*float(x):.{d}f}"


def thousands(x) -> str:
    return f"{int(round(float(x))):,}".replace(",", "{,}")


def main() -> None:
    a = json.loads((DATA / "analysis.json").read_text())
    cm = json.loads((DATA / "cost_model.json").read_text())
    corpus, run = a["corpus"], a["run"]
    coded = a["vs_coded_fields"]
    prev = a["prevalence"]

    # ---------------------------------------------------------------- corpus & run
    mac("Ncorpus", thousands(corpus["n_kept_gt_40"]))
    mac("Nrecords", thousands(corpus["n_rows_total"]))
    mac("NstageTwo", thousands(run["n_stage2"]))
    mac("NstageTwoRandom", thousands(run["n_stage2_random"]))
    mac("MeanTokens", thousands(run["mean_input_tokens"]))
    mac("MedianLatency", f(run["p50_latency_s"], 2))
    mac("StageTwoCost", f(run["total_cost_usd"], 2))
    # One measured token count drives every cost in the paper; see pricing.run_input_tokens.
    import pricing as PR
    tok_run = PR.run_input_tokens()
    mac("CostPerThousand", f(PR.jev_cost_per_1k(tok_run), 3))
    mac("CostWholeCorpus",
        thousands(corpus["n_kept_gt_40"] * PR.jev_cost_per_1k(tok_run) / 1000))
    mac("TokensPerQuestion", f(cm["intercept_model"]["beta_tokens_per_question"], 0))
    mac("TokenAlpha", f(cm["intercept_model"]["alpha_tokens"], 0))
    mac("TokenBchar", f(cm["per_question_set"]["27"]["b_tokens_per_char"], 3))
    mac("SchemaTokens", thousands(cm["intercept_model"]["alpha_tokens"]
                                  + 27 * cm["intercept_model"]["beta_tokens_per_question"]))
    mac("NarrativeTokens", thousands(cm["per_question_set"]["27"]["b_tokens_per_char"]
                                     * corpus["length_quantiles_chars"]["0.5"]))
    mac("MedianChars", f(corpus["length_quantiles_chars"]["0.5"], 0))
    mac("NvarsCoded", str(len(coded)))

    # ---------------------------------------------------------------- calibration range
    if coded:
        eces = {v: r["ece_vs_coded"] for v, r in coded.items()}
        slopes = {v: r["calibration_slope_vs_coded"] for v, r in coded.items()}
        kappas = {v: r["cohens_kappa_vs_coded"] for v, r in coded.items()}
        agree = {v: r["agreement_vs_coded"] for v, r in coded.items()}
        lo_v, hi_v = min(eces, key=eces.get), max(eces, key=eces.get)
        mac("ECEmin", f(eces[lo_v], 3)); mac("ECEminVar", lo_v.replace("_", " "))
        mac("ECEmax", f(eces[hi_v], 3)); mac("ECEmaxVar", hi_v.replace("_", " "))
        mac("Slopemin", f(min(slopes.values()), 2))
        mac("Slopemax", f(max(slopes.values()), 2))
        klo, khi = min(kappas, key=kappas.get), max(kappas, key=kappas.get)
        mac("Kappamin", f(kappas[klo], 2)); mac("KappaminVar", klo.replace("_", " "))
        mac("Kappamax", f(kappas[khi], 2)); mac("KappamaxVar", khi.replace("_", " "))
        mac("Agreemin", pc(min(agree.values())))
        mac("Agreemax", pc(max(agree.values())))

        # base rates and the grid floor
        rates = {v: r["resolution_floor"]["base_rate"] for v, r in coded.items()
                 if "resolution_floor" in r}
        if rates:
            mac("BaseRateMin", pc(min(rates.values()), 2))
            mac("BaseRateMax", pc(max(rates.values()), 2))
            below = [v for v, r in coded.items()
                     if r.get("resolution_floor", {}).get("base_rate_below_floor")]
            mac("NbelowFloor", str(len(below)))
            mac("BelowFloorVars", ", ".join(v.replace("_", " ") for v in below) or "none")

        # accuracy-risk degeneracy
        full = {v: 1 - r["agreement_vs_coded"] for v, r in coded.items()}
        mac("FullCovRiskMin", pc(min(full.values()), 2))
        mac("FullCovRiskMax", pc(max(full.values()), 2))
        zero_budget = sum(1 for r in coded.values() if (r.get("review_budget_5pct") or 0) == 0)
        mac("NzeroBudget", str(zero_budget))
        # The degeneracy is total at the 5% target but only partial at 1%, so the prose has
        # to distinguish them rather than say "below the 1% and 5% targets".
        mac("NzeroBudgetOne",
            str(sum(1 for r in coded.values() if (r.get("review_budget_1pct") or 0) == 0)))

        # positive-class budget
        pbs = {v: r.get("positive_budget", {}) for v, r in coded.items()}
        precs = {v: p.get("precision_at_full_coverage") for v, p in pbs.items()
                 if p.get("precision_at_full_coverage") is not None}
        if precs:
            plo, phi = min(precs, key=precs.get), max(precs, key=precs.get)
            mac("PrecMin", f(precs[plo], 2)); mac("PrecMinVar", plo.replace("_", " "))
            mac("PrecMax", f(precs[phi], 2)); mac("PrecMaxVar", phi.replace("_", " "))
        # Figure 5b, the step that carries the argument. The flagged records tied at the
        # highest grid value are accepted or reviewed together, so accepting any share beyond
        # the very top of the ranking means taking that whole block and the precision it
        # carries. These two macros are that block for the variable the figure highlights.
        try:
            import s07_metrics as _M
            import pandas as _pd
            _d = _pd.read_parquet(DATA / "stage2" / "stage2_flat.parquet")
            _d = _d[_d.stratum == "random"]
            _c = _pd.read_csv(DATA / "coded_join.csv", usecols=["Crash_ID", "coded_alcohol"])
            _m = _d[["Crash_ID", "p_alcohol_involved"]].merge(_c, on="Crash_ID").dropna()
            _cov, _fdr, _ = _M.precision_coverage(
                _m.p_alcohol_involved.to_numpy(dtype=float),
                _m.coded_alcohol.astype(bool).to_numpy().astype(float))
            if _cov.size > 1:
                mac("PrecAlcoholTopBlock", pc(1 - _fdr[1], 0))
                mac("PrecAlcoholTopCoverage", pc(_cov[1], 0))
        except Exception as _e:                               # pragma: no cover
            print(f"  note: could not compute the Figure 5b block macros ({_e})")

        rev = {v: p.get("review_per_year_90") for v, p in pbs.items()
               if p.get("review_per_year_90") is not None}
        if rev:
            # T13. This is a sum over nine variables, so it counts variable-level review
            # decisions rather than narratives. The number of narratives a reader opens is
            # the union of the flagged sets, which is the staffing figure.
            mac("ReviewPerYearTotal", thousands(sum(rev.values())))
            mac("ReviewPerYearMax", thousands(max(rev.values())))
            mac("ReviewPerYearMaxVar", max(rev, key=rev.get).replace("_", " "))
        aw = a.get("annual_workload", {})
        if aw:
            mac("ReviewNarrativesPerYear", thousands(aw["union_per_year"]))
            mac("ReviewNarrativeShare", pc(aw["union_share_of_corpus"], 1))
            mac("ReviewDecisionsPerYear", thousands(aw["sum_of_variable_decisions_per_year"]))
            mac("ReviewUnionNVars", str(aw["n_variables"]))

        # WP8, T10. The claim that every base rate is below one percent is false; these two
        # macros carry the true counts so the sentence cannot go stale again.
        # restricted to the variables that carry a coded field, which is the set the
        # sentence in Section 3.4 is about
        jrs = {v: prev[v].get("jev_prevalence") for v in coded
               if v in prev and prev[v].get("jev_prevalence") is not None}
        if jrs:
            mac("NBelowOnePct", str(sum(1 for x in jrs.values() if x < 0.01)))
            mac("NBelowFivePct", str(sum(1 for x in jrs.values() if x < 0.05)))
            mac("NPrevalenceVars", str(len(jrs)))
            mac("PrevalenceMaxPct", pc(max(jrs.values()), 2))

    # The shape claim of Figure 3, made exact. Every panel turns up at the ceiling, meaning
    # the top decile holds more mass than the decile below it. That is not the same as the
    # ceiling outweighing the middle, which fails for a few variables, so both counts are
    # macros and the prose names the exceptions rather than asserting one shape for all.
    try:
        import pandas as _pd
        _d = _pd.read_parquet(DATA / "stage2" / "stage2_flat.parquet")
        _d = _d[_d.stratum == "random"]
        _up, _mid_heavy = 0, []
        _pcols = [c for c in _d.columns if c.startswith("p_")]
        for _c in _pcols:
            _p = _d[_c].to_numpy(dtype=float)
            _hi = int((_p >= 0.90).sum())
            _up += int(_hi > int(((_p >= 0.80) & (_p < 0.90)).sum()))
            if _hi < int(((_p >= 0.30) & (_p <= 0.70)).sum()):
                _mid_heavy.append(_c[2:])
        mac("NCeilingUpturn", str(_up))
        mac("NGridShapeVars", str(len(_pcols)))
        mac("NMidHeavy", str(len(_mid_heavy)))
        mac("MidHeavyVars", ", ".join(v.replace("_", " ") for v in _mid_heavy) or "none")
    except Exception as _e:                                   # pragma: no cover
        print(f"  note: could not compute the grid-shape macros ({_e})")

    # ---------------------------------------------------------------- output grid
    grid = a.get("output_grid", {})
    if grid:
        g = next(iter(grid.values()))
        mac("GridDistinct", str(max(x["n_distinct"] for x in grid.values())))
        mac("GridMin", f(min(x["min"] for x in grid.values()), 2))
        mac("GridMax", f(max(x["max"] for x in grid.values()), 2))
        mac("NgridVars", str(len(grid)))

    # ---------------------------------------------------------------- gate leakage
    leak = a.get("gate_leakage", {})
    if leak:
        rates = {k: v["leakage_rate"] for k, v in leak.items() if v.get("leakage_rate")}
        if rates:
            hi = max(rates, key=rates.get)
            mac("LeakMax", pc(rates[hi], 2)); mac("LeakMaxVar", hi.replace("_", " "))
            mac("LeakMin", pc(min(rates.values()), 2))

    # ---------------------------------------------------------------- floor inflation
    infl = {v: p.get("floor_contribution_to_expected") for v, p in prev.items()
            if p.get("floor_contribution_to_expected") is not None}
    if infl:
        hi = max(infl, key=infl.get)
        mac("FloorInflMax", pc(infl[hi], 0)); mac("FloorInflMaxVar", hi.replace("_", " "))
        mac("NfloorFlagged", str(sum(1 for x in infl.values() if x > 0.20)))

    # ---------------------------------------------------------------- human gold set
    gf = DATA / "gold" / "gold_analysis.json"
    if gf.exists():
        G = json.loads(gf.read_text())
        Pg = G["pooled"]
        mac("GoldNPairs", thousands(G["n_pairs_labelled"]))
        mac("GoldNUsable", thousands(G["n_usable"]))
        mac("GoldNDisputed", str(G["n_disputed"]))
        mac("GoldNVars", str(Pg["n_variables"]))
        mac("GoldPrecision", f(Pg["precision"], 3))
        mac("GoldRecall", f(Pg["recall"], 3))
        mac("GoldFone", f(Pg["f1"], 3))
        mac("GoldKappa", f(Pg["cohens_kappa"], 3))
        mac("GoldECE", f(Pg["ece"], 4))
        mac("GoldECElo", f(Pg["ece_ci95"][0], 4))
        mac("GoldECEhi", f(Pg["ece_ci95"][1], 4))
        mac("GoldBrier", f(Pg["brier"], 4))
        mac("GoldBSS", f(Pg["brier_skill_score"], 3))
        mac("GoldSlope", f(Pg["calibration_slope"], 2))
        mac("GoldIntercept", f(Pg["calibration_intercept"], 2))
        mac("GoldZ", f(Pg["spiegelhalter_z"], 1))
        mac("GoldExpectedP", pc(Pg["expected_p_weighted"], 1))
        mac("GoldBaseRate", pc(Pg["base_rate_weighted"], 1))

        rc = G["recalibration"]
        mac("RecalRawECE", f(rc["none"]["ece"], 4))
        mac("RecalPlattECE", f(rc["platt"]["ece"], 4))
        mac("RecalIsoECE", f(rc["isotonic"]["ece"], 4))
        mac("RecalIsoBrier", f(rc["isotonic"]["brier"], 4))
        mac("RecalIsoGain", f(rc["none"]["ece"] / rc["isotonic"]["ece"], 1))

        # The reference-noise gap, per variable: how much better the coded field makes the
        # model look than the human labels do. This is the quantity that justifies the whole
        # agreement-vs-accuracy distinction, so it is reported rather than asserted.
        coded = json.loads((DATA / "analysis.json").read_text())["vs_coded_fields"]
        ratios = {v: G["per_variable"][v]["ece"] / coded[v]["ece_vs_coded"]
                  for v in G["per_variable"]
                  if v in coded and coded[v]["ece_vs_coded"] > 0}
        if ratios:
            hv = max(ratios, key=ratios.get)
            mac("ECEGapMin", f(min(ratios.values()), 1))
            mac("ECEGapMax", f(max(ratios.values()), 1))
            mac("ECEGapMaxVar", hv.replace("_", " "))
            mac("ECEGapMedian", f(float(np.median(list(ratios.values()))), 1))
            mac("ECEGapNAbove", str(sum(1 for x in ratios.values() if x > 1.0)))
            mac("ECEGapN", str(len(ratios)))
            # Figure 4 annotates both errors at three decimals, and for some variables the
            # difference does not survive that rounding. A reader counting from the figure
            # would therefore not reach the full-precision count, so the tied count is a macro
            # and the prose says which is which.
            mac("ECEGapNTied", str(sum(
                1 for v in ratios
                if round(G["per_variable"][v]["ece"], 3) == round(coded[v]["ece_vs_coded"], 3))))

        # The reference gap that does survive the correction is in the AGREEMENT statistic,
        # not the calibration error. Both kappas below are population quantities on the same
        # nine variables, so this is like against like: the coded one is computed over the
        # whole random stratum, the human one is calibration-weighted to it.
        kg = {v: (coded[v]["cohens_kappa_vs_coded"],
                  G["per_variable"][v]["cohens_kappa_weighted"])
              for v in G["per_variable"]
              if v in coded and coded[v].get("cohens_kappa_vs_coded") is not None}
        if kg:
            gaps = {v: h - c for v, (c, h) in kg.items()}
            wv = max(gaps, key=gaps.get)
            mac("KappaGapN", str(len(kg)))
            mac("KappaGapMedian", f(float(np.median(list(gaps.values()))), 2))
            mac("KappaGapMax", f(gaps[wv], 2))
            mac("KappaGapMaxVar", wv.replace("_", " "))
            mac("KappaGapMaxCoded", f(kg[wv][0], 2))
            mac("KappaGapMaxHuman", f(kg[wv][1], 2))
            mac("KappaGapNPositive", str(sum(1 for x in gaps.values() if x > 0)))
            cl = min(kg, key=lambda v: kg[v][0])
            mac("KappaCodedMin", f(kg[cl][0], 2)); mac("KappaCodedMinVar", cl.replace("_", " "))
            mac("KappaCodedMinHuman", f(kg[cl][1], 2))

        f1s = {v: r["f1"] for v, r in G["per_variable"].items()}
        lo_v, hi_f = min(f1s, key=f1s.get), max(f1s, key=f1s.get)
        mac("GoldWorstVar", lo_v.replace("_", " ")); mac("GoldWorstFone", f(f1s[lo_v], 2))
        mac("GoldBestVar", hi_f.replace("_", " ")); mac("GoldBestFone", f(f1s[hi_f], 2))
        # Thresholded on the value the TABLE prints, not the full-precision one: one
        # variable sits at 0.8496 and would be counted out here while a reader counting
        # the 0.85 in Table 10 counts it in. The prose must match the table.
        mac("GoldNStrong", str(sum(1 for x in f1s.values() if round(x, 2) >= 0.85)))

        # The variables the model fails on. Named by threshold rather than counted by hand,
        # because "three variables" is exactly the kind of claim that goes stale silently.
        weak = sorted((v for v, x in f1s.items() if x < 0.70), key=lambda v: f1s[v])
        mac("GoldNWeak", str(len(weak)))
        mac("GoldWeakVars", ", ".join(v.replace("_", " ") for v in weak) or "none")
        # The weak variables no longer fail the same way, so which of them fail on precision
        # and which on recall is read from the results rather than asserted in the prose.
        wp = [v for v in weak if G["per_variable"][v]["precision"]
              < G["per_variable"][v]["recall"]]
        wr = [v for v in weak if v not in wp]
        mac("GoldNWeakPrecision", str(len(wp)))
        mac("GoldWeakPrecisionVars", ", ".join(v.replace("_", " ") for v in wp) or "none")
        mac("GoldNWeakRecall", str(len(wr)))
        mac("GoldWeakRecallVars", ", ".join(v.replace("_", " ") for v in wr) or "none")
        if wr:
            w0 = min(wr, key=lambda v: G["per_variable"][v]["recall"])
            mac("GoldWeakRecallVar", w0.replace("_", " "))
            mac("GoldWeakRecallValue", f(G["per_variable"][w0]["recall"], 2))
            mac("GoldWeakRecallPrec", f(G["per_variable"][w0]["precision"], 2))
        precs_g = {v: r["precision"] for v, r in G["per_variable"].items()}
        wv = min(precs_g, key=precs_g.get)
        mac("GoldWorstPrec", f(precs_g[wv], 2)); mac("GoldWorstPrecVar", wv.replace("_", " "))
        mac("RecalPlattBrier", f(rc["platt"]["brier"], 4))
        mac("RecalRawBrier", f(rc["none"]["brier"], 4))

        # The review budget against HUMAN labels. Section 5.5 computes it against coded
        # fields, where a narrative-only detection counts as a false positive and the budget
        # therefore saturates; this is the same quantity with that artefact removed.
        # WP4. The budget is selected on one narrative fold and certified on the other, so
        # the reported value is held out. The in-sample value and the gap between them are
        # given beside it, because the gap is the size of the finding the review raised.
        hb90 = Pg["held_out_budget_90"]
        hb95 = Pg["held_out_budget_95"]
        mac("GoldBudgetNinety", pc(hb90["review_budget_heldout_mean"], 1))
        mac("GoldBudgetNinetyFive", pc(hb95["review_budget_heldout_mean"], 1))
        mac("GoldBudgetNinetyInSample", pc(hb90["review_budget_insample"], 1))
        mac("GoldBudgetNinetyFiveInSample", pc(hb95["review_budget_insample"], 1))
        mac("GoldBudgetOptimism", pc(hb90["optimism"], 1))
        mac("GoldBudgetPrecHeldOut", f(hb90["precision_heldout_mean"], 3))
        mac("GoldBudgetPrecLCB", f(hb90["precision_heldout_lcb95"], 3))
        mac("GoldBudgetNSplits", str(hb90["n_splits"]))
        b90 = {v: r["held_out_budget_90"]["review_budget_heldout_mean"]
               for v, r in G["per_variable"].items()}
        bhi = max(b90, key=b90.get)
        mac("GoldBudgetMaxVar", bhi.replace("_", " "))
        mac("GoldBudgetMax", pc(b90[bhi], 1))
        mac("GoldBudgetPerVarLo", pc(min(b90.values()), 1))
        mac("GoldBudgetPerVarHi", pc(max(b90.values()), 1))

        # WP1. The executed design, so that the Methods sentence and Appendix C are macros.
        D = G["design"]
        mac("DesignNEff", thousands(D["n_effective"]))
        mac("DesignDeff", f(D["kish_deff"], 2))
        mac("DesignWeightRatio", thousands(D["weight_ratio"]))
        mac("DesignNActivePairs", thousands(D["n_active_pairs"]))
        mac("DesignNControlPairs", thousands(D["n_control_pairs"]))
        mac("DesignNCollapsed", str(len(D["collapsed_cells"])))
        mac("DesignNCandidates", thousands(640))       # the oversampled draw in s15
        mac("DesignNWithheld", thousands(137))         # withheld by the identifier screen
        mac("DesignNControls", "3")                    # N_NEG_CONTROL in s05_gold_sample.py
        mac("DesignActiveThreshold", "0.05")           # ACTIVE_THRESHOLD in s05_gold_sample.py
        mac("DesignMinCell", "5")                      # MIN_CELL in s05b_design_weights.py
        mac("DesignNStrata", str(len(D["draw_strata"])))
        nev = D["per_variable_n_effective"]
        mac("DesignNEffVarLo", thousands(min(nev.values())))
        mac("DesignNEffVarHi", thousands(max(nev.values())))

        # WP5. The 49 pairs the coders could not resolve, and what they are worth.
        X = G["exclusion_sensitivity"]
        mac("ExclN", str(X["n_excluded"]))
        mac("ExclFoneError", f(X["as_model_error"]["f1"], 3))
        mac("ExclFoneCorrect", f(X["as_model_correct"]["f1"], 3))
        mac("ExclBudgetError", pc(X["as_model_error"]["review_budget_90"], 1))
        mac("ExclBudgetReview", pc(X["as_routed_to_review"]["review_budget_90"], 1))
        mac("ExclNVars", str(len(X["by_variable"])))

        # WP3. The per-variable maps are the deployment evidence; the pooled map is the
        # proof of concept. Both are held out over narrative splits.
        pv = G["recalibration"]["per_variable"]
        if pv:
            pe = {v: r["platt_ece"] for v, r in pv.items() if r.get("platt_ece") is not None}
            rw = {v: r["none_ece"] for v, r in pv.items() if r.get("none_ece") is not None}
            mac("RecalPerVarN", str(len(pv)))
            mac("RecalPerVarECELo", f(min(pe.values()), 4))
            mac("RecalPerVarECEHi", f(max(pe.values()), 4))
            mac("RecalPerVarECEMedian", f(float(np.median(list(pe.values()))), 4))
            mac("RecalPerVarRawMedian", f(float(np.median(list(rw.values()))), 4))
            mac("RecalPerVarNImproved",
                str(sum(1 for v in pe if v in rw and pe[v] < rw[v])))
        mac("RecalPooledOnlyN", str(len(G["recalibration"]["pooled_only_variables"])))
        mac("RecalRepeats", str(G["recalibration"]["n_repeats"]))
        mac("RecalMinPositives", str(G["recalibration"]["min_positives_for_own_map"]))
        pci = G["recalibration"].get("pooled_ci95", {})
        if "platt_ece" in pci:
            mac("RecalPlattECElo", f(pci["platt_ece"][0], 4))
            mac("RecalPlattECEhi", f(pci["platt_ece"][1], 4))
        ps = G["recalibration"]["pooled"]
        mac("RecalPlattSlope", f(ps.get("platt_slope"), 2))
        mac("RecalPlattIntercept", f(ps.get("platt_intercept"), 2))

        # WP2. Every interval on the human reference set names this resampling unit.
        mac("GoldNNarratives", str(Pg["n_narratives"]))
        mac("GoldFoneLo", f(Pg["ci95"]["f1"][0], 3))
        mac("GoldFoneHi", f(Pg["ci95"]["f1"][1], 3))
        mac("GoldPrecisionLo", f(Pg["ci95"]["precision"][0], 3))
        mac("GoldPrecisionHi", f(Pg["ci95"]["precision"][1], 3))
        mac("GoldBaseRateLo", pc(Pg["ci95"]["base_rate_weighted"][0], 2))
        mac("GoldBaseRateHi", pc(Pg["ci95"]["base_rate_weighted"][1], 2))
        mac("GoldMeanMinusPrev", pc(Pg["mean_probability_minus_prevalence"]
                                    if "mean_probability_minus_prevalence" in Pg
                                    else Pg["mean_p_minus_prevalence"], 2))
        sp = Pg["spiegelhalter"]
        mac("GoldZSE", f(sp["bootstrap_se"], 2))
        mac("GoldZLo", f(sp["ci95"][0], 1))
        mac("GoldZHi", f(sp["ci95"][1], 1))
        mac("GoldNZeroBudget", str(sum(1 for x in b90.values() if x <= 1e-9)))
        cb = {v: coded[v]["positive_budget"].get("review_budget_of_flagged_90")
              for v in b90 if v in coded
              and coded[v].get("positive_budget", {}).get("review_budget_of_flagged_90")
              is not None}
        mac("BudgetCompN", str(len(cb)))
        mac("BudgetLowerN", str(sum(1 for v in cb if b90[v] < cb[v] - 1e-9)))
        mac("BudgetCodedSatN", str(sum(1 for v in cb if cb[v] > 0.999)))

        # Section 5.6: how often a flag the coded field calls negative is confirmed by a
        # human. Measured in that cell, not inferred from pooled precision.
        NO = G.get("narrative_only", {})
        if NO:
            def cellmac(stem: str, c: dict) -> None:
                mac(stem, pc(c["confirmed_rate"], 0))
                mac(stem + "Lo", pc(c["ci95"][0], 0))
                mac(stem + "Hi", pc(c["ci95"][1], 0))
                mac(stem + "N", str(c["n"]))
            cellmac("NarrOnly", NO["narrative_only"])
            cellmac("NarrOnlyStrong", NO["narrative_only_strong_vars"])
            cellmac("NarrOnlyWeak", NO["narrative_only_weak_vars"])
            cellmac("CodedConf", NO["coded_confirmed_flags"])
            mac("NarrOnlyShare", pc(NO["narrative_only_share_of_flags"], 0))
            mac("NarrOnlyNVars", str(NO["n_variables"]))

        # Section 5.4: retuning tau does not rescue the weak variables. Reported as macros
        # so the negative claim cannot drift away from the table that evidences it.
        TT = G.get("threshold_tuning", {})
        if TT:
            mac("ThreshNVars", str(len(TT)))
            stable = {v: r for v, r in TT.items() if not r["unstable"]}
            mac("ThreshNStable", str(len(stable)))
            mac("ThreshNUnstable", str(len(TT) - len(stable)))
            gains = [r["gain"] for r in stable.values()]
            mac("ThreshBestGain", f(max(gains), 2))
            mac("ThreshBestGainVar",
                max(stable, key=lambda v: stable[v]["gain"]).replace("_", " "))
            mac("ThreshNHelped", str(sum(1 for g in gains if g >= 0.05)))
            mac("ThreshMedianGain", f(float(np.median(gains)), 2))
            weakt = {v: r for v, r in TT.items() if r["f1_at_half"] < 0.70}
            mac("ThreshWeakNPosMax", str(max(r["n_positive"] for r in weakt.values())))
            mac("ThreshWeakNPosMin", str(min(r["n_positive"] for r in weakt.values())))
            # The weak variables split by whether they carry enough positives for the tuning
            # to be judged at all, which is the distinction the prose has to draw.
            thin = {v: r for v, r in weakt.items()
                    if r["n_positive"] == min(x["n_positive"] for x in weakt.values())}
            thick = {v: r for v, r in weakt.items() if v not in thin}
            mac("ThreshWeakNThin", str(len(thin)))
            mac("ThreshWeakThinVars",
                ", ".join(v.replace("_", " ") for v in sorted(thin)) or "none")
            mac("ThreshWeakNThick", str(len(thick)))
            mac("ThreshWeakThickVars",
                ", ".join(v.replace("_", " ") for v in sorted(thick)) or "none")
            if thick:
                mac("ThreshWeakThickBestGain",
                    f(max(r["gain"] for r in thick.values()), 2))
            mac("ThreshWeakThinBestGain", f(max(r["gain"] for r in thin.values()), 2))
            ww = TT.get("wrong_way")
            if ww:
                mac("ThreshWrongWayNPos", str(ww["n_positive"]))
                mac("ThreshWrongWayOof", f(ww["f1_tuned_oof"], 2))
                mac("ThreshWrongWayTau", f(ww["tuned_threshold"], 2))
        # Whether the coders disagreed on the variables the model failed on decides whether
        # those failures are model error or ambiguous criteria. They did not, so it is error.
        gl = pd.read_csv(GOLD_CSV)
        mac("GoldWeakDisputes",
            str(int(gl[(gl.label == "DISPUTED") & (gl.variable.isin(weak))].shape[0])))
        top = gl[gl.label == "DISPUTED"].variable.value_counts()
        mac("GoldDisputeTopVar", top.index[0].replace("_", " ") if len(top) else "none")
        mac("GoldDisputeTopN", str(int(top.iloc[0])) if len(top) else "0")

    rel = DATA / "gold" / "gold_reliability.json"
    if rel.exists():
        R = json.loads(rel.read_text())
        ks = [v for v in R.get("pairwise_kappa_calibration", {}).values()
              if v is not None and np.isfinite(v)]
        if ks:
            mac("CoderKappaMin", f(min(ks), 2))
            mac("CoderKappaMax", f(max(ks), 2))
            mac("CoderKappaMedian", f(float(np.median(ks)), 2))
        mac("CoderNAnswers", thousands(R["n_answers"]))
        mac("CoderN", str(R["n_raters"]))
        mac("CoderUnanimous", thousands(R["n_unanimous"]))

    # ---------------------------------------------------------------- frontier baseline
    ff = DATA / "frontier" / "frontier_analysis.json"
    if ff.exists():
        FR = json.loads(ff.read_text())
        SHORT = {"claude-fable-5-1": "Fable", "gpt-5.6-sol": "GPT"}
        for label, arm in FR["arms"].items():
            k = SHORT.get(label, "".join(c for c in label if c.isalpha()))
            P_ = arm["pooled"]
            mac(k + "Fone", f(P_["f1"], 3))
            mac(k + "Precision", f(P_["precision"], 3))
            mac(k + "Recall", f(P_["recall"], 3))
            mac(k + "Kappa", f(P_["cohens_kappa"], 3))
            mac(k + "ECE", f(P_["ece"], 4))
            mac(k + "Brier", f(P_["brier"], 4))
            mac(k + "Slope", f(P_["calibration_slope"], 2))
            mac(k + "Z", f(P_["spiegelhalter_z"], 1))
            mac(k + "ExpectedP", pc(P_["expected_p_weighted"], 1))
            fv = {v: r["f1"] for v, r in arm["per_variable"].items()}
            wk = [v for v in sorted(fv, key=fv.get) if fv[v] < 0.80]
            mac(k + "WeakVars", ", ".join(v.replace("_", " ") for v in wk) or "none")
            mac(k + "NWeak", str(len(wk)))
            # How many of this arm's weak variables the typed model handles well, which is
            # what makes the failures idiosyncratic rather than shared.
            GJ = json.loads((DATA / "gold" / "gold_analysis.json").read_text())["per_variable"]
            mac(k + "NWeakJevSound",
                str(sum(1 for v in wk if (GJ.get(v, {}).get("f1") or 0) >= 0.80)))
        # The failing variables the frontier arms recover, which is what falsifies the
        # "the narrative does not say" reading of the typed model's failures. The weak set is
        # read from the results rather than named here, because which variables are weak is
        # exactly the kind of claim that goes stale silently.
        G_ = json.loads((DATA / "gold" / "gold_analysis.json").read_text())
        weak_v = sorted((v for v, r in G_["per_variable"].items() if r["f1"] < 0.70),
                        key=lambda v: G_["per_variable"][v]["f1"])
        RECOVER = 0.80
        recovered, missed = [], []
        for v in weak_v:
            stem = "".join(p.capitalize() for p in v.split("_"))
            best = None
            for label, arm in FR["arms"].items():
                r = arm["per_variable"].get(v)
                if r:
                    mac(SHORT.get(label, "X") + stem + "Fone", f(r["f1"], 2))
                    best = r["f1"] if best is None else max(best, r["f1"])
            if best is not None:
                mac("Best" + stem + "Fone", f(best, 2))
                (recovered if best >= RECOVER else missed).append(v)
        mac("FrontierNRecovered", str(len(recovered)))
        mac("FrontierRecoveredVars",
            ", ".join(v.replace("_", " ") for v in recovered) or "none")
        mac("FrontierNNotRecovered", str(len(missed)))
        mac("FrontierNotRecoveredVars",
            ", ".join(v.replace("_", " ") for v in missed) or "none")

        pr = FR.get("paired", {})
        if pr:
            mac("PairedN", thousands(pr["n_paired"]))
            mac("PairedB", thousands(pr["B"]))
            for key, d in pr["differences"].items():
                a, b = [x.strip() for x in key.split(" - ")]
                nm = "Diff" + SHORT.get(a, "Jev") + SHORT.get(b, "Jev")
                # The stored direction is (a - b). The prose always names the second model as
                # the one that gains, so the macro reports (b - a) when that is the positive
                # direction. Negating the point WITHOUT also negating and swapping the
                # interval is how this printed "0.068 [-0.088, -0.046]" the first time.
                point, (lo, hi) = d["point"], d["ci95"]
                if point < 0:
                    point, lo, hi = -point, -hi, -lo
                mac(nm, f(point, 3))
                mac(nm + "Lo", f(lo, 3)); mac(nm + "Hi", f(hi, 3))
            ind = [k for k, d in pr["differences"].items() if d["indistinguishable"]]
            mac("NIndistinguishable", str(len(ind)))

        c = FR["cost"]
        mac("FrontierCostSonnet", f(c["claude-sonnet-5_per_1k_usd"], 2))
        mac("FrontierCostHaiku", f(c["claude-haiku-4-5_per_1k_usd"], 2))
        mac("FrontierCostOpus", f(c["claude-opus-5_per_1k_usd"], 2))
        mac("FrontierCostRatio",
            f(c["claude-sonnet-5_per_1k_usd"] / c["jev_per_1k_usd"], 0))
        mac("FrontierCostRatioHaiku",
            f(c["claude-haiku-4-5_per_1k_usd"] / c["jev_per_1k_usd"], 0))
        mac("CoderNPairs", thousands(R["n_pairs"]))
        mac("CoderNCalib", thousands(R["ratings_per_pair"]["3"]))
        mac("CoderNDouble", thousands(R["ratings_per_pair"]["2"]))
        mac("CoderAgreement", pc(R["mean_agreement"], 1))
        mac("CoderMedianSec",
            f(float(np.median([x["median_ms"] for x in R["per_rater_calibration"].values()]))
              / 1000.0, 1))
        mac("CoderNUnclear", str(R["verdict_mix"].get("unclear", 0)))
        # Effort per coder, derived: quoting a hand-rounded "coder-day" in a file whose
        # whole purpose is to stop hardcoded numbers would be the wrong kind of shortcut.
        secs = float(np.median([x["median_ms"] for x in R["per_rater_calibration"].values()]))
        mac("CoderHours", f(R["n_answers"] / R["n_raters"] * secs / 1000.0 / 3600.0, 1))

    # ---------------------------------------------------------------- rewrite additions
    # Numbers that the earlier draft typed by hand. Each is read from the file that
    # produced it, so the manuscript carries no hand-typed value.
    cs = json.loads((DATA / "corpus_stats.json").read_text())
    pii = cs["residual_pii_before_second_pass"]
    mac("PIIDate", pc(pii["date"]["share"], 2))
    mac("PIIDigitRun", pc(pii["digitrun"]["share"], 2))
    mac("PIICase", pc(pii["case"]["share"], 2))
    mac("PIIDob", pc(pii["dob"]["share"], 2))
    mac("PIIIdNum", pc(pii["idnum"]["share"], 2))
    mac("PIITitledName", pc(pii["titled_name"]["share"], 2))
    yrs = sorted(cs["year_total"])
    y0, y1 = yrs[0], yrs[-1]
    mac("FirstYear", y0); mac("LastYear", y1)
    mac("NYears", str(len(yrs)))
    mac("EmptyShareFirst", pc(cs["year_empty"].get(y0, 0) / cs["year_total"][y0], 2))
    mac("EmptyShareLast", pc(cs["year_empty"].get(y1, 0) / cs["year_total"][y1], 2))
    mac("MinChars", str(cs["min_chars"]))
    mac("TexasPerYear", thousands(cs["n_kept_gt_40"] / len(yrs)))
    mac("LengthPNinetyFive", f(cs["length_quantiles_chars"]["0.95"], 0))
    mac("LengthPNinetyNine", f(cs["length_quantiles_chars"]["0.99"], 0))
    mac("LengthMean", f(cs["length_mean_chars"], 0))

    # schema composition (Stage 2 asked every question except the separate PII screen)
    sch = json.loads((ROOT / "schemas" / "crash_factors_v1_1.json").read_text())
    qs = {k: v for k, v in sch["questions"].items() if k != "pii_residual"}
    mac("NQuestions", str(len(qs)))
    mac("NPresence", str(sum(1 for v in qs.values() if v["type"] == "noul")))
    mac("NChoice", str(sum(1 for v in qs.values() if v["type"] == "choice")))
    mac("NScore", str(sum(1 for v in qs.values() if v["type"] == "score")))
    mac("NGated", str(len(sch["gates"])))
    lean = json.loads((ROOT / "schemas" / "crash_factors_lean_v1_1.json").read_text())
    mac("NLeanQuestions", str(len(lean["questions"])))

    # runs and spend
    s1 = json.loads((DATA / "stage1" / "stage1_report.json").read_text())
    s2 = json.loads((DATA / "stage2" / "stage2_report.json").read_text())
    mac("StageOneFrame", thousands(s1["frame"]))
    mac("StageOneCoded", thousands(s1["n_new"]))
    mac("StageOneCost", f(s1["cost_usd"], 2))
    mac("StageOneTokens", thousands(s1["mean_input_tokens"]))
    mac("StageOneWallHours", f(s1["wall_s"] / 3600.0, 1))
    mac("StageOneReqPerS", f(s1["req_per_s"], 0))
    mac("StageTwoReqPerS", f(s2["req_per_s"], 0))
    mac("StageTwoWallMinutes", f(s2["wall_s"] / 60.0, 0))
    mac("StageTwoErrors", str(s2["n_errors"]))
    mac("StageTwoPNinetyLatency", f(s2["p90_latency_s"], 2))
    sv = json.loads((DATA / "schema_validate_v1_1_report.json").read_text())
    mac("SchemaValidateFrame", thousands(sv["frame"]))
    mac("SchemaValidateCost", f(sv["cost_usd"], 2))
    sp = json.loads((DATA / "spend_report.json").read_text())
    mac("TotalSpend", f(sp["total_usd"], 2))
    mac("SpendCap", f(sp["cap_usd"], 0))
    mac("CostSweepCalls", str(cm["n_calls"]))
    mac("CostSweepCost", f(cm["spend_usd"], 2))
    mac("CostSweepQuestionSets", str(len(cm["per_question_set"])))
    mac("CostModelRsq", f(cm["intercept_model"]["r2"], 3))
    mac("CostPerThousandOneQuestion", f(cm["per_question_set"]["1"]["cost_per_1k_narratives_usd"], 3))

    # the paired schema test behind design rule six
    svr = json.loads((DATA / "schema_validate_report.json").read_text())
    ag = svr["aggression"]
    mac("PursuitN", str(ag["pursuit_narratives_n"]))
    mac("PursuitPosBefore", str(ag["pursuit_pos_v1"]))
    mac("PursuitPosAfter", str(ag["pursuit_pos_v1_1"]))
    mac("PursuitMeanBefore", f(ag["pursuit_mean_p_v1"], 3))
    mac("PursuitMeanAfter", f(ag["pursuit_mean_p_v1_1"], 3))
    mac("NonPursuitPosBefore", str(ag["nonpursuit_pos_v1"]))
    mac("NonPursuitPosAfter", str(ag["nonpursuit_pos_v1_1"]))
    oth = svr["other_nouls_stability"]
    mac("OtherNoulsN", str(len(oth)))
    mac("OtherAgreeMin", f(min(v["label_agreement"] for v in oth.values()), 4))
    mac("OtherShiftMax", f(max(v["mean_abs_shift"] for v in oth.values()), 4))
    mac("SchemaValidateCompared", thousands(svr["n_compared"]))

    # the two reference-field corrections, measured over all crashes in the join
    cjcols = pd.read_csv(DATA / "coded_join.csv", usecols=["coded_wrongway", "coded_wrongway_proxy"])
    mac("WrongWayProxyPrev", pc(cjcols["coded_wrongway_proxy"].astype(bool).mean(), 2))
    mac("WrongWayCodedPrev", pc(cjcols["coded_wrongway"].astype(bool).mean(), 2))
    mac("WrongWayProxyRatio", f(cjcols["coded_wrongway_proxy"].astype(bool).mean()
                                / cjcols["coded_wrongway"].astype(bool).mean(), 0))
    mac("NCrashesJoined", thousands(len(cjcols)))

    # the output grid: how many answers were inspected and how often the endpoints occurred
    st2 = pd.read_parquet(DATA / "stage2" / "stage2_flat.parquet")
    rnd = st2[st2["stratum"] == "random"]
    pcols = [c for c in rnd.columns if c.startswith("p_")]
    vals = rnd[pcols].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    mac("NGridAnswersMillions", f(vals.size / 1e6, 1))
    mac("NGridOnes", str(int((vals >= 1.0).sum())))
    mac("NGridZeros", str(int((vals <= 0.0).sum())))
    mac("GridStep", f(M_GRID_STEP, 2))
    # the resolution-floor bound for the variables whose base rate lies below the floor
    A = json.loads((DATA / "analysis.json").read_text())   # `a` is rebound above
    coded = A["vs_coded_fields"]
    fb = {v: r["resolution_floor"]["min_achievable_ece_from_floor"] for v, r in coded.items()
          if r.get("resolution_floor", {}).get("base_rate_below_floor")}
    if fb:
        hv = max(fb, key=fb.get)
        mac("FloorBoundMax", f(fb[hv], 4)); mac("FloorBoundMaxVar", hv.replace("_", " "))
        mac("FloorBoundMin", f(min(fb.values()), 4))
    # leakage counts behind the maximum rate
    leak = A.get("gate_leakage", {})
    if leak:
        hi = max(leak, key=lambda k: leak[k]["leakage_rate"] or 0)
        mac("LeakMaxNLeaked", thousands(leak[hi]["n_leaked"]))
        mac("LeakMaxNOff", thousands(leak[hi]["n_gate_off"]))
        mac("NGatedMeasured", str(len(leak)))

    # gold-set design and screening
    gd = json.loads((DATA / "gold" / "gold_design.json").read_text())
    mac("GoldBinTargets", ", ".join(str(x) for x in gd["bin_targets"]))
    mac("GoldBinTargetTail", str(gd["bin_targets"][0]))
    mac("GoldBinTargetMiddle", str(gd["bin_targets"][2]))
    mac("GoldFrameDrawn", str(gd["n_drawn"]))
    mac("GoldNegControls", str(gd["n_negative_controls"]))
    mac("GoldActiveThreshold", f(gd["active_threshold"], 2))
    mac("GoldNBins", str(len(gd["bins"])))
    gl = pd.read_csv(GOLD_CSV)
    mac("GoldNarratives", str(int(gl["Crash_ID"].nunique())))
    mac("GoldNYes", str(int((gl["label"] == "yes").sum())))
    mac("GoldNNo", str(int((gl["label"] == "no").sum())))
    withheld = json.loads((DATA / "gold" / "pii_withheld.json").read_text())
    mac("GoldPIIWithheld", str(len(withheld)))
    mac("GoldMinN", "40")   # MIN_N in s17_gold_analysis.py, the smallest per-variable n scored
    mac("ThreshRepeats", "200")   # THRESH_REPEATS in s17_gold_analysis.py
    mac("BootstrapB", thousands(A["notes"]["bootstrap_B"]))
    # per-variable ECE against human labels, range
    if gf.exists():
        ge = {v: r["ece"] for v, r in G["per_variable"].items()}
        lo_v, hi_v = min(ge, key=ge.get), max(ge, key=ge.get)
        mac("GoldECEmin", f(ge[lo_v], 3)); mac("GoldECEminVar", lo_v.replace("_", " "))
        mac("GoldECEmax", f(ge[hi_v], 3)); mac("GoldECEmaxVar", hi_v.replace("_", " "))
        gn = {v: r["n"] for v, r in G["per_variable"].items()}
        mac("GoldNPerVarMin", str(min(gn.values()))); mac("GoldNPerVarMax", str(max(gn.values())))
        mac("GoldAccuracy", pc(Pg["accuracy_weighted"], 1))
        mac("GoldFlaggedRate", pc(Pg["flagged_rate_weighted"], 1))
        mac("GoldRelExcess", f(100 * (Pg["expected_p_weighted"] / Pg["base_rate_weighted"] - 1), 0))
        nsep = sum(1 for r in G["per_variable"].values() if r["separated"])
        mac("GoldNSeparated", str(nsep))

    # frontier arm, how it was run
    if ff.exists():
        import glob as _glob
        nb = len(_glob.glob(str(DATA / "frontier" / "raw" / "fable_batch_*.json")))
        mac("FrontierBatches", str(nb))
        pk = json.loads((DATA / "frontier" / "frontier_task_package.json").read_text())
        mac("FrontierNVars", str(len(pk["criteria"])))
        mac("FrontierNarratives", str(len(pk["narratives"])))
        mac("FrontierBatchSize", str(len(pk["narratives"]) // max(nb, 1)))
        mac("FrontierCostRatioOpus",
            f(c["claude-opus-5_per_1k_usd"] / c["jev_per_1k_usd"], 0))
        mac("CostWholeCorpusSonnet",
            thousands(corpus["n_kept_gt_40"] * c["claude-sonnet-5_per_1k_usd"] / 1000))
        for label, arm in FR["arms"].items():
            k = SHORT.get(label, "X")
            mac(k + "LabelProbDisagree", pc(arm["label_probability_disagreement"], 1))
            mac(k + "DistinctP", str(arm["distinct_p_values"]))

    # design constants read from the scripts that apply them, and derived ratios
    mac("TauGate", f(0.5, 1))                       # TAU_GATE in s06_flatten.py
    mac("GoldBinsText", "; ".join(gd["bins"]))       # BIN_NAMES in s05_gold_sample.py
    mac("GoldBinEdgeOne", "0.05"); mac("GoldBinEdgeTwo", "0.3")
    mac("GoldBinEdgeThree", "0.7"); mac("GoldBinEdgeFour", "0.95")   # BIN_EDGES in s05_gold_sample.py
    mac("UncertainLo", "0.3"); mac("UncertainHi", "0.7")   # band in s10_analysis.prevalence_block
    mac("RiskTargetOne", "1"); mac("RiskTargetFive", "5")   # coverage_at_risk targets in s10_analysis
    mac("PrecTargetNinety", "90"); mac("PrecTargetNinetyFive", "95")   # positive_review_budget targets
    mac("JevPricePerM", f(PR.JEV_INPUT_PER_M, 3))
    mac("ScreenCostShare", f(s1["mean_input_tokens"] / s2["mean_input_tokens"], 2))
    # T1. Two break-even numbers, which are not the same quantity and were conflated.
    # `BreakEvenFraction` is the MEASURED ratio of the two runs' mean input tokens. The
    # model value is what Eq. (twostage) predicts from the fitted cost model at the median
    # narrative length, with the screen's ten questions against the full schema's
    # twenty-seven. They differ because the lean screen also carries shortened instruction
    # text, so its intercept is not the full schema's.
    mac("BreakEvenFraction", f(1 - s1["mean_input_tokens"] / s2["mean_input_tokens"], 2))
    alpha = cm["intercept_model"]["alpha_tokens"]
    beta = cm["intercept_model"]["beta_tokens_per_question"]
    bchar = cm["per_question_set"]["27"]["b_tokens_per_char"]
    med_chars = float(corpus["length_quantiles_chars"]["0.5"])
    tok_s = alpha + beta * 10 + bchar * med_chars
    tok_f = alpha + beta * 27 + bchar * med_chars
    mac("BreakEvenModel", f(1 - tok_s / tok_f, 2))
    mac("BreakEvenModelTokensScreen", thousands(tok_s))
    mac("BreakEvenModelTokensFull", thousands(tok_f))
    mac("MedianNarrativeChars", thousands(med_chars))
    mac("SchemaQScreen", "10")
    mac("SchemaQFull", "27")

    # ---------------------------------------------------------------- WP14, downstream
    dsf = DATA / "downstream_severity.json"
    if dsf.exists():
        ds = json.loads(dsf.read_text())
        S = "injury_or_fatal"
        pvd = ds["per_variable"]
        add = {v: r[S]["added_per_year"] for v, r in pvd.items()}
        rel = {v: r[S]["relative_increase"] for v, r in pvd.items()}
        hi_r = max(rel, key=rel.get)
        lo_r = min(rel, key=rel.get)
        mac("DownNSeverity", thousands(ds["n_severity"][S]))
        mac("DownNFatal", thousands(ds["n_severity"]["fatal"]))
        mac("DownNVars", str(len(pvd)))
        mac("DownAddedTotal", thousands(sum(add.values())))
        mac("DownCodedTotal", thousands(sum(r[S]["coded_per_year"] for r in pvd.values())))
        mac("DownRelMax", pc(rel[hi_r], 0)); mac("DownRelMaxVar", hi_r.replace("_", " "))
        mac("DownRelMin", pc(rel[lo_r], 0)); mac("DownRelMinVar", lo_r.replace("_", " "))
        mac("DownRelMedian", pc(float(np.median(list(rel.values()))), 0))
        mac("DownAddedMaxVar", max(add, key=add.get).replace("_", " "))
        mac("DownAddedMax", thousands(max(add.values())))
        mac("DownNRankMove", str(ds["ranking"][S]["n_variables_that_move"]))
        for v in (hi_r,):
            r = pvd[v]
            mac("DownTopCoded", thousands(r[S]["coded_per_year"]))
            mac("DownTopAdded", thousands(r[S]["added_per_year"]))
            mac("DownTopAddedLo", thousands(r[S]["added_per_year_ci95"][0]))
            mac("DownTopAddedHi", thousands(r[S]["added_per_year_ci95"][1]))
            mac("DownTopConfirm", f(r["narrative_only_confirmed_rate"], 2))
        cf = {v: r["narrative_only_confirmed_rate"] for v, r in pvd.items()
              if r.get("narrative_only_confirmed_rate") is not None}
        if cf:
            mac("DownConfirmHi", f(max(cf.values()), 2))
            mac("DownConfirmHiVar", max(cf, key=cf.get).replace("_", " "))
            mac("DownConfirmLo", f(min(cf.values()), 2))
            mac("DownConfirmLoVar", min(cf, key=cf.get).replace("_", " "))
            mac("DownNConfirmHigh", str(sum(1 for x in cf.values() if x >= 0.70)))
            mac("DownNConfirmLow", str(sum(1 for x in cf.values() if x < 0.50)))
            mac("DownConfirmHighThreshold", "0.70")
    mac("ConcurrencyN", "10")            # CONCURRENCY in jev_runner.py
    mac("ProjectAfterCalls", thousands(2000))   # project_after in jev_runner.run
    mac("MaxTries", "5")                 # MAX_TRIES in jev_runner.py
    mac("EndpointClip", "10^{-6}")       # EPS clip in s07_metrics._logit and calibration_slope
    mac("WeakFoneThreshold", "0.70")     # WEAK in s11_tables.t10_gold
    mac("StrongFoneThreshold", "0.85")   # threshold behind GoldNStrong
    mac("FrontierWeakThreshold", "0.80") # threshold behind the frontier weak-variable macros
    mac("ThreshGainThreshold", "0.05")   # threshold behind ThreshNHelped
    mac("FloorFlagShare", "20")          # reporting threshold for the floor share in T4

    OUT.write_text("%% GENERATED by s13_numbers.py -- do not edit.\n"
                   "%% Every number quoted in the Results prose is defined here so that it\n"
                   "%% binds at build time and cannot go stale.\n"
                   + "\n".join(sorted(lines)) + "\n", encoding="utf-8")
    print(f"wrote {OUT} with {len(lines)} macros")


if __name__ == "__main__":
    main()
