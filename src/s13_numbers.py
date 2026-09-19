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

ROOT = Path(r"D:/OneDrive - Texas State University/AIT/Papers/Jev")
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
        rev = {v: p.get("review_per_year_90") for v, p in pbs.items()
               if p.get("review_per_year_90") is not None}
        if rev:
            mac("ReviewPerYearTotal", thousands(sum(rev.values())))
            mac("ReviewPerYearMax", thousands(max(rev.values())))
            mac("ReviewPerYearMaxVar", max(rev, key=rev.get).replace("_", " "))

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
        precs_g = {v: r["precision"] for v, r in G["per_variable"].items()}
        wv = min(precs_g, key=precs_g.get)
        mac("GoldWorstPrec", f(precs_g[wv], 2)); mac("GoldWorstPrecVar", wv.replace("_", " "))
        mac("RecalPlattBrier", f(rc["platt"]["brier"], 4))
        mac("RecalRawBrier", f(rc["none"]["brier"], 4))

        # The review budget against HUMAN labels. Section 5.5 computes it against coded
        # fields, where a narrative-only detection counts as a false positive and the budget
        # therefore saturates; this is the same quantity with that artefact removed.
        mac("GoldBudgetNinety", pc(Pg["review_budget_90"], 1))
        mac("GoldBudgetNinetyFive", pc(Pg["review_budget_95"], 1))
        b90 = {v: r["review_budget_90"] for v, r in G["per_variable"].items()}
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
            mac(k + "WeakVars",
                ", ".join(v.replace("_", " ")
                          for v in sorted(fv, key=fv.get) if fv[v] < 0.80) or "none")
            mac(k + "NWeak", str(sum(1 for x in fv.values() if x < 0.80)))
        # The two failing variables the frontier arms recover, which is what falsifies the
        # "the narrative does not say" reading of Jev's failures.
        for v, stem in (("intentional", "Intentional"), ("unbelted", "Unbelted")):
            for label, arm in FR["arms"].items():
                r = arm["per_variable"].get(v)
                if r:
                    mac(SHORT.get(label, "X") + stem + "Fone", f(r["f1"], 2))

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

    OUT.write_text("%% GENERATED by s13_numbers.py -- do not edit.\n"
                   "%% Every number quoted in the Results prose is defined here so that it\n"
                   "%% binds at build time and cannot go stale.\n"
                   + "\n".join(sorted(lines)) + "\n", encoding="utf-8")
    print(f"wrote {OUT} with {len(lines)} macros")


if __name__ == "__main__":
    main()
