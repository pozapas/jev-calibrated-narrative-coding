"""Tables T2-T8 as LaTeX booktabs, from analysis.json plus the schema and run reports.

v1 emits T2, T3, T4, T5, T6, T7 (keyword arm only) and T8. T1 (literature positioning) is
prose written by hand; T9 (robustness) is v2.

Every table that compares against coded CRIS fields says AGREEMENT in its caption, per the
§7 desk-reject checklist, and proxy references are marked.
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:/OneDrive - Texas State University/AIT/Papers/Jev")
DATA = ROOT / "paper1" / "data"
OUT = ROOT / "paper1" / "outputs" / "tables"
NA = "--"


def w(name: str, df: pd.DataFrame, caption: str, label: str, note: str = "",
      col_format: str | None = None) -> None:
    """Emit a booktabs table.

    Hand-rolled rather than pandas `to_latex(caption=...)`, which routes through Styler and
    needs jinja2 (not installed). Cell content is already LaTeX (the builders escape
    underscores and emit maths), so it is written through verbatim.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    fmt = col_format or ("l" + "r" * (df.shape[1] - 1))
    # Wide tables ran off the page: the 11-column ones overflowed the elsarticle text block
    # by over 200pt, which LaTeX reports as an overfull hbox and the reader sees as numbers
    # sliding into the margin. Tightening the inter-column padding and dropping one font step
    # keeps them inside the block without rescaling the table as an image, which would leave
    # its type a different size from the rest of the paper.
    # Column count is the usual cause, but the schema table is only five columns and still
    # overflows because two of them are paragraph columns of prose, so that is caught too.
    wide = df.shape[1] >= 7 or "p{" in fmt
    size = r"\scriptsize" if wide else r"\footnotesize"
    lines = [
        r"\begin{table}[!htbp]", r"\centering", size,
        (r"\setlength{\tabcolsep}{3pt}" if wide else ""),
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        # A wide table is centred into both margins at \jevtablewidth (see floatsetup.tex).
        # The notes go inside the same minipage so they align with the table rather than
        # hanging at the narrower text width.
        (r"\makebox[\textwidth][c]{\begin{minipage}{\jevtablewidth}\centering" if wide else ""),
        # threeparttable sets the note to the TABULAR's width. That is what we want for a
        # wide table and wrong for a narrow one: the recalibration table is three columns,
        # so its note was being squeezed into a tall ragged strip about a third of the page
        # wide. Narrow tables therefore get the note in a minipage at a readable measure
        # instead.
        (r"\begin{threeparttable}" if (note and wide) else ""),
        rf"\begin{{tabular}}{{{fmt}}}", r"\toprule",
        " & ".join(df.columns) + r" \\", r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join("" if pd.isna(v) else str(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note and wide:
        lines += [r"\begin{tablenotes}\footnotesize", r"\item " + note,
                  r"\end{tablenotes}", r"\end{threeparttable}"]
    elif note:
        lines += [r"\par\vspace{5pt}",
                  r"\begin{minipage}{0.86\textwidth}\footnotesize " + note + r"\end{minipage}"]
    if wide:
        lines.append(r"\end{minipage}}")
    lines.append(r"\end{table}")
    (OUT / f"{name}.tex").write_text("\n".join(x for x in lines if x) + "\n", encoding="utf-8")
    (OUT / f"{name}.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    print(f"  {name}: {len(df)} rows x {df.shape[1]} cols")


def esc(s: str) -> str:
    """Escape free text for LaTeX. Only for cells that are prose or raw identifiers --
    cells the builders compose out of maths must not pass through here."""
    out = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        out = out.replace(a, b)
    return out.replace("`", "'")          # backticks would render as open quotes


def pct(x, d=2):
    return NA if x is None else f"{100*float(x):.{d}f}"


def num(x, d=3):
    return NA if x is None else f"{float(x):.{d}f}"


def t2_schema() -> pd.DataFrame:
    s = json.loads((ROOT / "schemas" / "crash_factors_v1_1.json").read_text())
    gates = s.get("gates", {})
    rows = []
    for qid, q in s["questions"].items():
        crit = q.get("criteria")
        if isinstance(crit, dict):
            opts = ", ".join(list(crit)[:6]) + ("..." if len(crit) > 6 else "")
        elif isinstance(crit, list):
            opts = f"{len(crit)} ordered levels"
        else:
            opts = NA
        instr = q["instructions"]
        rows.append({
            "id": esc(qid),
            "type": q["type"],
            "instruction": esc(instr[:95] + ("..." if len(instr) > 95 else "")),
            "options / levels": esc(opts),
            "gate": esc(gates[qid]) if gates.get(qid) else NA,
        })
    return pd.DataFrame(rows)


def t3_run(a: dict) -> pd.DataFrame:
    c, r = a["corpus"], a["run"]
    cm = json.loads((DATA / "cost_model.json").read_text())
    rows = [
        ("Crash records in the narrative file", f"{c['n_rows_total']:,}"),
        ("Narratives empty", f"{c['n_narrative_empty']:,} ({pct(c['empty_share'])}\\%)"),
        ("Narratives analysed ($>$40 chars)", f"{c['n_kept_gt_40']:,}"),
        ("Narrative length, median / p95 / p99 (chars)",
         f"{c['length_quantiles_chars']['0.5']:.0f} / {c['length_quantiles_chars']['0.95']:.0f} "
         f"/ {c['length_quantiles_chars']['0.99']:.0f}"),
        ("Stage-2 narratives coded", f"{r['n_stage2']:,}"),
        ("  of which random stratum", f"{r['n_stage2_random']:,}"),
        ("Model", ", ".join(f"{k} ({v:,})" for k, v in r["models_seen"].items()).replace("_", r"\_")),
        ("Mean input tokens per narrative", f"{r['mean_input_tokens']:,.0f}"),
        ("Median latency (s)", f"{r['p50_latency_s']:.2f}"),
        ("Token model (27 questions)",
         f"$a$ = {cm['per_question_set']['27']['a_intercept_tokens']:.0f}, "
         f"$b$ = {cm['per_question_set']['27']['b_tokens_per_char']:.3f}/char, "
         f"$R^2$ = {cm['per_question_set']['27']['r2']:.3f}"),
        ("Schema overhead per question (tokens)",
         f"{cm['intercept_model']['beta_tokens_per_question']:.0f} "
         f"($R^2$ = {cm['intercept_model']['r2']:.3f})"),
        ("Cost per 1{,}000 narratives (USD)",
         f"{cm['per_question_set']['27']['cost_per_1k_narratives_usd']:.4f}"),
        ("Stage-2 spend (USD)", f"{r['total_cost_usd']:.2f}"),
    ]
    return pd.DataFrame(rows, columns=["Quantity", "Value"])


def t4_prevalence(a: dict) -> pd.DataFrame:
    rows = []
    for v, p in sorted(a["prevalence"].items(),
                       key=lambda kv: -(kv[1].get("jev_prevalence") or 0)):
        if p.get("jev_prevalence") is None:
            continue
        ci = p["jev_prevalence_ci95"]
        rows.append({
            "variable": esc(v),
            "keyword \\%": pct(p.get("keyword_population_prevalence"), 3),
            "Jev \\% ($p>0.5$)": f"{pct(p['jev_prevalence'], 3)} [{pct(ci[0], 3)}, {pct(ci[1], 3)}]",
            "E[p] \\%": (pct(p.get("expected_prevalence"), 3)
                        + (r"$^{\ddagger}$"
                           if (p.get("floor_contribution_to_expected") or 0) > 0.20 else "")),
            "floor \\%": pct(p.get("floor_contribution_to_expected"), 0),
            "confirm.\\ \\%": pct(p.get("jev_confirmation_among_keyword_hits")),
            "uncertain \\%": pct(p.get("uncertain_band_share")),
        })
    return pd.DataFrame(rows)


def t5_calibration(a: dict) -> pd.DataFrame:
    rows = []
    for v, r in a["vs_coded_fields"].items():
        ci = r.get("ece_vs_coded_ci95") or [None, None]
        rows.append({
            "variable": esc(v) + (r"$^{\dagger}$" if r["proxy_only"] else ""),
            "coded field": esc(r["coded_field"]),
            "$n$": f"{r['n']:,}",
            "ECE": f"{num(r['ece_vs_coded'], 4)} [{num(ci[0], 4)}, {num(ci[1], 4)}]",
            "MCE": num(r["mce_vs_coded"], 3),
            "Brier": num(r["brier_vs_coded"], 4),
            "rel.": num(r["reliability_vs_coded"], 4),
            "res.": num(r["resolution_vs_coded"], 4),
            "slope": num(r["calibration_slope_vs_coded"], 2),
            "intcpt": num(r["calibration_intercept_vs_coded"], 2),
            "$z$": num(r["spiegelhalter_z_vs_coded"], 1),
        })
    return pd.DataFrame(rows)


def t6_selective(a: dict) -> pd.DataFrame:
    rows = []
    for v, r in a["vs_coded_fields"].items():
        pb = r.get("positive_budget", {})
        rows.append({
            "variable": esc(v),
            "AURC": num(r["aurc_vs_coded"], 4),
            "excess": num(r["excess_aurc_vs_coded"], 4),
            "prec.": num(pb.get("precision_at_full_coverage"), 3),
            "flagged \\%": pct(pb.get("flagged_share_of_corpus"), 3),
            "$H$(0.90) \\%": pct(pb.get("review_budget_of_flagged_90"), 1),
            "$H$(0.95) \\%": pct(pb.get("review_budget_of_flagged_95"), 1),
            "review/yr": (f"{pb['review_per_year_90']:,.0f}"
                          if pb.get("review_per_year_90") is not None else NA),
        })
    return pd.DataFrame(rows)


def t7_baseline(a: dict) -> pd.DataFrame:
    rows = []
    for v, k in a["keyword_baseline"].items():
        j = a["vs_coded_fields"].get(v, {})
        mc = k.get("mcnemar_jev_vs_keyword", {})
        rows.append({
            "variable": esc(v),
            "kw prec.": num(k["keyword_precision_vs_coded"], 3),
            "kw rec.": num(k["keyword_recall_vs_coded"], 3),
            "kw $\\kappa$": num(k["keyword_kappa_vs_coded"], 3),
            "Jev prec.": num(j.get("precision_vs_coded"), 3),
            "Jev rec.": num(j.get("recall_vs_coded"), 3),
            "Jev $\\kappa$": num(j.get("cohens_kappa_vs_coded"), 3),
            "McNemar $p$": ("$<$0.001" if (mc.get("p_value") or 1) < 1e-3
                            else num(mc.get("p_value"), 3)),
        })
    return pd.DataFrame(rows)


def t8_discrepancy(a: dict) -> pd.DataFrame:
    rows = []
    for v, r in a["vs_coded_fields"].items():
        d = r["discrepancy_taxonomy"]
        rows.append({
            "variable": esc(v) + (r"$^{\dagger}$" if r["proxy_only"] else ""),
            "$n$": f"{r['n']:,}",
            "agreement \\%": pct(r["agreement_vs_coded"]),
            "$\\kappa$": num(r["cohens_kappa_vs_coded"], 3),
            "both": f"{d['both']:,}",
            "narr.\\ only": f"{d['narrative_only']:,}",
            "code only": f"{d['code_only']:,}",
            "neither": f"{d['neither']:,}",
        })
    return pd.DataFrame(rows)


def t10_gold(a: dict) -> pd.DataFrame:
    """Accuracy against the human gold set. Unlike every other table here, these are NOT
    agreement statistics -- the reference is human judgement of the same criteria text."""
    G = json.loads((DATA / "gold" / "gold_analysis.json").read_text())
    rows = []
    for v, r in sorted(G["per_variable"].items(), key=lambda kv: -kv[1]["f1"]):
        cod = a["vs_coded_fields"].get(v, {})
        gap = (r["ece"] / cod["ece_vs_coded"]) if cod.get("ece_vs_coded") else None
        rows.append({
            "variable": esc(v),
            "$n$": f"{r['n']:,}",
            r"base \%": pct(r["base_rate_weighted"], 2),
            "prec.": num(r["precision"], 2),
            "rec.": num(r["recall"], 2),
            "$F_1$": num(r["f1"], 2),
            r"$\kappa$": num(r["cohens_kappa"], 3),
            "ECE": num(r["ece"], 4),
            r"ECE$_{\text{coded}}$": num(cod.get("ece_vs_coded"), 4),
            "gap": (rf"{gap:.1f}$\times$" if gap else NA),
            "slope": ("sep." if r["separated"] else num(r["calibration_slope"], 2)),
        })
    Pg = G["pooled"]
    rows.append({
        "variable": r"\textbf{pooled}", "$n$": f"{Pg['n']:,}",
        r"base \%": pct(Pg["base_rate_weighted"], 2),
        "prec.": num(Pg["precision"], 2), "rec.": num(Pg["recall"], 2),
        "$F_1$": num(Pg["f1"], 2), r"$\kappa$": num(Pg["cohens_kappa"], 3),
        "ECE": num(Pg["ece"], 4), r"ECE$_{\text{coded}}$": NA, "gap": NA,
        "slope": num(Pg["calibration_slope"], 2),
    })
    return pd.DataFrame(rows)


def t11_recal(a: dict) -> pd.DataFrame:
    G = json.loads((DATA / "gold" / "gold_analysis.json").read_text())
    rc = G["recalibration"]
    name = {"none": "none (raw model output)", "platt": "Platt (logistic on the logit)",
            "isotonic": "isotonic"}
    return pd.DataFrame([{"mapping": name[k], "ECE": num(rc[k]["ece"], 4),
                          "Brier": num(rc[k]["brier"], 4)}
                         for k in ("none", "platt", "isotonic")])


def t13_frontier(a: dict) -> pd.DataFrame:
    """Jev against frontier generative models on the SAME gold labels.

    Accuracy only. The cost column is deliberately absent: cost is not measured by the
    harness that produced these accuracies, and putting an unmeasured dollar figure beside a
    measured F1 in one table invites the reader to treat both as equally sourced. Cost is
    reported separately, analytically, in Figure~\\ref{fig:frontier}.
    """
    F = json.loads((DATA / "frontier" / "frontier_analysis.json").read_text())
    rows = []

    def row(label: str, p: dict, extra: str) -> dict:
        return {"model": label, "$n$": f"{p['n']:,}",
                "prec.": num(p["precision"], 3), "rec.": num(p["recall"], 3),
                "$F_1$": num(p["f1"], 3), r"$\kappa$": num(p["cohens_kappa"], 3),
                "ECE": num(p["ece"], 4), "Brier": num(p["brier"], 4),
                "slope": num(p["calibration_slope"], 2),
                "$z$": num(p["spiegelhalter_z"], 1),
                "$p$": extra}

    rows.append(row("Jev 1.13", F["jev"]["pooled"], "native"))
    for label, arm in F["arms"].items():
        rows.append(row(esc(label), arm["pooled"], "elicited"))
    return pd.DataFrame(rows)


def t12_threshold(a: dict) -> pd.DataFrame:
    """Evidence for a NEGATIVE claim: retuning tau does not rescue the weak variables.

    A reader's first reaction to a variable with precision 0.32 at recall 1.00 is that the
    threshold is simply wrong. That is a testable proposition and this table is the test, so
    it reports every variable rather than the three that fail -- a table showing only the
    failures could not establish that the well-behaved variables are not being helped either.
    """
    G = json.loads((DATA / "gold" / "gold_analysis.json").read_text())
    T = G["threshold_tuning"]
    rows = []
    for v, r in sorted(T.items(), key=lambda kv: kv[1]["f1_at_half"]):
        rows.append({
            "variable": esc(v),
            "$n^{+}$": str(r["n_positive"]),
            r"$F_1$ at $\tau=0.5$": num(r["f1_at_half"], 2),
            r"tuned $\tau^{*}$": num(r["tuned_threshold"], 2),
            r"$F_1$ tuned, out of fold": num(r["f1_tuned_oof"], 2),
            "s.d.": num(r["f1_tuned_oof_sd"], 2),
            "change": ("%+.2f" % r["gain"]),
            "stable?": ("no" if r["unstable"] else "yes"),
        })
    return pd.DataFrame(rows)


AGREE = ("Comparisons are with coded CRIS fields, which are themselves incomplete and noisy; "
         r"these are measures of \emph{agreement}, not accuracy. Accuracy against the human "
         r"gold set of \S4.4 is in Table~\ref{tab:gold}, which also gives the factor by which "
         r"this reference flatters the model. $^{\dagger}$ the coded field is a proxy for the "
         "narrative concept rather than a direct match.")


def main() -> None:
    a = json.loads((DATA / "analysis.json").read_text())
    print("writing tables:")
    w("T2_schema", t2_schema(),
      "The 27-question schema (v1.1), with gates.", "tab:schema",
      "Instructions are truncated for space; Appendix A gives the verbatim JSON. "
      "Gated questions are interpreted only when their gate exceeds $\\tau_g=0.5$.",
      col_format=r"llp{0.34\textwidth}p{0.20\textwidth}l")
    w("T3_run", t3_run(a), "Corpus and run statistics.", "tab:run",
      "Prices are TypeSafe list prices accessed 2026-09-17 and are quoted, not endorsed.")
    w("T4_prevalence", t4_prevalence(a),
      "Prevalence estimates on the Stage-2 random stratum.", "tab:prevalence",
      "Keyword \\% is over all 5{,}018{,}080 narratives. Jev \\% thresholds at $p>0.5$ with "
      "Clopper--Pearson intervals; E[p] is the mean probability, which a calibrated model "
      "would match to the true prevalence. ``floor'' is the share of E[p] contributed by "
      "narratives sitting at the smallest probability the model can express, $p=0.01$; "
      "$^{\\ddagger}$ marks variables where that exceeds 20\\%, for which E[p] is inflated by "
      "the output grid and must not be read as a prevalence estimate (\\S5.2). "
      "Confirmation \\% is the share of keyword hits Jev also calls positive. "
      "Uncertain \\% is the share with $p \\in [0.3, 0.7]$.")
    w("T5_calibration", t5_calibration(a),
      "Calibration against coded CRIS fields (agreement).", "tab:calibration",
      AGREE + " ECE is computed on the model's discrete output grid; intervals are stratified "
      "bootstrap (B = 1{,}000).")
    w("T6_selective", t6_selective(a),
      "Selective prediction and the human-review budget.", "tab:selective",
      AGREE + " $H(\\pi)$ is the share of \\emph{flagged positives} a human must review to "
      "reach precision $\\pi$; accuracy-based budgets are degenerate here because the "
      "positive class is rare (see \\S5.4).")
    w("T7_baseline", t7_baseline(a),
      "Keyword baseline versus Jev on the same coded reference.", "tab:baseline",
      AGREE + " The frontier-LLM arm is v2. McNemar tests paired per-narrative correctness.")
    w("T8_discrepancy", t8_discrepancy(a),
      "Discrepancy taxonomy against coded fields.", "tab:discrepancy",
      AGREE + " ``Narrative only'' is not an error rate: a factor stated in the narrative "
      "and absent from the coded field is the case this study sets out to find. Adjudication "
      "of 200 sampled discrepancies is reported in v2.")
    w("T10_gold", t10_gold(a),
      "Accuracy against the human gold set.", "tab:gold",
      "These are ACCURACY statistics: the reference is human judgement of the same criteria "
      "text the model was given, from "
      r"\GoldNUsable{} usable judgements by \CoderN{} independent coders. Rates are "
      "Horvitz--Thompson weighted to the population, because the gold frame over-samples "
      r"high-probability narratives by design. ECE$_{\text{coded}}$ repeats the "
      r"coded-field value from Table~\ref{tab:calibration} and ``gap'' is their ratio: the "
      "factor by which the noisy reference flatters the model. ``sep.'' marks quasi-complete "
      "separation, where the calibration slope has no finite estimate.",
      col_format="lrrrrrrrrrr")
    w("T11_recalibration", t11_recal(a),
      "Post-hoc recalibration on the gold set, split-half validated.", "tab:recal",
      "Each mapping is fitted on one half of the gold set and scored on the other, in both "
      "directions; the reported values are the mean of the two out-of-fold folds. An "
      "in-sample recalibration would be circular. The ``none'' row is scored under the "
      "same half-sample protocol so that it is comparable to the rows below it, and is "
      r"therefore not identical to the full-sample pooled ECE of Table~\ref{tab:gold}. "
      "Isotonic attains the lower ECE; Platt attains the lower Brier, so it trades less "
      "sharpness for its calibration gain.")
    w("T12_threshold", t12_threshold(a),
      "Per-variable decision threshold, retuned on the gold set and validated out of fold.",
      "tab:threshold",
      r"$n^{+}$ is the number of positive gold labels. $\tau^{*}$ is chosen to maximise "
      r"weighted $F_1$ on the full gold set and is the value one would deploy; the "
      r"out-of-fold column instead chooses $\tau$ on a random half and scores it on the "
      r"other, both directions, averaged over 200 repeats, and is the honest estimate of "
      r"what retuning buys. ``change'' is that estimate minus $F_1$ at $\tau=0.5$. A "
      r"variable is marked unstable when the out-of-fold spread exceeds 0.10 or fewer than "
      r"20 positives are available, in which case $\tau^{*}$ is fitted to noise and should "
      r"not be transported.",
      col_format="lrrrrrrl")
    if (DATA / "frontier" / "frontier_analysis.json").exists():
        w("T13_frontier", t13_frontier(a),
          "Jev against frontier generative models on the same human gold labels.",
          "tab:frontier",
          r"All arms answer the same narratives, for the same variables, under the same "
          r"criteria text, and are scored against the same human labels with the same "
          r"Horvitz--Thompson weights and the same metric code. ``interface'' distinguishes a "
          r"probability the model returns natively from one a generative model was asked to "
          r"state. Accuracy here is measured; cost is not, and is reported separately in "
          r"Figure~\ref{fig:frontier} from measured token counts at published list prices. "
          r"No arm is held out from the gold set, which is also the set the "
          r"Table~\ref{tab:recal} recalibration maps were fitted on; the comparison between "
          r"arms is therefore like-for-like, but none of these figures is an out-of-sample "
          r"estimate. Every arm is rejected by the Spiegelhalter test, and the sign of $z$ "
          r"together with the calibration slope shows they fail differently: a slope above "
          r"one with $z<0$ is a model whose probabilities are not extreme enough and which "
          r"over-states prevalence, a slope below one with $z>0$ the reverse.",
          col_format="lrrrrrrrrrl")


if __name__ == "__main__":
    main()
