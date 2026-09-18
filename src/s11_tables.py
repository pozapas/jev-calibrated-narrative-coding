"""Tables T2-T8 as LaTeX booktabs, from analysis.json plus the schema and run reports.

v1 emits T2, T3, T4, T5, T6, T7 (keyword arm only) and T8. T1 (literature positioning) is
prose written by hand; T9 (robustness) is v2.

Every table that compares against coded CRIS fields says AGREEMENT in its caption, per the
Section 7 desk-reject checklist, and proxy references are marked.
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:/OneDrive - Texas State University/AIT/Papers/Jev")
DATA = ROOT / "paper1" / "data"
OUT = ROOT / "paper1" / "outputs" / "tables"
NA = "--"


def bar(x, lo: float = 0.0, hi: float = 1.0) -> str:
    """An inline proportional bar for a metric, rendered next to its own number.

    Readers scan these tables for a pattern -- which variables fail, which model leads -- and
    comparing sixteen three-decimal numbers by eye is exactly the task a bar removes. The
    number stays in the adjacent column, so nothing is only encoded as length.
    """
    if x is None:
        return ""
    f = (float(x) - lo) / (hi - lo)
    f = min(max(f, 0.0), 1.0)
    return rf"\jevbar{{{f:.3f}}}{{{1 - f:.3f}}}"


def flag(text: str, bad: bool) -> str:
    """Colour a value that fails a threshold the caption states. Never decorative."""
    return rf"\jevflag{{{text}}}" if bad else text


def w(name: str, df: pd.DataFrame, caption: str, label: str, note: str = "",
      col_format: str | None = None, groups: list[tuple[str, int]] | None = None) -> None:
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
    ]
    # A spanning header row tells the reader that "prec./rec./F1" and "ECE/slope/z" are two
    # different questions about the same model, which eleven undifferentiated column heads
    # do not. The rules under each span are drawn short (lr) so they read as grouping marks
    # rather than as a second full-width rule competing with \midrule.
    if groups:
        assert sum(n for _, n in groups) == df.shape[1], \
            f"{name}: group spans {sum(n for _, n in groups)} != {df.shape[1]} columns"
        cells, rules, col = [], [], 1
        for title, n in groups:
            if title:
                cells.append(rf"\multicolumn{{{n}}}{{c}}{{\jevhead{{{title}}}}}")
                rules.append(rf"\cmidrule(lr){{{col}-{col + n - 1}}}")
            else:
                cells.append(rf"\multicolumn{{{n}}}{{c}}{{}}")
            col += n
        lines += [" & ".join(cells) + r" \\", "".join(rules)]
    lines += [" & ".join(rf"\jevhead{{{c}}}" for c in df.columns) + r" \\", r"\midrule"]
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
            "instruction": esc(instr[:88] + ("..." if len(instr) > 88 else "")),
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
            # The arm is named in the spanning header, so the per-column prefixes are
            # dropped: repeating "kw"/"Jev" six times is what pushed this table off the page.
            "prec.": num(k["keyword_precision_vs_coded"], 3),
            "rec.": num(k["keyword_recall_vs_coded"], 3),
            r"$\kappa$": num(k["keyword_kappa_vs_coded"], 3),
            "prec.\\ ": num(j.get("precision_vs_coded"), 3),
            "rec.\\ ": num(j.get("recall_vs_coded"), 3),
            r"$\kappa$\ ": num(j.get("cohens_kappa_vs_coded"), 3),
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
    WEAK = 0.70          # the threshold the caption states; below it a variable is flagged
    rows = []
    for v, r in sorted(G["per_variable"].items(), key=lambda kv: -kv[1]["f1"]):
        cod = a["vs_coded_fields"].get(v, {})
        gap = (r["ece"] / cod["ece_vs_coded"]) if cod.get("ece_vs_coded") else None
        bad = r["f1"] < WEAK
        rows.append({
            "variable": flag(esc(v), bad),
            "$n$": f"{r['n']:,}",
            r"base \%": pct(r["base_rate_weighted"], 1),
            "prec.": flag(num(r["precision"], 2), bad),
            "rec.": num(r["recall"], 2),
            "$F_1$": flag(num(r["f1"], 2), bad),
            # F1 is bar-scaled from 0.4 rather than 0: every variable scores above 0.45, so a
            # zero-based bar would compress the whole range into its right-hand third and show
            # nothing. The floor is stated in the caption.
            "": bar(r["f1"], 0.40, 1.0),
            r"$\kappa$": num(r["cohens_kappa"], 2),
            "ECE": num(r["ece"], 3),
            r"vs.\ coded": num(cod.get("ece_vs_coded"), 3),
            "gap": (rf"{gap:.1f}$\times$" if gap else NA),
            "slope": ("sep." if r["separated"] else num(r["calibration_slope"], 2)),
        })
    Pg = G["pooled"]
    rows.append({
        "variable": r"\textbf{pooled}", "$n$": f"{Pg['n']:,}",
        r"base \%": pct(Pg["base_rate_weighted"], 1),
        "prec.": num(Pg["precision"], 2), "rec.": num(Pg["recall"], 2),
        "$F_1$": num(Pg["f1"], 2), "": bar(Pg["f1"], 0.40, 1.0),
        r"$\kappa$": num(Pg["cohens_kappa"], 2),
        "ECE": num(Pg["ece"], 3), r"vs.\ coded": NA, "gap": NA,
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

    arms = [("Jev 1.13", F["jev"]["pooled"], "native")]
    arms += [(esc(label), arm["pooled"], "elicited") for label, arm in F["arms"].items()]

    # Bold the leader on each metric. With only three arms a reader will do this comparison
    # anyway; doing it for them is what makes the table answer "who wins, and on what" at a
    # glance, and it is the honest presentation because no arm leads on everything.
    best = {
        "precision": max(p["precision"] for _, p, _ in arms),
        "recall": max(p["recall"] for _, p, _ in arms),
        "f1": max(p["f1"] for _, p, _ in arms),
        "cohens_kappa": max(p["cohens_kappa"] for _, p, _ in arms),
        "ece": min(p["ece"] for _, p, _ in arms),
        "brier": min(p["brier"] for _, p, _ in arms),
    }

    def cell(p: dict, key: str, d: int) -> str:
        t = num(p[key], d)
        return rf"\jevbest{{{t}}}" if abs(p[key] - best[key]) < 1e-12 else t

    for label, p, kind in arms:
        rows.append({
            "model": label, "$n$": f"{p['n']:,}",
            "prec.": cell(p, "precision", 3), "rec.": cell(p, "recall", 3),
            "$F_1$": cell(p, "f1", 3), "": bar(p["f1"], 0.80, 1.0),
            r"$\kappa$": cell(p, "cohens_kappa", 3),
            "ECE": cell(p, "ece", 4), "Brier": cell(p, "brier", 4),
            "slope": num(p["calibration_slope"], 2),
            # Every arm is rejected, so z is flagged for all three rather than for an
            # outlier: the point of the column is that none of them passes.
            "$z$": flag(num(p["spiegelhalter_z"], 1), True),
            "prob.": kind,
        })
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
         r"gold set of Section~\ref{sec:gold} is in Table~\ref{tab:gold}, which also gives the factor by which "
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
      "the output grid and must not be read as a prevalence estimate (Section~\\ref{sec:landscape}). "
      "Confirmation \\% is the share of keyword hits Jev also calls positive. "
      "Uncertain \\% is the share with $p \\in [0.3, 0.7]$.")
    w("T5_calibration", t5_calibration(a),
      "Calibration against coded CRIS fields (agreement).", "tab:calibration",
      AGREE + " ECE is computed on the model's discrete output grid; intervals are stratified "
      "bootstrap (B = 1{,}000).",
      groups=[("", 3), ("Error", 5), ("Cox calibration", 3)])
    w("T6_selective", t6_selective(a),
      "Selective prediction and the human-review budget.", "tab:selective",
      AGREE + " $H(\\pi)$ is the share of \\emph{flagged positives} a human must review to "
      "reach precision $\\pi$; accuracy-based budgets are degenerate here because the "
      "positive class is rare (see Section~\\ref{sec:selective-results}).")
    w("T7_baseline", t7_baseline(a),
      "Keyword baseline versus Jev on the same coded reference.", "tab:baseline",
      AGREE + " McNemar tests paired per-narrative correctness. The frontier-model arm is "
      r"scored against human labels instead, in Table~\ref{tab:frontier}.",
      groups=[("", 1), ("Keyword rules", 3), ("Jev 1.13", 3), ("", 1)])
    w("T8_discrepancy", t8_discrepancy(a),
      "Discrepancy taxonomy against coded fields.", "tab:discrepancy",
      AGREE + " ``Narrative only'' is not an error rate: a factor stated in the narrative "
      "and absent from the coded field is the case this study sets out to find. Adjudication "
      "of 200 sampled discrepancies is reported in v2.")
    w("T10_gold", t10_gold(a),
      "Accuracy against the human gold set.", "tab:gold",
      r"Accuracy, not agreement: the reference is \GoldNUsable{} human judgements of the "
      r"criteria text the model was given, Horvitz--Thompson weighted to the population. "
      r"Bars scale $F_1$ from 0.40; \jevflag{red} marks $F_1<0.70$. ``gap'' is ECE divided "
      r"by the coded-field ECE of Table~\ref{tab:calibration}: the factor by which that "
      r"reference flatters the model. ``sep.'' is quasi-complete separation.",
      col_format="lrrrrr@{\\hspace{3pt}}lrrrrr",
      groups=[("", 3), ("Accuracy vs.\\ human labels", 5), ("Calibration", 4)])
    w("T11_recalibration", t11_recal(a),
      "Post-hoc recalibration on the gold set, split-half validated.", "tab:recal",
      "Each mapping is fitted on one half of the gold set and scored on the other, both "
      "ways; an in-sample recalibration would be circular. ``none'' uses the same "
      r"half-sample protocol, so it is comparable to the rows below it but not to the "
      r"full-sample ECE of Table~\ref{tab:gold}. Isotonic wins on ECE, Platt on Brier: the "
      r"calibration gain costs some sharpness.")
    w("T12_threshold", t12_threshold(a),
      "Per-variable decision threshold, retuned on the gold set and validated out of fold.",
      "tab:threshold",
      r"$n^{+}$ is the count of positive gold labels. $\tau^{*}$ maximises weighted $F_1$ "
      r"on the full gold set; the out-of-fold column instead picks $\tau$ on a random half "
      r"and scores it on the other, both ways, over 200 repeats, and is what retuning "
      r"actually buys. \jevflag{Unstable} marks a spread above 0.10 or fewer than 20 "
      r"positives, where $\tau^{*}$ is fitted to noise and must not be transported.",
      col_format="lrrrrrrl",
      groups=[("", 3), ("Retuned threshold, out of fold", 5)])
    if (DATA / "frontier" / "frontier_analysis.json").exists():
        w("T13_frontier", t13_frontier(a),
          "Jev against frontier generative models on the same human gold labels.",
          "tab:frontier",
          r"All arms answer the same narratives and variables under the same criteria text, "
          r"scored against the same labels with the same metric code. \jevbest{Bold} is the "
          r"leader per metric; bars scale $F_1$ from 0.80. ``prob.'' is whether the model "
          r"returns a probability natively or was asked to state one. Every $z$ is "
          r"\jevflag{red} because every arm is rejected, and they fail in opposite "
          r"directions: slope above one with $z<0$ over-states prevalence, below one with "
          r"$z>0$ the reverse. Accuracy is measured here; cost is not, and is given at list "
          r"prices in Figure~\ref{fig:frontier}. No arm is held out from the gold set.",
          col_format="lrrrrr@{\\hspace{3pt}}lrrrrl",
          groups=[("", 2), ("Agreement with human labels", 5), ("Calibration", 4), ("", 1)])


if __name__ == "__main__":
    main()
