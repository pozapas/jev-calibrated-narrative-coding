"""Tables T2-T8 as LaTeX booktabs, from analysis.json plus the schema and run reports.

v1 emits T2, T3, T4, T5, T6, T7 (keyword arm only) and T8. T1 (literature positioning) is
prose written by hand; T9 (robustness) is v2.

Every table that compares against coded CRIS fields says AGREEMENT in its caption, per the
Section 7 desk-reject checklist, and proxy references are marked.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import pandas as pd

import pricing as P

ROOT = Path(__file__).resolve().parents[2]   # project root, relative to this file (was a hard-coded D: path)
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


# The text block of the CAS single-column template, measured from the build. Every table is
# laid out to fit inside it upright, because a reader should not have to turn the page.
TEXTWIDTH_PT = 468.3
TABCOLSEP_PT = 4.0          # starting separation; narrowed automatically for a wide table
MIN_TABCOLSEP_PT = 3.0
CHAR_PT = 4.45              # mean advance of a character at \small in the document serif
SAFETY_PT = 8.0             # slack kept inside the text block, so a rounding error does not clip
ROTWIDTH_PT = 637.3          # a rotated table lies across the height of the text block
WRAP_CHARS = 26             # a column wider than this wraps in a p{} column instead of running on


def _plain(cell: str) -> str:
    """The visible text of a cell, with LaTeX markup removed, for width estimation."""
    s = str(cell)
    s = re.sub(r"\\jevbar\{[^}]*\}\{[^}]*\}", "xxxx", s)     # the inline bar occupies ~4 chars
    s = re.sub(r"\\(jevbr|newline)\b", "\n", s)
    s = re.sub(r"\\[A-Za-z]+\*?", "", s)                      # other control sequences
    s = re.sub(r"[{}$\\]", "", s)
    s = s.replace("~", " ").replace("&", "&")
    return s


def _col_widths(df) -> list:
    """Longest visible line in each column, in characters, header included."""
    out = []
    for c in df.columns:
        w = max(len(ln) for ln in _plain(c).split("\n"))
        for v in df[c]:
            if pd.isna(v):
                continue
            w = max(w, max(len(ln) for ln in _plain(v).split("\n")))
        out.append(w)
    return out


MIN_COL_CHARS = 5           # no column is squeezed below this, whatever the arithmetic says


def _min_token(df, i: int) -> float:
    """The longest unbreakable run in a column, which is the floor on its width.

    A p{} column wraps at spaces, so it can never be narrower than its longest single word
    without the content sticking out past the rule. Measuring that word is what keeps the
    automatic sizing from trading one overflow for another.
    """
    longest = 0
    col = df.columns[i]
    for cell in [col, *df[col].tolist()]:
        if pd.isna(cell):
            continue
        for line in _plain(cell).split("\n"):
            for tok in line.split():
                longest = max(longest, len(tok))
    return float(longest)


def auto_format(df, first_left: bool = True,
                width_pt: float = TEXTWIDTH_PT) -> tuple:
    """Choose a column format that fits the text block, and the separation it needs.

    A short column is set flush right, which is what a reader wants for a number. A column
    whose content is long is given a fixed width and wraps inside it. The fixed widths are not
    derived from the character estimate directly, because that estimate appears on both sides
    of the arithmetic and a conservative value then makes the table wider rather than narrower.
    They are solved for instead, so that the columns add up to the space actually left over
    once the numeric columns and the separation have taken theirs. No column goes below its
    longest unbreakable word, which is what stops a value being clipped by the right rule.
    """
    widths = _col_widths(df)
    n = len(widths)
    floors = [max(_min_token(df, i), MIN_COL_CHARS) for i in range(n)]

    def fits(wrapset, sep):
        fixed = sum(widths[i] for i in range(n) if i not in wrapset) * CHAR_PT
        return width_pt - SAFETY_PT - 2 * sep * n - fixed

    # columns that are long enough to be prose wrap from the start
    wrapset = {i for i, wd in enumerate(widths) if wd > WRAP_CHARS}
    sep = TABCOLSEP_PT

    def room():
        """Space left for the wrapped columns, or the overall slack when none wrap."""
        avail = fits(wrapset, sep)
        return avail - sum(widths[i] for i in wrapset) * CHAR_PT

    # give back separation first, then start wrapping the widest remaining numeric column
    while room() < 0 and sep > MIN_TABCOLSEP_PT:
        sep = max(MIN_TABCOLSEP_PT, sep - 0.5)
    while room() < 0:
        cand = [i for i in range(n) if i not in wrapset and widths[i] > floors[i] + 1]
        if not cand:
            break
        wrapset.add(max(cand, key=lambda i: widths[i] - floors[i]))

    parts = []
    if wrapset:
        avail = fits(wrapset, sep)
        floor_pt = {i: floors[i] * CHAR_PT for i in wrapset}
        need = sum(widths[i] for i in wrapset) * CHAR_PT
        spare = avail - sum(floor_pt.values())
        extra = max(need - sum(floor_pt.values()), 1e-6)
        scale = min(1.0, max(spare, 0.0) / extra)
        for i in range(n):
            if i in wrapset:
                pt = floor_pt[i] + scale * (widths[i] * CHAR_PT - floor_pt[i])
                align = "raggedleft" if _is_numeric_col(df, i) else "raggedright"
                parts.append(
                    f">{{\\{align}\\arraybackslash}}p{{{max(pt, floor_pt[i]):.1f}pt}}")
            else:
                parts.append("l" if (i == 0 and first_left) else "r")
    else:
        for i in range(n):
            parts.append("l" if (i == 0 and first_left) else "r")
    return "".join(parts), sep


def _is_numeric_col(df, i: int) -> bool:
    """Whether a column holds numbers, so a wrapped cell should stay flush right.

    A p{} column is a paragraph box and sets its content flush left by default, which breaks
    the alignment of a column of figures. Deciding this from the content keeps the numeric
    columns reading as numbers even when an interval forces them to wrap.
    """
    col = df.columns[i]
    vals = [str(v) for v in df[col].tolist() if not pd.isna(v) and str(v).strip() not in ("", NA)]
    if not vals:
        return False
    numeric = sum(1 for v in vals
                  if re.match(r"^[\s$\\]*[-+\[(]?\s*\d", _plain(v).strip()))
    return numeric >= 0.7 * len(vals)


def _col_kinds(fmt: str) -> list:
    """The per-column specifiers of a tabular format, so a header can be set to match.

    A header in a p{} column must wrap to that column's width. \jevhead sets its argument in
    a mini-tabular whose natural width is the longest line it contains, which cannot shrink,
    so a long header in a narrow p{} column pushed the row past the right rule. Knowing which
    columns are p{} lets the emitter use the wrapping form for exactly those.
    """
    kinds, i = [], 0
    while i < len(fmt):
        c = fmt[i]
        if c in "lrc":
            kinds.append(c); i += 1
        elif c == "p":
            j = fmt.index("}", i) + 1
            kinds.append("p"); i = j
        elif c in "@>":
            depth, i = 0, i + 1
            while i < len(fmt):
                if fmt[i] == "{":
                    depth += 1
                elif fmt[i] == "}":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
        else:
            i += 1
    return kinds


def w(name: str, df: pd.DataFrame, caption: str, label: str, note: str = "",
      col_format: str | None = None, groups: list[tuple[str, int]] | None = None,
      landscape: bool = False, longtable: bool = False) -> None:
    r"""Emit a booktabs table upright, in the document's text font.

    Three things here are deliberate and were all fixed together. The table is upright rather
    than rotated, because a rotated table takes a page to itself and the reader has to turn the
    paper. The body is set with \rmfamily, because the class opens every table with \sffamily
    and the surrounding text is serif; a rotated table escaped that and so the old build mixed
    two faces across the paper. The column format is computed from the content rather than
    guessed, because a hand-written format is what let cells run past the right rule.

    A rotated table is still available through `landscape`, and Table D.1 uses it.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    if col_format:
        fmt, sep = col_format, TABCOLSEP_PT
    else:
        fmt, sep = auto_format(
            df, width_pt=ROTWIDTH_PT if landscape else TEXTWIDTH_PT)

    head = []
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
        head += [" & ".join(cells) + r" \\", "".join(rules)]
    # A header that wants two lines asks for \newline, which does nothing in an l or r
    # column. \jevhead sets its argument in a small tabular, where \jevbr is the row break.
    kinds = _col_kinds(fmt)
    cells_h = []
    for k, c in enumerate(df.columns):
        wrapcol = k < len(kinds) and kinds[k] == "p"
        if wrapcol:
            # a wrapping column takes the wrapping header, which honours an explicit
            # line break and otherwise fills the column width
            cells_h.append(rf"\jevheadp{{{c}}}")
        else:
            cells_h.append(
                rf"\jevhead{{{c.replace(chr(92) + 'newline', chr(92) + 'jevbr ')}}}")
    head += [" & ".join(cells_h) + r" \\", r"\midrule"]
    body = [" & ".join("" if pd.isna(v) else str(v) for v in row) + r" \\"
            for _, row in df.iterrows()]

    note_block = []
    if note:
        # threeparttable sets the notes to the width of the tabular, so they align with the
        # rules rather than with the text block, and every note opens with the same word.
        note_block = [r"\begin{tablenotes}[flushleft]\small\rmfamily",
                      r"\item \textit{Note:}~" + note,
                      r"\end{tablenotes}"]

    if longtable:
        ncol = df.shape[1]
        # The note belongs to the last foot of the table rather than after it. Set after
        # \end{longtable} it is an ordinary paragraph, and when the table happens to end near
        # the foot of a page the note is carried to the next one and sits alone on an
        # otherwise empty page, which is what happened to the schema table.
        last_foot = [r"\bottomrule"]
        if note:
            last_foot.append(
                rf"\multicolumn{{{ncol}}}{{@{{}}p{{\linewidth}}@{{}}}}"
                r"{\rmfamily\small\vspace{2pt}\textit{Note:}~" + note + r"}\\")
        lines = [r"\begin{small}\rmfamily",
                 rf"\setlength{{\tabcolsep}}{{{sep:.1f}pt}}",
                 rf"\begin{{longtable}}{{{fmt}}}",
                 # the caption travels inside the first head, so it cannot be left behind on
                 # the previous page when the table breaks
                 rf"\caption{{{caption}}}\label{{{label}}}\\",
                 r"\toprule", *head, r"\endfirsthead",
                 rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\rmfamily\small\textit{{Table \ref{{{label}}}, continued}}}}\\",
                 r"\toprule", *head, r"\endhead",
                 r"\bottomrule",
                 rf"\multicolumn{{{ncol}}}{{@{{}}r}}{{\rmfamily\small\textit{{continued on the next page}}}}\\",
                 r"\endfoot", *last_foot, r"\endlastfoot", *body,
                 r"\end{longtable}", r"\end{small}"]
        (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (OUT / f"{name}.csv").write_text(df.to_csv(index=False), encoding="utf-8")
        print(f"  {name}: {len(df)} rows x {ncol} cols (longtable)")
        return

    env = "sidewaystable" if landscape else "table"
    pos = "!htbp" if landscape else "pos=!htbp"
    # The class styles the caption of its own table environment and of nothing else, so a
    # rotated table gets the caption written out to match rather than the LaTeX default.
    cap = (rf"\jevrotcaption{{{label}}}{{{caption}}}" if landscape
           else rf"\caption{{{caption}}}" + "\n" + rf"\label{{{label}}}")
    lines = [rf"\begin{{{env}}}[{pos}]", r"\centering", r"\rmfamily\small",
             rf"\setlength{{\tabcolsep}}{{{sep:.1f}pt}}",
             cap,
             (r"\begin{threeparttable}" if note else ""),
             rf"\begin{{tabular}}{{{fmt}}}", r"\toprule", *head, *body,
             r"\bottomrule", r"\end{tabular}",
             *note_block,
             (r"\end{threeparttable}" if note else ""),
             rf"\end{{{env}}}"]
    (OUT / f"{name}.tex").write_text("\n".join(x for x in lines if x) + "\n", encoding="utf-8")
    (OUT / f"{name}.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    est = sum(_col_widths(df)) * CHAR_PT + 2 * sep * df.shape[1]
    flag = "  WIDE" if est > TEXTWIDTH_PT and "p{" not in fmt else ""
    print(f"  {name}: {len(df)} rows x {df.shape[1]} cols ({env}, sep {sep:.1f}pt, "
          f"est {est:.0f}pt){flag}")


def esc(s: str) -> str:
    """Escape free text for LaTeX. Only for cells that are prose or raw identifiers --
    cells the builders compose out of maths must not pass through here."""
    # T12. The replacements are not commutative. Substituting the backslash first inserts
    # braces that the later brace rules then escape again, which is why the keyword table
    # printed \textbackslash\{\}b instead of \b. Each replacement therefore writes a sentinel
    # that no later rule can match, and the sentinels are expanded at the end.
    out = str(s)
    subs = (("\\", "<<BS>>"), ("{", "<<LB>>"), ("}", "<<RB>>"),
            ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
            ("~", "<<TILDE>>"), ("^", "<<CARET>>"))
    for a, b in subs:
        out = out.replace(a, b)
    for a, b in (("<<BS>>", r"\textbackslash{}"), ("<<LB>>", r"\{"), ("<<RB>>", r"\}"),
                 ("<<TILDE>>", r"\textasciitilde{}"), ("<<CARET>>", r"\textasciicircum{}")):
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
        ("Narratives analyzed ($>$40 chars)", f"{c['n_kept_gt_40']:,}"),
        ("Narrative length, median / p95 / p99 (chars)",
         f"{c['length_quantiles_chars']['0.5']:.0f} / {c['length_quantiles_chars']['0.95']:.0f} "
         f"/ {c['length_quantiles_chars']['0.99']:.0f}"),
        ("Stage-2 narratives coded", f"{r['n_stage2']:,}"),
        ("  of which random stratum", f"{r['n_stage2_random']:,}"),
        ("Model", ", ".join(f"{k} ({v:,})" for k, v in r["models_seen"].items()).replace("_", r"\_")),
        ("Mean input tokens per narrative", f"{r['mean_input_tokens']:,.0f}"),
        ("Median latency (s)", f"{r['p50_latency_s']:.2f}"),
        ("Token model, direct fit at 27 questions",
         f"$a$ = {cm['per_question_set']['27']['a_intercept_tokens']:.0f}, "
         f"$b$ = {cm['per_question_set']['27']['b_tokens_per_char']:.3f}/char, "
         f"$R^2$ = {cm['per_question_set']['27']['r2']:.3f}"),
        ("Schema overhead per question (tokens)",
         f"{cm['intercept_model']['beta_tokens_per_question']:.0f} "
         f"($R^2$ = {cm['intercept_model']['r2']:.3f})"),
        ("Cost per 1{,}000 narratives, realised (USD)",
         f"{P.jev_cost_per_1k(P.run_input_tokens()):.4f}"),
        ("Stage-2 spend (USD)", f"{r['total_cost_usd']:.2f}"),
    ]
    return pd.DataFrame(rows, columns=["Quantity", "Value"])


def t4_prevalence(a: dict) -> pd.DataFrame:
    """T12. Every column names the sample it is computed on and the denominator it uses.

    The keyword share is over the whole corpus, the thresholded and mean probabilities are
    over the Stage-2 random stratum, and the confirmation rate is among the keyword hits that
    fall inside that stratum, so it needs its own count and its own interval rather than
    sitting in a column the reader will read against the corpus.
    """
    n_corpus = a["corpus"]["n_kept_gt_40"]
    n_rand = a["run"]["n_stage2_random"]
    rows = []
    for v, p in sorted(a["prevalence"].items(),
                       key=lambda kv: -(kv[1].get("jev_prevalence") or 0)):
        if p.get("jev_prevalence") is None:
            continue
        ci = p["jev_prevalence_ci95"]
        cc = p.get("jev_confirmation_ci95") or [None, None]
        nh = p.get("n_keyword_hits_in_random")
        conf = p.get("jev_confirmation_among_keyword_hits")
        rows.append({
            "variable": esc(v),
            f"keyword \\%\\newline of {n_corpus:,}".replace(",", "{,}"):
                pct(p.get("keyword_population_prevalence"), 3),
            f"$p>0.5$ \\%\\newline of {n_rand:,}".replace(",", "{,}"):
                f"{pct(p['jev_prevalence'], 3)} [{pct(ci[0], 3)}, {pct(ci[1], 3)}]",
            f"E[p] \\%\\newline of {n_rand:,}".replace(",", "{,}"):
                (pct(p.get("expected_prevalence"), 3)
                 + (r"$^{\ddagger}$"
                    if (p.get("floor_contribution_to_expected") or 0) > 0.20 else "")),
            "floor \\%": pct(p.get("floor_contribution_to_expected"), 0),
            "hits": f"{nh:,}".replace(",", "{,}") if nh is not None else NA,
            "confirmed \\%\\newline of hits":
                (f"{pct(conf)} [{pct(cc[0])}, {pct(cc[1])}]" if conf is not None else NA),
            f"uncertain \\%\\newline of {n_rand:,}".replace(",", "{,}"):
                pct(p.get("uncertain_band_share")),
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
    gold = json.loads((DATA / "gold" / "gold_analysis.json").read_text())["per_variable"]
    rows = []
    for v, r in a["vs_coded_fields"].items():
        pb = r.get("positive_budget", {})
        hb = (gold.get(v, {}).get("held_out_budget_90") or {})
        rows.append({
            "variable": esc(v),
            "AURC": num(r["aurc_vs_coded"], 4),
            "excess": num(r["excess_aurc_vs_coded"], 4),
            "prec.": num(pb.get("precision_at_full_coverage"), 3),
            "flagged \\%": pct(pb.get("flagged_share_of_corpus"), 3),
            "$H$(0.90) \\%": pct(pb.get("review_budget_of_flagged_90"), 1),
            "$H$(0.95) \\%": pct(pb.get("review_budget_of_flagged_95"), 1),
            # WP4. The human-reference budget is cross-fitted, so the held-out value and the
            # in-sample value both appear and the gap between them is the optimism.
            "human $H$(0.90) \\%\\newline held out": pct(
                hb.get("review_budget_heldout_mean"), 1),
            "human $H$(0.90) \\%\\newline in sample": pct(
                hb.get("review_budget_insample"), 1),
            "review decisions/yr\\newline at 90\\% precision": (
                f"{pb['review_per_year_90']:,.0f}".replace(",", "{,}")
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
            # WP1, T3. Both kappas are population quantities on the same variable, the coded
            # one over the whole random stratum and the human one calibration-weighted to it,
            # so the two columns are comparable and the header says which is which.
            r"$\kappa$\newline weighted": num(r["cohens_kappa_weighted"], 2),
            r"$\kappa$\newline vs.\ coded": num(cod.get("cohens_kappa_vs_coded"), 2),
            "ECE": num(r["ece"], 3),
            r"vs.\ coded": num(cod.get("ece_vs_coded"), 3),
            "gap": (rf"{gap:.1f}$\times$" if gap else NA),
            "slope": ("sep." if r["separated"] else num(r["calibration_slope"], 2)),
            r"$H(0.90)$ \%\newline held out": pct(
                (r.get("held_out_budget_90") or {}).get("review_budget_heldout_mean"), 0),
        })
    Pg = G["pooled"]
    rows.append({
        "variable": r"\textbf{pooled}", "$n$": f"{Pg['n']:,}",
        r"base \%": pct(Pg["base_rate_weighted"], 1),
        "prec.": num(Pg["precision"], 2), "rec.": num(Pg["recall"], 2),
        "$F_1$": num(Pg["f1"], 2), "": bar(Pg["f1"], 0.40, 1.0),
        r"$\kappa$\newline weighted": num(Pg["cohens_kappa_weighted"], 2),
        r"$\kappa$\newline vs.\ coded": NA,
        "ECE": num(Pg["ece"], 3), r"vs.\ coded": NA, "gap": NA,
        "slope": num(Pg["calibration_slope"], 2),
        r"$H(0.90)$ \%\newline held out": pct(
            Pg["held_out_budget_90"]["review_budget_heldout_mean"], 0),
    })
    return pd.DataFrame(rows)


def t11_recal(a: dict) -> pd.DataFrame:
    """WP3. The pooled map is the proof of concept, the per-variable maps are the evidence.

    An analyst applies a recalibration map to one variable at a time, so a pooled map that
    lowers the aggregate calibration error is not by itself the deployment result. The pooled
    block keeps its three rows and the per-variable block follows, with the variables that
    carry too few positives for their own map named rather than silently pooled.
    """
    G = json.loads((DATA / "gold" / "gold_analysis.json").read_text())
    rc = G["recalibration"]
    name = {"none": "none (raw model output)", "platt": "Platt (logistic on the logit)",
            "isotonic": "isotonic"}
    ci = rc.get("pooled_ci95", {})
    rows = []
    for k in ("none", "platt", "isotonic"):
        c = ci.get(f"{k}_ece") or [None, None]
        rows.append({
            "scope": "pooled", "mapping": name[k],
            "$n$": f"{G['n_usable']:,}".replace(",", "{,}"),
            "ECE, held out": (f"{num(rc['pooled'][k + '_ece'], 4)} "
                              f"[{num(c[0], 4)}, {num(c[1], 4)}]"),
            "Brier": num(rc["pooled"][f"{k}_brier"], 4),
            "slope": num(rc["pooled"].get(f"{k}_slope"), 2),
            "intercept": num(rc["pooled"].get(f"{k}_intercept"), 2),
        })
    for v, r in sorted(rc["per_variable"].items(),
                       key=lambda kv: kv[1].get("platt_ece") or 0):
        rows.append({
            "scope": esc(v), "mapping": "raw / Platt",
            "$n$": f"{r['n']:,}".replace(",", "{,}"),
            "ECE, held out": f"{num(r.get('none_ece'), 4)} / {num(r.get('platt_ece'), 4)}",
            "Brier": f"{num(r.get('none_brier'), 4)} / {num(r.get('platt_brier'), 4)}",
            "slope": num(r.get("platt_slope"), 2),
            "intercept": num(r.get("platt_intercept"), 2),
        })
    return pd.DataFrame(rows)


def tc1_exclusions(a: dict) -> pd.DataFrame:
    """WP5. What the excluded pairs are, and what the headline numbers would be under three
    treatments of them. The first block is the composition of the exclusions, the second is
    the sensitivity, and both are needed because a reader has to see that the excluded pairs
    are not concentrated where the model is weakest before the bounds mean anything."""
    G = json.loads((DATA / "gold" / "gold_analysis.json").read_text())
    X = G["exclusion_sensitivity"]
    BIN = {"0": "[0,0.05)", "1": "[0.05,0.3)", "2": "[0.3,0.7)", "3": "[0.7,0.95)",
           "4": "[0.95,1]"}
    rows = []
    for v, n in sorted(X["by_variable"].items(), key=lambda kv: -kv[1]):
        rows.append({"block": "by variable", "item": esc(v), "count": str(n),
                     "$F_1$": NA, "ECE": NA, r"$H(0.90)$ \%": NA})
    for b, n in sorted(X["by_bin"].items()):
        rows.append({"block": "by probability bin", "item": BIN.get(b, b), "count": str(n),
                     "$F_1$": NA, "ECE": NA, r"$H(0.90)$ \%": NA})
    label = {"excluded": "excluded, as reported throughout",
             "as_model_error": "counted as model errors",
             "as_model_correct": "counted as model-correct",
             "as_routed_to_review": "routed to review regardless of probability"}
    for k, lab in label.items():
        r = X[k]
        rows.append({"block": "sensitivity", "item": lab,
                     "count": str(X["n_excluded"]),
                     "$F_1$": num(r["f1"], 3) if r.get("f1") is not None else NA,
                     "ECE": num(r["ece"], 4) if r.get("ece") is not None else NA,
                     r"$H(0.90)$ \%": pct(r["review_budget_90"], 1)})
    return pd.DataFrame(rows)


def t14_downstream(a: dict) -> pd.DataFrame:
    """WP14. What adding calibrated narrative variables does to a safety diagnosis."""
    f = DATA / "downstream_severity.json"
    if not f.exists():
        return pd.DataFrame()
    ds = json.loads(f.read_text())
    S = "injury_or_fatal"
    rows = []
    for v, r in sorted(ds["per_variable"].items(), key=lambda kv: -kv[1][S]["added_per_year"]):
        x = r[S]
        cr = r.get("narrative_only_confirmed_rate")
        rows.append({
            "factor": esc(v) + (r"$^{\dagger}$" if r["proxy_only"] else ""),
            "coded field": esc(r["coded_field"]),
            "coded\\newline per year": f"{x['coded_per_year']:,.0f}".replace(",", "{,}"),
            "added by narratives\\newline per year":
                f"{x['added_per_year']:,.0f}".replace(",", "{,}"),
            # the thousands separator has to be protected one number at a time, or the comma
            # between the two bounds is replaced along with them
            "95\\% interval": "[{}, {}]".format(
                *(f"{b:,.0f}".replace(",", "{,}") for b in x["added_per_year_ci95"])),
            "increase \\%": pct(x["relative_increase"], 0),
            "narrative-only flags\\newline confirmed": (num(cr, 2) if cr is not None else NA),
        })
    return pd.DataFrame(rows)


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
            # T12. The threshold chosen on a half sample is itself a random quantity, so its
            # spread belongs beside the out-of-fold score it produced.
            r"half-sample $\tau$\newline median [IQR]": (
                f"{num(r['threshold_median'], 2)} "
                f"[{num(r['threshold_iqr'][0], 2)}, {num(r['threshold_iqr'][1], 2)}]"
                if r.get("threshold_median") is not None else NA),
            r"$F_1$ tuned, out of fold": num(r["f1_tuned_oof"], 2),
            "s.d.": num(r["f1_tuned_oof_sd"], 2),
            "change": ("%+.2f" % r["gain"]),
            "stable?": ("no" if r["unstable"] else "yes"),
        })
    return pd.DataFrame(rows)


# Stated in full under the first table it applies to, and cross-referenced from the rest.
AGREE = ("Coded CRIS fields are themselves incomplete, so these measure "
         r"\emph{agreement} rather than accuracy; accuracy against the human reference set of "
         r"Section~\ref{sec:reference} is in Table~\ref{tab:gold}. Incompleteness costs "
         r"agreement rather than calibration, because a factor the narrative states and the "
         r"code omits turns a correct extraction into an apparent false positive.")
AGREE_REF = (r"Agreement against the incomplete coded fields rather than accuracy; see the "
             r"note to Table~\ref{tab:calibration}.")
DAGGER = r" The dagger marks a coded field that is a proxy rather than a direct match."


# ----------------------------------------------------------------------------- appendix tables
def _schema():
    return json.loads((ROOT / "schemas" / "crash_factors_v1_1.json").read_text())


def ta1_schema_full() -> pd.DataFrame:
    """Appendix A, Table A.1: every question with its full instruction and criteria text."""
    s = _schema()
    gates = s.get("gates", {})
    rows = []
    for qid, q in s["questions"].items():
        parts = [esc(q["instructions"])]
        crit = q.get("criteria")
        if isinstance(crit, dict):
            for k, v in crit.items():
                parts.append(rf"\textit{{{esc(k)}}}: {esc(v)}")
        elif isinstance(crit, list):
            for i, v in enumerate(crit):
                parts.append(rf"\textit{{level {i}}}: {esc(v)}")
        rows.append({
            "id": r"\texttt{" + esc(qid) + "}",
            "type": q["type"],
            "gate": (r"\texttt{" + esc(gates[qid]) + "}") if gates.get(qid) else NA,
            "instruction and criteria": r" \newline ".join(parts),
        })
    return pd.DataFrame(rows)


def _dict_literal(path: Path, name: str) -> dict:
    """Read a module-level dict literal such as KW = {...} without importing the module."""
    import ast
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def rx(pat: str) -> str:
    r"""A regular expression in a fixed-width column, allowed to break after an alternation.

    TeX does not hyphenate inside \texttt and finds no break point in a long pattern, so a
    wide family ran past the right margin and the overflow was clipped, which silently
    dropped the end of the expression from the printed table. Permitting a break after each
    alternation bar lets the cell wrap instead. \allowbreak contributes no character of its
    own, so what the table prints is still exactly the pattern the script applies.
    """
    return "\\texttt{" + esc(pat).replace("|", "|\\allowbreak{}") + "}"


def tb1_keyword_families() -> pd.DataFrame:
    """Appendix B, Table B.1: the regular-expression families of the rule baseline, read from
    the scripts that apply them so the table cannot drift from the code."""
    kw = _dict_literal(ROOT / "paper1" / "src" / "s01_corpus.py", "KW")
    extra = _dict_literal(ROOT / "paper1" / "src" / "s01b_extra_keywords.py", "EXTRA")
    from s10_analysis import KW_MAP, KW_MAP_EXTRA
    rows = []
    for fam, pat in kw.items():
        rows.append({"family": esc(fam), "narrative variable": esc(KW_MAP[fam]),
                     "scope": "all narratives",
                     "regular expression (case-insensitive)": rx(pat)})
    for fam, pat in extra.items():
        rows.append({"family": esc(fam), "narrative variable": esc(KW_MAP_EXTRA[fam]),
                     "scope": "working set",
                     "regular expression (case-insensitive)": rx(pat)})
    return pd.DataFrame(rows)


# The coded-field mapping implemented in src/s04_join_fst.R. Transcribed here as data so
# that Table B.2 states, for each narrative variable, which CRIS columns and values define
# its reference; the R script remains the executable definition.
CODED_MAPPING_ROWS = [
    ("alcohol_involved", "coded_alcohol",
     r"\texttt{Contrib\_Factr\_1\_ID} or \texttt{Contrib\_Factr\_2\_ID} matches ``Under Influence - Alcohol'' or ``Had Been Drinking''; or \texttt{Prsn\_Alc\_Rslt\_ID} = 1; or \texttt{Prsn\_Bac\_Test\_Rslt} $>$ 0",
     "any person or unit row of the crash"),
    ("drug_involved", "coded_drug",
     r"contributing factor matches ``Under Influence - Drug''; or \texttt{Prsn\_Drg\_Rslt\_ID} = 1",
     "any row"),
    ("fatigue", "coded_fatigue", r"contributing factor matches ``Fatigued Or Asleep''", "any row"),
    ("medical_episode", "coded_medical", r"contributing factor matches ``Ill (Explain In Narrative)''", "any row"),
    ("animal_involved", "coded_animal",
     r"contributing factor matches ``Animal On Road''; or \texttt{Harm\_Evnt\_ID} contains ``Animal''", "any row"),
    ("phone_use", "coded_phone", r"contributing factor matches ``Cell/Mobile Device Use''", "any row"),
    ("unbelted", "coded_unbelted", r"\texttt{Prsn\_Rest\_ID} equals ``None'' (the only unbelted code; ``Not Applicable'' excluded)", "any person row"),
    ("hydroplane", "coded_hydro_proxy",
     r"\texttt{Othr\_Factr\_ID} contains ``Lost Control Or Skidded'' and (\texttt{Surf\_Cond\_ID} contains ``Wet'' or ``Standing Water'', or \texttt{Wthr\_Cond\_ID} contains ``Rain'')",
     "any row; proxy reference"),
    ("wrong_way", "coded_wrongway",
     r"contributing factor matches ``Wrong Way - One Way Road'', ``Wrong Side - Not Passing'' or ``Wrong Side - Approach''",
     "any row"),
    ("(sensitivity only)", "coded_wrongway_proxy",
     r"\texttt{FHE\_Collsn\_ID} contains ``Opposite Direction''", "any row; not used as a reference"),
]


def tb2_coded_mapping() -> pd.DataFrame:
    return pd.DataFrame([{"narrative variable": esc(v), "reference column": r"\texttt{" + esc(c) + "}",
                          "CRIS fields and values": rule, "aggregation to crash level": agg}
                         for v, c, rule, agg in CODED_MAPPING_ROWS])


# Appendix D, Table D.1: where each formalized quantity is computed. The equation numbers are
# bound through \ref at build time; the script and function names are literal identifiers.
EQUATION_MAP_ROWS = [
    ("eq:gate", "gating operator and no-match override", "s06_flatten.py", "flatten", "stage2_flat.parquet"),
    ("eq:leak", r"leakage rate $\lambda_v$", "s06_flatten.py", "flatten", "stage2_leakage.json"),
    ("eq:ht", r"H\'ajek ratio with calibration weights $w_i$", "s05b_design_weights.py; s17_gold_analysis.py", "design_weights; wrate, block", "gold_design_weights.csv; gold_analysis.json"),
    ("eq:htvar", "stratified cluster bootstrap interval", "s07_metrics.py", "cluster_bootstrap; weighted_bootstrap", "analysis.json; gold_analysis.json"),
    ("eq:brier", "Brier score and Murphy decomposition", "s07_metrics.py", "brier, brier_decomposition", "analysis.json"),
    ("eq:ece", "expected and maximum calibration error, equal-mass bins", "s07_metrics.py", "ece, mce, equal_mass_bins", "analysis.json"),
    ("eq:ecegrid", "exact calibration error on the output grid", "s07_metrics.py", "ece_discrete, discrete_reliability", "analysis.json; gold_analysis.json"),
    ("eq:cox", "calibration slope and intercept", "s07_metrics.py", "calibration_slope", "analysis.json; gold_analysis.json"),
    ("eq:spieg", "Spiegelhalter statistic", "s07_metrics.py", "spiegelhalter_z", "analysis.json; gold_analysis.json"),
    ("eq:floor", "resolution-floor bound", "s07_metrics.py", "resolution_floor_report", "analysis.json"),
    ("eq:prec", "precision and coverage over flagged records", "s07_metrics.py", "precision_coverage, coverage_at_precision", "analysis.json"),
    ("eq:budget", r"review budget $H(\pi^*)$, cross-fitted", "s07_metrics.py; s17_gold_analysis.py", "positive_review_budget; held_out_budget", "analysis.json; gold_analysis.json"),
    ("eq:riskcov", "risk--coverage curve and area", "s07_metrics.py", "risk_coverage, aurc, coverage_at_risk", "analysis.json"),
    ("eq:noise", "one-sided reference-noise bound", "s17_gold_analysis.py", "narrative_only", "gold_analysis.json"),
    ("eq:cost", "token cost model", "s08_cost_model.py", "main", "cost_model.json"),
    ("eq:twostage", "two-stage cost and break-even fraction", "s08_cost_model.py; pricing.py", "main; jev_cost_per_1k", "cost_model.json"),
    ("eq:platt", "Platt and isotonic recalibration, split over narratives", "s17_gold_analysis.py", "recalibration", "gold_analysis.json"),
    ("eq:kappa", "Cohen's kappa, weighted against the reference set", "s07_metrics.py; s16_gold_ingest.py", "cohen_kappa", "gold_reliability.json; gold_analysis.json"),
    ("eq:cp", "Clopper--Pearson interval", "s07_metrics.py", "clopper_pearson", "analysis.json"),
    ("eq:mcnemar", "McNemar test", "s07_metrics.py", "mcnemar", "analysis.json"),
]


def td1_equation_map() -> pd.DataFrame:
    return pd.DataFrame([{"equation": rf"Eq.~(\ref{{{lab}}})", "quantity": q,
                          "script (\\texttt{paper1/src/})": r"\texttt{" + esc(sc) + "}",
                          "function": r"\texttt{" + esc(fn) + "}",
                          "result file (\\texttt{paper1/data/})": r"\texttt{" + esc(rf_) + "}"}
                         for lab, q, sc, fn, rf_ in EQUATION_MAP_ROWS])


def main() -> None:
    a = json.loads((DATA / "analysis.json").read_text())
    print("writing tables:")
    # The abbreviated schema table is not emitted. Appendix A carries the full schema in
    # Table A.1, so this one was generated into the release without any section including it,
    # and a table a reader can never reach is a file that only invites the question of where
    # it went. `t2_schema()` is kept, so restoring it is one line if a length limit ever makes
    # the short form preferable to the full one.
    w("T3_run", t3_run(a), "Corpus and run statistics.", "tab:run",
      "Prices are TypeSafe list prices accessed 2026-09-17 and are quoted rather than endorsed.")
    w("T4_prevalence", t4_prevalence(a),
      "Prevalence estimates on the Stage-2 random stratum.", "tab:prevalence",
      r"Keyword \% is over all 5{,}018{,}080 narratives. Jev \% thresholds at $p>0.5$ with "
      r"Clopper--Pearson intervals. E[p] is the mean probability. The floor column is the "
      r"share of E[p] from narratives at the smallest probability the model can express, "
      r"$p=0.01$; the double dagger marks a share above 20\%, where E[p] is inflated by the "
      r"output grid and is not a prevalence estimate (Section~\ref{sec:res-population}). "
      r"Confirmation \% is the share of keyword hits Jev also calls positive; uncertain \% "
      r"the share with $p \in [0.3, 0.7]$.")
    w("T5_calibration", t5_calibration(a),
      "Calibration against coded CRIS fields (agreement).", "tab:calibration",
      AGREE + DAGGER + " ECE and its interval are both the exact estimator on the model's "
      "two-decimal output grid, so every point estimate lies inside its own interval. The "
      "interval is a bootstrap over crashes (B = 1{,}000), the sampling unit of the stratum.",
      groups=[("", 3), ("Error", 5), ("Cox calibration", 3)])
    w("T6_selective", t6_selective(a),
      "Selective prediction and the human-review budget (agreement).", "tab:selective",
      AGREE_REF + r" $H(\pi)$ is the share of \emph{flagged positives} a human must review "
      r"to reach precision $\pi$; accuracy-based budgets are degenerate because the positive "
      r"class is rare (Section~\ref{sec:res-accuracy}). The two human columns are measured "
      r"against the human reference set. The first picks the threshold on one half of the "
      r"narratives and certifies it on the other, and the second does both on the same half, "
      r"so their difference is the optimism of in-sample selection. The last column counts "
      r"\emph{variable-level review decisions} per year rather than narratives.")
    if (DATA / "downstream_severity.json").exists():
        w("T14_downstream", t14_downstream(a),
          "Injury and fatal crashes attributed to each factor, from the coded fields alone and "
          "with calibrated narrative variables added.", "tab:downstream",
          r"Counts are estimated on the Stage-2 random stratum and scaled to Texas per year. "
          r"The added column is the expected number of injury or fatal crashes whose coded "
          r"field is negative and whose narrative states the factor, summed over the "
          r"recalibrated probability of Section~\ref{sec:res-calibration}, so it is a "
          r"calibrated total rather than a count of flags. Intervals resample the reference "
          r"narratives and refit the map. The last column is the share of narrative-only "
          r"flags a coder confirmed, from Table~\ref{tab:gold}, which disciplines the added "
          r"column rather than feeding it." + DAGGER)
    w("T7_baseline", t7_baseline(a),
      "Keyword baseline versus Jev on the same coded reference (agreement).", "tab:baseline",
      AGREE_REF + " McNemar tests paired per-narrative correctness. The frontier arm is "
      r"scored against human labels in Table~\ref{tab:frontier}.",
      groups=[("", 1), ("Keyword rules", 3), ("Jev 1.13", 3), ("", 1)])
    w("T8_discrepancy", t8_discrepancy(a),
      "Discrepancy taxonomy against coded fields (agreement).", "tab:discrepancy",
      AGREE_REF + DAGGER + " ``Narrative only'' is not an error rate, because a factor stated "
      "in the narrative and absent from the coded field is what this study sets out to find.")
    w("T10_gold", t10_gold(a),
      "Accuracy against the human reference set.", "tab:gold",
      r"These are accuracy rather than agreement measures, because the reference is "
      r"\GoldNUsable{} human judgments of the criteria text the model was given. Every rate "
      r"is calibration-weighted to the Stage-2 random stratum under the design of "
      r"Section~\ref{sec:reference}, so the base rate is the population rate and both kappa "
      r"columns are population quantities. Bars scale $F_1$ from 0.40 and \jevflag{red} marks "
      r"$F_1<0.70$. The gap column is ECE divided by the coded-field ECE of "
      r"Table~\ref{tab:calibration}. The budget column is cross-fitted, the threshold chosen "
      r"on one half of the narratives and certified on the other. ``sep.'' denotes "
      r"quasi-complete separation.",
      groups=[("", 3), ("Accuracy vs.\\ human labels", 6), ("Calibration", 4), ("", 1)])
    w("T11_recalibration", t11_recal(a),
      "Post-hoc recalibration on the human reference set, validated across narrative splits.",
      "tab:recal",
      "Each mapping is fitted on one half of the narratives and scored on the other, both ways "
      r"and over \RecalRepeats{} repeats, because an in-sample fit would be circular and a "
      r"split over pairs would place one narrative on both sides of it. The row ``none'' uses "
      r"the same protocol, so it is comparable to the rows below but not to the full-sample "
      r"ECE of Table~\ref{tab:gold}. The Platt map is unpenalised, fitting an intercept and a "
      r"slope. The pooled block is one map over all variables; the per-variable block is what "
      r"an analyst deploys, and variables with fewer than \RecalMinPositives{} positive "
      r"labels are omitted from it.")
    w("T12_threshold", t12_threshold(a),
      "Per-variable decision threshold, retuned on the human reference set and validated out of fold.",
      "tab:threshold",
      r"$n^{+}$ counts positive reference labels. $\tau^{*}$ maximizes weighted $F_1$ on the "
      r"full reference set; the out-of-fold column picks $\tau$ on a random half and scores "
      r"it on the other, both ways over 200 repeats, splitting on narratives rather than "
      r"pairs, and is what retuning actually buys. The median and interquartile range are of "
      r"the thresholds those half samples chose. An unstable threshold has a spread above "
      r"0.10 or fewer than 20 positives, is fitted to noise, and must not be transported.",
      groups=[("", 3), ("Retuned threshold, out of fold", 6)])
    if (DATA / "frontier" / "frontier_analysis.json").exists():
        w("T13_frontier", t13_frontier(a),
          "Jev against frontier generative models on the same human reference labels.",
          "tab:frontier",
          r"All arms answer the same narratives and variables under the same criteria text, "
          r"scored against the same labels with the same code. \jevbest{Bold} is the leader "
          r"per metric and bars scale $F_1$ from 0.80. The last column states whether the "
          r"model returns a probability natively or was asked to state one. Every $z$ is "
          r"\jevflag{red} because every arm is rejected, and they fail in opposite directions, "
          r"since a slope above one with $z<0$ over-states prevalence and a slope below one "
          r"with $z>0$ under-states it. Cost is not measured here, and "
          r"Figure~\ref{fig:frontier} prices a comparable tier "
          r"at list rates. No arm is held out from the reference set.",
          groups=[("", 2), ("Agreement with human labels", 5), ("Calibration", 4), ("", 1)])
    # appendix tables
    w("TA1_schema_full", ta1_schema_full(),
      "The full question schema, version 1.1, as sent to the model.", "tab:schema-full",
      r"Text is reproduced from \texttt{schemas/crash\_factors\_v1\_1.json} without "
      r"abbreviation. A gated question is interpreted only when its gate exceeds "
      r"$\tau_g = 0.5$, otherwise it takes its no-match option. \texttt{pii\_residual} ships "
      r"in the same file but was run separately as the display screen, outside the Stage 1 "
      r"and Stage 2 calls.", longtable=True)
    w("TC1_exclusions", tc1_exclusions(a),
      "Unresolved pairs and the sensitivity of the headline results to them.",
      "tab:exclusions",
      r"The \ExclN{} excluded pairs are those the coders marked unclear or disagreed on. "
      r"They are dropped from every rate in Section~\ref{sec:res-accuracy}, so those rates "
      r"are conditional on the pairs the coders found clear. The sensitivity block replaces "
      r"that conditioning with three assumptions. Routing the excluded pairs to review "
      r"leaves $F_1$ undefined, because a record sent to a human is not scored as a model "
      r"decision.", longtable=True)
    w("TB1_keyword_families", tb1_keyword_families(),
      "Regular-expression families of the rule baseline.", "tab:keywords",
      r"The first ten families apply to all narratives and provide the population keyword "
      r"rates of Table~\ref{tab:prevalence}. The last three were added so the baseline "
      r"covers every variable with a coded reference, and were scored on the working set "
      r"only. Patterns are read from the scripts that apply them.", longtable=True)
    w("TB2_coded_mapping", tb2_coded_mapping(),
      "Mapping from narrative variables to coded CRIS reference fields.", "tab:coded-map",
      r"Each reference is true for a crash when the rule holds on any of its person or unit "
      r"rows. The hydroplaning reference is a proxy built from surface, weather and "
      r"loss-of-control codes, because CRIS has no hydroplaning field. The final row is the "
      r"opposite-direction collision type the original plan proposed for wrong-way travel, "
      r"retained only as a sensitivity check.", longtable=True)
    w("TD1_equation_map", td1_equation_map(),
      "Where each formalized quantity is computed.", "tab:eqmap",
      r"Scripts are in \texttt{paper1/src/} and result files in \texttt{paper1/data/} of the "
      r"released repository. Every equation in Section~\ref{sec:methods} maps to a function "
      r"that a reader can execute on the released outputs.", landscape=True)


if __name__ == "__main__":
    main()
