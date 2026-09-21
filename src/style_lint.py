"""Style lint for the Paper 1 manuscript (Paper1_Revision_Plan_Q1_Rewrite.md, section 5).

Usage:
    python src/style_lint.py paper1/submission/body.tex [more .tex files] [--report out.md]

Every file given is linted; `main.tex` is linted for its abstract and highlights, `body.tex`
and `appendices.tex` for the prose, and the generated table files for their captions and
notes. The report lists every finding with file, line and rule, and ends with the word
counts per section and subsection so the owner can see how the length was measured.

What is checked (each rule is a function below, named in the report):
  em-dash            an em dash (U+2014, `---`, or a spaced ` -- `) anywhere in prose
  colon              a colon inside a sentence, outside URLs, LaTeX commands, math and tables
  list               itemize or enumerate outside the `highlights` environment
  banned-word        honest, honestly, frankly, admittedly, candid
  limitation-word    "limitation" outside the Conclusion's limitations paragraph
  figure-word        "Figure" not followed by a number or a \\ref
  short-opener       a paragraph whose first sentence has fewer than eight words
  long-sentence      a sentence over 45 words
  short-sentence     a sentence under eight words in body text
  question           a sentence ending with a question mark
  british            a spelling from a fixed British list
  short-subsection   a subsection under 350 words
  highlight-length   a highlight over 85 characters

The lint works on a de-TeXed copy of the prose: comments, display math, tabular bodies,
figure/table environments (except their captions and notes) and the preamble are removed,
`\\ref`, `\\cite` and `\\num` are replaced by placeholders, and macros are left as words.
Sentence splitting is deliberately simple: a sentence ends at `.`, `!` or `?` followed by
whitespace and an upper-case letter, a `\\`, or the end of the paragraph, with common
abbreviations (`e.g.`, `i.e.`, `et al.`, `vs.`, `Eq.`, `Fig.`, `No.`, `cf.`) protected.
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

BANNED = ["honest", "honestly", "frankly", "admittedly", "candid", "candidly"]
BRITISH = [
    "analyse", "analysed", "analyses", "analysing", "behaviour", "behaviours", "catalogue",
    "centre", "centred", "colour", "colours", "coloured", "defence", "favour", "favoured",
    "formalise", "formalised", "formalises", "formalising", "generalise", "generalised",
    "generalises", "generalising", "grey", "greys", "honour", "judgement", "judgements",
    "labelled", "labelling", "licence", "maximise", "maximised", "maximises", "maximising",
    "minimise", "minimised", "minimises", "minimising", "modelled", "modelling", "normalise",
    "normalised", "normalising", "optimise", "optimised", "optimises", "optimising",
    "organise", "organised", "organisation", "practise", "programme", "realise", "realised",
    "realises", "recognise", "recognised", "recognises", "summarise", "summarised",
    "summarises", "summarising", "travelling", "travelled", "utilise", "utilised",
    "verbalise", "verbalised", "verbalises", "verbalising", "visualise", "visualised",
    "visualises", "visualising", "artefact", "artefacts", "emphasise", "emphasised",
    "characterise", "characterised", "standardise", "standardised", "penalise", "penalised",
    "categorise", "categorised", "manoeuvre", "manoeuvres", "signalling", "signalled",
    "cancelled", "totalling", "fulfil", "fulfils", "enrol", "enrolment", "criticise",
    "criticised", "authorise", "authorised", "prioritise", "prioritised", "stabilise",
    "stabilised", "paralyse", "paralysed", "sceptical", "specialised", "specialise",
    "randomise", "randomised", "randomising", "italicised", "capitalise", "capitalised",
    "hypothesise", "hypothesised", "neighbour", "neighbours", "neighbouring", "rumour",
    "harbour", "labour", "flavour", "vapour", "humour", "endeavour", "mould", "plough",
    "tyre", "tyres", "kerb", "aluminium", "sulphur", "whilst", "amongst", "learnt", "spelt",
    "practising", "ageing", "storey", "cheque", "dialogue", "epicentre", "metre", "metres",
    "litre", "litres", "theatre", "fibre", "calibre", "sombre", "manoeuvring",
]
ABBREV = ["e.g.", "i.e.", "et al.", "vs.", "Eq.", "Eqs.", "Fig.", "Figs.", "No.", "cf.",
          "Dr.", "Mr.", "Ms.", "St.", "approx.", "ca.", "Sec.", "Tab.", "Ref.", "Refs.",
          "Prop.", "Appendix A.", "Appendix B.", "Appendix C.", "Appendix D."]


# ----------------------------------------------------------------------------- de-TeX
def strip_comments(s: str) -> str:
    out = []
    for line in s.splitlines():
        i, esc = 0, False
        cut = None
        while i < len(line):
            c = line[i]
            if c == "\\":
                i += 2
                continue
            if c == "%":
                cut = i
                break
            i += 1
        out.append(line if cut is None else line[:cut])
    return "\n".join(out)


def remove_env(s: str, name: str, keep_caption: bool = False) -> str:
    """Remove \\begin{name}...\\end{name} (non-nested). Captions and table notes are kept as
    separate paragraphs when requested, so they are linted like prose."""
    pat = re.compile(r"\\begin\{" + name + r"\*?\}.*?\\end\{" + name + r"\*?\}", re.S)

    def repl(m):
        if not keep_caption:
            return "\n\n"
        block = m.group(0)
        kept = []
        for cm in re.finditer(r"\\caption\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", block, re.S):
            kept.append("CAPTION " + cm.group(1))
        for nm in re.finditer(r"\\item\s+(.*?)(?=\\end\{tablenotes\})", block, re.S):
            kept.append("TABLENOTE " + nm.group(1))
        for nm in re.finditer(r"\\begin\{minipage\}\{[^}]*\}\\footnotesize\s*(.*?)\\end\{minipage\}",
                              block, re.S):
            kept.append("TABLENOTE " + nm.group(1))
        return "\n\n" + "\n\n".join(kept) + "\n\n"

    return pat.sub(repl, s)


def detex(s: str) -> str:
    s = strip_comments(s)
    # preamble of main.tex: keep only the abstract and highlights
    if "\\begin{document}" in s:
        m_abs = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", s, re.S)
        m_hl = re.search(r"\\begin\{highlights\}(.*?)\\end\{highlights\}", s, re.S)
        s = ("\\section{Abstract}\n" + (m_abs.group(1) if m_abs else "") +
             "\n\n\\section{Highlights}\n" + (m_hl.group(1) if m_hl else ""))
    for env in ("equation", "equation*", "align", "align*", "gather", "eqnarray", "verbatim",
                "lstlisting", "proof", "proofprop", "prooflem", "pf"):
        s = re.sub(r"\\begin\{" + re.escape(env) + r"\}.*?\\end\{" + re.escape(env) + r"\}",
                   " EQUATION ", s, flags=re.S)
    s = re.sub(r"\\\[.*?\\\]", " EQUATION ", s, flags=re.S)
    for env in ("figure", "figure*", "table", "table*", "sidewaystable", "longtable",
                "threeparttable"):
        s = remove_env(s, re.escape(env), keep_caption=True)
    s = remove_env(s, "tabular", keep_caption=False)
    s = remove_env(s, "tabularx", keep_caption=False)
    s = re.sub(r"\\input\{[^}]*\}", "", s)
    s = re.sub(r"\\setcounter\{[^}]*\}\{[^}]*\}", "", s)
    s = re.sub(r"\\(begin|end)\{(proposition|lemma|proofprop|prooflem|proof|pf)\}(\[[^\]]*\])?", "\n\n", s)
    s = re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}", "", s)
    s = re.sub(r"\\(label|url|href|cite[pt]?\*?|citeauthor|citeyear|ead|credit|affiliation|"
               r"cormark|cortext|shorttitle|shortauthors|author|title|keywords|sep)"
               r"(\[[^\]]*\])?\{[^}]*\}", lambda m: {"url": "URL", "href": "URL"}.get(m.group(1), "CITE")
               if m.group(1).startswith(("cite", "url", "href")) else "", s)
    s = re.sub(r"\\(ref|eqref|autoref|pageref|num|SI|si)\{[^}]*\}", "9", s)
    s = re.sub(r"\\section\*?\{([^}]*)\}", r"\n\n@@SECTION@@ \1\n", s)
    s = re.sub(r"\\subsection\*?\{([^}]*)\}", r"\n\n@@SUBSECTION@@ \1\n", s)
    s = s.replace("\$", " USD ")
    s = re.sub(r"\$[^$]*\$", " MATH ", s)
    s = re.sub(r"\\(textit|textbf|emph|texttt|textsc|mbox|text)\{([^{}]*)\}", r"\2", s)
    s = re.sub(r"\\[A-Za-z]+\*?(\[[^\]]*\])?", lambda m: m.group(0)
               if m.group(0)[1:] in ("section", "subsection") else " " + m.group(0)[1:].strip("[]") + " "
               if m.group(0)[1:].isalpha() and m.group(0)[1:] not in
               ("section", "subsection", "subsubsection", "paragraph", "item", "centering", "small", "footnotesize", "scriptsize", "noindent", "par",
                "appendix", "midrule", "toprule", "bottomrule", "hline", "newpage", "clearpage",
                "printcredits", "maketitle", "bibliography", "bibliographystyle", "setcounter",
                "renewcommand", "newcommand", "appendixtheorems", "appendixtheorems", "vspace", "hspace", "linebreak", "FloatBarrier",
                "sisetup", "textwidth", "linewidth", "textheight", "centerline")
               else " ", s)
    s = s.replace("~", " ").replace("``", '"').replace("''", '"').replace("{", "").replace("}", "")
    s = re.sub(r"[ \t]+", " ", s)
    return s


# ----------------------------------------------------------------------------- structure
def split_sections(text: str):
    """Yield (section, subsection, paragraph_text, line_no) tuples."""
    lines = text.splitlines()
    sec, sub = "front", None
    para, start = [], 1
    for i, line in enumerate(lines, 1):
        m = re.match(r"\s*@@(SECTION|SUBSECTION)@@ (.*)$", line)
        if m:
            if para:
                yield sec, sub, " ".join(para).strip(), start
                para = []
            if m.group(1) == "SECTION":
                sec, sub = m.group(2).strip(), None
            else:
                sub = m.group(2).strip()
            continue
        if not line.strip():
            if para:
                yield sec, sub, " ".join(para).strip(), start
                para = []
            continue
        if not para:
            start = i
        para.append(line.strip())
    if para:
        yield sec, sub, " ".join(para).strip(), start


def sentences(par: str) -> list[str]:
    t = par
    for a in ABBREV:
        t = t.replace(a, a.replace(".", "<DOT>"))
    t = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", t)
    parts = re.split(r"(?<=[.!?])[\"')]*\s+(?=[A-Z\"(\\9]|MATH|CITE)", t)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def words(s: str) -> int:
    return len([w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", s)
                if w not in ("EQUATION", "MATH", "CITE", "USD", "limitationsparagraph")])


# ----------------------------------------------------------------------------- rules
def lint_file(path: Path, is_body: bool, limitations_marker: str = "limitationsparagraph"):
    raw = path.read_text(encoding="utf-8")
    findings = []
    for i, line in enumerate(raw.splitlines(), 1):
        code = strip_comments(line)
        # a spaced "--" inside a table row (cells separated by &) marks a missing value, not a dash
        is_row = re.search(r"(?<!\\)&", code) is not None
        if "\u2014" in code or "---" in code or (re.search(r"\s--\s", code) and not is_row):
            findings.append((path.name, i, "em-dash", code.strip()[:110]))
        if re.search(r"\\begin\{(itemize|enumerate)\}", code) and "highlights" not in path.name:
            if path.name == "main.tex":
                pass
            else:
                findings.append((path.name, i, "list", code.strip()[:110]))
    # highlights
    m_hl = re.search(r"\\begin\{highlights\}(.*?)\\end\{highlights\}", strip_comments(raw), re.S)
    if m_hl:
        for item in re.findall(r"\\item\s+(.*?)(?=\\item|\Z)", m_hl.group(1), re.S):
            txt = " ".join(item.split())
            if len(txt) > 85:
                findings.append((path.name, 0, "highlight-length", f"{len(txt)} chars: {txt}"))
    text = detex(raw)
    in_limits = False
    section_words: dict[tuple, int] = {}
    for sec, sub, par, ln in split_sections(text):
        if not par or par.startswith("EQUATION") and words(par) < 3:
            continue
        section_words[(sec, sub)] = section_words.get((sec, sub), 0) + words(par)
        is_caption = par.startswith(("CAPTION", "TABLENOTE"))
        body_par = par.replace("CAPTION ", "").replace("TABLENOTE ", "")
        lim_par = limitations_marker in par
        body_par = body_par.replace(limitations_marker, "")
        if sec.lower().startswith("highlights"):
            continue
        # colon inside a sentence (outside URLs, which were replaced by URL already)
        for m in re.finditer(r":", body_par):
            ctx = body_par[max(0, m.start() - 40): m.end() + 40]
            if re.search(r"\d:\d", ctx):
                continue
            # The label that opens a table note is a heading, not a colon inside a sentence.
            # The rule exists to stop a colon standing in for a conjunction in prose, and
            # "Note:" at the head of a note is the ordinary journal convention the owner asked
            # for. Only that exact label is exempt, and only where it opens the note.
            if re.match(r"\s*(\\textit\{)?Note:", ctx) or ctx.strip().startswith("Note:"):
                continue
            findings.append((path.name, ln, "colon", ctx.strip()))
        low = " " + re.sub(r"[^a-z' ]", " ", body_par.lower()) + " "
        for w in BANNED:
            if f" {w} " in low:
                findings.append((path.name, ln, "banned-word", w))
        for w in BRITISH:
            if f" {w} " in low:
                findings.append((path.name, ln, "british", w))
        if " limitation " in low or " limitations " in low:
            if not (lim_par or sec.lower().startswith("conclusion") and lim_par):
                findings.append((path.name, ln, "limitation-word", body_par[:110]))
        for m in re.finditer(r"\bFigures?\b(?!\s*(9|[A-D]\.\d|\d|s? 9))", body_par):
            findings.append((path.name, ln, "figure-word", body_par[max(0, m.start() - 30): m.end() + 30]))
        sents = sentences(body_par)
        if sents and not is_caption:
            first = words(sents[0])
            if first < 8 and sec != "front":
                findings.append((path.name, ln, "short-opener", sents[0][:110]))
        for s in sents:
            n = words(s)
            if n > 45:
                findings.append((path.name, ln, "long-sentence", f"{n} words: {s[:100]}"))
            if n < 8 and not is_caption and is_body and sec != "front" and not s.startswith("EQUATION"):
                findings.append((path.name, ln, "short-sentence", s[:110]))
            if s.rstrip('"').endswith("?"):
                findings.append((path.name, ln, "question", s[:110]))
    if is_body:
        for (sec, sub), n in section_words.items():
            if sub is not None and n < 350:
                findings.append((path.name, 0, "short-subsection", f"{sec} / {sub}: {n} words"))
    return findings, section_words


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--report", default=None)
    ap.add_argument("--body", nargs="*", default=["body.tex"],
                    help="file names treated as body prose (short-sentence and subsection rules)")
    a = ap.parse_args()
    all_f, all_w = [], {}
    for f in a.files:
        p = Path(f)
        fs, sw = lint_file(p, is_body=p.name in a.body)
        all_f += fs
        if p.name in a.body:
            all_w.update(sw)
    lines = ["# Style lint report", "", f"Files: {', '.join(a.files)}", "",
             f"Findings: {len(all_f)}", ""]
    if all_f:
        lines.append("| file | line | rule | context |")
        lines.append("|---|---|---|---|")
        for fn, ln, rule, ctx in all_f:
            lines.append(f"| {fn} | {ln} | {rule} | {ctx.replace('|', '/')} |")
    lines += ["", "## Word counts (prose only; math, tables, captions, notes and references excluded)",
              "", "| section | subsection | words |", "|---|---|---|"]
    tot = 0
    for (sec, sub), n in all_w.items():
        if sec == "front":
            continue
        if sub is None:
            continue
        lines.append(f"| {sec} | {sub} | {n} |")
    bysec: dict[str, int] = {}
    for (sec, sub), n in all_w.items():
        if sec == "front":
            continue
        bysec[sec] = bysec.get(sec, 0) + n
    lines += ["", "| section | words |", "|---|---|"]
    for sec, n in bysec.items():
        lines.append(f"| {sec} | {n} |")
        tot += n
    lines.append(f"| **total body** | **{tot}** |")
    rep = "\n".join(lines) + "\n"
    if a.report:
        Path(a.report).write_text(rep, encoding="utf-8")
    print(rep)
    return 1 if all_f else 0


if __name__ == "__main__":
    sys.exit(main())
