"""Reference gate for the submission build (revision plan, section 5).

    python src/check_refs.py paper1/submission

Fails (exit 1) when
  * a key cited in body.tex, appendices.tex or main.tex is missing from the verified
    submission/refs.bib;
  * a cited entry lacks a `verified` field (written by the reference-verification subagent);
  * a citation instance whose citing sentence asserts a finding or a number has no row in
    submission/references/claims.csv (matched on key and on a normalised prefix of the
    citing sentence).

A citation "asserts a finding or a number" when the sentence containing it holds a digit, a
percent sign, or one of the verbs found, show, report, measure, reach, estimate, evaluate,
recommend, compare, demonstrate, observe, document, propose, define, prove, achieve, obtain,
audit, benchmark, or classify. Purely attributive citations (the name of a method or a data
source with none of those markers) need no claims row.
"""
from __future__ import annotations
import csv
import re
import sys
from pathlib import Path

FINDING_VERBS = ("found", "find", "finds", "show", "shows", "showed", "shown", "report",
                 "reports", "reported", "measure", "measures", "measured", "reach", "reaches",
                 "reached", "estimate", "estimates", "estimated", "evaluate", "evaluates",
                 "evaluated", "recommend", "recommends", "recommended", "compare", "compares",
                 "compared", "demonstrate", "demonstrates", "demonstrated", "observe",
                 "observes", "observed", "document", "documents", "documented", "propose",
                 "proposes", "proposed", "define", "defines", "defined", "prove", "proves",
                 "proved", "achieve", "achieves", "achieved", "obtain", "obtains", "obtained",
                 "audit", "audits", "audited", "benchmark", "benchmarks", "benchmarked",
                 "classify", "classifies", "classified", "argue", "argues", "argued",
                 "conclude", "concludes", "concluded", "identify", "identifies", "identified")


def strip_comments(s: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in s.splitlines())


def norm(s: str) -> str:
    s = re.sub(r"\\cite[pt]?\*?(\[[^\]]*\])?\{[^}]*\}", " ", s)
    s = re.sub(r"\\[A-Za-z]+\*?", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def bib_keys(bib: Path) -> dict[str, bool]:
    text = bib.read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r"@\w+\{([^,\s]+),(.*?)(?=\n@|\Z)", text, re.S):
        out[m.group(1)] = bool(re.search(r"\bverified\s*=", m.group(2)))
    return out


def citations(tex_files: list[Path]):
    """Yield (key, sentence, file, line) for every citation instance."""
    for f in tex_files:
        text = strip_comments(f.read_text(encoding="utf-8"))
        # join paragraphs into sentences
        paras = re.split(r"\n\s*\n", text)
        for para in paras:
            flat = " ".join(para.split())
            sents = re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", flat)
            for s in sents:
                for m in re.finditer(r"\\cite[pt]?\*?(?:\[[^\]]*\])?\{([^}]*)\}", s):
                    for key in m.group(1).split(","):
                        yield key.strip(), s, f.name


def asserts_finding(sent: str) -> bool:
    low = re.sub(r"\\cite[pt]?\*?(\[[^\]]*\])?\{[^}]*\}", " ", sent).lower()
    if re.search(r"\d", low) or "%" in low or "\\%" in low:
        return True
    return any(re.search(rf"\b{v}\b", low) for v in FINDING_VERBS)


def main() -> int:
    sub = Path(sys.argv[1] if len(sys.argv) > 1 else "paper1/submission")
    bib = sub / "refs.bib"
    claims = sub / "references" / "claims.csv"
    tex_files = [p for p in (sub / "body.tex", sub / "appendices.tex", sub / "main.tex") if p.exists()]
    keys = bib_keys(bib)
    rows = []
    if claims.exists():
        with open(claims, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    claim_index = {}
    for r in rows:
        claim_index.setdefault(r["key"].strip(), []).append(norm(r["citing_sentence"])[:80])
    problems = []
    seen = set()
    n_cit = 0
    for key, sent, fname in citations(tex_files):
        n_cit += 1
        if key not in keys:
            problems.append(f"MISSING KEY   {key}  ({fname})")
            continue
        if not keys[key]:
            if key not in seen:
                problems.append(f"UNVERIFIED    {key}")
            seen.add(key)
        if asserts_finding(sent):
            pref = norm(sent)[:80]
            if not any(pref[:60] == c[:60] or pref[:40] in c or c[:40] in pref
                       for c in claim_index.get(key, [])):
                problems.append(f"NO CLAIM ROW  {key}: {sent[:120]}")
    cited = {k for k, _, _ in citations(tex_files)}
    print(f"{n_cit} citation instances, {len(cited)} distinct keys, "
          f"{len(keys)} bib entries, {sum(keys.values())} verified, {len(rows)} claims rows")
    uncited = sorted(k for k in keys if k not in cited)
    if uncited:
        print(f"note: {len(uncited)} bib entries are not cited: {', '.join(uncited)}")
    for p in problems:
        print(p)
    print("OK" if not problems else f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
