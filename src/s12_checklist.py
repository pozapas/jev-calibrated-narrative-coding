"""Mechanically verify the outline's §7 desk-reject and §8 arXiv-v1 checklists.

Every item that CAN be checked from the artefacts is checked; items that need human
judgement are printed as MANUAL with the evidence needed. Run before any submission:

    python src/s12_checklist.py
"""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(r"D:/OneDrive - Texas State University/AIT/Papers/Jev")
P1 = ROOT / "paper1"
MANDIR = P1 / "manuscript"
# The manuscript is split across a wrapper plus a shared body/backmatter (two Elsevier
# builds, elsarticle and CAS, include the same body), so every content check has to read
# all of them or it reports the body's contents as missing.
TEX_FILES = ["main.tex", "body.tex", "backmatter.tex"]
TEX = MANDIR / "main.tex"
FIGS = P1 / "outputs" / "figures"
TABS = P1 / "outputs" / "tables"
DATA = P1 / "data"

REQUIRED_FIGS = ["F1_framework", "F2_corpus_cost", "F3_probability_landscape",
                 "F4_calibration", "F5_risk_coverage", "F7_coded_agreement", "F9_frontier"]
REQUIRED_TABS = ["T2_schema", "T3_run", "T4_prevalence", "T5_calibration",
                 "T6_selective", "T7_baseline", "T8_discrepancy"]

results: list[tuple[str, str, str]] = []


def check(ok: bool | None, item: str, detail: str = "") -> None:
    results.append(("PASS" if ok else ("MANUAL" if ok is None else "FAIL"), item, detail))


def main() -> None:
    tex = "\n".join((MANDIR / f).read_text(encoding="utf-8")
                    for f in TEX_FILES if (MANDIR / f).exists())
    # title and abstract live in the wrapper; everything else lives in the shared body
    wrapper = TEX.read_text(encoding="utf-8") if TEX.exists() else ""
    title = re.search(r"\\title\{(.+?)\}\s*\n", wrapper, re.S)
    title = title.group(1) if title else ""
    abstract = re.search(r"\\begin\{abstract\}(.+?)\\end\{abstract\}", wrapper, re.S)
    abstract = abstract.group(1) if abstract else ""

    # ---------------------------------------------------------------- §8 arXiv v1
    check("Jev" in title, "§8 title contains 'Jev'", title.strip()[:90])
    check(bool(re.search(r"[Cc]alibrated [Dd]ecisions", title)),
          "§8 title contains 'calibrated decisions'")
    check(bool(re.search(r"5[,\s]?018[,\s]?080|\\num\{5018080\}|\\Ncorpus", abstract)),
          "§8 abstract states corpus size", "via the \\Ncorpus macro")
    check(bool(re.search(r"per thousand narratives|\\CostPerThousand", abstract)),
          "§8 abstract states cost per 1,000 narratives", "via the \\CostPerThousand macro")
    check("calibrat" in abstract.lower(), "§8 abstract states the calibration audit")
    check(bool(re.search(r"review|precision", abstract, re.I)),
          "§8 abstract states the review-budget idea")
    missing_f = [f for f in REQUIRED_FIGS if not (FIGS / f"{f}.pdf").exists()]
    check(not missing_f, "§8 figures F1-F5, F7, F9 present",
          f"missing: {missing_f}" if missing_f else f"{len(REQUIRED_FIGS)} PDFs")
    check("v2" in tex and "gold set" in tex.lower(),
          "§8 F4 human panel marked v2")
    check("jev-1.13.0" in tex, "§8 model pinned in the manuscript")
    not_included = [f for f in REQUIRED_FIGS if f not in tex]
    check(not not_included, "§8 every required figure is included in the text",
          rf"built but never \includegraphics'd: {not_included}" if not_included else "")
    check(bool(re.search(r"[Aa]ccessed 2026", (P1 / 'manuscript' / 'refs.bib').read_text(encoding='utf-8')))
          if (P1 / "manuscript" / "refs.bib").exists() else False,
          "§8 vendor pricing quoted with an access date")
    check(bool(re.search(r"in preparation", tex, re.I)),
          "§8 companion Paper 2 cited as 'in preparation'")

    # ---------------------------------------------------------------- §7 desk-reject
    check(bool(re.search(r"first ever|first-ever", tex, re.I)) is False,
          "§7 no 'first ever' claim")
    check(bool(re.search(r"quote these claims rather than adopt", tex, re.I)),
          "§7 vendor claims quoted, not adopted")
    check("AGREEMENT" in tex or "agreement measures, not accuracy" in tex,
          "§7 coded-field comparisons called agreement")
    check(bool(re.search(r"closed and commercial|closed commercial", tex, re.I)),
          "§7 limitations name the closed-model problem")
    check(bool(re.search(r"Ethics and data statement", tex)),
          "§7 ethics/data statement present")
    check(bool(re.search(r"No narrative is reproduced", tex)),
          "§7 PII protocol: no narrative reproduced")
    check(bool(re.search(r"kappa|\\kappa", tex)), "§7 kappa reported")

    missing_t = [t for t in REQUIRED_TABS if not (TABS / f"{t}.tex").exists()]
    check(not missing_t, "tables T2-T8 present",
          f"missing: {missing_t}" if missing_t else f"{len(REQUIRED_TABS)} files")

    # ---------------------------------------------------------------- data integrity
    a = DATA / "analysis.json"
    if a.exists():
        j = json.loads(a.read_text())
        models = j.get("run", {}).get("models_seen", {})
        check(set(models) == {"jev-1.13.0"},
              "every call returned the pinned model id", str(models))
        ci_ok = all("ece_vs_coded_ci95" in r for r in j.get("vs_coded_fields", {}).values())
        check(ci_ok, "§7 calibration estimates carry intervals")
        grid = j.get("output_grid", {})
        two_dp = all(g.get("is_two_decimal") for g in grid.values()) if grid else False
        check(two_dp, "output-grid finding holds across every variable",
              f"{len(grid)} variables checked")
    else:
        check(None, "analysis.json checks", "run s10_analysis.py first")

    # ---------------------------------------------------------------- freshness
    # A figure or table older than the analysis it derives from is stale: it was built from
    # an earlier run and would silently ship the wrong numbers. This is the failure mode the
    # Stage-1 dry run created once already.
    if a.exists():
        t_analysis = a.stat().st_mtime
        stale = [f.name for f in list(FIGS.glob("*.pdf")) + list(TABS.glob("T*.tex"))
                 if f.stat().st_mtime < t_analysis and f.stem != "T2_schema"]
        check(not stale, "no figure or table is older than analysis.json",
              f"STALE: {stale}" if stale else "all outputs rebuilt after the analysis")
        s2 = DATA / "stage2" / "stage2_flat.parquet"
        if s2.exists():
            check(t_analysis >= s2.stat().st_mtime,
                  "analysis.json is newer than stage2_flat.parquet")

    check(None, "§7 every empirical claim carries a CI",
          "read the Results prose once written")
    check(None, "§8 quoted narratives redacted and manually read",
          "N/A while the manuscript reproduces no narrative -- re-check if that changes")
    check(None, "refs.bib [M] entries verified against publisher records",
          "classic methodology citations are from memory; verify before submission")

    # ---------------------------------------------------------------- report
    width = max(len(i) for _, i, _ in results) + 2
    print("=" * (width + 30))
    print("PAPER 1 SUBMISSION CHECKLIST")
    print("=" * (width + 30))
    for status, item, detail in results:
        mark = {"PASS": "[x]", "FAIL": "[ ]", "MANUAL": "[?]"}[status]
        print(f"{mark} {item:<{width}} {detail}")
    n_fail = sum(1 for s, _, _ in results if s == "FAIL")
    n_man = sum(1 for s, _, _ in results if s == "MANUAL")
    print("-" * (width + 30))
    print(f"{sum(1 for s,_,_ in results if s=='PASS')} pass, {n_fail} FAIL, {n_man} manual")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
