"""Compile the Elsevier (elsarticle) manuscript and report anything a reviewer would see.

    python src/s14_build_pdf.py

Runs pdflatex -> bibtex -> pdflatex x2, then fails loudly on undefined citations or
references rather than shipping a PDF full of [?] marks.
"""
from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"D:/OneDrive - Texas State University/AIT/Papers/Jev")
MAN = ROOT / "paper1" / "manuscript"
# Two official Elsevier builds share body.tex/backmatter.tex:
#   main      -> elsarticle (the long-standing Elsevier class)
#   main_cas  -> cas-sc (els-cas-templates, the current Elsevier/Overleaf template)
JOBS = ["main", "main_cas"]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=MAN, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def build(JOB: str) -> bool:
    need = [MAN / "numbers.tex"] + [
        ROOT / "paper1" / "outputs" / "tables" / f"{t}.tex"
        for t in ("T2_schema", "T3_run", "T4_prevalence", "T5_calibration",
                  "T6_selective", "T7_baseline", "T8_discrepancy",
                  "T10_gold", "T11_recalibration", "T12_threshold", "T13_frontier")]
    missing = [p.name for p in need if not p.exists()]
    if missing:
        raise SystemExit(f"cannot build: missing generated inputs {missing} "
                         f"-- run src/run_v1.py first")

    print(f"\n=== building {JOB} ===")
    run(["pdflatex", "-interaction=nonstopmode", f"{JOB}.tex"])
    run(["bibtex", JOB])
    run(["pdflatex", "-interaction=nonstopmode", f"{JOB}.tex"])
    run(["pdflatex", "-interaction=nonstopmode", f"{JOB}.tex"])

    log = (MAN / f"{JOB}.log").read_text(encoding="utf-8", errors="replace")
    errors = re.findall(r"^! (.+)$", log, re.M)
    undef_cite = sorted(set(re.findall(r"Citation `([^']+)' .*undefined", log)))
    undef_ref = sorted(set(re.findall(r"Reference `([^']+)' .*undefined", log)))
    undef_mac = sorted(set(re.findall(r"Undefined control sequence.*?\\(\w+)", log, re.S)))
    pages = re.findall(r"\((\d+) pages", log)
    pdf = MAN / f"{JOB}.pdf"

    print(f"pages: {pages[-1] if pages else '?'}   size: "
          f"{pdf.stat().st_size/1e6:.2f} MB" if pdf.exists() else "NO PDF PRODUCED")
    ok = True
    if errors:
        ok = False
        print(f"\nLaTeX errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  ! {e}")
    for label, items in (("undefined citations", undef_cite),
                         ("undefined references", undef_ref)):
        if items:
            ok = False
            print(f"\n{label}: {items}")
    if ok:
        print("clean build: no errors, no undefined citations or references")
    return ok


def main() -> None:
    results = {j: build(j) for j in JOBS if (MAN / f"{j}.tex").exists()}
    print("\n" + "=" * 60)
    for j, ok in results.items():
        print(f"{'OK  ' if ok else 'FAIL'}  {j}.pdf")
    if not all(results.values()):
        raise SystemExit("at least one build has problems")


if __name__ == "__main__":
    main()
