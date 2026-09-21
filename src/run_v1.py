"""Run every v1 step that follows the Stage-2 API call, in order, and report spend.

    python src/run_v1.py            # flatten -> gold frame -> analysis -> tables -> figures
    python src/run_v1.py --figures  # figures only
    python src/run_v1.py --spend    # spend report only

v1 scope is outline §6 line 295: Steps 1-4, 6, 7-keyword, figures F1-F5/F7/F9, tables T2-T8.
The human gold set labels, the frontier-LLM baseline and the robustness runs are v2; this
script emits the gold-set FRAME (free, no API) because v1 must specify that design fully.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # repository root, resolved from this file
SRC = ROOT / "paper1" / "src"
DATA = ROOT / "paper1" / "data"
PY = r"C:/Users/nib37/AppData/Local/Programs/Python/Python313/python.exe"

FIGURES = ["f01_framework.py", "f02_corpus_cost.py", "f03_prob_landscape.py",
           "f04_calibration.py", "f05_risk_coverage.py", "f07_coded_agreement.py",
           "f09_frontier.py"]

PRICE = 0.042 / 1e6
BUDGET = {
    "stage1": ("Stage 1 lean screen (500k)", 19.50, "line item + authorised reserve draw"),
    "stage2": ("Stage 2 full schema", 30.60, "line item"),
    "schema_validate_v1_1": ("Schema validation re-run (2k)", 0.90, "schema-iteration line"),
    "cost_model": ("Cost-model sweep", None, "schema-iteration line"),
    "pii_screen": ("Residual-PII screen", None, "schema-iteration line"),
}
CAP_TOTAL = 100.00
PAPER1_LINE = 55.00


def run(cmd: list[str], label: str, fatal: bool = True) -> int:
    """Run a build step. `fatal=False` for steps whose non-zero exit is a report rather
    than a failure (the submission checklist exits non-zero when items are still open)."""
    print(f"\n{'='*72}\n{label}\n{'='*72}", flush=True)
    t = time.time()
    r = subprocess.run(cmd, cwd=SRC)
    if r.returncode != 0:
        if fatal:
            raise SystemExit(f"FAILED: {label} (exit {r.returncode})")
        print(f"-- {label} reported open items (exit {r.returncode}); continuing", flush=True)
    else:
        print(f"-- ok in {time.time()-t:.1f}s", flush=True)
    return r.returncode


def spend() -> dict:
    """Read every JSONL the runs produced and total the realised input tokens."""
    rows, total = [], 0
    for f in sorted(DATA.rglob("*.jsonl")):
        if f.name.startswith("_"):
            continue
        tok = n = 0
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    u = json.loads(line).get("usage") or {}
                except Exception:
                    continue
                tok += u.get("input_tokens") or 0
                n += 1
        if not n:
            continue
        cost = tok * PRICE
        total += cost
        key = f.stem
        label, line_item, note = BUDGET.get(key, (key, None, ""))
        rows.append({"file": f.name, "label": label, "n_calls": n, "tokens": tok,
                     "cost_usd": round(cost, 4), "line_item_usd": line_item, "note": note})

    print(f"\n{'='*78}\nSPEND REPORT\n{'='*78}")
    print(f"{'stage':42s}{'calls':>9s}{'cost':>10s}{'line':>9s}")
    print("-" * 78)
    for r in rows:
        li = f"${r['line_item_usd']:.2f}" if r["line_item_usd"] else "--"
        print(f"{r['label'][:41]:42s}{r['n_calls']:>9,d}{'$'+format(r['cost_usd'],'.2f'):>10s}{li:>9s}")
    print("-" * 78)
    print(f"{'PAPER 1 TOTAL':42s}{sum(r['n_calls'] for r in rows):>9,d}{'$'+format(total,'.2f'):>10s}"
          f"{'$'+format(PAPER1_LINE,'.2f'):>9s}")
    print(f"{'Remaining against the $100 cap':42s}{'':>9s}{'$'+format(CAP_TOTAL-total,'.2f'):>10s}")
    rep = {"rows": rows, "total_usd": round(total, 4),
           "paper1_line_usd": PAPER1_LINE, "cap_usd": CAP_TOTAL,
           "remaining_usd": round(CAP_TOTAL - total, 4)}
    (DATA / "spend_report.json").write_text(json.dumps(rep, indent=1))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--spend", action="store_true")
    ap.add_argument("--skip-gold", action="store_true")
    a = ap.parse_args()

    if a.spend:
        spend()
        return

    if not a.figures:
        if (DATA / "stage1" / "stage1.jsonl").exists():
            run([PY, "s06_flatten.py", "--run", "stage1"], "Flatten Stage 1")
        if not (DATA / "stage2" / "stage2.jsonl").exists():
            raise SystemExit("stage2.jsonl missing -- run s03_full.py first")
        run([PY, "s06_flatten.py", "--run", "stage2"], "Flatten Stage 2 (applies §4.1 gating)")
        if not a.skip_gold:
            run([PY, "s05_gold_sample.py"], "Gold-set frame (§4.4 design; labels are v2)")
        run([PY, "s10_analysis.py"], "Analysis (§4.5-§4.7)")
        run([PY, "s11_tables.py"], "Tables T2-T8")
        run([PY, "s13_numbers.py"], "Results macros (manuscript/numbers.tex)")

    for f in FIGURES:
        run([PY, f"s09_figures/{f}"], f"Figure {f.split('_')[0].upper()}")

    # Its exit code reports open checklist items, which is information, not a build failure.
    rc_check = run([PY, "s12_checklist.py"], "Submission checklist (sections 7 and 8)",
                   fatal=False)

    spend()
    if rc_check:
        print("\n!! submission checklist has open FAIL items -- see above", flush=True)
    print("\nv1 artefacts:")
    for p in sorted((ROOT / "paper1" / "outputs").rglob("*.pdf")):
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
