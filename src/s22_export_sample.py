"""Export the coded sample as CSV, one row per narrative and one column per variable.

    python src/s22_export_sample.py

The analysis reads parquet, which is the right format for it and the wrong one for anyone who
wants to open the data in Stata, R or a spreadsheet. This writes the same rows as CSV with a
data dictionary beside it.

Column order follows the schema rather than the parquet's, so a question added upstream cannot
quietly land in a different place, and the script asserts that every question in the schema
has a column and every variable column belongs to a question.

Two things a user of this file has to know, so both are columns rather than footnotes:

  `stratum` and `incl_prob`. The frame is not a simple random sample. 150,000 rows are random
  and the rest enrich rare classes, so an unweighted mean over the whole file is biased toward
  whatever was enriched. Filter to stratum == "random" for a population estimate, or weight by
  1 / incl_prob.

  the gated answers. A detail question is only interpreted when its presence question fires;
  where it did not, the answer is the no-match option. The model's ungated reply is kept in the
  matching `raw_` column for anyone studying the gate itself, and should not be read as a
  finding.

NOT for redistribution: the rows carry CRIS crash identifiers, so this file falls under the
data agreement even though it holds no narrative text.

Output: outputs/cris_coded_sample.csv  +  outputs/cris_coded_sample_dictionary.csv
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SRC = ROOT / "data" / "stage2" / "stage2_flat.parquet"
OUT = ROOT / "outputs"
SCHEMA = REPO / "schemas" / "crash_factors_v1_1.json"

# Not a model answer: how the row was drawn, and what it cost to code.
DESIGN = ["Crash_ID", "Year", "Crash_Sev_ID", "stratum", "incl_prob", "enrich_class"]
RUNMETA = ["model", "latency_s", "input_tokens"]


def main() -> None:
    d = pd.read_parquet(SRC)
    schema = json.loads(SCHEMA.read_text())
    gates = schema.get("gates", {})

    # The schema carries 28 questions; pii_residual screens narratives before release and is
    # not a crash variable, so it is not part of the coded output.
    questions = [q for q in schema["questions"] if q != "pii_residual"]

    ordered, rows = [], []
    for qid in questions:
        q = schema["questions"][qid]
        kind = q["type"]
        if kind == "noul":
            col = f"p_{qid}"
            desc = "probability the factor is present, on the model's two-decimal grid"
        elif kind == "choice":
            col = qid
            desc = "selected option, or the no-match option where the gate did not fire"
        else:
            col = f"score_{qid}"
            desc = "ordinal level"
        if col not in d.columns:
            print(f"  schema question with no column, skipped: {qid} ({kind})")
            continue
        ordered.append(col)
        rows.append({"column": col, "question": qid, "type": kind,
                     "gated_by": gates.get(qid, ""), "description": desc})
        for extra, note in ((f"conf_{qid}", "confidence in the selected option"),
                            (f"raw_{qid}", "ungated reply; not a finding, kept for gate audit")):
            if extra in d.columns:
                ordered.append(extra)
                rows.append({"column": extra, "question": qid, "type": kind,
                             "gated_by": gates.get(qid, ""), "description": note})

    assert len(rows) >= len(questions), "a schema question lost its column"
    cols = [c for c in DESIGN if c in d.columns] + ordered + \
           [c for c in RUNMETA if c in d.columns]
    missed = [c for c in d.columns if c not in cols]

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "cris_coded_sample.csv"
    d[cols].to_csv(out, index=False)

    for c, note in [("Crash_ID", "CRIS crash identifier"),
                    ("Year", "crash year"),
                    ("Crash_Sev_ID", "CRIS severity code"),
                    ("stratum", "random | enrich_kw | enrich_s1; filter to random for "
                                "population estimates"),
                    ("incl_prob", "probability this narrative entered the frame; weight by "
                                  "1/incl_prob if the enriched rows are used"),
                    ("enrich_class", "which rare class this row was drawn to enrich"),
                    ("model", "pinned model identifier returned by the service"),
                    ("latency_s", "wall-clock seconds for the call"),
                    ("input_tokens", "input tokens billed for the call")]:
        if c in cols:
            rows.insert(0 if c in DESIGN else len(rows),
                        {"column": c, "question": "", "type": "design/run",
                         "gated_by": "", "description": note})

    dic = OUT / "cris_coded_sample_dictionary.csv"
    pd.DataFrame(rows).drop_duplicates("column").to_csv(dic, index=False)

    # Count questions, not columns: a choice or score question also contributes a confidence
    # column and sometimes an ungated one, so counting rows triples some of them.
    n_pres = sum(1 for r in rows if r["type"] == "noul"
                 and r["column"] == f"p_{r['question']}")
    n_ch = sum(1 for r in rows if r["type"] == "choice" and r["column"] == r["question"])
    n_sc = sum(1 for r in rows if r["type"] == "score"
               and r["column"] == f"score_{r['question']}")
    print(f"wrote {out.name}  {len(d):,} rows x {len(cols)} cols  "
          f"({out.stat().st_size / 1e6:.1f} MB)")
    print(f"      {n_pres} presence, {n_ch} choice, {n_sc} score = "
          f"{n_pres + n_ch + n_sc} questions")
    print(f"      strata: " + ", ".join(f"{k} {v:,}" for k, v in
                                        d.stratum.value_counts().items()))
    print(f"wrote {dic.name}  {len(pd.DataFrame(rows).drop_duplicates('column'))} entries")
    if missed:
        print(f"      columns not carried over: {missed}")


if __name__ == "__main__":
    main()
