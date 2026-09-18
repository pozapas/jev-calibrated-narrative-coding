# Calibrated Decisions at Scale

Code, schema and aggregated results for *Calibrated Decisions at Scale: Converting Police
Crash Narratives into Probabilistic Crash Variables with a System One Model (Jev)*.

The paper converts 5,018,080 Texas CRIS crash narratives (2017–2025) into probabilistic
variables using a typed-decision model, audits whether those probabilities are calibrated
against both coded CRIS fields and a human gold set, and turns calibration into an
operational quantity: the share of flagged records a human must review to reach a target
precision.

## What is here

```
src/                  the full pipeline, in run order (s01 … s18)
schemas/              the question schema, verbatim as sent to the model
manuscript/           LaTeX source and both built PDFs (elsarticle and cas-sc)
outputs/tables/       every table, as .tex and .csv
outputs/figures/      every figure, as PDF
data/                 aggregated results only — see below
```

The pipeline is numbered in execution order. The parts most likely to be useful on their own:

| file | what it is |
|---|---|
| `src/s07_metrics.py` | calibration and selective-prediction metrics with unit tests: ECE on a discrete grid, Murphy decomposition, Cox calibration slope, Spiegelhalter *z*, risk–coverage, and the positive-class review budget |
| `src/s08_cost_model.py` | the token/cost model showing input cost is governed by schema size, not narrative length |
| `src/s17_gold_analysis.py` | accuracy against the human gold set, Horvitz–Thompson weighted, with split-half recalibration |
| `src/s18_frontier_analysis.py` | the frontier-model baseline and the paired bootstrap comparing arms |
| `src/s13_numbers.py` | every number quoted in the prose, emitted as a LaTeX macro so the manuscript cannot go stale |

## What is not here, and why

**The narratives themselves are not redistributable** under the data agreement with the Texas
Department of Transportation, and are not included in any form. Neither are the record-level
intermediate files, because they carry crash identifiers: the per-judgment gold labels, the
narrative-to-coded-field join, the raw model outputs, and the two-stage run artefacts all stay
out. No narrative is reproduced in the paper either; every example shown there is synthetic
text written to illustrate structure.

What *is* released is everything aggregated: the schema, the code, and the summary JSON behind
every figure and table. The analysis scripts can be read and audited end to end, and the
figures and tables can be regenerated from the aggregated files, but the pipeline cannot be
re-run from scratch without the licensed corpus.

Researchers with their own CRIS access under an equivalent agreement can run the pipeline
directly: the paths are set at the top of each script.

## Reproducing the manuscript

With the aggregated data in place:

```bash
python src/s13_numbers.py      # regenerate numbers.tex from the analysis JSON
python src/s11_tables.py       # regenerate every table
python src/s14_build_pdf.py    # build both PDFs and fail loudly on undefined references
```

`src/run_v1.py` drives the full pipeline including the API calls, and is included for
completeness rather than because it can be run without the corpus.

## Model and pricing

All model calls use a pinned identifier (`jev-1.13.0`), recorded per call. Vendor prices are
in `src/pricing.py` with their access date and should be re-verified before reuse — they
change, and the cost findings are stated at list prices.

## Status

The manuscript is a preprint under submission. Results, numbering and text may change.
