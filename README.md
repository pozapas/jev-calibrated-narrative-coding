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
src/                  the full pipeline, in run order (s01 … s23)
schemas/              the question schema, verbatim as sent to the model
outputs/tables/       every table in the paper, as .tex and .csv
outputs/figures/      every figure in the paper, as PDF and PNG
outputs/numbers.tex   every number quoted in the paper, as LaTeX macros
data/                 aggregated results only, see below
```

The pipeline is numbered in execution order. The parts most likely to be useful on their own:

| file | what it is |
|---|---|
| `src/s07_metrics.py` | calibration and selective-prediction metrics with unit tests: ECE on a discrete grid, Murphy decomposition, Cox calibration slope, Spiegelhalter *z*, risk–coverage, and the positive-class review budget |
| `src/s08_cost_model.py` | the token/cost model showing input cost is governed by schema size, not narrative length |
| `src/s17_gold_analysis.py` | accuracy against the human gold set, Horvitz–Thompson weighted, with split-half recalibration |
| `src/s18_frontier_analysis.py` | the frontier-model baseline and the paired bootstrap comparing arms |
| `src/s13_numbers.py` | every number quoted in the prose, emitted as a LaTeX macro so the paper cannot go stale |
| `src/s05b_design_weights.py` | the calibration weights that make the human reference set stand for the population, with the design report behind them |
| `src/s21_downstream.py` | what changes in a severity diagnosis when the calibrated narrative variables are added to the coded fields |
| `src/tests/test_metrics.py` | 44 unit tests over the metric implementations |

## What is not here, and why

**The narratives themselves are not redistributable** under the data agreement with the Texas
Department of Transportation, and are not included in any form. Neither are the record-level
intermediate files, because they carry crash identifiers: the per-judgment gold labels, the
narrative-to-coded-field join, the raw model outputs, and the two-stage run artefacts all stay
out. No narrative is reproduced in the paper either; every example shown there is synthetic
text written to illustrate structure.

**The manuscript is not published here.** The LaTeX source and the built PDF are kept out
of this repository so that distribution of the paper stays with the journal. What this
repository carries is the code, the schema, the aggregated results, and every figure,
table and number the paper reports.

What *is* released is everything aggregated: the schema, the code, and the summary JSON behind
every figure and table. The analysis scripts can be read and audited end to end, and the
figures and tables can be regenerated from the aggregated files, but the pipeline cannot be
re-run from scratch without the licensed corpus.

Researchers with their own CRIS access under an equivalent agreement can run the pipeline
directly: the paths are set at the top of each script.

## Reproducing the figures, tables and numbers

With the aggregated data in place:

```bash
python src/s13_numbers.py          # outputs/numbers.tex from the analysis JSON
python src/s11_tables.py           # every table, as .tex and .csv
python -m src.s09_figures          # every figure
python -m pytest src/tests         # 44 metric tests
```

Each of these reads only the aggregated files in `data/`, so they run without the corpus.

`src/run_v1.py` drives the full pipeline including the API calls, and is included for
completeness rather than because it can be run without the corpus.

## Model and pricing

All model calls use a pinned identifier (`jev-1.13.0`), recorded per call. Vendor prices are
in `src/pricing.py` with their access date and should be re-verified before reuse, since they
change, and the cost findings are stated at list prices.

## Status

The manuscript is a preprint under submission. Results, numbering and text may change.
