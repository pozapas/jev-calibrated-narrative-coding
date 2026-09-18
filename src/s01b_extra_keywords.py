"""Extra keyword families for the §4.8 baseline.

`spike/sample2k.R` defines 10 families chosen to ENRICH rare classes, so it has no family
for alcohol, phone use or seat belts -- three of the variables that do have a coded-field
reference. Comparing a keyword baseline that covers 5 variables against Jev on 6 would be a
macro-F1 over different variable sets, so the missing families are added here.

§6 Step 1 explicitly allows this ("keyword families ... extend as needed"). These are
computed on `narratives_work.parquet` (the Stage-1 sample plus keyword hits) rather than on
all 5.02 M rows, because they are needed only to SCORE the baseline on the evaluated
narratives, not to report a population prevalence. The original 10 families remain the ones
reported at population scale in T4.

Output: data/keyword_flags_extra.parquet (Crash_ID + the new boolean columns).
"""
from __future__ import annotations
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DATA = Path(r"D:/OneDrive - Texas State University/AIT/Papers/Jev/paper1/data")

EXTRA = {
    # tuned to the same spirit as the spike families: over-find, let the model filter
    "alcohol": r"\balcohol|intoxicat|\bdwi\b|\bdui\b|drunk|\bbac\b|been drinking|"
               r"odor of an alcoholic|open container|breathalyz|field sobriety",
    "phone": r"cell ?phone|cellular|mobile (phone|device)|texting|\btext(ed|ing)?\b|"
             r"on (the|his|her|their) phone|looking at (his|her|their) phone",
    "unbelted": r"not wearing (a )?(seat ?belt|safety belt)|unbelted|unrestrained|"
                r"no (seat ?belt|restraint)|seat ?belt was not|without a seat ?belt",
}


def main() -> None:
    t = pq.read_table(DATA / "narratives_work.parquet", columns=["Crash_ID", "narrative"])
    ids = t.column("Crash_ID").to_pylist()
    narr = t.column("narrative").to_pylist()
    print(f"scanning {len(ids):,} working narratives for {len(EXTRA)} extra families")

    cols = {"Crash_ID": ids}
    for name, pat in EXTRA.items():
        rx = re.compile(pat, re.IGNORECASE)
        flags = [bool(rx.search(n)) for n in narr]
        cols[f"kw_{name}"] = pa.array(flags, type=pa.bool_())
        print(f"  kw_{name:9s} {sum(flags):8,d}  ({100*sum(flags)/len(flags):6.3f}% of working set)")

    pq.write_table(pa.table(cols), DATA / "keyword_flags_extra.parquet", compression="zstd")
    print(f"wrote {DATA/'keyword_flags_extra.parquet'}")


if __name__ == "__main__":
    main()
