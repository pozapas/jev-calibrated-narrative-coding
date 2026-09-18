"""Step 1 -- corpus preparation, keyword families, second-pass redaction audit, sampling frames.

Outline: Paper1 §6 Step 1. Implemented as a two-pass Python stream rather than R `fread`
because only ~6 GB RAM is free on this machine and the 2.69 GB CSV expands well past that
when loaded whole. Streaming is constant-memory and, per the outline's own note, faster
(~5 min/pass vs ~15 min for fread).

Pass 1: row counts, per-year / severity / length / casing descriptives, the 10 keyword
        families over ALL narratives (population prevalence, free), residual-PII regex
        audit, and the full Crash_ID list.
Pass 2: materialise narratives for the working set = 500k Stage-1 sample UNION all
        keyword hits (Stage-2 enrichment draws from these), with a redacted display copy.

Outputs (paper1/data/):
  corpus_stats.json          descriptives for F2 / T3 / §3
  keyword_flags.parquet      Crash_ID + 10 booleans, all 5.02 M rows
  stage1_frame.parquet       500k Crash_ID sample, seed 42
  stage2_random_frame.parquet  first 150k of the Stage-1 sample
  narratives_work.parquet    Crash_ID, Year, Crash_Sev_ID, narrative, narrative_redacted
Idempotent: skips any pass whose outputs already exist unless --force.
"""
from __future__ import annotations
import argparse, csv, json, re, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CSV = Path(r"D:/OneDrive - Texas State University/Das, Subasish's files - 2027_TRBAM/CRIS/CRIS_17_25_impWithNarr_pii_clean.csv")
DATA = Path(__file__).resolve().parents[1] / "data"
SEED = 42
N_STAGE1 = 500_000
N_STAGE2_RANDOM = 150_000
MIN_CHARS = 40
MAX_ENRICH_PER_CLASS = 5_000  # §6 Step 3; we materialise all hits and subset later

csv.field_size_limit(1 << 24)

# ---------------------------------------------------------------- keyword families
# From spike/sample2k.R, applied here to the full corpus (§4.3 "population keyword pass").
KW = {
    "preg": r"pregnan|unborn|fetus|fetal|trimester|weeks along|miscarr",
    "medical": r"seizure|diabet|passed out|blacked out|fainted|heart attack|stroke|unconscious|medical (episode|condition|emergency|event)|dementia|alzheimer",
    "drug": r"marijuana|methamphetamine|meth |cocaine|fentanyl|xanax|heroin|pills|narcotic|controlled substance|under the influence of (a )?drug",
    "hydro": r"hydroplan",
    "defect": r"tire blow|blowout|blew out|brake fail|brakes fail|mechanical fail|steering fail",
    "rage": r"road rage|angry|anger|enraged|argument|fight|confront|brandish|assault",
    "fatigue": r"fell asleep|asleep|drowsy|fatigue|nodded off",
    "intent": r"suicid|intentional|deliberate|on purpose|rammed",
    "wrongway": r"wrong way|wrong direction",
    "animal": r"deer|hog|cow|horse|dog|coyote|cattle",
}
KW_RE = {k: re.compile(v, re.IGNORECASE) for k, v in KW.items()}
KW_KEYS = list(KW)

# ---------------------------------------------------------------- second-pass redaction
# §6 Step 1. Order matters: DOB and titled names before the bare-date rule.
REDACT = [
    (re.compile(r"D\.?O\.?B\.?\s*[:]?\s*\S+", re.IGNORECASE), "DOB [DATE]"),
    (re.compile(r"\b(TX ID|DL#|DL #|ID#|ID #|LIC#|DL:)\s*\w+", re.IGNORECASE), "[ID]"),
    (re.compile(r"\b(CASE|REPORT|PR)\s*#\s*[\w-]+", re.IGNORECASE), "[CASE]"),
    (re.compile(r"\b(OFC|OFFICER|CPL|SGT|DEPUTY|TROOPER|DR|MR|MRS|MS)\.?\s+[A-Z][A-Za-z.]+(\s+[A-Z][A-Za-z]+)?"), r"\1 [NAME]"),
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), "[DATE]"),
    (re.compile(r"\b\d{7,}\b"), "[NUM]"),
]
# audit-only patterns (measure residual PII rate BEFORE the second pass, for §3)
AUDIT = {
    "date": re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    "dob": re.compile(r"D\.?O\.?B\.?\s*[:]?\s*\S+", re.IGNORECASE),
    "idnum": re.compile(r"\b(TX ID|DL#|DL #|ID#|ID #|LIC#|DL:)\s*\w+", re.IGNORECASE),
    "digitrun": re.compile(r"\b\d{7,}\b"),
    "case": re.compile(r"\b(CASE|REPORT|PR)\s*#\s*[\w-]+", re.IGNORECASE),
    "titled_name": re.compile(r"\b(OFC|OFFICER|CPL|SGT|DEPUTY|TROOPER|DR|MR|MRS|MS)\.?\s+[A-Z][A-Za-z.]+"),
}


def redact(text: str) -> str:
    for pat, rep in REDACT:
        text = pat.sub(rep, text)
    return text


def _open():
    return open(CSV, newline="", encoding="utf-8", errors="replace")


def pass1() -> dict:
    """Stream once: counts, descriptives, keyword flags, PII audit, Crash_ID list."""
    t0 = time.time()
    ids: list[np.int64] = []
    flags = {k: [] for k in KW_KEYS}
    n_total = n_kept = n_empty = n_short = 0
    year_total, year_kept, year_empty = Counter(), Counter(), Counter()
    sev_kept = Counter()
    casing_by_year = {}          # year -> Counter({"upper","mixed"})
    audit_hits = Counter()
    lengths = []                  # reservoir for the length distribution (F2b)
    rng = np.random.default_rng(SEED)
    RESERVOIR = 400_000

    with _open() as f:
        rdr = csv.DictReader(f)
        cols = rdr.fieldnames
        print(f"columns: {cols}", flush=True)
        for row in rdr:
            n_total += 1
            yr = row.get("Year") or "NA"
            year_total[yr] += 1
            narr = row.get("Investigator_Narrative") or ""
            L = len(narr)
            if L == 0:
                n_empty += 1
                year_empty[yr] += 1
                continue
            if L <= MIN_CHARS:
                n_short += 1
                continue
            n_kept += 1
            year_kept[yr] += 1
            sev_kept[row.get("Crash_Sev_ID") or "NA"] += 1
            try:
                ids.append(int(row["Crash_ID"]))
            except (TypeError, ValueError):
                ids.append(-1)
            for k, rx in KW_RE.items():
                flags[k].append(bool(rx.search(narr)))
            # casing proxy for agency style
            up = "upper" if narr == narr.upper() else "mixed"
            casing_by_year.setdefault(yr, Counter())[up] += 1
            for k, rx in AUDIT.items():
                if rx.search(narr):
                    audit_hits[k] += 1
            if len(lengths) < RESERVOIR:
                lengths.append(L)
            elif rng.random() < RESERVOIR / n_kept:
                lengths[rng.integers(RESERVOIR)] = L
            if n_total % 500_000 == 0:
                print(f"  pass1 {n_total:,} rows  kept {n_kept:,}  {time.time()-t0:.0f}s", flush=True)

    ids_arr = np.asarray(ids, dtype=np.int64)
    del ids
    tbl = pa.table({"Crash_ID": ids_arr, **{f"kw_{k}": pa.array(flags[k], type=pa.bool_()) for k in KW_KEYS}})
    pq.write_table(tbl, DATA / "keyword_flags.parquet", compression="zstd")
    kw_counts = {k: int(np.count_nonzero(flags[k])) for k in KW_KEYS}
    del flags

    lengths_arr = np.asarray(lengths)
    stats = {
        "csv": str(CSV),
        "n_rows_total": n_total,
        "n_narrative_empty": n_empty,
        "n_narrative_le_40": n_short,
        "n_kept_gt_40": n_kept,
        "min_chars": MIN_CHARS,
        "empty_share": round(n_empty / max(n_total, 1), 5),
        "year_total": dict(sorted(year_total.items())),
        "year_kept": dict(sorted(year_kept.items())),
        "year_empty": dict(sorted(year_empty.items())),
        "severity_kept": dict(sev_kept.most_common()),
        "casing_by_year": {y: dict(c) for y, c in sorted(casing_by_year.items())},
        "length_quantiles_chars": {
            str(q): float(np.quantile(lengths_arr, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
        },
        "length_mean_chars": float(lengths_arr.mean()),
        "length_reservoir_n": int(lengths_arr.size),
        "keyword_counts": kw_counts,
        "keyword_prevalence": {k: round(v / max(n_kept, 1), 6) for k, v in kw_counts.items()},
        "residual_pii_before_second_pass": {
            k: {"n": int(v), "share": round(v / max(n_kept, 1), 5)} for k, v in audit_hits.items()
        },
        "pass1_seconds": round(time.time() - t0, 1),
    }
    (DATA / "corpus_stats.json").write_text(json.dumps(stats, indent=1))
    np.save(DATA / "_length_reservoir.npy", lengths_arr)
    print(json.dumps({k: stats[k] for k in ("n_rows_total", "n_kept_gt_40", "keyword_counts")}, indent=1), flush=True)
    return stats


def build_frames() -> set[int]:
    """Sample 500k Stage-1 Crash_IDs (seed 42); first 150k -> Stage-2 random frame.
    Returns the working-set ID list (Stage-1 sample UNION all keyword hits)."""
    t = pq.read_table(DATA / "keyword_flags.parquet")
    ids = t.column("Crash_ID").to_numpy()
    n = ids.size
    rng = np.random.default_rng(SEED)
    sel = rng.choice(n, size=min(N_STAGE1, n), replace=False)
    sel.sort()                       # positional order == corpus order
    stage1_ids = ids[sel]
    rng.shuffle(stage1_ids)          # randomise so "first 150k" is a random subset
    stage2_ids = stage1_ids[:N_STAGE2_RANDOM]

    pq.write_table(pa.table({"Crash_ID": stage1_ids}), DATA / "stage1_frame.parquet")
    pq.write_table(pa.table({"Crash_ID": stage2_ids}), DATA / "stage2_random_frame.parquet")

    any_hit = np.zeros(n, dtype=bool)
    for k in KW_KEYS:
        any_hit |= t.column(f"kw_{k}").to_numpy(zero_copy_only=False)
    work = set(ids[any_hit].tolist()) | set(stage1_ids.tolist())
    print(f"stage1={stage1_ids.size:,} stage2_random={stage2_ids.size:,} "
          f"keyword_hits={int(any_hit.sum()):,} working_set={len(work):,}", flush=True)
    return work


def pass2(work: set[int]) -> None:
    """Materialise narratives (raw + redacted) for the working set."""
    t0 = time.time()
    out = {"Crash_ID": [], "Year": [], "Crash_Sev_ID": [], "Cnty_ID": [],
           "narrative": [], "narrative_redacted": [], "nchar": []}
    n_seen = 0
    with _open() as f:
        for row in csv.DictReader(f):
            n_seen += 1
            try:
                cid = int(row["Crash_ID"])
            except (TypeError, ValueError):
                continue
            if cid not in work:
                continue
            narr = row.get("Investigator_Narrative") or ""
            if len(narr) <= MIN_CHARS:
                continue
            out["Crash_ID"].append(cid)
            out["Year"].append(row.get("Year"))
            out["Crash_Sev_ID"].append(row.get("Crash_Sev_ID"))
            out["Cnty_ID"].append(row.get("Cnty_ID"))
            out["narrative"].append(narr)
            out["narrative_redacted"].append(redact(narr))
            out["nchar"].append(len(narr))
            if n_seen % 1_000_000 == 0:
                print(f"  pass2 {n_seen:,} scanned  {len(out['Crash_ID']):,} kept  {time.time()-t0:.0f}s", flush=True)
    pq.write_table(pa.table(out), DATA / "narratives_work.parquet", compression="zstd")
    print(f"narratives_work.parquet rows={len(out['Crash_ID']):,} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if a.force or not (DATA / "keyword_flags.parquet").exists():
        pass1()
    else:
        print("pass1 outputs exist, skipping", flush=True)
    work = build_frames()
    if a.force or not (DATA / "narratives_work.parquet").exists():
        pass2(work)
    else:
        print("pass2 output exists, skipping", flush=True)
    print("STEP 1 DONE", flush=True)
