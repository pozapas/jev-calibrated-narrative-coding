"""Step 3 -- Stage 2 full schema (§6): 27 questions on 200k narratives
= 150k random (subset of Stage 1) + up to 50k enriched for rare classes.

The enrichment strata and the inclusion probability are recorded PER NARRATIVE. This is the
one thing that cannot be reconstructed after the fact: every population estimate in §4.3
uses either the random part alone or inverse-inclusion weights, so a JSONL written without
`stratum` and `incl_prob` would have to be re-run at full cost.

Budget line: $30.6 (200k x 3,640 input tokens).
"""
from __future__ import annotations
import argparse, asyncio, json, re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import jev_runner as J

ROOT = Path(__file__).resolve().parents[2]   # repository root, resolved from this file
DATA = ROOT / "paper1" / "data"
OUT = DATA / "stage2" / "stage2.jsonl"
SCHEMA = ROOT / "schemas" / "crash_factors_v1_1.json"
COST_CAP = 30.60
SEED = 42

RARE = ["preg", "medical", "drug", "hydro", "defect", "rage", "fatigue", "intent", "wrongway", "animal"]
LEAN_FOR_CLASS = {"preg": "preg_mentioned", "medical": "medical_episode", "drug": "drug_involved",
                  "hydro": "hydroplane", "defect": "vehicle_defect", "rage": "aggression",
                  "fatigue": "fatigue", "intent": "intentional", "wrongway": "wrong_way"}
MAX_PER_CLASS_KW = 5_000
MAX_PER_CLASS_S1 = 5_000
N_RANDOM = 150_000
# Measured on the 2k schema-validation run (s02b): the v1.1 27-question schema costs
# 3,682.7 input tokens per narrative, not the 3,640 the outline assumed, because rule 6
# lengthened the aggression criteria. 200k x 3,682.7 would be $30.93 against a $30.60 line,
# so the frame is trimmed to fit. The 150k RANDOM part is the inferential backbone and is
# never trimmed; only the enriched allocation gives way.
MEAN_TOKENS_MEASURED = 3682.7


def stage1_positive_ids() -> dict[str, set[int]]:
    """Crash_IDs with Stage-1 p > 0.5 per lean variable (empty if Stage 1 has not run)."""
    pos: dict[str, set[int]] = {v: set() for v in LEAN_FOR_CLASS.values()}
    f = DATA / "stage1" / "stage1.jsonl"
    if not f.exists():
        print("NOTE: stage1.jsonl absent -- enrichment will use keyword hits only", flush=True)
        return pos
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            cid = r.get("Crash_ID")
            for qid, a in (r.get("answers") or {}).items():
                if qid in pos and (a.get("noul") or 0) > 0.5:
                    pos[qid].add(cid)
    print("stage1 positives: " + ", ".join(f"{k}={len(v):,}" for k, v in pos.items()), flush=True)
    return pos


def build_frame(write: bool = True) -> list[dict]:
    """150k random + enriched, with stratum and inclusion probability recorded.

    `write=False` exercises the whole construction (enrichment pools, inclusion
    probabilities, budget trim) without persisting a frame -- used to test the logic against
    a partially complete Stage 1 before committing to the real run.
    """
    rng = np.random.default_rng(SEED)
    stage1_ids = pq.read_table(DATA / "stage1_frame.parquet").column("Crash_ID").to_pylist()
    random_ids = pq.read_table(DATA / "stage2_random_frame.parquet").column("Crash_ID").to_pylist()
    n_stage1 = len(stage1_ids)

    kw = pq.read_table(DATA / "keyword_flags.parquet").to_pydict()
    kw_ids = np.asarray(kw["Crash_ID"])
    n_corpus = kw_ids.size
    hits = {c: kw_ids[np.asarray(kw[f"kw_{c}"], dtype=bool)] for c in RARE}

    # inclusion probability for the random part: P(in Stage-2 random | in corpus)
    p_random = len(random_ids) / n_corpus
    frame: dict[int, dict] = {
        cid: {"Crash_ID": cid, "stratum": "random", "incl_prob": p_random, "enrich_class": ""}
        for cid in random_ids
    }

    s1pos = stage1_positive_ids()
    p_stage1 = n_stage1 / n_corpus
    for c in RARE:
        pool = np.array([i for i in hits[c] if i not in frame], dtype=np.int64)
        take = pool if pool.size <= MAX_PER_CLASS_KW else rng.choice(pool, MAX_PER_CLASS_KW, replace=False)
        # P(selected) = P(keyword hit -- deterministic, so 1) x P(drawn from the hit pool)
        pr = min(1.0, MAX_PER_CLASS_KW / max(pool.size, 1))
        for i in take.tolist():
            frame[i] = {"Crash_ID": i, "stratum": "enrich_kw", "incl_prob": pr, "enrich_class": c}

        qid = LEAN_FOR_CLASS.get(c)
        if not qid:
            continue
        pool2 = np.array([i for i in s1pos.get(qid, ()) if i not in frame], dtype=np.int64)
        if pool2.size == 0:
            continue
        take2 = pool2 if pool2.size <= MAX_PER_CLASS_S1 else rng.choice(pool2, MAX_PER_CLASS_S1, replace=False)
        pr2 = p_stage1 * min(1.0, MAX_PER_CLASS_S1 / max(pool2.size, 1))
        for i in take2.tolist():
            frame[i] = {"Crash_ID": i, "stratum": "enrich_s1", "incl_prob": pr2, "enrich_class": c}

    rows = list(frame.values())

    # ---- trim the enriched part to fit the budget line -------------------------------
    # SAFETY_MARGIN keeps the projection clear of the cap. Sizing the frame to exactly the
    # cap leaves no room for the realised mean token count to come in above the estimate,
    # and a tripwire stop mid-run would truncate whichever enriched classes happen to sit
    # last in the frame -- a biased enrichment rather than a smaller one. 1% of headroom
    # costs ~2,000 enriched narratives and removes that failure mode.
    SAFETY_MARGIN = 0.99
    n_max = int(SAFETY_MARGIN * COST_CAP / (MEAN_TOKENS_MEASURED * J.PRICE_PER_M_INPUT / 1e6))
    if len(rows) > n_max:
        rnd = [r for r in rows if r["stratum"] == "random"]
        enr = [r for r in rows if r["stratum"] != "random"]
        keep_enr = max(0, n_max - len(rnd))
        # drop proportionally within each enrich class so no class is wiped out
        by_class: dict[str, list] = {}
        for r in enr:
            by_class.setdefault(r["enrich_class"], []).append(r)
        # note: rng has already been consumed by the per-class draws above; the trim sample is
        # therefore reproducible for a fixed class list but shifts if RARE gains a member.
        frac = keep_enr / max(len(enr), 1)
        # Allocate with largest-remainder rather than independent rounding: rounding each
        # class separately can overshoot keep_enr by up to one per class, and overshooting
        # by even one narrative pushes the projected spend past the cap, which would make
        # the tripwire fire at the very end of an hour-long run.
        quotas = {c: len(lst) * frac for c, lst in by_class.items()}
        alloc = {c: min(int(q), len(by_class[c])) for c, q in quotas.items()}
        spare = keep_enr - sum(alloc.values())
        for c, _ in sorted(quotas.items(), key=lambda kv: -(kv[1] - int(kv[1]))):
            if spare <= 0:
                break
            if alloc[c] < len(by_class[c]):
                alloc[c] += 1
                spare -= 1

        trimmed = []
        for c, lst in by_class.items():
            k = alloc[c]
            sel = lst if k >= len(lst) else [lst[i] for i in
                                             rng.choice(len(lst), k, replace=False)]
            # the trim is part of the inclusion probability, so record it
            adj = k / max(len(lst), 1)
            for r in sel:
                r["incl_prob"] = r["incl_prob"] * adj
            trimmed.extend(sel)
        rows = rnd + trimmed
        assert len(trimmed) <= keep_enr, (len(trimmed), keep_enr)
        print(f"TRIMMED enriched {len(enr):,} -> {len(trimmed):,} to fit ${COST_CAP:.2f} "
              f"at {MEAN_TOKENS_MEASURED:.0f} tok/narrative (n_max={n_max:,})", flush=True)

    from collections import Counter
    if write:
        pq.write_table(pa.table({k: [r[k] for r in rows] for k in
                                ("Crash_ID", "stratum", "incl_prob", "enrich_class")}),
                       DATA / "stage2_frame.parquet")
    else:
        print("DRY RUN: frame not written", flush=True)
    by_strat = Counter(r["stratum"] for r in rows)
    by_class = Counter(r["enrich_class"] for r in rows if r["enrich_class"])
    proj = len(rows) * MEAN_TOKENS_MEASURED * J.PRICE_PER_M_INPUT / 1e6
    print(f"stage2 frame: {len(rows):,}  {dict(by_strat)}", flush=True)
    print(f"  enrichment achieved per class: {dict(by_class)}", flush=True)
    print(f"  projected cost ${proj:.2f} against cap ${COST_CAP:.2f}", flush=True)
    report = {
        "n_total": len(rows), "by_stratum": dict(by_strat),
        "enrichment_per_class": dict(by_class),
        "mean_tokens_measured": MEAN_TOKENS_MEASURED,
        "projected_cost_usd": round(proj, 2), "cost_cap_usd": COST_CAP,
        "n_max_within_cap": n_max,
        "note": "150k random stratum never trimmed; enriched allocation trimmed "
                "proportionally within class and incl_prob adjusted for the trim.",
    }
    if write:
        (DATA / "stage2_frame_report.json").write_text(json.dumps(report, indent=1))
    # invariants that must hold however the pools came out
    assert all(0 < r["incl_prob"] <= 1 for r in rows), "inclusion probability out of range"
    assert sum(1 for r in rows if r["stratum"] == "random") == len(set(
        r["Crash_ID"] for r in rows if r["stratum"] == "random")), "duplicate random ids"
    assert len(rows) == len({r["Crash_ID"] for r in rows}), "duplicate Crash_IDs in frame"
    assert len(rows) <= n_max, f"frame {len(rows)} exceeds budget cap {n_max}"
    return rows


def attach_narratives(rows: list[dict], drug_re: re.Pattern) -> list[dict]:
    t = pq.read_table(DATA / "narratives_work.parquet",
                      columns=["Crash_ID", "Year", "Crash_Sev_ID", "narrative"]).to_pydict()
    by_id = {c: (y, s, n) for c, y, s, n in
             zip(t["Crash_ID"], t["Year"], t["Crash_Sev_ID"], t["narrative"])}
    out, missing = [], 0
    for r in rows:
        v = by_id.get(r["Crash_ID"])
        if v is None:
            missing += 1
            continue
        y, s, n = v
        r = dict(r, Year=y, Crash_Sev_ID=s, narrative=n,
                 # §4.2 rule 3: record which drug tokens the narrative actually contains, so an
                 # `unspecified_drug` answer can be told apart from a missed named drug.
                 drug_tokens_found=sorted({m.group(0).lower() for m in drug_re.finditer(n)}))
        out.append(r)
    if missing:
        print(f"WARNING: {missing:,} frame ids have no narrative in narratives_work", flush=True)
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cap", type=float, default=COST_CAP)
    ap.add_argument("--rebuild-frame", action="store_true")
    ap.add_argument("--dry-run-frame", action="store_true",
                    help="build and validate the frame, write nothing, make no API calls")
    a = ap.parse_args()

    schema = json.loads(SCHEMA.read_text())
    drug_re = re.compile(schema["drug_token_regex"], re.IGNORECASE)

    # pii_residual lives in the same schema file for convenience but is NOT one of the 27
    # crash variables: it is the §6 Step 1 display screen, run separately on the handful of
    # narratives that might be quoted. Sending it here would make every call a 28-question
    # call -- about 122 extra tokens each, ~$1.00 over the frame, which is enough to push the
    # run past its budget line -- and would contradict the "27-question schema" the cost
    # model and the manuscript are both built on.
    qs = {k: v for k, v in schema["questions"].items() if k != "pii_residual"}
    assert len(qs) == 27, f"expected 27 crash questions, got {len(qs)}: {sorted(qs)}"
    questions = J.build_questions({"questions": qs})

    if a.dry_run_frame:
        build_frame(write=False)
        return

    fp = DATA / "stage2_frame.parquet"
    if a.rebuild_frame or not fp.exists():
        rows = build_frame()
    else:
        rows = pq.read_table(fp).to_pylist()
        print(f"stage2 frame loaded: {len(rows):,}", flush=True)

    items = attach_narratives(rows, drug_re)
    # Interleave the strata. The frame is built as random-then-enriched, so if anything did
    # stop the run early (tripwire, rate limit, crash) an unshuffled order would lose whole
    # enriched classes. Shuffled, any prefix is a representative sample of the frame, and the
    # recorded incl_prob stays interpretable. Seeded, so the order is reproducible.
    np.random.default_rng(SEED).shuffle(items)
    if a.limit:
        items = items[:a.limit]
    print(f"stage2: {len(items):,} narratives, {len(questions)} questions, cap ${a.cap:.2f}", flush=True)

    await J.run(items, questions, OUT, cost_cap_usd=a.cap, label="stage2",
                meta_fn=lambda r: {"Year": r["Year"], "Crash_Sev_ID": r["Crash_Sev_ID"],
                                   "stratum": r["stratum"], "incl_prob": r["incl_prob"],
                                   "enrich_class": r["enrich_class"],
                                   "drug_tokens_found": r["drug_tokens_found"]})


if __name__ == "__main__":
    asyncio.run(main())
