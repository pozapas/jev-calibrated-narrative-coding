"""Step 2 -- Stage 1 lean screen (§6): 10 presence Nouls over the 500k Crash_ID sample.

Budget line: $19.5 (500k x ~930 input tokens). The cap is enforced by jev_runner's
tripwire, and a projection after the first 2,000 calls aborts the run if the realised
tokens/narrative would blow the line rather than discovering it three hours in.

Output: paper1/data/stage1/stage1.jsonl (append-only, resumable) + a spend report.
"""
from __future__ import annotations
import argparse, asyncio, json
from pathlib import Path

import pyarrow.parquet as pq

import jev_runner as J

ROOT = Path(__file__).resolve().parents[2]   # repository root, resolved from this file
DATA = ROOT / "paper1" / "data"
OUT = DATA / "stage1" / "stage1.jsonl"
SCHEMA = ROOT / "schemas" / "crash_factors_lean_v1_1.json"
# $19.50 line item + ~$5.10 drawn from the $38 reserve, authorised by the user, because the
# measured lean-schema cost is 1,101 + 0.237*nchar tokens (~1,170/narrative), not the ~930 the
# outline assumed. 500k is kept because it is a headline number in §4.3 and the abstract.
COST_CAP = 25.50


def load_items(limit: int = 0) -> list[dict]:
    frame = set(pq.read_table(DATA / "stage1_frame.parquet").column("Crash_ID").to_pylist())
    t = pq.read_table(DATA / "narratives_work.parquet",
                      columns=["Crash_ID", "Year", "Crash_Sev_ID", "narrative"])
    d = t.to_pydict()
    items = [{"Crash_ID": c, "Year": y, "Crash_Sev_ID": s, "narrative": n}
             for c, y, s, n in zip(d["Crash_ID"], d["Year"], d["Crash_Sev_ID"], d["narrative"])
             if c in frame]
    missing = len(frame) - len(items)
    if missing:
        print(f"WARNING: {missing:,} Stage-1 frame ids have no narrative in narratives_work", flush=True)
    return items[:limit] if limit else items


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="run only the first N (smoke test)")
    ap.add_argument("--cap", type=float, default=COST_CAP)
    a = ap.parse_args()

    schema = json.loads(SCHEMA.read_text())
    questions = J.build_questions(schema)
    items = load_items(a.limit)
    print(f"stage1: {len(items):,} narratives, {len(questions)} questions, cap ${a.cap:.2f}", flush=True)

    await J.run(items, questions, OUT, cost_cap_usd=a.cap, label="stage1",
                meta_fn=lambda r: {"Year": r["Year"], "Crash_Sev_ID": r["Crash_Sev_ID"]})


if __name__ == "__main__":
    asyncio.run(main())
