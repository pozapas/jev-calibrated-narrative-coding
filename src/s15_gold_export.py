"""Build the human-labelling dataset for the gold set, PII-screened.

This is the gate between the corpus and any external coder. Nothing reaches the labelling
app that has not been through all three steps:

  1. Second-pass regex redaction (§6 Step 1) on top of the data owner's placeholder pass.
  2. The `pii_residual` Jev screen (§6 Step 1) over the REDACTED text. Any narrative with
     p > 0.2 is withheld from the app entirely and replaced from a reserve list.
  3. A final regex assertion: no narrative shipped may still match a date, DOB, ID, long
     digit run or case-number pattern.

Blinding: the export carries NO model output. Jev's probabilities are what the gold set is
meant to audit, so showing them to a labeller would anchor the labels and destroy the
reference. `incl_prob` and the probabilities stay in gold_frame.csv, which the analysis
joins back by Crash_ID after labelling.

Output: tool-gold/public/data/gold_tasks.json  (+ a withheld-narratives report)
"""
from __future__ import annotations
import argparse, asyncio, json, re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

import jev_runner as J
from s01_corpus import REDACT, AUDIT, redact

ROOT = Path(__file__).resolve().parents[2]   # repository root, resolved from this file
DATA = ROOT / "paper1" / "data"
GOLD = DATA / "gold"
# NOT public/: a file under public/ is served without any auth check, which would
# leave all 400 narratives downloadable by anyone who guessed the path. The app
# serves them through /api/tasks behind the access-code cookie instead.
OUT_APP = ROOT / "tool-gold" / "data"
PII_SCHEMA = ROOT / "schemas" / "pii_residual_v1.json"
PII_THRESHOLD = 0.2          # §6 Step 1
N_DOUBLE_CODED = 100         # §4.4: double-coded subset for Cohen's kappa


def load_criteria() -> dict:
    """The criteria text the model saw, which §4.4 requires labellers to see too."""
    s = json.loads((ROOT / "schemas" / "crash_factors_v1_1.json").read_text())
    out = {}
    for qid, q in s["questions"].items():
        if q["type"] != "noul" or qid == "pii_residual":
            continue
        c = q.get("criteria") or {}
        # Backticks mark `narrative` as a state key for the model. §4.4 requires labellers
        # to see the same criteria text, so the wording is untouched; only the backticks are
        # dropped, because to a human they read as noise rather than as a variable name.
        strip = lambda t: t.replace("`", "")
        out[qid] = {"instructions": strip(q["instructions"]),
                    "true": strip(c.get("true", "")), "false": strip(c.get("false", ""))}
    return out


def narratives_for(ids: list[int]) -> dict[int, str]:
    t = pq.read_table(DATA / "narratives_work.parquet",
                      columns=["Crash_ID", "narrative"]).to_pydict()
    want = set(ids)
    return {c: n for c, n in zip(t["Crash_ID"], t["narrative"]) if c in want}


def residual_hits(text: str) -> list[str]:
    return [k for k, rx in AUDIT.items() if rx.search(text)]


async def screen(rows: list[dict], out_path: Path) -> dict[int, float]:
    """Run the pii_residual Noul over the REDACTED text."""
    schema = json.loads(PII_SCHEMA.read_text())
    qs = J.build_questions(schema)
    items = [{"Crash_ID": r["Crash_ID"], "narrative": r["narrative_redacted"]} for r in rows]
    await J.run(items, qs, out_path, cost_cap_usd=0.50, label="pii_screen")
    p = {}
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            a = (rec.get("answers") or {}).get("pii_residual") or {}
            p[int(rec["Crash_ID"])] = a.get("noul")
    return p


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-only", action="store_true")
    ap.add_argument("--oversample", type=float, default=1.6,
                    help="draw this multiple of the 400-narrative target before screening")
    ap.add_argument("--target", type=int, default=400)
    a = ap.parse_args()

    # Oversample the frame, then screen down to the design size. 27% of a 400-narrative
    # frame was withheld by the PII screen in the first pass, which would have left the gold
    # set below its §4.4 target; drawing OVERSAMPLE x 400 and keeping the first 400 cleared
    # narratives preserves both the size and the probability-bin coverage.
    import s05_gold_sample as G
    frame_path = GOLD / "gold_frame_oversampled.csv"
    if a.oversample > 1.0 and not frame_path.exists():
        big = G.build_frame(G.load_random("stage2"), n_target=int(a.oversample * 400))
        big.to_csv(frame_path, index=False)
    g = pd.read_csv(frame_path if frame_path.exists() else GOLD / "gold_frame.csv")
    crit = load_criteria()
    narr = narratives_for(g["Crash_ID"].tolist())
    print(f"gold frame {len(g)}; narratives found {len(narr)}")

    rows = []
    for _, r in g.iterrows():
        cid = int(r["Crash_ID"])
        raw = narr.get(cid)
        if raw is None:
            continue
        red = redact(raw)
        active = [v for v in str(r["active_vars"]).split("|") if v and v in crit]
        controls = [v for v in str(r["control_vars"]).split("|") if v and v in crit]
        rows.append({"Crash_ID": cid, "narrative_redacted": red,
                     "vars": active + controls,
                     "n_active": len(active),
                     "regex_hits_before": residual_hits(raw),
                     "regex_hits_after": residual_hits(red)})

    before = sum(1 for r in rows if r["regex_hits_before"])
    after = sum(1 for r in rows if r["regex_hits_after"])
    print(f"regex PII hits: {before}/{len(rows)} before redaction, {after} after")

    pii = await screen(rows, GOLD / "pii_screen_oversampled.jsonl")
    withheld = [r for r in rows if (pii.get(r["Crash_ID"]) or 0) > PII_THRESHOLD
                or r["regex_hits_after"]]
    kept = [r for r in rows if r not in withheld]
    print(f"pii_residual > {PII_THRESHOLD}: "
          f"{sum(1 for r in rows if (pii.get(r['Crash_ID']) or 0) > PII_THRESHOLD)}")
    print(f"WITHHELD {len(withheld)} narratives; {len(kept)} cleared for labelling")

    (GOLD / "pii_withheld.json").write_text(json.dumps(
        [{"Crash_ID": r["Crash_ID"], "pii_p": pii.get(r["Crash_ID"]),
          "regex_hits_after": r["regex_hits_after"]} for r in withheld], indent=1))

    if a.screen_only:
        return

    # final assertion: nothing shipped may still match a PII pattern
    for r in kept:
        assert not residual_hits(r["narrative_redacted"]), \
            f"PII survived into the export for Crash_ID {r['Crash_ID']}"

    # keep the design size; anything beyond it is a reserve, not extra work for the coders
    kept = kept[: a.target]
    tasks = [{"id": r["Crash_ID"], "text": r["narrative_redacted"], "vars": r["vars"]}
             for r in kept]
    payload = {
        "generated": pd.Timestamp.utcnow().isoformat(),
        "n_tasks": len(tasks),
        "n_judgments": sum(len(t["vars"]) for t in tasks),
        "double_coded_ids": [t["id"] for t in tasks[:N_DOUBLE_CODED]],
        "pii": {"threshold": PII_THRESHOLD, "n_withheld": len(withheld),
                "screen": "second-pass regex redaction + Jev pii_residual over the "
                          "redacted text; regex assertion on export"},
        "blinding": "no model output is included; labels must be independent of Jev",
        "criteria": crit,
        "tasks": tasks,
    }
    OUT_APP.mkdir(parents=True, exist_ok=True)
    (OUT_APP / "gold_tasks.json").write_text(json.dumps(payload, indent=1))
    print(f"wrote {OUT_APP/'gold_tasks.json'}: {len(tasks)} narratives, "
          f"{payload['n_judgments']} judgments, first {N_DOUBLE_CODED} double-coded")


if __name__ == "__main__":
    asyncio.run(main())
