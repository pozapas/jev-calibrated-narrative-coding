"""Pre-flight: re-run the v1.1 full schema on the 2,000-narrative spike sample and compare
against the v1 answers already in spike/spike_results.jsonl.

Two jobs:
  1. Validate the §4.2 rule-6 fix -- aggression must stop firing on police pursuits while
     leaving genuine road-rage cases alone. Gives the paired before/after evidence the
     manuscript cites for design rule 6.
  2. Smoke-test the exact schema object that Stage 2 will send, at 2k scale, before
     committing $30 to a 200k run.

Budget line: schema iteration re-runs, $0.9 total. This run is ~$0.31.
"""
from __future__ import annotations
import argparse, asyncio, csv, json, re
from pathlib import Path

import numpy as np

import jev_runner as J
import s07_metrics as M

ROOT = Path(__file__).resolve().parents[2]   # repository root, resolved from this file
DATA = ROOT / "paper1" / "data"
OUT = DATA / "schema_validate_v1_1.jsonl"
SCHEMA = ROOT / "schemas" / "crash_factors_v1_1.json"
SPIKE_CSV = ROOT / "spike" / "spike_sample_2000.csv"
SPIKE_V1 = ROOT / "spike" / "spike_results.jsonl"
COST_CAP = 0.45

PURSUIT = re.compile(r"pursuit|pursuing|evading|evaded|police chase|fleeing from|chased by", re.IGNORECASE)


def load_spike() -> list[dict]:
    import sys
    csv.field_size_limit(1 << 24)
    with open(SPIKE_CSV, newline="", encoding="utf-8") as f:
        return [{"Crash_ID": int(r["Crash_ID"]), "narrative": r["Investigator_Narrative"],
                 "stratum": r["stratum"], "Year": r["Year"]}
                for r in csv.DictReader(f)]


def load_v1() -> dict[int, dict]:
    out = {}
    with open(SPIKE_V1, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                out[int(r["Crash_ID"])] = {k: v for k, v in (r.get("answers") or {}).items()}
            except Exception:
                pass
    return out


def compare() -> None:
    v1 = load_v1()
    narr = {r["Crash_ID"]: r["narrative"] for r in load_spike()}
    v11 = {}
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            v11[int(r["Crash_ID"])] = r.get("answers") or {}

    ids = [i for i in v11 if i in v1]
    def noul(d, i, q):
        a = (d[i].get(q) or {})
        return a.get("noul")

    a1 = np.array([noul(v1, i, "aggression") for i in ids], dtype=float)
    a2 = np.array([noul(v11, i, "aggression") for i in ids], dtype=float)
    pur = np.array([bool(PURSUIT.search(narr.get(i, ""))) for i in ids])

    rep: dict = {"n_compared": len(ids), "model_v1_1": None}
    rep["aggression"] = {
        "pursuit_narratives_n": int(pur.sum()),
        "pursuit_pos_v1": int((a1[pur] > 0.5).sum()),
        "pursuit_pos_v1_1": int((a2[pur] > 0.5).sum()),
        "pursuit_mean_p_v1": round(float(a1[pur].mean()), 4) if pur.any() else None,
        "pursuit_mean_p_v1_1": round(float(a2[pur].mean()), 4) if pur.any() else None,
        "nonpursuit_pos_v1": int((a1[~pur] > 0.5).sum()),
        "nonpursuit_pos_v1_1": int((a2[~pur] > 0.5).sum()),
        "nonpursuit_mean_p_v1": round(float(a1[~pur].mean()), 4),
        "nonpursuit_mean_p_v1_1": round(float(a2[~pur].mean()), 4),
    }

    # every other Noul should be essentially unchanged (only aggression's criteria moved)
    stab = {}
    for q in ("preg_mentioned", "medical_episode", "drug_involved", "alcohol_involved",
              "hydroplane", "vehicle_defect", "fatigue", "intentional", "hit_and_run",
              "animal_involved", "wrong_way", "phone_use", "unbelted", "transported"):
        x = np.array([noul(v1, i, q) for i in ids], dtype=float)
        z = np.array([noul(v11, i, q) for i in ids], dtype=float)
        ok = np.isfinite(x) & np.isfinite(z)
        if ok.sum():
            stab[q] = {"label_agreement": round(float(((x[ok] > .5) == (z[ok] > .5)).mean()), 4),
                       "mean_abs_shift": round(float(np.abs(x[ok] - z[ok]).mean()), 4)}
    rep["other_nouls_stability"] = stab

    # gate leakage (§4.1 lambda) under v1.1
    gates = json.loads(SCHEMA.read_text())["gates"]
    nomatch = {"medical_type": "none", "drug_class": "none", "preg_role": "no_pregnancy",
               "preg_outcome": "no_pregnancy", "preg_stage": "no_pregnancy",
               "animal_type": "none", "defect_type": "none"}
    leak = {}
    for c, g in gates.items():
        off = [i for i in ids if (noul(v11, i, g) or 0) <= 0.5]
        if off:
            bad = sum(1 for i in off if (v11[i].get(c) or {}).get("choice") != nomatch[c])
            lo, hi = M.clopper_pearson(bad, len(off))
            leak[c] = {"gate": g, "n_gate_off": len(off), "n_leaked": bad,
                       "leakage_rate": round(bad / len(off), 5),
                       "ci95": [round(lo, 5), round(hi, 5)]}
    rep["gate_leakage_v1_1"] = leak

    (DATA / "schema_validate_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if not a.compare_only:
        schema = json.loads(SCHEMA.read_text())
        qs = {k: v for k, v in schema["questions"].items() if k != "pii_residual"}
        questions = J.build_questions({"questions": qs})
        items = load_spike()[:a.limit] if a.limit else load_spike()
        print(f"validate: {len(items):,} narratives, {len(questions)} questions", flush=True)
        await J.run(items, questions, OUT, cost_cap_usd=COST_CAP, label="schema_validate",
                    meta_fn=lambda r: {"stratum": r["stratum"]})
    compare()


if __name__ == "__main__":
    asyncio.run(main())
