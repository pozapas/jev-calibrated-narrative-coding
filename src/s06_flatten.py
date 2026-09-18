"""Flatten a Jev JSONL run into one row per narrative, applying the §4.1 gating rule.

Gating (§4.1): for a dependent variable v with gate g(v), the answer is interpreted only if
p_{i,g(v)} > tau_g (default 0.5); otherwise it is set to the no-match option. The RAW answer
is kept alongside the gated one so the leakage rate

    lambda_v = P(choice != no-match | p_gate <= tau)

can be reported as the empirical justification for the rule, exactly as §4.1 requires.

Usage:
  python s06_flatten.py --run stage2   # -> data/stage2/stage2_flat.parquet
  python s06_flatten.py --run stage1   # -> data/stage1/stage1_flat.parquet
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:/OneDrive - Texas State University/AIT/Papers/Jev")
DATA = ROOT / "paper1" / "data"
TAU_GATE = 0.5

NO_MATCH = {"medical_type": "none", "drug_class": "none", "preg_role": "no_pregnancy",
            "preg_outcome": "no_pregnancy", "preg_stage": "no_pregnancy",
            "animal_type": "none", "defect_type": "none", "surface_narr": "not_stated"}


def flatten(run: str, schema_path: Path) -> pd.DataFrame:
    schema = json.loads(schema_path.read_text())
    gates: dict[str, str] = schema.get("gates", {})
    # pii_residual ships in the same schema file but is never asked during Stage 1 or 2 (it
    # is the display screen, run separately). Including it here would add an all-null
    # p_pii_residual column, which noul_ids() in the analysis and the gold-set sampler would
    # then treat as a real variable.
    qtype = {k: v["type"] for k, v in schema["questions"].items() if k != "pii_residual"}

    src = DATA / run / f"{run}.jsonl"
    rows, n_bad = [], 0
    with open(src, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                n_bad += 1
                continue
            out = {"Crash_ID": r.get("Crash_ID"), "Year": r.get("Year"),
                   "Crash_Sev_ID": r.get("Crash_Sev_ID"),
                   "stratum": r.get("stratum", "random"),
                   "incl_prob": r.get("incl_prob"), "enrich_class": r.get("enrich_class"),
                   "model": r.get("model"), "latency_s": r.get("latency_s"),
                   "input_tokens": (r.get("usage") or {}).get("input_tokens"),
                   "n_drug_tokens": len(r.get("drug_tokens_found") or [])}
            ans = r.get("answers") or {}
            for qid, t in qtype.items():
                a = ans.get(qid) or {}
                if t == "noul":
                    out[f"p_{qid}"] = a.get("noul")
                elif t == "choice":
                    out[f"raw_{qid}"] = a.get("choice")
                    out[f"conf_{qid}"] = a.get("confidence")
                elif t == "score":
                    out[f"score_{qid}"] = a.get("score")
                    out[f"conf_{qid}"] = a.get("confidence")
            rows.append(out)
    d = pd.DataFrame(rows)
    if n_bad:
        print(f"skipped {n_bad} unparseable lines")
    print(f"{run}: {len(d):,} narratives")

    # ---- apply gating, record leakage ------------------------------------------------
    leak = {}
    for dep, g in gates.items():
        if f"raw_{dep}" not in d.columns or f"p_{g}" not in d.columns:
            continue
        nm = NO_MATCH[dep]
        gate_on = d[f"p_{g}"].fillna(0) > TAU_GATE
        raw = d[f"raw_{dep}"]
        d[dep] = np.where(gate_on, raw, nm)
        off = ~gate_on
        n_off = int(off.sum())
        n_leak = int(((raw != nm) & off).sum())
        leak[dep] = {"gate": g, "tau": TAU_GATE, "n_gate_off": n_off, "n_leaked": n_leak,
                     "leakage_rate": round(n_leak / n_off, 6) if n_off else None,
                     "n_overridden_to_no_match": n_leak}
    # ungated choices pass through
    for qid, t in qtype.items():
        if t == "choice" and qid not in gates and f"raw_{qid}" in d.columns:
            d[qid] = d[f"raw_{qid}"]

    outdir = DATA / run
    d.to_parquet(outdir / f"{run}_flat.parquet", index=False)
    (outdir / f"{run}_leakage.json").write_text(json.dumps(leak, indent=1))
    print(f"wrote {outdir/f'{run}_flat.parquet'}")
    if leak:
        print("gate leakage (lambda_v, §4.1):")
        for k, v in leak.items():
            print(f"  {k:14s} {100*(v['leakage_rate'] or 0):6.3f}%  "
                  f"({v['n_leaked']:,} of {v['n_gate_off']:,} gate-off)")
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=["stage1", "stage2"])
    a = ap.parse_args()
    sch = ROOT / "schemas" / ("crash_factors_v1_1.json" if a.run == "stage2"
                              else "crash_factors_lean_v1_1.json")
    flatten(a.run, sch)
