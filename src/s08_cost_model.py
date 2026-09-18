"""Cost and latency model (§4.3, F2d, T3).

§4.3 asks for input tokens = a + b*len(n_i) per question set, with a and b reported. The
spike gave a single point (27 questions). This sweeps the question count so both the
per-narrative slope b and the way the intercept a grows with schema size are measured
rather than assumed -- which matters, because the pre-flight measurements showed cost is
dominated by a (the schema), not b*len (the narrative).

Runs `--n` narratives at each question count and fits:
    within a question set:  tokens = a_k + b_k * nchar
    across question sets:   a_k    = alpha + beta * k

Budget: ~$0.04 at the default 60 narratives x 7 question counts, from the $0.90
schema-iteration line.
"""
from __future__ import annotations
import argparse, asyncio, csv, json
from pathlib import Path

import numpy as np

import jev_runner as J

ROOT = Path(r"D:/OneDrive - Texas State University/AIT/Papers/Jev")
DATA = ROOT / "paper1" / "data"
SCHEMA = ROOT / "schemas" / "crash_factors_v1_1.json"
OUT = DATA / "cost_model.jsonl"
COUNTS = [1, 3, 6, 10, 15, 21, 27]


def sample(n: int) -> list[dict]:
    csv.field_size_limit(1 << 24)
    with open(ROOT / "spike" / "spike_sample_2000.csv", newline="", encoding="utf-8") as f:
        rows = [{"Crash_ID": int(r["Crash_ID"]), "narrative": r["Investigator_Narrative"]}
                for r in csv.DictReader(f)]
    # spread across the length distribution so b is identified, not just a
    rows.sort(key=lambda r: len(r["narrative"]))
    idx = np.linspace(0, len(rows) - 1, n).astype(int)
    return [rows[i] for i in idx]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--fit-only", action="store_true")
    a = ap.parse_args()

    schema = json.loads(SCHEMA.read_text())
    qids = [q for q in schema["questions"] if q != "pii_residual"]
    items = sample(a.n)

    if not a.fit_only:
        if OUT.exists():
            OUT.unlink()
        for k in COUNTS:
            qs = J.build_questions({"questions": {q: schema["questions"][q] for q in qids[:k]}})
            tagged = [dict(r, Crash_ID=f"{k}_{r['Crash_ID']}") for r in items]
            await J.run(tagged, qs, OUT, cost_cap_usd=0.20, label=f"nq={k}",
                        meta_fn=lambda r: {"n_questions": k, "nchar": len(r["narrative"])},
                        project_after=10_000)

    # ---- fit -------------------------------------------------------------------------
    recs = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    fits, rows = {}, []
    for k in COUNTS:
        sub = [r for r in recs if r.get("n_questions") == k]
        if len(sub) < 5:
            continue
        x = np.array([r["nchar"] for r in sub], float)
        y = np.array([(r["usage"] or {}).get("input_tokens") for r in sub], float)
        ok = np.isfinite(y)
        x, y = x[ok], y[ok]
        b, a_ = np.polyfit(x, y, 1)
        r2 = 1 - ((y - (a_ + b * x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        lat = np.array([r["latency_s"] for r in sub], float)
        fits[k] = {"n": int(x.size), "a_intercept_tokens": round(float(a_), 1),
                   "b_tokens_per_char": round(float(b), 4), "r2": round(float(r2), 4),
                   "mean_tokens": round(float(y.mean()), 1),
                   "p50_latency_s": round(float(np.median(lat)), 3),
                   "cost_per_1k_narratives_usd": round(float(y.mean()) * 1000 * J.PRICE_PER_M_INPUT / 1e6, 4)}
        rows.append((k, float(a_), float(b), float(np.median(lat))))

    ks = np.array([r[0] for r in rows], float)
    as_ = np.array([r[1] for r in rows], float)
    beta, alpha = np.polyfit(ks, as_, 1)
    r2a = 1 - ((as_ - (alpha + beta * ks)) ** 2).sum() / ((as_ - as_.mean()) ** 2).sum()

    total_tokens = sum((r["usage"] or {}).get("input_tokens") or 0 for r in recs)
    out = {
        "per_question_set": fits,
        "intercept_model": {"alpha_tokens": round(float(alpha), 1),
                            "beta_tokens_per_question": round(float(beta), 1),
                            "r2": round(float(r2a), 4)},
        "interpretation": (
            f"Schema overhead grows at ~{beta:.0f} tokens per question and dominates: a "
            f"27-question call spends ~{alpha + beta*27:.0f} tokens on the schema against "
            f"~{np.mean([r[2] for r in rows])*350:.0f} tokens for a median 350-character "
            f"narrative. Cost therefore scales with schema size, not narrative length, which "
            f"inverts the usual assumption that a lean screen is cheap per narrative."),
        "spend_usd": round(total_tokens * J.PRICE_PER_M_INPUT / 1e6, 4),
        "n_calls": len(recs),
    }
    (DATA / "cost_model.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
