"""Shared async Jev runner: concurrency 10, backoff, append-only JSONL with resume,
and a HARD SPEND TRIPWIRE enforced in code.

§6 says "stop any run whose projected spend exceeds its line above". On a 500k-narrative
run that takes ~3.5 h unattended, that has to be code, not vigilance. Two controls:

  1. Hard cap: realised spend (sum of usage.input_tokens x PRICE) is checked after every
     completed call. Once it crosses `cost_cap_usd`, no new work is dispatched and the run
     exits cleanly. Already-written JSONL is valid and resumable.
  2. Early projection: after `project_after` completed calls, the realised mean tokens per
     narrative is extrapolated to the full frame. If the projection exceeds the cap, the run
     aborts immediately rather than burning the budget to find out.

Every record carries `model` (so a silent vendor version change is detectable per §6 risks),
`usage`, and `latency_s`.
"""
from __future__ import annotations
import asyncio, json, os, statistics, time
from pathlib import Path
from typing import Any, Callable, Iterable

from dotenv import load_dotenv

ROOT = Path(r"D:/OneDrive - Texas State University/AIT/Papers/Jev")
load_dotenv(ROOT / ".env")

from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score  # noqa: E402

PRICE_PER_M_INPUT = 0.042   # USD per million input tokens, TypeSafe pricing, accessed 2026-09-17
MODEL = "jev-1.13.0"
CONCURRENCY = 10            # §6: 50 req/s at 16 produced 429s; 10 is the documented safe setting
MAX_TRIES = 5


class SpendCapExceeded(Exception):
    pass


def build_questions(schema: dict) -> dict[str, Any]:
    """Turn a schema JSON's `questions` block into SDK objects."""
    out: dict[str, Any] = {}
    for qid, q in schema["questions"].items():
        t = q["type"]
        if t == "noul":
            crit = q.get("criteria")
            out[qid] = Noul(instructions=q["instructions"], criteria=crit) if crit else Noul(instructions=q["instructions"])
        elif t == "choice":
            out[qid] = Choice(instructions=q["instructions"], criteria=q["criteria"])
        elif t == "score":
            out[qid] = Score(instructions=q["instructions"], criteria=q["criteria"])
        else:
            raise ValueError(f"unknown question type {t!r} for {qid}")
    return out


def serialize(resp) -> dict:
    out: dict[str, Any] = {"answers": {}}
    answers = getattr(resp, "answers", None) or {}
    for k, a in answers.items():
        out["answers"][k] = {x: getattr(a, x) for x in
                             ("type", "noul", "choice", "probabilities", "confidence", "score", "legend")
                             if hasattr(a, x)}
    u = getattr(resp, "usage", None)
    out["usage"] = {"input_tokens": getattr(u, "input_tokens", None),
                    "output_tokens": getattr(u, "output_tokens", None)} if u else None
    out["model"] = getattr(resp, "model", None)
    return out


def load_done(path: Path, key: str = "Crash_ID") -> set:
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)[key])
                except Exception:
                    pass
    return done


async def run(
    items: list[dict],
    questions: dict,
    out_path: Path,
    *,
    cost_cap_usd: float,
    state_fn: Callable[[dict], dict] = lambda r: {"narrative": r["narrative"]},
    meta_fn: Callable[[dict], dict] = lambda r: {},
    key: str = "Crash_ID",
    concurrency: int = CONCURRENCY,
    project_after: int = 2_000,
    label: str = "run",
) -> dict:
    """Run `questions` over `items`, appending to `out_path`. Returns a spend report."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path, key)
    todo = [r for r in items if r[key] not in done]
    n_frame = len(items)

    # Spend already banked in a previous (resumed) run counts against the cap.
    banked_tokens = 0
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    u = json.loads(line).get("usage") or {}
                    banked_tokens += u.get("input_tokens") or 0
                except Exception:
                    pass

    print(f"[{label}] frame={n_frame:,} done={len(done):,} todo={len(todo):,} "
          f"banked=${banked_tokens * PRICE_PER_M_INPUT / 1e6:.2f} cap=${cost_cap_usd:.2f} "
          f"concurrency={concurrency}", flush=True)
    if not todo:
        return {"label": label, "n_new": 0, "tokens": banked_tokens,
                "cost_usd": banked_tokens * PRICE_PER_M_INPUT / 1e6, "stopped_early": False}

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    state = {"tokens": banked_tokens, "n": 0, "lat": [], "tok": [], "errs": [],
             "stop": False, "models": {}}
    t0 = time.time()
    fh = open(out_path, "a", encoding="utf-8")

    async def one(r: dict) -> None:
        if state["stop"]:
            return
        async with sem:
            if state["stop"]:
                return
            st = time.time()
            for k in range(MAX_TRIES):
                try:
                    resp = await client.system_one(state=state_fn(r), questions=questions)
                    break
                except Exception as e:
                    msg = repr(e)
                    retriable = any(c in msg for c in ("429", "529", "502", "503", "504", "Timeout", "timeout"))
                    if retriable and k < MAX_TRIES - 1:
                        await asyncio.sleep(2 ** k)
                        continue
                    async with lock:
                        state["errs"].append((r.get(key), msg[:300]))
                        if len(state["errs"]) <= 5:
                            print(f"[{label}] ERR {r.get(key)} {msg[:220]}", flush=True)
                    return
            rec = serialize(resp)
            rec[key] = r[key]
            rec.update(meta_fn(r))
            rec["latency_s"] = round(time.time() - st, 3)
            tok = (rec.get("usage") or {}).get("input_tokens") or 0
            async with lock:
                fh.write(json.dumps(rec) + "\n")
                state["tokens"] += tok
                state["n"] += 1
                state["lat"].append(rec["latency_s"])
                if tok:
                    state["tok"].append(tok)
                state["models"][rec.get("model")] = state["models"].get(rec.get("model"), 0) + 1
                cost = state["tokens"] * PRICE_PER_M_INPUT / 1e6

                if state["n"] == project_after and state["tok"]:
                    mean_tok = statistics.mean(state["tok"])
                    proj = (banked_tokens + mean_tok * len(todo)) * PRICE_PER_M_INPUT / 1e6
                    print(f"[{label}] PROJECTION after {state['n']}: mean {mean_tok:.0f} tok/item "
                          f"-> ${proj:.2f} for the frame (cap ${cost_cap_usd:.2f})", flush=True)
                    if proj > cost_cap_usd:
                        state["stop"] = True
                        print(f"[{label}] ABORT: projected ${proj:.2f} exceeds cap ${cost_cap_usd:.2f}", flush=True)

                if cost >= cost_cap_usd and not state["stop"]:
                    state["stop"] = True
                    print(f"[{label}] TRIPWIRE: realised ${cost:.2f} reached cap ${cost_cap_usd:.2f}; "
                          f"stopping after {state['n']:,} new calls", flush=True)

                if state["n"] % 5_000 == 0:
                    fh.flush()
                    el = time.time() - t0
                    print(f"[{label}] {state['n']:,}/{len(todo):,}  ${cost:.2f}  "
                          f"{state['n']/max(el,1):.1f} req/s  p50 lat {statistics.median(state['lat']):.2f}s  "
                          f"mean tok {statistics.mean(state['tok']):.0f}", flush=True)

    try:
        async with AsyncTypeSafeClient(timeout=60) as client:
            await asyncio.gather(*(one(r) for r in todo))
    finally:
        fh.close()

    el = time.time() - t0
    cost = state["tokens"] * PRICE_PER_M_INPUT / 1e6
    report = {
        "label": label, "frame": n_frame, "n_new": state["n"], "n_errors": len(state["errs"]),
        "tokens_total": state["tokens"], "cost_usd": round(cost, 4),
        "cost_cap_usd": cost_cap_usd, "stopped_early": state["stop"],
        "mean_input_tokens": round(statistics.mean(state["tok"]), 1) if state["tok"] else None,
        "p50_latency_s": round(statistics.median(state["lat"]), 3) if state["lat"] else None,
        "p90_latency_s": round(sorted(state["lat"])[int(0.9 * len(state["lat"])) - 1], 3) if state["lat"] else None,
        "req_per_s": round(state["n"] / max(el, 1), 2), "wall_s": round(el, 1),
        "models_seen": state["models"],
    }
    print(f"[{label}] DONE {json.dumps(report)}", flush=True)
    (out_path.parent / f"{out_path.stem}_report.json").write_text(json.dumps(report, indent=1))
    if state["errs"]:
        (out_path.parent / f"{out_path.stem}_errors.json").write_text(json.dumps(state["errs"][:2000], indent=1))
    return report
