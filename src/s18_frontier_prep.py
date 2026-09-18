"""Prepare the frontier-model baseline arm (§4.6) from the gold task file.

The comparison is only meaningful if the frontier model is asked *exactly* what Jev and the
human coders were asked. So the batches built here carry:

  - the same 400 narratives, already PII-screened, that the coders saw;
  - the same criteria text, verbatim, including the `true`/`false` boundary cases;
  - the same per-narrative variable assignment, so the judgment set is identical.

What deliberately differs is the answer format. Jev returns a probability on a fixed
two-decimal grid as part of its interface; a generative model has to be *asked* for one, and
whether that elicited number is calibrated is precisely what this arm tests. We therefore
impose no grid and no rounding, and we do not tell the model what the base rates are --
supplying them would let it anchor on prevalence rather than read the narrative, which is not
what Jev was given either.

Outputs batches under data/frontier/batches/ plus a single self-contained task package that
can be handed to any other model's harness without this repository.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

ROOT = Path(r"D:/OneDrive - Texas State University/AIT/Papers/Jev")
TASKS = ROOT / "tool-gold" / "data" / "gold_tasks.json"
OUT = ROOT / "paper1" / "data" / "frontier"
BATCH_SIZE = 20

INSTRUCTIONS = """\
You are coding police crash narratives for a transportation-safety study.

For each narrative you are given, answer ONLY the questions listed for that narrative. Each
question asks whether a factor is present in the narrative, and comes with the criteria that
define a `true` and a `false` answer. Apply those criteria exactly as written; they are the
same criteria given to the human coders this work is scored against.

Rules:
  - Judge ONLY what the narrative states or clearly implies. Do not infer from what is
    typical of crashes in general.
  - `answer` is true or false.
  - `p` is your probability that the correct answer is true, as a number between 0 and 1.
    Use the full range. Do not round to convenient values; 0.12 and 0.87 are as legitimate
    as 0.1 and 0.9. This number is scored for calibration, not just for its side of 0.5.
  - If the narrative genuinely does not say, that is a `false` with a low `p`, not a refusal.

Return ONE JSON object and nothing else, of the form:

{"results": [{"id": "<narrative id>", "answers": {"<variable>": {"answer": true, "p": 0.93}}}]}

Every narrative id you were given must appear exactly once, and every variable listed for
that narrative must appear in its `answers` object.
"""


def main() -> None:
    d = json.loads(TASKS.read_text(encoding="utf-8"))
    tasks, criteria = d["tasks"], d["criteria"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "batches").mkdir(exist_ok=True)

    n_batches = math.ceil(len(tasks) / BATCH_SIZE)
    for b in range(n_batches):
        chunk = tasks[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        used = sorted({v for t in chunk for v in t["vars"]})
        payload = {
            "batch": b,
            "instructions": INSTRUCTIONS,
            "criteria": {v: criteria[v] for v in used},
            "narratives": [{"id": str(t["id"]), "text": t["text"], "questions": t["vars"]}
                           for t in chunk],
        }
        (OUT / "batches" / f"batch_{b:02d}.json").write_text(
            json.dumps(payload, indent=1), encoding="utf-8")

    # One self-contained package, for running this arm in another vendor's harness. It has to
    # stand alone: no repository paths, no reliance on anything above.
    pkg = {
        "purpose": ("Frontier-LLM baseline arm for a crash-narrative coding study. Answer "
                    "every question for every narrative and return the JSON described in "
                    "`instructions`."),
        "instructions": INSTRUCTIONS,
        "output_file": "frontier_<modelname>.json",
        "n_narratives": len(tasks),
        "n_judgments": sum(len(t["vars"]) for t in tasks),
        "criteria": criteria,
        "narratives": [{"id": str(t["id"]), "text": t["text"], "questions": t["vars"]}
                       for t in tasks],
    }
    (OUT / "frontier_task_package.json").write_text(json.dumps(pkg, indent=1),
                                                    encoding="utf-8")

    print(f"{len(tasks)} narratives, {pkg['n_judgments']} judgments")
    print(f"wrote {n_batches} batches of <= {BATCH_SIZE} to {OUT / 'batches'}")
    print(f"wrote standalone package to {OUT / 'frontier_task_package.json'}")


if __name__ == "__main__":
    main()
