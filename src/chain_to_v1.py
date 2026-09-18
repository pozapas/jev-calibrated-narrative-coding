"""Unattended chain: wait for Stage 1, run Stage 2, then build every v1 artefact.

Started once and left alone. Each step's output is appended to logs/chain.log, and the
script stops at the first failure rather than carrying a broken intermediate forward.

    python src/chain_to_v1.py
"""
from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"D:/OneDrive - Texas State University/AIT/Papers/Jev")
SRC = ROOT / "paper1" / "src"
LOGS = ROOT / "paper1" / "logs"
DATA = ROOT / "paper1" / "data"
PY = r"C:/Users/nib37/AppData/Local/Programs/Python/Python313/python.exe"

STAGE1_LOG = LOGS / "s02_stage1.log"
DONE_MARKERS = ("[stage1] DONE",)
FAIL_MARKERS = ("Traceback", "ABORT:")


def say(msg: str) -> None:
    print(f"[chain {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_for_stage1() -> None:
    say("waiting for Stage 1 to finish...")
    while True:
        txt = STAGE1_LOG.read_text(encoding="utf-8", errors="replace") if STAGE1_LOG.exists() else ""
        if any(m in txt for m in DONE_MARKERS):
            say("Stage 1 reported DONE")
            return
        if any(m in txt for m in FAIL_MARKERS):
            raise SystemExit("Stage 1 failed -- see s02_stage1.log")
        time.sleep(30)


def step(cmd: list[str], label: str, log: str) -> None:
    say(f"START {label}")
    t = time.time()
    with open(LOGS / log, "a", encoding="utf-8") as fh:
        fh.write(f"\n===== {label} @ {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        fh.flush()
        r = subprocess.run(cmd, cwd=SRC, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {label} (exit {r.returncode}) -- see logs/{log}")
    say(f"OK    {label} in {time.time()-t:.0f}s")


def main() -> None:
    wait_for_stage1()

    # Stage 2 needs the completed Stage-1 positives, so the frame is always rebuilt here.
    step([PY, "s03_full.py", "--rebuild-frame"], "Stage 2 (27 questions)", "s03_stage2.log")

    # run_v1 does flatten -> gold frame -> analysis -> tables -> figures -> checklist -> spend
    step([PY, "run_v1.py"], "Build all v1 artefacts", "run_v1.log")

    say("CHAIN COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        say(f"CHAIN FAILED: {e}")
        sys.exit(1)
