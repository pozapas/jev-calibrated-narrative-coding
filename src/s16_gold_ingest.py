"""Ingest rater label files from the labelling app into the analysis.

    python src/s16_gold_ingest.py ~/Downloads/gold-labels-*.json

The app runs a PANEL: every narrative is rated independently by `replication` raters, and a
shared calibration set is rated by everyone. So this script:

  1. merges every rater's file, dropping duplicate answers within a rater;
  2. measures inter-rater reliability on the CALIBRATION set, where all raters overlap --
     pairwise Cohen's kappa plus a per-rater agreement-with-majority score, which is what
     identifies a rater who is drifting;
  3. resolves each (narrative, variable) pair by majority vote, marking ties and
     no-majority cases DISPUTED rather than inventing agreement;
  4. joins to the Stage-2 model output by Crash_ID and writes gold_labels.csv with the
     Horvitz-Thompson weights, ready for the §4.5 metrics.

The join happens HERE, never in the app: the app is blind to model output by design, so a
rater cannot be anchored by Jev's probabilities.
"""
from __future__ import annotations
import argparse, glob, json
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd

import s07_metrics as M

ROOT = Path(r"D:/OneDrive - Texas State University/AIT/Papers/Jev")
DATA = ROOT / "paper1" / "data"
GOLD = DATA / "gold"
APP = ROOT / "tool-gold" / "data"
TASKS = APP / "gold_tasks.json"
ROSTER = APP / "roster.json"


def resolve(vals: list[str]) -> tuple[str, int, float]:
    """Majority label, number of raters, and the majority's share.

    A tie, or a plurality that is not a majority, is DISPUTED. A reference set that
    manufactures agreement is worse than one that says where it has none.
    """
    n = len(vals)
    if n == 0:
        return "NONE", 0, float("nan")
    cnt = Counter(vals).most_common()
    top, k = cnt[0]
    tie = len(cnt) > 1 and cnt[1][1] == k
    label = "DISPUTED" if (tie or k * 2 <= n) else top
    return label, n, k / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="rater JSON exports (globs allowed)")
    a = ap.parse_args()

    paths = sorted({p for pat in a.files for p in glob.glob(pat)})
    if not paths:
        raise SystemExit(f"no files matched {a.files}")

    tasks = json.loads(TASKS.read_text())
    roster = json.loads(ROSTER.read_text())
    cal_n = int(roster.get("calibration_n", 0))
    calibration_ids = {t["id"] for t in tasks["tasks"][:cal_n]}
    valid = {t["id"]: set(t["vars"]) for t in tasks["tasks"]}

    rows = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        who = d.get("coder", "?")
        if d.get("format") != "jev-gold-labels/v1":
            print(f"  SKIP {Path(p).name}: unexpected format {d.get('format')!r}")
            continue
        if d.get("dataset_generated") != tasks["generated"]:
            print(f"  WARNING {Path(p).name}: labelled against a different dataset build")
        seen = set()
        for lab in d["labels"]:
            key = (lab["taskId"], lab["variable"])
            if key in seen:
                continue
            seen.add(key)
            if lab["variable"] not in valid.get(lab["taskId"], ()):
                print(f"  WARNING {who}: answer for a pair not in the frame {key}")
                continue
            rows.append({"rater": who, "Crash_ID": lab["taskId"],
                         "variable": lab["variable"], "verdict": lab["verdict"],
                         "ms": lab.get("ms")})
        print(f"  {Path(p).name}: {who}, {len(seen)} answers")

    lab = pd.DataFrame(rows)
    if lab.empty:
        raise SystemExit("no usable labels")
    print(f"\n{len(lab):,} answers from {lab.rater.nunique()} raters over "
          f"{lab.Crash_ID.nunique()} narratives")

    wide = lab.pivot_table(index=["Crash_ID", "variable"], columns="rater",
                           values="verdict", aggfunc="first")

    # ---------------------------------------------------------------- reliability
    cal = wide[wide.index.get_level_values("Crash_ID").isin(calibration_ids)]
    kappas: dict[str, float] = {}
    per_rater: dict[str, dict] = {}
    if len(cal) and cal.shape[1] >= 2:
        for x, y in combinations(list(cal.columns), 2):
            pair = cal[[x, y]].dropna()
            if len(pair) >= 20:
                kappas[f"{x}|{y}"] = M.cohen_kappa(pair[x].to_numpy(), pair[y].to_numpy())
        maj = cal.apply(lambda r: resolve([v for v in r if isinstance(v, str)])[0], axis=1)
        for r in cal.columns:
            m = cal[r].notna() & maj.isin(["yes", "no", "unclear"])
            if m.sum() >= 10:
                per_rater[r] = {"n": int(m.sum()),
                                "agree_with_majority": float((cal[r][m] == maj[m]).mean()),
                                "median_ms": float(lab[lab.rater == r].ms.median())}
        print(f"\ncalibration set: {len(cal)} pairs, {cal.shape[1]} raters")
        if kappas:
            ks = pd.Series(kappas)
            print(f"  pairwise kappa: median {ks.median():.3f}  "
                  f"range {ks.min():.3f}-{ks.max():.3f}  ({len(ks)} pairs)")
        # Flag relative to the panel rather than against a fixed cut: what counts as poor
        # agreement depends on how hard the items are, and a constant threshold either
        # spares a drifting rater on easy items or condemns the whole panel on hard ones.
        # An absolute floor still applies, because a rater below it is not usable whatever
        # the rest of the panel does.
        agr = pd.Series({r: v["agree_with_majority"] for r, v in per_rater.items()})
        mad = float((agr - agr.median()).abs().median()) or 0.01
        cut = max(0.80, float(agr.median()) - 3 * mad)
        print(f"  review threshold {cut:.2f} "
              f"(panel median {agr.median():.2f}, MAD {mad:.3f}, floor 0.80)")
        for r, v in sorted(per_rater.items(), key=lambda kv: kv[1]["agree_with_majority"]):
            flag = "  <-- REVIEW" if v["agree_with_majority"] < cut else ""
            print(f"  {r:8s} agrees with majority {v['agree_with_majority']:.2f} "
                  f"on {v['n']:3d}  median {v['median_ms']/1000:5.1f}s/answer{flag}")
    else:
        print("\nno calibration overlap yet -- raters have not reached the shared set")

    # ---------------------------------------------------------------- resolve
    res = wide.apply(lambda r: pd.Series(
        resolve([v for v in r if isinstance(v, str)]),
        index=["label", "n_raters", "agreement"]), axis=1).reset_index()
    res["n_raters"] = res["n_raters"].astype(int)
    n_disp = int((res.label == "DISPUTED").sum())
    unan = int((res.agreement == 1.0).sum())
    print(f"\nresolved {len(res):,} pairs; {unan:,} unanimous, {n_disp} disputed")
    print("  ratings per pair:", res.n_raters.value_counts().sort_index().to_dict())

    # ---------------------------------------------------------------- join model output
    flat = pd.read_parquet(DATA / "stage2" / "stage2_flat.parquet")
    fp = GOLD / ("gold_frame_oversampled.csv"
                 if (GOLD / "gold_frame_oversampled.csv").exists() else "gold_frame.csv")
    frame = pd.read_csv(fp)
    out = res.merge(frame[["Crash_ID", "incl_prob", "ht_weight"]], on="Crash_ID", how="left")
    pcols = [f"p_{v}" for v in out.variable.unique() if f"p_{v}" in flat.columns]
    out = out.merge(flat[["Crash_ID"] + pcols], on="Crash_ID", how="left")
    out["p"] = [row.get(f"p_{row['variable']}") for _, row in out.iterrows()]
    out["y"] = out.label.map({"yes": 1.0, "no": 0.0})   # unclear / disputed stay NaN
    out = out[["Crash_ID", "variable", "label", "n_raters", "agreement",
               "p", "incl_prob", "ht_weight", "y"]]

    GOLD.mkdir(parents=True, exist_ok=True)
    out.to_csv(GOLD / "gold_labels.csv", index=False)
    (GOLD / "gold_reliability.json").write_text(json.dumps({
        "n_answers": int(len(lab)), "n_raters": int(lab.rater.nunique()),
        "n_pairs": int(len(res)), "n_unanimous": unan, "n_disputed": n_disp,
        "mean_agreement": float(res.agreement.mean()),
        "ratings_per_pair": {str(k): int(v)
                             for k, v in res.n_raters.value_counts().sort_index().items()},
        "n_usable_yes_no": int(out.y.notna().sum()),
        "pairwise_kappa_calibration": kappas,
        "per_rater_calibration": per_rater,
        "verdict_mix": lab.verdict.value_counts().to_dict(),
    }, indent=1, default=float))
    print(f"\nwrote {GOLD/'gold_labels.csv'}  ({int(out.y.notna().sum()):,} usable yes/no)")
    print(f"      {GOLD/'gold_reliability.json'}")
    print("\nNext: point the §4.5 metrics at gold_labels.csv (y, p, ht_weight) for "
          "calibration against HUMAN labels rather than coded fields.")


if __name__ == "__main__":
    main()
