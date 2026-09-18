"""Apply the §4.2 schema design-rule fixes to crash_factors_v1 -> crash_factors_v1_1.

Two fixes, each motivated by the spike error analysis (recorded in schema_diff.json so the
manuscript can cite the before/after):

  Rule 6 (exclusions written into criteria after error analysis) -- `aggression`.
    Spike evidence: of 16 spike narratives containing pursuit language, 4 scored p>0.5 for
    aggression, two of them at 0.93 and 0.95. A police pursuit is an enforcement action, not
    aggression directed at another road user, and hit-and-run is coded separately by
    `hit_and_run`. Both are now named as boundary cases in the `false` criteria.

  Rule 3 (select, do not generate) -- `drug_class`.
    `drug_class` is already a Choice over a closed category set, so it never asks the model
    to generate a string; the "pre-parsed value extraction" requirement is therefore met at
    the schema level. What was missing is the regex candidate pass that documents WHICH drug
    tokens the narrative actually contains, so that an `unspecified_drug` answer can be
    distinguished from a missed named drug. That pass is emitted here as DRUG_TOKEN_RE and
    applied in s03_full.py as a recorded covariate, NOT as a per-narrative option set --
    varying the option set per narrative would make the confidence values incomparable
    across narratives and break the calibration analysis in §4.5.

Also adds the `pii_residual` Noul required by §6 Step 1 to the Stage-1 lean schema.
"""
from __future__ import annotations
import json, re
from pathlib import Path

ROOT = Path(r"D:/OneDrive - Texas State University/AIT/Papers/Jev")
SCHEMAS = ROOT / "schemas"
DATA = ROOT / "paper1" / "data"

# Regex candidate finder for §4.2 rule 3 (recorded, not injected into the option set).
DRUG_TOKEN_RE = re.compile(
    r"\b(marijuana|cannabis|thc|weed|methamphetamine|meth|cocaine|crack|amphetamine|"
    r"heroin|fentanyl|oxycodone|hydrocodone|opioid|opiate|morphine|xanax|alprazolam|"
    r"benzodiazepine|valium|klonopin|ambien|soma|muscle relaxer|adderall|ecstasy|mdma|"
    r"pcp|lsd|narcotic|controlled substance|prescription medication)\b", re.IGNORECASE)

AGGRESSION_FALSE = (
    "No aggression or confrontation is described. Ordinary traffic violations without "
    "hostility do not count. A police pursuit, a traffic stop, an officer chasing or "
    "stopping a vehicle, and a driver evading or fleeing from police do not count. "
    "A driver leaving the scene of the crash does not count by itself."
)

PII_RESIDUAL = {
    "type": "noul",
    "instructions": ("The `narrative` contains a person's full name, date of birth, driver "
                     "license or ID number, home address, or phone number that is not already "
                     "replaced by a placeholder."),
    "criteria": {
        "true": ("An actual name, date of birth, license or ID number, street address, or "
                 "phone number appears as literal text."),
        "false": ("No such identifier appears, or every identifier has been replaced by a "
                  "placeholder such as [NAME] or [PHONE]."),
    },
}

# Stage-1 lean screen (§6 Step 2): 10 presence Nouls, <=25-word instructions, <=20-word criteria.
LEAN_IDS = ["preg_mentioned", "medical_episode", "drug_involved", "alcohol_involved",
            "hydroplane", "vehicle_defect", "fatigue", "aggression", "intentional", "wrong_way"]

LEAN = {
    "preg_mentioned": ("The `narrative` states that a person involved in the crash was pregnant.",
                       "Pregnancy of an involved person is explicitly stated.",
                       "No pregnancy is stated. Children, infants, or car seats do not count."),
    "medical_episode": ("The `narrative` states that a driver had a medical event such as a seizure, fainting, diabetic episode, heart attack, or stroke before the crash.",
                        "A driver medical event before or during the crash is described, confirmed or suspected.",
                        "No driver medical event. Injuries caused by the crash do not count."),
    "drug_involved": ("The `narrative` states or suspects that a driver was under the influence of drugs or medication, not alcohol.",
                      "Drug, narcotic, or medication impairment of a driver is stated or suspected.",
                      "No drug involvement. Alcohol-only impairment does not count."),
    "alcohol_involved": ("The `narrative` states or suspects that a driver had been drinking alcohol or was intoxicated by alcohol.",
                         "Drinking, odour of alcohol, DWI, DUI, or suspected alcohol intoxication of a driver.",
                         "No alcohol involvement of a driver is mentioned."),
    "hydroplane": ("The `narrative` states that a vehicle hydroplaned or lost traction because of water on the roadway.",
                   "Hydroplaning, or loss of traction or control caused by rain, wet road, or standing water.",
                   "No loss of traction due to water. Wet conditions alone do not count."),
    "vehicle_defect": ("The `narrative` states that a mechanical failure of a vehicle contributed to the crash.",
                       "Tire blowout, brake, steering, or other mechanical failure contributed to the crash.",
                       "No mechanical failure. Damage caused by the crash does not count."),
    "fatigue": ("The `narrative` states that a driver fell asleep, dozed off, was drowsy, or was fatigued.",
                "A driver was asleep, falling asleep, drowsy, tired, or fatigued before the crash.",
                "No sleep or fatigue of a driver is mentioned."),
    "aggression": ("The `narrative` describes road rage, aggressive driving directed at another road user, or a confrontation between people involved.",
                   "Road rage, brake-checking, chasing, threatening, arguing, or fighting between involved people.",
                   "No aggression. A police pursuit, traffic stop, evading police, or leaving the scene does not count."),
    "intentional": ("The `narrative` states or suspects that a vehicle was used intentionally to cause the crash.",
                    "The crash is described as deliberate, intentional, an assault with a vehicle, or a suicide attempt.",
                    "The crash is unintentional, or intent is not mentioned."),
    "wrong_way": ("The `narrative` states that a vehicle was traveling the wrong way on a roadway, ramp, or lane.",
                  "A vehicle travelled the wrong way or against traffic on a one-way road, divided highway, or ramp.",
                  "No wrong-way travel. Crossing the center line during a loss of control does not count."),
}


def main() -> None:
    src = json.loads((SCHEMAS / "crash_factors_v1.json").read_text())
    out = json.loads(json.dumps(src))  # deep copy
    diff = []

    before = out["questions"]["aggression"]["criteria"]["false"]
    out["questions"]["aggression"]["criteria"]["false"] = AGGRESSION_FALSE
    diff.append({"question": "aggression", "field": "criteria.false", "rule": "§4.2 rule 6",
                 "before": before, "after": AGGRESSION_FALSE,
                 "evidence": "16 spike narratives with pursuit language; 4 scored p>0.5 (max 0.95)"})

    out["questions"]["pii_residual"] = PII_RESIDUAL
    diff.append({"question": "pii_residual", "field": "(new)", "rule": "§6 Step 1",
                 "before": None, "after": "added presence Noul for residual PII screening"})

    out["name"] = "crash_factors_v1_1"
    out["parent"] = "crash_factors_v1"
    out["drug_token_regex"] = DRUG_TOKEN_RE.pattern
    out["notes"] = ("v1.1 applies §4.2 rule 6 (aggression excludes police pursuit / evading / "
                    "leaving the scene) and adds pii_residual. drug_class stays a fixed closed "
                    "Choice; the regex candidate pass is recorded per narrative as "
                    "drug_tokens_found rather than injected into the option set, so that "
                    "confidence values stay comparable across narratives for §4.5.")
    (SCHEMAS / "crash_factors_v1_1.json").write_text(json.dumps(out, indent=1))

    lean = {"name": "crash_factors_lean_v1_1", "model": "jev-1.13.0", "state_key": "narrative",
            "questions": {k: {"type": "noul", "instructions": LEAN[k][0],
                              "criteria": {"true": LEAN[k][1], "false": LEAN[k][2]}}
                          for k in LEAN_IDS},
            "gates": {},
            "notes": "§6 Step 2 lean screen. Same 10 presence concepts as the full schema, "
                     "shortened instructions and criteria to hold the token budget near 930. "
                     "pii_residual is deliberately NOT included: an 11th Noul would add ~9% to "
                     "every one of 500k Stage-1 calls (~$1.8, over the $19.5 line item) to screen "
                     "narratives that will never be displayed. §6 Step 1 needs it only for "
                     "narratives that are quoted or shown publicly, so it ships as its own "
                     "one-question schema (pii_residual_v1.json) run on display candidates."}
    (SCHEMAS / "crash_factors_lean_v1_1.json").write_text(json.dumps(lean, indent=1))

    pii = {"name": "pii_residual_v1", "model": "jev-1.13.0", "state_key": "narrative",
           "questions": {"pii_residual": PII_RESIDUAL}, "gates": {},
           "notes": "§6 Step 1 residual-PII screen. Run on any narrative considered for "
                    "quotation in the manuscript or display in the public tool; exclude p>0.2."}
    (SCHEMAS / "pii_residual_v1.json").write_text(json.dumps(pii, indent=1))

    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "schema_diff.json").write_text(json.dumps(diff, indent=1))

    def words(q):
        n = len(q["instructions"].split())
        c = q.get("criteria") or {}
        vals = c.values() if isinstance(c, dict) else c
        return n, max((len(str(v).split()) for v in vals), default=0)

    print("full schema questions:", len(out["questions"]))
    print("lean schema questions:", len(lean["questions"]))
    worst = max((words(q) for q in lean["questions"].values()))
    print(f"lean: max instruction words={max(words(q)[0] for q in lean['questions'].values())} "
          f"(target <=25), max criteria words={max(words(q)[1] for q in lean['questions'].values())} (target <=20)")
    print("wrote crash_factors_v1_1.json, crash_factors_lean_v1_1.json, data/schema_diff.json")


if __name__ == "__main__":
    main()
