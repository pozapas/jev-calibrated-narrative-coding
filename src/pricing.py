"""Pricing constants, with access dates, for the cost model (§4.3, F2d, F9, T3, T7).

Every price here must be quoted in the manuscript with its access date (§8 arXiv checklist:
"TypeSafe pricing quoted with access date"). Re-verify before submission.
"""
# TypeSafe Jev -- typed decisions only, no output tokens billed.
JEV_INPUT_PER_M = 0.042          # USD / 1M input tokens, accessed 2026-09-17
JEV_MODEL = "jev-1.13.0"

# Anthropic Claude, first-party API list prices. Source: the vendor's published model
# pricing table, accessed 2026-09-17. Re-verify before submission.
CLAUDE = {
    "claude-sonnet-5": {"input_per_m": 2.00, "output_per_m": 10.00},
    "claude-opus-5":   {"input_per_m": 5.00, "output_per_m": 25.00},
    "claude-haiku-4-5": {"input_per_m": 1.00, "output_per_m": 5.00},
}
LLM_BASELINE = "claude-sonnet-5"   # §4.8 frontier-LLM baseline

def llm_cost_per_1k(input_tokens: float, output_tokens: float = 60.0,
                    model: str = LLM_BASELINE) -> float:
    """USD per 1,000 narratives for a generative baseline that must emit JSON."""
    p = CLAUDE[model]
    return 1000 * (input_tokens * p["input_per_m"] + output_tokens * p["output_per_m"]) / 1e6

def jev_cost_per_1k(input_tokens: float) -> float:
    return 1000 * input_tokens * JEV_INPUT_PER_M / 1e6


def run_input_tokens() -> float:
    """The measured mean input tokens per narrative for the production run.

    Every cost the paper quotes is derived from this one number, so the prose, the tables and
    the figures cannot disagree. Three bases were in circulation before: this run mean, the
    cost model's 27-question fit, and the random stratum's own mean, giving $0.1543, $0.1553
    and $0.1539 per thousand. They are all defensible and that is exactly the problem, since a
    reader who finds two of them stops trusting the third.

    This one is chosen because it is checkable by division: multiplied by the price and by the
    narratives coded it reproduces the realised spend. The cost model's fit remains the right
    basis for the scaling law in Figure 2, which is a different claim about how cost grows
    with schema size, not about what this run cost.
    """
    import json
    from pathlib import Path
    a = json.loads((Path(__file__).resolve().parent.parent / "data" /
                    "analysis.json").read_text())
    return float(a["run"]["mean_input_tokens"])
