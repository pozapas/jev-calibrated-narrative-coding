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
