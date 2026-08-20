"""Cross-family judge dispatch.

Per paper Methods (Zheng 2023 / AgentHallu 2026 best practice):
  • Output from family X → judge from family Y (different vendor)
  • Avoids self-preference bias documented at ~5-10%

Mapping ():
  GPT-5.4 / GPT-5.4-nano (OpenAI) → DeepSeek V4 Flash judge
  DeepSeek V4 Flash (DeepSeek) → GPT-4o-mini judge (via proxy)
"""
from __future__ import annotations

# Map: subject model_key → judge model_key
JUDGE_FOR = {
    "gpt-5.4":      "deepseek-v4",
    "gpt-5.4-native": "deepseek-v4",
    "gpt-5.4-mini": "deepseek-v4",
    "gpt-5.4-nano": "deepseek-v4",
    "deepseek-v4":  "gpt-4o-mini",
    # back-compat aliases
    "gpt-4o":       "deepseek-v4",
    "gpt-4o-mini":  "deepseek-v4",
    "deepseek":     "gpt-4o-mini",
}


def judge_model_for(subject_model_key: str) -> str:
    """Return the cross-family judge for a given subject model."""
    if subject_model_key not in JUDGE_FOR:
        raise KeyError(f"No judge mapping for subject '{subject_model_key}'. "
                       f"Available: {list(JUDGE_FOR.keys())}")
    return JUDGE_FOR[subject_model_key]


def judge_family_note() -> str:
    """Methods-paragraph-ready description of the dispatch policy."""
    return (
        "All LLM-as-judge evaluations (atomic claim extraction, entailment "
        "scoring, tool-use hallucination detection) were assigned to a judge "
        "model from a different vendor family than the system under test. "
        "Outputs from GPT-5.4 and GPT-5.4-nano were judged by DeepSeek V4 "
        "Flash; outputs from DeepSeek V4 were judged by GPT-4o-mini (accessed "
        "via the proxy proxy). This mitigates the self-preference bias "
        "documented by Zheng et al. (2023, MT-Bench)."
    )
