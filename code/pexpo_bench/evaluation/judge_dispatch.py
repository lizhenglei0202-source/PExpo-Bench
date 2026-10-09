"""Cross-family model assignments for the two evaluation tasks."""

OPEN_ANSWER_JUDGES = {
    "gpt-5.4": ("deepseek-v4", 600),
    "gpt-5.4-mini": ("deepseek-v4", 600),
    "gpt-5.4-nano": ("deepseek-v4", 600),
    "deepseek-v4": ("gpt-5.4-nano", 16),
}
GROUNDING_JUDGES = {
    "gpt-5.4": "deepseek-v4",
    "gpt-5.4-mini": "deepseek-v4",
    "gpt-5.4-nano": "deepseek-v4",
    "deepseek-v4": "gpt-4o-mini",
}

def judge_model_for(subject_model_key):
    """Return the model used for claim extraction and entailment judgments."""
    return GROUNDING_JUDGES[subject_model_key]
