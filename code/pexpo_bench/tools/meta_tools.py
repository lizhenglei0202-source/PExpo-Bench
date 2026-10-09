"""Meta-tools: planning trace.

These tools don't compute anything domain-specific. Their purpose is to make
the agent's planning process VISIBLE in the tool_calls log.

  submit_plan(steps)              — required first call in A3/A4
  revise_plan(reason, new_steps)  — when a tool result invalidates the plan

The agent harness enforces: 'submit_plan' must be the first tool_call.
If it isn't, the runner logs an instruction-following violation.
"""
from __future__ import annotations
from typing import List


def submit_plan(steps: List[str], expected_tools: List[str] = None) -> dict:
    """Submit an initial reasoning plan.

    Args:
        steps:           ordered list of natural-language plan steps (≥ 2)
        expected_tools:  optional list of tool names the plan anticipates calling
    """
    if not isinstance(steps, list) or len(steps) < 2:
        return {"plan_accepted": False,
                "error": "plan must contain ≥ 2 steps"}
    return {
        "plan_accepted": True,
        "n_steps": len(steps),
        "steps": [str(s)[:200] for s in steps],
        "expected_tools": list(expected_tools or []),
        "ack": "plan recorded; proceed with tool calls",
    }


def revise_plan(reason: str, new_steps: List[str]) -> dict:
    """Revise the plan when a previous tool result invalidates it.

    Args:
        reason:    natural-language explanation (≥ 10 chars)
        new_steps: revised plan (≥ 1 step)
    """
    if not isinstance(new_steps, list) or len(new_steps) < 1:
        return {"plan_accepted": False, "error": "new_steps must be ≥ 1"}
    if not isinstance(reason, str) or len(reason.strip()) < 10:
        return {"plan_accepted": False,
                "error": "reason must be ≥ 10 chars explaining what changed"}
    return {
        "plan_revised": True,
        "reason": reason[:300],
        "n_steps": len(new_steps),
        "new_steps": [str(s)[:200] for s in new_steps],
        "ack": "revised plan recorded; proceed with new steps",
    }


META_TOOLS = {
    "submit_plan": submit_plan,
    "revise_plan": revise_plan,
}
