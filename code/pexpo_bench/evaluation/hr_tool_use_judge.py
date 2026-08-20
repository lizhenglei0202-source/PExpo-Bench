"""HR_tool_use — tool-call-level hallucination detection.

Per AgentHallu (Lin et al. 2026, arxiv 2601.06818), tool-use hallucinations are
the HARDEST subcategory to catch (SOTA step-localization accuracy 11.6%).
We catch a *subset*: structural / parameter errors that are cheap to detect
without LLM judging.

5 categories of tool-use hallucination flagged:

  1. UNKNOWN_TOOL   — model called a tool name not in our registry
  2. INVALID_ARGS   — required arg missing OR wrong type
  3. INVALID_VALUE  — arg value out of declared enum / range (e.g. iris_lookup
                      chemical not in our 10-chem table)
  4. UNIT_MISMATCH  — heuristic: unit_converter from→to doesn't make physical sense
  5. CALL_WITHOUT_PLAN — A3/A4 first call was not submit_plan

HR_tool_use = (tool_calls_with_any_flag) / (total_tool_calls)

For LLM-judge layer (deeper diagnosis like "wrong tool for task"), an
optional LLM pass can be added; see judge_tool_choice_with_llm().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pexpo_bench.tools import TOOL_REGISTRY


# ==========================================================================
# Tool def cache — to validate args
# ==========================================================================
def _get_tool_defs():
    """Pull A4's TOOL_DEFS (superset of A3) for arg-spec lookup."""
    from pexpo_bench.architectures.orchestrator import A4_Hybrid
    return {td["function"]["name"]: td["function"]["parameters"]
            for td in A4_Hybrid.TOOL_DEFS}


# ==========================================================================
# Heuristic checks
# ==========================================================================
def _check_known_tool(name: str, defs: dict) -> Optional[str]:
    if name not in TOOL_REGISTRY and name not in defs:
        return f"UNKNOWN_TOOL: '{name}' not in registry"
    return None


def _check_required_args(name: str, args: dict, defs: dict) -> Optional[str]:
    spec = defs.get(name)
    if not spec: return None
    required = spec.get("required", []) or []
    missing = [r for r in required if r not in args]
    if missing:
        return f"INVALID_ARGS: missing {missing}"
    return None


def _check_enum_values(name: str, args: dict, defs: dict) -> Optional[str]:
    spec = defs.get(name)
    if not spec: return None
    props = spec.get("properties", {})
    bad = []
    for k, v in args.items():
        if k not in props: continue
        enum = props[k].get("enum")
        if enum and v not in enum:
            bad.append(f"{k}={v!r} ∉ {enum}")
    if bad:
        return f"INVALID_VALUE: " + "; ".join(bad)
    return None


def _check_iris_chemical(name: str, args: dict) -> Optional[str]:
    """iris_lookup with chemical not in our 10-chem table."""
    if name != "iris_lookup": return None
    chem = (args.get("chemical") or "").lower().strip()
    if not chem: return "INVALID_ARGS: iris_lookup missing chemical"
    from pexpo_bench.tools.health_tools import IRIS_DB, _CHEM_ALIASES
    key = _CHEM_ALIASES.get(chem, chem.replace(" ", "_"))
    if key not in IRIS_DB and key is not None:
        return f"INVALID_VALUE: chemical '{chem}' not in IRIS 10-chem table"
    return None


def _check_unit_converter_pair(name: str, args: dict) -> Optional[str]:
    """unit_converter from/to should be units of same dimension (heuristic)."""
    if name != "unit_converter": return None
    f = (args.get("from_") or args.get("from") or "").lower()
    t = (args.get("to") or "").lower()
    if not f or not t: return None
    # crude category mapping
    cat = lambda u: ("time" if u in {"s","min","h","day","hour","second","minute"} else
                     "volume" if u in {"l","ml","m3","liter","milliliter"} else
                     "mass" if u in {"g","mg","kg","ng","µg","ug","gram"} else
                     "conc" if u in {"mg/m³","µg/m³","ug/m³","ppb","ppm","mg/l","µg/l"} else
                     "other")
    cf, ct = cat(f), cat(t)
    if cf != "other" and ct != "other" and cf != ct:
        return f"UNIT_MISMATCH: '{f}'→'{t}' dimensions don't match ({cf}/{ct})"
    return None


def _check_first_call_is_plan(arch: str, tool_calls: list[dict]) -> Optional[str]:
    """A3/A4 family must submit_plan first."""
    if not arch.startswith(("A3", "A4")): return None
    if not tool_calls: return None
    first = tool_calls[0].get("tool") if isinstance(tool_calls[0], dict) else None
    if first != "submit_plan":
        return f"CALL_WITHOUT_PLAN: first tool was '{first}', expected submit_plan"
    return None


# ==========================================================================
# Public API
# ==========================================================================
@dataclass
class ToolCallAudit:
    qid: str
    arch: str
    model: str
    n_total_calls: int
    n_flagged: int
    flags: list[dict] = field(default_factory=list)   # per-call flags
    plan_violation: Optional[str] = None
    HR_tool_use: float | None = None


def audit_tool_use(result: dict) -> ToolCallAudit:
    """Audit one (qid, arch, model) record for tool-use hallucinations.

    Heuristic checks only (no LLM call). Cheap, deterministic, runs at
    aggregate-merge time, no API cost.

    HR_tool_use = (flagged calls) / (total calls); NaN-like if no calls.
    """
    qid = result.get("qid", "")
    arch = result.get("architecture", "")
    model = result.get("_model_key") or result.get("model", "")
    tool_calls = result.get("tool_calls") or []
    defs = _get_tool_defs()

    plan_v = _check_first_call_is_plan(arch, tool_calls)

    flag_records = []
    n_flagged = 0
    for i, tc in enumerate(tool_calls):
        if not isinstance(tc, dict): continue
        name = tc.get("tool") or tc.get("name") or ""
        args = tc.get("args", {}) or {}
        if isinstance(args, str):
            try:
                import json
                args = json.loads(args)
            except Exception:
                args = {}
        flags = []
        for chk in (_check_known_tool(name, defs),
                    _check_required_args(name, args, defs),
                    _check_enum_values(name, args, defs),
                    _check_iris_chemical(name, args),
                    _check_unit_converter_pair(name, args)):
            if chk: flags.append(chk)
        if flags:
            n_flagged += 1
            flag_records.append({"call_idx": i, "tool": name,
                                 "args_preview": str(args)[:200], "flags": flags})

    n_total = len(tool_calls)
    if n_total == 0:
        hr = None
    else:
        hr = n_flagged / n_total

    return ToolCallAudit(
        qid=qid, arch=arch, model=model,
        n_total_calls=n_total, n_flagged=n_flagged,
        flags=flag_records, plan_violation=plan_v,
        HR_tool_use=hr,
    )


def audit_corpus(results: list[dict]) -> dict:
    """Aggregate HR_tool_use across a corpus.

    Returns:
      {
        per_arch_model: {(arch, model): {mean_hr, n_calls_total, n_flagged_total, plan_v_rate}},
        per_question:   [ToolCallAudit, ...],
        flag_category_counts: {category: n}
      }
    """
    audits = [audit_tool_use(r) for r in results]
    from collections import defaultdict
    cell = defaultdict(lambda: {"flagged": 0, "total": 0, "plan_v": 0, "n_q": 0})
    cat_counts = defaultdict(int)
    for a in audits:
        c = cell[(a.arch, a.model)]
        c["flagged"] += a.n_flagged
        c["total"] += a.n_total_calls
        c["n_q"] += 1
        if a.plan_violation: c["plan_v"] += 1
        for fr in a.flags:
            for f in fr["flags"]:
                cat = f.split(":")[0]
                cat_counts[cat] += 1
    summary = {}
    for k, v in cell.items():
        summary[k] = {
            "mean_hr_tool_use": (v["flagged"] / v["total"]) if v["total"] else None,
            "total_tool_calls": v["total"],
            "flagged_calls": v["flagged"],
            "plan_violation_rate": v["plan_v"] / v["n_q"] if v["n_q"] else 0,
            "n_questions": v["n_q"],
        }
    return {
        "per_arch_model": summary,
        "per_question": audits,
        "flag_category_counts": dict(cat_counts),
    }
