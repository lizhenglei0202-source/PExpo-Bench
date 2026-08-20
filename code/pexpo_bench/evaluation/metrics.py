"""Three-layer evaluation metrics for PExpo-Bench.

Layer 1 — Output quality:
    accuracy(), instruction_following_rate(), hallucination_rate()

Layer 2 — Process quality:
    retrieval_precision_recall(), tool_use_f1(), tool_sequence_edit_distance(),
    grounding_rate(), physical_consistency()

Layer 3 — Cost & robustness:
    token_cost(), latency_stats(), self_consistency()

Each metric accepts a list of (Result, gold_item) tuples so it can be computed
over any slice (by subdomain / difficulty / question type).

 schema notes
---------------
• `gold_references[*].doc_id` and `chunk_id` are METADATA for human readability
  and audit only. They are NOT used by any scoring metric. All process metrics
  match retrieved chunk TEXT against gold `quote` TEXT via semantic similarity.
• `gold_tools` is descriptive. `tool_use_f1` is retired as a primary metric
  (penalizing alternative solution paths conflicts with the agentic-planning
  hypothesis under test). It is retained only for descriptive tool-usage
  histograms in SI.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Callable


# ==========================================================================
# Helpers
# ==========================================================================
def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _rel_err(pred: float, gold: float) -> float:
    if gold == 0:
        return abs(pred)
    return abs(pred - gold) / abs(gold)


def _normalize(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


# Schema adapters — v3_release uses flat fields (answer/unit/tolerance,
# question_type) while the original seed schema used nested (gold_answer.value,
# type). Accept either.
def _gold_value(g: dict):
    if "gold_answer" in g and isinstance(g["gold_answer"], dict):
        return g["gold_answer"].get("value")
    return g.get("answer")


def _gold_tolerance(g: dict, default: float = 0.05) -> float:
    if "gold_answer" in g and isinstance(g["gold_answer"], dict):
        t = g["gold_answer"].get("tolerance")
        if t is not None:
            return float(t)
    t = g.get("tolerance")
    return float(t) if t is not None else default


def _gold_type(g: dict) -> str:
    return g.get("question_type") or g.get("type") or ""


# ==========================================================================
# Layer 1 — Output quality
# ==========================================================================
def accuracy(results: list[dict], gold: list[dict]) -> float:
    """Per-type accuracy; call separately for T/F, calculation, open_ended."""
    if not results:
        return 0.0
    hit = 0
    for r, g in zip(results, gold):
        pred = r.get("answer")
        gans = _gold_value(g)
        gtype = _gold_type(g)
        if gtype == "true_false":
            # Accept textual "True"/"False" on either side.
            pb = pred
            gb = gans
            if isinstance(pb, str):
                pb = pb.strip().lower() in ("true", "yes", "1", "correct")
            if isinstance(gb, str):
                gb = gb.strip().lower() in ("true", "yes", "1", "correct")
            hit += int(bool(pb) == bool(gb))
        elif gtype == "calculation":
            if _is_number(pred) and _is_number(gans):
                tol = _gold_tolerance(g)
                hit += int(_rel_err(pred, gans) <= tol)
        else:  # open_ended
            # Deferred to LLM-judge; placeholder — see llm_judge_score()
            hit += int(llm_judge_score(pred, g) >= 0.5)
    return hit / len(results)


def instruction_following_rate(results: list[dict]) -> float:
    """Valid JSON schema with required keys present."""
    required = {"answer", "reasoning", "citations", "tool_calls"}
    ok = sum(
        1 for r in results
        if not r.get("parse_error") and required.issubset(r.keys())
    )
    return ok / max(1, len(results))


def hallucination_rate(results: list[dict], gold: list[dict],
                       judge: Callable | None = None) -> float:
    """Fraction of open-ended answers unsupported by gold references."""
    judge = judge or llm_judge_supported
    n = unsupported = 0
    for r, g in zip(results, gold):
        if _gold_type(g) != "open_ended":
            continue
        n += 1
        if not judge(r.get("reasoning", ""), g):
            unsupported += 1
    return unsupported / max(1, n)


def llm_judge_score(pred, gold: dict) -> float:
    """Placeholder: in production call GPT-4o-as-judge with gold.acceptable_forms."""
    acceptable = (gold.get("gold_answer") or {}).get("acceptable_forms") or []
    pred_s = _normalize(pred)
    if not acceptable:
        return 1.0 if pred_s == _normalize(_gold_value(gold)) else 0.0
    hits = sum(1 for form in acceptable if _normalize(form) in pred_s)
    return hits / max(1, len(acceptable))


def llm_judge_supported(reasoning: str, gold: dict,
                        sim_threshold: float = 0.45) -> bool:
    """Content-based 'is the reasoning supported by gold quotes' check ().

    Decompose reasoning into sentences; for each sentence compute cosine
    similarity to each gold-reference `quote`. The reasoning is considered
    "supported" if at least HALF of its substantive sentences cover ≥1 gold
    quote at cosine ≥ sim_threshold.

    This is a SEMANTIC heuristic (sentence-transformer MiniLM). For the 
    headline HR figure, replace with an LLM-judge that decomposes claims into
    atomic propositions and runs entailment — see RefChecker (Hu et al. 2024).

    No doc_id is used.
    """
    gold_quotes = [r.get("quote", "") for r in gold.get("gold_references", [])
                   if isinstance(r, dict) and r.get("quote")]
    if not gold_quotes:
        return True  # nothing to ground against → not penalized
    sents = [s.strip() for s in re.split(r"(?<=[\.!?。!?])\s+", reasoning or "")
             if len(s.strip()) > 15]
    if not sents:
        return False
    sim = _cosine_sim_batch(sents, gold_quotes)
    covered = (sim.max(axis=1) >= sim_threshold).sum()
    return covered >= max(1, len(sents) // 2)


# ==========================================================================
# Layer 2 — Process quality
# ==========================================================================
# --- semantic similarity cache (lazy-loaded MiniLM) ---
_SBERT_MODEL = None
def _sbert():
    """Return cached sentence-transformer model."""
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SBERT_MODEL


def _cosine_sim_batch(queries: list[str], targets: list[str]) -> "np.ndarray":
    """Pairwise cosine between len(queries) × len(targets) (both lists of str).
    Returns numpy array of shape (len(queries), len(targets))."""
    import numpy as np
    if not queries or not targets:
        return np.zeros((len(queries), len(targets)))
    m = _sbert()
    qv = m.encode([q[:1200] for q in queries], normalize_embeddings=True)
    tv = m.encode([t[:1200] for t in targets], normalize_embeddings=True)
    return qv @ tv.T


def retrieval_precision_recall(result: dict, gold: dict, k: int = 5,
                               sim_threshold: float = 0.50) -> tuple[float, float]:
    """Content-based retrieval P/R ( — doc_id NOT used).

    A retrieved chunk counts as a hit if its TEXT semantically matches any
    gold-reference `quote` at cosine ≥ sim_threshold.
    Threshold 0.50 is calibrated from the bank's curated gold refs
    (curated refs have median cosine 0.60 to question+rationale).

    Returns:
        (precision, recall) ∈ [0,1]² or (nan, nan) if gold has no refs.
    """
    gold_quotes = [r.get("quote", "") for r in gold.get("gold_references", [])
                   if isinstance(r, dict) and r.get("quote")]
    if not gold_quotes:
        return float("nan"), float("nan")

    retrieved_texts = [d.get("text", "") for d in result.get("retrieved_docs", [])[:k]
                       if d.get("text")]
    if not retrieved_texts:
        return 0.0, 0.0

    sim = _cosine_sim_batch(retrieved_texts, gold_quotes)
    # A retrieved chunk i is a "hit" iff it covers ≥1 gold quote
    retrieved_hits = (sim.max(axis=1) >= sim_threshold).sum()
    # A gold quote j is "found" iff ≥1 retrieved chunk covers it
    gold_found = (sim.max(axis=0) >= sim_threshold).sum()

    precision = float(retrieved_hits) / len(retrieved_texts)
    recall = float(gold_found) / len(gold_quotes)
    return precision, recall


def tool_use_f1(result: dict, gold: dict) -> float:
    """**DEPRECATED in **. Retained for backward compatibility.

    Rationale for retirement: gold_tools encodes the question-author's
    expected solution path. Penalizing alternative valid paths (e.g., using
    python_sandbox where dose_calculator was "expected") conflicts with the
    paper's central hypothesis that agentic *planning* — the model choosing
    its own tools — is the unit under test.

    The bank's `gold_tools` field is retained as descriptive metadata.
    For paper figures, use `tool_usage_descriptive()` below to report tool-call
    histograms per architecture without grading.

    This function still computes recall-only against gold_tools for SI tables
    that compare (graded) and (descriptive) framings.
    """
    gold_req = Counter(t["tool"] for t in gold.get("gold_tools", [])
                       if isinstance(t, dict) and t.get("tool")
                       and not t.get("optional"))
    pred = Counter(t["tool"] for t in result.get("tool_calls", []))
    if not gold_req:
        return float("nan")  # : no penalty / no reward when gold is empty
    hit = sum((gold_req & pred).values())
    return hit / sum(gold_req.values())


def tool_usage_descriptive(result: dict) -> dict:
    """ descriptive metric — what tools did the model actually use?

    Returns:
        {n_calls, n_unique_tools, histogram, first_tool, sequence}
    No comparison against gold; pure observation of agent behavior.
    """
    calls = result.get("tool_calls", []) or []
    names = [t.get("tool") for t in calls if isinstance(t, dict) and t.get("tool")]
    return {
        "n_calls": len(names),
        "n_unique_tools": len(set(names)),
        "histogram": dict(Counter(names)),
        "first_tool": names[0] if names else None,
        "sequence": names,
    }


def tool_sequence_edit_distance(result: dict, gold: dict) -> float:
    gold_seq = [t["tool"] for t in sorted(gold.get("gold_tools", []), key=lambda x: x.get("order", 0))]
    pred_seq = [t["tool"] for t in result.get("tool_calls", [])]
    if not gold_seq:
        return 0.0 if not pred_seq else float("nan")
    # Levenshtein
    m, n = len(gold_seq), len(pred_seq)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if gold_seq[i - 1] == pred_seq[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n] / max(m, n)


def grounding_rate(result: dict, gold: dict,
                   sim_threshold: float = 0.50) -> float:
    """Content-based grounding: fraction of model citations whose claimed
    evidence actually appears among its retrieved chunks.

     — no doc_id matching. A citation is "grounded" if its quoted text (or
    surrounding reasoning context) has cosine ≥ sim_threshold to ≥1 retrieved
    chunk's text.

    Returns NaN if model didn't cite or didn't retrieve.
    """
    cites = result.get("citations", []) or []
    if not cites:
        return float("nan")
    retrieved_texts = [d.get("text", "") for d in result.get("retrieved_docs", []) if d.get("text")]
    if not retrieved_texts:
        return float("nan")
    # Citation can be a string or dict; extract any text-ish content
    cite_texts = []
    for c in cites:
        if isinstance(c, dict):
            cite_texts.append(c.get("quote") or c.get("section") or c.get("doc_id") or "")
        else:
            cite_texts.append(str(c))
    cite_texts = [c for c in cite_texts if c.strip()]
    if not cite_texts:
        return float("nan")
    sim = _cosine_sim_batch(cite_texts, retrieved_texts)
    return float((sim.max(axis=1) >= sim_threshold).mean())


def physical_consistency(result: dict, gold: dict) -> bool:
    """Check dimensional/range/sign constraints for calculation answers."""
    if _gold_type(gold) != "calculation":
        return True
    pred = result.get("answer")
    unit = result.get("unit")
    gold_unit = gold.get("unit")
    cons = gold.get("physical_constraints", {})
    rng = cons.get("range")

    if not _is_number(pred):
        return False
    if gold_unit and unit and _normalize(unit) != _normalize(gold_unit):
        return False
    if rng:
        try:
            lo, hi = float(rng[0]), float(rng[1])
            if not (lo <= pred <= hi):
                return False
        except (TypeError, ValueError):
            pass  # malformed range → skip this check
    return True


def multi_hop_success(result: dict, gold: dict) -> bool:
    """Hard-only: require tool_use_f1 == 1.0 AND final answer correct."""
    if gold.get("difficulty") != "hard":
        return True
    if tool_use_f1(result, gold) < 1.0:
        return False
    if _gold_type(gold) == "calculation":
        return physical_consistency(result, gold) and \
               _rel_err(result.get("answer", math.nan),
                        _gold_value(gold)) <= _gold_tolerance(gold, 0.05)
    return llm_judge_score(result.get("answer"), gold) >= 0.5


# ==========================================================================
# Layer 3 — Cost & robustness
# ==========================================================================
def token_cost(results: list[dict]) -> dict:
    inp = sum(r.get("input_tokens", 0) for r in results)
    out = sum(r.get("output_tokens", 0) for r in results)
    return {"input": inp, "output": out, "total": inp + out,
            "per_question": (inp + out) / max(1, len(results))}


def latency_stats(results: list[dict]) -> dict:
    lats = sorted(r.get("total_latency_s", 0.0) for r in results)
    if not lats:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    def pct(p):  # noqa
        return lats[min(len(lats) - 1, int(len(lats) * p))]
    return {"p50": pct(0.5), "p95": pct(0.95),
            "mean": sum(lats) / len(lats)}


def self_consistency(repeat_runs: list[list[dict]], gold: list[dict]) -> float:
    """repeat_runs: list of independent runs, each aligned to `gold`."""
    if len(repeat_runs) < 2:
        return float("nan")
    n = len(gold)
    consistent = 0
    for i in range(n):
        answers = [run[i].get("answer") for run in repeat_runs]
        if gold[i]["type"] == "calculation":
            tol = gold[i]["gold_answer"].get("tolerance", 0.05)
            ref = answers[0]
            if _is_number(ref) and all(
                _is_number(a) and _rel_err(a, ref) <= tol for a in answers):
                consistent += 1
        else:
            normed = {_normalize(a) for a in answers}
            if len(normed) == 1:
                consistent += 1
    return consistent / n


# ==========================================================================
# Failure mode classifier
# ==========================================================================
def classify_failure(result: dict, gold: dict) -> str | None:
    """Return one of: retrieval_failure, tool_failure, reasoning_failure,
    hallucination, None (=success)."""
    if _gold_type(gold) == "calculation":
        tol = _gold_tolerance(gold, 0.05)
        if _is_number(result.get("answer")) and \
           _rel_err(result["answer"], _gold_value(gold)) <= tol:
            return None
    elif _gold_type(gold) == "true_false":
        if bool(result.get("answer")) == bool(_gold_value(gold)):
            return None
    else:
        if llm_judge_score(result.get("answer"), gold) >= 0.5:
            return None

    # Error attribution
    if result.get("retrieved_docs") is not None:
        p, r = retrieval_precision_recall(result, gold)
        if not math.isnan(p) and p == 0:
            return "retrieval_failure"
    if result.get("tool_calls"):
        # : only flag actual tool execution errors; do NOT penalize
        # alternative tool choices (that's the model's prerogative).
        errored = [t for t in result["tool_calls"] if t.get("error")]
        if errored:
            return "tool_failure"
    if not llm_judge_supported(result.get("reasoning", ""), gold):
        return "hallucination"
    return "reasoning_failure"
