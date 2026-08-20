"""Automated scoring for PExpo-Bench results.

Scores each (result, gold) pair using type-specific logic:
  - true_false: exact match (case-insensitive)
  - calculation: numerical extraction + relative tolerance
  - open_ended: LLM-as-judge (GPT-4o) with rubric

Usage:
    python -m pexpo_bench.evaluation.score \
        --results pexpo_bench/runs/exp_gpt4o_v2/A0_naive.jsonl \
        --gold pexpo_bench/samples/pexpo_bench_v2.yaml \
        --out pexpo_bench/runs/exp_gpt4o_v2/A0_naive_scored.jsonl
"""
from __future__ import annotations

import json
import re
import sys
import pathlib
import yaml
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


# ==========================================================================
# Numerical extraction
# ==========================================================================
def extract_number(text: str) -> float | None:
    """Extract the first meaningful number from text."""
    if text is None:
        return None
    s = str(text).strip()
    # Direct number
    try:
        return float(s)
    except (ValueError, TypeError):
        pass
    # Scientific notation
    m = re.search(r'[-+]?\d*\.?\d+\s*[×x×]\s*10\s*[⁻\-]?\s*\d+', s)
    if m:
        clean = re.sub(r'[×x×]\s*10\s*[⁻\-]?\s*', 'e-', m.group())
        try:
            return float(clean)
        except:
            pass
    m = re.search(r'[-+]?\d+\.?\d*[eE][-+]?\d+', s)
    if m:
        try:
            return float(m.group())
        except:
            pass
    # Plain number (first occurrence)
    m = re.search(r'[-+]?\d+\.?\d*', s)
    if m:
        try:
            return float(m.group())
        except:
            pass
    return None


# ==========================================================================
# T/F scoring
# ==========================================================================
def score_tf(pred_answer: Any, gold_answer: str) -> float:
    """Returns 1.0 for correct, 0.0 for incorrect."""
    if pred_answer is None:
        return 0.0
    pred = str(pred_answer).strip().lower()
    gold = str(gold_answer).strip().lower()
    # Normalize
    pred_bool = None
    if pred in ("true", "yes", "correct", "1"):
        pred_bool = True
    elif pred in ("false", "no", "incorrect", "0"):
        pred_bool = False
    elif "true" in pred and "false" not in pred:
        pred_bool = True
    elif "false" in pred and "true" not in pred:
        pred_bool = False

    gold_bool = None
    if gold in ("true", "yes", "correct", "1"):
        gold_bool = True
    elif gold in ("false", "no", "incorrect", "0"):
        gold_bool = False

    if pred_bool is None or gold_bool is None:
        return 0.0
    return 1.0 if pred_bool == gold_bool else 0.0


# ==========================================================================
# Calculation scoring
# ==========================================================================
def score_calc(pred_answer: Any, gold_answer: str, tolerance: float = 0.10) -> float:
    """Extract numbers and compare with relative tolerance."""
    pred_num = extract_number(str(pred_answer))
    gold_num = extract_number(str(gold_answer))
    if pred_num is None or gold_num is None:
        return 0.0
    if gold_num == 0:
        return 1.0 if abs(pred_num) < 1e-9 else 0.0
    rel_err = abs(pred_num - gold_num) / abs(gold_num)
    if rel_err <= tolerance:
        return 1.0
    elif rel_err <= tolerance * 2:
        return 0.5  # partial credit
    return 0.0


# ==========================================================================
# Open-ended scoring (LLM-as-judge)
# ==========================================================================
JUDGE_PROMPT = """You are an expert evaluator for an environmental health science exam.

Score the student's answer against the reference answer on a scale of 0-5:
  5: Completely correct, covers all key points with accurate details
  4: Mostly correct, minor omissions or imprecisions
  3: Partially correct, captures main idea but misses important details
  2: Weakly correct, shows some understanding but significant gaps
  1: Mostly incorrect, only tangentially related
  0: Completely wrong, irrelevant, or no answer

QUESTION: {question}

REFERENCE ANSWER: {gold}

STUDENT ANSWER: {pred}

Output ONLY a single integer (0-5), nothing else."""


def score_open_llm(question: str, pred_answer: str, gold_answer: str,
                   judge_client=None) -> float:
    """Use LLM to score open-ended answers. Returns 0.0-1.0."""
    if not pred_answer or pred_answer.strip() == "":
        return 0.0
    if judge_client is None:
        # Fallback: keyword overlap
        return _score_open_keyword(pred_answer, gold_answer)

    prompt = JUDGE_PROMPT.format(
        question=question, gold=gold_answer, pred=pred_answer
    )
    try:
        resp = judge_client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=10
        )
        score = int(re.search(r'\d', resp.content).group())
        return min(score, 5) / 5.0
    except Exception:
        return _score_open_keyword(pred_answer, gold_answer)


def _score_open_keyword(pred: str, gold: str) -> float:
    """Fallback: fraction of gold keywords present in prediction."""
    gold_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', gold.lower()))
    pred_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', pred.lower()))
    if not gold_words:
        return 0.5
    overlap = len(gold_words & pred_words) / len(gold_words)
    return min(overlap, 1.0)


# ==========================================================================
# Main scorer
# ==========================================================================
def score_all(results_path: str, gold_path: str, out_path: str,
              judge_model: str | None = None) -> dict:
    """Score all results against gold answers."""
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).resolve().parents[2] / ".env")

    results = [json.loads(l) for l in pathlib.Path(results_path).read_text().splitlines() if l.strip()]
    gold_list = yaml.safe_load(pathlib.Path(gold_path).read_text())
    gold_map = {q["qid"]: q for q in gold_list}

    judge = None
    if judge_model:
        from pexpo_bench.llm_clients import LLMClient
        judge = LLMClient(judge_model, temperature=0.0, max_tokens=20)

    scored = []
    totals = {"true_false": [], "calculation": [], "open_ended": []}

    for r in results:
        qid = r.get("qid", "")
        g = gold_map.get(qid)
        if g is None:
            continue

        qtype = g.get("question_type", "")
        gold_ans = str(g.get("answer", ""))
        pred_ans = r.get("answer") or r.get("raw_output", "")

        if qtype == "true_false":
            score = score_tf(pred_ans, gold_ans)
        elif qtype == "calculation":
            score = score_calc(pred_ans, gold_ans)
        elif qtype == "open_ended":
            score = score_open_llm(g.get("question",""), str(pred_ans), gold_ans, judge)
        else:
            score = 0.0

        r["score"] = score
        r["question_type"] = qtype
        r["subdomain"] = g.get("subdomain", "")
        r["difficulty"] = g.get("difficulty", "")
        scored.append(r)
        totals.setdefault(qtype, []).append(score)

    # Save scored results
    with open(out_path, "w") as f:
        for r in scored:
            f.write(json.dumps(r, default=str) + "\n")

    # Summary
    summary = {"total": len(scored)}
    for qtype, scores in totals.items():
        if scores:
            summary[f"{qtype}_accuracy"] = sum(scores) / len(scores)
            summary[f"{qtype}_n"] = len(scores)
    all_scores = [r["score"] for r in scored]
    summary["overall_accuracy"] = sum(all_scores) / len(all_scores) if all_scores else 0
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--gold", default="pexpo_bench/samples/pexpo_bench_v2.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge", default=None, help="Model key for LLM judge (e.g. gpt-4o)")
    args = ap.parse_args()
    summary = score_all(args.results, args.gold, args.out, args.judge)
    print(json.dumps(summary, indent=2))
