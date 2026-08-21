"""Mechanism analysis for A4 (harness) sub-additivity.

Tests the hypothesis that A4 sub-additivity is an agentic CONTROL-FLOW collapse
on weak base models (enlarged action space the agent cannot govern), NOT
retrieved-default displacement.

For each base model it contrasts A3 (tool agent) with A4 (A3 + retrieve tool +
larger step budget) on the SAME items and reports, per question type:
  - open-ended answer-type collapse: prose question answered with a
    boolean / bare number / <15-char string (a tool-style output)
  - calculation non-numeric rate: no parseable number returned
  - mean tool-call steps, and the max_steps_exceeded (budget-exhaustion) rate
  - the rate at which A4 actually invokes retrieve (to show the harm is the
    action-space overhead, not retrieved content)

Output: runs/v4_rerun/_mechanism/summary.json  (+ console table)
"""
from __future__ import annotations
import os
import json, re, pathlib
import pandas as pd

ROOT = pathlib.Path(os.environ.get("PEXPO_ROOT", "."))
PARQ = ROOT / 'runs/v4_scored/all_scored_v4_main.parquet'
OUT = ROOT / 'runs/v4_rerun/_mechanism'
MODELS = ['gpt-5.4', 'gpt-5.4-mini', 'gpt-5.4-nano', 'deepseek-v4']
A3, A4 = 'A3_agent', 'A4p_hybrid_constrained'


def load(model, arch):
    d = {}
    for base in ('runs/v4_rerun', 'runs/_none_v4'):
        p = ROOT / base / model / arch / 'run_1.jsonl'
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line); d[r['qid']] = r
    return d


def is_numeric(x):
    s = str(x or '').strip().lower()
    return bool(re.search(r'\d', s)) and s not in ('true', 'false', 'none', '')


def open_collapsed(x):
    """An open-ended answer that collapsed to a tool-style output."""
    s = str(x or '').strip().lower()
    if len(s) < 15:
        return True
    if s in ('true', 'false', 'none', 'yes', 'no'):
        return True
    if re.fullmatch(r'[-+]?\d*\.?\d+(?:[ee][-+]?\d+)?\s*[a-zµ³/%]*', s):
        return True
    return False


def n_steps(r):
    return len(r.get('tool_calls') or [])


def used_retrieve(r):
    return any(t.get('tool') == 'retrieve' for t in (r.get('tool_calls') or []))


def budget_exhausted(r):
    return 'max_steps_exceeded' in (str(r.get('error_msg') or '') + str(r.get('reasoning') or ''))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PARQ)
    open_q = set(df[df.question_type == 'open_ended'].qid)
    calc_q = set(df[df.question_type == 'calculation'].qid)

    def acc(m, arch, qt):
        return df[(df.model == m) & (df.arch == arch) & (df.question_type == qt)].score.mean() * 100

    summary = {}
    rows = []
    for m in MODELS:
        a3, a4 = load(m, A3), load(m, A4)
        oq = [q for q in open_q if q in a3 and q in a4]
        cq = [q for q in calc_q if q in a3 and q in a4]
        open_k_a3 = sum(open_collapsed(a3[q].get('answer')) for q in oq)
        open_k_a4 = sum(open_collapsed(a4[q].get('answer')) for q in oq)
        calc_k_a3 = sum(not is_numeric(a3[q].get('answer')) for q in cq)
        calc_k_a4 = sum(not is_numeric(a4[q].get('answer')) for q in cq)
        rec = {
            'open_acc_A3': round(acc(m, A3, 'open_ended'), 1),
            'open_acc_A4': round(acc(m, A4, 'open_ended'), 1),
            'open_collapse_A3': round(open_k_a3 / len(oq) * 100, 1),
            'open_collapse_A4': round(open_k_a4 / len(oq) * 100, 1),
            # raw counts for binomial confidence intervals (k = collapsed, n = items)
            'open_k_A3': int(open_k_a3), 'open_k_A4': int(open_k_a4), 'open_n': len(oq),
            'calc_k_A3': int(calc_k_a3), 'calc_k_A4': int(calc_k_a4), 'calc_n': len(cq),
            'calc_acc_A3': round(acc(m, A3, 'calculation'), 1),
            'calc_acc_A4': round(acc(m, A4, 'calculation'), 1),
            'calc_nonnum_A3': round(calc_k_a3 / len(cq) * 100, 1),
            'calc_nonnum_A4': round(calc_k_a4 / len(cq) * 100, 1),
            'steps_A3': round(sum(n_steps(a3[q]) for q in cq + oq) / len(cq + oq), 1),
            'steps_A4': round(sum(n_steps(a4[q]) for q in cq + oq) / len(cq + oq), 1),
            'budget_exhausted_A4': round(sum(budget_exhausted(a4[q]) for q in cq + oq) / len(cq + oq) * 100, 1),
            'retrieve_call_rate_A4': round(sum(used_retrieve(a4[q]) for q in cq + oq) / len(cq + oq) * 100, 1),
        }
        summary[m] = rec
        rows.append((m, rec))

    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))

    print("=== A4 sub-additivity mechanism: control-flow collapse on weak models ===\n")
    hdr = f"{'model':13} | open acc A3->A4 | open collapse% A3->A4 | calc acc A3->A4 | calc non-num% A3->A4 | steps A3->A4 | budget-exh% | retrieve-call%"
    print(hdr)
    for m, r in rows:
        print(f"{m:13} | {r['open_acc_A3']:5.1f}->{r['open_acc_A4']:5.1f}    | "
              f"{r['open_collapse_A3']:5.1f}->{r['open_collapse_A4']:5.1f}        | "
              f"{r['calc_acc_A3']:5.1f}->{r['calc_acc_A4']:5.1f}   | "
              f"{r['calc_nonnum_A3']:5.1f}->{r['calc_nonnum_A4']:5.1f}         | "
              f"{r['steps_A3']:4.1f}->{r['steps_A4']:4.1f}  | "
              f"{r['budget_exhausted_A4']:5.1f}      | {r['retrieve_call_rate_A4']:5.1f}")
    print(f"\nsaved -> {OUT/'summary.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
