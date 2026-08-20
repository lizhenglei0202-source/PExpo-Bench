"""Classify retrieval-induced calculation failures (A3-correct, A4-wrong) by
failure mechanism, with emphasis on scenario-parameter displacement.

Cross-family judge (OpenAI outputs -> DeepSeek; DeepSeek outputs -> GPT-5.4-nano),
matching the paper's judge-dispatch policy. Two passes:
  1. classify mechanism + displaced flag
  2. for displaced==true, a skeptical refutation pass to drop false positives

Input : runs/v3_main/_displacement/payloads.json
Output: runs/v3_main/_displacement/per_row.jsonl
"""
from __future__ import annotations
import json, re, sys, time, pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

ROOT = pathlib.Path('${HOME}/Desktop/lzl')
PAY = ROOT / (sys.argv[1] if len(sys.argv) > 1 else 'runs/v3_main/_displacement/payloads.json')
OUT = ROOT / (sys.argv[2] if len(sys.argv) > 2 else 'runs/v3_main/_displacement/per_row.jsonl')

JUDGE_FOR = {'gpt-5.4': 'deepseek-v4', 'gpt-5.4-mini': 'deepseek-v4',
             'gpt-5.4-nano': 'deepseek-v4', 'deepseek-v4': 'gpt-5.4-nano'}

CLASSIFY = """You are auditing why a tool-using AI got an environmental-health CALCULATION wrong.

The AI had a retrieve() tool over an EPA/WHO/IRIS handbook knowledge base plus a calculator. On this item the SAME agent WITHOUT retrieval answered correctly, but WITH retrieval it answered wrong. Your job is to classify the failure mechanism.

QUESTION (contains the scenario-specific parameters the model should use):
{question}

CORRECT ANSWER: {gold} {unit}
MODEL'S WRONG ANSWER: {model_answer}
MODEL TOOL-CALL TRACE (lookups, retrieved values, calculator inputs):
{trace}

Classify into exactly ONE mechanism:
- "displacement": the model used a generic handbook/default value (e.g. standard adult 80 or 70 kg body weight, 2.0 L/day water intake, 30-year exposure duration, 16 or 20 m3/day inhalation, 70-year averaging) IN PLACE OF a different value the QUESTION explicitly gave for that same quantity.
- "wrong_formula": used a different/incorrect formula or method, not a parameter swap.
- "arithmetic_error": correct inputs and formula but a math/unit-conversion slip.
- "format_error": did not return a usable number (e.g. answered True/False, refused, empty).
- "retrieval_distraction": derailed by retrieved text but NOT a clean default-for-scenario swap.
- "other": none of the above / cannot tell.

Return ONLY JSON: {{"mechanism":"<one>","displaced":<true|false>,"scenario_param":"<param and the scenario value the question gave, or none>","default_used":"<the default value the model substituted, or none>","evidence":"<=200 chars from the trace>","confidence":"high|medium|low"}}
displaced must be true only for mechanism "displacement"."""

REFUTE = """Skeptically re-examine a claim that an AI calculation failed due to SCENARIO-PARAMETER DISPLACEMENT (using a handbook default instead of a value the question explicitly gave).

QUESTION:
{question}
CORRECT ANSWER: {gold} {unit}
MODEL WRONG ANSWER: {model_answer}
TRACE:
{trace}

CLAIMED displacement: param={scenario_param}; default_used={default_used}; evidence={evidence}

Try to REFUTE this. The claim is only confirmed if ALL hold:
1. the question genuinely supplies a specific value for that quantity,
2. the model used a DIFFERENT generic/default value instead,
3. that swap plausibly explains the wrong answer.
Default to refuted if the question did not actually give that parameter, or the trace does not show a default replacing it, or the error is really arithmetic/formula/format.

Return ONLY JSON: {{"confirmed":<true|false>,"reason":"<=200 chars"}}"""


def parse_json(txt):
    m = re.search(r'\{.*\}', txt or '', re.S)
    if not m: return None
    try: return json.loads(m.group())
    except Exception:
        try: return json.loads(m.group().replace("'", '"'))
        except Exception: return None


def main():
    load_dotenv(ROOT / '.env')
    from pexpo_bench.llm_clients import LLMClient
    payloads = json.loads(PAY.read_text())
    print(f"[init] {len(payloads)} retrieval-induced failures to classify")

    clients = {}
    def judge_for(model):
        jk = JUDGE_FOR[model]
        if jk not in clients:
            clients[jk] = LLMClient(jk, temperature=0.0, max_tokens=700, seed=42)
        return clients[jk], jk

    def one(p):
        j, jk = judge_for(p['model'])
        c = CLASSIFY.format(question=p['question'], gold=p['gold'], unit=p['unit'],
                            model_answer=p['model_answer'], trace=p['trace'] or '(no trace recorded)')
        cls = parse_json(j.chat([{'role': 'user', 'content': c}]).content) or \
              {'mechanism': 'other', 'displaced': False, 'confidence': 'low', 'evidence': 'parse_fail'}
        rec = {'model': p['model'], 'qid': p['qid'], 'judge': jk, **cls, 'confirmed': None}
        if cls.get('displaced'):
            r = REFUTE.format(question=p['question'], gold=p['gold'], unit=p['unit'],
                              model_answer=p['model_answer'], trace=p['trace'] or '(none)',
                              scenario_param=cls.get('scenario_param', ''),
                              default_used=cls.get('default_used', ''), evidence=cls.get('evidence', ''))
            v = parse_json(j.chat([{'role': 'user', 'content': r}]).content) or {'confirmed': False, 'reason': 'parse_fail'}
            rec['confirmed'] = bool(v.get('confirmed'))
            rec['refute_reason'] = v.get('reason', '')
        return rec

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(one, p): p for p in payloads}
        for n, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if n % 20 == 0:
                print(f"  [{n}/{len(payloads)}] {(time.time()-t0)/60:.1f}min")
    OUT.write_text('\n'.join(json.dumps(r) for r in results) + '\n')
    print(f"[done] {len(results)} judged in {(time.time()-t0)/60:.1f}min -> {OUT}")

    # aggregate
    import collections
    by_model = collections.defaultdict(lambda: collections.Counter())
    disp_conf = collections.defaultdict(int)
    for r in results:
        by_model[r['model']]['n'] += 1
        by_model[r['model']][r.get('mechanism', 'other')] += 1
        if r.get('displaced') and r.get('confirmed'):
            disp_conf[r['model']] += 1
    print("\n=== mechanism breakdown of retrieval-induced calc failures ===")
    allc = collections.Counter()
    for m, c in by_model.items():
        n = c['n']
        print(f"{m:14} n={n:3}  displacement(confirmed)={disp_conf[m]:3} ({disp_conf[m]/n*100:.0f}%)  "
              + " ".join(f"{k}={c[k]}" for k in c if k != 'n'))
        for k in c:
            if k != 'n': allc[k] += c[k]
    N = sum(c['n'] for c in by_model.values()); D = sum(disp_conf.values())
    print(f"\nOVERALL n={N}  confirmed displacement={D} ({D/N*100:.0f}%)")
    print("mechanism totals:", dict(allc))
    return 0


if __name__ == '__main__':
    sys.exit(main())
