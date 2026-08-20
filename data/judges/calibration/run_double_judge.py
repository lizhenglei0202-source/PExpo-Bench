#!/usr/bin/env python3
"""Double-judge scoring for the judge-calibration sample.

Scores all 400 blinded answers in judge_inputs.jsonl with BOTH judges
(gpt-5.4-nano and deepseek-v4) using the EXACT prompt and 0-5 rubric from
pexpo_bench/runners/run_open_judge.py. Writes per_row_double_judge.jsonl
(one line per (sample_id, judge); resumable).

Usage:
    python run_double_judge.py [--concurrency 8] [--max-rows N]

NOTE: this script makes API calls and costs money (see README.md estimate).
It has NOT been executed as part of package construction.
"""
from __future__ import annotations
import argparse, json, pathlib, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

PKG = pathlib.Path(__file__).resolve().parent
REPO = pathlib.Path('~/Library/CloudStorage/OneDrive-Personal/macmini/lzl')

# EXACT prompt from pexpo_bench/runners/run_open_judge.py
JUDGE_PROMPT = """You are an expert evaluator for an environmental health science exam.

Score the student's answer against the reference answer on a 0 to 5 scale.

5: Completely correct, covers all key points with accurate detail.
4: Mostly correct, minor omissions or minor imprecision.
3: Partially correct, captures main idea but misses important detail.
2: Weak, shows some understanding but with significant gaps or partial inaccuracies.
1: Mostly incorrect, only tangentially related.
0: Completely wrong, irrelevant, or no substantive answer.

QUESTION: {question}

REFERENCE ANSWER: {gold}

STUDENT ANSWER: {pred}

Respond with a single integer 0 to 5 and nothing else."""

# Same max_tokens settings as run_open_judge.py JUDGE_FOR table:
# deepseek-v4 is a reasoning model and needs >500 tokens; nano emits one token.
JUDGES = {
    'gpt-5.4-nano': 16,
    'deepseek-v4': 600,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', default=str(PKG / 'judge_inputs.jsonl'))
    p.add_argument('--out', default=str(PKG / 'per_row_double_judge.jsonl'))
    p.add_argument('--concurrency', type=int, default=8)
    p.add_argument('--max-rows', type=int, default=None)
    args = p.parse_args()

    from dotenv import load_dotenv
    load_dotenv(REPO / '.env')
    sys.path.insert(0, str(REPO))
    from pexpo_bench.llm_clients import LLMClient

    inputs = [json.loads(l) for l in
              pathlib.Path(args.inputs).read_text().splitlines() if l.strip()]
    if args.max_rows:
        inputs = inputs[:args.max_rows]

    out_file = pathlib.Path(args.out)
    done = set()
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            try:
                r = json.loads(line)
                if r.get('score_0_5', -1) >= 0:
                    done.add((r['sample_id'], r['judge']))
            except Exception:
                pass
    print(f'[init] {len(inputs)} rows x {len(JUDGES)} judges; '
          f'{len(done)} already done')

    clients = {j: LLMClient(j, temperature=0.0, max_tokens=mtok, seed=42)
               for j, mtok in JUDGES.items()}

    tasks = [(r, j) for r in inputs for j in JUDGES
             if (r['sample_id'], j) not in done]
    print(f'[load] {len(tasks)} judgments pending')
    if not tasks:
        print('nothing to do')
        return 0

    def judge_one(task):
        r, jname = task
        base = {'sample_id': r['sample_id'], 'qid': r['qid'],
                'model': r['model'], 'judge': jname}
        try:
            prompt = JUDGE_PROMPT.format(question=r['question'],
                                         gold=r['gold'], pred=r['pred'])
            resp = clients[jname].chat([{'role': 'user', 'content': prompt}])
            m = re.search(r'\d', resp.content)
            score = int(m.group()) if m else 0
            score = max(0, min(5, score))
            return {**base, 'score_0_5': score, 'score': score / 5.0}
        except Exception as e:
            return {**base, 'score_0_5': -1, 'score': 0.0,
                    'error': f'{type(e).__name__}: {str(e)[:200]}'}

    n_done, t0 = 0, time.time()
    with open(out_file, 'a') as fout:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(judge_one, t): t for t in tasks}
            for fut in as_completed(futs):
                fout.write(json.dumps(fut.result(), default=str) + '\n')
                fout.flush()
                n_done += 1
                if n_done % 50 == 0:
                    rate = n_done / max(time.time() - t0, 0.1)
                    eta = (len(tasks) - n_done) / rate / 60
                    print(f' [{n_done:>4}/{len(tasks)}] '
                          f'rate={rate:.1f}/s eta={eta:.1f}min')
    print(f'done: {n_done} judged in {(time.time() - t0) / 60:.1f}min')
    return 0


if __name__ == '__main__':
    sys.exit(main())
