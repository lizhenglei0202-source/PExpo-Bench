"""LLM-judge for open-ended answers.
Uses a cross-family judge (DeepSeek-V4) to score each prediction 0-5 against the gold reference.
Output: per_row_open_judge.jsonl"""
from __future__ import annotations
import argparse, json, pathlib, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

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

def main():
    # load project-root .env (fix 2026-08-17: judge calls previously went out with key
    # 'EMPTY' and failed 401 on every row)
    from dotenv import load_dotenv
    load_dotenv(__import__('pathlib').Path(__file__).resolve().parents[2] / '.env')
    p = argparse.ArgumentParser()
    p.add_argument('--runs', required=True)
    p.add_argument('--bank', default='${HOME}/Desktop/lzl/pexpo_bench/samples/pexpo_bench_v3_release.yaml')
    p.add_argument('--out', required=True)
    p.add_argument('--judge', default='auto', help='auto = cross-family per evaluated model')
    p.add_argument('--concurrency', type=int, default=8)
    p.add_argument('--max-rows', type=int, default=None)
    args = p.parse_args()

    load_dotenv('${HOME}/Desktop/lzl/.env')
    sys.path.insert(0, '${HOME}/Desktop/lzl')
    from pexpo_bench.llm_clients import LLMClient
    import yaml

    runs_dir = pathlib.Path(args.runs)
    out_file = pathlib.Path(args.out); out_file.parent.mkdir(parents=True, exist_ok=True)

    bank = yaml.unsafe_load(open(args.bank).read())
    gold = {q['qid']: q for q in bank if q.get('question_type') == 'open_ended'}
    print(f'[init] {len(gold)} open-ended gold items')

    # Resume: load already done
    done = set()
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r['model'], r['arch'], r['qid']))
            except: pass
    print(f'[init] resuming with {len(done)} already-done')

    # Collect pending
    rows = []
    for jl in sorted(runs_dir.glob('*/*/run_1.jsonl')):
        model = jl.parts[-3]; arch = jl.parts[-2]
        for line in jl.read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            qid = r.get('qid','')
            if qid not in gold: continue
            if (model, arch, qid) in done: continue
            rows.append({
                'model': model, 'arch': arch, 'qid': qid,
                'question': str(gold[qid]['question']),
                'gold': str(gold[qid]['answer'])[:2500],
                'pred': str(r.get('answer') or '')[:2500],
            })
            if args.max_rows and len(rows) >= args.max_rows: break
        if args.max_rows and len(rows) >= args.max_rows: break
    print(f'[load] {len(rows)} judgments pending')
    if not rows:
        print('nothing to do'); return 0

    # Cross-family judges: OpenAI outputs judged by DeepSeek, DeepSeek by OpenAI
    JUDGE_FOR = {
        'gpt-5.4':       ('deepseek-v4', 600),    # DeepSeek reasoning model, needs >500 tok
        'gpt-5.4-native': ('deepseek-v4', 600),
        'gpt-5.4-mini':  ('deepseek-v4', 600),
        'gpt-5.4-nano':  ('deepseek-v4', 600),
        'deepseek-v4':   ('gpt-5.4-nano', 16),    # nano: single-token output, cheap
    }
    if args.judge != 'auto':
        JUDGE_FOR = {k: (args.judge, 600 if 'deepseek' in args.judge else 16) for k in JUDGE_FOR}
    clients = {}
    def get_client(model_eval):
        jmodel, mtok = JUDGE_FOR[model_eval]
        key = (jmodel, mtok)
        if key not in clients:
            clients[key] = LLMClient(jmodel, temperature=0.0, max_tokens=mtok, seed=42)
        return clients[key], jmodel

    def judge_one(r):
        try:
            judge, jname = get_client(r['model'])
            prompt = JUDGE_PROMPT.format(question=r['question'], gold=r['gold'], pred=r['pred'])
            resp = judge.chat([{'role':'user','content':prompt}])
            m = re.search(r'\d', resp.content)
            score = int(m.group()) if m else 0
            score = max(0, min(5, score))
            return {**{k:r[k] for k in ('model','arch','qid')},
                    'judge': jname, 'score_0_5': score, 'score': score/5.0}
        except Exception as e:
            return {**{k:r[k] for k in ('model','arch','qid')},
                    'judge': 'auto', 'score_0_5': -1, 'score': 0.0,
                    'error': f'{type(e).__name__}: {str(e)[:200]}'}

    n_done = 0; t0 = time.time()
    with open(out_file, 'a') as fout:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(judge_one, r): r for r in rows}
            for fut in as_completed(futs):
                out = fut.result()
                fout.write(json.dumps(out, default=str) + '\n')
                fout.flush()
                n_done += 1
                if n_done % 50 == 0:
                    rate = n_done / max(time.time()-t0, 0.1)
                    eta = (len(rows)-n_done) / rate / 60
                    print(f'  [{n_done:>5}/{len(rows)}]  rate={rate:.1f}/s  eta={eta:.1f}min')
    print(f'done: {n_done} judged in {(time.time()-t0)/60:.1f}min')
    return 0

if __name__ == '__main__':
    sys.exit(main())
