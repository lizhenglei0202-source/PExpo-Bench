"""HR judge runner — iterates over all 21k rows, runs claim extraction + KB
entailment via cross-family LLM judge. Outputs per-row HR scores with caching.

Usage:
    python3 -m pexpo_bench.runners.run_hr_judge \\
        --runs runs/v3_main \\
        --out runs/v3_main/_hr \\
        --concurrency 8 \\
        --max-rows 10   # for smoke; remove for full
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from dotenv import load_dotenv

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--qids-file", default=None,
                   help="optional newline-separated qid allowlist; rows outside it are skipped")
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--max-rows", type=int, default=None)
    args = p.parse_args()

    load_dotenv(pathlib.Path(__file__).resolve().parents[2] / ".env")  # fix 2026-08-18: stale Desktop path
    from pexpo_bench.llm_clients import LLMClient
    from pexpo_bench.evaluation.hr_atomic_judge import evaluate_hr_reasoning
    from pexpo_bench.evaluation.judge_dispatch import judge_model_for
    from pexpo_bench.retrieval import Retriever

    runs_dir = pathlib.Path(args.runs)
    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "_cache"; cache_dir.mkdir(exist_ok=True)
    out_file = out_dir / "per_row_hr.jsonl"

    # Load already-processed qids
    done_keys = set()
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            try:
                r = json.loads(line)
                # fix 2026-08-18: zero-claim rows with no recorded error were produced by
                # silently failing extractor calls (provider 402) — treat as NOT done.
                if r.get('n_claims', 0) == 0 and not r.get('error'):
                    continue
                done_keys.add((r['model'], r['arch'], r['qid']))
            except: pass
    print(f"[init] resuming with {len(done_keys)} already-done")

    # Load all per-row data from Phase 1
    print("[load] reading 21k rows from", runs_dir)
    rows = []
    for jsonl in sorted(runs_dir.glob('*/*/run_1.jsonl')):
        for line in jsonl.read_text().splitlines():
            try:
                r = json.loads(line)
                key = (r.get('_model_key',''), r.get('architecture',''), r.get('qid',''))
                if key in done_keys: continue
                rows.append(r)
            except: pass
            if args.max_rows and len(rows) >= args.max_rows: break
        if args.max_rows and len(rows) >= args.max_rows: break
    if args.qids_file:
        allow = set(pathlib.Path(args.qids_file).read_text().split())
        rows = [r for r in rows if r.get("qid") in allow]
        print(f"[load] qid allowlist: {len(allow)} qids -> {len(rows)} rows retained")
    print(f"[load] {len(rows)} rows pending HR judge")
    if not rows: print("nothing to do"); return 0

    # Load retriever (with dedup + low-info filter)
    print("[load] retriever ...")
    retriever = Retriever.load(
        str(pathlib.Path(__file__).resolve().parents[1] / "knowledge_base" / "index"),  # fix 2026-08-18
        embed_model_name="all-MiniLM-L6-v2",
        reranker_name=None, use_bm25=False,
    )
    # fix 2026-08-18b: the embedding/reranker/faiss stack is not thread-safe; the
    # threaded judge workers crashed natively without traceback. Serialize retrieval.
    import threading as _th
    class _LockedRetriever:
        def __init__(self, r): self._r, self._lk = r, _th.Lock()
        def retrieve(self, *a, **k):
            with self._lk: return self._r.retrieve(*a, **k)
        def __getattr__(self, n): return getattr(self._r, n)
    retriever = _LockedRetriever(retriever)

    # Cache judge clients by model_key
    judge_clients: dict[str, LLMClient] = {}
    def get_judge(model_key: str) -> tuple[LLMClient, str]:
        jk = judge_model_for(model_key)
        if jk not in judge_clients:
            judge_clients[jk] = LLMClient(jk, temperature=0.0, max_tokens=512, seed=42)
        return judge_clients[jk], jk

    # Process one row
    def process_one(r: dict) -> dict:
        try:
            mk = r.get('_model_key','')
            judge_client, jk = get_judge(mk)
            hr_res = evaluate_hr_reasoning(
                r, retriever, judge_client, jk, cache_dir=cache_dir,
            )
            return {
                'model': mk, 'arch': r.get('architecture',''),
                'qid': r.get('qid',''),
                'subdomain': r.get('_subdomain',''),
                'question_type': r.get('_question_type',''),
                'judge_model': jk,
                'n_claims': hr_res.n_claims,
                'n_supported': hr_res.n_supported,
                'n_contradicted': hr_res.n_contradicted,
                'n_no_info': hr_res.n_no_info,
                'HR_reasoning': hr_res.HR_reasoning,
            }
        except Exception as e:
            return {
                'model': r.get('_model_key',''),
                'arch': r.get('architecture',''),
                'qid': r.get('qid',''),
                'HR_reasoning': None, 'error': f"{type(e).__name__}: {str(e)[:200]}",
            }

    t0 = time.time()
    n_done = 0
    with open(out_file, 'a') as fp:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {ex.submit(process_one, r): r for r in rows}
            for i, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                fp.write(json.dumps(rec, ensure_ascii=False) + '\n')
                fp.flush()
                n_done += 1
                if i % 100 == 0 or i == len(rows):
                    rate = i / (time.time() - t0)
                    eta = (len(rows)-i) / rate if rate > 0 else 0
                    print(f"  [{i:>5d}/{len(rows)}] rate={rate:.2f} q/s  eta={eta/60:.1f}m")

    print(f"\n✓ HR judge done: {n_done} rows in {(time.time()-t0)/60:.1f}m")
    print(f"  output: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
