""" experiment driver: 3 model × 7 arch × 1004 question × n runs.

Features:
  • Concurrency: ThreadPoolExecutor, default 10 workers per (model, arch)
  • Checkpoint: write one JSONL line per question; resume by scanning existing rows
  • n-run support: separate run_idx output files (run_1.jsonl, run_2.jsonl, ...)
  • Manifest: auto-dumped per run via reproducibility_manifest.py
  • Balance check: pre-flight; abort if any key/endpoint is broken
  • Retry policy: per-question retry ≤2 on transient failures; mark parse_error else

Usage:
    python -m pexpo_bench.runners.run_experiment \\
        --models gpt-5.4 gpt-5.4-nano deepseek-v4 \\
        --archs A0_naive A1_context_eng A2_rag A2p_rag_constrained \\
                A3_agent A4_hybrid A4p_hybrid_constrained \\
        --bank pexpo_bench/samples/pexpo_bench_v3_release.yaml \\
        --out runs/v3_main \\
        --run-idx 1 \\
        --concurrency 10

Resume: just re-invoke same command; already-completed (qid) lines are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

import yaml
from dotenv import load_dotenv


# ==========================================================================
# Per-model concurrency caps (override default --concurrency)
# Based on observed rate-limit tier:
#   gpt-5.4-nano  — 200k TPM, hit at concurrency 5; safe at 2
#   deepseek-v4   — generous TPM
# ==========================================================================
PER_MODEL_CONC = {
    "gpt-5.4":      10,
    "gpt-5.4-nano":  2,
    "deepseek-v4":  10,
}

def _conc_for(model_key: str, default: int) -> int:
    return PER_MODEL_CONC.get(model_key, default)


# ==========================================================================
# Helpers
# ==========================================================================
def load_bank(path: pathlib.Path) -> list[dict]:
    qs = yaml.safe_load(path.read_text())  # unsafe_load executed arbitrary tags (fix 2026-08-12)
    print(f"  loaded {len(qs)} questions from {path}")
    return qs


def existing_qids(jsonl_path: pathlib.Path) -> set[str]:
    if not jsonl_path.exists(): return set()
    out = set()
    for line in jsonl_path.read_text().splitlines():
        try:
            r = json.loads(line)
            if r.get("qid"): out.add(r["qid"])
        except Exception:
            continue
    return out


# ==========================================================================
# Per-question worker
# ==========================================================================
def _run_one_question(runner, q: dict, model_key: str,
                       retry_on_transient: int = 2) -> dict:
    """Run one question through one architecture; catch errors + retry transient.

    Transient failures retried (parse_error with empty raw_output, API errors).
    Real model-behavior failures (max_steps_exceeded) NOT retried — that's a finding.
    """
    last_d = None
    for attempt in range(retry_on_transient + 1):
        try:
            result = runner.run(q)
            d = asdict(result)
            d["_model_key"] = model_key
            d["_question_text"] = q.get("question", "")
            d["_question_type"] = q.get("question_type", "")
            d["_subdomain"] = q.get("subdomain", "")
            d["_difficulty"] = q.get("difficulty", "")
            d["_attempt"] = attempt + 1
            # Decide if this is a transient failure worth retrying:
            #   parse_error=True AND raw_output is empty/very short  → transient
            #   error_msg='max_steps_exceeded'                       → model behavior, KEEP
            is_transient = (
                d.get("parse_error") and
                len((d.get("raw_output") or "").strip()) < 20 and
                (d.get("error_msg") or "") != "max_steps_exceeded"
            )
            if not is_transient:
                return d
            last_d = d
            import time as _t
            _t.sleep(2 ** attempt)  # 1, 2, 4 seconds
        except Exception as e:
            last_d = {
                "qid": q["qid"], "_model_key": model_key,
                "architecture": getattr(runner, "name", "?"),
                "answer": None, "unit": None,
                "reasoning": "", "citations": [], "tool_calls": [],
                "raw_output": "", "retrieved_docs": [],  # schema parity with asdict(Result) rows (fix 2026-08-12)
                "input_tokens": 0, "output_tokens": 0,
                "total_latency_s": 0, "_attempt": attempt + 1,
                "_question_text": q.get("question", ""),
                "_question_type": q.get("question_type", ""),
                "_subdomain": q.get("subdomain", ""),
                "_difficulty": q.get("difficulty", ""),
                "parse_error": True,
                "error_msg": f"{type(e).__name__}: {str(e)[:400]}",
                "_traceback": traceback.format_exc()[:1000],
            }
            import time as _t
            _t.sleep(2 ** attempt)
    return last_d


# ==========================================================================
# Per-(model, arch, run_idx) loop
# ==========================================================================
def run_cell(
    bank: list[dict],
    model_key: str, arch_name: str, run_idx: int,
    out_dir: pathlib.Path,
    concurrency: int = 10,
    temperature: float = 0.3, seed: int = 42,
    max_questions: int | None = None,
    retry_failed: bool = False,
) -> dict:
    """Run one (model, arch, run_idx) cell.  Writes JSONL incrementally."""
    from pexpo_bench.architectures.orchestrator import ARCHITECTURES

    arch_cls = ARCHITECTURES[arch_name]
    arch_dir = out_dir / model_key / arch_name
    arch_dir.mkdir(parents=True, exist_ok=True)
    out_path = arch_dir / f"run_{run_idx}.jsonl"

    if retry_failed and out_path.exists():
        # compact the file to non-failed rows so retried qids don't produce duplicates
        kept = []
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
                if not r.get("parse_error"):
                    kept.append(line)
            except Exception:
                continue
        out_path.write_text("\n".join(kept) + ("\n" if kept else ""))
        print(f"       retry-failed: compacted to {len(kept)} clean rows")

    done_qids = existing_qids(out_path)
    pending = [q for q in bank if q["qid"] not in done_qids]
    if max_questions: pending = pending[:max_questions]

    print(f"\n[cell] model={model_key:18s} arch={arch_name:25s} run={run_idx}")
    print(f"       output: {out_path}")
    print(f"       done: {len(done_qids)}, pending: {len(pending)}, concurrency: {concurrency}")
    if not pending:
        print("       ✓ already complete")
        return {"cell": (model_key, arch_name, run_idx), "n_done": len(done_qids),
                "n_new": 0, "elapsed_s": 0}

    # Instantiate arch: A3/A4 support model_key; A0/A1/A2 do too via base BaseArch
    # If arch class doesn't accept model_key, fall back to no-arg constructor.
    try:
        runner = arch_cls(model_key=model_key,
                          temperature=temperature, seed=seed)
    except TypeError:
        # Older archs may not accept these kwargs
        runner = arch_cls()
        if hasattr(runner, "model_key"):
            runner.model_key = model_key
        if hasattr(runner, "temperature"):
            runner.temperature = temperature

    t0 = time.time()
    n_new = 0
    n_fail = 0
    # Open in append mode; flush per-question for checkpoint safety
    with open(out_path, "a") as out_fp:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(_run_one_question, runner, q, model_key): q
                       for q in pending}
            for i, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                out_fp.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
                out_fp.flush()
                n_new += 1
                if rec.get("parse_error"): n_fail += 1
                if i % 50 == 0 or i == len(pending):
                    elapsed = time.time() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (len(pending) - i) / rate if rate > 0 else 0
                    print(f"       [{i:>4d}/{len(pending)}] "
                          f"rate={rate:.1f} q/s  eta={eta/60:.1f}m  "
                          f"fails={n_fail}")
    elapsed = time.time() - t0
    print(f"       ✓ done {n_new}/{len(pending)} in {elapsed/60:.1f}m  ({n_fail} failures)")
    return {"cell": (model_key, arch_name, run_idx),
            "n_done": len(done_qids), "n_new": n_new,
            "n_fail": n_fail, "elapsed_s": elapsed}


# ==========================================================================
# Top-level
# ==========================================================================
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--bank", required=True, help="Path to question bank YAML")
    p.add_argument("--out", required=True, help="Output dir (e.g. runs/v3_main)")
    p.add_argument("--models", nargs="+", required=True,
                   help="Model keys (e.g. gpt-5.4 gpt-5.4-nano deepseek-v4)")
    p.add_argument("--archs", nargs="+", required=True,
                   help="Architecture names (A0_naive, A3_agent, ...)")
    p.add_argument("--run-idx", type=int, default=1,
                   help="Which run this is (1, 2, or 3). Outputs to run_<idx>.jsonl")
    p.add_argument("--concurrency", type=int, default=10,
                   help="Default concurrency; overridden per-model via PER_MODEL_CONC")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-questions", type=int, default=None,
                   help="(debug) cap questions per cell")
    p.add_argument("--skip-balance-check", action="store_true",
                   help="Don't pre-check API balances (not recommended)")
    p.add_argument("--retry-failed", action="store_true",
                   help="On resume, re-run qids whose recorded row has parse_error=True "
                        "(default keeps them, matching historical behavior)")
    args = p.parse_args(argv)

    # project-root .env, resolved relative to this file (stale Desktop path fix 2026-08-12)
    env_path = pathlib.Path(__file__).resolve().parents[2] / ".env"
    if not load_dotenv(env_path):
        raise SystemExit(f"FATAL: could not load {env_path} — refusing to run with missing API keys")

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-flight: balance check
    if not args.skip_balance_check:
        from pexpo_bench.runners.balance_check import pre_flight_balance_check
        if not pre_flight_balance_check():
            print("\nABORTING — fix API key/balance issues, or rerun with --skip-balance-check")
            return 1

    # Manifest
    from pexpo_bench.analysis.reproducibility_manifest import write_manifest
    manifest_path = write_manifest(
        out_dir=out_dir, run_idx=args.run_idx,
        bank_path=args.bank, models=args.models, archs=args.archs,
        temperature=args.temperature, seed=args.seed,
        concurrency=args.concurrency,
    )
    print(f"\nManifest: {manifest_path}")

    # Load bank
    bank = load_bank(pathlib.Path(args.bank))

    # Run all cells
    summary = []
    for model_key in args.models:
        cell_conc = _conc_for(model_key, args.concurrency)
        if cell_conc != args.concurrency:
            print(f"\n[conc] {model_key}: using concurrency={cell_conc} "
                  f"(per-model override; --concurrency was {args.concurrency})")
        for arch in args.archs:
            res = run_cell(
                bank, model_key, arch, args.run_idx, out_dir,
                concurrency=cell_conc,
                temperature=args.temperature, seed=args.seed,
                max_questions=args.max_questions,
                retry_failed=args.retry_failed,
            )
            summary.append(res)

    # Save summary
    summary_path = out_dir / f"summary_run_{args.run_idx}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary: {summary_path}")
    print("\n=== ALL CELLS DONE ===")
    for s in summary:
        m, a, ri = s["cell"]
        print(f"  {m:18s} {a:25s} run{ri}  +{s['n_new']:>4d} new  ({s.get('n_fail',0)} fails)  "
              f"{s.get('elapsed_s',0)/60:.1f}m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
