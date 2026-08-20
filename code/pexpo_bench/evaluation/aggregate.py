"""Aggregate per-question results into the tables / figures reported in the paper.

Input:  runs/<arch>/<repeat>.jsonl (per-question Result records)
        pexpo_bench/samples/*.yaml (gold)

Output: runs/_agg/metrics_by_arch.parquet
        runs/_agg/metrics_by_arch_subdomain.parquet
        runs/_agg/failure_modes.parquet
        runs/_agg/pareto.parquet
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import pandas as pd
import yaml

from pexpo_bench.evaluation.metrics import (
    accuracy, instruction_following_rate, hallucination_rate,
    retrieval_precision_recall, tool_use_f1, tool_sequence_edit_distance,
    grounding_rate, physical_consistency, multi_hop_success,
    token_cost, latency_stats, self_consistency, classify_failure,
)

# Pricing ($/1M tokens) lookup — keep in sync with llm_clients.MODEL_REGISTRY
PRICE_TABLE = {
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "qwen2.5-72b": (0.0, 0.0),
    "deepseek-v3": (0.27, 1.1),
}


def usd(rs: list[dict], model_key: str) -> float:
    p_in, p_out = PRICE_TABLE.get(model_key, (0.0, 0.0))
    return sum(r.get("input_tokens", 0) * p_in + r.get("output_tokens", 0) * p_out
               for r in rs) / 1_000_000


def load_gold(samples_dir: pathlib.Path) -> dict:
    all_gold = {}
    for f in samples_dir.glob("*.yaml"):
        for item in yaml.safe_load(f.read_text()) or []:
            all_gold[item["qid"]] = item
    return all_gold


def load_runs(runs_dir: pathlib.Path) -> dict:
    """Return {arch: {repeat_id: [results...]}}."""
    out: dict[str, dict[int, list[dict]]] = defaultdict(dict)
    for arch_dir in runs_dir.iterdir():
        if not arch_dir.is_dir() or arch_dir.name.startswith("_"):
            continue
        for jf in arch_dir.glob("*.jsonl"):
            rid = int(jf.stem) if jf.stem.isdigit() else 0
            out[arch_dir.name][rid] = [json.loads(l) for l in jf.read_text().splitlines() if l.strip()]
    return out


def main(runs_dir: str, samples_dir: str, out_dir: str, model_key: str) -> None:
    runs = load_runs(pathlib.Path(runs_dir))
    gold_map = load_gold(pathlib.Path(samples_dir))
    out_dir_p = pathlib.Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    rows_by_arch = []
    rows_by_arch_sub = []
    failure_rows = []

    for arch, repeats in runs.items():
        # Use repeat 0 for per-question metrics; all repeats for self-consistency
        r0 = repeats[min(repeats)]
        aligned_gold = [gold_map[r["qid"]] for r in r0 if r["qid"] in gold_map]
        aligned_res = [r for r in r0 if r["qid"] in gold_map]

        # Per-architecture summary
        rows_by_arch.append({
            "arch": arch,
            "accuracy_all": accuracy(aligned_res, aligned_gold),
            "IF_rate": instruction_following_rate(aligned_res),
            "HR_open": hallucination_rate(aligned_res, aligned_gold),
            "tool_F1": sum(tool_use_f1(r, g) for r, g in zip(aligned_res, aligned_gold)) / len(aligned_res),
            "physical_ok": sum(physical_consistency(r, g) for r, g in zip(aligned_res, aligned_gold)) / len(aligned_res),
            "ret_prec@5": _mean_safe(retrieval_precision_recall(r, g)[0] for r, g in zip(aligned_res, aligned_gold)),
            "ret_recall@5": _mean_safe(retrieval_precision_recall(r, g)[1] for r, g in zip(aligned_res, aligned_gold)),
            "grounding": _mean_safe(grounding_rate(r, g) for r, g in zip(aligned_res, aligned_gold)),
            "multi_hop": sum(multi_hop_success(r, g) for r, g in zip(aligned_res, aligned_gold)) / len(aligned_res),
            "self_consistency": self_consistency(list(repeats.values()), aligned_gold) if len(repeats) > 1 else float("nan"),
            **{f"lat_{k}": v for k, v in latency_stats(aligned_res).items()},
            **{f"tok_{k}": v for k, v in token_cost(aligned_res).items()},
            "usd": usd(aligned_res, model_key),
        })

        # Per-subdomain breakdown
        per_sub: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
        for r, g in zip(aligned_res, aligned_gold):
            per_sub[g["subdomain"]][0].append(r)
            per_sub[g["subdomain"]][1].append(g)
        for sub, (rs, gs) in per_sub.items():
            rows_by_arch_sub.append({
                "arch": arch, "subdomain": sub, "n": len(rs),
                "accuracy": accuracy(rs, gs),
                "IF": instruction_following_rate(rs),
                "HR": hallucination_rate(rs, gs),
            })

        # Failure modes
        for r, g in zip(aligned_res, aligned_gold):
            mode = classify_failure(r, g)
            if mode:
                failure_rows.append({"arch": arch, "qid": g["qid"],
                                     "subdomain": g["subdomain"],
                                     "difficulty": g.get("difficulty"),
                                     "failure": mode})

    pd.DataFrame(rows_by_arch).to_parquet(out_dir_p / "metrics_by_arch.parquet")
    pd.DataFrame(rows_by_arch_sub).to_parquet(out_dir_p / "metrics_by_arch_subdomain.parquet")
    pd.DataFrame(failure_rows).to_parquet(out_dir_p / "failure_modes.parquet")
    pd.DataFrame([{"arch": r["arch"], "usd": r["usd"], "accuracy": r["accuracy_all"]}
                  for r in rows_by_arch]).to_parquet(out_dir_p / "pareto.parquet")
    print(f"[agg] wrote tables to {out_dir_p}")


def _mean_safe(it):
    vals = [v for v in it if v == v]  # drop NaN
    return sum(vals) / len(vals) if vals else float("nan")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="pexpo_bench/runs")
    ap.add_argument("--samples", default="pexpo_bench/samples")
    ap.add_argument("--out", default="pexpo_bench/runs/_agg")
    ap.add_argument("--model_key", default="gpt-4o")
    args = ap.parse_args()
    main(args.runs, args.samples, args.out, args.model_key)
