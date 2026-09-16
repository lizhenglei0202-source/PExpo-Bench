"""Recompute the 20-cell main evaluation from the released trajectories and recorded judge scores. No model/API calls are made. The original data/scored/ files are not overwritten."""
import os
from pexpo_bench.data_schema import configuration_id
import json, pathlib, sys
import pandas as pd
import yaml

from pexpo_bench.analysis.scoring import score_row

ROOT = pathlib.Path(os.environ.get("PEXPO_ROOT", "."))
(ROOT / "analysis_outputs").mkdir(parents=True, exist_ok=True)
gold = {q["qid"]: q for q in yaml.safe_load((ROOT / "data/bank/bank_evaluation_set.yaml").read_text())}
judge = {}
for l in (ROOT / "data/judges/open_ended_judgments.jsonl").read_text().splitlines():
    if l.strip():
        r = json.loads(l)
        judge[(r["model"], configuration_id(r["arch"]), r["qid"])] = r["score"]

rows = []
for f in (ROOT / "data/trajectories/main").glob("*/*/run_1.jsonl"):
    model, arch = f.parent.parent.name, configuration_id(f.parent.name)
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        gq = gold.get(r.get("qid"))
        if not gq:
            continue
        qt = gq["question_type"]
        if qt == "open_ended":
            sc = judge.get((model, arch, r["qid"]))
            if sc is None:
                sc = 0.0 if r.get("parse_error") else float("nan")
        else:
            sc, _ = score_row(r, gq, model, arch)
        rows.append({"model": model, "arch": arch, "qid": r["qid"],
                     "subdomain": gq.get("subdomain"), "question_type": qt,
                     "difficulty": gq.get("difficulty", "medium"), "score": sc,
                     "in_tokens": r.get("input_tokens", 0) or 0,
                     "out_tokens": r.get("output_tokens", 0) or 0,
                     "latency_s": r.get("total_latency_s", 0) or 0,
                     "n_tools": len(r.get("tool_calls") or []),
                     "parse_error": bool(r.get("parse_error")),
                     "error_msg": r.get("error_msg") or ""})

df = pd.DataFrame(rows).drop_duplicates(subset=["model", "arch", "qid"], keep="last")
out = ROOT / "analysis_outputs/results_main_recomputed.parquet"
df.to_parquet(out)
print(f"{out.name}: {len(df)} rows, {df.groupby(['model','arch']).ngroups} cells, "
      f"open-ended NaN: {int(df[df.question_type=='open_ended'].score.isna().sum())}")
print((df.groupby('arch').score.mean() * 100).round(1).to_string())
