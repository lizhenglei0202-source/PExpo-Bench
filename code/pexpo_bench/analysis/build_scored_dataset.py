import os
"""Build the CANONICAL paper dataset: runs/v4_scored/all_scored_v4_main.parquet.

Phase A cells only (4 models x 5 paper arms, curated bank n=1,027), full -compatible
schema (incl. latency_s, error_msg), open-ended scores from the corrected judge pass
(runs/v4_rerun/_open_judge). This file is THE single source for all figures/tables.
 files (runs/v3_scored/*) are frozen history — never overwritten.
"""
import json, pathlib, sys
import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from scoring import score_row
import scoring as r2
r2._OPEN_JUDGE = {}  # open-ended comes from the judge file below

ROOT = pathlib.Path(os.environ.get("PEXPO_ROOT", "."))
gold = {q["qid"]: q for q in yaml.safe_load((ROOT / "pexpo_bench/samples/pexpo_bench_v3_full.patched_20260811.yaml").read_text())}
judge = {}
for l in (ROOT / "runs/v4_rerun/_open_judge").read_text().splitlines():
    if l.strip():
        r = json.loads(l)
        judge[(r["model"], r["arch"], r["qid"])] = r["score"]

rows = []
for f in (ROOT / "runs/v4_rerun").glob("*/*/run_1.jsonl"):
    model, arch = f.parent.parent.name, f.parent.name
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        gq = gold.get(r.get("qid"))
        if not gq or gq.get("_retired_20260811"):
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
out = ROOT / "runs/v4_scored/all_scored_v4_main.parquet"
df.to_parquet(out)
print(f"{out.name}: {len(df)} rows, {df.groupby(['model','arch']).ngroups} cells, "
      f"open-ended NaN: {int(df[df.question_type=='open_ended'].score.isna().sum())}")
print((df.groupby('arch').score.mean() * 100).round(1).to_string())
