"""Score v7 results: A4, A4+, A2_dense for 3 models. LLM judge for open-ended."""
import os, sys, json, time
sys.path.insert(0, "${HOME}/Desktop/lzl")
from dotenv import load_dotenv
load_dotenv("${HOME}/Desktop/lzl/.env")
from pexpo_bench.evaluation.score import score_tf, score_calc, score_open_llm, _score_open_keyword
from pexpo_bench.llm_clients import LLMClient
import yaml

GOLD = {q["qid"]: q for q in yaml.safe_load(
    open("${HOME}/Desktop/lzl/pexpo_bench/samples/pexpo_bench_v3_release.yaml"))}
print("Init judge...", flush=True)
judge = LLMClient("gpt-4o")

def score_file(rows_path, out_path):
    out_rows = []
    n_judge = 0
    for ln in open(rows_path):
        r = json.loads(ln)
        g = GOLD.get(r["qid"])
        if not g: continue
        qt = g.get("question_type", "")
        ga = str(g.get("answer", ""))
        pa = r.get("answer") or r.get("raw_output", "")
        if qt == "true_false":
            s = score_tf(pa, ga)
        elif qt == "calculation":
            s = score_calc(pa, ga)
        elif qt == "open_ended":
            try:
                s = score_open_llm(g.get("question", ""), str(pa), ga, judge)
                n_judge += 1
            except Exception as e:
                s = _score_open_keyword(str(pa), ga)
                print(f"    judge fail {r['qid']}: {str(e)[:60]}", flush=True)
        else:
            s = 0.0
        r["score"] = s
        r["question_type"] = qt
        r["subdomain"] = g.get("subdomain", "")
        r["difficulty"] = g.get("difficulty", "")
        out_rows.append(r)
    with open(out_path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, default=str) + "\n")
    overall = sum(r["score"] for r in out_rows) / len(out_rows)
    return overall, n_judge

configs = [
    ("gpt4o",     "A4_hybrid"),
    ("gpt4o",     "A4p_hybrid_constrained"),
    ("gpt4omini", "A4_hybrid"),
    ("gpt4omini", "A4p_hybrid_constrained"),
    ("deepseek",  "A4_hybrid"),
    ("deepseek",  "A4p_hybrid_constrained"),
]
STATUS = open("${HOME}/Desktop/lzl/pexpo_bench/runs/exp_v7_fixes/_status.log","w",buffering=1)
STATUS.write("started\n")
for label, arch in configs:
    fp = f"${HOME}/Desktop/lzl/pexpo_bench/runs/exp_v7_fixes/{label}/{arch}/0.jsonl"
    out = f"${HOME}/Desktop/lzl/pexpo_bench/runs/exp_v7_fixes/{label}/{arch}/0.scored.jsonl"
    t0 = time.time()
    acc, n = score_file(fp, out)
    STATUS.write(f"{label}/{arch}: acc={acc:.3f} ({time.time()-t0:.0f}s)\n"); STATUS.flush(); print(f"  {label}/{arch}: acc={acc:.3f}", flush=True)

# A2 dense (sensitivity)
for label in ["gpt4o", "gpt4omini", "deepseek"]:
    fp = f"${HOME}/Desktop/lzl/pexpo_bench/runs/exp_v7_fixes/{label}/_dense/A2_rag/0.jsonl"
    out = fp.replace("/0.jsonl", "/0.scored.jsonl")
    t0 = time.time()
    acc, n = score_file(fp, out)
    STATUS.write(f"{label}/A2_dense: acc={acc:.3f} ({time.time()-t0:.0f}s)\n"); STATUS.flush(); print(f"  {label}/A2_dense: acc={acc:.3f}", flush=True)

STATUS.write("ALL SCORED\n"); STATUS.close(); print("ALL SCORED", flush=True)
