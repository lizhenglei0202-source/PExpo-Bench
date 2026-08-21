"""Grounding (HR) statistics for the corrected rerun, computed on the stratified
299-item reference-quoted subsample (seed 42, identical items for all models/archs).

Dedups per_row_hr.jsonl (best row per model×arch×qid, claims>0 preferred), then emits:
  - Table S1 markdown (atomic-claim breakdown incl. legacy ratios) -> stdout + file
  - grounding_subsample block appended to V4_NUMBERS_20260818.json
  - the ranges needed for the two manuscript placeholders -> stdout
"""
import os
import json, pathlib

ROOT = pathlib.Path(os.environ.get("PEXPO_ROOT", "."))
PAPER = ["A0_naive", "A1_context_eng", "A2p_rag_constrained", "A3_agent", "A4p_hybrid_constrained"]
LAB = dict(zip(PAPER, ["A0", "A1", "A2", "A3", "A4"]))
MODELS = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "deepseek-v4"]
MNAME = {"gpt-5.4": "GPT-5.4", "gpt-5.4-mini": "GPT-5.4-mini",
         "gpt-5.4-nano": "GPT-5.4-nano", "deepseek-v4": "DeepSeek-V4"}

sub = set((ROOT / "runs/v4_rerun/_hr/grounding_subsample_qids.txt").read_text().split())
best = {}
for l in (ROOT / "runs/v4_rerun/_hr/per_row_hr.jsonl").read_text().splitlines():
    if not l.strip():
        continue
    r = json.loads(l)
    k = (r.get("model"), r.get("arch"), r.get("qid"))
    cur = best.get(k)
    if cur is None or ((r.get("n_claims", 0) or 0) > 0 and (cur.get("n_claims", 0) or 0) == 0):
        best[k] = r

cells, lines = {}, []
lines.append("| Model | Arch | Items with claims | Claims | SUPPORTED | CONTRADICTED | NO_INFO | Coverage % | Contradiction (adjudicated) % | Legacy strict % | Legacy wide % |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
for m in MODELS:
    for a in PAPER:
        rs = [r for (mm, aa, q), r in best.items() if mm == m and aa == a and q in sub]
        nc = sum(r.get("n_claims", 0) or 0 for r in rs)
        ns = sum(r.get("n_supported", 0) or 0 for r in rs)
        nx = sum(r.get("n_contradicted", 0) or 0 for r in rs)
        ni = sum(r.get("n_no_info", 0) or 0 for r in rs)
        items = sum(1 for r in rs if (r.get("n_claims", 0) or 0) > 0)
        cov = 100 * (ns + nx) / nc if nc else 0.0
        con = 100 * nx / (ns + nx) if (ns + nx) else 0.0
        strict = 100 * nx / nc if nc else 0.0
        wide = 100 * (nx + ni) / nc if nc else 0.0
        cells[(m, a)] = dict(items=items, n_rows=len(rs), claims=nc, sup=ns, contra=nx,
                             no_info=ni, coverage=round(cov, 1), contra_adj=round(con, 1),
                             strict=round(strict, 1), wide=round(wide, 1))
        lines.append(f"| {MNAME[m]} | {LAB[a]} | {items} | {nc} | {ns} | {nx} | {ni} "
                     f"| {cov:.1f} | {con:.1f} | {strict:.1f} | {wide:.1f} |")

table_md = "\n".join(lines)
(ROOT / "article/final/table_S1_grounding_v4.md").write_text(table_md + "\n")
print(table_md)

def rng(vals):
    return (min(vals), max(vals))

gpt = [c for (m, a), c in cells.items() if m != "deepseek-v4"]
ds = [c for (m, a), c in cells.items() if m == "deepseek-v4"]
summary = {
    "design": "stratified subsample, 299 reference-quoted items, seed 42, "
              "proportional over subdomain x question_type, same items all models/archs",
    "cells": {f"{m}|{LAB[a]}": c for (m, a), c in cells.items()},
    "coverage_range_gpt": rng([c["coverage"] for c in gpt]),
    "coverage_range_deepseek": rng([c["coverage"] for c in ds]),
    "contra_adj_range_gpt": rng([c["contra_adj"] for c in gpt]),
    "contra_adj_range_deepseek": rng([c["contra_adj"] for c in ds]),
}
for m in MODELS:
    summary[f"contra_adj_by_arch_{m}"] = {LAB[a]: cells[(m, a)]["contra_adj"] for a in PAPER}
    summary[f"coverage_by_arch_{m}"] = {LAB[a]: cells[(m, a)]["coverage"] for a in PAPER}

vn = ROOT / "article/final/V4_NUMBERS_20260818.json"
d = json.loads(vn.read_text())
d["grounding_subsample"] = summary
vn.write_text(json.dumps(d, indent=1))

md = ROOT / "article/final/V4_NUMBERS_20260818.md"
txt = md.read_text()
marker = "## Grounding (Table S1, stratified 299-item reference-quoted subsample)"
block = (f"\n\n{marker}\n\nDesign: {summary['design']}.\n\n{table_md}\n")
if marker in txt:
    txt = txt[:txt.index(marker)].rstrip() + block
else:
    txt = txt.rstrip() + block
md.write_text(txt)
print("\n--- placeholder inputs ---")
for k in ("coverage_range_gpt", "coverage_range_deepseek",
          "contra_adj_range_gpt", "contra_adj_range_deepseek"):
    print(k, summary[k])
for m in MODELS:
    print(m, "contra by arch:", summary[f"contra_adj_by_arch_{m}"])
print("\nappended grounding_subsample to", vn)
