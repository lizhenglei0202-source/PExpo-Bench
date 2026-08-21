"""All statistics for the (reframed) manuscript, from the canonical dataset.
Sources: runs/v4_scored/all_scored_v4_main.parquet (Phase A), all_scored_v4.parquet
(factorial B + seeds C), runs/v3_scored/all_scored_v2.parquet (the frozen line, used
ONLY for the before/after exhibit). Grounding (S11/Fig4bc) is appended by the HR step.
Outputs: article/final/V4_NUMBERS_20260818.md + .json
"""
import os
import json, math, pathlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = pathlib.Path(os.environ.get("PEXPO_ROOT", "."))
PAPER = ["A0_naive", "A1_context_eng", "A2p_rag_constrained", "A3_agent", "A4p_hybrid_constrained"]
LAB = dict(zip(PAPER, ["A0", "A1", "A2", "A3", "A4"]))
MODELS = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "deepseek-v4"]
MNAME = {"gpt-5.4": "GPT-5.4", "gpt-5.4-mini": "GPT-5.4-mini", "gpt-5.4-nano": "GPT-5.4-nano", "deepseek-v4": "DeepSeek-V4"}
SUBS = [("S1_exposure_factors", "S1 Exposure factors"), ("S2_microenv_conc", "S2 Microenv conc."),
        ("S3_trajectory_activity", "S3 Trajectory/activity"), ("S4_dosimetry", "S4 Dosimetry"),
        ("S5_health", "S5 Health effects")]

def fmt_p(p):
    if p >= 0.995: return "1.00"
    if p >= 0.10: return f"{p:.2f}"
    if p >= 0.001: return f"{p:.3f}" if p >= 0.01 else f"{p:.4f}".rstrip("0")
    e = int(math.floor(math.log10(p)))
    return f"{p/10**e:.1f} × 10^{e}"

def holm(ps):
    order = np.argsort(ps); out = np.empty(len(ps)); prev = 0.0
    for rank, i in enumerate(order):
        prev = max(prev, min(1.0, (len(ps) - rank) * ps[i])); out[i] = prev
    return out

a = pd.read_parquet(ROOT / "runs/v4_scored/all_scored_v4_main.parquet")
allp = pd.read_parquet(ROOT / "runs/v4_scored/all_scored_v4.parquet")
prior = pd.read_parquet(ROOT / "runs/v3_scored/all_scored_v2.parquet"); prior = prior[~prior.retired]
piv = {m: a[a.model == m].pivot_table(index="qid", columns="arch", values="score") for m in MODELS}
M = {"tag": "corrected-20260818"}
L = ["# Results manifest — 2026-08-18 (canonical: results_main.parquet)", ""]

# cells + cross-model means
cells = {m: {LAB[ar]: float(a[(a.model == m) & (a.arch == ar)].score.mean()) for ar in PAPER} for m in MODELS}
M["cells"] = cells
M["cross_model_means"] = {LAB[ar]: float(np.mean([cells[m][LAB[ar]] for m in MODELS])) for ar in PAPER}
L += ["## Cell accuracies (%)", "", "| Model | A0 | A1 | A2 | A3 | A4 |", "|---|---|---|---|---|---|"]
for m in MODELS:
    L.append(f"| {MNAME[m]} | " + " | ".join(f"{cells[m][x]*100:.1f}" for x in ["A0","A1","A2","A3","A4"]) + " |")
L.append("| **Mean** | " + " | ".join(f"{M['cross_model_means'][x]*100:.1f}" for x in ["A0","A1","A2","A3","A4"]) + " |")

# by question type per model + cross-model
M["by_type"] = {m: {qt: {LAB[ar]: float(a[(a.model==m)&(a.arch==ar)&(a.question_type==qt)].score.mean())
                          for ar in PAPER} for qt in ["calculation","true_false","open_ended"]} for m in MODELS}

# Table S1: within-model contrasts (Holm across 20)
CONTRASTS = [("A0_naive","A1_context_eng"),("A0_naive","A2p_rag_constrained"),("A0_naive","A3_agent"),
             ("A3_agent","A4p_hybrid_constrained"),("A0_naive","A4p_hybrid_constrained")]
s1, ps = [], []
for x, y in CONTRASTS:
    for m in MODELS:
        pp = piv[m][[x, y]].dropna()
        d = pp[y] - pp[x]
        p = 1.0 if (d == 0).all() else wilcoxon(pp[x], pp[y], zero_method="wilcox", method="approx").pvalue
        s1.append({"contrast": f"{LAB[x]} vs {LAB[y]}", "model": m, "p": float(p), "diff_pp": float(d.mean()*100)})
        ps.append(p)
for r, ph in zip(s1, holm(np.array(ps))): r["p_holm"] = float(ph)
M["table_S1"] = s1
L += ["", "## Table S1 (within-model Wilcoxon; Holm across 20)", "",
      "| Contrast | " + " | ".join(MNAME[m] for m in MODELS) + " |", "|---|---|---|---|---|"]
for x, y in CONTRASTS:
    row = [next(r for r in s1 if r["contrast"]==f"{LAB[x]} vs {LAB[y]}" and r["model"]==m) for m in MODELS]
    L.append(f"| {LAB[x]} vs {LAB[y]} | " + " | ".join(f"{fmt_p(r['p'])} ({r['diff_pp']:+.1f} pp)" for r in row) + " |")

# Table S2: between-model at fixed arch (Holm across 30)
pairs = [(x, y) for i, x in enumerate(MODELS) for y in MODELS[i+1:]]
s2, ps2 = [], []
for m1, m2 in pairs:
    for ar in PAPER:
        x = piv[m1][ar].dropna(); y = piv[m2][ar].dropna()
        idx = x.index.intersection(y.index); d = y[idx]-x[idx]
        p = 1.0 if (d==0).all() else wilcoxon(x[idx], y[idx], zero_method="wilcox", method="approx").pvalue
        s2.append({"pair": f"{MNAME[m1]} vs {MNAME[m2]}", "arch": LAB[ar], "p": float(p), "diff_pp": float(d.mean()*100)})
        ps2.append(p)
for r, ph in zip(s2, holm(np.array(ps2))): r["p_holm"] = float(ph)
M["table_S2"] = s2
L += ["", "## Table S2 (between-model at fixed arch; Holm across 30)", "",
      "| Pair | A0 | A1 | A2 | A3 | A4 |", "|---|---|---|---|---|---|"]
for m1, m2 in pairs:
    row = [next(r for r in s2 if r["pair"]==f"{MNAME[m1]} vs {MNAME[m2]}" and r["arch"]==LAB[ar]) for ar in PAPER]
    L.append(f"| {MNAME[m1]} vs {MNAME[m2]} | " + " | ".join(f"{fmt_p(r['p'])} ({r['diff_pp']:+.1f})" for r in row) + " |")

# Fig5 contrasts: A4-A3 with bootstrap CI
rng = np.random.default_rng(42); f5 = []
for m in MODELS:
    pp = piv[m][["A3_agent","A4p_hybrid_constrained"]].dropna()
    d = (pp.A4p_hybrid_constrained - pp.A3_agent).values
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
    p = wilcoxon(pp.A3_agent, pp.A4p_hybrid_constrained, zero_method="wilcox", method="approx").pvalue
    f5.append({"model": m, "diff_pp": float(d.mean()*100), "ci_lo": float(np.percentile(boots,2.5)*100),
               "ci_hi": float(np.percentile(boots,97.5)*100), "p": float(p), "n": int(len(d))})
for r, ph in zip(f5, holm(np.array([r["p"] for r in f5]))): r["p_holm"] = float(ph)
M["fig5_A4_vs_A3"] = f5
L += ["", "## A4−A3 focal contrasts (bootstrap CI, Holm across 4)", ""]
for r in f5:
    L.append(f"- {MNAME[r['model']]}: {r['diff_pp']:+.1f} pp (CI {r['ci_lo']:+.1f} to {r['ci_hi']:+.1f}), p={fmt_p(r['p'])}, p_Holm={fmt_p(r['p_holm'])}")

# factorial (phase B) — nano + deepseek, calc stream, deltas vs A3
fact = allp[allp.phase=="B"]; calcA = allp[(allp.phase=="A") & (allp.question_type=="calculation")]
ARMS = [("A3","A","A3_agent"),("+R","B","fA3_R"),("+P","B","fA3_P"),("+B","B","fA3_B"),
        ("+R+P","B","fA3_RP"),("+R+B","B","A4_hybrid"),("+P+B","B","fA3_PB"),("A4 (+R+P+B)","A","A4p_hybrid_constrained")]
M["factorial"] = {}
L += ["", "## Factorial (calc stream n=361): accuracy % (delta vs A3)", "",
      "| Arm | GPT-5.4-nano | DeepSeek-V4 |", "|---|---|---|"]
for label, ph, ar in ARMS:
    row = []
    for m in ["gpt-5.4-nano","deepseek-v4"]:
        qids = set(fact[fact.model==m].qid.unique())
        src = calcA if ph=="A" else fact
        acc = float(src[(src.model==m)&(src.arch==ar)&(src.qid.isin(qids))].score.mean()*100)
        M["factorial"].setdefault(m, {})[label] = acc
        base = M["factorial"][m]["A3"]
        row.append(f"{acc:.1f} ({acc-base:+.1f})")
    L.append(f"| {label} | " + " | ".join(row) + " |")

# seeds (phase C) — objective subset
c = allp[(allp.phase=="C") & (allp.question_type!="open_ended")]
M["seeds"] = {}
L += ["", "## Seeds 43-45 (objective subset n=545): A4−A3 pp", "",
      "| Model | 42 (main) | 43 | 44 | 45 | mean ± SD |", "|---|---|---|---|---|---|"]
for m in ["gpt-5.4-nano","deepseek-v4"]:
    obj = a[(a.model==m)&(a.question_type!="open_ended")].pivot_table(index="qid",columns="arch",values="score")[["A3_agent","A4p_hybrid_constrained"]].dropna()
    main = float((obj.A4p_hybrid_constrained-obj.A3_agent).mean()*100)
    ds = []
    for s in [43,44,45]:
        pp = c[(c.model==m)&(c.seed==s)].pivot_table(index="qid",columns="arch",values="score")[["A3_agent","A4p_hybrid_constrained"]].dropna()
        ds.append(float((pp.A4p_hybrid_constrained-pp.A3_agent).mean()*100))
    M["seeds"][m] = {"main": main, "seeds": ds, "mean": float(np.mean(ds)), "sd": float(np.std(ds, ddof=1))}
    L.append(f"| {MNAME[m]} | {main:+.2f} | " + " | ".join(f"{d:+.2f}" for d in ds) + f" | {np.mean(ds):+.2f} ± {np.std(ds, ddof=1):.2f} |")

# before/after exhibit ( vs )
M["before_after"] = {}
L += ["", "## Before/after exhibit (original campaign, defective env -> corrected rerun)", "",
      "| Model | " + " | ".join(f"{LAB[ar]}" for ar in PAPER) + " |", "|---|---|---|---|---|---|"]
for m in MODELS:
    row = []
    for ar in PAPER:
        x2 = float(prior[(prior.model==m)&(prior.arch==ar)].score.mean()*100)
        x4 = cells[m][LAB[ar]]*100
        M["before_after"].setdefault(m, {})[LAB[ar]] = {"original": x2, "corrected": x4}
        row.append(f"{x2:.1f}→{x4:.1f}")
    L.append(f"| {MNAME[m]} | " + " | ".join(row) + " |")
M["format_collapse"] = {"nano_A4_original": 63.9, "nano_A4_corrected": 9.8, "nano_A3_corrected": 9.5}

# sub-domain tables
M["subdomain"] = {m: {sd: {LAB[ar]: float(a[(a.model==m)&(a.arch==ar)&(a.subdomain==sd)].score.mean())
                            for ar in PAPER} for sd, _ in SUBS} for m in MODELS}
for m in MODELS:
    L += ["", f"## Sub-domain × arch: {MNAME[m]}", "", "| Sub-domain | A0 | A1 | A2 | A3 | A4 |", "|---|---|---|---|---|---|"]
    for sd, sdlab in SUBS:
        L.append(f"| {sdlab} | " + " | ".join(f"{M['subdomain'][m][sd][x]:.3f}" for x in ["A0","A1","A2","A3","A4"]) + " |")

# costs ( tokens × registry prices)
PRICE = {"gpt-5.4": (2.5, 15.0), "gpt-5.4-mini": (0.25, 2.0), "gpt-5.4-nano": (0.20, 1.25), "deepseek-v4": (0.14, 0.28)}
M["cost"] = {}
L += ["", "## Cost per 100 questions (corrected-rerun tokens × 2026-08 registry prices, USD)", ""]
for m in MODELS:
    parts = []
    for ar in PAPER:
        d = a[(a.model==m)&(a.arch==ar)]
        c100 = float((d.in_tokens.mean()*PRICE[m][0] + d.out_tokens.mean()*PRICE[m][1]) / 1e6 * 100)
        M["cost"][f"{m}|{LAB[ar]}"] = c100
        parts.append(f"{LAB[ar]} {c100:.3f}")
    L.append(f"- {MNAME[m]}: " + ", ".join(parts))

(ROOT / "article/final/V4_NUMBERS_20260818.json").write_text(json.dumps(M, indent=1))
(ROOT / "article/final/V4_NUMBERS_20260818.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L[:40]))
print(f"\n... saved V4_NUMBERS_20260818.md/.json ({len(L)} lines)")
