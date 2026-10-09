"""Compute statistics for the manuscript from the released main, factorial and seed-replication datasets. Outputs are written to analysis_outputs/."""
import os
from pexpo_bench.data_schema import configuration_id, record_id
import json, math, pathlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = pathlib.Path(os.environ.get("PEXPO_ROOT", "."))
(ROOT / "analysis_outputs").mkdir(parents=True, exist_ok=True)
PAPER = ["A0", "A1", "A2", "A3", "A4"]
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

a = pd.read_parquet(ROOT / "data/scored/results_main.parquet")
allp = pd.read_parquet(ROOT / "data/scored/results_all_phases.parquet")
a["arch"] = a["arch"].map(configuration_id)
allp["arch"] = allp["arch"].map(configuration_id)
piv = {m: a[a.model == m].pivot_table(index="qid", columns="arch", values="score") for m in MODELS}
M = {"release": "publication-snapshot-20261009"}
L = ["# Results for the manuscript experiment", ""]

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

# Table S2: within-model contrasts (Holm across 20)
CONTRASTS = [("A0","A1"),("A0","A2"),("A0","A3"),
             ("A3","A4"),("A0","A4")]
s1, ps = [], []
for x, y in CONTRASTS:
    for m in MODELS:
        pp = piv[m][[x, y]].dropna()
        d = pp[y] - pp[x]
        p = 1.0 if (d == 0).all() else wilcoxon(pp[x], pp[y], zero_method="wilcox", method="approx").pvalue
        s1.append({"contrast": f"{LAB[x]} vs {LAB[y]}", "model": m, "p": float(p), "diff_pp": float(d.mean()*100)})
        ps.append(p)
for r, ph in zip(s1, holm(np.array(ps))): r["p_holm"] = float(ph)
M["within_model_contrasts"] = s1
L += ["", "## Table S2 (within-model Wilcoxon; Holm across 20)", "",
      "| Contrast | " + " | ".join(MNAME[m] for m in MODELS) + " |", "|---|---|---|---|---|"]
for x, y in CONTRASTS:
    row = [next(r for r in s1 if r["contrast"]==f"{LAB[x]} vs {LAB[y]}" and r["model"]==m) for m in MODELS]
    L.append(f"| {LAB[x]} vs {LAB[y]} | " + " | ".join(f"{fmt_p(r['p'])} ({r['diff_pp']:+.1f} pp)" for r in row) + " |")

# Table S3: between-model at fixed arch (Holm across 30)
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
M["between_model_contrasts"] = s2
L += ["", "## Table S3 (between-model at fixed arch; Holm across 30)", "",
      "| Pair | A0 | A1 | A2 | A3 | A4 |", "|---|---|---|---|---|---|"]
for m1, m2 in pairs:
    row = [next(r for r in s2 if r["pair"]==f"{MNAME[m1]} vs {MNAME[m2]}" and r["arch"]==LAB[ar]) for ar in PAPER]
    L.append(f"| {MNAME[m1]} vs {MNAME[m2]} | " + " | ".join(f"{fmt_p(r['p'])} ({r['diff_pp']:+.1f})" for r in row) + " |")

# A4/A3 contrasts: A4-A3 with bootstrap CI
rng = np.random.default_rng(42); f5 = []
for m in MODELS:
    pp = piv[m][["A3","A4"]].dropna()
    d = (pp.A4 - pp.A3).values
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
    p = wilcoxon(pp.A3, pp.A4, zero_method="wilcox", method="approx").pvalue
    f5.append({"model": m, "diff_pp": float(d.mean()*100), "ci_lo": float(np.percentile(boots,2.5)*100),
               "ci_hi": float(np.percentile(boots,97.5)*100), "p": float(p), "n": int(len(d))})
for r, ph in zip(f5, holm(np.array([r["p"] for r in f5]))): r["p_holm"] = float(ph)
M["a4_vs_a3"] = f5
L += ["", "## A4−A3 focal contrasts (bootstrap CI, Holm across 4)", ""]
for r in f5:
    L.append(f"- {MNAME[r['model']]}: {r['diff_pp']:+.1f} pp (CI {r['ci_lo']:+.1f} to {r['ci_hi']:+.1f}), p={fmt_p(r['p'])}, p_Holm={fmt_p(r['p_holm'])}")

# Factorial experiment — all four base models calc stream, deltas vs A3
fact = allp[allp.phase=="B"]; calcA = allp[(allp.phase=="A") & (allp.question_type=="calculation")]
ARMS = [("A3","A","A3"),("+R","B","F100"),("+P","B","F010"),("+B","B","F001"),
        ("+R+P","B","F110"),("+R+B","B","F101"),("+P+B","B","F011"),("A4 (+R+P+B)","A","A4")]
M["factorial"] = {}
L += ["", "## Factorial (calc stream n=361): accuracy % (delta vs A3)", "",
      "| Arm | " + " | ".join(MNAME[m] for m in MODELS) + " |", "|---|---|---|---|---|"]
for label, ph, ar in ARMS:
    row = []
    for m in MODELS:
        qids = set(fact[fact.model==m].qid.unique())
        src = calcA if ph=="A" else fact
        acc = float(src[(src.model==m)&(src.arch==ar)&(src.qid.isin(qids))].score.mean()*100)
        M["factorial"].setdefault(m, {})[label] = acc
        base = M["factorial"][m]["A3"]
        row.append(f"{acc:.1f} ({acc-base:+.1f})")
    L.append(f"| {label} | " + " | ".join(row) + " |")

# Seed-replication experiment — objective subset (545 items, every configuration); seed 42 = main grid, seeds 43-45 = repeats
c = allp[(allp.phase=="C") & (allp.question_type!="open_ended")]
SEEDS = [43, 44, 45]
def _piv(df): return df.pivot_table(index="qid", columns="arch", values="score")
M["seeds_single_arm"] = {}
L += ["", "## Seeds: single-arm accuracy % (objective n=545); seed 42 = main grid; SD over seeds 43-45", "",
      "| Model | Arm | 42 (main) | 43 | 44 | 45 | mean 43-45 | SD 43-45 | main − mean | parse-error % 43/44/45 |", "|---|---|---|---|---|---|---|---|---|---|"]
for m in MODELS:
    for ar in [x for x in PAPER if ((c.model==m)&(c.arch==x)).any()]:
        main = float(a[(a.model==m)&(a.arch==ar)&(a.question_type!="open_ended")].score.mean()*100)
        vals = [float(c[(c.model==m)&(c.seed==s)&(c.arch==ar)].score.mean()*100) for s in SEEDS]
        pe = [float(c[(c.model==m)&(c.seed==s)&(c.arch==ar)].parse_error.mean()*100) for s in SEEDS]
        M["seeds_single_arm"][f"{m}|{LAB[ar]}"] = {"main": main, "seeds": vals, "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)),
                                                   "parse_error_pct": pe, "n_items": int(c[(c.model==m)&(c.seed==SEEDS[0])&(c.arch==ar)].shape[0])}
        L.append(f"| {MNAME[m]} | {LAB[ar]} | {main:.2f} | " + " | ".join(f"{v:.2f}" for v in vals)
                 + f" | {np.mean(vals):.2f} | {np.std(vals, ddof=1):.2f} | {main-np.mean(vals):+.2f} | " + "/".join(f"{v:.1f}" for v in pe) + " |")

M["seeds"] = {}
SEED_CONTRASTS = [("A1","A0"),("A2","A0"),("A3","A0"),("A4","A0"),("A3","A2"),("A4","A3")]
L += ["", "## Seeds: paired contrasts pp (objective n=545); seed 42 = main grid", "",
      "| Model | Contrast | 42 (main) | 43 | 44 | 45 | mean ± SD (43-45) | same sign in all repeats |", "|---|---|---|---|---|---|---|---|"]
for m in MODELS:
    pm = _piv(a[(a.model==m)&(a.question_type!="open_ended")])
    for x, y in SEED_CONTRASTS:
        if not all(((c.model==m)&(c.arch==z)).any() for z in (x, y)): continue
        main = float((pm[x]-pm[y]).dropna().mean()*100)
        ds = []
        for s in SEEDS:
            pp = _piv(c[(c.model==m)&(c.seed==s)])[[x, y]].dropna(); ds.append(float((pp[x]-pp[y]).mean()*100))
        stable = bool(all(np.sign(d) == np.sign(ds[0]) and d != 0 for d in ds))
        M["seeds"].setdefault(m, {})[f"{x}-{y}"] = {"main": main, "seeds": ds, "mean": float(np.mean(ds)), "sd": float(np.std(ds, ddof=1)), "sign_stable": stable}
        L.append(f"| {MNAME[m]} | {x}−{y} | {main:+.2f} | " + " | ".join(f"{d:+.2f}" for d in ds)
                 + f" | {np.mean(ds):+.2f} ± {np.std(ds, ddof=1):.2f} | {'yes' if stable else 'no'} |")

# Seed-replication experiment — A3/A4 on the full evaluation set (objective items plus the 472 distributed open-ended
# repeat items); every open-ended score in this block comes from the same judge session as the main grid.
cf = allp[(allp.phase=="C") & (allp.arch.isin(["A3", "A4"]))]
if (cf.question_type=="open_ended").any():
    full_items = sorted(set(cf.qid)); af = a[a.qid.isin(full_items)]
    M["seeds_full_set"] = {"n_items": len(full_items), "single_arm": {}, "A4-A3": {}, "cross_model": {}}
    L += ["", f"## Seeds: A3/A4 single-arm accuracy % on the full evaluation set (n={len(full_items)}); seed 42 = main grid", "",
          "| Model | Arm | 42 (main) | 43 | 44 | 45 | mean 43-45 | SD 43-45 | main − mean |", "|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        for ar in ["A3", "A4"]:
            main = float(af[(af.model==m)&(af.arch==ar)].score.mean()*100)
            vals = [float(cf[(cf.model==m)&(cf.seed==s)&(cf.arch==ar)].score.mean()*100) for s in SEEDS]
            M["seeds_full_set"]["single_arm"][f"{m}|{LAB[ar]}"] = {"main": main, "seeds": vals, "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))}
            L.append(f"| {MNAME[m]} | {LAB[ar]} | {main:.2f} | " + " | ".join(f"{v:.2f}" for v in vals)
                     + f" | {np.mean(vals):.2f} | {np.std(vals, ddof=1):.2f} | {main-np.mean(vals):+.2f} |")
    L += ["", f"## Seeds: paired A4−A3 contrast pp on the full evaluation set (n={len(full_items)}); seed 42 = main grid", "",
          "| Model | 42 (main) | 43 | 44 | 45 | mean ± SD (43-45) | same sign in all repeats |", "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        pm = _piv(af[af.model==m]); main = float((pm["A4"]-pm["A3"]).dropna().mean()*100); ds = []
        for s in SEEDS:
            pp = _piv(cf[(cf.model==m)&(cf.seed==s)])[["A4", "A3"]].dropna(); ds.append(float((pp["A4"]-pp["A3"]).mean()*100))
        stable = bool(all(np.sign(d) == np.sign(ds[0]) and d != 0 for d in ds))
        M["seeds_full_set"]["A4-A3"][m] = {"main": main, "seeds": ds, "mean": float(np.mean(ds)), "sd": float(np.std(ds, ddof=1)), "sign_stable": stable}
        L.append(f"| {MNAME[m]} | {main:+.2f} | " + " | ".join(f"{d:+.2f}" for d in ds) + f" | {np.mean(ds):+.2f} ± {np.std(ds, ddof=1):.2f} | {'yes' if stable else 'no'} |")
    L += ["", f"## Seeds: cross-model cell differences pp on the full evaluation set (n={len(full_items)}); each column pairs the two cells run with the same seed", "",
          "| Contrast | 42 (main) | 43 | 44 | 45 | mean ± SD (43-45) |", "|---|---|---|---|---|---|"]
    def _cell(df, m, ar, s=None):
        d = df[(df.model==m)&(df.arch==ar)] if s is None else df[(df.model==m)&(df.arch==ar)&(df.seed==s)]
        return d.set_index("qid").score
    for (xm, xa), (ym, ya) in [(("gpt-5.4", "A3"), ("gpt-5.4-mini", "A3")), (("gpt-5.4", "A4"), ("gpt-5.4-mini", "A3")),
                               (("gpt-5.4", "A4"), ("gpt-5.4-mini", "A4")), (("deepseek-v4", "A3"), ("gpt-5.4-mini", "A3"))]:
        lab = f"{MNAME[xm]} {LAB[xa]} − {MNAME[ym]} {LAB[ya]}"
        x, y = _cell(af, xm, xa), _cell(af, ym, ya); main = float((x - y.reindex(x.index)).mean()*100); ds = []
        for s in SEEDS:
            x, y = _cell(cf, xm, xa, s), _cell(cf, ym, ya, s); ds.append(float((x - y.reindex(x.index)).mean()*100))
        M["seeds_full_set"]["cross_model"][lab] = {"main": main, "seeds": ds, "mean": float(np.mean(ds)), "sd": float(np.std(ds, ddof=1))}
        L.append(f"| {lab} | {main:+.2f} | " + " | ".join(f"{d:+.2f}" for d in ds) + f" | {np.mean(ds):+.2f} ± {np.std(ds, ddof=1):.2f} |")

# Tool-removal configuration (A3-T): the A3 prompt and planning functions without external tools (phase D).
# A3 − A3-T estimates the effect of tool access, A3-T − A0 the effect of the prompt and planning scaffold.
td = allp[allp.phase == "D"]
if len(td):
    rngT = np.random.default_rng(42)
    META_T = {"submit_plan", "revise_plan"}
    def _conT(x, y, nb=5000):
        idx = x.index.intersection(y.index); d = (x[idx] - y[idx]).values
        boots = np.array([d[rngT.integers(0, len(d), len(d))].mean() for _ in range(nb)])
        p = 1.0 if (d == 0).all() else wilcoxon(x[idx], y[idx], zero_method="wilcox", method="approx").pvalue
        return {"diff_pp": float(d.mean()*100), "ci_lo": float(np.percentile(boots, 2.5)*100), "ci_hi": float(np.percentile(boots, 97.5)*100), "p": float(p), "n": int(len(d))}
    def _fT(r): return f"{r['diff_pp']:+.1f} (CI {r['ci_lo']:+.1f} to {r['ci_hi']:+.1f}; p={fmt_p(r['p'])}" + (f", p_Holm={fmt_p(r['p_holm'])})" if "p_holm" in r else ")")
    M["tool_removal"] = {"cells": {}, "contrasts": {}, "by_type": {}, "tool_use_split": {}}
    L += ["", "## Tool removal: A3-T (A3 prompt and planning functions, no external tools) against A0 and A3", ""]
    for scope, keep in (("full", lambda df: df.index.notna() if False else np.ones(len(df), bool)), ("objective", lambda df: (df.question_type != "open_ended").values)):
        rows_c = {}
        for m in MODELS:
            t = td[td.model == m]; t = t[keep(t)].set_index("qid").score
            am = a[a.model == m]; am = am[keep(am)]; pm = am.pivot_table(index="qid", columns="arch", values="score")
            M["tool_removal"]["cells"][f"{m}|{scope}"] = {"A0": float(pm["A0"].mean()*100), "A3-T": float(t.mean()*100), "A3": float(pm["A3"].mean()*100), "n": int(len(t))}
            rows_c[m] = {"A3-A3T": _conT(pm["A3"], t), "A3T-A0": _conT(t, pm["A0"]), "A3-A0": _conT(pm["A3"], pm["A0"])}
        for name in ("A3-A3T", "A3T-A0"):
            for m, ph in zip(MODELS, holm(np.array([rows_c[m][name]["p"] for m in MODELS]))): rows_c[m][name]["p_holm"] = float(ph)
        M["tool_removal"]["contrasts"][scope] = rows_c
        n = rows_c[MODELS[0]]["A3-A0"]["n"]
        L += [f"### {scope} items (n = {n}): accuracy % and paired contrasts pp (item bootstrap 95% CI, paired Wilcoxon, Holm across the four models per contrast)", "",
              "| Model | A0 | A3-T | A3 | A3 − A3-T (tool access) | A3-T − A0 (prompt and planning) | A3 − A0 |", "|---|---|---|---|---|---|---|"]
        for m in MODELS:
            c = M["tool_removal"]["cells"][f"{m}|{scope}"]; k = rows_c[m]
            L.append(f"| {MNAME[m]} | {c['A0']:.1f} | {c['A3-T']:.1f} | {c['A3']:.1f} | {_fT(k['A3-A3T'])} | {_fT(k['A3T-A0'])} | {_fT(k['A3-A0'])} |")
    L += ["", "### By question type: accuracy % (A0 / A3-T / A3) and contrasts pp (bootstrap 95% CI, paired Wilcoxon)", "",
          "| Model | Type | A0 | A3-T | A3 | A3 − A3-T | A3-T − A0 |", "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        for qt in ["calculation", "true_false", "open_ended"]:
            t = td[(td.model == m) & (td.question_type == qt)].set_index("qid").score
            pm = a[(a.model == m) & (a.question_type == qt)].pivot_table(index="qid", columns="arch", values="score")
            k = {"A0": float(pm["A0"].mean()*100), "A3-T": float(t.mean()*100), "A3": float(pm["A3"].mean()*100), "A3-A3T": _conT(pm["A3"], t), "A3T-A0": _conT(t, pm["A0"])}
            M["tool_removal"]["by_type"][f"{m}|{qt}"] = k
            L.append(f"| {MNAME[m]} | {qt} | {k['A0']:.1f} | {k['A3-T']:.1f} | {k['A3']:.1f} | {_fT(k['A3-A3T'])} | {_fT(k['A3T-A0'])} |")
    L += ["", "### Split by whether the recorded A3 trace executed an external tool (full evaluation set)", "",
          "| Model | A3 trace | Items | A0 | A3-T | A3 | A3 − A3-T | A3-T − A0 |", "|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        traj = {}
        for _l in (ROOT / "data/trajectories/main" / m / record_id("A3") / "run_1.jsonl").read_text().splitlines():
            if _l.strip():
                _r = json.loads(_l); traj[_r["qid"]] = _r
        t = td[td.model == m].set_index("qid").score; pm = a[a.model == m].pivot_table(index="qid", columns="arch", values="score")
        used = pd.Series({q: any(isinstance(c, dict) and c.get("tool") not in META_T for c in (traj[q].get("tool_calls") or [])) for q in t.index})
        for lab, sel in (("external tool called", used), ("no external tool", ~used)):
            qq = used.index[sel]
            k = {"n": int(len(qq)), "A0": float(pm.loc[qq, "A0"].mean()*100), "A3-T": float(t.loc[qq].mean()*100), "A3": float(pm.loc[qq, "A3"].mean()*100),
                 "A3-A3T": _conT(pm.loc[qq, "A3"], t.loc[qq]), "A3T-A0": _conT(t.loc[qq], pm.loc[qq, "A0"])}
            M["tool_removal"]["tool_use_split"][f"{m}|{lab}"] = k
            L.append(f"| {MNAME[m]} | {lab} | {k['n']} | {k['A0']:.1f} | {k['A3-T']:.1f} | {k['A3']:.1f} | {_fT(k['A3-A3T'])} | {_fT(k['A3T-A0'])} |")

# Judge reproducibility: the reported open-ended judgments (one session) against an independent scoring pass of the
# same main-grid answers by the same judge models on another deployment of the DeepSeek-V4 judge.
pp_path = ROOT / "data/judges/open_ended_judgments_prior_pass.jsonl"
if pp_path.exists():
    def _readj(p):
        d = {}
        for _l in p.read_text().splitlines():
            if _l.strip():
                _r = json.loads(_l); d[(_r["model"], configuration_id(_r["arch"]), _r["qid"])] = int(_r["score_0_5"])
        return d
    prior, rep = _readj(pp_path), _readj(ROOT / "data/judges/open_ended_judgments.jsonl")
    M["judge_reproducibility"] = {}
    L += ["", "## Judge reproducibility: reported open-ended judgments vs an independent scoring pass on another deployment (n = 482 answers per cell)", "",
          "| Model | Configuration | Judge | Exact agreement | Reported − other pass, open-ended mean (pp) | Reported − other pass, full-set cell (pp) |", "|---|---|---|---|---|---|"]
    for m in MODELS:
        for ar in PAPER:
            keys = [k for k in rep if k[0] == m and k[1] == ar and k in prior]
            x = np.array([rep[k] for k in keys]); y = np.array([prior[k] for k in keys])
            jname = "DeepSeek-V4" if m != "deepseek-v4" else "GPT-5.4-nano"
            d = {"judge": jname, "n": int(len(keys)), "exact_agreement": float(np.mean(x == y)), "diff_open_pp": float((x.mean()-y.mean())/5*100),
                 "diff_cell_pp": float((x.mean()-y.mean())/5*100*len(keys)/1027)}
            M["judge_reproducibility"][f"{m}|{LAB[ar]}"] = d
            L.append(f"| {MNAME[m]} | {LAB[ar]} | {jname} | {d['exact_agreement']:.3f} | {d['diff_open_pp']:+.2f} | {d['diff_cell_pp']:+.2f} |")

# sub-domain tables
M["subdomain"] = {m: {sd: {LAB[ar]: float(a[(a.model==m)&(a.arch==ar)&(a.subdomain==sd)].score.mean())
                            for ar in PAPER} for sd, _ in SUBS} for m in MODELS}
for m in MODELS:
    L += ["", f"## Sub-domain × arch: {MNAME[m]}", "", "| Sub-domain | A0 | A1 | A2 | A3 | A4 |", "|---|---|---|---|---|---|"]
    for sd, sdlab in SUBS:
        L.append(f"| {sdlab} | " + " | ".join(f"{M['subdomain'][m][sd][x]:.3f}" for x in ["A0","A1","A2","A3","A4"]) + " |")

# costs ( tokens × registry prices)
# USD per 1M tokens (input, output), provider list prices as of 2026-08-17, the date of the
# reported runs (manifest_run_1.yaml). DeepSeek-V4-flash is the rate recorded in that
# manifest at run time. These are the dated prices used in the manuscript.
PRICE = {"gpt-5.4": (2.5, 15.0), "gpt-5.4-mini": (0.75, 4.50), "gpt-5.4-nano": (0.20, 1.25), "deepseek-v4": (0.14, 0.28)}
M["cost"] = {}
L += ["", "## Cost per 100 questions (recorded tokens × 2026-08 registry prices, USD)", ""]
for m in MODELS:
    parts = []
    for ar in PAPER:
        d = a[(a.model==m)&(a.arch==ar)]
        c100 = float((d.in_tokens.mean()*PRICE[m][0] + d.out_tokens.mean()*PRICE[m][1]) / 1e6 * 100)
        M["cost"][f"{m}|{LAB[ar]}"] = c100
        parts.append(f"{LAB[ar]} {c100:.3f}")
    L.append(f"- {MNAME[m]}: " + ", ".join(parts))


# ---- instruction following, recomputed from raw trajectories (not 1 - parse_error) ----
import re as _re
_NUM = _re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
def _if_pass(r):
    if r.get("parse_error"): return False
    qt = r.get("_question_type") or r.get("question_type"); a = r.get("answer")
    if qt == "true_false":
        return isinstance(a, bool) or str(a).strip().lower() in {"true", "false"}
    if qt == "calculation":
        u = str(r.get("unit") or "").strip().lower()
        return bool(_NUM.search(str(a or ""))) and u not in {"", "none", "null"}
    return a is not None and bool(str(a).strip())
_IF = {}
for _m in MODELS:
    _vals = []
    for _ar in PAPER:
        _rec = {}
        for _f in sorted((ROOT / "data/trajectories/main" / _m / record_id(_ar)).glob("*.jsonl")):
            for _l in _f.read_text().splitlines():
                if _l.strip():
                    _r = json.loads(_l); _rec[_r.get("qid")] = _r
        _vals.append(round(100 * float(np.mean([_if_pass(x) for x in _rec.values()])), 2))
    _IF[_m] = dict(zip(["A0", "A1", "A2", "A3", "A4"], _vals))
M["instruction_following_pct"] = {"definition": "type-aware IF from raw trajectories; n=1027/cell", **_IF}
L += ["", "## Instruction following (%) — type-aware, from raw trajectories", "",
      "| Model | A0 | A1 | A2 | A3 | A4 |", "|---|---|---|---|---|---|"]
for _m in MODELS:
    L.append(f"| {MNAME[_m]} | " + " | ".join(f"{_IF[_m][x]:.1f}" for x in ["A0","A1","A2","A3","A4"]) + " |")

(ROOT / "analysis_outputs/statistics.json").write_text(json.dumps(M, indent=1))
(ROOT / "analysis_outputs/statistics.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L[:40]))
print(f"\n... saved statistics.md/.json ({len(L)} lines)")
