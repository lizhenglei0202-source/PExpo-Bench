"""Compute the supplementary statistics described in Methods 2.6 that make_tables.py does not emit: per-cell bootstrap confidence intervals, Kendall rank correlations, Pareto-frontier membership, cross-cell contrasts and the headline equivalence tests. Outputs are written to analysis_outputs/."""
import os
from pexpo_bench.data_schema import configuration_id
import json, math, pathlib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, wilcoxon

ROOT = pathlib.Path(os.environ.get("PEXPO_ROOT", "."))
(ROOT / "analysis_outputs").mkdir(parents=True, exist_ok=True)
PAPER = ["A0", "A1", "A2", "A3", "A4"]
LAB = dict(zip(PAPER, ["A0", "A1", "A2", "A3", "A4"]))
MODELS = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "deepseek-v4"]
MNAME = {"gpt-5.4": "GPT-5.4", "gpt-5.4-mini": "GPT-5.4-mini", "gpt-5.4-nano": "GPT-5.4-nano", "deepseek-v4": "DeepSeek-V4"}

# USD per 1M tokens (input, output), provider list prices as of 2026-08-17, the date of the
# reported runs (manifest_run_1.yaml). DeepSeek-V4-flash is the rate recorded in that
# manifest at run time. These are the dated prices used in the manuscript, identical to
# the table in make_tables.py; the Pareto panel of Figure 2c uses this dated-price scenario.
PRICE = {"gpt-5.4": (2.5, 15.0), "gpt-5.4-mini": (0.75, 4.50), "gpt-5.4-nano": (0.20, 1.25), "deepseek-v4": (0.14, 0.28)}

# Random-number discipline, matching make_tables.py: one np.random.default_rng(SEED),
# bootstrap indices drawn with rng.integers(0, n, n), percentiles via np.percentile.
# make_tables.py shares a single generator across its four focal A4-A3 contrasts; the
# statistics below are each independent, so each one opens its own default_rng(SEED).
# Every figure in this module is therefore deterministic and order-stable.
SEED = 42
N_CELL_BOOT = 500      # Methods 2.6: per-cell 95% confidence intervals
N_ITEM_BOOT = 5000     # Methods 2.6: paired item bootstrap for contrasts and equivalence
TOST_MARGINS_PP = (2.0, 1.5)

# The six small-model cells compared against the flagship no-tool cell, Holm-corrected
# as one family (Methods 2.6; SI Section E).
CROSS_CELL_REFERENCE = ("gpt-5.4", "A0")
CROSS_CELL_FAMILY = [("gpt-5.4-mini", "A1"), ("gpt-5.4-mini", "A2"),
                     ("gpt-5.4-mini", "A3"), ("gpt-5.4-mini", "A4"),
                     ("gpt-5.4-nano", "A3"), ("gpt-5.4-nano", "A4")]
# The headline cell (GPT-5.4-mini A3, the recommended configuration) and the same model's A4
# cell, each against the flagship's A3 cell and its best cell (A4). Signed as flagship minus
# small model, so a positive difference is the flagship's advantage.
HEADLINE_SMALL = ("gpt-5.4-mini", "A3")
SECONDARY_SMALL = ("gpt-5.4-mini", "A4")
HEADLINE_FLAGSHIP = [("gpt-5.4", "A3"), ("gpt-5.4", "A4")]


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


def boot_means_pp(values, n, seed=SEED):
    """Bootstrap distribution of the mean, in percentage points."""
    v = np.asarray(values, dtype=float); rng = np.random.default_rng(seed)
    return np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]) * 100


def interval(boots, level):
    """Two-sided percentile interval at the given confidence level (0.95, 0.90, ...)."""
    tail = (1.0 - level) / 2 * 100
    return float(np.percentile(boots, tail)), float(np.percentile(boots, 100 - tail))


def paired_contrast(x, y, n_boot=N_ITEM_BOOT):
    """Paired mean difference y - x over the shared item intersection, with bootstrap draws."""
    idx = x.index.intersection(y.index)
    d = (y[idx] - x[idx]).values
    p = 1.0 if (d == 0).all() else wilcoxon(x[idx], y[idx], zero_method="wilcox", method="approx").pvalue
    return d, float(p), boot_means_pp(d, n_boot)


def pareto_frontier(cells_pct, cost_per_100q):
    """Cells not dominated on (cost, accuracy): no other cell is both cheaper and more accurate."""
    pts = [(cost_per_100q[f"{m}|{LAB[ar]}"], cells_pct[m][LAB[ar]], m, LAB[ar])
           for m in MODELS for ar in PAPER]
    out = []
    for c, acc, m, ar in pts:
        dominated = any((c2 <= c and a2 >= acc and (c2 < c or a2 > acc)) for c2, a2, _, _ in pts)
        out.append({"model": m, "arch": ar, "cost_usd_per_100q": float(c),
                    "accuracy_pct": float(acc), "on_frontier": bool(not dominated)})
    return out


a = pd.read_parquet(ROOT / "data/scored/results_main.parquet")
a["arch"] = a["arch"].map(configuration_id)
obj = a[a.question_type != "open_ended"]
piv = {m: a[a.model == m].pivot_table(index="qid", columns="arch", values="score") for m in MODELS}
piv_obj = {m: obj[obj.model == m].pivot_table(index="qid", columns="arch", values="score") for m in MODELS}
M = {"release": "publication-snapshot-20261009",
     "seed": SEED, "n_cell_bootstrap": N_CELL_BOOT, "n_item_bootstrap": N_ITEM_BOOT}
L = ["# Supplementary statistics for the manuscript experiment", "",
     f"Deterministic: every bootstrap uses numpy default_rng({SEED}).", ""]

# ---------------------------------------------------------------------------
# 1. Per-cell accuracy with a 95% bootstrap confidence interval (500 resamples)
# ---------------------------------------------------------------------------
# Items are resampled in the stored row order of results_main.parquet (.score.values
# without re-sorting), which is the order the published per-cell intervals were drawn in.
cell_ci = {}
for m in MODELS:
    for ar in PAPER:
        v = a[(a.model == m) & (a.arch == ar)].score.values
        lo, hi = interval(boot_means_pp(v, N_CELL_BOOT), 0.95)
        cell_ci[f"{m}|{LAB[ar]}"] = {"n": int(len(v)), "accuracy_pct": float(np.mean(v) * 100),
                                     "ci_lo_pct": lo, "ci_hi_pct": hi}
M["cell_accuracy_ci95"] = cell_ci
L += [f"## Per-cell accuracy with 95% bootstrap CI ({N_CELL_BOOT} resamples)", "",
      "| Model | Arch | n | Accuracy % | CI low % | CI high % |", "|---|---|---|---|---|---|"]
for m in MODELS:
    for ar in PAPER:
        r = cell_ci[f"{m}|{LAB[ar]}"]
        L.append(f"| {MNAME[m]} | {LAB[ar]} | {r['n']} | {r['accuracy_pct']:.2f} | "
                 f"{r['ci_lo_pct']:.2f} | {r['ci_hi_pct']:.2f} |")

# ---------------------------------------------------------------------------
# 2. Kendall's tau over configuration rankings of mean accuracy (SI Table S4)
# ---------------------------------------------------------------------------
acc_vec = {m: [float(a[(a.model == m) & (a.arch == ar)].score.mean()) for ar in PAPER] for m in MODELS}
tau_rows, tau_matrix = [], {}
for i, m1 in enumerate(MODELS):
    for m2 in MODELS[i + 1:]:
        tau, p = kendalltau(acc_vec[m1], acc_vec[m2])
        tau_rows.append({"model_a": m1, "model_b": m2, "tau": float(tau), "p": float(p)})
        tau_matrix.setdefault(m1, {})[m2] = float(tau)
        tau_matrix.setdefault(m2, {})[m1] = float(tau)
GPT_FAMILY = [m for m in MODELS if m.startswith("gpt-")]
gpt_taus = [r["tau"] for r in tau_rows if r["model_a"] in GPT_FAMILY and r["model_b"] in GPT_FAMILY]
other_taus = [r["tau"] for r in tau_rows if (r["model_a"] in GPT_FAMILY) != (r["model_b"] in GPT_FAMILY)]
M["kendall_tau"] = {"pairs": tau_rows,
                    "gpt_family_range": [float(min(gpt_taus)), float(max(gpt_taus))],
                    "gpt_vs_deepseek_range": [float(min(other_taus)), float(max(other_taus))],
                    "ranking_basis": "mean accuracy over the five configurations, n=5 per model"}
L += ["", "## Kendall tau over configuration rankings of mean accuracy (Table S4)", "",
      "| | " + " | ".join(MNAME[m] for m in MODELS) + " |", "|---|---|---|---|---|"]
for m1 in MODELS:
    L.append(f"| {MNAME[m1]} | " + " | ".join("1.000" if m1 == m2 else f"{tau_matrix[m1][m2]:+.3f}"
                                              for m2 in MODELS) + " |")
L += ["", f"- GPT family: tau = {min(gpt_taus):+.1f} to {max(gpt_taus):+.1f}",
      f"- GPT family vs DeepSeek-V4: tau = {min(other_taus):+.1f} to {max(other_taus):+.1f}"]

# ---------------------------------------------------------------------------
# 3. Pareto-frontier membership by a dominance test on (cost, accuracy), 20 cells
# ---------------------------------------------------------------------------
cost = {}
for m in MODELS:
    for ar in PAPER:
        d = a[(a.model == m) & (a.arch == ar)]
        cost[f"{m}|{LAB[ar]}"] = float((d.in_tokens.mean() * PRICE[m][0]
                                        + d.out_tokens.mean() * PRICE[m][1]) / 1e6 * 100)
# The subset variant keeps this dated full-set token charge (as Figure 2c does) and
# changes only the accuracy coordinate, so the two frontiers are directly comparable.
cells_full = {m: {LAB[ar]: float(a[(a.model == m) & (a.arch == ar)].score.mean() * 100) for ar in PAPER}
              for m in MODELS}
cells_obj = {m: {LAB[ar]: float(obj[(obj.model == m) & (obj.arch == ar)].score.mean() * 100) for ar in PAPER}
             for m in MODELS}
n_full = int(a.groupby(["model", "arch"]).size().max())
n_obj = int(obj.groupby(["model", "arch"]).size().max())
M["pareto"] = {}
for key, cz, n in [("full_set", cells_full, n_full), ("objective_subset", cells_obj, n_obj)]:
    rows = pareto_frontier(cz, cost)
    M["pareto"][key] = {"n_items": n, "cells": rows,
                        "frontier": [f"{MNAME[r['model']]} {r['arch']}" for r in rows if r["on_frontier"]]}
    L += ["", f"## Pareto frontier on (cost, accuracy), 20 cells — {key} (n={n})", "",
          "| Model | Arch | Cost USD/100q | Accuracy % | Frontier |", "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {MNAME[r['model']]} | {r['arch']} | {r['cost_usd_per_100q']:.3f} | "
                 f"{r['accuracy_pct']:.2f} | {'yes' if r['on_frontier'] else '-'} |")
    L.append("")
    L.append("Frontier: " + ", ".join(M["pareto"][key]["frontier"]))

# ---------------------------------------------------------------------------
# 4 + 6. Cross-cell contrasts between a small-model cell and a flagship cell
# ---------------------------------------------------------------------------
# Family of six against the flagship no-tool cell, Holm-corrected together.
rm, ra = CROSS_CELL_REFERENCE
six, ps = [], []
for m, ar in CROSS_CELL_FAMILY:
    d, p, boots = paired_contrast(piv[rm][ra], piv[m][ar])
    lo, hi = interval(boots, 0.95)
    six.append({"cell": f"{MNAME[m]} {LAB[ar]}", "reference": f"{MNAME[rm]} {LAB[ra]}",
                "model": m, "arch": LAB[ar], "n": int(len(d)), "diff_pp": float(d.mean() * 100),
                "ci_lo_pp": lo, "ci_hi_pp": hi, "p": p})
    ps.append(p)
for r, ph in zip(six, holm(np.array(ps))):
    r["p_holm"] = float(ph)
M["cross_cell_vs_flagship_a0"] = {
    "holm_family_size": len(six),
    "note": "paired Wilcoxon on the shared item intersection; positive = small-model cell higher",
    "contrasts": six}
L += ["", f"## Cross-cell contrasts vs {MNAME[rm]} {LAB[ra]} (Holm across {len(six)})", ""]
for r in six:
    L.append(f"- {r['cell']} − {r['reference']}: {r['diff_pp']:+.1f} pp "
             f"(95% CI {r['ci_lo_pp']:+.1f} to {r['ci_hi_pp']:+.1f}), n={r['n']}, "
             f"p={fmt_p(r['p'])}, p_Holm={fmt_p(r['p_holm'])}")

# Headline contrasts: flagship minus the small-model cell, on both item sets, for the headline
# cell (GPT-5.4-mini A3) and for the same model's A4 cell.
head = []
for role, (sm, sa) in (("headline", HEADLINE_SMALL), ("secondary", SECONDARY_SMALL)):
    for fm, fa in HEADLINE_FLAGSHIP:
        for set_name, P, n_set in [("full_set", piv, n_full), ("objective_subset", piv_obj, n_obj)]:
            d, p, boots = paired_contrast(P[sm][sa], P[fm][fa])
            lo95, hi95 = interval(boots, 0.95)
            lo90, hi90 = interval(boots, 0.90)
            head.append({"comparison": f"{MNAME[fm]} {LAB[fa]} vs {MNAME[sm]} {LAB[sa]}",
                         "role": role, "fixed_configuration": sa == fa,
                         "item_set": set_name, "n": int(len(d)), "n_expected": n_set,
                         "flagship_advantage_pp": float(d.mean() * 100),
                         "ci95_lo_pp": lo95, "ci95_hi_pp": hi95,
                         "ci90_lo_pp": lo90, "ci90_hi_pp": hi90, "p": p,
                         "equivalence": {f"margin_{mg:g}pp": {
                             "margin_pp": float(mg),
                             "pass": bool(lo90 > -mg and hi90 < mg)} for mg in TOST_MARGINS_PP}})
M["headline_cross_cell"] = {
    "sign_convention": "flagship cell minus small-model cell; positive = flagship advantage",
    "correction": "comparisons at a fixed configuration (GPT-5.4 A3 vs GPT-5.4-mini A3, GPT-5.4 A4 vs "
                  "GPT-5.4-mini A4) are Holm-corrected inside the 30-contrast family of Table S3 "
                  "(statistics.json: between_model_contrasts); the cross-configuration comparisons "
                  "belong to no family and are reported uncorrected",
    "contrasts": head}
L += ["", f"## Headline cross-cell contrasts ({MNAME[HEADLINE_SMALL[0]]} {LAB[HEADLINE_SMALL[1]]} (headline) "
      f"and {LAB[SECONDARY_SMALL[1]]} against the flagship)", ""]
for r in head:
    L.append(f"- {r['comparison']} [{r['item_set']}, n={r['n']}]: {r['flagship_advantage_pp']:+.2f} pp "
             f"(95% CI {r['ci95_lo_pp']:+.1f} to {r['ci95_hi_pp']:+.1f}), p={fmt_p(r['p'])}")

# Two one-sided equivalence tests, read off the 90% paired item-bootstrap interval.
M["equivalence_tests"] = {
    "procedure": "two one-sided tests; equivalence at margin +/-m is concluded when the 90% "
                 f"paired item-bootstrap interval ({N_ITEM_BOOT} resamples) of the mean difference "
                 "lies strictly inside (-m, +m)",
    "margins_pp": [float(x) for x in TOST_MARGINS_PP],
    "results": [{"comparison": r["comparison"], "item_set": r["item_set"], "n": r["n"],
                 "ci90_lo_pp": r["ci90_lo_pp"], "ci90_hi_pp": r["ci90_hi_pp"],
                 "pass": {k: v["pass"] for k, v in r["equivalence"].items()}} for r in head]}
L += ["", "## Equivalence tests (90% item-bootstrap interval; margins "
      + " and ".join(f"±{mg:g} pp" for mg in TOST_MARGINS_PP) + ")", "",
      "| Comparison | Item set | n | 90% CI low pp | 90% CI high pp | "
      + " | ".join(f"±{mg:g} pp" for mg in TOST_MARGINS_PP) + " |",
      "|---|---|---|---|---|---|---|"]
for r in head:
    L.append(f"| {r['comparison']} | {r['item_set']} | {r['n']} | {r['ci90_lo_pp']:+.1f} | "
             f"{r['ci90_hi_pp']:+.1f} | "
             + " | ".join("pass" if r["equivalence"][f"margin_{mg:g}pp"]["pass"] else "fail"
                          for mg in TOST_MARGINS_PP) + " |")
L += ["", "Non-significance is not equivalence; the margins are reported descriptively."]

(ROOT / "analysis_outputs/supplementary_statistics.json").write_text(json.dumps(M, indent=1))
(ROOT / "analysis_outputs/supplementary_statistics.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L[:40]))
print(f"\n... saved supplementary_statistics.md/.json ({len(L)} lines)")
