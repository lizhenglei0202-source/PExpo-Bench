"""Error magnitude and abstention on calculation items (main grid, 361 items x 20 cells).

Every calculation response is placed in one of six classes from the scorer's own quantities:
  correct      score 1.0 (within tolerance, or within the item's rounding allowance);
  near         score 0.5 (between one and two tolerances);
  <2x          wrong, predicted/gold ratio within a factor of 2;
  2-10x        wrong, ratio between 2x and 10x;
  >=10x        wrong by at least an order of magnitude, or wrong sign, or zero;
  non-numeric  no extractable number (parse error, execution error, empty or non-numeric answer field).
The predicted value is the one the scorer used (the unit-converted value when that scored higher).
Among wrong numeric answers the script also reports the share that are exact decimal-prefix slips
(the answer divided by 10^k, k != 0, falls within tolerance of the gold value), the over/under split,
and the median |log10(predicted/gold)|. The one two-component item (named_numeric_components) is
classified by score only and excluded from the ratio statistics. Writes analysis_outputs/error_magnitude.json
and .md (SI Section K, Table S12, Figure S4)."""
import json, math, os, pathlib
import pandas as pd, yaml
from pexpo_bench.data_schema import configuration_id, record_id
from pexpo_bench.analysis.scoring import parse_unit, score_row
from pexpo_bench.analysis.numeric_scoring import number, scalar_score, default_rounding

ROOT = pathlib.Path(os.environ.get("PEXPO_ROOT", "."))
OUT = ROOT / "analysis_outputs"; OUT.mkdir(exist_ok=True)
MODELS = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "deepseek-v4"]
MNAME = {"gpt-5.4": "GPT-5.4", "gpt-5.4-mini": "GPT-5.4-mini", "gpt-5.4-nano": "GPT-5.4-nano", "deepseek-v4": "DeepSeek-V4"}
ARCHS = ["A0", "A1", "A2", "A3", "A4"]
CLASSES = ["correct", "near", "<2x", "2-10x", ">=10x", "non-numeric"]
LOG2 = math.log10(2.0)

bank = yaml.safe_load((ROOT / "data/bank/bank_evaluation_set.yaml").read_text())
calc = {q["qid"]: q for q in bank if q["question_type"] == "calculation"}
res = pd.read_parquet(ROOT / "data/scored/results_main.parquet")
res["arch"] = res["arch"].map(configuration_id)
score_of = {(r.model, r.arch, r.qid): float(r.score) for r in res[res.question_type == "calculation"].itertuples()}


def classify(r, gq, model, arch):
    sc = score_of[(model, arch, r["qid"])]
    chk, _ = score_row(r, gq, model, arch)
    assert abs(chk - sc) < 1e-9, (model, arch, r["qid"], chk, sc)
    if sc >= 1.0:
        return "correct", None
    if sc > 0:
        return "near", None
    err = r.get("error_msg") or ""
    if r.get("parse_error") or (err and not err.startswith("route=")):
        return "non-numeric", None
    if (gq.get("scoring") or {}).get("kind") == "named_numeric_components":
        return "component", None
    pn = number(r.get("answer"))
    if pn is None or not math.isfinite(pn):
        return "non-numeric", None
    gn = gq.get("numeric_reference", number(gq["answer"]))
    tol = gq.get("tolerance", 0.10); policy = gq.get("rounding", default_rounding(gn, gq.get("unit")))
    pe = pn
    pu, gu = parse_unit(r.get("unit")), parse_unit(gq.get("unit"))
    if pu and gu and pu[0] == gu[0] and pu[1] != gu[1]:
        conv = pn * pu[1] / gu[1]
        if scalar_score(conv, gn, tol, policy) > scalar_score(pn, gn, tol, policy):
            pe = conv
    info = {"pred": pe, "gold": gn, "tol": tol}
    if gn == 0 or pe == 0 or (pe > 0) != (gn > 0):
        info.update(mag=None, sign_or_zero=True, prefix_slip=False, over=None)
        return ">=10x", info
    mag = abs(math.log10(abs(pe) / abs(gn)))
    slip = any(k != 0 and abs(pe / 10 ** k - gn) / abs(gn) <= tol for k in range(-12, 13))
    info.update(mag=mag, sign_or_zero=False, prefix_slip=slip, over=abs(pe) > abs(gn))
    return ("<2x" if mag < LOG2 else "2-10x" if mag < 1.0 else ">=10x"), info


cells, rows_by = {}, {}
for m in MODELS:
    for a in ARCHS:
        n_rows = 0
        for line in (ROOT / f"data/trajectories/main/{m}/{record_id(a)}/run_1.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["qid"] not in calc:
                continue
            gq = calc[r["qid"]]
            cls, info = classify(r, gq, m, a)
            rows_by.setdefault((m, a), []).append((r["qid"], gq["subdomain"], gq.get("source"), cls, info))
            n_rows += 1
        assert n_rows == len(calc), (m, a, n_rows)


def summarize(rows):
    n = len(rows)
    counts = {c: sum(1 for x in rows if x[3] == c) for c in CLASSES}
    counts["component"] = sum(1 for x in rows if x[3] == "component")
    wrong = [x for x in rows if x[4] is not None]
    mags = sorted(x[4]["mag"] for x in wrong if x[4]["mag"] is not None)
    big = [x for x in wrong if x[3] == ">=10x"]
    d = {"n": n, "counts": counts, "pct": {c: 100.0 * counts[c] / n for c in CLASSES},
         "wrong_numeric": len(wrong),
         "pct_wrong_numeric_over": (100.0 * sum(1 for x in wrong if x[4]["over"]) / len(wrong)) if wrong else None,
         "median_abs_log10_ratio_wrong": (mags[len(mags) // 2] if mags else None),
         "n_ge10x": len(big),
         "pct_ge10x_prefix_slip": (100.0 * sum(1 for x in big if x[4]["prefix_slip"]) / len(big)) if big else None,
         "n_ge10x_sign_or_zero": sum(1 for x in big if x[4]["sign_or_zero"])}
    return d


M = {"classes": CLASSES, "note": __doc__.strip().splitlines()[0], "cells": {}, "pooled_by_config": {},
     "pooled_by_config_stream": {}, "pooled_by_config_subdomain": {}}
for (m, a), rows in rows_by.items():
    M["cells"][f"{m}|{a}"] = summarize(rows)
for a in ARCHS:
    pooled = [x for m in MODELS for x in rows_by[(m, a)]]
    M["pooled_by_config"][a] = summarize(pooled)
    for stream in ("tool_required", "release"):
        M["pooled_by_config_stream"][f"{a}|{stream}"] = summarize([x for x in pooled if x[2] == stream])
    for sd in sorted(set(x[1] for x in pooled)):
        M["pooled_by_config_subdomain"][f"{a}|{sd}"] = summarize([x for x in pooled if x[1] == sd])
# per-item worst-case: items wrong by >=10x in every configuration of a model
M["items_ge10x_all_configs"] = {m: sorted(q for q in calc if all(any(x[0] == q and x[3] == ">=10x" for x in rows_by[(m, a)]) for a in ARCHS)) for m in MODELS}

L = ["# Error magnitude and abstention on calculation items (main grid, n = 361 per cell)", "",
     "Classes: correct = score 1.0; near = score 0.5; <2x, 2-10x, >=10x = wrong numeric answer by that factor "
     "(>=10x includes wrong sign and zero); non-numeric = no extractable number. Percent of the 361 items.", "",
     "| Base model | Config | Correct | Near | <2x | 2-10x | >=10x | Non-numeric | Prefix slips among >=10x | Over-estimates among wrong | Median \\|log10 ratio\\| (wrong) |",
     "|---|---|---|---|---|---|---|---|---|---|---|"]
def fmt(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"
for m in MODELS:
    for a in ARCHS:
        d = M["cells"][f"{m}|{a}"]; p = d["pct"]
        L.append(f"| {MNAME[m]} | {a} | {p['correct']:.1f} | {p['near']:.1f} | {p['<2x']:.1f} | {p['2-10x']:.1f} | {p['>=10x']:.1f} | {p['non-numeric']:.1f} | "
                 f"{fmt(d['pct_ge10x_prefix_slip'])}% (n={d['n_ge10x']}) | {fmt(d['pct_wrong_numeric_over'])}% (n={d['wrong_numeric']}) | {fmt(d['median_abs_log10_ratio_wrong'], 2)} |")
L += ["", "## Pooled over the four base models (n = 1,444 responses per configuration)", "",
      "| Config | Correct | Near | <2x | 2-10x | >=10x | Non-numeric | Prefix slips among >=10x | Over-estimates among wrong |", "|---|---|---|---|---|---|---|---|---|"]
for a in ARCHS:
    d = M["pooled_by_config"][a]; p = d["pct"]
    L.append(f"| {a} | {p['correct']:.1f} | {p['near']:.1f} | {p['<2x']:.1f} | {p['2-10x']:.1f} | {p['>=10x']:.1f} | {p['non-numeric']:.1f} | "
             f"{fmt(d['pct_ge10x_prefix_slip'])}% (n={d['n_ge10x']}) | {fmt(d['pct_wrong_numeric_over'])}% (n={d['wrong_numeric']}) |")
L += ["", "## Pooled, by item stream", "", "| Config | Stream | n | Correct | Near | <2x | 2-10x | >=10x | Non-numeric |", "|---|---|---|---|---|---|---|---|---|"]
for a in ARCHS:
    for stream, lab in (("tool_required", "programmatic (100)"), ("release", "LLM-guided (261)")):
        d = M["pooled_by_config_stream"][f"{a}|{stream}"]; p = d["pct"]
        L.append(f"| {a} | {lab} | {d['n']} | {p['correct']:.1f} | {p['near']:.1f} | {p['<2x']:.1f} | {p['2-10x']:.1f} | {p['>=10x']:.1f} | {p['non-numeric']:.1f} |")
L += ["", "## Pooled, by sub-domain", "", "| Config | Sub-domain | n | Correct | >=10x | Non-numeric |", "|---|---|---|---|---|---|"]
for a in ARCHS:
    for sd in sorted(set(q["subdomain"] for q in calc.values())):
        d = M["pooled_by_config_subdomain"][f"{a}|{sd}"]; p = d["pct"]
        L.append(f"| {a} | {sd} | {d['n']} | {p['correct']:.1f} | {p['>=10x']:.1f} | {p['non-numeric']:.1f} |")
L += ["", "Items wrong by >=10x in all five configurations of a model: " + "; ".join(f"{MNAME[m]}: {len(v)}" for m, v in M["items_ge10x_all_configs"].items()), ""]
(OUT / "error_magnitude.json").write_text(json.dumps(M, indent=1))
(OUT / "error_magnitude.md").write_text("\n".join(L))
print("\n".join(L[:40]))
