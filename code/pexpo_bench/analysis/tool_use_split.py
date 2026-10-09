"""Split the A3-A0 gain of every base model by whether the A3 trace invoked an external tool.

Groups are assigned from the recorded A3 trajectory of each item (plan meta-tools submit_plan and
revise_plan excluded): "none" = no external tool call, "sandbox_only" = python_sandbox was the only
external tool called, "domain" = at least one domain lookup or calculation function was called
(with or without the sandbox). The split is descriptive: the model chooses when to call a tool, and
it does so on the items it finds hardest, so group differences reflect item selection as well as
tool effect. Writes analysis_outputs/tool_use_split.json and .md (SI Table S11)."""
import json, os, pathlib
import pandas as pd
from pexpo_bench.data_schema import configuration_id, record_id

ROOT = pathlib.Path(os.environ.get("PEXPO_ROOT", "."))
OUT = ROOT / "analysis_outputs"; OUT.mkdir(exist_ok=True)
MODELS = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "deepseek-v4"]
MNAME = {"gpt-5.4": "GPT-5.4", "gpt-5.4-mini": "GPT-5.4-mini", "gpt-5.4-nano": "GPT-5.4-nano", "deepseek-v4": "DeepSeek-V4"}
META = {"submit_plan", "revise_plan"}
GROUPS = ["none", "sandbox_only", "domain"]
GLABEL = {"none": "no external tool", "sandbox_only": "python_sandbox only", "domain": "domain tool"}

res = pd.read_parquet(ROOT / "data/scored/results_main.parquet")
res["arch"] = res["arch"].map(configuration_id)


def group_of(row):
    ext = [c.get("tool") for c in (row.get("tool_calls") or []) if isinstance(c, dict) and c.get("tool") not in META]
    if not ext:
        return "none"
    return "sandbox_only" if set(ext) == {"python_sandbox"} else "domain"


M, L = {"groups": {g: GLABEL[g] for g in GROUPS}, "models": {}}, []
L += ["# A3-A0 gain split by tool use in the A3 trace (main grid, n = 1,027 per model)", "",
      "| Base model | Trace group | Items | Share of items % | A0 accuracy % | A3 accuracy % | A3-A0 pp |", "|---|---|---|---|---|---|---|"]
for m in MODELS:
    traj = {}
    for line in (ROOT / f"data/trajectories/main/{m}/{record_id('A3')}/run_1.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line); traj[r["qid"]] = r
    piv = res[res.model == m].pivot_table(index="qid", columns="arch", values="score")
    qt = res[res.model == m].drop_duplicates("qid").set_index("qid")["question_type"]
    assert len(piv) == 1027 and set(traj) == set(piv.index), m
    grp = pd.Series({q: group_of(traj[q]) for q in piv.index})
    M["models"][m] = {"n_items": int(len(piv)), "groups": {}, "by_question_type": {}}
    for g in GROUPS:
        sel = grp == g
        d = {"n": int(sel.sum()), "share_pct": float(100 * sel.mean()),
             "a0_accuracy_pct": float(100 * piv.loc[sel, "A0"].mean()) if sel.any() else None,
             "a3_accuracy_pct": float(100 * piv.loc[sel, "A3"].mean()) if sel.any() else None,
             "a3_minus_a0_pp": float(100 * (piv.loc[sel, "A3"] - piv.loc[sel, "A0"]).mean()) if sel.any() else None}
        M["models"][m]["groups"][g] = d
        L.append(f"| {MNAME[m]} | {GLABEL[g]} | {d['n']} | {d['share_pct']:.1f} | {d['a0_accuracy_pct']:.1f} | {d['a3_accuracy_pct']:.1f} | {d['a3_minus_a0_pp']:+.1f} |")
    for t in ("true_false", "calculation", "open_ended"):
        sub = qt[qt == t].index
        M["models"][m]["by_question_type"][t] = {g: {"n": int((grp[sub] == g).sum()),
                                                      "a3_minus_a0_pp": float(100 * (piv.loc[sub[grp[sub] == g], "A3"] - piv.loc[sub[grp[sub] == g], "A0"]).mean()) if (grp[sub] == g).any() else None}
                                                  for g in GROUPS}
L += ["", "Descriptive split: tool use is chosen by the model and concentrates on the items with the lowest A0 accuracy."]
(OUT / "tool_use_split.json").write_text(json.dumps(M, indent=1))
(OUT / "tool_use_split.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
