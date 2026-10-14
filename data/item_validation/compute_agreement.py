# -*- coding: utf-8 -*-
"""Recompute agreement_report.json from the two completed rating sheets.
Usage: python3 compute_agreement.py [pass1_sheet.csv] [pass2_sheet.csv] [agreement_report.json]"""
import sys, os, csv, json
CRIT = ["clarity", "gold_correctness", "reference_support", "difficulty_appropriateness"]
def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    out = {}
    for r in rows:
        q = str(r["qid"]).strip(); v = {}
        for c in CRIT:
            x = (r.get(c + "_0_1") or "").strip()
            if x not in ("0", "1"): raise SystemExit(f"{os.path.basename(path)}: {q} {c} = {x!r} (must be 0 or 1)")
            v[c] = int(x)
        v["comment"] = r.get("comment") or ""
        v["programmatic"] = q.startswith("TS")
        out[q] = v
    return out
def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    p1 = sum(a) / n; p2 = sum(b) / n; pe = p1 * p2 + (1 - p1) * (1 - p2)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    f1 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "pass1_sheet.csv")
    f2 = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "pass2_sheet.csv")
    fo = sys.argv[3] if len(sys.argv) > 3 else os.path.join(here, "agreement_report.json")
    r1 = load(f1); r2 = load(f2)
    assert set(r1) == set(r2), "qid sets differ"
    qids = sorted(r1); n = len(qids)
    rep = {"n": n, "sample": "stratified seed 20260820 over the 1,027-item evaluation set; items as in the released bank",
           "raters": "two independent LLM passes (OpenAI Codex), checked by PhD-level exposure scientists",
           "kappa_note": "agreement between the two passes", "per_criterion": {}}
    for c in CRIT:
        a = [r1[q][c] for q in qids]; b = [r2[q][c] for q in qids]
        both = sum(x == 1 and y == 1 for x, y in zip(a, b))
        rep["per_criterion"][c] = {"both": both, "pct": round(100 * both / n, 1), "pass1": sum(a), "pass2": sum(b),
                                   "agree": round(100 * sum(x == y for x, y in zip(a, b)) / n, 1), "kappa": round(kappa(a, b), 2)}
    np_ = [q for q in qids if not r1[q]["programmatic"]]; pr = [q for q in qids if r1[q]["programmatic"]]
    both_np = sum(r1[q]["reference_support"] == 1 and r2[q]["reference_support"] == 1 for q in np_)
    both_pr = sum(r1[q]["reference_support"] == 1 and r2[q]["reference_support"] == 1 for q in pr)
    rep["reference_support_excluding_programmatic"] = {"n": len(np_), "both_accept": both_np, "pct": round(100 * both_np / len(np_), 1)}
    rep["reference_support_programmatic"] = {"n": len(pr), "both_accept": both_pr}
    acc4 = [q for q in qids if all(r1[q][c] == 1 and r2[q][c] == 1 for c in CRIT)]
    acc3 = [q for q in qids if all(r1[q][c] == 1 and r2[q][c] == 1 for c in CRIT if c != "reference_support")]
    rep["acceptance_4of4_pct"] = round(100 * len(acc4) / n, 1); rep["acceptance_3of3_pct"] = round(100 * len(acc3) / n, 1)
    rep["programmatic_items"] = len(pr)
    rep["rejected_items"] = [{"qid": q, "failed": {c: [k for k, r in (("pass1", r1), ("pass2", r2)) if r[q][c] == 0] for c in CRIT if r1[q][c] == 0 or r2[q][c] == 0},
                              "comments": {"pass1": r1[q]["comment"], "pass2": r2[q]["comment"]}} for q in qids if q not in acc4]
    json.dump(rep, open(fo, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "rejected_items"}, ensure_ascii=False, indent=1))
    print("rejected items:", len(rep["rejected_items"]), "->", fo)
if __name__ == "__main__": main()
