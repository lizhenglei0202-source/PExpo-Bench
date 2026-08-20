import os
"""Curated rescore (2026-08-11): re-scores ALL archived run trajectories against the patched
golds, with unit-aware numeric comparison, excluding retired items. Zero model/API calls.

Replicates analysis/build_v3_scored.py scoring exactly, then adds:
  - golds from pexpo_bench_v3_release.patched_20260811.yaml (+ unpatched TS extension)
  - unit-aware calculation compare: if the answer's recorded unit and the gold unit parse
    to the same base signature with a known factor, the converted value is also tried and
    the BETTER of raw/converted comparison is kept (credit can only be added, never removed)
  - `retired` column; primary curated outputs exclude retired items

Outputs (baseline files are untouched):
  runs/v3_scored/all_scored_v2.parquet (all rows incl. retired, with column)
  runs/v3_scored/per_cell_summary_v2.parquet (excluding retired)
  article/final/RESCORE_DIFF_20260811.md ( -> per-cell and headline diff)
"""
import json, math, pathlib, re, unicodedata, yaml
import pandas as pd

ROOT = pathlib.Path(os.environ.get("PEXPO_ROOT", "."))
P1, T1 = ROOT / "runs/v3_main", ROOT / "runs/v3_tool_required"
OUT = ROOT / "runs/v3_scored"
GOLD1 = ROOT / "pexpo_bench/samples/pexpo_bench_v3_release.patched_20260811.yaml"
GOLD2 = ROOT / "pexpo_bench/samples/tool_required_extension_v3.yaml"
DIFF_MD = ROOT / "article/final/RESCORE_DIFF_20260811.md"
TAG = "20260811"

ARCHS = ["A0_naive", "A1_context_eng", "A2_rag", "A2p_rag_constrained",
         "A3_agent", "A4_hybrid", "A4p_hybrid_constrained"]
PAPER = ["A0_naive", "A1_context_eng", "A2p_rag_constrained", "A3_agent", "A4p_hybrid_constrained"]
MODELS = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "deepseek-v4"]
LAB = dict(zip(PAPER, ["A0", "A1", "A2", "A3", "A4"]))


# ---------- identical scoring primitives (from build_v3_scored.py) ----------
def norm_tf(x):
    if isinstance(x, bool):
        return x
    s = str(x or "").strip().lower()
    s = re.sub(r"[*`'\".,;:!?#\-\s]+$", "", s)
    s = re.sub(r"^[*`'\".,;:!?#\-\s]+", "", s)
    if not s:
        return None
    if s.startswith("true") or s in ("yes", "correct", "1", "t"):
        return True
    if s.startswith("false") or s in ("no", "incorrect", "0", "f"):
        return False
    return None


def extract_num(x):
    if isinstance(x, (int, float)):
        return float(x)
    if x is None:
        return None
    s = str(x)
    m = re.search(r"[-+]?\d*\.?\d+\s*[×x]\s*10\s*[⁻\-]?\s*\d+", s)
    if m:
        clean = re.sub(r"[×x]\s*10\s*[⁻\-]?\s*", "e-", m.group())
        try:
            return float(clean)
        except Exception:
            pass
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    return float(m.group()) if m else None


# ---------- unit-aware comparison ----------
_ATOM = {
    # base name, factor to canonical base
    "kg": ("g", 1e3), "g": ("g", 1.0), "mg": ("g", 1e-3), "μg": ("g", 1e-6), "ng": ("g", 1e-9), "pg": ("g", 1e-12),
    "m3": ("l", 1e3), "cm3": ("l", 1e-3), "l": ("l", 1.0), "dl": ("l", 1e-1), "ml": ("l", 1e-3), "μl": ("l", 1e-6),
    "m2": ("m2", 1.0), "cm2": ("m2", 1e-4),
    "m": ("m", 1.0), "cm": ("m", 1e-2), "mm": ("m", 1e-3), "km": ("m", 1e3),
    "day": ("day", 1.0), "d": ("day", 1.0), "days": ("day", 1.0),
    "hour": ("day", 1 / 24), "hr": ("day", 1 / 24), "h": ("day", 1 / 24), "hours": ("day", 1 / 24),
    "min": ("day", 1 / 1440), "s": ("day", 1 / 86400), "sec": ("day", 1 / 86400),
    "year": ("year", 1.0), "yr": ("year", 1.0), "years": ("year", 1.0),
    "deaths": ("deaths", 1.0), "death": ("deaths", 1.0), "cases": ("cases", 1.0), "case": ("cases", 1.0),
    "ppb": ("ppb", 1.0), "ppm": ("ppb", 1e3),
}
_DIMLESS = {"", "fraction", "unitless", "dimensionless", "ratio", "(dimensionless)", "-", "none"}


def parse_unit(u):
    """Return (signature, factor) or None. Signature = (tuple(num bases), tuple(den bases))."""
    if u is None:
        return None
    s = unicodedata.normalize("NFKC", str(u)).strip().lower()
    s = s.replace("µ", "μ").replace("³", "3").replace("²", "2").replace(" ", "")
    s = s.replace("(", "").replace(")", "")
    if s in _DIMLESS:
        return (("1",), ()), 1.0
    parts = s.split("/")
    if not parts or len(parts) > 3:
        return None
    num_toks = [t for t in re.split(r"[·.*]", parts[0]) if t]
    den_toks = []
    for p in parts[1:]:
        den_toks += [t for t in re.split(r"[·.*\-]", p) if t]
    if not num_toks:
        return None
    factor, num, den = 1.0, [], []
    for t in num_toks:
        if t not in _ATOM:
            return None
        b, f = _ATOM[t]
        num.append(b)
        factor *= f
    for t in den_toks:
        if t not in _ATOM:
            return None
        b, f = _ATOM[t]
        den.append(b)
        factor /= f
    return (tuple(sorted(num)), tuple(sorted(den))), factor


def calc_score(pn, gn, tol):
    if pn is None or gn is None:
        return 0.0
    if gn == 0:
        return 1.0 if abs(pn) < 1e-9 else 0.0
    rel = abs(pn - gn) / abs(gn)
    if rel <= tol:
        return 1.0
    if rel <= 2 * tol:
        return 0.5
    return 0.0


# ---------- judge scores ----------
_OPEN_JUDGE = {}
jf = ROOT / "runs/v3_main/_open_judge/per_row.jsonl"
if jf.exists():
    for line in jf.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            _OPEN_JUDGE[(r["model"], r["arch"], r["qid"])] = r["score"]
print(f"open-judge rows: {len(_OPEN_JUDGE)}")


def score_row(r, gq, model, arch):
    err = r.get("error_msg") or ""
    if r.get("parse_error") or (err and not err.startswith("route=")):
        return 0.0, False
    qt = gq.get("question_type", "")
    ga, pred = gq.get("answer"), r.get("answer")
    if qt == "true_false":
        p, g = norm_tf(pred), norm_tf(ga)
        return (1.0 if (p is not None and g is not None and p == g) else 0.0), False
    if qt == "calculation":
        pn = extract_num(pred)
        gn = float(ga) if isinstance(ga, (int, float)) else extract_num(ga)
        tol = gq.get("tolerance", 0.10)
        raw = calc_score(pn, gn, tol)
        conv = 0.0
        pu, gu = parse_unit(r.get("unit")), parse_unit(gq.get("unit"))
        if pn is not None and pu and gu and pu[0] == gu[0] and pu[1] != gu[1]:
            conv = calc_score(pn * pu[1] / gu[1], gn, tol)
        return max(raw, conv), conv > raw
    if qt == "open_ended":
        key = (model, arch, r.get("qid"))
        if key in _OPEN_JUDGE:
            return _OPEN_JUDGE[key], False
        gw = set(re.findall(r"\b[a-zA-Z]{3,}\b", str(ga or "").lower()))
        pw = set(re.findall(r"\b[a-zA-Z]{3,}\b", str(pred or "").lower()))
        if not gw:
            return 0.5, False
        return min(len(gw & pw) / len(gw), 1.0), False
    return 0.0, False


# ---------- load gold ----------
g1 = {q["qid"]: q for q in yaml.safe_load(GOLD1.read_text())}
g2 = {q["qid"]: q for q in yaml.safe_load(GOLD2.read_text())}
gold = {**g1, **g2}
retired = {q for q, it in gold.items() if it.get(f"_retired_{TAG}")}
print(f"gold items: {len(gold)}; retired: {len(retired)}")

# ---------- consolidate ----------
rows = []
for model in MODELS:
    for arch in ARCHS:
        for root in [P1, T1]:
            f = root / model / arch / "run_1.jsonl"
            if not f.exists():
                continue
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                gq = gold.get(r.get("qid"))
                if not gq:
                    continue
                sc, via_unit = score_row(r, gq, model, arch)
                rows.append({
                    "model": model, "arch": arch, "qid": r["qid"],
                    "subdomain": gq.get("subdomain"), "question_type": gq.get("question_type"),
                    "difficulty": gq.get("difficulty", "medium"),
                    "score": sc, "unit_converted": via_unit,
                    "retired": r["qid"] in retired,
                    "in_tokens": r.get("input_tokens", 0) or 0,
                    "out_tokens": r.get("output_tokens", 0) or 0,
                    "latency_s": r.get("total_latency_s", 0) or 0,
                    "n_tools": len(r.get("tool_calls") or []),
                    "parse_error": bool(r.get("parse_error")),
                    "error_msg": r.get("error_msg") or "",
                })
df = pd.DataFrame(rows)
df.to_parquet(OUT / "all_scored_v2.parquet")
act = df[~df.retired].copy()
print(f"rows: {len(df)} (active {len(act)}); cells: {df.groupby(['model','arch']).ngroups}")

# per-cell summary (active only), same cost model as for comparability
COST = {"gpt-5.4": {"in": 5e-3, "out": 15e-3}, "gpt-5.4-nano": {"in": 1.5e-4, "out": 6e-4},
        "deepseek-v4": {"in": 2.7e-4, "out": 1.1e-3}}
act["cost_usd"] = act.apply(lambda r: (r.in_tokens / 1000) * COST.get(r.model, {"in": 1e-3})["in"]
                            + (r.out_tokens / 1000) * COST.get(r.model, {"out": 3e-3})["out"], axis=1)
cell = act.groupby(["model", "arch"]).agg(n=("score", "size"), acc=("score", "mean"),
                                          in_tok=("in_tokens", "mean"), out_tok=("out_tokens", "mean"),
                                          cost_per_q=("cost_usd", "mean"),
                                          cost_per_100q=("cost_usd", lambda x: x.sum() * 100 / len(x)))
cell.to_parquet(OUT / "per_cell_summary_v2.parquet")

# ---------- diff vs baseline ----------
baseline = pd.read_parquet(OUT / "all_scored.parquet")
m = baseline.merge(df[["model", "arch", "qid", "score", "unit_converted", "retired"]],
             on=["model", "arch", "qid"], suffixes=("_v1", "_v2"))
chg = m[m.score_v1 != m.score_v2]
gain_qids = chg.groupby("qid").size().sort_values(ascending=False)
unit_gain = m[m.unit_converted & (m.score_v2 > m.score_v1)].qid.nunique()

L = ["# Curated-rescore diff — 2026-08-11", "",
     "Baseline = `all_scored.parquet` (original golds, unit-blind scorer, all 1,104 items).",
     "Curated = `all_scored_v2.parquet` (patched golds, unit-aware scorer; primary numbers exclude "
     f"the {len(retired)} retired items → n = {len(gold) - len(retired)}).",
     "No model or judge was re-run; every score derives from archived trajectories.", "",
     f"- Rows with changed scores: {len(chg)} (across {chg.qid.nunique()} qids)",
     f"- qids gaining via unit conversion: {unit_gain}",
     f"- Score decreases: {len(chg[chg.score_v2 < chg.score_v1])} rows "
     "(possible only via gold fixes; expected ~0 since fixed items scored 0 under the original golds)", "",
     "## Changed items (rows changed across 28 cells)", ""]
for q, n in gain_qids.head(20).items():
    mean1 = m[m.qid == q].score_v1.mean()
    mean2 = m[m.qid == q].score_v2.mean()
    L.append(f"- {q}: {n} cells changed (item mean {mean1:.2f} → {mean2:.2f})")

L += ["", "## Paper-arm accuracy (%), baseline (n=1,104) → curated (n=1,030)", "",
      "| Model | " + " | ".join(LAB[a] for a in PAPER) + " |", "|---|" + "---|" * len(PAPER)]
for mod in MODELS:
    parts = []
    for a in PAPER:
        a1 = baseline[(baseline.model == mod) & (baseline.arch == a)].score.mean() * 100
        a2 = act[(act.model == mod) & (act.arch == a)].score.mean() * 100
        parts.append(f"{a1:.1f} → {a2:.1f}")
    L.append(f"| {mod} | " + " | ".join(parts) + " |")

L += ["", "## Cross-model means (paper arms)", "", "| Arm | baseline | curated | Δ |", "|---|---|---|---|"]
for a in PAPER:
    a1 = baseline[baseline.arch == a].groupby("model").score.mean().mean() * 100
    a2 = act[act.arch == a].groupby("model").score.mean().mean() * 100
    L.append(f"| {LAB[a]} | {a1:.1f} | {a2:.1f} | {a2 - a1:+.1f} |")

L += ["", "## By question type, cross-model means (baseline → curated)", ""]
for qt in ["calculation", "true_false", "open_ended"]:
    row = []
    for a in PAPER:
        a1 = baseline[(baseline.arch == a) & (baseline.question_type == qt)].groupby("model").score.mean().mean() * 100
        a2 = act[(act.arch == a) & (act.question_type == qt)].groupby("model").score.mean().mean() * 100
        row.append(f"{LAB[a]} {a1:.1f}→{a2:.1f}")
    L.append(f"- **{qt}**: " + "; ".join(row))

L += ["", f"Retired qids and reasons: see `BANK_CHANGELOG_{TAG}.md`. Cost columns keep the original "
      "price table for comparability and remain descriptive only.", ""]
DIFF_MD.write_text("\n".join(L), encoding="utf-8")
print(f"diff report: {DIFF_MD.name}")
print("\n=== Cross-model means (paper arms), baseline -> curated ===")
for a in PAPER:
    a1 = baseline[baseline.arch == a].groupby("model").score.mean().mean() * 100
    a2 = act[act.arch == a].groupby("model").score.mean().mean() * 100
    print(f"  {LAB[a]}: {a1:.1f} -> {a2:.1f}")
