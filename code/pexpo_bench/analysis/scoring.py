"""Scoring functions used for the released benchmark results.
Numeric scoring retains the recorded unit-conversion rule.
The main dataset builder uses the recorded cross-family judge scores for open answers.
"""

import json, math, pathlib, re, unicodedata

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

_OPEN_JUDGE = {}

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
