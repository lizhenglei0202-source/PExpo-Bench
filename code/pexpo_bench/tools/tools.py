"""PExpo-Bench reference tool implementations (v0.1).

All tools return dicts. Agent harness wraps these with JSON-schema adapters.
Keep implementations deterministic so evaluation is reproducible.
"""
from __future__ import annotations

import math
import signal
from typing import Any

# --------------------------------------------------------------------------
# 1. unit_converter
# --------------------------------------------------------------------------
_UNIT_TABLE = {
    ("min", "h"): 1 / 60,
    ("h", "min"): 60,
    ("s", "h"): 1 / 3600,
    ("day", "h"): 24,
    ("L", "mL"): 1000,
    ("mL", "L"): 1 / 1000,
    ("m3", "L"): 1000,
    ("μg/m³", "mg/m³"): 1 / 1000,
    ("mg/m³", "μg/m³"): 1000,
    ("kg", "g"): 1000,
    ("g", "mg"): 1000,
}


def unit_converter(value: float, from_: str, to: str) -> dict:
    if from_ == to:
        return {"value": value, "unit": to}
    k = _UNIT_TABLE.get((from_, to))
    if k is None:
        raise ValueError(f"Unsupported conversion: {from_} -> {to}")
    return {"value": value * k, "unit": to}


# --------------------------------------------------------------------------
# 2. dose_calculator
# --------------------------------------------------------------------------
def dose_calculator(C: float, IR: float, ET: float, BW: float | None = None) -> dict:
    """Dose = C × IR × ET (optionally / BW).

    Units expected:
      C  [μg/m³] or [mg/L]
      IR [m³/h]  or [L/day]
      ET [h]     or [day]
    """
    dose = C * IR * ET
    if BW is not None:
        dose = dose / BW
        return {"dose": dose, "unit": "μg/(kg·h) or mg/(kg·day)"}
    return {"dose": dose, "unit": "μg or mg"}


# --------------------------------------------------------------------------
# 3. mppd_deposition (simplified ICRP 66 lookup)
# --------------------------------------------------------------------------
# 极简查表，仅用于 benchmark；真实分析应用 MPPD software
_MPPD_TABLE = {
    # (size_um, region): fraction, breathing=resting
    (0.01, "alveolar"): 0.25,
    (0.1,  "alveolar"): 0.35,
    (1.0,  "alveolar"): 0.18,
    (2.5,  "alveolar"): 0.10,
    (10.0, "alveolar"): 0.02,
    (0.01, "head"): 0.10,
    (0.1,  "head"): 0.05,
    (1.0,  "head"): 0.15,
    (2.5,  "head"): 0.45,
    (10.0, "head"): 0.85,
    (0.01, "TB"): 0.20,
    (0.1,  "TB"): 0.10,
    (1.0,  "TB"): 0.08,
    (2.5,  "TB"): 0.10,
    (10.0, "TB"): 0.05,
}


def mppd_deposition(particle_size_um: float, region: str, breathing: str = "resting") -> dict:
    sizes = sorted({s for (s, _) in _MPPD_TABLE})
    nearest = min(sizes, key=lambda x: abs(math.log10(x) - math.log10(particle_size_um)))
    if region == "total":
        frac = sum(_MPPD_TABLE[(nearest, r)] for r in ("head", "TB", "alveolar"))
    else:
        frac = _MPPD_TABLE[(nearest, region)]
    return {"fraction": frac, "nearest_size_um": nearest, "breathing": breathing}


# --------------------------------------------------------------------------
# 4. airquality_lookup  (offline cache stub)
# --------------------------------------------------------------------------
_AQ_CACHE: dict[tuple, dict] = {}  # populated from snapshot CSV in production


def airquality_lookup(lat: float, lon: float, time: str, pollutant: str) -> dict:
    key = (round(lat, 2), round(lon, 2), time[:10], pollutant)
    if key in _AQ_CACHE:
        return _AQ_CACHE[key]
    # Fallback default (for pilot); replace with real snapshot later.
    defaults = {"PM2.5": 20.0, "PM10": 35.0, "NO2": 30.0, "O3": 50.0, "CO": 0.5, "SO2": 5.0}
    return {
        "value": defaults.get(pollutant, 0.0),
        "unit": "μg/m³" if pollutant != "CO" else "mg/m³",
        "station_id": "default_fallback",
    }


# --------------------------------------------------------------------------
# 5. trajectory_match  (stub; expects CSV of lat,lon,t,speed)
# --------------------------------------------------------------------------
def trajectory_match(gps_csv: str, microenv_db: str = "default") -> dict:
    # Pilot stub: in production this runs land-use + dwell-time heuristic
    return {
        "segments": [
            {"start": "07:00", "end": "08:00", "microenv": "home_indoor", "duration_h": 1.0},
            {"start": "08:00", "end": "09:00", "microenv": "in_transit", "duration_h": 1.0},
            {"start": "09:00", "end": "18:00", "microenv": "office_indoor", "duration_h": 9.0},
        ]
    }


# --------------------------------------------------------------------------
# 6. exposure_factor_lookup  (EFH default values subset)
# --------------------------------------------------------------------------
_EFH = {
    # Inhalation rates (m³/day) — EFH Table 6-1
    ("inhalation_rate", "both", "<1",     "long_term"): (5.4,  "m³/day", "EFH Table 6-1"),
    ("inhalation_rate", "both", "1-2",    "long_term"): (8.0,  "m³/day", "EFH Table 6-1"),
    ("inhalation_rate", "both", "2-3",    "long_term"): (8.9,  "m³/day", "EFH Table 6-1"),
    ("inhalation_rate", "both", "3-6",    "long_term"): (10.1, "m³/day", "EFH Table 6-1"),
    ("inhalation_rate", "both", "6-11",   "long_term"): (12.0, "m³/day", "EFH Table 6-1"),
    ("inhalation_rate", "both", "11-16",  "long_term"): (15.2, "m³/day", "EFH Table 6-1"),
    ("inhalation_rate", "both", "16-21",  "long_term"): (16.3, "m³/day", "EFH Table 6-1"),
    ("inhalation_rate", "both", "adult",  "long_term"): (15.7, "m³/day", "EFH Table 6-1"),
    ("inhalation_rate", "M",    "adult",  "long_term"): (16.0, "m³/day", "EFH Table 6-1"),
    ("inhalation_rate", "F",    "adult",  "long_term"): (12.0, "m³/day", "EFH Table 6-1"),
    # Activity-specific rates (m³/h) — EFH Table 6-2
    ("inhalation_rate", "M",    "adult",  "sleep"):     (0.45, "m³/h", "EFH Table 6-2"),
    ("inhalation_rate", "M",    "adult",  "light"):     (1.5,  "m³/h", "EFH Table 6-2"),
    ("inhalation_rate", "M",    "adult",  "moderate"):   (2.5,  "m³/h", "EFH Table 6-2"),
    ("inhalation_rate", "M",    "adult",  "heavy"):      (4.8,  "m³/h", "EFH Table 6-2"),
    ("inhalation_rate", "F",    "adult",  "light"):      (1.25, "m³/h", "EFH Table 6-2"),
    ("inhalation_rate", "F",    "adult",  "moderate"):   (1.6,  "m³/h", "EFH Table 6-2"),
    ("inhalation_rate", "F",    "adult",  "heavy"):      (2.9,  "m³/h", "EFH Table 6-2"),
    # Body weight (kg) — EFH Table 8-1
    ("body_weight", "both", "<1",     "long_term"): (7.8,  "kg", "EFH Table 8-1"),
    ("body_weight", "both", "1-2",    "long_term"): (11.4, "kg", "EFH Table 8-1"),
    ("body_weight", "both", "2-3",    "long_term"): (13.8, "kg", "EFH Table 8-1"),
    ("body_weight", "both", "3-6",    "long_term"): (18.6, "kg", "EFH Table 8-1"),
    ("body_weight", "both", "6-11",   "long_term"): (31.8, "kg", "EFH Table 8-1"),
    ("body_weight", "both", "11-16",  "long_term"): (56.8, "kg", "EFH Table 8-1"),
    ("body_weight", "both", "16-21",  "long_term"): (71.6, "kg", "EFH Table 8-1"),
    ("body_weight", "both", "adult",  "long_term"): (80.0, "kg", "EFH Table 8-1"),
    ("body_weight", "M",    "adult",  "long_term"): (86.4, "kg", "EFH Table 8-1"),
    ("body_weight", "F",    "adult",  "long_term"): (73.9, "kg", "EFH Table 8-1"),
    # Skin surface area (m²) — EFH Table 7-1
    ("skin_area", "both", "<1",     "long_term"): (0.30, "m²", "EFH Table 7-1"),
    ("skin_area", "both", "1-2",    "long_term"): (0.53, "m²", "EFH Table 7-1"),
    ("skin_area", "both", "3-6",    "long_term"): (0.76, "m²", "EFH Table 7-1"),
    ("skin_area", "both", "6-11",   "long_term"): (1.08, "m²", "EFH Table 7-1"),
    ("skin_area", "both", "11-16",  "long_term"): (1.55, "m²", "EFH Table 7-1"),
    ("skin_area", "M",    "adult",  "long_term"): (2.03, "m²", "EFH Table 7-1"),
    ("skin_area", "F",    "adult",  "long_term"): (1.79, "m²", "EFH Table 7-1"),
    # Drinking water (mL/day) — EFH Table 3-1
    ("drinking_water", "both", "<1",     "long_term"): (302,  "mL/day", "EFH Table 3-1"),
    ("drinking_water", "both", "1-2",    "long_term"): (271,  "mL/day", "EFH Table 3-1"),
    ("drinking_water", "both", "3-6",    "long_term"): (378,  "mL/day", "EFH Table 3-1"),
    ("drinking_water", "both", "6-11",   "long_term"): (503,  "mL/day", "EFH Table 3-1"),
    ("drinking_water", "both", "11-16",  "long_term"): (681,  "mL/day", "EFH Table 3-1"),
    ("drinking_water", "both", "16-21",  "long_term"): (770,  "mL/day", "EFH Table 3-1"),
    ("drinking_water", "both", "adult",  "long_term"): (1053, "mL/day", "EFH Table 3-1"),
    # Soil ingestion (mg/day) — EFH Table 5-1
    ("soil_ingestion", "both", "child",  "long_term"): (100,  "mg/day", "EFH Table 5-1"),
    ("soil_ingestion", "both", "adult",  "long_term"): (50,   "mg/day", "EFH Table 5-1"),
    ("soil_ingestion", "both", "pica",   "long_term"): (1000, "mg/day", "EFH Table 5-1"),
    # Time-activity (hours/day) — EFH Table 16-1
    ("time_indoors",    "both", "adult", "long_term"): (20.0, "h/day", "EFH Table 16-1"),
    ("time_outdoors",   "both", "adult", "long_term"): (1.8,  "h/day", "EFH Table 16-1"),
    ("time_in_vehicle", "both", "adult", "long_term"): (1.5,  "h/day", "EFH Table 16-1"),
}

# Aliases: model may use various names
_FACTOR_ALIASES = {
    "inhalation rate": "inhalation_rate",
    "inhalation_rate": "inhalation_rate",
    "ir": "inhalation_rate",
    "breathing rate": "inhalation_rate",
    "ventilation rate": "inhalation_rate",
    "body weight": "body_weight",
    "body_weight": "body_weight",
    "bw": "body_weight",
    "weight": "body_weight",
    "skin area": "skin_area",
    "skin_area": "skin_area",
    "skin surface area": "skin_area",
    "bsa": "skin_area",
    "body surface area": "skin_area",
    "skin surface area to body weight ratio": "skin_area",
    "drinking water": "drinking_water",
    "drinking_water": "drinking_water",
    "water intake": "drinking_water",
    "water ingestion": "drinking_water",
    "water ingestion rate": "drinking_water",
    "drinking water ingestion rate": "drinking_water",
    "soil ingestion": "soil_ingestion",
    "soil_ingestion": "soil_ingestion",
    "soil ingestion rate": "soil_ingestion",
    "food intake": "food_intake",
    "food_intake": "food_intake",
    "time indoors": "time_indoors",
    "time outdoors": "time_outdoors",
}


def _resolve_age_group(age) -> str:
    """Map numeric age or string to EFH age bracket."""
    if age is None:
        return "adult"
    if isinstance(age, str):
        a = age.lower().strip()
        if a in ("adult", "adults"): return "adult"
        if a in ("child", "children", "kid"): return "child"
        if a in ("infant", "baby", "newborn"): return "<1"
        if a in ("pica",): return "pica"
        if a in ("all",): return "adult"  # fallback
        # Recognize EFH bracket labels passed directly: '<1', '1-2', '3-6', etc.
        if a in ("<1", "1-2", "2-3", "3-6", "6-11", "11-16", "16-21"):
            return a
        # Some models write '0-1' for '<1'
        if a in ("0-1", "0 to 1", "<1y", "<1 year"): return "<1"
        # Try parsing "6" or numeric value
        try:
            age = float(a)
        except ValueError:
            return "adult"
    # Numeric age → bracket
    age = float(age)
    if age < 1:    return "<1"
    if age < 2:    return "1-2"
    if age < 3:    return "2-3"
    if age < 6:    return "3-6"
    if age < 11:   return "6-11"
    if age < 16:   return "11-16"
    if age < 21:   return "16-21"
    return "adult"


def exposure_factor_lookup(factor: str, age: float | None = None,
                           sex: str = "both", duration: str = "long_term") -> dict:
    # Normalize factor name
    factor_key = _FACTOR_ALIASES.get(factor.lower().strip(), factor.lower().strip().replace(" ", "_"))
    age_group = _resolve_age_group(age)
    sex = (sex or "both").upper()[0] if sex and sex.lower() not in ("both",) else "both"

    # Try exact match
    key = (factor_key, sex, age_group, duration)
    if key not in _EFH:
        key = (factor_key, "both", age_group, duration)
    if key not in _EFH:
        # Fallback: try adult if child not found
        key = (factor_key, "both", "adult", duration)
    if key not in _EFH:
        key = (factor_key, sex, "adult", duration)
    if key not in _EFH:
        # List available options for this factor
        available = [k for k in _EFH if k[0] == factor_key]
        return {"value": None, "unit": None,
                "error": f"No match for {factor_key}/{sex}/{age_group}/{duration}",
                "available": [f"{k[2]}/{k[1]}/{k[3]}" for k in available],
                "source": "EPA EFH 2011"}
    val, unit, source = _EFH[key]
    return {"value": val, "unit": unit, "age_group": age_group, "source": source}


# --------------------------------------------------------------------------
# 7. python_sandbox
# --------------------------------------------------------------------------
class _Timeout(Exception):
    pass


def _sandbox_worker(code: str, conn) -> None:
    """Runs in a separate PROCESS (thread-safe, hard-killable). Sends one dict on conn."""
    try:
        import math as math_mod
        import statistics as stats_mod
        import numpy as np_mod
        import re as _re

        stdout_buf: list = []

        def _print(*args, **kwargs):
            stdout_buf.append(" ".join(str(a) for a in args))

        safe_builtins = {
            "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
            "round": round, "range": range, "pow": pow, "float": float, "int": int,
            "zip": zip, "all": all, "any": any, "enumerate": enumerate,
            "sorted": sorted, "list": list, "dict": dict, "str": str,
            "tuple": tuple, "set": set, "map": map, "filter": filter,
            "bool": bool, "divmod": divmod, "reversed": reversed,
            "isinstance": isinstance, "print": _print,
        }
        # single exec namespace: avoids the split-globals NameError class of failures
        g = {"__builtins__": safe_builtins, "math": math_mod, "np": np_mod,
             "numpy": np_mod, "statistics": stats_mod}
        # Strip import lines (incl. "import math, statistics") and inject names directly
        code = _re.sub(r"^\s*import\s+(?:math|numpy|statistics)(?:\s*,\s*(?:math|numpy|statistics))*(?:\s+as\s+\w+)?\s*$",
                       "", code, flags=_re.MULTILINE)
        for m in _re.finditer(r"^\s*from\s+(math|numpy|statistics)\s+import\s+(.+)$", code, _re.MULTILINE):
            mod = {"math": math_mod, "numpy": np_mod, "statistics": stats_mod}[m.group(1)]
            for name in m.group(2).split(","):
                name = name.strip().split(" as ")[0].strip()
                if hasattr(mod, name):
                    g[name] = getattr(mod, name)
        code = _re.sub(r"^\s*from\s+(math|numpy|statistics)\s+import\s+.+$", "", code, flags=_re.MULTILINE)
        code = code.strip()
        if not code:
            conn.send({"result": None, "stdout": "no computation after stripping imports"})
            return

        def _coerce(v):
            try:
                if hasattr(v, "item"):
                    v = v.item()
                if hasattr(v, "tolist"):
                    v = v.tolist()
                if isinstance(v, (int, float, bool, str, list, tuple, dict, type(None))):
                    return v
                return str(v)
            except Exception:
                return str(v)

        try:
            result = eval(code, g)
            conn.send({"result": _coerce(result), "stdout": "\n".join(stdout_buf)})
            return
        except SyntaxError:
            pass
        lines = [ln for ln in code.splitlines() if ln.strip()]
        pre_keys = set(g)
        if len(lines) > 1 and (lines[-1][:1].isspace() or lines[-2].rstrip().endswith(":")):
            # last line belongs to an indented block: run whole snippet as-is
            exec(code, g)
            lines = []
        if len(lines) > 1:
            exec("\n".join(lines[:-1]), g)
            last = lines[-1].strip()
            try:
                result = eval(last, g)
                conn.send({"result": _coerce(result), "stdout": "\n".join(stdout_buf)})
                return
            except Exception:
                exec(last, g)
        else:
            exec(code, g)
        new_vars = [k for k in g if k not in pre_keys and not k.startswith("_")]
        result = g.get("_result", g[new_vars[-1]] if new_vars else None)
        conn.send({"result": _coerce(result), "stdout": "\n".join(stdout_buf)})
    except Exception as e:  # noqa: BLE001
        conn.send({"error": f"{type(e).__name__}: {e}"})


def python_sandbox(code: str, timeout_s: int = 5) -> dict:
    """Execute a small arithmetic snippet in an isolated subprocess with a hard timeout.

    Thread-safe (no signals): callable from ThreadPoolExecutor workers. The 2026-08-12 fix
    replaces the previous signal.alarm implementation, which raised
    'signal only works in main thread' under the threaded runner and silently failed
    every call (see QUESTION_BANK_AUDIT/PEER_REVIEW P0 items).
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")  # fork is unsafe under a threaded parent
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_sandbox_worker, args=(code, child), daemon=True)
    proc.start()
    child.close()
    try:
        if parent.poll(timeout_s + 8):  # +8 s covers spawn/numpy import overhead
            out = parent.recv()
        else:
            raise _Timeout(f"python_sandbox timed out after {timeout_s}s")
    finally:
        parent.close()
        if proc.is_alive():
            proc.terminate()
        proc.join(2)
    if "error" in out:
        raise RuntimeError(out["error"])
    return {"result": out.get("result"), "stdout": out.get("stdout", "")}


# --------------------------------------------------------------------------
# Tool registry for agent loop
# --------------------------------------------------------------------------
TOOL_REGISTRY = {
    "unit_converter": unit_converter,
    "dose_calculator": dose_calculator,
    "mppd_deposition": mppd_deposition,
    "airquality_lookup": airquality_lookup,
    "trajectory_match": trajectory_match,
    "exposure_factor_lookup": exposure_factor_lookup,
    "python_sandbox": python_sandbox,
}
