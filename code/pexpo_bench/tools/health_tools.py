"""Tier-1 + Tier-3 health & risk tools.

8 tools, all deterministic, no network. Designed to match the bank's actual
needs derived from question-bank audit:
  • indoor_air_mass_balance — 134 questions
  • af_calc                 — 77
  • gbd_mortality           — ~50
  • who_aqg_lookup          — 39
  • ier_pm25_rr             — 20
  • iris_lookup             — 20  (10 chemicals only — see IRIS_DB below)
  • noncancer_hq_calc       — 6
  • cotinine_pk_calc        — 11

IRIS values verified against EPA IRIS portal as of 2024.  Each entry carries
`last_revised` and `url` for audit transparency.
"""
from __future__ import annotations
from typing import Optional, Union

# ==========================================================================
# IRIS toxicity database — 10 chemicals (the only ones referenced in the
# 1004-question bank, per chemical-frequency audit 2026-05).
# Units:
#   RfD        mg/kg-day  (oral chronic reference dose)
#   RfC        mg/m³      (inhalation chronic reference concentration)
#   IUR        per µg/m³  (inhalation unit risk, lifetime cancer)
#   oral_CSF   per mg/kg-day  (oral cancer slope factor)
# ==========================================================================
IRIS_DB = {
    "benzene": {
        "RfD": 4e-3, "RfC": 0.03, "IUR": (2.2e-6, 7.8e-6),
        "oral_CSF": (0.015, 0.055), "carcinogen_class": "A",
        "last_revised": "2003-01",
        "url": "https://iris.epa.gov/ChemicalLanding/&substance_nmbr=276",
        "casrn": "71-43-2",
    },
    "formaldehyde": {
        "RfD": 0.2, "RfC": 9.8e-3, "IUR": 1.3e-5,
        "carcinogen_class": "B1", "last_revised": "1991-06",
        "url": "https://iris.epa.gov/ChemicalLanding/&substance_nmbr=419",
        "casrn": "50-00-0",
    },
    "arsenic": {
        "RfD": 3e-4, "IUR": 4.3e-3, "oral_CSF": 1.5,
        "carcinogen_class": "A", "last_revised": "1995-04",
        "url": "https://iris.epa.gov/ChemicalLanding/&substance_nmbr=278",
        "casrn": "7440-38-2",
        "notes": "Inorganic arsenic; 2024 draft proposes lower RfD but not yet effective",
    },
    "lead": {
        "RfD": None, "RfC": None,
        "carcinogen_class": "B2", "last_revised": "1993-07",
        "url": "https://iris.epa.gov/ChemicalLanding/&substance_nmbr=277",
        "casrn": "7439-92-1",
        "notes": "EPA does not derive RfD/RfC for Pb; uses blood-lead level (BLL) targets",
    },
    "cadmium": {
        "RfD": 1e-3, "carcinogen_class": "B1", "last_revised": "1994-02",
        "url": "https://iris.epa.gov/ChemicalLanding/&substance_nmbr=141",
        "casrn": "7440-43-9",
        "notes": "RfD_water=5e-4, RfD_food=1e-3; no IUR derived",
    },
    "chromium_vi": {
        "RfD": 3e-3, "RfC": 1e-4, "IUR": 1.2e-2,
        "carcinogen_class": "A", "last_revised": "1998-09",
        "url": "https://iris.epa.gov/ChemicalLanding/&substance_nmbr=144",
        "casrn": "18540-29-9",
    },
    "benzo_a_pyrene": {
        "RfD": 3e-4, "RfC": 2e-6, "IUR": 6e-4, "oral_CSF": 1.0,
        "carcinogen_class": "B2", "last_revised": "2017-01",
        "url": "https://iris.epa.gov/ChemicalLanding/&substance_nmbr=136",
        "casrn": "50-32-8",
    },
    "toluene": {
        "RfD": 0.08, "RfC": 5.0, "carcinogen_class": "I",
        "last_revised": "2005-09",
        "url": "https://iris.epa.gov/ChemicalLanding/&substance_nmbr=118",
        "casrn": "108-88-3",
    },
    "manganese": {
        "RfD": 0.14, "RfC": 5e-5,
        "carcinogen_class": "D", "last_revised": "1996-11",
        "url": "https://iris.epa.gov/ChemicalLanding/&substance_nmbr=373",
        "casrn": "7439-96-5",
    },
    "cotinine": {
        "RfD": None, "RfC": None,
        "carcinogen_class": "not derived",
        "url": "n/a",
        "notes": "Cotinine is a tobacco-exposure biomarker, not a toxicant with EPA-derived RfD/RfC.",
    },
}

_CHEM_ALIASES = {
    "pb": "lead", "hg": None, "as": "arsenic",
    "cd": "cadmium", "cr": "chromium_vi", "cr6": "chromium_vi",
    "cr(vi)": "chromium_vi", "hexavalent chromium": "chromium_vi",
    "bap": "benzo_a_pyrene", "b(a)p": "benzo_a_pyrene",
    "benzo[a]pyrene": "benzo_a_pyrene", "benzo(a)pyrene": "benzo_a_pyrene",
    "mn": "manganese",
}


# ==========================================================================
# Tool 1: iris_lookup
# ==========================================================================
def iris_lookup(chemical: str, value_type: str = "RfD") -> dict:
    """EPA IRIS toxicity lookup.

    Args:
        chemical: name or alias (case-insensitive).
        value_type: RfD / RfC / IUR / oral_CSF

    Returns: {chemical, value, unit, last_revised, url, casrn, notes}
    """
    key = _CHEM_ALIASES.get(chemical.lower().strip(),
                            chemical.lower().strip().replace(" ", "_"))
    if key is None:
        return {"value": None, "error": f"'{chemical}' is ambiguous; specify (e.g. mercury_elemental, methylmercury)"}
    rec = IRIS_DB.get(key)
    if rec is None:
        from difflib import get_close_matches
        cands = get_close_matches(key, list(IRIS_DB.keys()), n=3, cutoff=0.4)
        return {"value": None,
                "error": f"'{chemical}' not in IRIS table (10 chemicals supported)",
                "did_you_mean": cands,
                "available": sorted(IRIS_DB.keys())}
    v = rec.get(value_type)
    units = {"RfD": "mg/kg-day", "RfC": "mg/m³",
             "IUR": "per µg/m³", "oral_CSF": "per mg/kg-day"}
    if v is None:
        return {"value": None, "chemical": key,
                "error": f"{value_type} not derived by EPA IRIS for {key}",
                "carcinogen_class": rec.get("carcinogen_class"),
                "notes": rec.get("notes"), "url": rec.get("url")}
    return {
        "chemical": key, "value": v, "unit": units.get(value_type, ""),
        "carcinogen_class": rec.get("carcinogen_class"),
        "last_revised": rec.get("last_revised"),
        "casrn": rec.get("casrn"), "url": rec.get("url"),
        "notes": rec.get("notes"),
    }


# ==========================================================================
# Tool 2: indoor_air_mass_balance
# ==========================================================================
def indoor_air_mass_balance(
    C_out: float, P: float = 1.0, AER: float = 0.5,
    k_dep: float = 0.0, S: float = 0.0, V: float = 100.0,
) -> dict:
    """Steady-state indoor concentration:
       C_in = (P·AER·C_out + S/V) / (AER + k_dep)

    Special case: when k_dep = 0 and S = 0, the AER cancels:
       C_in = P · C_out

    Args:
        C_out:  outdoor concentration (µg/m³ or any [Y])
        P:      penetration factor (0-1)
        AER:    air exchange rate (h⁻¹)
        k_dep:  indoor deposition rate (h⁻¹)
        S:      indoor source emission (mass/h)
        V:      room volume (m³)
    """
    denom = AER + k_dep
    if denom <= 0:
        return {"C_in": None, "error": "AER + k_dep must be > 0"}
    C_in = (P * AER * C_out + S / V) / denom
    io_ratio = (P * AER / denom) if C_out > 0 else None
    regime = ("no sinks → C_in = P·C_out (AER cancels)" if k_dep == 0 and S == 0
              else "source-dominated" if S/V > P*AER*C_out and C_out > 0
              else "outdoor-driven" if S == 0
              else "mixed")
    return {
        "C_in": C_in, "unit": "same as C_out",
        "io_ratio": io_ratio,
        "time_constant_h": 1.0 / denom,
        "formula": "C_in = (P·AER·C_out + S/V) / (AER + k_dep)",
        "regime": regime,
        "inputs": {"C_out": C_out, "P": P, "AER": AER, "k_dep": k_dep, "S": S, "V": V},
    }


# ==========================================================================
# Tool 3: af_calc — Attributable Fraction
# ==========================================================================
def af_calc(RR: float, prevalence_exposed: Optional[float] = None) -> dict:
    """Attributable fraction.

    For the exposed group:        AF_e = (RR - 1) / RR
    Population attributable:      PAF  = p(RR - 1) / (1 + p(RR - 1))

    Args:
        RR: relative risk
        prevalence_exposed: optional, fraction exposed in population [0,1]
    """
    af_e = (RR - 1) / RR if RR != 0 else None
    out = {
        "AF_exposed": af_e,
        "formula": "AF = (RR - 1) / RR",
        "interpretation": "Fraction of cases in exposed group attributable to exposure",
    }
    if prevalence_exposed is not None:
        paf = prevalence_exposed * (RR - 1) / (1 + prevalence_exposed * (RR - 1))
        out["PAF_population"] = paf
        out["formula_PAF"] = "PAF = p(RR-1) / (1 + p(RR-1))"
    return out


# ==========================================================================
# Tool 4: gbd_mortality — Attributable cases/deaths
# ==========================================================================
def gbd_mortality(
    population: float,
    baseline_rate_per_100k: float,
    AF: Optional[float] = None,
    RR: Optional[float] = None,
) -> dict:
    """Attributable cases (or deaths) per year.

       cases = population × (baseline_rate / 100,000) × AF

    Args:
        population:              size of cohort
        baseline_rate_per_100k:  cases per 100,000 per year (incidence or mortality)
        AF:                      attributable fraction (preferred)
        RR:                      if AF not given, AF will be computed from (RR-1)/RR
    """
    if AF is None:
        if RR is None:
            return {"cases": None, "error": "Provide AF or RR"}
        AF = (RR - 1) / RR if RR != 0 else 0
    rate = baseline_rate_per_100k / 100_000.0
    cases = population * rate * AF
    return {
        "attributable_cases_per_year": cases,
        "baseline_rate_fraction": rate,
        "AF_used": AF,
        "formula": "cases = pop × (baseline/100k) × AF",
        "inputs": {"population": population, "baseline_per_100k": baseline_rate_per_100k,
                   "AF": AF, "RR": RR},
    }


# ==========================================================================
# Tool 5: who_aqg_lookup — WHO Air Quality Guidelines 2021
# ==========================================================================
WHO_AQG_2021 = {
    # pollutant → window → (value, unit)
    "PM2.5": {"annual": (5, "µg/m³"), "24h": (15, "µg/m³")},
    "PM10":  {"annual": (15, "µg/m³"), "24h": (45, "µg/m³")},
    "NO2":   {"annual": (10, "µg/m³"), "24h": (25, "µg/m³")},
    "O3":    {"peak-season": (60, "µg/m³"), "8h": (100, "µg/m³")},
    "SO2":   {"24h": (40, "µg/m³")},
    "CO":    {"24h": (4, "mg/m³")},
}
WHO_AQG_2005 = {  # previous; for compare/contrast questions
    "PM2.5": {"annual": (10, "µg/m³"), "24h": (25, "µg/m³")},
    "PM10":  {"annual": (20, "µg/m³"), "24h": (50, "µg/m³")},
    "NO2":   {"annual": (40, "µg/m³"), "1h": (200, "µg/m³")},
    "O3":    {"8h": (100, "µg/m³")},
    "SO2":   {"24h": (20, "µg/m³"), "10min": (500, "µg/m³")},
    "CO":    {"15min": (100, "mg/m³"), "1h": (35, "mg/m³"), "8h": (10, "mg/m³")},
}


def who_aqg_lookup(pollutant: str, window: str = "annual",
                   version: str = "2021") -> dict:
    """WHO Air Quality Guideline lookup.

    Args:
        pollutant: PM2.5, PM10, NO2, O3, SO2, CO (case-sensitive standard)
        window:    annual, 24h, 8h, peak-season (depends on pollutant)
        version:   "2021" (default) or "2005"
    """
    db = WHO_AQG_2021 if version == "2021" else WHO_AQG_2005
    pol = pollutant.upper().replace(' ', '').replace('-', '')
    # Normalize
    aliases = {"PM25": "PM2.5", "PM2.5": "PM2.5", "PM10": "PM10",
               "NO2": "NO2", "O3": "O3", "OZONE": "O3",
               "SO2": "SO2", "CO": "CO"}
    pol = aliases.get(pol, pollutant)
    if pol not in db:
        return {"value": None,
                "error": f"Pollutant '{pollutant}' not in AQG {version}",
                "available_pollutants": list(db.keys())}
    if window not in db[pol]:
        return {"value": None,
                "error": f"Averaging window '{window}' not in AQG {version} for {pol}",
                "available_windows": list(db[pol].keys())}
    v, unit = db[pol][window]
    return {
        "pollutant": pol, "window": window, "version": version,
        "value": v, "unit": unit,
        "url": "https://www.who.int/publications/i/item/9789240034228" if version == "2021"
               else "https://www.who.int/publications/i/item/WHO-SDE-PHE-OEH-06-02",
    }


# ==========================================================================
# Tool 6: ier_pm25_rr — GBD 2019 IER for PM2.5 mortality / morbidity
# ==========================================================================
# Per-10 µg/m³ relative-risk coefficients (GBD 2019 supplementary)
IER_COEF = {
    "IHD":    {"RR_per_10": 1.23, "TMREL_low": 2.4, "TMREL_high": 5.9},
    "stroke": {"RR_per_10": 1.24, "TMREL_low": 2.4, "TMREL_high": 5.9},
    "COPD":   {"RR_per_10": 1.14, "TMREL_low": 2.4, "TMREL_high": 5.9},
    "LC":     {"RR_per_10": 1.10, "TMREL_low": 2.4, "TMREL_high": 5.9},   # lung cancer
    "LRI":    {"RR_per_10": 1.07, "TMREL_low": 2.4, "TMREL_high": 5.9},   # lower respiratory infections
}


def ier_pm25_rr(C: float, endpoint: str = "IHD",
                tmrel: float = 5.0) -> dict:
    """GBD 2019 IER-based RR for PM2.5 exposure above TMREL.

    Simplified log-linear extrapolation: RR(C) = RR_per_10 ^ ((C - tmrel) / 10)
    (GBD's actual IER is piecewise; this is the linear-log approximation used
    in policy analyses.)

    Args:
        C:        ambient PM2.5 concentration (µg/m³)
        endpoint: one of IHD, stroke, COPD, LC, LRI
        tmrel:    TMREL value (default 5; range 2.4-5.9)
    """
    if endpoint not in IER_COEF:
        return {"RR": None, "error": f"Endpoint '{endpoint}' not supported",
                "available": list(IER_COEF.keys())}
    coef = IER_COEF[endpoint]
    delta = max(0, C - tmrel)
    RR = coef["RR_per_10"] ** (delta / 10)
    AF = (RR - 1) / RR if RR > 0 else 0
    return {
        "RR": RR, "AF": AF,
        "delta_C": delta, "tmrel_used": tmrel,
        "endpoint": endpoint,
        "RR_per_10": coef["RR_per_10"],
        "formula": "RR = RR_per_10 ^ ((C - TMREL) / 10)",
        "source": "GBD 2019 supplementary; log-linear approximation of IER",
    }


# ==========================================================================
# Tool 7: noncancer_hq_calc — Hazard Quotient
# ==========================================================================
def noncancer_hq_calc(exposure: float, reference: float,
                      route: str = "inhalation") -> dict:
    """Hazard Quotient = exposure / reference.

    Args:
        exposure:  C (mg/m³) for inhalation, or ADD (mg/kg-day) for oral
        reference: RfC (mg/m³) or RfD (mg/kg-day) accordingly
        route:     'oral' or 'inhalation'
    """
    if reference is None or reference == 0:
        return {"HQ": None, "error": "reference must be > 0"}
    hq = exposure / reference
    return {
        "HQ": hq, "exposure": exposure, "reference": reference, "route": route,
        "formula": f"HQ = {'C' if route=='inhalation' else 'ADD'} / "
                   f"{'RfC' if route=='inhalation' else 'RfD'}",
        "interpretation": ("HQ < 1: no appreciable risk" if hq < 1
                           else "HQ ≥ 1: potential concern"),
        "note": "HQ is a ratio; HI = Σ HQ across chemicals.",
    }


# ==========================================================================
# Tool 8: cotinine_pk_calc — Cotinine → blood/body burden
# ==========================================================================
def cotinine_pk_calc(concentration_ng_per_mL: float,
                     blood_volume_L: float = 5.0,
                     output_unit: str = "µg") -> dict:
    """Convert plasma/blood cotinine concentration to total body amount.

    total_amount = concentration (ng/mL) × blood_volume (mL)
                 = concentration × blood_volume_L × 1000 (ng)

    Then convert to µg or mg as requested.
    """
    total_ng = concentration_ng_per_mL * (blood_volume_L * 1000)
    out = {
        "total_ng": total_ng,
        "concentration_ng_per_mL": concentration_ng_per_mL,
        "blood_volume_L": blood_volume_L,
        "formula": "total_ng = concentration_ng/mL × blood_volume_mL",
    }
    if output_unit == "µg":
        out["total_µg"] = total_ng / 1000
        out["unit"] = "µg"
    elif output_unit == "mg":
        out["total_mg"] = total_ng / 1_000_000
        out["unit"] = "mg"
    else:
        out["unit"] = "ng"
    return out


# ==========================================================================
# Module-level registry
# ==========================================================================
HEALTH_TOOLS = {
    "iris_lookup": iris_lookup,
    "indoor_air_mass_balance": indoor_air_mass_balance,
    "af_calc": af_calc,
    "gbd_mortality": gbd_mortality,
    "who_aqg_lookup": who_aqg_lookup,
    "ier_pm25_rr": ier_pm25_rr,
    "noncancer_hq_calc": noncancer_hq_calc,
    "cotinine_pk_calc": cotinine_pk_calc,
}
