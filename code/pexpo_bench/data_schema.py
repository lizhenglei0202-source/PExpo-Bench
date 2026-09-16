"""Configuration identifiers used by the paper and its recorded datasets."""

MAIN_CONFIGURATIONS = ("A0", "A1", "A2", "A3", "A4")
FACTORIAL_CONFIGURATIONS = ("A3", "F100", "F010", "F001", "F110", "F101", "F011", "A4")
SUBJECT_MODELS = ("gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "deepseek-v4")

# Stored identifiers are data keys, not additional experimental configurations.
RECORD_IDS = {
    "A0": "A0_naive",
    "A1": "A1_context_eng",
    "A2": "A2p_rag_constrained",
    "A3": "A3_agent",
    "A4": "A4p_hybrid_constrained",
    "F100": "fA3_R",
    "F010": "fA3_P",
    "F001": "fA3_B",
    "F110": "fA3_RP",
    "F101": "A4_hybrid",
    "F011": "fA3_PB",
}
CONFIGURATION_IDS = {record: config for config, record in RECORD_IDS.items()}

def configuration_id(value):
    """Read either a stored data key or a manuscript configuration label."""
    if value in RECORD_IDS:
        return value
    return CONFIGURATION_IDS[value]

def record_id(configuration):
    """Locate the unchanged data files associated with a configuration."""
    return RECORD_IDS[configuration_id(configuration)]
