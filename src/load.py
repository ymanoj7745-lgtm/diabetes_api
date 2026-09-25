"""
Data loading utilities for NFHS-5 .DTA files.
"""
import pyreadstat
import pandas as pd
from src.config import DATA_RAW, HOUSEHOLD_FILE, WOMEN_FILE, MEN_FILE


def load_household():
    """Load household recode (contains biomarker/CAB data)."""
    df, meta = pyreadstat.read_dta(
        DATA_RAW / HOUSEHOLD_FILE,
        convert_categoricals=False,
    )
    return df, meta


def load_women():
    """Load individual women's recode."""
    df, meta = pyreadstat.read_dta(
        DATA_RAW / WOMEN_FILE,
        convert_categoricals=False,
    )
    return df, meta


def load_men():
    """Load individual men's recode."""
    df, meta = pyreadstat.read_dta(
        DATA_RAW / MEN_FILE,
        convert_categoricals=False,
    )
    return df, meta


def scan_variables(df, meta, keywords=None):
    """
    Search variable labels for keywords.
    Returns list of (variable_name, label) tuples.
    """
    if keywords is None:
        keywords = [
            "glucose", "diab", "systolic", "diastolic",
            "blood pressure", "waist", "arm", "sugar",
            "hemoglobin", "hba1c",
        ]

    matches = []
    for var in df.columns:
        label = meta.column_names_to_labels.get(var, "").lower()
        if any(kw in label for kw in keywords) or var.startswith(("sb", "sh")):
            matches.append((var, meta.column_names_to_labels.get(var, "")))
    return matches
