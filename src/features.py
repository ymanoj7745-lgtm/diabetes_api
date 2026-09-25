"""
Feature engineering for NFHS-5 diabetes prediction.
"""
import pandas as pd
import numpy as np
from src.config import SBP_VARS, DBP_VARS


# DHS special missing codes
DHS_MISSING = [9, 99, 999, 9999, 99999, 999999]


def clean_dhs(df):
    """Replace DHS special missing codes with NaN."""
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace(DHS_MISSING, np.nan)
    return df


def build_target(df, glucose_col, medicine_col=None, self_report_col=None, threshold=140):
    """
    Create binary diabetes target per NFHS-5 Report Table 12.5.1:
      diabetes = (RBG > threshold) OR (taking glucose medicine) OR (self-reported diabetes)

    Parameters
    ----------
    df : DataFrame
    glucose_col : str — random blood glucose variable (sb74)
    medicine_col : str or None — taking medicine to lower glucose (sb57)
    self_report_col : str or None — currently has diabetes (s728a)
    threshold : int — mg/dL cutoff (default 140)

    Returns
    -------
    DataFrame with 'diabetes' column appended
    """
    df = df.copy()
    df["diabetes"] = 0
    df.loc[df[glucose_col] > threshold, "diabetes"] = 1
    if medicine_col and medicine_col in df.columns:
        df.loc[df[medicine_col] == 1, "diabetes"] = 1
    if self_report_col and self_report_col in df.columns:
        df.loc[df[self_report_col] == 1, "diabetes"] = 1
    # Missing glucose → missing target
    df.loc[df[glucose_col].isna(), "diabetes"] = np.nan
    return df


def build_features(df):
    """
    Build ML feature matrix from an NFHS-5 individual recode DataFrame.

    Expected input columns (DHS standard names):
        v012, v025, v106, v190, v024, v445,
        sb17a-c, sb18a-c, v463a, v463c
    """
    out = pd.DataFrame(index=df.index)

    # ── Demographics ──
    out["age"] = df.get("v012")
    out["sex_encoded"] = df.get("sex_encoded", pd.Series(0, index=df.index))
    out["urban"] = (df.get("v025", pd.Series(np.nan, index=df.index)) == 1).astype(int)
    out["education"] = df.get("v106")
    out["wealth_index"] = df.get("v190")
    out["state"] = df.get("v024")

    # ── Anthropometry ──
    bmi_raw = df.get("v445", pd.Series(np.nan, index=df.index))
    out["bmi"] = bmi_raw / 100.0
    out.loc[(out["bmi"] < 10) | (out["bmi"] > 60), "bmi"] = np.nan

    # ── Blood pressure (average of up to 3 readings) ──
    sbp_cols = [c for c in SBP_VARS if c in df.columns]
    dbp_cols = [c for c in DBP_VARS if c in df.columns]

    if sbp_cols:
        out["sbp"] = df[sbp_cols].mean(axis=1)
    if dbp_cols:
        out["dbp"] = df[dbp_cols].mean(axis=1)

    # ── Hypertension flag ──
    if "sbp" in out.columns and "dbp" in out.columns:
        out["hypertension"] = (
            (out["sbp"] >= 140) | (out["dbp"] >= 90)
        ).astype(int)
        out.loc[out["sbp"].isna() | out["dbp"].isna(), "hypertension"] = np.nan

    # ── Behavioural ──
    out["tobacco"] = df.get("v463a", pd.Series(np.nan, index=df.index))
    out["alcohol"] = df.get("v463c", pd.Series(np.nan, index=df.index))

    return out
