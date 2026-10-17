"""Group individual measurement rows into training sessions.

A session is defined as (date, AM/PM). Reps within a session are ordered by
the source rep_number if present, otherwise by original row order.
"""
from __future__ import annotations

import pandas as pd


def assign_sessions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["date", "session_period", "source_row"], kind="stable").reset_index(drop=True)

    df["session_id"] = df["date"].dt.strftime("%Y%m%d") + "_" + df["session_period"]

    original_rep_number = pd.to_numeric(df.get("rep_number"), errors="coerce")
    df["rep_number_inferred"] = original_rep_number.isna()
    fallback = df.groupby("session_id").cumcount() + 1
    df["rep_number"] = original_rep_number.fillna(fallback).astype(int)

    df["session_rep_count"] = df.groupby("session_id")["session_id"].transform("count")

    df = df.sort_values(["session_id", "rep_number"], kind="stable").reset_index(drop=True)
    return df
