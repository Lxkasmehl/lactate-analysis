"""Data-completeness checks that feed the report's data-quality section."""
from __future__ import annotations

import pandas as pd

TRACKED_COLUMNS = [
    "temperature_c",
    "heart_rate",
    "humidity_pct",
    "incline_pct",
    "rest_duration_s",
    "rep_number",
    "surface",
    "rest_type",
    "shoes",
    "workout_type",
]


def compute_fill_rates(df: pd.DataFrame) -> dict[str, float]:
    return {
        col: round(float(df[col].notna().mean()), 3) if col in df.columns else 0.0 for col in TRACKED_COLUMNS
    }


def build_validation_report(df: pd.DataFrame) -> dict:
    return {
        "n_rows_total": int(len(df)),
        "n_sessions": int(df["session_id"].nunique()),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "covariate_fill_rates": compute_fill_rates(df),
        "inferred_rep_numbers": int(df.get("rep_number_inferred", pd.Series(dtype=bool)).sum()),
    }
