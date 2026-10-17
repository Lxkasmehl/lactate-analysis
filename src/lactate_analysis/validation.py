"""Data-quality checks that feed directly into the report's caveats section.

Nothing here is hand-written prose -- every caveat in the final report is
generated mechanically from this module's output, so it can't go stale.
"""
from __future__ import annotations

import pandas as pd

REQUIRED_MODEL_COLUMNS = ["lactate_mmol", "pace_s_per_km"]
OPTIONAL_NUMERIC_COVARIATES = [
    "temperature_c",
    "heart_rate",
    "humidity_pct",
    "incline_pct",
    "rest_duration_s",
    "rep_number",
    "prior_lactate",
]
OPTIONAL_CATEGORICAL_COVARIATES = [
    "surface",
    "rest_type",
    "indoor_outdoor",
    "shoes",
    "workout_type",
    "had_earlier_session_today",
]


def compute_fill_rates(df: pd.DataFrame) -> dict[str, float]:
    fill_rates: dict[str, float] = {}
    for col in OPTIONAL_NUMERIC_COVARIATES + OPTIONAL_CATEGORICAL_COVARIATES:
        if col in df.columns:
            fill_rates[col] = round(float(df[col].notna().mean()), 3)
        else:
            fill_rates[col] = 0.0
    return fill_rates


def build_validation_report(df: pd.DataFrame, min_fill_rate: float) -> dict:
    fill_rates = compute_fill_rates(df)
    usable_rows = df.dropna(subset=REQUIRED_MODEL_COLUMNS)

    numeric_included = [c for c in OPTIONAL_NUMERIC_COVARIATES if fill_rates.get(c, 0) >= min_fill_rate]
    numeric_excluded = [c for c in OPTIONAL_NUMERIC_COVARIATES if fill_rates.get(c, 0) < min_fill_rate]

    # Same fill-rate bar as numeric covariates: including a mostly-missing
    # categorical in the model's required-columns dropna would silently gut
    # the sample (e.g. "shoes" only ever recorded in the most recent cycle
    # would drop every row from every other cycle).
    categorical_included = [c for c in OPTIONAL_CATEGORICAL_COVARIATES if fill_rates.get(c, 0) >= min_fill_rate]
    categorical_excluded = [c for c in OPTIONAL_CATEGORICAL_COVARIATES if fill_rates.get(c, 0) < min_fill_rate]

    unlabeled_count = int((df.get("cycle_id", pd.Series(dtype=str)) == "unlabeled").sum())
    inferred_rep_numbers = int(df.get("rep_number_inferred", pd.Series(dtype=bool)).sum())

    return {
        "n_rows_total": int(len(df)),
        "n_rows_excluded_missing_required": int(len(df) - len(usable_rows)),
        "n_rows_modeled": int(len(usable_rows)),
        "n_sessions": int(df["session_id"].nunique()),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "covariate_fill_rates": fill_rates,
        "covariates_included_numeric": numeric_included,
        "covariates_excluded_low_fill": numeric_excluded,
        "covariates_included_categorical": categorical_included,
        "covariates_excluded_categorical_low_fill": categorical_excluded,
        "unlabeled_rows": unlabeled_count,
        "inferred_rep_numbers": inferred_rep_numbers,
    }
