"""Per-session pace-at-fixed-lactate estimates, for the raw "trend over time"
scatter view.

Each point here is a log-linear interpolation between the two reps that
bracket a target lactate value *within that one session's own reps* --
useful as a raw, individual data point. It is deliberately NOT used to build
per-cycle or per-workout-type averages: averaging across whichever sessions
happen to bracket each target mixes different, non-overlapping subsets of
sessions per target, which has no guarantee of staying internally consistent
(a target of 4.0 mmol/L might only ever be bracketed by your hardest, fastest
sessions, while 2.0 mmol/L is bracketed by easy days too -- averaging those
separately can produce a "faster at lower lactate" result that makes no
physiological sense). The per-cycle / per-workout-type numbers in the report
come from model.compute_pace_at_lactate() instead, which inverts a single
fitted lactate-vs-pace line per cycle and is mathematically guaranteed to
stay monotonic in the target lactate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UNKNOWN_WORKOUT_TYPE = "Unknown"


def _pace_at_lactate_for_session(pace: np.ndarray, lactate: np.ndarray, target: float) -> float | None:
    """Log-linear interpolation of pace as a function of lactate, within one session.

    Returns None if the session doesn't have at least 2 valid reps or the
    target falls outside that session's own observed lactate range --
    extrapolating a 2-6 point single-session curve is not reliable.
    """
    valid = ~(np.isnan(pace) | np.isnan(lactate)) & (lactate > 0)
    pace, lactate = pace[valid], lactate[valid]
    if len(pace) < 2:
        return None

    order = np.argsort(lactate)
    pace, lactate = pace[order], lactate[order]
    if target < lactate[0] or target > lactate[-1]:
        return None
    if target == lactate[0]:
        return float(pace[0])
    if target == lactate[-1]:
        return float(pace[-1])

    idx = int(np.searchsorted(lactate, target))
    l0, l1 = lactate[idx - 1], lactate[idx]
    p0, p1 = pace[idx - 1], pace[idx]
    if l0 == l1:
        return float((p0 + p1) / 2)

    frac = (np.log(target) - np.log(l0)) / (np.log(l1) - np.log(l0))
    return float(p0 + frac * (p1 - p0))


def compute_session_estimates(df: pd.DataFrame, targets: list[float]) -> pd.DataFrame:
    """One row per session: date, cycle_label, workout_type, pace-at-target per target."""
    rows = []
    for session_id, group in df.groupby("session_id"):
        group = group.sort_values("rep_number")
        pace = group["pace_s_per_km"].to_numpy(dtype=float)
        lactate = group["lactate_mmol"].to_numpy(dtype=float)
        workout_type = group["workout_type"].iloc[0] if "workout_type" in group.columns else None
        row = {
            "session_id": session_id,
            "date": group["date"].iloc[0],
            "cycle_label": group["cycle_label"].iloc[0],
            "workout_type": workout_type if pd.notna(workout_type) else UNKNOWN_WORKOUT_TYPE,
        }
        for target in targets:
            row[f"pace_at_{target}"] = _pace_at_lactate_for_session(pace, lactate, target)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def compute_trend_points(df: pd.DataFrame, targets: list[float]) -> list[dict]:
    session_estimates = compute_session_estimates(df, targets)

    # Not itertuples(): column names like "pace_at_2.0" aren't valid Python
    # identifiers and itertuples silently renames them to positional fields.
    trend_points = []
    for record in session_estimates.to_dict(orient="records"):
        point = {
            "date": str(record["date"].date()),
            "cycle_label": record["cycle_label"],
            "workout_type": record["workout_type"],
        }
        for target in targets:
            value = record[f"pace_at_{target}"]
            point[f"pace_at_{target}"] = None if pd.isna(value) else round(float(value), 2)
        trend_points.append(point)

    return trend_points
