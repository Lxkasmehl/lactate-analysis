"""Assemble the single structured JSON summary that drives the HTML report.

This dict is the hand-off contract for later phases too (an LLM
interpretation layer, a future web backend) -- nothing downstream of this
function should need to touch the raw DataFrame or posterior draws again.
All paces are in seconds per km; the report converts for display.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .config import MAX_EFFORT_WORKOUT_TYPES

# Comparisons for the status tile: "now" vs. this many weeks earlier.
COMPARISON_WEEKS = {"8 weeks ago": 8, "a year ago": 52}

# A change counts as real when the posterior is at least this sure of its
# direction; otherwise it's reported as "no clear change".
CLEAR_CHANGE_PROBABILITY = 0.8


def _interval(values: np.ndarray) -> dict:
    return {
        "median": round(float(np.median(values)), 1),
        "lo95": round(float(np.percentile(values, 2.5)), 1),
        "hi95": round(float(np.percentile(values, 97.5)), 1),
    }


def _change(diff_draws: np.ndarray) -> dict:
    """diff = later pace - earlier pace (s/km); negative means faster."""
    # Rounded before the verdict so the shown percentage and the verdict agree.
    prob_faster = round(float(np.mean(diff_draws < 0)), 2)
    if prob_faster >= CLEAR_CHANGE_PROBABILITY:
        verdict = "faster"
    elif prob_faster <= 1 - CLEAR_CHANGE_PROBABILITY:
        verdict = "slower"
    else:
        verdict = "no clear change"
    return {**_interval(diff_draws), "prob_faster": prob_faster, "verdict": verdict}


def _status(fitness_summary: dict) -> dict:
    fitness = fitness_summary["_fitness_draws"]
    week_starts = fitness_summary["_week_starts"]
    last_idx = fitness_summary["_last_week_idx"]

    current = {
        target: {k: week[k] for k in ("median", "lo95", "hi95")}
        for target, weeks in fitness_summary["trend"].items()
        for week in [weeks[last_idx]]
    }

    comparisons = []
    for label, weeks in COMPARISON_WEEKS.items():
        earlier_idx = last_idx - weeks
        if earlier_idx < 0:
            continue
        comparisons.append(
            {
                "label": label,
                "week_start": week_starts[earlier_idx],
                # The curve's shape is shared over time, so the change is the
                # same at every target lactate.
                **_change(fitness[:, last_idx] - fitness[:, earlier_idx]),
            }
        )
    return {"as_of_week": week_starts[last_idx], "current": current, "comparisons": comparisons}


def _cycle_summaries(fitness_summary: dict, cycle_defs: list[dict]) -> list[dict]:
    """Average standardized fitness over the weeks of each cycle that have data."""
    fitness = fitness_summary["_fitness_draws"]
    week_starts = pd.to_datetime(fitness_summary["_week_starts"])
    has_data = np.array([w["has_data"] for w in fitness_summary["trend"][str(fitness_summary["reference_lactate"])]])
    sessions = pd.DataFrame(fitness_summary["sessions"])
    sessions["date"] = pd.to_datetime(sessions["date"])

    summaries: list[dict] = []
    previous_draws = None
    for cycle in cycle_defs:
        in_cycle = (week_starts >= cycle["start_date"] - pd.Timedelta(days=6)) & (week_starts <= cycle["end_date"])
        weeks = np.where(in_cycle & has_data)[0]
        n_sessions = int(sessions["date"].between(cycle["start_date"], cycle["end_date"]).sum())
        entry = {
            "label": cycle["label"],
            "start_date": str(cycle["start_date"].date()),
            "end_date": None if cycle["end_date"] == pd.Timestamp.max else str(cycle["end_date"].date()),
            "n_sessions": n_sessions,
            "pace_at_reference": None,
            "change_vs_previous": None,
        }
        if len(weeks):
            draws = fitness[:, weeks].mean(axis=1)
            entry["pace_at_reference"] = _interval(draws)
            if previous_draws is not None:
                entry["change_vs_previous"] = _change(draws - previous_draws)
            previous_draws = draws
        summaries.append(entry)
    return summaries


def _headline(status: dict) -> str:
    recent = next((c for c in status["comparisons"] if c["label"] == "8 weeks ago"), None)
    if recent is None:
        return "Not enough history yet to judge a recent trend."
    if recent["verdict"] == "faster":
        return (
            f"You're getting fitter: at the same lactate you now run about {abs(recent['median']):.0f} s/km "
            f"faster than 8 weeks ago ({recent['prob_faster']:.0%} probability of a real improvement)."
        )
    if recent["verdict"] == "slower":
        return (
            f"Your standardized pace is about {abs(recent['median']):.0f} s/km slower than 8 weeks ago "
            f"({1 - recent['prob_faster']:.0%} probability this is a real decline, not noise)."
        )
    return "No clear change in your standardized fitness over the last 8 weeks."


def build_summary(
    df: pd.DataFrame,
    validation_report: dict,
    fitness_summary: dict,
    cycle_defs: list[dict],
    heart_rate_summary: dict | None = None,
) -> dict:
    status = _status(fitness_summary)
    max_effort_rows = int(df["workout_type"].isin(MAX_EFFORT_WORKOUT_TYPES).sum()) if "workout_type" in df else 0

    public_fitness = {k: v for k, v in fitness_summary.items() if not k.startswith("_")}
    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "data_summary": {
            "n_rows_total": validation_report["n_rows_total"],
            "n_rows_modeled": public_fitness["diagnostics"]["n_reps"],
            "n_sessions": public_fitness["diagnostics"]["n_sessions"],
            "date_range": validation_report["date_range"],
        },
        "headline": _headline(status),
        "status": status,
        "cycles": _cycle_summaries(fitness_summary, cycle_defs),
        "fitness": public_fitness,
        "heart_rate": heart_rate_summary,
        "data_quality": {
            "fill_rates": validation_report["covariate_fill_rates"],
            "outdoor_temperature_missing_sessions": public_fitness["diagnostics"]["outdoor_temperature_missing_sessions"],
            "assumed_indoor_track_sessions": public_fitness["diagnostics"]["assumed_indoor_track_sessions"],
            "max_effort_rows_excluded": max_effort_rows,
            "max_effort_workout_types": MAX_EFFORT_WORKOUT_TYPES,
        },
    }
