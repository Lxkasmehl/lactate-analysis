"""Derive per-repetition features used by the statistical model and report."""
from __future__ import annotations

import numpy as np
import pandas as pd

MILE_TO_KM = 1.60934


def _pace_deviation_pct(df: pd.DataFrame) -> pd.Series:
    """Heuristic 'was this rep (or the ones before it) too fast' proxy.

    Target = median pace of prior reps in the same session (if at least 2
    prior reps exist), else the session's fastest rep. There is no real
    target-pace column in the source data, so this is always a proxy --
    surfaced as such in the report's caveats section, never as ground truth.
    """
    result = pd.Series(index=df.index, dtype=float)
    for _, group in df.groupby("session_id"):
        idx = group.index
        paces = group["pace_s_per_km"].to_numpy()
        session_fastest = np.nanmin(paces) if len(paces) else np.nan
        targets = np.empty(len(paces))
        for i in range(len(paces)):
            prior = paces[:i]
            prior = prior[~np.isnan(prior)]
            targets[i] = np.median(prior) if len(prior) >= 2 else session_fastest
        result.loc[idx] = (paces - targets) / targets
    return result


_SESSION_PERIOD_ORDER = {"AM": 0, "PM": 1}


def _had_earlier_session_today(df: pd.DataFrame) -> pd.Series:
    """True if another session on the same date has an earlier period (AM
    before PM) -- i.e. this is the second training session of that day.

    This is a real, derivable stand-in for "did I already train earlier
    today", not something that has to stay an unrecorded blind spot: the
    date and AM/PM are always logged, so whether a second same-day session
    happened is a fact already implicit in the data, not a guess.
    """
    period_rank = df["session_period"].map(_SESSION_PERIOD_ORDER).fillna(0)
    min_rank_today = period_rank.groupby(df["date"]).transform("min")
    return period_rank > min_rank_today


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["session_id", "rep_number"], kind="stable").copy()

    df["pace_s_per_km"] = df["rep_time_s"] / (df["rep_distance_m"] / 1000.0)
    df["pace_min_per_mi"] = df["pace_s_per_km"] * MILE_TO_KM / 60.0

    grouped = df.groupby("session_id")
    df["cum_distance_m"] = grouped["rep_distance_m"].cumsum()
    df["cum_time_s"] = grouped["rep_time_s"].cumsum()
    df["prior_lactate"] = grouped["lactate_mmol"].shift(1)
    df["lactate_delta"] = df["lactate_mmol"] - df["prior_lactate"]
    df["had_earlier_session_today"] = _had_earlier_session_today(df)

    project_start = df["date"].min()
    df["days_since_project_start"] = (df["date"] - project_start).dt.days

    cycle_start = df.groupby("cycle_id")["date"].transform("min")
    df["days_since_cycle_start"] = (df["date"] - cycle_start).dt.days

    df["session_mean_pace"] = grouped["pace_s_per_km"].transform("mean")
    df["session_mean_lactate"] = grouped["lactate_mmol"].transform("mean")

    df["pace_deviation_pct"] = _pace_deviation_pct(df)

    return df.reset_index(drop=True)
