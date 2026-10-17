"""Heart rate at a given lactate: what HR to run by on days without a lactate meter.

Per rep i in session s, week w:

    hr_i = hr_at_2_w + k * log(lactate_i / 2.0) + rep_length + heat + treadmill + session_s + noise

- ``hr_at_2_w``: average heart rate that goes with 2.0 mmol/L in week w
  (weekly random walk -- it drifts as fitness and heat acclimatization change).
- ``k``: how many bpm a given lactate ratio is worth (2 -> 3 mmol/L is
  k * log(1.5)).
- Rep length matters for the *average* HR logged per rep: in short reps HR
  never reaches its plateau, so the logged average is lower at the same lactate.
- Heat raises HR at the same effort (cardiovascular drift).
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from .fitness_model import (
    REFERENCE_LACTATE,
    STANDARD_REP_TIME_S,
    STANDARD_TEMPERATURE_C,
    WORKOUT_STRUCTURES,
    _impute_temperature,
    prepare_model_frame,
)

MIN_REPS_WITH_HR = 20


def _design(model_df: pd.DataFrame) -> dict[str, np.ndarray]:
    temperature, _ = _impute_temperature(model_df)
    return {
        "log_lactate_ratio": np.log(model_df["lactate_mmol"].to_numpy(float) / REFERENCE_LACTATE),
        "rep_duration": np.log2(model_df["rep_time_s"].to_numpy(float) / STANDARD_REP_TIME_S),
        "heat": (temperature - STANDARD_TEMPERATURE_C).clip(lower=0.0).to_numpy(float),
        "treadmill": model_df["surface"].eq("Treadmill").to_numpy(float),
    }


def fit_and_summarize(df: pd.DataFrame, targets: list[float], draws: int = 1000, tune: int = 1000, chains: int = 4, seed: int = 0) -> dict | None:
    """Returns None when too few reps have a heart rate to say anything."""
    model_df = prepare_model_frame(df)
    model_df = model_df[model_df["heart_rate"].notna()].reset_index(drop=True)
    if len(model_df) < MIN_REPS_WITH_HR:
        return None
    # Re-index weeks on the HR subset so the random walk has no leading gap.
    model_df["week_idx"] = ((model_df["week_start"] - model_df["week_start"].min()).dt.days // 7).astype(int)
    n_weeks = int(model_df["week_idx"].max()) + 1
    session_codes, session_ids = pd.factorize(model_df["session_id"])
    x = _design(model_df)
    hr = model_df["heart_rate"].to_numpy(float)

    with pm.Model(coords={"week": np.arange(n_weeks), "session": session_ids}) as model:
        bpm_per_log_lactate = pm.Normal("bpm_per_log_lactate", 15.0, 7.0)
        rep_duration = pm.Normal("rep_duration", 3.0, 2.0)
        heat = pm.Normal("heat", 0.5, 0.3)
        treadmill = pm.Normal("treadmill", 0.0, 3.0)
        start = pm.Normal("start", float(np.median(hr)), 10.0)
        weekly_change_sd = pm.HalfNormal("weekly_change_sd", 1.0)
        steps = pm.Normal("steps", 0.0, 1.0, dims="week")
        hr_at_2 = pm.Deterministic(
            "hr_at_2", start + pm.math.cumsum(steps * weekly_change_sd) - steps[0] * weekly_change_sd, dims="week"
        )
        session_sd = pm.HalfNormal("session_sd", 4.0)
        session_raw = pm.Normal("session_raw", 0.0, 1.0, dims="session")
        mu = (
            hr_at_2[model_df["week_idx"].to_numpy()]
            + bpm_per_log_lactate * x["log_lactate_ratio"]
            + rep_duration * x["rep_duration"]
            + heat * x["heat"]
            + treadmill * x["treadmill"]
            + session_raw[session_codes] * session_sd
        )
        noise_sd = pm.HalfNormal("noise_sd", 4.0)
        pm.StudentT("hr", nu=4, mu=mu, sigma=noise_sd, observed=hr)
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, target_accept=0.95, random_seed=seed, progressbar=False, cores=1
        )

    def draws_of(name: str) -> np.ndarray:
        values = idata.posterior[name].to_numpy()
        return values.reshape(-1, *values.shape[2:])

    def interval(values: np.ndarray) -> dict:
        return {
            "median": round(float(np.median(values))),
            "lo95": round(float(np.percentile(values, 2.5))),
            "hi95": round(float(np.percentile(values, 97.5))),
        }

    hr_at_2_draws = draws_of("hr_at_2")
    k = draws_of("bpm_per_log_lactate")
    rep_effect = draws_of("rep_duration")
    last_week = n_weeks - 1

    structures = []
    for label, _, rep_time_s, _ in WORKOUT_STRUCTURES:
        shift = rep_effect * np.log2(rep_time_s / STANDARD_REP_TIME_S)
        structures.append(
            {
                "label": label,
                "hr": {
                    str(t): interval(hr_at_2_draws[:, last_week] + k * np.log(t / REFERENCE_LACTATE) + shift)
                    for t in targets
                },
            }
        )

    week_starts = pd.date_range(model_df["week_start"].min(), periods=n_weeks, freq="7D")
    weeks_with_data = set(model_df["week_idx"])
    trend = [
        {
            "week_start": str(week.date()),
            "has_data": i in weeks_with_data,
            "median": round(float(np.median(hr_at_2_draws[:, i])), 1),
            "lo95": round(float(np.percentile(hr_at_2_draws[:, i], 2.5)), 1),
            "hi95": round(float(np.percentile(hr_at_2_draws[:, i], 97.5)), 1),
        }
        for i, week in enumerate(week_starts)
    ]

    max_rhat = float(az.rhat(idata, var_names=["hr_at_2", "bpm_per_log_lactate", "rep_duration"]).to_dataset().to_array().max())
    return {
        "n_reps": int(len(model_df)),
        "structures": structures,
        "trend_at_2": trend,
        "bpm_2_to_3": interval(k * np.log(1.5)),
        "heat_bpm_per_c": round(float(np.median(draws_of("heat"))), 2),
        "converged": max_rhat <= 1.05 and int(idata.sample_stats["diverging"].sum()) == 0,
    }
