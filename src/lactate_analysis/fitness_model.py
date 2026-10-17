"""Latent fitness trend + condition normalization, all in seconds per km.

The model, per rep i in session s, week w:

    log(lactate_i) = log(L_ref) + b * (effective_pace_i - fitness_w) + form_s + noise

- ``fitness_w`` is the pace (s/km) that produces exactly L_ref (2.0 mmol/L)
  under standard conditions in week w. It follows a weekly random walk, so
  every session informs its neighbours and gaps simply widen the
  uncertainty -- no manual cycle boundaries are needed to see a trend.
- ``b`` (< 0) is the local slope of the exponential lactate curve: how much
  log-lactate rises per s/km faster. It is identified mostly from pace
  changes *within* a session and between same-day AM/PM sessions, which is
  why it can be estimated well even though the overall pace range is narrow.
  It is what turns "slightly slower but much lower lactate" into a fair
  comparison instead of the misleading lactate/speed ratio.
- ``effective_pace_i = pace_i - sum_k cost_k * x_ik``: every condition
  (heat, treadmill, rep length, rest, ...) is expressed as a pace cost in
  s/km (positive = makes the rep harder, i.e. like running that much
  faster). Each cost has a literature-based prior, so a sparsely observed
  condition still contributes a sensible, appropriately uncertain number
  instead of being dropped or overfit.
- ``form_s`` is the day-to-day deviation of one session (sleep, fatigue,
  nutrition, measurement noise) that the trend shouldn't absorb.

Heart rate is deliberately *not* a covariate: it is a consequence of
intensity, not a condition, and controlling for it would remove part of the
fitness signal itself.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from . import config

REFERENCE_LACTATE = 2.0
STANDARD_TEMPERATURE_C = 15.0
STANDARD_REST_S = 60.0
STANDARD_REP_TIME_S = 300.0
TREADMILL_DEFAULT_TEMPERATURE_C = 18.0
OFFROAD_SURFACES = {"Grass / Turf", "Dirtroad", "XC", "Trail", "Grass"}

# A factor counts as "learned from your data" once the data has narrowed its
# uncertainty to at most this fraction of the prior's; above it, the reported
# number is mostly the literature prior.
LEARNED_FROM_DATA_SD_RATIO = 0.7


@dataclass(frozen=True)
class Factor:
    key: str
    label: str
    unit: str
    prior_mean: float  # s/km per unit, positive = harder
    prior_sd: float
    rationale: str


FACTORS: list[Factor] = [
    Factor(
        "treadmill", "Treadmill (vs. track)", "on/off", -5.0, 4.0,
        "No air resistance and a moving belt: at 0% incline the treadmill is easier than "
        "outdoors (Jones & Doust 1996 recommend 1% incline to match).",
    ),
    Factor(
        "road", "Road (vs. track)", "on/off", 0.0, 3.0,
        "Road and track are both firm surfaces; no systematic difference expected.",
    ),
    Factor(
        "offroad", "Grass / dirt / XC (vs. track)", "on/off", 5.0, 5.0,
        "Soft or uneven ground costs extra energy at the same pace.",
    ),
    Factor(
        "incline", "Treadmill incline", "per 1%", 6.0, 2.0,
        "Each 1% grade adds roughly 3-4% metabolic cost (Minetti et al. 2002).",
    ),
    Factor(
        "heat", "Heat", f"per °C above {STANDARD_TEMPERATURE_C:.0f}°C", 0.6, 0.4,
        "Heat shifts blood to the skin and raises lactate at a given pace; endurance "
        "performance drops ~0.3-0.4% per °C above ~15°C (Ely et al. 2007).",
    ),
    Factor(
        "rep_index", "Later rep in the session", "per rep", 1.0, 2.0,
        "Lactate tends to accumulate over a session even at constant pace.",
    ),
    Factor(
        "rep_duration", "Longer reps", "per doubling of rep duration", 4.0, 4.0,
        "Longer continuous efforts produce more lactate at the same pace -- why short reps are "
        "run faster than long reps at the same lactate target.",
    ),
    Factor(
        "rest", "Longer rest between reps", f"per extra minute (vs. {STANDARD_REST_S:.0f}s)", -2.0, 3.0,
        "More recovery lets lactate clear before the next rep.",
    ),
    Factor(
        "jog_rest", "Jog/float rest (vs. standing)", "on/off", 0.0, 3.0,
        "Active recovery clears lactate faster but keeps working; net effect unclear.",
    ),
    Factor(
        "second_session", "Second session of the day", "on/off", 0.0, 5.0,
        "Double-threshold PM sessions start pre-fatigued and partly glycogen-depleted; the "
        "direction of the effect on lactate is debated.",
    ),
]


def _impute_temperature(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Fill missing temperatures: indoor (treadmill) -> typical room temperature,
    outdoor -> the mean of recorded temperatures in the same calendar month
    (seasonal), else the overall mean. Returns (temperature, was_imputed)."""
    temp = df["temperature_c"].astype(float)
    imputed = temp.isna()
    month = df["date"].dt.month
    month_mean = temp.groupby(month).transform("mean")
    overall = temp.mean() if temp.notna().any() else STANDARD_TEMPERATURE_C
    is_treadmill = df["surface"].eq("Treadmill")
    filled = temp.copy()
    filled[imputed & is_treadmill] = TREADMILL_DEFAULT_TEMPERATURE_C
    rest = imputed & ~is_treadmill
    filled[rest] = month_mean[rest].fillna(overall)
    return filled, imputed


def build_design(df: pd.DataFrame) -> pd.DataFrame:
    """Condition matrix (one column per Factor, standard conditions = 0)."""
    surface = df["surface"].fillna("Track")
    temperature, _ = _impute_temperature(df)
    rest_s = df["rest_duration_s"].astype(float).fillna(STANDARD_REST_S)
    rest_type = df["rest_type"].fillna("Stand")
    rep_time = df["rep_time_s"].astype(float)

    design = pd.DataFrame(index=df.index)
    design["treadmill"] = surface.eq("Treadmill").astype(float)
    design["road"] = surface.eq("Road").astype(float)
    design["offroad"] = surface.isin(OFFROAD_SURFACES).astype(float)
    design["incline"] = df["incline_pct"].astype(float).fillna(0.0) * design["treadmill"]
    design["heat"] = (temperature - STANDARD_TEMPERATURE_C).clip(lower=0.0)
    design["rep_index"] = (df["rep_number"].astype(float) - 1.0).clip(lower=0.0)
    design["rep_duration"] = np.log2(rep_time / STANDARD_REP_TIME_S)
    design["rest"] = (rest_s - STANDARD_REST_S) / 60.0
    design["jog_rest"] = rest_type.isin(["Jog", "Float"]).astype(float)
    design["second_session"] = df["had_earlier_session_today"].astype(float)
    return design[[f.key for f in FACTORS]]


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Submaximal reps with a usable lactate and pace, plus a week index."""
    keep = df["lactate_mmol"].gt(0) & df["pace_s_per_km"].notna()
    if "workout_type" in df.columns:
        keep &= ~df["workout_type"].isin(config.MAX_EFFORT_WORKOUT_TYPES)
    model_df = df[keep].copy().reset_index(drop=True)
    if model_df.empty:
        raise ValueError("No reps with both lactate and pace available to model.")
    week_start = model_df["date"].dt.to_period("W-SUN").dt.start_time
    model_df["week_start"] = week_start
    model_df["week_idx"] = ((week_start - week_start.min()).dt.days // 7).astype(int)
    return model_df


def build_model(model_df: pd.DataFrame, design: pd.DataFrame) -> pm.Model:
    n_weeks = int(model_df["week_idx"].max()) + 1
    session_codes, session_ids = pd.factorize(model_df["session_id"])
    prior_means = np.array([f.prior_mean for f in FACTORS])
    prior_sds = np.array([f.prior_sd for f in FACTORS])

    pace = model_df["pace_s_per_km"].to_numpy(float)
    log_lactate = np.log(model_df["lactate_mmol"].to_numpy(float))
    # Rough data-driven starting level: observed pace shifted to 2.0 mmol/L
    # with a typical slope -- only centers the prior, the data decides.
    start_guess = float(np.median(pace + (log_lactate - np.log(REFERENCE_LACTATE)) / 0.025))

    coords = {"week": np.arange(n_weeks), "session": session_ids, "factor": [f.key for f in FACTORS]}
    with pm.Model(coords=coords) as model:
        cost = pm.Normal("cost", mu=prior_means, sigma=prior_sds, dims="factor")
        # Slope of log(lactate) per s/km, constrained negative (faster ->
        # more lactate). Centered on ~2.5%/s/km, a typical value around the
        # lactate threshold, but allowed to range ~2x either way.
        log_neg_slope = pm.Normal("log_neg_slope", mu=np.log(0.025), sigma=0.4)
        slope = pm.Deterministic("slope", -pm.math.exp(log_neg_slope))

        fitness_start = pm.Normal("fitness_start", mu=start_guess, sigma=15.0)
        weekly_change_sd = pm.HalfNormal("weekly_change_sd", sigma=2.0)
        weekly_steps = pm.Normal("weekly_steps", 0.0, 1.0, dims="week")
        fitness = pm.Deterministic(
            "fitness",
            fitness_start + pm.math.cumsum(weekly_steps * weekly_change_sd) - weekly_steps[0] * weekly_change_sd,
            dims="week",
        )

        form_sd = pm.HalfNormal("form_sd", sigma=0.2)
        form_raw = pm.Normal("form_raw", 0.0, 1.0, dims="session")
        form = pm.Deterministic("form", form_raw * form_sd, dims="session")

        effective_pace = pace - pm.math.dot(design.to_numpy(float), cost)
        mu = (
            np.log(REFERENCE_LACTATE)
            + slope * (effective_pace - fitness[model_df["week_idx"].to_numpy()])
            + form[session_codes]
        )
        noise_sd = pm.HalfNormal("noise_sd", sigma=0.2)
        pm.StudentT("log_lactate", nu=4, mu=mu, sigma=noise_sd, observed=log_lactate)
    return model


@dataclass
class FitnessFit:
    model_df: pd.DataFrame
    design: pd.DataFrame
    idata: object
    n_divergences: int
    max_rhat: float


def fit(df: pd.DataFrame, draws: int = 1000, tune: int = 1000, chains: int = 4, seed: int = 0) -> FitnessFit:
    model_df = prepare_model_frame(df)
    design = build_design(model_df)
    model = build_model(model_df, design)
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.95,
            random_seed=seed,
            progressbar=False,
            # Sequential chains: the model samples in seconds, and parallel
            # chain processes (and the NUMBA backend) hung on this Windows setup.
            cores=1,
        )
    max_rhat = float(
        az.rhat(idata, var_names=["cost", "slope", "fitness", "form_sd", "noise_sd"]).to_dataset().to_array().max()
    )
    n_divergences = int(idata.sample_stats["diverging"].to_numpy().sum())
    return FitnessFit(model_df, design, idata, n_divergences, max_rhat)


def _draws(fit_result: FitnessFit, name: str) -> np.ndarray:
    """Posterior draws flattened over chains: shape (n_draws, *dims)."""
    values = fit_result.idata.posterior[name].to_numpy()
    return values.reshape(-1, *values.shape[2:])


def _interval(values: np.ndarray, axis: int = 0) -> dict:
    return {
        "median": np.median(values, axis=axis),
        "lo80": np.percentile(values, 10, axis=axis),
        "hi80": np.percentile(values, 90, axis=axis),
        "lo95": np.percentile(values, 2.5, axis=axis),
        "hi95": np.percentile(values, 97.5, axis=axis),
    }


def pace_offset_for_lactate(slope_draws: np.ndarray, target_lactate: float) -> np.ndarray:
    """s/km to add to the 2.0-mmol pace to get the pace at target_lactate
    (negative = faster), per posterior draw."""
    return np.log(target_lactate / REFERENCE_LACTATE) / slope_draws


def summarize(fit_result: FitnessFit, targets: list[float]) -> dict:
    model_df, design = fit_result.model_df, fit_result.design
    slope = _draws(fit_result, "slope")
    fitness = _draws(fit_result, "fitness")
    cost = _draws(fit_result, "cost")
    form = _draws(fit_result, "form")

    # Trend only over weeks between first and last data, per target lactate.
    week_dates = model_df.groupby("week_idx")["week_start"].first()
    all_weeks = pd.date_range(week_dates.min(), week_dates.max(), freq="7D")
    weeks_with_data = set(model_df["week_idx"])
    trend: dict[str, list[dict]] = {}
    for target in targets:
        series = fitness + pace_offset_for_lactate(slope, target)[:, None]
        band = _interval(series)
        trend[str(target)] = [
            {
                "week_start": str(week.date()),
                "has_data": i in weeks_with_data,
                **{k: round(float(v[i]), 1) for k, v in band.items()},
            }
            for i, week in enumerate(all_weeks)
        ]

    # Per-rep normalized pace: what this rep is worth at target lactate under
    # standard conditions (posterior median over draws).
    pace = model_df["pace_s_per_km"].to_numpy(float)
    log_lac = np.log(model_df["lactate_mmol"].to_numpy(float))
    effective = pace[None, :] - cost @ design.to_numpy(float).T
    reps = []
    for target in targets:
        normalized = effective - (log_lac[None, :] - np.log(target)) / slope[:, None]
        model_df[f"normalized_pace_{target}"] = np.median(normalized, axis=0)
    session_codes, session_ids = pd.factorize(model_df["session_id"])
    # form > 0 means more lactate than the trend predicts (a worse day);
    # dividing by the negative slope turns that into s/km with positive =
    # better day than the trend.
    form_s_per_km = form / slope[:, None]
    sessions = []
    for code, session_id in enumerate(session_ids):
        rows = model_df[session_codes == code]
        first = rows.iloc[0]
        sessions.append(
            {
                "session_id": session_id,
                "date": str(first["date"].date()),
                "period": first.get("session_period"),
                "workout_type": first.get("workout_type") if pd.notna(first.get("workout_type")) else None,
                "surface": first.get("surface") if pd.notna(first.get("surface")) else None,
                "n_reps": int(len(rows)),
                "mean_pace_s_per_km": round(float(rows["pace_s_per_km"].mean()), 1),
                "mean_lactate": round(float(rows["lactate_mmol"].mean()), 2),
                "normalized_pace": {
                    str(t): round(float(rows[f"normalized_pace_{t}"].mean()), 1) for t in targets
                },
                "form_s_per_km": round(float(np.median(form_s_per_km[:, code])), 1),
            }
        )
    for _, row in model_df.iterrows():
        reps.append(
            {
                "date": str(row["date"].date()),
                "session_id": row["session_id"],
                "rep_number": int(row["rep_number"]),
                "pace_s_per_km": round(float(row["pace_s_per_km"]), 1),
                "lactate_mmol": float(row["lactate_mmol"]),
                "normalized_pace": {str(t): round(float(row[f"normalized_pace_{t}"]), 1) for t in targets},
            }
        )

    factors = []
    for j, factor in enumerate(FACTORS):
        values = cost[:, j]
        observed = design[factor.key].abs().gt(1e-9)
        posterior_sd = float(values.std())
        factors.append(
            {
                "key": factor.key,
                "label": factor.label,
                "unit": factor.unit,
                "rationale": factor.rationale,
                "prior_mean": factor.prior_mean,
                "prior_sd": factor.prior_sd,
                "median": round(float(np.median(values)), 1),
                "ci95": [round(float(np.percentile(values, 2.5)), 1), round(float(np.percentile(values, 97.5)), 1)],
                "prob_harder": round(float(np.mean(values > 0)), 2),
                "n_reps_affected": int(observed.sum()),
                "learned_from_data": posterior_sd <= LEARNED_FROM_DATA_SD_RATIO * factor.prior_sd,
            }
        )

    lactate_ratio_1_5 = np.log(1.5) / -slope
    return {
        "reference_lactate": REFERENCE_LACTATE,
        "targets": targets,
        "trend": trend,
        "sessions": sessions,
        "reps": reps,
        "factors": factors,
        "curve": {
            "slope_pct_per_s_per_km": {
                k: round(float(v), 2) for k, v in _interval(-slope * 100).items()
            },
            "s_per_km_for_1_5x_lactate": {k: round(float(v), 1) for k, v in _interval(lactate_ratio_1_5).items()},
        },
        "day_to_day_form_sd_s_per_km": round(float(np.median(_draws(fit_result, "form_sd") / -slope)), 1),
        "diagnostics": {
            "n_reps": int(len(model_df)),
            "n_sessions": int(len(session_ids)),
            "n_temperature_imputed": int(_impute_temperature(model_df)[1].sum()),
            "n_divergences": fit_result.n_divergences,
            "max_rhat": round(fit_result.max_rhat, 3),
            "converged": fit_result.max_rhat <= 1.05 and fit_result.n_divergences == 0,
        },
        "_fitness_draws": fitness,
        "_slope_draws": slope,
        "_week_starts": [str(d.date()) for d in all_weeks],
    }
