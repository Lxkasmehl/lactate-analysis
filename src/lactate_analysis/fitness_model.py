"""Latent fitness trend + condition normalization, all in seconds per km.

The model, per rep i in session s, week w:

    lactate_i = L0 + (2.0 - L0) * exp(c * (fitness_w - effective_pace_i) + form_s)   (x noise)

- ``L0`` is the resting/baseline lactate (~0.8-1 mmol/L). The classic
  lactate curve is baseline + exponential, not a pure exponential: without
  the baseline the step from 2 to 3 mmol/L comes out far too large, because
  a pure log curve treats 1.5 -> 2.0 and 2.0 -> 3.0 alike.
- ``fitness_w`` is the pace (s/km) that produces exactly 2.0 mmol/L under
  standard conditions in week w, as a weekly random walk: every session
  informs its neighbours and gaps simply widen the uncertainty.
- ``c`` is the steepness of the exponential part, learned from your data
  with a literature prior (trained runners go from 2 to 4 mmol/L over ~1-1.5
  km/h in a step test). It turns "slightly slower but much lower lactate"
  into a fair comparison instead of the misleading lactate/speed ratio.
- ``effective_pace_i = pace_i - sum_k cost_k * x_ik``: every condition
  (heat, treadmill, rep length, work:rest, ...) is a pace cost in s/km
  (positive = makes the rep harder, like running that much faster). Each
  cost has a literature-based prior, so a sparsely observed condition still
  contributes a sensible, honestly uncertain number instead of being
  dropped or overfit.
- ``form_s`` is the day-to-day deviation of one session (sleep, fatigue,
  nutrition, measurement noise) that the trend shouldn't absorb, and each
  session also gets its own lactate drift over accumulated work: because
  pace is steered by lactate (slow down when it drifts up), a shared drift
  would let that feedback flatten the estimated curve.

Heart rate is not a condition here (it's a consequence of intensity); it
gets its own model in heart_rate_model.py.
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
STANDARD_HUMIDITY_PCT = 50.0
STANDARD_REP_TIME_S = 300.0
STANDARD_REST_FRACTION = 60.0 / (300.0 + 60.0)
STANDARD_ACCUMULATED_WORK_MIN = 15.0
INDOOR_TEMPERATURE_C = 18.0
DEFAULT_REST_S = 60.0
OFFROAD_SURFACES = {"Grass / Turf", "Dirtroad", "XC", "Trail", "Grass"}
# Carbon-plate "super shoes"; matched case-insensitively as substrings of the
# Shoes column.
SUPER_SHOE_KEYWORDS = [
    "vaporfly", "alphafly", "cloudboom echo", "metaspeed", "adios pro", "endorphin pro",
    "endorphin elite", "rocket x", "fast-r", "sc elite", "carbon x", "deviate elite", "hyperion elite",
]

# Prior for the steepness of the exponential part of the curve (per s/km
# faster), putting 2 -> 3 mmol/L at roughly 8-12 s/km as in step tests of
# trained runners. Deliberately fairly tight: lactate-guided training keeps
# pace within each workout structure almost constant, so the data can't
# separate "steep curve + big rep-length effect" from "flat curve + small
# rep-length effect" (leave-one-out fit was identical for both on the real
# data). A wide prior let the flat solution win by accident and produced
# 3.0 mmol/L paces ~10 s/km faster than the raw long-rep data supports.
STEEPNESS_PRIOR_MEDIAN = 0.085
STEEPNESS_PRIOR_LOG_SD = 0.2

# A factor counts as "learned from your data" once the data has narrowed its
# uncertainty to at most this fraction of the prior's; above it, the reported
# number is mostly the literature prior.
LEARNED_FROM_DATA_SD_RATIO = 0.7

# Typical workout structures for the "target paces" table: (label, rep
# distance m, rep duration s at threshold pace, rest s). Durations are
# approximate; the rep-duration effect is logarithmic, so a few seconds off
# barely matters.
WORKOUT_STRUCTURES = [
    ("400 m reps, 30 s rest", 400, 72.0, 30.0),
    ("1000 m reps, 60 s rest", 1000, 185.0, 60.0),
    ("1 mile reps, 60 s rest", 1609, 310.0, 60.0),
    ("10 min reps, 60 s rest", None, 600.0, 60.0),
]


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
        "rep_duration", "Longer reps", "per doubling of rep duration (vs. 5 min)", 6.0, 3.0,
        "In short reps, oxygen stored in muscle and creatine phosphate cover part of each bout, so "
        "less lactate is produced at the same pace. That's why 400s can be run much faster than "
        "miles at the same lactate.",
    ),
    Factor(
        "rest_fraction", "More rest relative to work", "per +10% of time spent resting", -4.0, 3.0,
        "A larger share of rest (e.g. 400 m / 30 s vs. mile / 60 s) gives lactate more time to clear "
        "between reps.",
    ),
    Factor(
        "accumulated_work", "Later in the session", "per 10 min of reps already run (vs. 15 min)", 2.0, 2.0,
        "Lactate drifts up over a session even at constant pace as fatigue builds.",
    ),
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
        "humidity", "Humidity", f"per +10% above {STANDARD_HUMIDITY_PCT:.0f}% (outdoors)", 1.0, 1.0,
        "Humid air makes sweat evaporate less, so the body heats up faster -- mostly relevant on warm days.",
    ),
    Factor(
        "super_shoes", "Carbon super shoes (vs. not recorded)", "on/off", -5.0, 3.0,
        "Carbon-plated shoes improve running economy by ~4% (Hoogkamer et al. 2018), roughly "
        "2-3% faster at the same effort.",
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


def _is_indoor(df: pd.DataFrame) -> pd.Series:
    surface = df["surface"].fillna("")
    return surface.eq("Treadmill") | surface.str.contains("indoor", case=False)


def _impute_temperature(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Fill missing temperatures: indoor -> room temperature (expected, not a
    gap), outdoor -> mean recorded temperature in the same calendar month,
    else overall mean. Returns (temperature, outdoor_value_was_missing)."""
    temp = df["temperature_c"].astype(float)
    indoor = _is_indoor(df)
    month_mean = temp.where(~indoor).groupby(df["date"].dt.month).transform("mean")
    overall = temp.mean() if temp.notna().any() else STANDARD_TEMPERATURE_C
    filled = temp.copy()
    filled[temp.isna() & indoor] = INDOOR_TEMPERATURE_C
    outdoor_missing = temp.isna() & ~indoor
    filled[outdoor_missing] = month_mean[outdoor_missing].fillna(overall)
    return filled, outdoor_missing


def _is_super_shoe(shoes: pd.Series) -> pd.Series:
    lowered = shoes.fillna("").str.lower()
    return lowered.apply(lambda s: any(k in s for k in SUPER_SHOE_KEYWORDS))


def build_design(df: pd.DataFrame) -> pd.DataFrame:
    """Condition matrix (one column per Factor, standard conditions = 0)."""
    surface = df["surface"].fillna("Track")
    indoor = _is_indoor(df)
    temperature, _ = _impute_temperature(df)
    rest_s = df["rest_duration_s"].astype(float).fillna(DEFAULT_REST_S)
    rest_type = df["rest_type"].fillna("Stand")
    rep_time = df["rep_time_s"].astype(float)
    # "Rep #" in the log is the rep's position in the whole workout (e.g.
    # lactate taken after reps 3, 6, 9), so reps run so far ~ rep_number.
    accumulated_min = df["rep_number"].astype(float) * rep_time / 60.0
    humidity = df["humidity_pct"].astype(float).where(~indoor)

    design = pd.DataFrame(index=df.index)
    design["rep_duration"] = np.log2(rep_time / STANDARD_REP_TIME_S)
    design["rest_fraction"] = (rest_s / (rep_time + rest_s) - STANDARD_REST_FRACTION) * 10.0
    design["accumulated_work"] = (accumulated_min - STANDARD_ACCUMULATED_WORK_MIN) / 10.0
    design["treadmill"] = surface.eq("Treadmill").astype(float)
    design["road"] = surface.eq("Road").astype(float)
    design["offroad"] = surface.isin(OFFROAD_SURFACES).astype(float)
    design["incline"] = df["incline_pct"].astype(float).fillna(0.0) * design["treadmill"]
    design["heat"] = (temperature - STANDARD_TEMPERATURE_C).clip(lower=0.0)
    design["humidity"] = ((humidity - STANDARD_HUMIDITY_PCT).clip(lower=0.0) / 10.0).fillna(0.0)
    design["super_shoes"] = _is_super_shoe(df["shoes"]).astype(float)
    design["jog_rest"] = rest_type.isin(["Jog", "Float"]).astype(float)
    design["second_session"] = df["had_earlier_session_today"].astype(float)
    return design[[f.key for f in FACTORS]]


def structure_design(rep_time_s: float, rest_s: float) -> np.ndarray:
    """Condition vector for a workout structure under otherwise standard
    conditions (track, cool, mid-session, first session of the day)."""
    row = {f.key: 0.0 for f in FACTORS}
    row["rep_duration"] = np.log2(rep_time_s / STANDARD_REP_TIME_S)
    row["rest_fraction"] = (rest_s / (rep_time_s + rest_s) - STANDARD_REST_FRACTION) * 10.0
    return np.array([row[f.key] for f in FACTORS])


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Submaximal reps with a usable lactate and pace, plus a week index."""
    keep = df["lactate_mmol"].gt(0) & df["pace_s_per_km"].notna()
    if "workout_type" in df.columns:
        keep &= ~df["workout_type"].isin(config.MAX_EFFORT_WORKOUT_TYPES)
    model_df = df[keep].copy().reset_index(drop=True)
    if model_df.empty:
        raise ValueError("No reps with both lactate and pace available to model.")
    for col in ("humidity_pct", "shoes", "incline_pct", "rest_duration_s", "rest_type", "temperature_c"):
        if col not in model_df.columns:
            model_df[col] = np.nan
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
    lactate = model_df["lactate_mmol"].to_numpy(float)
    start_guess = float(np.median(pace[(lactate > 1.5) & (lactate < 2.5)])) if ((lactate > 1.5) & (lactate < 2.5)).any() else float(np.median(pace))

    coords = {"week": np.arange(n_weeks), "session": session_ids, "factor": [f.key for f in FACTORS]}
    with pm.Model(coords=coords) as model:
        cost = pm.Normal("cost", mu=prior_means, sigma=prior_sds, dims="factor")
        log_steepness = pm.Normal("log_steepness", mu=np.log(STEEPNESS_PRIOR_MEDIAN), sigma=STEEPNESS_PRIOR_LOG_SD)
        steepness = pm.Deterministic("steepness", pm.math.exp(log_steepness))
        baseline = pm.TruncatedNormal(
            "baseline", mu=0.8, sigma=0.25, lower=0.3, upper=min(0.95 * float(lactate.min()), 1.5)
        )

        fitness_start = pm.Normal("fitness_start", mu=start_guess, sigma=15.0)
        weekly_change_sd = pm.HalfNormal("weekly_change_sd", sigma=2.0)
        weekly_steps = pm.Normal("weekly_steps", 0.0, 1.0, dims="week")
        fitness = pm.Deterministic(
            "fitness",
            fitness_start + pm.math.cumsum(weekly_steps * weekly_change_sd) - weekly_steps[0] * weekly_change_sd,
            dims="week",
        )

        form_sd = pm.HalfNormal("form_sd", sigma=0.3)
        form_raw = pm.Normal("form_raw", 0.0, 1.0, dims="session")
        form = pm.Deterministic("form", form_raw * form_sd, dims="session")

        # Session-specific lactate drift over the session. Pace is steered by
        # lactate (slow down when it drifts up), so without this the drift
        # leaks into the pace-lactate relationship and flattens the curve.
        drift_sd = pm.HalfNormal("drift_sd", sigma=0.3)
        drift_raw = pm.Normal("drift_raw", 0.0, 1.0, dims="session")
        drift = drift_raw * drift_sd

        effective_pace = pace - pm.math.dot(design.to_numpy(float), cost)
        exponent = (
            steepness * (fitness[model_df["week_idx"].to_numpy()] - effective_pace)
            + form[session_codes]
            + drift[session_codes] * design["accumulated_work"].to_numpy(float)
        )
        mu = pm.math.log(baseline + (REFERENCE_LACTATE - baseline) * pm.math.exp(exponent))
        noise_sd = pm.HalfNormal("noise_sd", sigma=0.2)
        pm.StudentT("log_lactate", nu=4, mu=mu, sigma=noise_sd, observed=np.log(lactate))
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
            # Sequential chains: the model samples in about a minute, and
            # parallel chain processes (and the NUMBA backend) hung on this
            # Windows setup.
            cores=1,
        )
    max_rhat = float(
        az.rhat(idata, var_names=["cost", "steepness", "baseline", "fitness", "form_sd", "drift_sd", "noise_sd"])
        .to_dataset()
        .to_array()
        .max()
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


def _rounded(values: np.ndarray, digits: int = 1) -> dict:
    return {k: round(float(v), digits) for k, v in _interval(values).items()}


def pace_offset_for_lactate(steepness: np.ndarray, baseline: np.ndarray, target_lactate: float) -> np.ndarray:
    """s/km to add to the 2.0-mmol pace to get the pace at target_lactate
    (negative = faster), per posterior draw."""
    return -np.log((target_lactate - baseline) / (REFERENCE_LACTATE - baseline)) / steepness


def summarize(fit_result: FitnessFit, targets: list[float]) -> dict:
    model_df, design = fit_result.model_df, fit_result.design
    steepness = _draws(fit_result, "steepness")
    baseline = _draws(fit_result, "baseline")
    fitness = _draws(fit_result, "fitness")
    cost = _draws(fit_result, "cost")
    form = _draws(fit_result, "form")
    offsets = {t: pace_offset_for_lactate(steepness, baseline, t) for t in targets}

    week_dates = model_df.groupby("week_idx")["week_start"].first()
    all_weeks = pd.date_range(week_dates.min(), week_dates.max(), freq="7D")
    weeks_with_data = set(model_df["week_idx"])
    last_week = max(weeks_with_data)
    trend: dict[str, list[dict]] = {}
    for target in targets:
        band = _interval(fitness + offsets[target][:, None])
        trend[str(target)] = [
            {
                "week_start": str(week.date()),
                "has_data": i in weeks_with_data,
                **{k: round(float(v[i]), 1) for k, v in band.items()},
            }
            for i, week in enumerate(all_weeks)
        ]

    # Target paces for typical workout structures, current week.
    structures = []
    for label, distance, rep_time_s, rest_s in WORKOUT_STRUCTURES:
        structure_cost = cost @ structure_design(rep_time_s, rest_s)
        structures.append(
            {
                "label": label,
                "rep_distance_m": distance,
                "rep_time_s": rep_time_s,
                "rest_s": rest_s,
                "pace": {
                    str(t): _rounded(fitness[:, last_week] + offsets[t] + structure_cost) for t in targets
                },
            }
        )

    # Per-rep normalized pace: the pace at target lactate under standard
    # conditions that this rep implies. Readings at or below the baseline
    # carry no pace information, so they're clipped just above it.
    pace = model_df["pace_s_per_km"].to_numpy(float)
    lactate = model_df["lactate_mmol"].to_numpy(float)
    effective = pace[None, :] - cost @ design.to_numpy(float).T
    excess = np.maximum(lactate[None, :] - baseline[:, None], 0.05)
    for target in targets:
        normalized = effective + np.log(excess / (target - baseline[:, None])) / steepness[:, None]
        model_df[f"normalized_pace_{target}"] = np.median(normalized, axis=0)

    session_codes, session_ids = pd.factorize(model_df["session_id"])
    # form > 0 means more lactate than the trend predicts (a worse day); as
    # s/km with positive = better day than the trend.
    form_s_per_km = -form / steepness[:, None]
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
                "rep_distance_m": float(first["rep_distance_m"]) if pd.notna(first.get("rep_distance_m")) else None,
                "n_reps": int(len(rows)),
                "mean_pace_s_per_km": round(float(rows["pace_s_per_km"].mean()), 1),
                "mean_lactate": round(float(rows["lactate_mmol"].mean()), 2),
                "normalized_pace": {
                    str(t): round(float(rows[f"normalized_pace_{t}"].mean()), 1) for t in targets
                },
                "form_s_per_km": round(float(np.median(form_s_per_km[:, code])), 1),
            }
        )

    factors = []
    for j, factor in enumerate(FACTORS):
        values = cost[:, j]
        observed = design[factor.key].abs().gt(1e-9)
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
                "n_reps_affected": int(observed.sum()),
                "learned_from_data": float(values.std()) <= LEARNED_FROM_DATA_SD_RATIO * factor.prior_sd,
            }
        )

    _, outdoor_temp_missing = _impute_temperature(model_df)
    missing_temp_rows = model_df.loc[outdoor_temp_missing].drop_duplicates("session_id").sort_values("date")
    missing_temp_sessions = [
        f"{row.date.date()} {row.session_period}" for row in missing_temp_rows.itertuples()
    ]

    return {
        "reference_lactate": REFERENCE_LACTATE,
        "targets": targets,
        "trend": trend,
        "structures": structures,
        "sessions": sessions,
        "factors": factors,
        "curve": {
            "baseline_mmol": _rounded(baseline, 2),
            "steepness_learned_from_data": float(np.log(steepness).std())
            <= LEARNED_FROM_DATA_SD_RATIO * STEEPNESS_PRIOR_LOG_SD,
            "s_per_km_2_to_3": _rounded(-offsets[3.0]) if 3.0 in offsets else None,
            "lactate_drop_5s_slower_pct": _rounded(
                100 * (1 - (baseline + (REFERENCE_LACTATE - baseline) * np.exp(-5 * steepness)) / REFERENCE_LACTATE)
            ),
        },
        "day_to_day_form_sd_s_per_km": round(float(np.median(_draws(fit_result, "form_sd") / steepness)), 1),
        "diagnostics": {
            "n_reps": int(len(model_df)),
            "n_sessions": int(len(session_ids)),
            "outdoor_temperature_missing_sessions": missing_temp_sessions,
            "n_divergences": fit_result.n_divergences,
            "max_rhat": round(fit_result.max_rhat, 3),
            "converged": fit_result.max_rhat <= 1.05 and fit_result.n_divergences == 0,
        },
        "_fitness_draws": fitness,
        "_last_week_idx": last_week,
        "_week_starts": [str(d.date()) for d in all_weeks],
    }
