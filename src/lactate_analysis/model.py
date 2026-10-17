"""Bayesian hierarchical statistical modeling: log(lactate) ~ pace + covariates.

log(lactate) ~ pace (rather than lactate ~ pace) makes this an exponential
lactate-vs-pace curve in the original scale -- the physiologically correct
shape for graded exercise tests (blood lactate rises exponentially with
intensity, not linearly). Fit with `bambi` (a formula-based Bayesian GLMM
interface on top of PyMC): session_id and workout_type are *varying*
(partially-pooled) intercepts, so a workout type with only 1-2 sessions in a
given cycle still gets an estimate shrunk toward -- not absent from, not
naively equal to -- what that type looks like across the whole dataset.
Cycle stays a *fixed* effect (with a pace:cycle interaction): with only 3
training cycles, there are too few groups for hierarchical pooling to
usefully estimate cycle-to-cycle variance (a well-known small-group-count
caveat for hierarchical models). Falls back to cluster-robust OLS on
log(lactate) (same exponential fix, ordinary fixed dummies instead of partial
pooling) if sampling fails to run or doesn't converge.

PYTENSOR_FLAGS=cxx= disables PyTensor's C++ compilation backend, which needs
a working C++ toolchain PyTensor could not use on this machine (Windows,
Strawberry Perl's bundled g++ failed to compile PyTensor's linker
extension). The pure-Python fallback linker is slower but needs no compiler.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf

from . import config
from .validation import OPTIONAL_CATEGORICAL_COVARIATES

BAYESIAN_DRAWS = 1000
BAYESIAN_TUNE = 1000
BAYESIAN_CHAINS = 4
BAYESIAN_TARGET_ACCEPT = 0.9
MAX_RHAT_FOR_CONVERGENCE = 1.05
MAX_DIVERGENCE_FRACTION = 0.01
UNSEEN_SESSION_ID = "__unseen_session__"


class ModelResult:
    def __init__(
        self,
        method,
        df,
        pace_mean,
        pace_std,
        covariates_used,
        workout_type_included,
        numeric_covariates,
        fixed_categoricals,
        rep_number_varies_by_workout_type,
        redundant_categoricals,
        bambi_model=None,
        idata=None,
        fit_result=None,
        formula_rhs=None,
    ):
        self.method = method
        self.df = df
        self.pace_mean = pace_mean
        self.pace_std = pace_std
        self.covariates_used = covariates_used
        self.workout_type_included = workout_type_included
        self.numeric_covariates = numeric_covariates
        self.fixed_categoricals = fixed_categoricals
        self.rep_number_varies_by_workout_type = rep_number_varies_by_workout_type
        self.redundant_categoricals = redundant_categoricals
        self.bambi_model = bambi_model
        self.idata = idata
        self.fit_result = fit_result
        self.formula_rhs = formula_rhs


def _collapse_rare_categories(series: pd.Series, min_count: int) -> pd.Series:
    """Merge categories with fewer than min_count rows into "Other".

    Only used for *fixed*-effect categoricals (not session_id/workout_type,
    which get real partial pooling instead): a level backed by only 1-2 rows
    makes an ordinary dummy coefficient unstable or barely inestimable.
    """
    counts = series.value_counts()
    rare_levels = counts[counts < min_count].index
    if len(rare_levels) == 0:
        return series
    return series.where(~series.isin(rare_levels), "Other")


def select_covariates(validation_report: dict) -> list[str]:
    return list(validation_report["covariates_included_numeric"]) + list(
        validation_report["covariates_included_categorical"]
    )


def _prepare_model_df(df: pd.DataFrame, validation_report: dict):
    df = df[df["cycle_label"] != config.UNLABELED_CYCLE_LABEL]

    # An all-out/race effort is a different physiological regime (different
    # lactate kinetics, different measurement timing relative to the effort)
    # from a repeatable submaximal training structure -- excluded from the
    # curve fit by workout_type, not by lactate value (a hard *training*
    # session, whatever its lactate reading, still belongs in the fit; only
    # rows actually logged as a maximal-effort type don't).
    if "workout_type" in df.columns:
        df = df[~df["workout_type"].isin(config.MAX_EFFORT_WORKOUT_TYPES)]

    covariates = select_covariates(validation_report)
    required = ["lactate_mmol", "pace_s_per_km", "cycle_label"] + covariates
    model_df = df.dropna(subset=required).copy()
    if model_df.empty:
        raise ValueError("No rows with complete lactate/pace/cycle data available to fit a model.")

    pace_mean = float(model_df["pace_s_per_km"].mean())
    pace_std = float(model_df["pace_s_per_km"].std() or 1.0)
    model_df["pace_z"] = (model_df["pace_s_per_km"] - pace_mean) / pace_std
    model_df["log_lactate"] = np.log(model_df["lactate_mmol"])

    # workout_type gets a varying (partially-pooled) intercept instead of
    # ordinary fixed dummies -- pulled out of the "fixed categoricals" list.
    workout_type_included = "workout_type" in covariates and model_df["workout_type"].nunique(dropna=True) >= 2
    if workout_type_included:
        model_df["workout_type"] = model_df["workout_type"].astype(str)

    # rep_number's relationship with lactate is workout-type-dependent (some
    # sessions build fatigue through the reps, some settle down as the athlete
    # warms up) -- not one fixed direction/magnitude for every workout type.
    # When workout_type is available, rep_number becomes a *varying slope* per
    # type (partially pooled, like the intercept) instead of one fixed
    # coefficient forced across all of them.
    rep_number_available = "rep_number" in covariates and model_df["rep_number"].std(skipna=True) > 1e-8
    rep_number_varies_by_workout_type = rep_number_available and workout_type_included

    numeric_covariates = [
        c
        for c in covariates
        if c not in OPTIONAL_CATEGORICAL_COVARIATES
        and not (c == "rep_number" and rep_number_varies_by_workout_type)
        and model_df[c].std(skipna=True) > 1e-8
    ]

    fixed_categorical_candidates = [
        c for c in covariates if c in OPTIONAL_CATEGORICAL_COVARIATES and c != "workout_type"
    ]
    for cat_col in fixed_categorical_candidates:
        model_df[cat_col] = _collapse_rare_categories(model_df[cat_col].astype(str), config.MIN_CATEGORY_COUNT)
    fixed_categoricals = [c for c in fixed_categorical_candidates if model_df[c].nunique(dropna=True) >= 2]

    # Drop a fixed categorical fully determined by one already accepted (e.g.
    # every "Treadmill" surface row is always "Indoor") -- an exact linear
    # combination of the other's dummies, making the design matrix
    # rank-deficient. Recorded (not just silently dropped): "why did X
    # disappear" is a real question the report should be able to answer.
    accepted_categoricals: list[str] = []
    redundant_categoricals: list[tuple[str, str]] = []
    for col in fixed_categoricals:
        redundant_with = next(
            (
                kept
                for kept in accepted_categoricals
                if (model_df.groupby(kept, observed=True)[col].nunique(dropna=False) <= 1).all()
            ),
            None,
        )
        if redundant_with is None:
            accepted_categoricals.append(col)
        else:
            redundant_categoricals.append((col, redundant_with))
    fixed_categoricals = accepted_categoricals

    return (
        model_df,
        pace_mean,
        pace_std,
        numeric_covariates,
        fixed_categoricals,
        workout_type_included,
        rep_number_varies_by_workout_type,
        redundant_categoricals,
    )


def _formula_terms(numeric_covariates: list[str], fixed_categoricals: list[str], multi_cycle: bool) -> list[str]:
    # With only one training cycle (e.g. a brand-new user who hasn't finished
    # a second block yet), cycle_label is a constant and the pace_z:cycle_label
    # interaction is a single-level factor -- formulae/patsy can't build a
    # design column for that ("need at least one array to concatenate") --
    # so drop both terms entirely rather than crash.
    cycle_terms = ["cycle_label", "pace_z:cycle_label"] if multi_cycle else []
    return ["pace_z"] + cycle_terms + numeric_covariates + fixed_categoricals


def fit_model(
    df: pd.DataFrame,
    validation_report: dict,
    draws: int = BAYESIAN_DRAWS,
    tune: int = BAYESIAN_TUNE,
    chains: int = BAYESIAN_CHAINS,
    target_accept: float = BAYESIAN_TARGET_ACCEPT,
) -> ModelResult:
    (
        model_df,
        pace_mean,
        pace_std,
        numeric_covariates,
        fixed_categoricals,
        workout_type_included,
        rep_number_varies_by_workout_type,
        redundant_categoricals,
    ) = _prepare_model_df(df, validation_report)
    covariates_used = (
        numeric_covariates
        + fixed_categoricals
        + (["workout_type"] if workout_type_included else [])
        + (["rep_number"] if rep_number_varies_by_workout_type else [])
    )
    n_sessions = model_df["session_id"].nunique()
    multi_cycle = model_df["cycle_label"].nunique() >= 2

    if n_sessions >= config.MIN_SESSIONS_FOR_BAYESIAN:
        try:
            fixed_terms = _formula_terms(numeric_covariates, fixed_categoricals, multi_cycle)
            if rep_number_varies_by_workout_type:
                varying_terms = "(1|session_id) + (1 + rep_number|workout_type)"
            elif workout_type_included:
                varying_terms = "(1|session_id) + (1|workout_type)"
            else:
                varying_terms = "(1|session_id)"
            formula = f"log_lactate ~ {' + '.join(fixed_terms)} + {varying_terms}"

            bambi_model = bmb.Model(formula, model_df)
            idata = bambi_model.fit(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                progressbar=False,
                random_seed=0,
            )

            # az.rhat() returns an xarray DataTree in this arviz version, not
            # a plain Dataset -- .to_dataset() first to get something
            # .to_array() can act on.
            max_rhat = float(az.rhat(idata).to_dataset().to_array().max())
            n_draws_total = draws * chains
            n_divergences = int(idata.sample_stats["diverging"].to_numpy().sum())
            if max_rhat <= MAX_RHAT_FOR_CONVERGENCE and n_divergences <= MAX_DIVERGENCE_FRACTION * n_draws_total:
                return ModelResult(
                    "bayesian",
                    model_df,
                    pace_mean,
                    pace_std,
                    covariates_used,
                    workout_type_included,
                    numeric_covariates,
                    fixed_categoricals,
                    rep_number_varies_by_workout_type,
                    redundant_categoricals,
                    bambi_model=bambi_model,
                    idata=idata,
                )
        except Exception:
            pass

    # Fallback: cluster-robust OLS on log(lactate) -- keeps the exponential
    # functional-form fix, but workout_type (and rep_number's per-type slope)
    # become ordinary fixed effects (no partial pooling) since there's no
    # Bayesian machinery to shrink them.
    fallback_terms = _formula_terms(numeric_covariates, [f"C({c})" for c in fixed_categoricals], multi_cycle)
    if rep_number_varies_by_workout_type:
        fallback_terms += ["C(workout_type)", "rep_number:C(workout_type)"]
    elif workout_type_included:
        fallback_terms.append("C(workout_type)")
    formula_rhs = " + ".join(fallback_terms)
    ols = smf.ols(f"log_lactate ~ {formula_rhs}", data=model_df)
    fit_result = ols.fit(cov_type="cluster", cov_kwds={"groups": model_df["session_id"]})

    return ModelResult(
        "ols_cluster_robust",
        model_df,
        pace_mean,
        pace_std,
        covariates_used,
        workout_type_included,
        numeric_covariates,
        fixed_categoricals,
        rep_number_varies_by_workout_type,
        redundant_categoricals,
        fit_result=fit_result,
        formula_rhs=formula_rhs,
    )


def reference_covariate_values(df: pd.DataFrame, covariates: list[str]) -> dict:
    values: dict[str, object] = {}
    for c in covariates:
        if c in OPTIONAL_CATEGORICAL_COVARIATES:
            mode = df[c].mode()
            values[c] = mode.iloc[0] if not mode.empty else "unknown"
        else:
            values[c] = float(df[c].mean())
    return values


def _row_dict(model_result: ModelResult, cycle_label: str, pace_z: float, covariate_reference: dict) -> dict:
    row = {"pace_z": pace_z, "cycle_label": cycle_label}
    for cov in model_result.covariates_used:
        row[cov] = covariate_reference.get(cov)
    return row


def _bayesian_log_lactate_draws(model_result: ModelResult, rows: list[dict]) -> np.ndarray:
    """Posterior draws of log(lactate) for each row, shape (n_draws, n_rows).

    All rows share one placeholder "unseen" session_id so `sample_new_groups`
    draws a *single* new session-level offset per posterior sample, applied
    identically to every row in this call -- required so that differencing
    predictions at pace_z=0 and pace_z=1 (to invert for pace-at-lactate)
    isn't contaminated by comparing two independently-sampled session
    offsets. Population-average prediction (marginalizing over sessions)
    rather than picking a specific real session.

    A fixed random_seed makes this deterministic given the same fitted
    model: without it, `sample_new_groups` draws from numpy's global random
    state, so re-running the same report (or even just calling this twice
    within one run, as predict_pace_at_lactate does for two different
    targets) could silently produce slightly different numbers each time --
    at odds with "the same data always produces the same report".
    """
    rows_df = pd.DataFrame(rows)
    rows_df["session_id"] = UNSEEN_SESSION_ID
    pred = model_result.bambi_model.predict(
        model_result.idata, data=rows_df, inplace=False, sample_new_groups=True, random_seed=0
    )
    mu = pred.posterior["mu"].to_numpy()  # (chain, draw, obs)
    return mu.reshape(-1, mu.shape[-1])


def _build_design_row(model_result: ModelResult, row_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Design vector, params, and their covariance for the OLS fallback path."""
    fit_result = model_result.fit_result
    design_info = getattr(fit_result.model.data, "model_spec", None) or fit_result.model.data.design_info
    design_df = patsy.build_design_matrices([design_info], row_df, return_type="dataframe")[0]

    param_names = fit_result.params.index
    params = fit_result.params.to_numpy()
    cov = fit_result.cov_params().loc[param_names, param_names].to_numpy()

    design_df = design_df.reindex(columns=param_names, fill_value=0.0)
    x = design_df.iloc[0].to_numpy()
    return x, params, cov


def predict_at_reference_pace(
    model_result: ModelResult,
    cycle_label: str,
    reference_pace_s_per_km: float,
    covariate_reference: dict,
) -> dict:
    df = model_result.df
    cycle_rows = df[df["cycle_label"] == cycle_label]
    n_sessions = int(cycle_rows["session_id"].nunique())

    pace_min, pace_max = cycle_rows["pace_s_per_km"].min(), cycle_rows["pace_s_per_km"].max()
    if not (pace_min <= reference_pace_s_per_km <= pace_max):
        return {
            "cycle_label": cycle_label,
            "out_of_range": True,
            "pace_range_s_per_km": [round(float(pace_min), 2), round(float(pace_max), 2)],
        }

    pace_z = (reference_pace_s_per_km - model_result.pace_mean) / model_result.pace_std

    if model_result.method == "bayesian":
        log_draws = _bayesian_log_lactate_draws(
            model_result, [_row_dict(model_result, cycle_label, pace_z, covariate_reference)]
        )[:, 0]
        lactate_draws = np.exp(log_draws)
        mean = float(np.median(lactate_draws))
        ci = (float(np.percentile(lactate_draws, 2.5)), float(np.percentile(lactate_draws, 97.5)))
    else:
        row_df = pd.DataFrame([_row_dict(model_result, cycle_label, pace_z, covariate_reference)])
        x, params, cov = _build_design_row(model_result, row_df)
        log_mean = float(x @ params)
        log_se = float(np.sqrt(max(x @ cov @ x, 0.0)))
        mean = float(np.exp(log_mean))
        ci = (float(np.exp(log_mean - 1.96 * log_se)), float(np.exp(log_mean + 1.96 * log_se)))

    return {
        "cycle_label": cycle_label,
        "predicted_lactate": round(mean, 3),
        "ci": [round(ci[0], 3), round(ci[1], 3)],
        "n_sessions": n_sessions,
        "n_reps": int(len(cycle_rows)),
        "low_confidence": n_sessions < config.LOW_CONFIDENCE_MIN_SESSIONS,
        "out_of_range": False,
    }


def predict_pace_at_lactate(
    model_result: ModelResult,
    cycle_label: str,
    target_lactate: float,
    covariate_reference: dict,
    observed_rows: pd.DataFrame | None = None,
) -> dict:
    """Invert the fitted log(lactate) ~ pace (+ covariates) relationship for
    one cycle to find the pace at a fixed lactate concentration.

    Inverting a single fitted (log-linear = exponential in the original
    scale) relationship per cycle, rather than averaging raw per-session
    point-estimates across whichever sessions happened to reach each target,
    is what guarantees a higher target lactate always maps to a faster pace.
    For the Bayesian path this is done per posterior draw (no linearization,
    can't produce a negative pace since the inversion happens in the space
    where the model actually is linear); the OLS fallback uses the
    delta-method approximation on the log scale instead.
    """
    df = model_result.df
    cycle_rows = df[df["cycle_label"] == cycle_label]
    lactate_min, lactate_max = cycle_rows["lactate_mmol"].min(), cycle_rows["lactate_mmol"].max()
    pace_obs_min, pace_obs_max = float(cycle_rows["pace_s_per_km"].min()), float(cycle_rows["pace_s_per_km"].max())
    n_sessions = int(cycle_rows["session_id"].nunique())

    if observed_rows is None:
        observed_rows = cycle_rows
    n_observed = int(observed_rows["session_id"].nunique())

    # Every branch below returns this same set of keys (None/False where not
    # applicable) so downstream code and templates never have to distinguish
    # "key absent" from "key is None" (Jinja's Undefined, not None).
    result = {
        "cycle_label": cycle_label,
        "target_lactate": target_lactate,
        "pace_s_per_km": None,
        "ci_s_per_km": None,
        "n_sessions": n_sessions,
        "n_observed_sessions": n_observed,
        "low_confidence": n_sessions < config.LOW_CONFIDENCE_MIN_SESSIONS,
        "unexpected_relationship": False,
        "implausible": False,
        "wide_uncertainty": False,
        "out_of_range": False,
        "unsolvable": False,
        "lactate_range": [round(float(lactate_min), 2), round(float(lactate_max), 2)],
    }

    if not (lactate_min <= target_lactate <= lactate_max):
        result["out_of_range"] = True
        return result

    log_target = float(np.log(target_lactate))

    if model_result.method == "bayesian":
        rows = [
            _row_dict(model_result, cycle_label, 0.0, covariate_reference),
            _row_dict(model_result, cycle_label, 1.0, covariate_reference),
        ]
        log_draws = _bayesian_log_lactate_draws(model_result, rows)
        intercept_draws = log_draws[:, 0]
        slope_draws = log_draws[:, 1] - intercept_draws

        # Only draws with the physiologically valid sign (lactate rises as
        # pace_s_per_km falls, i.e. a *negative* slope here): a draw with
        # slope near zero -- even technically nonzero -- gives a ratio with
        # an enormous or wrong-signed pace_z, and mixing those into the
        # posterior alongside valid draws produces exactly the kind of wild,
        # sometimes-negative credible interval this replaced the delta-method
        # approximation to avoid. Wrong-signed draws are what
        # unexpected_relationship already tracks; they don't further belong
        # in the pace-at-lactate distribution itself.
        usable = slope_draws < -1e-6
        unexpected_relationship = bool(np.mean(slope_draws >= 0) > 0.5)
        if usable.mean() < 0.5:
            result["unsolvable"] = True
            result["unexpected_relationship"] = unexpected_relationship
            return result

        pace_z_draws = (log_target - intercept_draws[usable]) / slope_draws[usable]
        pace_draws = pace_z_draws * model_result.pace_std + model_result.pace_mean
        pace_s_per_km = float(np.median(pace_draws))
        ci = (float(np.percentile(pace_draws, 2.5)), float(np.percentile(pace_draws, 97.5)))
    else:
        row0_df = pd.DataFrame([_row_dict(model_result, cycle_label, 0.0, covariate_reference)])
        row1_df = pd.DataFrame([_row_dict(model_result, cycle_label, 1.0, covariate_reference)])
        x0, params, cov = _build_design_row(model_result, row0_df)
        x1_full, _, _ = _build_design_row(model_result, row1_df)
        x1 = x1_full - x0

        intercept_term = float(x0 @ params)
        slope = float(x1 @ params)
        if abs(slope) < 1e-6:
            result["unsolvable"] = True
            return result

        pace_z = (log_target - intercept_term) / slope
        pace_s_per_km = pace_z * model_result.pace_std + model_result.pace_mean

        grad = (-x0 * slope - (log_target - intercept_term) * x1) / (slope**2)
        se_pace_z = float(np.sqrt(max(grad @ cov @ grad, 0.0)))
        se_pace_s_per_km = se_pace_z * model_result.pace_std
        ci = (pace_s_per_km - 1.96 * se_pace_s_per_km, pace_s_per_km + 1.96 * se_pace_s_per_km)
        unexpected_relationship = slope > 0

    result["unexpected_relationship"] = unexpected_relationship

    # Plausibility relative to *this cycle's own* observed pace range, not an
    # absolute human-population bound -- a bound loose enough to cover all
    # humans is far too loose to catch "unrealistic for this specific
    # athlete" (e.g. 4:14/mile from a recreational runner's own data).
    margin = max(
        config.PLAUSIBLE_PACE_MARGIN_FRACTION * (pace_obs_max - pace_obs_min),
        config.PLAUSIBLE_PACE_MARGIN_MIN_S_PER_KM,
    )
    lo_bound, hi_bound = pace_obs_min - margin, pace_obs_max + margin

    # Only the *point estimate* being absurd hides the number entirely (that
    # was the actual bug: a wrong central estimate presented with false
    # confidence). A credible interval that stretches into implausible
    # territory while the point estimate itself is fine is a different,
    # honest situation -- genuinely wide uncertainty, not a wrong answer --
    # so it's kept and flagged, with its displayed bounds clamped to the
    # plausible range rather than printing something like "-868 min/mi".
    result["implausible"] = not (lo_bound <= pace_s_per_km <= hi_bound)
    if result["implausible"]:
        return result

    result["pace_s_per_km"] = round(pace_s_per_km, 2)
    ci_lo, ci_hi = ci
    result["wide_uncertainty"] = ci_lo < lo_bound or ci_hi > hi_bound
    if result["wide_uncertainty"]:
        ci_lo, ci_hi = max(ci_lo, lo_bound), min(ci_hi, hi_bound)
    result["ci_s_per_km"] = [round(ci_lo, 2), round(ci_hi, 2)]
    return result


def _covariate_effect_pct(model_result: ModelResult, base_row: dict, changed_row: dict) -> dict:
    """% change in lactate between two prediction rows differing in exactly
    one covariate, with the rest held fixed. Computed by differencing model
    predictions (the same predict() pathway used everywhere else) rather
    than reading raw coefficients off the posterior directly, so it's
    correct regardless of the model's internal parameter coding (e.g.
    formulae's dummy-coding choice can differ term to term)."""
    if model_result.method == "bayesian":
        log_draws = _bayesian_log_lactate_draws(model_result, [base_row, changed_row])
        diff_draws = log_draws[:, 1] - log_draws[:, 0]
        pct_draws = (np.exp(diff_draws) - 1) * 100
        return {
            "pct_change": round(float(np.median(pct_draws)), 1),
            "pct_change_ci": [round(float(np.percentile(pct_draws, 2.5)), 1), round(float(np.percentile(pct_draws, 97.5)), 1)],
        }

    x0, params, cov = _build_design_row(model_result, pd.DataFrame([base_row]))
    x1, _, _ = _build_design_row(model_result, pd.DataFrame([changed_row]))
    diff = x1 - x0
    diff_mean = float(diff @ params)
    diff_se = float(np.sqrt(max(diff @ cov @ diff, 0.0)))
    return {
        "pct_change": round((np.exp(diff_mean) - 1) * 100, 1),
        "pct_change_ci": [
            round((np.exp(diff_mean - 1.96 * diff_se) - 1) * 100, 1),
            round((np.exp(diff_mean + 1.96 * diff_se) - 1) * 100, 1),
        ],
    }


def compute_covariate_effects(model_result: ModelResult) -> list[dict]:
    """Estimated effect of each covariate on lactate (% change, with
    uncertainty) -- the "which factors matter and by how much" answer this
    project was actually built to give, not just a single derived
    pace-at-lactate number. Each entry compares a reference row against one
    covariate changed, everything else held fixed; numeric covariates are
    compared per +1 SD (a comparable unit across differently-scaled
    covariates), categorical covariates per non-reference level.

    Deliberately does NOT attempt to summarize rep_number's effect with one
    number or a directional story ("busier workouts -> rising lactate"):
    that's workout-type-specific by construction here (a varying slope per
    type), and even that is only a rough proxy for the real drivers (what you
    ate, whether you'd already trained earlier that day, whether the first
    rep was paced too fast...), most of which aren't recorded at all. Each
    workout type's own estimated rep_number slope is reported separately,
    with its own uncertainty, instead of forcing one summary.
    """
    covariate_reference = reference_covariate_values(model_result.df, model_result.covariates_used)
    cycle_label = str(model_result.df["cycle_label"].iloc[0])
    effects: list[dict] = []

    for cov in model_result.numeric_covariates:
        sd = float(model_result.df[cov].std())
        if not sd or sd < 1e-8:
            continue
        base_row = _row_dict(model_result, cycle_label, 0.0, covariate_reference)
        changed_row = dict(base_row, **{cov: covariate_reference[cov] + sd})
        effects.append(
            {
                "covariate": cov,
                "comparison": f"+1 SD (~{sd:.1f})",
                "partially_pooled": False,
                **_covariate_effect_pct(model_result, base_row, changed_row),
            }
        )

    for cov in model_result.fixed_categoricals:
        reference_level = covariate_reference[cov]
        for level in sorted(model_result.df[cov].dropna().unique()):
            if level == reference_level:
                continue
            base_row = _row_dict(model_result, cycle_label, 0.0, covariate_reference)
            changed_row = dict(base_row, **{cov: level})
            effects.append(
                {
                    "covariate": cov,
                    "comparison": f"{level} vs {reference_level}",
                    "partially_pooled": False,
                    **_covariate_effect_pct(model_result, base_row, changed_row),
                }
            )

    if model_result.workout_type_included:
        reference_level = covariate_reference["workout_type"]
        for level in sorted(model_result.df["workout_type"].dropna().unique()):
            if level == reference_level:
                continue
            base_row = _row_dict(model_result, cycle_label, 0.0, covariate_reference)
            changed_row = dict(base_row, workout_type=level)
            effects.append(
                {
                    "covariate": "workout_type",
                    "comparison": f"{level} vs {reference_level}",
                    "partially_pooled": True,
                    **_covariate_effect_pct(model_result, base_row, changed_row),
                }
            )

    if model_result.rep_number_varies_by_workout_type:
        rep_mean = covariate_reference["rep_number"]
        for workout_type in sorted(model_result.df["workout_type"].dropna().unique()):
            base_row = _row_dict(model_result, cycle_label, 0.0, covariate_reference)
            base_row["workout_type"] = workout_type
            changed_row = dict(base_row, rep_number=rep_mean + 1)
            effects.append(
                {
                    "covariate": f"rep_number (within {workout_type})",
                    "comparison": "+1 rep",
                    "partially_pooled": True,
                    **_covariate_effect_pct(model_result, base_row, changed_row),
                }
            )

    return effects


def compute_pace_at_lactate(model_result: ModelResult, cycle_labels: list[str], targets: list[float]) -> dict:
    """Model-based pace-at-fixed-lactate per cycle, plus a per-workout-type
    breakdown.

    A (cycle, workout_type) combo uses this cycle's fitted slope together
    with that workout type's partially-pooled intercept offset (estimated
    from the entire dataset) -- so even a combo with only 1-2 sessions in
    this cycle gets an estimate shrunk toward, not absent from, what that
    type looks like elsewhere.
    """
    covariate_reference = reference_covariate_values(model_result.df, model_result.covariates_used)
    workout_types = (
        sorted(model_result.df["workout_type"].dropna().unique()) if model_result.workout_type_included else []
    )

    by_cycle: dict[str, dict] = {}
    by_cycle_workout_type: dict[str, dict] = {}

    for cycle_label in cycle_labels:
        by_cycle[cycle_label] = {
            str(t): predict_pace_at_lactate(model_result, cycle_label, t, covariate_reference) for t in targets
        }

        if not model_result.workout_type_included:
            continue

        cycle_rows = model_result.df[model_result.df["cycle_label"] == cycle_label]
        breakdown = {}
        for workout_type in workout_types:
            observed = cycle_rows[cycle_rows["workout_type"] == workout_type]
            if observed.empty:
                continue  # only report combos actually observed in this cycle
            type_reference = dict(covariate_reference, workout_type=workout_type)
            breakdown[workout_type] = {
                str(t): predict_pace_at_lactate(model_result, cycle_label, t, type_reference, observed_rows=observed)
                for t in targets
            }
        by_cycle_workout_type[cycle_label] = breakdown

    return {"by_cycle": by_cycle, "by_cycle_workout_type": by_cycle_workout_type}


def run_model_analysis(
    df: pd.DataFrame,
    validation_report: dict,
    reference_pace_s_per_km: float | None = None,
    target_lactate_levels: list[float] | None = None,
    draws: int = BAYESIAN_DRAWS,
    tune: int = BAYESIAN_TUNE,
    chains: int = BAYESIAN_CHAINS,
) -> dict:
    model_result = fit_model(df, validation_report, draws=draws, tune=tune, chains=chains)
    # Use the median of the actually-modeled rows (not the full raw df) so the
    # default reference pace is consistent with the per-cycle observed-range
    # check below, which also runs against model_result.df.
    reference_pace = reference_pace_s_per_km or float(model_result.df["pace_s_per_km"].median())
    covariate_reference = reference_covariate_values(model_result.df, model_result.covariates_used)

    # Chronological order by each cycle's earliest observed date -- cycle
    # labels are free text (e.g. "XC 2025", "Track 2026") and sorting them
    # alphabetically would not reflect training-cycle sequence.
    cycle_labels = model_result.df.groupby("cycle_label")["date"].min().sort_values().index.tolist()
    cycle_predictions = [
        predict_at_reference_pace(model_result, label, reference_pace, covariate_reference) for label in cycle_labels
    ]

    pace_at_lactate = compute_pace_at_lactate(model_result, cycle_labels, target_lactate_levels or config.TARGET_LACTATE_LEVELS)
    covariate_effects = compute_covariate_effects(model_result)

    return {
        "method": model_result.method,
        "covariates_used": model_result.covariates_used,
        "reference_pace_s_per_km": round(reference_pace, 2),
        "cycle_predictions": cycle_predictions,
        "cycle_order": cycle_labels,
        "pace_at_lactate": pace_at_lactate,
        "covariate_effects": covariate_effects,
        "redundant_categoricals": model_result.redundant_categoricals,
    }
