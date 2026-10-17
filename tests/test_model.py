from __future__ import annotations

import numpy as np
import pandas as pd

from lactate_analysis import model, validation

# Small draws/tune/chains keep the test suite's MCMC fits fast; tests check
# qualitative properties (direction, monotonicity, shrinkage), not tight
# numerical precision, so noisier posteriors are an acceptable trade.
FAST_SAMPLE_KWARGS = dict(draws=150, tune=150, chains=2)


def _make_synthetic_df(n_sessions: int, reps_per_session: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base_date = pd.Timestamp("2025-01-01")
    for s in range(n_sessions):
        date = base_date + pd.Timedelta(days=s * 5)
        cycle_label = "Cycle 1" if s < n_sessions // 2 else "Cycle 2"
        session_id = f"s{s}"
        # Cycle 2 is the "fitter" cycle: lower lactate at the *same* absolute
        # pace (an intercept shift), not just a different pace range sampled.
        base_pace = 240
        cycle_lactate_shift = -0.8 if cycle_label == "Cycle 2" else 0.0
        for r in range(reps_per_session):
            pace = base_pace + r * 5 + rng.normal(0, 1)
            lactate = max(
                2 + 0.03 * (300 - pace) + r * 0.5 + cycle_lactate_shift + rng.normal(0, 0.2), 0.5
            )
            rows.append(
                {
                    "date": date,
                    "session_id": session_id,
                    "rep_number": r + 1,
                    "pace_s_per_km": pace,
                    "lactate_mmol": lactate,
                    "cycle_label": cycle_label,
                    "surface": "Track",
                    "rest_type": "jogging",
                    "rest_duration_s": 90.0,
                    "temperature_c": 15.0,
                    "indoor_outdoor": "Outdoor",
                    "heart_rate": np.nan,
                    "humidity_pct": np.nan,
                    "incline_pct": np.nan,
                    "shoes": np.nan,
                }
            )
    df = pd.DataFrame(rows)
    df["session_mean_pace"] = df.groupby("session_id")["pace_s_per_km"].transform("mean")
    df["session_mean_lactate"] = df.groupby("session_id")["lactate_mmol"].transform("mean")
    df["prior_lactate"] = df.groupby("session_id")["lactate_mmol"].shift(1)
    df["days_since_project_start"] = (df["date"] - df["date"].min()).dt.days
    return df


def test_fit_model_uses_bayesian_when_enough_sessions():
    df = _make_synthetic_df(n_sessions=20)
    validation_report = validation.build_validation_report(df, min_fill_rate=0.6)
    analysis = model.run_model_analysis(df, validation_report, **FAST_SAMPLE_KWARGS)

    assert analysis["method"] == "bayesian"
    assert len(analysis["cycle_predictions"]) >= 1


def test_fit_model_falls_back_to_ols_with_few_sessions():
    df = _make_synthetic_df(n_sessions=6)
    validation_report = validation.build_validation_report(df, min_fill_rate=0.6)
    analysis = model.run_model_analysis(df, validation_report, **FAST_SAMPLE_KWARGS)

    assert analysis["method"] == "ols_cluster_robust"


def test_cycle_comparison_detects_improvement_direction():
    df = _make_synthetic_df(n_sessions=20)
    validation_report = validation.build_validation_report(df, min_fill_rate=0.6)
    analysis = model.run_model_analysis(df, validation_report, **FAST_SAMPLE_KWARGS)

    predictions = {c["cycle_label"]: c for c in analysis["cycle_predictions"] if not c.get("out_of_range")}
    assert "Cycle 1" in predictions and "Cycle 2" in predictions
    assert predictions["Cycle 2"]["predicted_lactate"] < predictions["Cycle 1"]["predicted_lactate"]


def test_pace_at_lactate_is_monotonic_in_target():
    """A higher target lactate must always map to a faster (lower) pace,
    since it comes from inverting one fitted (log-linear = exponential)
    relationship per cycle. This is the guarantee a naive per-session average
    across different sessions per target does not have -- see
    pace_at_lactate.py's module docstring for the real-data bug this
    replaced (e.g. "faster" estimated at 2.0 mmol/L than at 4.0 mmol/L)."""
    df = _make_synthetic_df(n_sessions=20)
    validation_report = validation.build_validation_report(df, min_fill_rate=0.6)
    analysis = model.run_model_analysis(
        df, validation_report, target_lactate_levels=[2.0, 3.0, 4.0], **FAST_SAMPLE_KWARGS
    )

    for cycle_label, targets in analysis["pace_at_lactate"]["by_cycle"].items():
        paces = {t: entry["pace_s_per_km"] for t, entry in targets.items() if not entry.get("out_of_range")}
        ordered_targets = sorted(paces, key=float)
        ordered_paces = [paces[t] for t in ordered_targets]
        assert ordered_paces == sorted(ordered_paces, reverse=True), (
            f"{cycle_label}: pace-at-lactate not monotonic: {paces}"
        )


def _make_workout_type_synthetic_df(seed: int = 2) -> tuple[pd.DataFrame, float]:
    """One cycle, five workout types of varying size (25/5/4/3/2 sessions),
    all following the exact same true pace-lactate relationship except the
    sparsest one ("Rare", 2 sessions), which gets a real, deliberately
    injected +0.6 mmol/L bump -- simulating a small subgroup whose only
    evidence is a couple of sessions. Returns (df, naive_raw_diff):
    naive_raw_diff is the plain mean(lactate | Rare) - mean(lactate | Common)
    an unpooled/stratified average would report at face value.
    """
    rng = np.random.default_rng(seed)
    rows = []
    base_date = pd.Timestamp("2025-01-01")
    session_idx = 0

    def _add_session(session_id: str, workout_type: str, bump: float):
        nonlocal session_idx
        date = base_date + pd.Timedelta(days=session_idx * 3)
        session_idx += 1
        for r in range(4):
            pace = 240 + r * 5 + rng.normal(0, 1)
            lactate = max(2 + 0.03 * (300 - pace) + r * 0.5 + bump + rng.normal(0, 0.15), 0.5)
            rows.append(
                {
                    "date": date,
                    "session_id": session_id,
                    "rep_number": r + 1,
                    "pace_s_per_km": pace,
                    "lactate_mmol": lactate,
                    "cycle_label": "Cycle 1",
                    "workout_type": workout_type,
                    "surface": "Track",
                    "rest_type": "jogging",
                    "rest_duration_s": 90.0,
                    "temperature_c": 15.0,
                    "indoor_outdoor": "Outdoor",
                    "heart_rate": np.nan,
                    "humidity_pct": np.nan,
                    "incline_pct": np.nan,
                    "shoes": np.nan,
                }
            )

    # Several groups of varying size (not just Common vs. Rare): a
    # workout_type factor with only 2 levels is a notoriously hard funnel
    # geometry for the sampler (its group-level variance is barely
    # identifiable from just 2 groups). The real data has ~7 types, so 5
    # here is more representative and lets the sampler estimate the
    # between-type variance far more stably, without changing what the test
    # is actually checking (shrinkage of the sparse "Rare" group).
    for s in range(25):
        _add_session(f"common{s}", "Common", bump=0.0)
    for s in range(5):
        _add_session(f"typeb{s}", "TypeB", bump=0.1)
    for s in range(4):
        _add_session(f"typec{s}", "TypeC", bump=-0.1)
    for s in range(3):
        _add_session(f"typed{s}", "TypeD", bump=0.2)
    for s in range(2):
        _add_session(f"rare{s}", "Rare", bump=0.6)

    df = pd.DataFrame(rows)
    df["session_mean_pace"] = df.groupby("session_id")["pace_s_per_km"].transform("mean")
    df["session_mean_lactate"] = df.groupby("session_id")["lactate_mmol"].transform("mean")
    df["prior_lactate"] = df.groupby("session_id")["lactate_mmol"].shift(1)
    df["days_since_project_start"] = (df["date"] - df["date"].min()).dt.days

    naive_raw_diff = df.loc[df["workout_type"] == "Rare", "lactate_mmol"].mean() - df.loc[
        df["workout_type"] == "Common", "lactate_mmol"
    ].mean()
    return df, float(naive_raw_diff)


def test_workout_type_partial_pooling_shrinks_sparse_group():
    """A workout type backed by only 2 sessions should get a model-based
    estimate pulled toward the dominant type's pattern -- not the sparse
    group's own noisy/small-sample average taken at face value. This is the
    concrete, testable signature of partial pooling actually working (the
    principled fix for "not enough data per category", vs. either ignoring
    the grouping entirely or splitting into disjoint per-category averages)."""
    df, naive_raw_diff = _make_workout_type_synthetic_df()
    validation_report = validation.build_validation_report(df, min_fill_rate=0.6)

    # rep_number now varies by workout_type (a correlated random slope +
    # intercept) -- a harder fit than a plain varying intercept, needs more
    # sampling budget than FAST_SAMPLE_KWARGS to converge reliably.
    model_result = model.fit_model(df, validation_report, draws=300, tune=300, chains=4, target_accept=0.95)
    assert model_result.method == "bayesian"
    assert model_result.workout_type_included

    covariate_reference = model.reference_covariate_values(model_result.df, model_result.covariates_used)
    reference_pace = float(model_result.df["pace_s_per_km"].median())
    cycle_label = "Cycle 1"

    common_pred = model.predict_at_reference_pace(
        model_result, cycle_label, reference_pace, dict(covariate_reference, workout_type="Common")
    )
    rare_pred = model.predict_at_reference_pace(
        model_result, cycle_label, reference_pace, dict(covariate_reference, workout_type="Rare")
    )
    model_diff = rare_pred["predicted_lactate"] - common_pred["predicted_lactate"]

    assert naive_raw_diff > 0.3, "test setup sanity check: the injected bump should show up in the raw means"
    assert 0 < model_diff < naive_raw_diff, (
        f"expected the model's Rare-vs-Common difference ({model_diff}) to be shrunk toward zero "
        f"relative to the naive raw difference ({naive_raw_diff}), while keeping the right direction"
    )
