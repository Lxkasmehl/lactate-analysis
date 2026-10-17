from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lactate_analysis import fitness_model, report_data, report_html

# Small draws keep the suite fast; tests check direction and rough size of
# known synthetic effects, not tight numerical precision.
FAST_SAMPLE_KWARGS = dict(draws=300, tune=300, chains=2)

TRUE_SLOPE = -0.02  # log(lactate) per s/km
TRUE_TREADMILL_COST = -12.0  # s/km, treadmill easier
START_FITNESS, END_FITNESS = 205.0, 195.0  # pace at 2.0 mmol/L, s/km


def _make_synthetic_df(n_weeks: int = 40, seed: int = 0) -> pd.DataFrame:
    """Double-threshold days (AM ~2.0, PM ~3.0 mmol/L) with steadily improving
    fitness and a known treadmill effect; every other session on the treadmill."""
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2025-01-06")
    for week in range(n_weeks):
        fitness = START_FITNESS + (END_FITNESS - START_FITNESS) * week / (n_weeks - 1)
        date = start + pd.Timedelta(days=7 * week + 1)
        for period, target in (("AM", 2.0), ("PM", 3.0)):
            treadmill = (week + (period == "PM")) % 2 == 1
            day_form = rng.normal(0, 0.05)
            # The athlete aims for the lactate target, so the pace is chosen
            # around the pace that would produce it.
            target_pace = fitness + np.log(target / 2.0) / TRUE_SLOPE + (TRUE_TREADMILL_COST if treadmill else 0)
            for rep in range(1, 4):
                pace = target_pace + rng.normal(0, 3)
                effective = pace - (TRUE_TREADMILL_COST if treadmill else 0)
                log_lac = np.log(2.0) + TRUE_SLOPE * (effective - fitness) + day_form + rng.normal(0, 0.05)
                rows.append(
                    {
                        "date": date,
                        "session_period": period,
                        "session_id": f"{date:%Y%m%d}_{period}",
                        "rep_number": rep,
                        "pace_s_per_km": pace,
                        "lactate_mmol": float(np.exp(log_lac)),
                        "rep_time_s": 300.0,
                        "rest_duration_s": 60.0,
                        "rest_type": "Stand",
                        "surface": "Treadmill" if treadmill else "Track",
                        "incline_pct": 0.0 if treadmill else np.nan,
                        "temperature_c": 12.0,
                        "had_earlier_session_today": period == "PM",
                        "workout_type": "Double Threshold - Long Reps",
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted():
    df = _make_synthetic_df()
    fit = fitness_model.fit(df, **FAST_SAMPLE_KWARGS)
    return df, fit, fitness_model.summarize(fit, [2.0, 3.0])


def test_detects_fitness_improvement(fitted):
    _, _, summary = fitted
    trend = [w for w in summary["trend"]["2.0"] if w["has_data"]]
    first, last = trend[0]["median"], trend[-1]["median"]
    assert last < first - 4  # true improvement is 10 s/km
    assert abs(first - START_FITNESS) < 6
    assert abs(last - END_FITNESS) < 6


def test_recovers_treadmill_effect_and_slope(fitted):
    _, _, summary = fitted
    treadmill = next(f for f in summary["factors"] if f["key"] == "treadmill")
    assert treadmill["ci95"][0] < TRUE_TREADMILL_COST < treadmill["ci95"][1]
    assert treadmill["learned_from_data"]
    slope_pct = summary["curve"]["slope_pct_per_s_per_km"]
    assert slope_pct["lo95"] < -TRUE_SLOPE * 100 < slope_pct["hi95"]


def test_three_mmol_trend_is_faster_than_two(fitted):
    _, _, summary = fitted
    for w2, w3 in zip(summary["trend"]["2.0"], summary["trend"]["3.0"]):
        assert w3["median"] < w2["median"]


def test_normalized_session_pace_removes_treadmill_advantage(fitted):
    _, _, summary = fitted
    sessions = pd.DataFrame(summary["sessions"])
    raw_gap = (
        sessions.loc[sessions["surface"] == "Treadmill", "mean_pace_s_per_km"].mean()
        - sessions.loc[sessions["surface"] == "Track", "mean_pace_s_per_km"].mean()
    )
    normalized = sessions["normalized_pace"].map(lambda p: p["2.0"])
    normalized_gap = (
        normalized[sessions["surface"] == "Treadmill"].mean() - normalized[sessions["surface"] == "Track"].mean()
    )
    assert abs(normalized_gap) < abs(raw_gap) / 2


def test_summary_and_report_render(fitted, tmp_path):
    df, _, fitness_summary = fitted
    cycle_defs = [
        {"id": "c1", "label": "Early", "start_date": pd.Timestamp("2025-01-01"), "end_date": pd.Timestamp("2025-05-31")},
        {"id": "c2", "label": "Late", "start_date": pd.Timestamp("2025-06-01"), "end_date": pd.Timestamp.max},
    ]
    validation_report = {
        "n_rows_total": len(df),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "covariate_fill_rates": {"temperature_c": 1.0},
    }
    summary = report_data.build_summary(df, validation_report, fitness_summary, cycle_defs)
    assert summary["cycles"][1]["change_vs_previous"]["verdict"] == "faster"
    out = tmp_path / "report.html"
    report_html.write_report(summary, out)
    assert "Lactate Fitness Report" in out.read_text(encoding="utf-8")


def test_format_pace():
    assert report_html.format_pace(195.0) == "3:15/km (5:14/mi)"
