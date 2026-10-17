"""Assemble the single structured JSON summary that drives the HTML report.

This dict is the hand-off contract for later phases too (an LLM
interpretation layer, a future web backend) -- nothing downstream of this
function should need to touch the raw DataFrame again.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from .config import MAX_EFFORT_WORKOUT_TYPES, MILE_TO_KM


def _pace_s_per_km_to_min_per_mi(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value * MILE_TO_KM / 60, 2)


def _format_pace_at_lactate_entry(entry: dict) -> dict:
    """Convert one model.predict_pace_at_lactate() result to display units.

    Always sets pace_min_per_mi/ci_min_per_mi (to None if not applicable) so
    every entry has the same keys -- a key that's simply absent reads back as
    Jinja's Undefined in the template, not None, which silently breaks an
    "is not none" check.
    """
    formatted = dict(entry)
    ci = entry.get("ci_s_per_km")
    formatted["pace_min_per_mi"] = _pace_s_per_km_to_min_per_mi(entry.get("pace_s_per_km"))
    formatted["ci_min_per_mi"] = (
        [_pace_s_per_km_to_min_per_mi(ci[0]), _pace_s_per_km_to_min_per_mi(ci[1])] if ci else None
    )
    return formatted


def _format_pace_at_lactate(pace_at_lactate_result: dict, cycle_order: list[str]) -> dict:
    by_cycle = {
        cycle_label: {target: _format_pace_at_lactate_entry(entry) for target, entry in targets.items()}
        for cycle_label, targets in pace_at_lactate_result["by_cycle"].items()
    }
    by_cycle_workout_type = {
        cycle_label: {
            workout_type: {target: _format_pace_at_lactate_entry(entry) for target, entry in targets.items()}
            for workout_type, targets in breakdown.items()
        }
        for cycle_label, breakdown in pace_at_lactate_result["by_cycle_workout_type"].items()
    }
    return {"cycle_order": cycle_order, "by_cycle": by_cycle, "by_cycle_workout_type": by_cycle_workout_type}


def _format_trend_points(trend_points: list[dict]) -> list[dict]:
    formatted = []
    for point in trend_points:
        converted = {
            "date": point["date"],
            "cycle_label": point["cycle_label"],
            "workout_type": point["workout_type"],
        }
        for key, value in point.items():
            if key.startswith("pace_at_"):
                converted[key] = _pace_s_per_km_to_min_per_mi(value)
        formatted.append(converted)
    return formatted


def _trend_direction(cycle_predictions: list[dict]) -> str:
    usable = [c for c in cycle_predictions if not c.get("out_of_range")]
    if len(usable) < 2:
        return "insufficient_data"
    first, last = usable[0], usable[-1]
    if last["predicted_lactate"] < first["predicted_lactate"] - 0.05:
        return "improving"
    if last["predicted_lactate"] > first["predicted_lactate"] + 0.05:
        return "declining"
    return "stable"


def _build_conclusion(trend_direction: str, cycle_predictions: list[dict], validation_report: dict) -> dict:
    headline = {
        "improving": (
            "Your estimated lactate response at a standardized pace has decreased across "
            "training cycles, consistent with improving fitness."
        ),
        "declining": (
            "Your estimated lactate response at a standardized pace has increased across training cycles."
        ),
        "stable": "Your estimated lactate response at a standardized pace has stayed roughly stable across cycles.",
        "insufficient_data": "Not enough comparable in-range cycles to assess a fitness trend yet.",
    }[trend_direction]

    supporting_points = []
    for c in cycle_predictions:
        if c.get("out_of_range"):
            supporting_points.append(
                f"{c['cycle_label']}: reference pace outside this cycle's observed range "
                f"({c['pace_range_s_per_km'][0]}-{c['pace_range_s_per_km'][1]} s/km) -- excluded from comparison."
            )
        else:
            note = " (low confidence)" if c.get("low_confidence") else ""
            supporting_points.append(
                f"{c['cycle_label']}: predicted lactate {c['predicted_lactate']} mmol/L "
                f"(95% CI {c['ci'][0]}-{c['ci'][1]}, n={c['n_sessions']} sessions){note}"
            )

    if validation_report["covariates_excluded_low_fill"]:
        supporting_points.append(
            "Excluded from the model due to insufficient data: "
            + ", ".join(validation_report["covariates_excluded_low_fill"])
        )

    return {"headline": headline, "supporting_points": supporting_points}


def build_summary(
    df: pd.DataFrame,
    validation_report: dict,
    analysis: dict,
    reference_pace_min_per_mi: float,
    trend_points: list[dict],
) -> dict:
    cycle_predictions = analysis["cycle_predictions"]
    trend_direction = _trend_direction(cycle_predictions)
    latest = cycle_predictions[-1] if cycle_predictions else None
    latest_usable = latest if latest and not latest.get("out_of_range") else None

    cycle_order = analysis["cycle_order"]
    pace_at_lactate_summary = _format_pace_at_lactate(analysis["pace_at_lactate"], cycle_order)
    pace_at_lactate_summary["trend_points"] = _format_trend_points(trend_points)

    latest_cycle_label = cycle_order[-1] if cycle_order else None
    pace_at_3mmol_latest = None
    if latest_cycle_label:
        entry = pace_at_lactate_summary["by_cycle"].get(latest_cycle_label, {}).get("3.0", {})
        if not entry.get("implausible"):
            pace_at_3mmol_latest = entry.get("pace_min_per_mi")

    unexpected_relationship_cycles = sorted(
        {
            cycle_label
            for cycle_label, targets in pace_at_lactate_summary["by_cycle"].items()
            for entry in targets.values()
            if entry.get("unexpected_relationship")
        }
    )
    implausible_estimates = [
        f"{cycle_label} @ {target} mmol/L"
        for cycle_label, targets in pace_at_lactate_summary["by_cycle"].items()
        for target, entry in targets.items()
        if entry.get("implausible")
    ]

    curves: dict[str, dict] = {}
    for label in df["cycle_label"].dropna().unique():
        sub = df[df["cycle_label"] == label]
        curves[label] = {
            "points": [
                {"pace_s_per_km": round(float(p), 2), "lactate_mmol": round(float(l), 3)}
                for p, l in zip(sub["pace_s_per_km"], sub["lactate_mmol"])
                if pd.notna(p) and pd.notna(l)
            ]
        }

    surface_mix: dict[str, dict[str, float]] = {}
    for label, sub in df.groupby("cycle_label"):
        counts = sub["surface"].fillna("unknown").value_counts(normalize=True)
        surface_mix[str(label)] = {str(k): round(float(v), 3) for k, v in counts.items()}

    max_effort_mask = df["workout_type"].isin(MAX_EFFORT_WORKOUT_TYPES) if "workout_type" in df.columns else None
    max_effort_rows_excluded = int(max_effort_mask.sum()) if max_effort_mask is not None else 0

    # Largest-magnitude effects first -- "which factors matter most", not
    # just an unordered dump of every covariate the model happened to use.
    covariate_effects = sorted(
        analysis.get("covariate_effects", []), key=lambda e: abs(e["pct_change"]), reverse=True
    )

    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "data_summary": {
            "n_rows_total": validation_report["n_rows_total"],
            "n_rows_modeled": validation_report["n_rows_modeled"],
            "n_sessions": validation_report["n_sessions"],
            "date_range": validation_report["date_range"],
        },
        "current_status": {
            "latest_cycle": latest["cycle_label"] if latest else None,
            "predicted_lactate_at_ref_pace": latest_usable["predicted_lactate"] if latest_usable else None,
            "ci": latest_usable["ci"] if latest_usable else None,
            "reference_pace_min_per_mi": round(reference_pace_min_per_mi, 2),
            "trend_direction": trend_direction,
            "pace_at_3mmol_latest_cycle_min_per_mi": pace_at_3mmol_latest,
        },
        "lactate_curves": curves,
        "cycle_comparison": {
            "reference_pace_min_per_mi": round(reference_pace_min_per_mi, 2),
            "cycles": cycle_predictions,
        },
        "pace_at_lactate": pace_at_lactate_summary,
        "covariate_effects": covariate_effects,
        "evidence_and_caveats": {
            "model_used": analysis["method"],
            "covariates_included": analysis["covariates_used"],
            "covariates_excluded_low_fill": validation_report["covariates_excluded_low_fill"],
            "covariates_excluded_categorical_low_fill": validation_report["covariates_excluded_categorical_low_fill"],
            "missingness": validation_report["covariate_fill_rates"],
            "low_confidence_cycles": [c["cycle_label"] for c in cycle_predictions if c.get("low_confidence")],
            "pace_deviation_caveat": (
                "pace_deviation_pct is a heuristic proxy based on prior reps in the same session, "
                "not a true target-pace comparison -- there is no target-pace column in the source data."
            ),
            "max_effort_rows_excluded": max_effort_rows_excluded,
            "max_effort_workout_types": MAX_EFFORT_WORKOUT_TYPES,
            "redundant_categoricals": [
                f"{dropped} (redundant with {kept})" for dropped, kept in analysis.get("redundant_categoricals", [])
            ],
            "unlabeled_rows": validation_report["unlabeled_rows"],
            "inferred_rep_numbers": validation_report["inferred_rep_numbers"],
            "surface_mix_by_cycle": surface_mix,
            "unexpected_relationship_cycles": unexpected_relationship_cycles,
            "implausible_estimates": implausible_estimates,
        },
        "conclusion": _build_conclusion(trend_direction, cycle_predictions, validation_report),
    }
