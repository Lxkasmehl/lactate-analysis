"""Find earlier sessions most like a given one and compare them fairly.

Comparing similar sessions keeps the model's corrections small: when two
sessions share rep length, rest, surface and time of day, almost all of the
normalized difference comes straight from the measured pace and lactate.
Each comparison therefore reports how the difference splits into raw pace,
condition correction and lactate-curve correction, so it's visible how much
of the verdict depends on the model.
"""
from __future__ import annotations

import math

N_SIMILAR = 3

# Distance weights: one "unit" of dissimilarity is roughly a doubling of rep
# duration. Rest share, surface and time of day matter less once rep length
# matches; workout type is a coarse label and only breaks ties.
WEIGHT_REP_DURATION = 1.0
WEIGHT_REST_FRACTION = 4.0
WEIGHT_SURFACE = 1.0
WEIGHT_PERIOD = 0.5
WEIGHT_WORKOUT_TYPE = 0.3
WEIGHT_TEMPERATURE_PER_10C = 0.3


def _rest_fraction(session: dict) -> float:
    return session["rest_s"] / (session["mean_rep_time_s"] + session["rest_s"])


def distance(a: dict, b: dict) -> float:
    d = WEIGHT_REP_DURATION * abs(math.log2(a["mean_rep_time_s"] / b["mean_rep_time_s"]))
    d += WEIGHT_REST_FRACTION * abs(_rest_fraction(a) - _rest_fraction(b))
    d += WEIGHT_SURFACE * (a["surface_group"] != b["surface_group"])
    d += WEIGHT_PERIOD * (a["period"] != b["period"])
    d += WEIGHT_WORKOUT_TYPE * (a["workout_type"] != b["workout_type"])
    if a["temperature_c"] is not None and b["temperature_c"] is not None:
        d += WEIGHT_TEMPERATURE_PER_10C * abs(a["temperature_c"] - b["temperature_c"]) / 10
    return d


def compare(current: dict, earlier: dict, reference_lactate: str) -> dict:
    """Differences current - earlier in s/km; negative normalized = current better."""
    raw = current["mean_pace_s_per_km"] - earlier["mean_pace_s_per_km"]
    conditions = -(current["conditions_cost_s_per_km"] - earlier["conditions_cost_s_per_km"])
    curve = current["curve_shift_s_per_km"] - earlier["curve_shift_s_per_km"]
    return {
        "session_id": earlier["session_id"],
        "date": earlier["date"],
        "period": earlier["period"],
        "workout_type": earlier["workout_type"],
        "surface_group": earlier["surface_group"],
        "mean_pace_s_per_km": earlier["mean_pace_s_per_km"],
        "mean_lactate": earlier["mean_lactate"],
        "distance": round(distance(current, earlier), 2),
        "raw_pace_diff": round(raw, 1),
        "conditions_correction": round(conditions, 1),
        "curve_correction": round(curve, 1),
        "normalized_diff": round(
            current["normalized_pace"][reference_lactate] - earlier["normalized_pace"][reference_lactate], 1
        ),
    }


def find_similar(sessions: list[dict], reference_lactate: float) -> dict[str, list[dict]]:
    """For every session: its most similar *earlier* sessions, closest first."""
    ref = str(reference_lactate)
    ordered = sorted(sessions, key=lambda s: (s["date"], s["period"] or ""))
    result: dict[str, list[dict]] = {}
    for i, current in enumerate(ordered):
        earlier = ordered[:i]
        ranked = sorted(earlier, key=lambda e: distance(current, e))[:N_SIMILAR]
        result[current["session_id"]] = [compare(current, e, ref) for e in ranked]
    return result
