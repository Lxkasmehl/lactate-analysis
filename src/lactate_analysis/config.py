"""Configuration constants for the lactate analysis pipeline."""
from __future__ import annotations

# Canonical column name -> list of possible source header variants, matched
# case-insensitively after stripping/collapsing whitespace. Extend this if the
# real Excel file uses header text not covered here.
COLUMN_ALIASES: dict[str, list[str]] = {
    "date": ["date", "datum"],
    "session_period": ["am/pm", "ampm", "period", "session period", "session", "tageszeit"],
    "rep_number": ["rep", "rep number", "rep #", "rep no", "wiederholung", "wdh"],
    "rep_distance_m": ["rep distance", "distance", "distanz", "rep distance (m)", "rep dist"],
    "rep_time_s": ["rep time", "time", "zeit", "rep time (s)"],
    "surface": ["surface", "untergrund"],
    "rest_type": ["rest type", "pausenart", "pause typ"],
    "rest_duration_s": ["rest duration", "rest", "pause", "pausendauer"],
    "temperature_c": ["temperature", "temp", "temperatur"],
    "indoor_outdoor": ["indoor/outdoor", "indoor outdoor", "in/out"],
    "heart_rate": ["heart rate", "hr", "hr avg", "puls"],
    "shoes": ["shoes", "schuhe"],
    "humidity_pct": ["humidity", "luftfeuchtigkeit"],
    "incline_pct": ["incline", "steigung"],
    "lactate_mmol": ["lactate", "laktat", "lac"],
    "workout_type": ["workout type", "workout"],
}

# Default cycle_label assigned to rows that fall outside every range defined
# in cycles.yaml. These rows are excluded from cycle comparisons but kept in
# the descriptive fitness-trend view.
UNLABELED_CYCLE_LABEL = "Unlabeled"

# Environment variable (usually set in the gitignored .env) holding the Google
# Sheets link or Excel path used when --input is omitted. An env var rather
# than a file so the same setting works unchanged on a hosted backend later.
SOURCE_ENV_VAR = "LACTATE_SOURCE"

DEFAULT_CYCLES_PATH = "data/cycles.yaml"
DEFAULT_OUTPUT_HTML = "output/report.html"
DEFAULT_OUTPUT_JSON = "output/report.json"

MILE_TO_KM = 1.60934

# Lactate levels (mmol/L) the fitness trend is reported at: the double-
# threshold targets actually trained at (AM ~2.0, PM ~3.0). 4.0 is left out on
# purpose -- training data almost never reaches it, so any number there would
# be extrapolation rather than measurement.
TARGET_LACTATE_LEVELS = [2.0, 3.0]

# Workout types that represent an all-out/maximal effort (a race), not a
# repeatable submaximal training structure. A maximal effort is a genuinely
# different physiological regime (different lactate kinetics, different
# measurement timing relative to the effort) -- these rows are excluded from
# the fitness model, not because the values are wrong, but because they don't
# belong on the same submaximal training curve. This is a workout_type-based
# exclusion, not a lactate-value cutoff: a hard,
# fast *training* session (any other workout_type) still belongs in the fit.
MAX_EFFORT_WORKOUT_TYPES = ["Race"]
