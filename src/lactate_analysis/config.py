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

# A numeric covariate is only offered to the statistical model if its fill
# rate across all rows is at least this high; otherwise it is dropped rather
# than imputed. Overridable via --min-covariate-fill-rate.
MIN_COVARIATE_FILL_RATE = 0.6

# Below this many usable sessions, the Bayesian hierarchical fit is skipped
# (too little data for the sampler to do better than a simpler model) and the
# pipeline goes straight to cluster-robust OLS.
MIN_SESSIONS_FOR_BAYESIAN = 15

# A per-cycle prediction is flagged "low_confidence" (not hidden) below this
# many sessions in that cycle.
LOW_CONFIDENCE_MIN_SESSIONS = 5

# A *fixed*-effect categorical covariate level (e.g. surface="Treadmill")
# backed by fewer rows than this is merged into "Other" before modeling
# rather than getting its own unstable dummy coefficient. Does not apply to
# workout_type, which gets a real partially-pooled (varying) intercept
# instead of fixed dummies, so sparse levels are shrunk rather than merged.
MIN_CATEGORY_COUNT = 5

# A model-predicted pace-at-lactate is flagged "implausible" if it (or its
# credible/confidence interval) falls more than this margin outside *that
# cycle's own observed* pace range -- relative to what this specific athlete
# has actually demonstrated, not an absolute human-population bound (which
# is too loose to catch e.g. "4:14/mile" as wrong for a given recreational
# runner even though it's a realistic pace for some humans). The margin is
# max(fraction * observed range width, the floor below), so a cycle with a
# very narrow observed pace range still gets a sane minimum margin.
PLAUSIBLE_PACE_MARGIN_FRACTION = 0.25
PLAUSIBLE_PACE_MARGIN_MIN_S_PER_KM = 20.0

# Default cycle_label assigned to rows that fall outside every range defined
# in cycles.yaml. These rows are excluded from cycle comparisons but kept in
# the descriptive fitness-trend view.
UNLABELED_CYCLE_LABEL = "Unlabeled"

# Local, gitignored file holding the Google Sheets link (or an Excel path) used
# when --input is omitted, so the data source isn't committed to the repo.
DEFAULT_SOURCE_FILE = "data/source.txt"

DEFAULT_CYCLES_PATH = "data/cycles.yaml"
DEFAULT_OUTPUT_HTML = "output/report.html"
DEFAULT_OUTPUT_JSON = "output/report.json"

MILE_TO_KM = 1.60934

# Standard fixed lactate reference points (mmol/L) used to report "how fast
# can you run at this lactate concentration" per cycle -- the conventional
# way lactate step-test results are reported (aerobic ~2.0, threshold ~4.0).
TARGET_LACTATE_LEVELS = [2.0, 3.0, 4.0]

# Workout types that represent an all-out/maximal effort (a race), not a
# repeatable submaximal training structure. A maximal effort is a genuinely
# different physiological regime (different lactate kinetics, different
# measurement timing relative to the effort) -- these rows are excluded from
# the pace-at-lactate curve fit, not because the values are wrong, but
# because they don't belong on the same submaximal training curve. They stay
# in the raw data and are reported separately as an observed maximum. This is
# a workout_type-based exclusion, not a lactate-value cutoff: a hard,
# fast *training* session (any other workout_type) still belongs in the fit.
MAX_EFFORT_WORKOUT_TYPES = ["Race"]
