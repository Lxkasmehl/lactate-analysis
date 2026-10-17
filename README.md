# Lactate Analysis (Phase 1)

Local, no-server, no-LLM statistics engine that turns a lactate-testing training log
(Excel) into a fixed-structure HTML report: pace at fixed lactate concentrations
(2.0/3.0/4.0 mmol/L) per training cycle -- broken down by workout type via partial
pooling, since a "Continuous Threshold" run and a "Fartlek" aren't directly comparable
even at the same lactate reading -- plus a mechanically-generated evidence/caveats
section.

These numbers come from a **Bayesian hierarchical model**: `log(lactate) ~ pace +
covariates`, fit per cycle with `bambi`/PyMC. Two design choices came directly out of
real problems found while testing this against actual training data, not from
theory alone:

- **log(lactate), not lactate, as a linear function of pace.** Blood lactate rises
  *exponentially* with intensity in graded exercise tests (well established in exercise
  physiology) -- `log(lactate) ~ pace` is exactly that exponential curve, fit as an
  ordinary linear regression. A plain linear `lactate ~ pace` model, once inverted to
  solve for "pace at 4.0 mmol/L", extrapolated to a physiologically absurd 4:14/mile for
  a recreational runner's own data.
- **Partial pooling for `session_id` and `workout_type`** (varying/hierarchical
  intercepts) instead of splitting the data into disjoint per-category averages. A
  workout type backed by only 1-2 sessions in a cycle still gets an estimate -- shrunk
  toward, not absent from, what that type looks like across the whole dataset -- rather
  than either being silently too sparse to report, or an earlier version's naive
  cross-session average, which produced *physiologically backwards* results (a faster
  estimated pace at 2.0 mmol/L than at 4.0 mmol/L) because different lactate targets get
  reached by different, non-overlapping sessions with their own baseline-speed bias.
  Cycle itself stays a fixed effect: with only a handful of training cycles, there
  usually aren't enough of them for hierarchical pooling to usefully estimate
  cycle-to-cycle variance.

Inverting the model to solve for "pace at a fixed lactate" is done per posterior draw
(not a linearized delta-method approximation), which is mathematically guaranteed
monotonic (a higher lactate target always implies a faster pace) and can't produce a
negative pace. Two things are still surfaced explicitly rather than hidden:
a cycle whose own fitted slope has the wrong sign (`unexpected_relationship`), and a
point estimate that's implausible for *this athlete's own observed pace range*
(`implausible`, dropped from the number/chart) versus a merely *very uncertain* one
whose point estimate is fine but whose interval reaches into implausible territory
(`wide_uncertainty`, kept and flagged rather than hidden).

This is Phase 1 of a larger project. Later phases (not part of this code) will add an
LLM layer for relevant-session selection/interpretation, and eventually a hosted web
app. This phase's modules are plain functions (DataFrame/dict in, DataFrame/dict out)
so those phases can build on top without rework -- see `output/report.json` for the
structured hand-off contract.

## Setup

```
cd C:\Users\LukasMehl\Documents\lactate-analysis
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
```

1. Put your Google Sheets link into `data/source.txt` (gitignored). The sheet must be
   shared as "anyone with the link can view"; it is downloaded fresh on every run.
   Alternatively pass `--input` with a local Excel file or another Sheets link.
2. Edit `data/cycles.yaml` with the real start/end dates of your training cycles.
3. Run:

```
.venv\Scripts\lactate-report
```

Open `output/report.html` in a browser. The same run also writes
`output/report.json` (the structured summary used to build the HTML).

Useful flags:
- `--reference-pace-min-per-mile 6.75` -- pin the comparison pace instead of using the
  dataset median.
- `--offline` -- inline the Plotly JS so the report works without internet access
  (bigger file, good for sending to your trainer).
- `--min-covariate-fill-rate 0.6` -- fill-rate threshold below which a numeric
  covariate (temperature, HR, humidity, incline, rest duration, rep number, prior
  lactate) is dropped from the model instead of imputed.

## Running the tests

```
.venv\Scripts\pytest
```

## Known limitations (Phase 1)

1. Small N (~130 rows / ~20-30 sessions per cycle) means genuinely wide posterior
   uncertainty is expected and correct, not a bug -- `wide_uncertainty` and
   `low_confidence` flag this rather than hiding it. No modeling approach (Bayesian,
   classical, or an LLM handed the same rows) can manufacture more precision than the
   data contains.
2. Hierarchical partial pooling only helps where there are enough *groups* to estimate
   between-group variance from (session_id: ~20-30; workout_type: ~5+). With only a
   handful of training cycles, cycle-to-cycle variance itself would be poorly estimated
   by pooling, so cycle stays a fixed effect -- comparisons *across* cycles don't get
   the same partial-pooling benefit that workout types do.
3. `pace_deviation_pct` ("was this rep too fast") is a heuristic proxy from existing
   columns, not a true target-pace comparison -- there's no target-pace column in the
   source data yet.
4. Training-cycle boundaries in `data/cycles.yaml` are manual, not detected -- a wrong
   boundary silently shifts which reps count toward which cycle. The report shows the
   boundary dates used so this is checkable at a glance.
5. Shoes/humidity/incline/HR will likely auto-exclude from the model due to
   missingness -- the underlying questions stay unanswerable until that data exists. A
   literature-backed informative prior (e.g. carbon-plated shoes' ~2-3% metabolic-cost
   effect) could let a sparsely-observed covariate like `shoes` still contribute a
   sensible, appropriately-uncertain estimate -- not built yet, since current fill rate
   (~5%) is too low for it to matter either way; worth revisiting once shoe logging
   improves.
6. Surface/indoor-outdoor are included as plain covariates, not interacted with pace --
   a surface-mix shift between cycles could partly masquerade as a fitness change. Per-
   cycle surface mix is shown in the caveats section as a cheap sanity check.
7. The default report needs internet access for Plotly's CDN JS; use `--offline` for a
   fully self-contained file.
8. Workout-type breakdowns use partial pooling: a (cycle, workout type) combo borrows
   the cycle's fitted slope and that workout type's fitted intercept offset, both
   estimated from the *entire* dataset. This means even a combo with 1-2 sessions gets
   an estimate -- grounded in other cycles' data for that type, not invented -- rather
   than being silently too sparse to report. `n_sessions` in that breakdown is how many
   sessions of that exact type were actually observed in that cycle, so you can judge
   how much of the number is "this cycle" vs. "borrowed from elsewhere."
9. A cycle with too little/too noisy data can still produce a technically-computed but
   physiologically nonsensical *point estimate* (e.g. faster than this athlete has ever
   actually run). This is flagged `implausible` and omitted from the numbers/chart
   entirely. A point estimate that's fine but whose credible interval reaches into
   implausible territory is a different, more honest situation (real uncertainty, not a
   wrong answer) -- it's kept, flagged `wide_uncertainty`, and its displayed interval is
   capped at the plausible range rather than printing something like "-868 min/mi".
10. MCMC sampling takes tens of seconds per report (vs. instant for a closed-form fit);
    `PYTENSOR_FLAGS=cxx=` is set programmatically in `model.py` to force PyTensor's
    pure-Python backend, since this environment's C++ toolchain couldn't compile
    PyTensor's linker extension -- slower, but needs no working compiler.

## Project layout

```
data/           cycles.yaml, plus gitignored source.txt (Sheets link) / local Excel files
output/         generated report.html + report.json (gitignored)
src/lactate_analysis/
  ingest.py         Excel -> normalized DataFrame (only module that reads the input)
  sessions.py       group rows into sessions (date + AM/PM + contiguous reps)
  cycles.py         attach cycle_label from cycles.yaml by date range
  features.py       derived columns (pace, cumulative load, lags, pace-deviation proxy)
  validation.py     missing-data checks -> caveats
  model.py          Bayesian hierarchical log-linear model (bambi/PyMC), falling back
                    to cluster-robust OLS on log(lactate) if sampling doesn't converge;
                    per-cycle lactate-at-pace AND pace-at-lactate predictions (the
                    model-based, monotonic numbers)
  pace_at_lactate.py  raw per-session pace-at-lactate points, for the "trend over time"
                    scatter chart only -- NOT used for per-cycle/workout-type averages
                    (see module docstring for why that averaging is unsound)
  report_data.py    assembles the structured JSON summary (the hand-off contract)
  report_html.py    renders the JSON summary into a self-contained HTML file
  main.py           CLI entry point
tests/          pytest suite against synthetic fixtures
```
