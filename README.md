# Lactate Analysis

Turns a lactate-guided training log (Google Sheet or Excel) into an HTML report that
answers: **am I getting fitter, and what makes a session harder or easier?**

## The idea

Training data is not a lab step test. Most sessions are double threshold (AM ~2.0,
PM ~3.0 mmol/L) where pace is *adjusted* to hit a lactate target, and every session
happens under different conditions (treadmill or track, heat, rep length, rest...).
Raw numbers, or a lactate ÷ speed ratio, aren't comparable across sessions:
because lactate rises exponentially with pace, running 5 s/km slower lowers lactate
by ~10% with no change in fitness at all.

The model (`fitness_model.py`, Bayesian, PyMC) normalizes every rep to a common
standard:

```
lactate = L0 + (2.0 - L0) * exp(c * (fitness[week] - effective_pace) + day_form[session] + drift[session] * work_done)
effective_pace = pace - Σ cost_k * condition_k
```

- **`fitness[week]`**: pace (s/km) at exactly 2.0 mmol/L under standard conditions,
  as a weekly random walk. Every session informs its neighbours; gaps just widen the
  uncertainty. No manual cycle boundaries are needed to see the trend.
- **`L0`, `c`**: your lactate curve, a resting baseline plus an exponential part. It
  converts any (pace, lactate) pair into the pace at 2.0 or 3.0 mmol/L. Because pace is
  steered by lactate, each workout structure is run at nearly the same pace, so the data
  alone can't separate curve steepness from the rep-length effect (leave-one-out fit is
  identical either way). `c` therefore has a fairly tight literature prior (2 → 3 mmol/L
  ≈ 8-12 s/km).
- **`cost_k`**: each condition as a pace cost in s/km (treadmill, road, offroad,
  incline, heat, rep number, rep duration, rest, jog rest, second session of the
  day). Each has a **literature-based prior**, so sparse factors still get a sensible,
  honestly uncertain estimate instead of being dropped or overfit. The report shows
  prior vs. estimate, so you see where your data taught the model something.
- **`day_form`, `drift`**: how a single session deviated from the trend (sleep, fatigue,
  measurement noise) and how fast its lactate drifted up, kept separate so neither
  masquerades as a fitness change.

The report turns this into target paces per workout structure (400 m / 30 s, 1000 m,
mile, 10 min). `heart_rate_model.py` adds the heart rate that goes with 2.0/3.0 mmol/L
per structure and over time, for sessions without a lactate meter. Heart rate is not
used to normalize lactate: it's a consequence of intensity, not a condition.
4.0 mmol/L isn't reported because training data almost never reaches it.

## Setup

```
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
```

1. Create `.env` (gitignored) with your sheet link. The sheet must be shared as
   "anyone with the link can view"; it's downloaded fresh on every run.
   ```
   LACTATE_SOURCE=https://docs.google.com/spreadsheets/d/<id>/edit
   ```
   Alternatively pass `--input` with a local Excel file or another Sheets link.
2. Edit `data/cycles.yaml` with your training cycles (used for shading and the
   per-cycle table, not for the trend itself).
3. Run `.venv\Scripts\lactate-report` (about a minute) and open `output/report.html`.
   `--offline` inlines Plotly so the file works without internet (e.g. to send to a coach).

The report also contains two interactive tools that run entirely in the browser on
posterior draws exported by the model (no server, no AI/API calls):

- **Workout planner**: type e.g. `3x10min/60s` or `4x1600m/60s + 4x400m/30s`, pick target
  lactate, surface (incl. treadmill incline), temperature, shoe, and get a per-rep pace,
  treadmill speed (km/h, mph) and heart-rate guide with 80% ranges.
- **Similar sessions**: for any session, the most similar earlier ones, with the
  normalized difference split into raw pace + conditions correction + lactate-curve
  correction, so you can see how much the verdict depends on the model.

`output/report.json` holds the same results as structured data, the hand-off
contract for a later web frontend or LLM explanation layer.

## Tests

```
.venv\Scripts\pytest
```

The model tests use synthetic data with a known fitness improvement and treadmill
effect, and check that the model recovers both.

## Known limitations

- One lactate-curve shape for the whole period (only its position moves with fitness):
  the data cannot yet identify even one steepness, so a time-varying one would be pure
  prior.
- A large treadmill effect can partly be belt-speed calibration; the model can't tell
  the two apart.
- Indoor sessions (treadmill, "Indoor ..." surfaces, track without a temperature) get a
  learned "indoors" offset instead of an assumed hall temperature. Missing outdoor
  temperatures are imputed (same-month average of recorded temperatures); the report
  lists the affected sessions.
- Shoes are compared per logged shoe against your typical shoe (partially pooled);
  reps without a logged shoe count as typical.
- "Rep #" is read as the rep's position in the whole workout (lactate after reps 3, 6,
  9, ...), which is how the sheet is filled in.
- Sampling runs chains sequentially with PyTensor's pure-Python backend
  (`PYTENSOR_FLAGS=cxx=`): parallel chains and the Numba backend hung on Windows here.

## Project layout

```
src/lactate_analysis/
  ingest.py         Sheet/Excel -> normalized DataFrame (only module that reads input)
  sessions.py       group rows into sessions (date + AM/PM)
  cycles.py         attach cycle labels from data/cycles.yaml
  features.py       derived columns (pace, second session of the day, ...)
  validation.py     column fill rates for the data-quality section
  fitness_model.py  the Bayesian normalization + fitness-trend model
  heart_rate_model.py  heart rate at 2.0/3.0 mmol/L per workout structure and over time
  report_data.py    status, cycle comparison, JSON summary
  similar_sessions.py  nearest earlier sessions + decomposed comparison
  report_html.py    Plotly charts + Jinja template -> HTML
  templates/interactive.js  in-browser workout planner + session comparison
  main.py           CLI entry point
```
