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
log(lactate) = log(2.0) + slope * (effective_pace - fitness[week]) + day_form[session] + noise
effective_pace = pace - Σ cost_k * condition_k
```

- **`fitness[week]`**: pace (s/km) at exactly 2.0 mmol/L under standard conditions,
  as a weekly random walk. Every session informs its neighbours; gaps just widen the
  uncertainty. No manual cycle boundaries are needed to see the trend.
- **`slope`**: your own lactate curve (≈2% more lactate per s/km faster), learned
  mainly from pace changes within sessions and between AM/PM sessions on the same
  day. It converts any (pace, lactate) pair into the pace at 2.0 or 3.0 mmol/L.
- **`cost_k`**: each condition as a pace cost in s/km (treadmill, road, offroad,
  incline, heat, rep number, rep duration, rest, jog rest, second session of the
  day). Each has a **literature-based prior**, so sparse factors still get a sensible,
  honestly uncertain estimate instead of being dropped or overfit. The report shows
  prior vs. estimate, so you see where your data taught the model something.
- **`day_form`**: how a single session deviated from the trend (sleep, fatigue,
  measurement noise), kept separate so it doesn't masquerade as a fitness change.

Heart rate is deliberately not a factor: it's a consequence of intensity, not a
condition. 4.0 mmol/L isn't reported because training data almost never reaches it.

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

`output/report.json` holds the same results as structured data, the hand-off
contract for a later web frontend or LLM explanation layer.

## Tests

```
.venv\Scripts\pytest
```

The model tests use synthetic data with a known fitness improvement and treadmill
effect, and check that the model recovers both.

## Known limitations

- One lactate-curve slope for the whole period: if the curve's *shape* changes with
  fitness (not just its position), that's hidden.
- A large treadmill effect can partly be belt-speed calibration; the model can't tell
  the two apart.
- Missing temperatures are imputed (treadmill: 18°C, outdoor: same-month average of
  recorded temperatures).
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
  report_data.py    status, cycle comparison, JSON summary
  report_html.py    Plotly charts + Jinja template -> HTML
  main.py           CLI entry point
```
