"""Read the source Excel file into a normalized, canonical DataFrame.

This is the only module allowed to read the input file. Everything downstream
works with the canonical column names defined in config.COLUMN_ALIASES.
"""
from __future__ import annotations

import datetime as dt
import re
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

from .config import COLUMN_ALIASES

REQUIRED_COLUMNS = ["date", "rep_distance_m", "rep_time_s", "lactate_mmol"]

# Preferred sheet names for the raw per-rep data table, tried in order before
# falling back to the first sheet. Handles workbooks that put a dashboard /
# pivot view (e.g. "Analysis") ahead of the raw data.
PREFERRED_SHEET_NAMES = ["rawdata", "raw data", "raw_data", "data"]

CATEGORICAL_COLUMNS = ["surface", "rest_type", "indoor_outdoor", "shoes", "workout_type"]
NUMERIC_COLUMNS = ["temperature_c", "heart_rate", "humidity_pct", "incline_pct", "lactate_mmol", "rep_number"]

_TIME_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)$")
_DISTANCE_RE = re.compile(r"^([\d.]+)\s*(m|meter|meters|km|kilometer|kilometers|mi|mile|miles)?$", re.IGNORECASE)


def _normalize_header(header: object) -> str:
    return re.sub(r"\s+", " ", str(header).strip().lower())


def _strip_unit_suffix(header: str) -> str:
    """Strip a trailing parenthesized unit, e.g. 'Lactate (mmol/L)' -> 'Lactate'."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", header).strip()


def build_rename_map(columns) -> dict[str, str]:
    """Map raw source headers to canonical column names via config.COLUMN_ALIASES.

    Matches against the full normalized header first, then against the header
    with a trailing "(unit)" suffix stripped -- real headers commonly look
    like "Temp (C)" or "Lactate (mmol/L)".
    """
    normalized_lookup = {_normalize_header(c): c for c in columns}
    stripped_lookup = {_strip_unit_suffix(_normalize_header(c)): c for c in columns}
    rename: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalized_lookup:
                rename[normalized_lookup[alias]] = canonical
                break
            if alias in stripped_lookup:
                rename[stripped_lookup[alias]] = canonical
                break
    return rename


def _parse_time_to_seconds(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dt.time):
        return value.hour * 3600 + value.minute * 60 + value.second
    if isinstance(value, dt.timedelta):
        return value.total_seconds()
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    match = _TIME_RE.match(text)
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2))
        seconds = float(match.group(3))
        return hours * 3600 + minutes * 60 + seconds
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"Could not parse time/duration value '{value}'") from exc


def _parse_distance_to_meters(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    match = _DISTANCE_RE.match(text)
    if not match:
        raise ValueError(f"Could not parse distance value '{value}'")
    number = float(match.group(1))
    unit = (match.group(2) or "m").lower()
    if unit in ("m", "meter", "meters"):
        return number
    if unit in ("km", "kilometer", "kilometers"):
        return number * 1000.0
    if unit in ("mi", "mile", "miles"):
        return number * 1609.34
    raise ValueError(f"Unknown distance unit in '{value}'")


def _select_sheet(path: Path) -> str | int:
    workbook = pd.ExcelFile(path, engine="openpyxl")
    for name in workbook.sheet_names:
        if name.strip().lower() in PREFERRED_SHEET_NAMES:
            return name
    return 0


_GOOGLE_SHEET_RE = re.compile(r"https://docs\.google\.com/spreadsheets/d/([A-Za-z0-9_-]+)")


def google_sheet_export_url(source: str) -> str | None:
    """Return the xlsx export URL for a Google Sheets link, or None if it isn't one.

    Exporting the whole workbook as xlsx (rather than one tab as CSV) keeps the
    exact same sheet-selection and parsing path as a local Excel file.
    """
    match = _GOOGLE_SHEET_RE.match(source.strip())
    if not match:
        return None
    return f"https://docs.google.com/spreadsheets/d/{match.group(1)}/export?format=xlsx"


def resolve_source(source: str | Path) -> Path:
    """Turn a local path or a Google Sheets link into a local xlsx path.

    A Google Sheet must be shared as "anyone with the link can view"; a
    private sheet makes Google return a login page instead of the file.
    """
    export_url = google_sheet_export_url(str(source))
    if export_url is None:
        return Path(source)
    with urllib.request.urlopen(export_url, timeout=60) as response:
        content_type = response.headers.get("Content-Type", "")
        payload = response.read()
    if "spreadsheetml" not in content_type:
        raise ValueError(
            "Google did not return a spreadsheet for this link (got "
            f"'{content_type}'). Make sure the sheet is shared as 'anyone with the link can view'."
        )
    target = Path(tempfile.gettempdir()) / "lactate-analysis-google-sheet.xlsx"
    target.write_bytes(payload)
    return target


def load(path: str | Path) -> pd.DataFrame:
    """Load and normalize the source (Excel path or Google Sheets link) into the canonical schema."""
    path = resolve_source(path)
    sheet = _select_sheet(path)
    raw = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    # +2: pandas 0-index -> 1-index, plus the header row, for user-facing messages.
    # Computed before any row is dropped so it reflects the true Excel row number.
    raw["source_row"] = raw.index + 2
    raw = raw.dropna(how="all").dropna(axis=1, how="all")

    rename = build_rename_map(raw.columns)
    df = raw.rename(columns=rename).reset_index(drop=True)

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"Could not find required column(s) {missing_required} in {path.name}. "
            f"Columns found after normalization: {list(df.columns)}. "
            "Update COLUMN_ALIASES in config.py if your header text differs."
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Spreadsheets commonly have trailing "blank" rows that survive
    # dropna(how="all") only because a filled-down formula column (e.g. a
    # computed "Month" or "Session Key") left a stray value there. A row with
    # no date can never be a real measurement, so drop those here rather than
    # letting NaT propagate into session grouping downstream.
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    if "session_period" in df.columns:
        df["session_period"] = df["session_period"].astype(str).str.strip().str.upper()
        df.loc[~df["session_period"].isin(["AM", "PM"]), "session_period"] = "AM"
    else:
        df["session_period"] = "AM"

    df["rep_distance_m"] = df["rep_distance_m"].apply(_parse_distance_to_meters)
    df["rep_time_s"] = df["rep_time_s"].apply(_parse_time_to_seconds)

    if "rest_duration_s" in df.columns:
        df["rest_duration_s"] = df["rest_duration_s"].apply(_parse_time_to_seconds)
    else:
        df["rest_duration_s"] = pd.NA

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.NA

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin(["nan", "None", ""]), col] = pd.NA
        else:
            df[col] = pd.NA

    return df
