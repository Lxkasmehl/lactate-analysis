"""Attach manually-defined training-cycle labels to rows by date range.

Cycles come from a "Cycles" tab in the source spreadsheet (columns: Label,
Start, End; an empty End means the cycle is still running), so they can be
edited next to the training log without touching code. data/cycles.yaml is
the fallback when the workbook has no such tab. Cycles only drive the
shading and the per-cycle table; the fitness trend itself doesn't need them.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .config import UNLABELED_CYCLE_LABEL

CYCLES_SHEET_NAMES = ["cycles", "zyklen"]
_HEADER_ALIASES = {
    "label": ["label", "name", "cycle", "zyklus"],
    "start_date": ["start", "start date", "startdatum", "beginn", "from", "von"],
    "end_date": ["end", "end date", "enddatum", "ende", "to", "bis"],
}


def load_from_workbook(path: str | Path) -> list[dict] | None:
    """Cycle definitions from the workbook's "Cycles" tab, or None if absent."""
    workbook = pd.ExcelFile(path, engine="openpyxl")
    sheet = next((n for n in workbook.sheet_names if n.strip().lower() in CYCLES_SHEET_NAMES), None)
    if sheet is None:
        return None
    raw = workbook.parse(sheet).dropna(how="all")
    columns = {}
    for canonical, aliases in _HEADER_ALIASES.items():
        match = next((c for c in raw.columns if str(c).strip().lower() in aliases), None)
        if match is None:
            raise ValueError(f'The "{sheet}" tab needs a column named one of: {", ".join(aliases)}')
        columns[canonical] = match

    parsed = []
    for i, row in enumerate(raw.to_dict(orient="records")):
        label = row[columns["label"]]
        start = pd.to_datetime(row[columns["start_date"]], errors="coerce")
        end = pd.to_datetime(row[columns["end_date"]], errors="coerce")
        if pd.isna(label) or pd.isna(start):
            continue
        parsed.append(
            {
                "id": f"cycle_{i + 1}",
                "label": str(label).strip(),
                "start_date": start,
                "end_date": end if pd.notna(end) else pd.Timestamp.max,
            }
        )
    if not parsed:
        raise ValueError(f'The "{sheet}" tab has no cycles with a label and a start date.')
    return sorted(parsed, key=lambda c: c["start_date"])


def load_cycle_definitions(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cycle definition file not found at {path}. Fill in your real training-cycle "
            "date ranges in data/cycles.yaml (a template is included)."
        )
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cycles = raw.get("cycles", [])
    if not cycles:
        raise ValueError(f"No cycles defined in {path}")

    parsed = []
    for c in cycles:
        parsed.append(
            {
                "id": c["id"],
                "label": c.get("label", c["id"]),
                "start_date": pd.Timestamp(c["start_date"]),
                "end_date": pd.Timestamp(c["end_date"]) if c.get("end_date") else pd.Timestamp.max,
            }
        )
    return sorted(parsed, key=lambda c: c["start_date"])


def attach_cycle_labels(df: pd.DataFrame, cycle_defs: list[dict]) -> pd.DataFrame:
    df = df.copy()
    df["cycle_id"] = "unlabeled"
    df["cycle_label"] = UNLABELED_CYCLE_LABEL
    for c in cycle_defs:
        mask = (df["date"] >= c["start_date"]) & (df["date"] <= c["end_date"])
        df.loc[mask, "cycle_id"] = c["id"]
        df.loc[mask, "cycle_label"] = c["label"]
    return df
