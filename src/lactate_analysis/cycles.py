"""Attach manually-defined training-cycle labels to rows by date range.

Cycle boundaries are defined in data/cycles.yaml (Lukas fills in the real
dates for his 3 training builds). A manual file is simpler and easier to
correct than automatic changepoint detection on ~130 noisy, unevenly-spaced
rows -- see the Phase 1 plan for the tradeoff discussion.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .config import UNLABELED_CYCLE_LABEL


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
