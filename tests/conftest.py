from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def raw_excel_path(tmp_path):
    rows = [
        ["2025-04-05", "AM", 1, 1609, "6:30", "Track", "jogging", "90", 15, "Outdoor", 2.1],
        ["2025-04-05", "AM", 2, 1609, "6:25", "Track", "jogging", "90", 15, "Outdoor", 3.0],
        ["2025-04-05", "AM", 3, 1609, "6:20", "Track", "jogging", "90", 15, "Outdoor", 4.2],
        ["2025-06-10", "AM", 1, 1609, "6:20", "Track", "jogging", "90", 18, "Outdoor", 2.0],
        ["2025-06-10", "AM", 2, 1609, "6:15", "Track", "jogging", "90", 18, "Outdoor", 2.9],
        ["2025-06-10", "AM", 3, 1609, "6:10", "Track", "jogging", "90", 18, "Outdoor", 4.0],
        ["2026-04-01", "AM", 1, 1609, "6:10", "Road", "standing", "60", 10, "Outdoor", 1.8],
        ["2026-04-01", "AM", 2, 1609, "6:05", "Road", "standing", "60", 10, "Outdoor", 2.5],
        ["2026-04-01", "AM", 3, 1609, "6:00", "Road", "standing", "60", 10, "Outdoor", 3.4],
    ]
    columns = [
        "Date",
        "AM/PM",
        "Rep",
        "Distance",
        "Time",
        "Surface",
        "Rest Type",
        "Rest",
        "Temperature",
        "Indoor/Outdoor",
        "Lactate",
    ]
    df = pd.DataFrame(rows, columns=columns)
    path = tmp_path / "lactate.xlsx"
    df.to_excel(path, index=False, engine="openpyxl")
    return path


@pytest.fixture
def cycles_yaml_path(tmp_path):
    content = """
cycles:
  - id: cycle_1
    label: "Cycle 1"
    start_date: 2025-01-01
    end_date: 2025-12-31
  - id: cycle_2
    label: "Cycle 2"
    start_date: 2026-01-01
    end_date: null
"""
    path = tmp_path / "cycles.yaml"
    path.write_text(content, encoding="utf-8")
    return path
