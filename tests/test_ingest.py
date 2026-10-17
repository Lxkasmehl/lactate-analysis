from __future__ import annotations

import pandas as pd
import pytest

from lactate_analysis import ingest


def test_load_normalizes_headers_and_parses_units(raw_excel_path):
    df = ingest.load(raw_excel_path)

    assert "date" in df.columns
    assert "rep_distance_m" in df.columns
    assert df["rep_distance_m"].iloc[0] == 1609
    assert abs(df["rep_time_s"].iloc[0] - 390) < 1e-6  # 6:30 -> 390s
    assert df["session_period"].iloc[0] == "AM"
    assert df["lactate_mmol"].iloc[0] == 2.1
    assert df["rest_duration_s"].iloc[0] == 90.0


def test_load_missing_required_column_raises(tmp_path):
    bad = pd.DataFrame({"Date": ["2025-01-01"], "Foo": [1]})
    path = tmp_path / "bad.xlsx"
    bad.to_excel(path, index=False, engine="openpyxl")

    with pytest.raises(ValueError, match="required column"):
        ingest.load(path)


def test_google_sheet_link_maps_to_xlsx_export():
    url = "https://docs.google.com/spreadsheets/d/abc_DEF-123/edit?usp=sharing"
    assert (
        ingest.google_sheet_export_url(url)
        == "https://docs.google.com/spreadsheets/d/abc_DEF-123/export?format=xlsx"
    )
    assert ingest.google_sheet_export_url("data/lactate.xlsx") is None


def test_cycles_tab_in_workbook(tmp_path):
    from lactate_analysis import cycles

    path = tmp_path / "log.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"Date": ["2025-01-01"]}).to_excel(writer, sheet_name="RawData", index=False)
        pd.DataFrame(
            {"Label": ["Spring", "Fall"], "Start": ["2025-08-01", "2025-01-01"], "End": [None, "2025-06-30"]}
        ).to_excel(writer, sheet_name="Cycles", index=False)
    defs = cycles.load_from_workbook(path)
    assert [c["label"] for c in defs] == ["Fall", "Spring"]
    assert defs[1]["end_date"] == pd.Timestamp.max


def test_workbook_without_cycles_tab_returns_none(raw_excel_path):
    from lactate_analysis import cycles

    assert cycles.load_from_workbook(raw_excel_path) is None
