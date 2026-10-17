from __future__ import annotations

from lactate_analysis import ingest, sessions


def test_assign_sessions_groups_by_date_and_period(raw_excel_path):
    df = ingest.load(raw_excel_path)
    out = sessions.assign_sessions(df)

    assert out["session_id"].nunique() == 3
    assert set(out.groupby("session_id").size()) == {3}

    first_session_id = out["session_id"].iloc[0]
    first_session = out[out["session_id"] == first_session_id]
    assert list(first_session["rep_number"]) == [1, 2, 3]
    assert not first_session["rep_number_inferred"].any()
