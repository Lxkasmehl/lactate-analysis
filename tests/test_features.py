from __future__ import annotations

from lactate_analysis import cycles, features, ingest, sessions


def test_build_feature_frame(raw_excel_path, cycles_yaml_path):
    df = ingest.load(raw_excel_path)
    df = sessions.assign_sessions(df)
    cycle_defs = cycles.load_cycle_definitions(cycles_yaml_path)
    df = cycles.attach_cycle_labels(df, cycle_defs)
    out = features.build_feature_frame(df)

    assert "pace_s_per_km" in out.columns

    first_session_id = out["session_id"].iloc[0]
    first_session = out[out["session_id"] == first_session_id].reset_index(drop=True)
    assert first_session["prior_lactate"].isna().iloc[0]
    assert first_session["prior_lactate"].iloc[1] == first_session["lactate_mmol"].iloc[0]

    # rep 3 is the fastest in the session, so it should have the largest (most
    # negative-pace / fastest) pace_deviation_pct relative to earlier reps.
    assert first_session["pace_deviation_pct"].iloc[-1] <= 0

    assert (out["cycle_label"] != "Unlabeled").all()
