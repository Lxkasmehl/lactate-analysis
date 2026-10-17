"""CLI entry point: Excel in, fixed-structure HTML + JSON report out."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config, cycles, features, ingest, model, pace_at_lactate, report_data, report_html, sessions, validation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze lactate training data and generate a fixed-structure HTML report."
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Source Excel file or Google Sheets link. Defaults to the link stored in "
            f"{config.DEFAULT_SOURCE_FILE}."
        ),
    )
    parser.add_argument("--cycles", default=config.DEFAULT_CYCLES_PATH, help="Path to cycles.yaml")
    parser.add_argument("--output-html", default=config.DEFAULT_OUTPUT_HTML)
    parser.add_argument("--output-json", default=config.DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--reference-pace-min-per-mile",
        type=float,
        default=None,
        help="Reference pace (min/mi) for the cycle comparison. Defaults to the dataset median pace.",
    )
    parser.add_argument("--min-covariate-fill-rate", type=float, default=config.MIN_COVARIATE_FILL_RATE)
    parser.add_argument("--offline", action="store_true", help="Inline Plotly JS for fully offline viewing")
    return parser


def resolve_input(cli_input: str | None) -> str:
    if cli_input:
        return cli_input
    source_file = Path(config.DEFAULT_SOURCE_FILE)
    if not source_file.exists():
        raise ValueError(
            f"No --input given and {source_file} does not exist. Pass --input or put your "
            f"Google Sheets link into {source_file}."
        )
    return source_file.read_text(encoding="utf-8").strip()


def run(args: argparse.Namespace) -> dict:
    raw_df = ingest.load(resolve_input(args.input))
    sessioned_df = sessions.assign_sessions(raw_df)
    cycle_defs = cycles.load_cycle_definitions(args.cycles)
    cycled_df = cycles.attach_cycle_labels(sessioned_df, cycle_defs)
    feature_df = features.build_feature_frame(cycled_df)

    validation_report = validation.build_validation_report(feature_df, args.min_covariate_fill_rate)

    reference_pace_s_per_km = None
    if args.reference_pace_min_per_mile:
        reference_pace_s_per_km = args.reference_pace_min_per_mile * 60 / config.MILE_TO_KM

    analysis = model.run_model_analysis(feature_df, validation_report, reference_pace_s_per_km)
    reference_pace_min_per_mi = analysis["reference_pace_s_per_km"] * config.MILE_TO_KM / 60

    trend_points = pace_at_lactate.compute_trend_points(feature_df, config.TARGET_LACTATE_LEVELS)

    summary = report_data.build_summary(
        feature_df, validation_report, analysis, reference_pace_min_per_mi, trend_points
    )

    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_html.write_report(summary, args.output_html, offline=args.offline)

    return summary


def cli(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        summary = run(args)
    except Exception as exc:  # noqa: BLE001 - CLI boundary, report and exit non-zero
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Report written to {args.output_html}")
    print(f"JSON summary written to {args.output_json}")
    print(f"Headline: {summary['conclusion']['headline']}")
    return 0


if __name__ == "__main__":
    sys.exit(cli())
