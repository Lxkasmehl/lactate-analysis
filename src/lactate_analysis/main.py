"""CLI entry point: spreadsheet in, HTML + JSON fitness report out."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from . import config, cycles, features, fitness_model, ingest, report_data, report_html, sessions, validation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze lactate training data and generate an HTML fitness report."
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Source Excel file or Google Sheets link. Defaults to the "
            f"{config.SOURCE_ENV_VAR} environment variable (or .env)."
        ),
    )
    parser.add_argument("--cycles", default=config.DEFAULT_CYCLES_PATH, help="Path to cycles.yaml")
    parser.add_argument("--output-html", default=config.DEFAULT_OUTPUT_HTML)
    parser.add_argument("--output-json", default=config.DEFAULT_OUTPUT_JSON)
    parser.add_argument("--offline", action="store_true", help="Inline Plotly JS for fully offline viewing")
    return parser


def resolve_input(cli_input: str | None) -> str:
    if cli_input:
        return cli_input
    load_dotenv(find_dotenv(usecwd=True))
    source = os.environ.get(config.SOURCE_ENV_VAR, "").strip()
    if not source:
        raise ValueError(
            f"No --input given and {config.SOURCE_ENV_VAR} is not set. Pass --input or add "
            f"{config.SOURCE_ENV_VAR}=<Google Sheets link> to .env."
        )
    return source


def run(args: argparse.Namespace) -> dict:
    raw_df = ingest.load(resolve_input(args.input))
    sessioned_df = sessions.assign_sessions(raw_df)
    cycle_defs = cycles.load_cycle_definitions(args.cycles)
    cycled_df = cycles.attach_cycle_labels(sessioned_df, cycle_defs)
    feature_df = features.build_feature_frame(cycled_df)

    validation_report = validation.build_validation_report(feature_df)
    fit = fitness_model.fit(feature_df)
    fitness_summary = fitness_model.summarize(fit, config.TARGET_LACTATE_LEVELS)
    summary = report_data.build_summary(feature_df, validation_report, fitness_summary, cycle_defs)

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
    print(f"Headline: {summary['headline']}")
    return 0


if __name__ == "__main__":
    sys.exit(cli())
