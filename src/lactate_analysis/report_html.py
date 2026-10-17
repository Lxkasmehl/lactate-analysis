"""Render the structured JSON summary into a self-contained HTML report."""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATE_DIR = Path(__file__).parent / "templates"
_PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]


def _format_mmss(decimal_minutes: float | None) -> str:
    """4.75 -> "4:45", not the meaningless decimal-minutes reading it replaces."""
    if decimal_minutes is None:
        return "?"
    total_seconds = round(decimal_minutes * 60)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}"


def _lactate_curve_figure(summary: dict) -> go.Figure:
    fig = go.Figure()
    for i, (label, data) in enumerate(summary["lactate_curves"].items()):
        points = data["points"]
        if not points:
            continue
        fig.add_trace(
            go.Scatter(
                x=[p["pace_s_per_km"] for p in points],
                y=[p["lactate_mmol"] for p in points],
                mode="markers",
                name=label,
                marker=dict(color=_PALETTE[i % len(_PALETTE)]),
            )
        )
    fig.update_xaxes(title="Pace (s/km)", autorange="reversed")
    fig.update_yaxes(title="Lactate (mmol/L)")
    fig.update_layout(title="Lactate response by cycle", template="plotly_white")
    return fig


def _cycle_comparison_figure(summary: dict) -> go.Figure:
    cycles = [c for c in summary["cycle_comparison"]["cycles"] if not c.get("out_of_range")]
    fig = go.Figure()
    if cycles:
        fig.add_trace(
            go.Bar(
                x=[c["cycle_label"] for c in cycles],
                y=[c["predicted_lactate"] for c in cycles],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[c["ci"][1] - c["predicted_lactate"] for c in cycles],
                    arrayminus=[c["predicted_lactate"] - c["ci"][0] for c in cycles],
                ),
                text=[
                    f"n={c['n_sessions']}" + (" (low confidence)" if c.get("low_confidence") else "")
                    for c in cycles
                ],
                marker=dict(color=_PALETTE[0]),
            )
        )
    fig.update_yaxes(title="Predicted lactate at reference pace (mmol/L)")
    fig.update_layout(
        title=f"Cycle comparison @ {summary['cycle_comparison']['reference_pace_min_per_mi']} min/mi",
        template="plotly_white",
    )
    return fig


def _pace_at_lactate_by_cycle_figure(summary: dict) -> go.Figure:
    """Per cycle: model-based pace at fixed lactate concentrations (2/3/4 mmol/L).

    This is the standard lactate-step-test readout -- how fast you can run at
    a given metabolic stress level -- rather than the raw lactate value on
    its own, which says nothing about fitness without the pace it came at.
    Values come from inverting one fitted lactate-vs-pace line per cycle, so
    they are guaranteed monotonic (higher target -> faster pace), unlike a
    naive average across different sessions per target.
    """
    by_cycle = summary["pace_at_lactate"]["by_cycle"]
    cycle_order = summary["pace_at_lactate"]["cycle_order"]
    targets = ["2.0", "3.0", "4.0"]

    fig = go.Figure()
    for i, target in enumerate(targets):
        ys, errs_hi, errs_lo, texts = [], [], [], []
        for c in cycle_order:
            entry = by_cycle.get(c, {}).get(target, {})
            # Implausible extrapolations (e.g. a negative or faster-than-world-
            # record pace from a noisy/limited-data cycle) are left off the
            # chart entirely rather than distorting its scale -- they're still
            # shown as text in the table below with an explicit caveat.
            pace = entry.get("pace_min_per_mi") if not entry.get("implausible") else None
            ys.append(pace)
            if pace is not None and entry.get("ci_min_per_mi"):
                lo, hi = entry["ci_min_per_mi"]
                errs_hi.append(hi - pace)
                errs_lo.append(pace - lo)
            else:
                errs_hi.append(0)
                errs_lo.append(0)
            if entry.get("out_of_range"):
                texts.append("outside this cycle's observed lactate range")
            elif entry.get("unsolvable"):
                texts.append("flat lactate-pace relationship, cannot solve")
            elif entry.get("implausible"):
                texts.append("implausible extrapolation, omitted -- see table below")
            elif entry.get("unexpected_relationship"):
                texts.append(f"n={entry.get('n_sessions')} sessions (unexpected slope sign -- treat with caution)")
            else:
                texts.append(f"n={entry.get('n_sessions')} sessions")
        fig.add_trace(
            go.Scatter(
                x=cycle_order,
                y=ys,
                mode="markers+lines",
                name=f"{target} mmol/L",
                marker=dict(color=_PALETTE[i % len(_PALETTE)]),
                error_y=dict(type="data", symmetric=False, array=errs_hi, arrayminus=errs_lo),
                text=texts,
                hovertemplate="%{x}<br>%{y} min/mi<br>%{text}<extra></extra>",
            )
        )
    fig.update_yaxes(title="Pace (min/mi) -- lower is faster", autorange="reversed")
    fig.update_layout(title="Pace at fixed lactate concentrations, per cycle (model-based)", template="plotly_white")
    return fig


def _pace_at_lactate_trend_figure(summary: dict) -> go.Figure:
    """Per session over time: pace at 2/3/4 mmol/L -- the actual fitness signal
    (lactate alone says nothing; running the same lactate faster does)."""
    points = summary["pace_at_lactate"]["trend_points"]
    fig = go.Figure()
    for i, target in enumerate([2.0, 3.0, 4.0]):
        key = f"pace_at_{target}"
        relevant = [p for p in points if p.get(key) is not None]
        fig.add_trace(
            go.Scatter(
                x=[p["date"] for p in relevant],
                y=[p[key] for p in relevant],
                mode="markers",
                name=f"{target} mmol/L",
                marker=dict(color=_PALETTE[i % len(_PALETTE)], size=6),
                text=[p["workout_type"] for p in relevant],
                hovertemplate="%{x}<br>%{y} min/mi<br>%{text}<extra></extra>",
            )
        )
    fig.update_yaxes(title="Pace (min/mi) -- lower is faster", autorange="reversed")
    fig.update_xaxes(title="Date")
    fig.update_layout(title="Pace at fixed lactate over time (per session)", template="plotly_white")
    return fig


def render(summary: dict, offline: bool = False) -> str:
    include_plotlyjs = True if offline else "cdn"
    figs_html = {
        "lactate_curve": _lactate_curve_figure(summary).to_html(
            full_html=False, include_plotlyjs=include_plotlyjs
        ),
        "cycle_comparison": _cycle_comparison_figure(summary).to_html(full_html=False, include_plotlyjs=False),
        "pace_at_lactate_by_cycle": _pace_at_lactate_by_cycle_figure(summary).to_html(
            full_html=False, include_plotlyjs=False
        ),
        "pace_at_lactate_trend": _pace_at_lactate_trend_figure(summary).to_html(
            full_html=False, include_plotlyjs=False
        ),
    }
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=select_autoescape(["html"]))
    env.filters["mmss"] = _format_mmss
    template = env.get_template("report.html.j2")
    return template.render(summary=summary, figs=figs_html)


def write_report(summary: dict, output_path: str | Path, offline: bool = False) -> None:
    html = render(summary, offline=offline)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
