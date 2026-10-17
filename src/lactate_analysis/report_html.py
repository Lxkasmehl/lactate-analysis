"""Render the structured JSON summary into a self-contained HTML report."""
from __future__ import annotations

import math
from pathlib import Path

import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import MILE_TO_KM

TEMPLATE_DIR = Path(__file__).parent / "templates"

# Reference categorical palette slots 1-2 (validated CVD-safe as a pair) --
# one fixed color per target lactate across every chart.
TARGET_COLORS = {"2.0": "#2a78d6", "3.0": "#eb6834"}
NEUTRAL = "#8a8986"
PRIOR_COLOR = "#b5b4ae"
TEXT_SECONDARY = "#52514e"
CYCLE_SHADES = ["rgba(138,137,134,0.07)", "rgba(138,137,134,0.0)"]


def _mmss(seconds: float | None) -> str:
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "?"
    total = round(seconds)
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"


def format_pace(s_per_km: float | None) -> str:
    """195.0 -> "3:15/km (5:14/mi)"."""
    if s_per_km is None:
        return "?"
    return f"{_mmss(s_per_km)}/km ({_mmss(s_per_km * MILE_TO_KM)}/mi)"


def format_pace_short(s_per_km: float | None) -> str:
    return f"{_mmss(s_per_km)}/km" if s_per_km is not None else "?"


def format_signed_seconds(value: float | None) -> str:
    if value is None:
        return "?"
    if abs(value) < 0.05:
        return "±0.0 s/km"
    return f"{value:+.1f} s/km"


def format_certainty(change: dict) -> str:
    """Verdict plus how sure the model is of *that* direction."""
    if change["verdict"] == "faster":
        return f"faster, {change['prob_faster']:.0%} sure"
    if change["verdict"] == "slower":
        return f"slower, {1 - change['prob_faster']:.0%} sure"
    return f"no clear change ({change['prob_faster']:.0%} chance faster)"


def _pace_axis(fig: go.Figure, values: list[float]) -> None:
    """Pace y-axis: faster at the top, mm:ss tick labels every 5 s/km."""
    lo, hi = min(values), max(values)
    start, end = 5 * math.floor(lo / 5), 5 * math.ceil(hi / 5)
    ticks = list(range(start, end + 1, 5))
    fig.update_yaxes(
        autorange="reversed",
        tickvals=ticks,
        ticktext=[f"{_mmss(t)}/km" for t in ticks],
        gridcolor="#ecebe8",
        zeroline=False,
    )


def _nearest_target(mean_lactate: float, targets: list[str]) -> str:
    return min(targets, key=lambda t: abs(math.log(mean_lactate / float(t))))


def _fitness_trend_figure(summary: dict) -> go.Figure:
    fitness = summary["fitness"]
    targets = [str(t) for t in fitness["targets"]]
    fig = go.Figure()
    all_values: list[float] = []

    for cycle_idx, cycle in enumerate(summary["cycles"]):
        end = cycle["end_date"] or summary["data_summary"]["date_range"][1]
        fig.add_vrect(
            x0=cycle["start_date"], x1=end, fillcolor=CYCLE_SHADES[cycle_idx % 2], line_width=0,
            annotation_text=cycle["label"], annotation_position="top left",
            annotation_font=dict(size=11, color=TEXT_SECONDARY),
        )

    for target in targets:
        color = TARGET_COLORS.get(target, NEUTRAL)
        weeks = fitness["trend"][target]
        x = [w["week_start"] for w in weeks]
        all_values += [w["lo95"] for w in weeks] + [w["hi95"] for w in weeks]
        for lo_key, hi_key, opacity in (("lo95", "hi95", 0.12), ("lo80", "hi80", 0.2)):
            fig.add_trace(
                go.Scatter(
                    x=x + x[::-1],
                    y=[w[hi_key] for w in weeks] + [w[lo_key] for w in weeks][::-1],
                    fill="toself", fillcolor=_rgba(color, opacity), line=dict(width=0),
                    hoverinfo="skip", showlegend=False, legendgroup=target,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=x, y=[w["median"] for w in weeks], mode="lines",
                line=dict(color=color, width=2), name=f"Pace at {target} mmol/L (trend)", legendgroup=target,
                customdata=[[format_pace(w["median"]), format_pace_short(w["lo95"]), format_pace_short(w["hi95"])] for w in weeks],
                hovertemplate="Week of %{x}<br>%{customdata[0]}<br>95%: %{customdata[1]} – %{customdata[2]}<extra>"
                + f"{target} mmol/L</extra>",
            )
        )

        sessions = [s for s in fitness["sessions"] if _nearest_target(s["mean_lactate"], targets) == target]
        fig.add_trace(
            go.Scatter(
                x=[s["date"] for s in sessions],
                y=[s["normalized_pace"][target] for s in sessions],
                mode="markers", name=f"Sessions near {target} mmol/L (normalized)", legendgroup=target,
                marker=dict(color=color, size=8, line=dict(color="#fcfcfb", width=2)),
                customdata=[
                    [s["workout_type"] or "", format_pace_short(s["mean_pace_s_per_km"]), s["mean_lactate"],
                     format_pace(s["normalized_pace"][target]), s["form_s_per_km"]]
                    for s in sessions
                ],
                hovertemplate=(
                    "%{x} – %{customdata[0]}<br>Ran %{customdata[1]} at %{customdata[2]} mmol/L"
                    "<br>Normalized: %{customdata[3]}<br>Day form vs. trend: %{customdata[4]:+.1f} s/km"
                    f"<extra>{target} mmol/L</extra>"
                ),
            )
        )
        all_values += [s["normalized_pace"][target] for s in sessions]

    _pace_axis(fig, all_values)
    fig.update_xaxes(gridcolor="#ecebe8")
    fig.update_layout(
        template="plotly_white", height=480, margin=dict(l=70, r=20, t=30, b=40),
        legend=dict(orientation="h", y=-0.12), hovermode="closest",
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
    )
    return fig


def _factor_figure(summary: dict) -> go.Figure:
    factors = list(reversed(summary["fitness"]["factors"]))
    labels = [f"{f['label']}<br><span style='font-size:11px;color:{TEXT_SECONDARY}'>{f['unit']}</span>" for f in factors]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[f["prior_mean"] for f in factors], y=labels, mode="markers", name="Literature prior (before your data)",
            marker=dict(color="#fcfcfb", size=9, symbol="circle", line=dict(color=PRIOR_COLOR, width=2)),
            error_x=dict(type="data", array=[1.96 * f["prior_sd"] for f in factors], color=PRIOR_COLOR, thickness=1),
            hovertemplate="Prior: %{x:+.1f} s/km<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[f["median"] for f in factors], y=labels, mode="markers", name="Estimate with your data (95%)",
            marker=dict(color=TARGET_COLORS["2.0"], size=10, line=dict(color="#fcfcfb", width=2)),
            error_x=dict(
                type="data", symmetric=False,
                array=[f["ci95"][1] - f["median"] for f in factors],
                arrayminus=[f["median"] - f["ci95"][0] for f in factors],
                color=TARGET_COLORS["2.0"], thickness=2,
            ),
            customdata=[[f["ci95"][0], f["ci95"][1], f["n_reps_affected"]] for f in factors],
            hovertemplate="%{x:+.1f} s/km (95%: %{customdata[0]:+.1f} to %{customdata[1]:+.1f})"
            "<br>%{customdata[2]} reps affected<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color=NEUTRAL, width=1))
    fig.update_xaxes(title="← makes it easier · pace cost in s/km · makes it harder →", gridcolor="#ecebe8", zeroline=False)
    fig.update_yaxes(gridcolor="#f3f2ef")
    fig.update_layout(
        template="plotly_white", height=90 + 46 * len(factors), margin=dict(l=20, r=20, t=20, b=60),
        legend=dict(orientation="h", y=1.04, x=0, yanchor="bottom"),
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
    )
    return fig


def _heart_rate_figure(summary: dict) -> go.Figure:
    weeks = summary["heart_rate"]["trend_at_2"]
    x = [w["week_start"] for w in weeks]
    color = TARGET_COLORS["2.0"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x + x[::-1], y=[w["hi95"] for w in weeks] + [w["lo95"] for w in weeks][::-1],
            fill="toself", fillcolor=_rgba(color, 0.15), line=dict(width=0), hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=[w["median"] for w in weeks], mode="lines", line=dict(color=color, width=2), showlegend=False,
            customdata=[[w["lo95"], w["hi95"]] for w in weeks],
            hovertemplate="Week of %{x}<br>%{y:.0f} bpm (95%: %{customdata[0]:.0f}–%{customdata[1]:.0f})<extra></extra>",
        )
    )
    fig.update_yaxes(title="bpm at 2.0 mmol/L (5-min reps)", gridcolor="#ecebe8", zeroline=False)
    fig.update_xaxes(gridcolor="#ecebe8")
    fig.update_layout(
        template="plotly_white", height=300, margin=dict(l=70, r=20, t=20, b=40),
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
    )
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def render(summary: dict, offline: bool = False) -> str:
    include_plotlyjs = True if offline else "cdn"
    figs_html = {
        "fitness_trend": _fitness_trend_figure(summary).to_html(full_html=False, include_plotlyjs=include_plotlyjs),
        "factors": _factor_figure(summary).to_html(full_html=False, include_plotlyjs=False),
    }
    if summary.get("heart_rate"):
        figs_html["heart_rate"] = _heart_rate_figure(summary).to_html(full_html=False, include_plotlyjs=False)
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=select_autoescape(["html"]))
    env.filters["pace"] = format_pace
    env.filters["pace_short"] = format_pace_short
    env.filters["signed_s"] = format_signed_seconds
    env.filters["certainty"] = format_certainty
    template = env.get_template("report.html.j2")
    sessions_latest_first = sorted(summary["fitness"]["sessions"], key=lambda s: s["date"], reverse=True)
    return template.render(summary=summary, figs=figs_html, sessions=sessions_latest_first)


def write_report(summary: dict, output_path: str | Path, offline: bool = False) -> None:
    html = render(summary, offline=offline)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
