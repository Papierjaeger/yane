"""Run report generation — HTML, JSON, and Markdown formats.

Generates per-run postmortem reports from ``Run`` objects (backed by
``RunDatabase``).  No Qt or graphviz dependency — the HTML report uses
an inline SVG fitness curve built from the stored fitness history.
"""
from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane.util.run_database import Run


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_run_report(run: Run, format: str = "html") -> str:
    """Generate a postmortem report for *run*.

    Args:
        run:    A ``Run`` object (from ``RunDatabase``).
        format: ``"html"``, ``"json"``, or ``"md"`` (Markdown).

    Returns:
        Report as a string in the requested format.
    """
    if format == "json":
        return _report_json(run)
    if format == "md":
        return _report_markdown(run)
    if format == "html":
        return _report_html(run)
    raise ValueError(f"Unknown report format {format!r}. Use 'html', 'json', or 'md'.")


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def _report_json(run: Run) -> str:
    data = {
        "run_id": run.run_id,
        "name": run.name,
        "experiment": run.experiment_id,
        "start_time": run.start_time,
        "end_time": run.end_time,
        "stop_reason": run.stop_reason,
        "final_best": run.final_best,
        "seed": run.seed,
        "config": run.config,
        "diagnostics": run.diagnostics,
        "fitness_history": run.fitness_history,
        "artifacts": run.artifacts,
    }
    return json.dumps(data, indent=2, default=str)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _report_markdown(run: Run) -> str:
    lines: list[str] = []
    _name = run.name or run.run_id
    lines.append(f"# Run Report: {_name}\n")

    lines.append("## Summary\n")
    lines.append(f"| Field | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Run ID | `{run.run_id}` |")
    lines.append(f"| Experiment | {run.experiment_id or '—'} |")
    lines.append(f"| Start | {run.start_time or '—'} |")
    lines.append(f"| End | {run.end_time or '—'} |")
    lines.append(f"| Stop reason | {run.stop_reason or '—'} |")
    lines.append(f"| Best fitness | {run.final_best:.6f} |")
    lines.append(f"| Seed | {run.seed} |")
    lines.append("")

    lines.append("## Fitness History\n")
    if run.fitness_history:
        lines.append("| Iteration | Best Fitness |")
        lines.append("|---|---|")
        step = max(1, len(run.fitness_history) // 20)
        for i, row in enumerate(run.fitness_history):
            if i % step == 0 or i == len(run.fitness_history) - 1:
                lines.append(f"| {row.get('iteration', i)} | {row.get('best_fitness', 0):.6f} |")
        lines.append("")
    else:
        lines.append("*(no fitness history)*\n")

    if run.config:
        lines.append("## Configuration\n")
        lines.append("| Parameter | Value |")
        lines.append("|---|---|")
        for k, v in sorted(run.config.items()):
            lines.append(f"| {k} | {v} |")
        lines.append("")

    if run.diagnostics:
        lines.append("## Final Diagnostics\n")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for k, v in sorted(run.diagnostics.items()):
            if not isinstance(v, (dict, list)):
                lines.append(f"| {k} | {v} |")
        lines.append("")

    if run.artifacts:
        lines.append("## Artifacts\n")
        for k, v in run.artifacts.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML (inline SVG fitness curve, no external dependencies)
# ---------------------------------------------------------------------------

def _report_html(run: Run) -> str:
    _name = html.escape(run.name or run.run_id)
    svg = _svg_fitness_curve(run.fitness_history)
    summary_rows = _kv_rows_html({
        "Run ID": run.run_id,
        "Experiment": run.experiment_id or "—",
        "Start": run.start_time or "—",
        "End": run.end_time or "—",
        "Stop reason": run.stop_reason or "—",
        "Best fitness": f"{run.final_best:.6f}",
        "Seed": run.seed,
    })
    config_rows = _kv_rows_html(run.config or {})
    diag_flat = {k: v for k, v in (run.diagnostics or {}).items()
                 if not isinstance(v, (dict, list))}
    diag_rows = _kv_rows_html(diag_flat)
    artifact_rows = _kv_rows_html(run.artifacts or {})

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Run Report — {_name}</title>
<style>
  body {{font-family: "Segoe UI", sans-serif; background: #1e1e2e; color: #cdd6f4;
         max-width: 960px; margin: 40px auto; padding: 0 20px;}}
  h1   {{color: #89b4fa; border-bottom: 1px solid #313244; padding-bottom: 8px;}}
  h2   {{color: #a6adc8; margin-top: 32px;}}
  table{{border-collapse: collapse; width: 100%; margin-top: 8px;}}
  th,td{{border: 1px solid #313244; padding: 6px 12px; text-align: left;}}
  th   {{background: #313244; color: #cba6f7;}}
  tr:nth-child(even){{background: #181825;}}
  .svg-wrap{{background: #181825; border: 1px solid #313244; border-radius: 6px;
             padding: 12px; margin-top: 8px;}}
</style>
</head>
<body>
<h1>Run Report: {_name}</h1>
<h2>Summary</h2>
<table><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>
{summary_rows}
</tbody></table>
<h2>Fitness Curve</h2>
<div class="svg-wrap">{svg}</div>
<h2>Configuration</h2>
<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>
{config_rows}
</tbody></table>
<h2>Final Diagnostics</h2>
<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>
{diag_rows}
</tbody></table>
<h2>Artifacts</h2>
<table><thead><tr><th>Type</th><th>Path</th></tr></thead><tbody>
{artifact_rows}
</tbody></table>
</body>
</html>"""


def _kv_rows_html(data: dict) -> str:
    rows = []
    for k, v in sorted(data.items()):
        ek = html.escape(str(k))
        ev = html.escape(str(v))
        rows.append(f"<tr><td>{ek}</td><td>{ev}</td></tr>")
    return "\n".join(rows)


def _svg_fitness_curve(fitness_history: list[dict]) -> str:
    """Return an inline SVG of the fitness-over-iteration curve."""
    if not fitness_history:
        return "<svg width='100%' height='60'><text x='10' y='30' fill='#585b70'>No data</text></svg>"

    W, H, PAD = 860, 200, 40
    xs = [r.get("iteration", i) for i, r in enumerate(fitness_history)]
    ys = [r.get("best_fitness", 0.0) for r in fitness_history]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = max(x_max - x_min, 1)
    y_range = max(y_max - y_min, 1e-9)

    def px(x):
        return PAD + (x - x_min) / x_range * (W - 2 * PAD)

    def py(y):
        return H - PAD - (y - y_min) / y_range * (H - 2 * PAD)

    # Sample at most 400 points to keep SVG small.
    step = max(1, len(fitness_history) // 400)
    pts = list(zip(xs, ys))[::step]
    if pts[-1] != (xs[-1], ys[-1]):
        pts.append((xs[-1], ys[-1]))

    polyline = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)

    # Axis labels
    x_labels = ""
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        lx = x_min + frac * x_range
        sx = px(lx)
        x_labels += (f'<text x="{sx:.0f}" y="{H - PAD + 16}" '
                     f'fill="#585b70" font-size="11" text-anchor="middle">'
                     f'{int(lx)}</text>')
    y_labels = ""
    for frac in (0.0, 0.5, 1.0):
        ly = y_min + frac * y_range
        sy = py(ly)
        y_labels += (f'<text x="{PAD - 4}" y="{sy:.0f}" '
                     f'fill="#585b70" font-size="11" text-anchor="end" dominant-baseline="middle">'
                     f'{ly:.3g}</text>')

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{H-PAD}" stroke="#313244" stroke-width="1"/>
  <line x1="{PAD}" y1="{H-PAD}" x2="{W-PAD}" y2="{H-PAD}" stroke="#313244" stroke-width="1"/>
  {x_labels}
  {y_labels}
  <polyline points="{polyline}" fill="none" stroke="#89b4fa" stroke-width="2" stroke-linejoin="round"/>
  <circle cx="{px(xs[-1]):.1f}" cy="{py(ys[-1]):.1f}" r="4" fill="#a6e3a1"/>
</svg>"""
