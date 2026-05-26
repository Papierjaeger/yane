"""Backward-compatible run report API.

The current report implementation lives in :mod:`yane.util.report`; this
module keeps older ``from yane.util.run_report import generate_run_report``
callers working for ``RunDatabase`` run objects.
"""
from __future__ import annotations

import html
import json


def generate_run_report(run, format: str = "html") -> str:
    """Generate a report from a ``RunDatabase`` Run object."""
    if format == "json":
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
    if format == "md":
        return _report_markdown(run)
    if format == "html":
        return _report_html(run)
    raise ValueError(f"Unknown report format {format!r}. Use 'html', 'json', or 'md'.")


def _report_markdown(run) -> str:
    name = run.name or run.run_id
    final_best = run.final_best
    best_text = f"{final_best:.6f}" if final_best is not None else "-"
    lines = [
        f"# Run Report: {name}",
        "",
        "## Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Run ID | `{run.run_id}` |",
        f"| Experiment | {run.experiment_id or '-'} |",
        f"| Start | {run.start_time or '-'} |",
        f"| End | {run.end_time or '-'} |",
        f"| Stop reason | {run.stop_reason or '-'} |",
        f"| Best fitness | {best_text} |",
        f"| Seed | {run.seed} |",
        "",
        "## Fitness History",
        "",
    ]
    if run.fitness_history:
        lines.append("| Iteration | Best Fitness |")
        lines.append("|---|---|")
        step = max(1, len(run.fitness_history) // 20)
        for i, row in enumerate(run.fitness_history):
            if i % step == 0 or i == len(run.fitness_history) - 1:
                lines.append(f"| {row.get('iteration', i)} | {row.get('best_fitness', 0):.6f} |")
    else:
        lines.append("*(no fitness history)*")
    return "\n".join(lines)


def _report_html(run) -> str:
    name = html.escape(str(run.name or run.run_id))
    rows = _kv_rows_html({
        "Run ID": run.run_id,
        "Experiment": run.experiment_id or "-",
        "Start": run.start_time or "-",
        "End": run.end_time or "-",
        "Stop reason": run.stop_reason or "-",
        "Best fitness": run.final_best if run.final_best is not None else "-",
        "Seed": run.seed,
    })
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>Run Report: {name}</title></head><body>"
        f"<h1>Run Report: {name}</h1>"
        "<h2>Summary</h2>"
        "<table><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>"
        f"{rows}</tbody></table></body></html>"
    )


def _kv_rows_html(data: dict) -> str:
    return "\n".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in data.items()
    )
