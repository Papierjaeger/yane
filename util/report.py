"""Run report generation for YANE training runs.

Supports three output formats:

- ``json`` — machine-readable dict (for programmatic consumption)
- ``md``   — human-readable Markdown (for GitHub / documentation)
- ``html`` — self-contained HTML with inline SVG fitness curve
"""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

from yane.core.genome import Genome


# ---------------------------------------------------------------------------
# SVG helpers (no external dependencies)
# ---------------------------------------------------------------------------

def _svg_polyline(points: list[tuple[float, float]], stroke: str = "#2196F3",
                  width: float = 2) -> str:
    """Return an SVG polyline element."""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{pts}" stroke="{stroke}" stroke-width="{width}" fill="none"/>'


def _fitness_svg(
    fitness_history: list[dict[str, Any]],
    width: int = 600,
    height: int = 250,
) -> str:
    """Generate an inline SVG fitness curve from training history.

    Plots ``best_fitness`` (blue), ``mean_fitness`` (green), and
    ``validation_fitness`` (orange) if available.
    """
    if not fitness_history:
        return "<!-- no fitness data -->"

    margin_l, margin_r, margin_t, margin_b = 50, 20, 20, 30
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    # Collect series
    iterations = [r.get("iteration", i) for i, r in enumerate(fitness_history)]
    best_vals: list[float] = [
        r.get("best_fitness", 0) or 0 for r in fitness_history
    ]
    mean_vals: list[float] | None = None
    if any(r.get("mean_fitness") is not None for r in fitness_history):
        mean_vals = [r.get("mean_fitness", 0) or 0 for r in fitness_history]
    valid_vals: list[float] | None = None
    if any(r.get("validation_fitness") is not None for r in fitness_history):
        valid_vals = [
            r.get("validation_fitness", 0) or 0 for r in fitness_history
        ]

    lo = min(
        min(v for v in best_vals if v == v),  # skip NaN
        min((v for v in (mean_vals or []) if v == v), default=0),
        min((v for v in (valid_vals or []) if v == v), default=0),
    )
    hi = max(
        max(v for v in best_vals if v == v),
        max((v for v in (mean_vals or []) if v == v), default=0),
        max((v for v in (valid_vals or []) if v == v), default=0),
    )
    if hi - lo < 1e-9:
        hi = lo + 1.0
    pad = (hi - lo) * 0.05
    lo -= pad
    hi += pad
    it_lo, it_hi = iterations[0], iterations[-1] or 1
    if it_hi - it_lo < 1:
        it_hi = it_lo + 1

    def _scale(it: float, val: float) -> tuple[float, float]:
        x = margin_l + (it - it_lo) / (it_hi - it_lo) * plot_w
        y = margin_t + (hi - val) / (hi - lo) * plot_h
        return x, y

    best_pts = [_scale(it, v) for it, v in zip(iterations, best_vals)]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
        f' viewBox="0 0 {width} {height}" font-family="monospace" font-size="10">',
        f'<rect width="{width}" height="{height}" fill="#fafafa"/>',
        # Y-axis
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{height - margin_b}" stroke="#ccc"/>',
        f'<text x="{margin_l - 4}" y="{margin_t + 10}" text-anchor="end" fill="#666">{hi:.2f}</text>',
        f'<text x="{margin_l - 4}" y="{height - margin_b + 4}" text-anchor="end" fill="#666">{lo:.2f}</text>',
        # X-axis
        f'<line x1="{margin_l}" y1="{height - margin_b}" x2="{width - margin_r}" y2="{height - margin_b}" stroke="#ccc"/>',
        f'<text x="{margin_l}" y="{height - 4}" fill="#666">{it_lo}</text>',
        f'<text x="{width - margin_r}" y="{height - 4}" text-anchor="end" fill="#666">{it_hi}</text>',
    ]

    # Mean fitness (green, behind best)
    if mean_vals and len(mean_vals) > 1:
        mean_pts = [_scale(it, v) for it, v in zip(iterations, mean_vals)]
        parts.append(_svg_polyline(mean_pts, "#4CAF50", 1.5))

    # Validation fitness (orange)
    if valid_vals:
        valid_pts = [
            _scale(it, v) for it, v in zip(iterations, valid_vals) if v == v
        ]
        if len(valid_pts) > 1:
            parts.append(_svg_polyline(valid_pts, "#FF9800", 1.5))

    # Best fitness (blue, front)
    if len(best_pts) > 1:
        parts.append(_svg_polyline(best_pts, "#2196F3", 2))

    # Legend
    ly = height - margin_b + 14
    parts.append(
        f'<line x1="{margin_l}" y1="{ly}" x2="{margin_l + 12}" y2="{ly}" '
        f'stroke="#2196F3" stroke-width="2"/>'
        f'<text x="{margin_l + 16}" y="{ly + 3}" fill="#555">Best</text>'
    )
    if mean_vals:
        lx2 = margin_l + 70
        parts.append(
            f'<line x1="{lx2}" y1="{ly}" x2="{lx2 + 12}" y2="{ly}" '
            f'stroke="#4CAF50" stroke-width="1.5"/>'
            f'<text x="{lx2 + 16}" y="{ly + 3}" fill="#555">Mean</text>'
        )
    if valid_vals:
        lx3 = margin_l + 145 if mean_vals else margin_l + 70
        parts.append(
            f'<line x1="{lx3}" y1="{ly}" x2="{lx3 + 12}" y2="{ly}" '
            f'stroke="#FF9800" stroke-width="1.5"/>'
            f'<text x="{lx3 + 16}" y="{ly + 3}" fill="#555">Validation</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Report data collector
# ---------------------------------------------------------------------------

def _collect_report_data(
    ne: Any,  # NeuroEvolution instance
    stop_reason: str,
    iterations: int,
) -> dict[str, Any]:
    """Collect all report data from a finished training run."""
    best: Genome = ne.get_best()
    mem = ne.population_memory_info()

    # Load fitness history from CSV
    csv_path = ne._log_run_dir / "fitness_history.csv" if ne._log_run_dir else None
    fitness_history: list[dict[str, Any]] = []
    if csv_path and csv_path.exists():
        from yane.util.run_database import load_fitness_history_from_csv
        fitness_history = load_fitness_history_from_csv(csv_path)

    data: dict[str, Any] = {
        "run_name": ne._log_run_name or "untitled",
        "stop_reason": stop_reason,
        "iterations": iterations,
        "generations": mem.get("generation", iterations // max(1, ne._population_size)),
        "n_evaluations": ne._n_evaluations_done,
        "duration_sec": None,
        "seed": ne._seed,
        "best_fitness": best.fitness,
        "best_nodes": len(best.nodes),
        "best_connections": best.connection_count,
        "best_genome_topology": [
            {
                "innovation": n.innovation,
                "type": n.type.name if hasattr(n.type, "name") else str(n.type),
                "activation": n.activation if isinstance(n.activation, str) else n.activation.value,
                "bias": round(n.bias, 6),
            }
            for n in best.nodes
        ],
        "best_genome_connections": [
            {
                "innovation": c.innovation,
                "source": src.innovation,
                "target": c.target.innovation,
                "weight": round(c.weight, 6),
                "enabled": c.enabled,
            }
            for src in best.nodes
            for c in src.connections
        ],
        "fitness_history": fitness_history,
        "species_count": mem.get("species_count", 0),
        "stagnation_count": mem.get("stagnation_count", 0),
        "recovery_events": ne._recovery_events,
        "stopped_early": ne.stopped_early,
        "config": ne._config_dict(),
        "lamarck_n_applied": mem.get("lamarck_n_applied", 0),
        "lamarck_n_steps_total": mem.get("lamarck_n_steps_total", 0),
        "n_invalid_fitness": mem.get("n_invalid_fitness", 0),
        "n_clipped_fitness": mem.get("n_clipped_fitness", 0),
    }

    # Duration
    if hasattr(ne, "_train_start_time") and ne._train_start_time:
        import time as _time
        data["duration_sec"] = round(_time.time() - ne._train_start_time, 2)

    # Anomaly events
    if ne._anomaly_detectors is not None:
        try:
            data["anomalies"] = ne._anomaly_detectors.get_diagnostics()
        except Exception:
            pass

    return data


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def _render_json(data: dict[str, Any]) -> str:
    import json
    return json.dumps(data, indent=2, default=str)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _render_markdown(data: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Run Report: {data['run_name']}")
    lines.append("")
    lines.append(f"- **Stop reason:** {data['stop_reason']}")
    lines.append(f"- **Iterations:** {data['iterations']}")
    lines.append(f"- **Generations:** {data['generations']}")
    lines.append(f"- **Evaluations:** {data['n_evaluations']}")
    if data.get("duration_sec"):
        lines.append(f"- **Duration:** {data['duration_sec']:.1f}s")
    lines.append(f"- **Seed:** {data['seed']}")
    lines.append("")

    # Best genome
    lines.append("## Best Genome")
    lines.append("")
    lines.append(f"- **Fitness:** {data['best_fitness']:.6f}")
    lines.append(f"- **Nodes:** {data['best_nodes']}")
    lines.append(f"- **Connections:** {data['best_connections']}")
    lines.append("")

    # Fitness summary
    fh = data.get("fitness_history", [])
    if fh:
        bests = [r.get("best_fitness", 0) or 0 for r in fh]
        lines.append("## Fitness Progress")
        lines.append("")
        lines.append(f"- **Initial best:** {bests[0]:.6f}")
        lines.append(f"- **Final best:** {bests[-1]:.6f}")
        if bests[-1] - bests[0] > 1e-12:
            lines.append(f"- **Improvement:** {bests[-1] - bests[0]:.6f}")
        lines.append("")

    # Config
    lines.append("## Configuration")
    lines.append("")
    cfg = data.get("config", {})
    for key, val in cfg.items():
        if isinstance(val, float):
            lines.append(f"- **{key}:** {val:.6g}")
        else:
            lines.append(f"- **{key}:** {val!r}")
    lines.append("")

    # Population diagnostics
    lines.append("## Population")
    lines.append("")
    lines.append(f"- **Species:** {data['species_count']}")
    lines.append(f"- **Stagnation:** {data['stagnation_count']}")
    if data.get("stopped_early"):
        lines.append("- ⚠ **Early stopped**")
    lines.append("")

    # Recovery
    if data.get("recovery_events"):
        lines.append("## Recovery Events")
        lines.append("")
        for ev in data["recovery_events"]:
            lines.append(f"- Gen {ev.get('generation', '?'):>4}: "
                         f"{ev.get('strategy', '?')} "
                         f"(fitness Δ: {ev.get('fitness_delta', '?'):+})")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
max-width:900px;margin:2em auto;padding:0 1em;color:#333;line-height:1.6}
h1{color:#1565C0;border-bottom:2px solid #1565C0;padding-bottom:0.3em}
h2{color:#333;margin-top:2em}
table{border-collapse:collapse;width:100%;margin:1em 0}
th,td{border:1px solid #ddd;padding:8px;text-align:left}
th{background:#f5f5f5}
code{background:#f5f5f5;padding:2px 4px;border-radius:3px;font-size:0.9em}
.section{margin:1em 0}
.badge{display:inline-block;padding:2px 8px;border-radius:3px;font-size:0.85em;
font-weight:600}
.badge-ok{background:#E8F5E9;color:#2E7D32}
.badge-warn{background:#FFF3E0;color:#E65100}
.badge-stop{background:#FFEBEE;color:#C62828}
"""


def _render_html(data: dict[str, Any]) -> str:
    fh = data.get("fitness_history", [])
    svg = _fitness_svg(fh)
    best = data.get("best_genome_topology", [])
    conns = data.get("best_genome_connections", [])
    esc = lambda value: html.escape(str(value), quote=True)

    stop_cls = "badge-stop" if data.get("stopped_early") else "badge-ok"

    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>Run Report: {esc(data['run_name'])}</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<h1>Run Report: {esc(data['run_name'])}</h1>",
        f"<p><span class='badge {stop_cls}'>{esc(data['stop_reason'])}</span></p>",
        "<table>",
        f"<tr><td><strong>Iterations</strong></td><td>{data['iterations']}</td></tr>",
        f"<tr><td><strong>Generations</strong></td><td>{data['generations']}</td></tr>",
        f"<tr><td><strong>Evaluations</strong></td><td>{data['n_evaluations']}</td></tr>",
    ]
    if data.get("duration_sec"):
        parts.append(
            f"<tr><td><strong>Duration</strong></td><td>{data['duration_sec']:.1f}s</td></tr>"
        )
    parts.extend([
        f"<tr><td><strong>Seed</strong></td><td>{esc(data['seed'])}</td></tr>",
        "</table>",
        # Fitness curve
        "<h2>Fitness Progress</h2>",
        svg,
        # Best genome
        "<h2>Best Genome</h2>",
        "<table>",
        f"<tr><td><strong>Fitness</strong></td><td>{data['best_fitness']:.6f}</td></tr>",
        f"<tr><td><strong>Nodes</strong></td><td>{data['best_nodes']}</td></tr>",
        f"<tr><td><strong>Connections</strong></td><td>{data['best_connections']}</td></tr>",
        "</table>",
    ])

    # Node table
    if best:
        parts.extend([
            "<h3>Nodes</h3><table><tr><th>ID</th><th>Type</th><th>Activation</th><th>Bias</th></tr>"
        ])
        for n in best:
            parts.append(
                f"<tr><td>{n['innovation']}</td><td>{n['type']}</td>"
                f"<td>{esc(n['activation'])}</td><td>{n['bias']}</td></tr>"
            )
        parts.append("</table>")

    # Connection table
    if conns:
        parts.extend([
            "<h3>Connections</h3>",
            "<table><tr><th>Innovation</th><th>Source</th><th>Target</th>"
            "<th>Weight</th><th>Enabled</th></tr>"
        ])
        for c in conns:
            status = "✓" if c["enabled"] else "✗"
            parts.append(
                f"<tr><td>{c['innovation']}</td><td>{c['source']}</td>"
                f"<td>{c['target']}</td><td>{c['weight']}</td><td>{status}</td></tr>"
            )
        parts.append("</table>")

    # Recovery events
    if data.get("recovery_events"):
        parts.append("<h2>Recovery Events</h2><table>"
                      "<tr><th>Generation</th><th>Strategy</th><th>Fitness Δ</th></tr>")
        for ev in data["recovery_events"]:
            parts.append(
                f"<tr><td>{ev.get('generation', '?')}</td>"
                f"<td>{esc(ev.get('strategy', '?'))}</td>"
                f"<td>{ev.get('fitness_delta', '?'):+}</td></tr>"
            )
        parts.append("</table>")

    # Config
    cfg = data.get("config", {})
    parts.append("<h2>Configuration</h2><table>")
    for key, val in cfg.items():
        if isinstance(val, float):
            parts.append(f"<tr><td><code>{esc(key)}</code></td><td>{val:.6g}</td></tr>")
        else:
            parts.append(f"<tr><td><code>{esc(key)}</code></td><td>{esc(repr(val))}</td></tr>")
    parts.append("</table>")

    parts.append("</body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

FORMAT_RENDERERS = {
    "json": _render_json,
    "md": _render_markdown,
    "markdown": _render_markdown,
    "html": _render_html,
}


def export_run_report(
    ne: Any,
    path: str | Path | None = None,
    fmt: str = "html",
    *,
    stop_reason: str = "manual",
    iterations: int = 0,
) -> str:
    """Generate a post-training report for a finished run.

    Parameters
    ----------
    ne : NeuroEvolution
        A configured instance that has completed ``train()``.
    path : str, Path, or None
        Destination file path.  Pass ``None`` to only return the content
        without writing to disk.
    fmt : str
        Output format: ``"html"`` (default), ``"md"`` / ``"markdown"``, or
        ``"json"``.
    stop_reason, iterations :
        Passed through from the training loop; usually obtained from the
        return value of ``train()``.

    Returns
    -------
    str
        The rendered content (written to *path* if given).
    """
    renderer = FORMAT_RENDERERS.get(fmt)
    if renderer is None:
        raise ValueError(f"Unknown report format {fmt!r}; supported: {list(FORMAT_RENDERERS)}")

    data = _collect_report_data(ne, stop_reason, iterations)
    content = renderer(data)

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return content
