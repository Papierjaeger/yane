"""Regression detection, baseline management, and trend tracking for YANE benchmarks.

Core classes
------------
``RegressionSeverity``   — NONE / MINOR / MAJOR / CRITICAL
``RegressionDetector``   — compares current results against a saved baseline
``BaselineStore``        — load / save per-benchmark JSON baselines
``HistoryStore``         — append / load time-series entries per benchmark
``TrendReport``          — wraps history entries; provides an ASCII plot
``BenchmarkReport``      — top-level result from ``run_benchmark_suite()``

Top-level function
------------------
``run_benchmark_suite(path, update_baseline, ...)``
    Loads a ``benchmarks.yaml``, runs each benchmark, compares against
    baselines, appends to history, and returns a ``BenchmarkReport``.
"""
from __future__ import annotations

import enum
import json
import os
import statistics
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------

class RegressionSeverity(enum.Enum):
    NONE     = "none"
    MINOR    = "minor"    # relative degradation  0% –  5%
    MAJOR    = "major"    # relative degradation  5% – 20%
    CRITICAL = "critical" # relative degradation >20%, or success_rate drops significantly


def _worst(*severities: RegressionSeverity) -> RegressionSeverity:
    order = [RegressionSeverity.NONE, RegressionSeverity.MINOR,
             RegressionSeverity.MAJOR, RegressionSeverity.CRITICAL]
    return max(severities, key=lambda s: order.index(s))


# ---------------------------------------------------------------------------
# Regression result
# ---------------------------------------------------------------------------

@dataclass
class RegressionResult:
    benchmark:       str
    metric:          str
    baseline_value:  float
    current_value:   float
    relative_change: float          # positive = improvement, negative = regression
    severity:        RegressionSeverity
    p_value:         float | None   # Mann-Whitney-U p-value, None if unavailable


# ---------------------------------------------------------------------------
# Regression detector
# ---------------------------------------------------------------------------

def _mannwhitney_p(a: list[float], b: list[float]) -> float | None:
    """Return Mann-Whitney-U p-value, or None if scipy is not installed."""
    if len(a) < 2 or len(b) < 2:
        return None
    try:
        from scipy.stats import mannwhitneyu
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(p)
    except ImportError:
        return None
    except Exception:
        return None


def _severity_from_relative_change(rel: float) -> RegressionSeverity:
    """Map a relative change (negative = regression) to severity."""
    if rel >= -0.05:
        return RegressionSeverity.NONE
    if rel >= -0.10:
        return RegressionSeverity.MINOR
    if rel >= -0.20:
        return RegressionSeverity.MAJOR
    return RegressionSeverity.CRITICAL


class RegressionDetector:
    """Compare current benchmark results against a saved baseline.

    Inputs are dicts with keys:
        ``fitnesses``     list[float]  — best fitness per seed run
        ``iterations``    list[int]    — iterations to solve (solved runs only)
        ``success_rate``  float        — fraction of seeds that hit target fitness
        ``median_fitness`` float       — pre-computed (optional)
    """

    # Success-rate drop threshold for a CRITICAL regression
    CRITICAL_SUCCESS_DROP: float = 0.20

    def compare(
        self,
        name: str,
        baseline: dict,
        current: dict,
    ) -> list[RegressionResult]:
        """Return a list of regression findings for *name*."""
        results: list[RegressionResult] = []

        # --- median fitness -------------------------------------------------
        base_fits = baseline.get("fitnesses", [])
        curr_fits = current.get("fitnesses", [])
        if base_fits and curr_fits:
            base_med = statistics.median(base_fits)
            curr_med = statistics.median(curr_fits)
            rel = self._relative_change(base_med, curr_med)
            sev = _severity_from_relative_change(rel)
            p = _mannwhitney_p(base_fits, curr_fits)
            # Downgrade to NONE if not statistically significant
            if p is not None and p >= 0.05 and sev == RegressionSeverity.MINOR:
                sev = RegressionSeverity.NONE
            results.append(RegressionResult(
                benchmark=name, metric="median_fitness",
                baseline_value=base_med, current_value=curr_med,
                relative_change=rel, severity=sev, p_value=p,
            ))

        # --- success rate ---------------------------------------------------
        base_sr = baseline.get("success_rate")
        curr_sr = current.get("success_rate")
        if base_sr is not None and curr_sr is not None:
            drop = curr_sr - base_sr   # negative = regression
            if drop < -self.CRITICAL_SUCCESS_DROP:
                sev = RegressionSeverity.CRITICAL
            elif drop < -0.10:
                sev = RegressionSeverity.MAJOR
            elif drop < 0.0:
                sev = RegressionSeverity.MINOR
            else:
                sev = RegressionSeverity.NONE
            results.append(RegressionResult(
                benchmark=name, metric="success_rate",
                baseline_value=base_sr, current_value=curr_sr,
                relative_change=drop, severity=sev, p_value=None,
            ))

        # --- convergence speed (iterations to solve) ------------------------
        base_iters = baseline.get("iterations", [])
        curr_iters = current.get("iterations", [])
        if base_iters and curr_iters:
            base_med_i = statistics.median(base_iters)
            curr_med_i = statistics.median(curr_iters)
            # More iterations = regression (invert sign relative to fitness)
            rel = self._relative_change(base_med_i, curr_med_i, higher_is_worse=True)
            sev = _severity_from_relative_change(rel)
            p = _mannwhitney_p(base_iters, curr_iters)
            if p is not None and p >= 0.05 and sev == RegressionSeverity.MINOR:
                sev = RegressionSeverity.NONE
            results.append(RegressionResult(
                benchmark=name, metric="median_iterations",
                baseline_value=base_med_i, current_value=curr_med_i,
                relative_change=rel, severity=sev, p_value=p,
            ))

        return results

    @staticmethod
    def _relative_change(
        base: float,
        current: float,
        higher_is_worse: bool = False,
    ) -> float:
        """Return relative change as a fraction where negative = regression."""
        if abs(base) < 1e-12:
            return 0.0 if abs(current - base) < 1e-12 else (-1.0 if higher_is_worse else 1.0)
        raw = (current - base) / abs(base)
        return -raw if higher_is_worse else raw


# ---------------------------------------------------------------------------
# Baseline storage
# ---------------------------------------------------------------------------

class BaselineStore:
    """Persist per-benchmark baseline data as JSON files in *directory*."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def _path(self, name: str) -> Path:
        safe = name.replace("/", "_").replace(" ", "_")
        return self.directory / f"{safe}.json"

    def load(self, name: str) -> dict | None:
        path = self._path(name)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def save(self, name: str, data: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(name)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def list_names(self) -> list[str]:
        if not self.directory.exists():
            return []
        return [p.stem for p in sorted(self.directory.glob("*.json"))]


# ---------------------------------------------------------------------------
# History storage
# ---------------------------------------------------------------------------

class HistoryStore:
    """Persist time-series benchmark data as JSONL files in *directory*."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def _path(self, name: str) -> Path:
        safe = name.replace("/", "_").replace(" ", "_")
        return self.directory / f"{safe}.jsonl"

    def append(self, name: str, entry: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(name)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def load(self, name: str) -> list[dict]:
        path = self._path(name)
        if not path.exists():
            return []
        entries: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries


# ---------------------------------------------------------------------------
# Trend report
# ---------------------------------------------------------------------------

@dataclass
class TrendReport:
    """Historical trend data for a single benchmark."""

    name:    str
    entries: list[dict]

    def ascii_plot(self, metric: str = "median_fitness", width: int = 60) -> str:
        """Return a compact ASCII sparkline of *metric* over time."""
        values = [e.get(metric) for e in self.entries if e.get(metric) is not None]
        if not values:
            return f"[{self.name}] no data for '{metric}'"

        lo, hi = min(values), max(values)
        span = hi - lo
        blocks = " ▁▂▃▄▅▆▇█"
        n = min(len(values), width)
        # Subsample if more entries than width
        step = max(1, len(values) // n)
        sample = values[::step][:n]

        bar = ""
        for v in sample:
            if span < 1e-12:
                idx = 4
            else:
                idx = int((v - lo) / span * (len(blocks) - 1))
            bar += blocks[max(0, min(idx, len(blocks) - 1))]

        last = values[-1]
        first = values[0]
        trend_sym = "↑" if last > first else "↓" if last < first else "→"
        return (
            f"[{self.name}] {metric}  "
            f"{first:.4g} → {last:.4g} {trend_sym}\n"
            f"  {bar}"
        )


# ---------------------------------------------------------------------------
# Benchmark report
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkReport:
    """Top-level result from ``run_benchmark_suite()``."""

    timestamp:        str
    commit:           str
    benchmark_results: list[dict]            # one dict per benchmark
    regressions:      list[RegressionResult]
    overall_severity: RegressionSeverity

    def exit_code(self) -> int:
        """0 = NONE/MINOR, 1 = MAJOR, 2 = CRITICAL."""
        if self.overall_severity == RegressionSeverity.CRITICAL:
            return 2
        if self.overall_severity == RegressionSeverity.MAJOR:
            return 1
        return 0

    def to_dict(self) -> dict:
        regressions = [
            {
                "benchmark":       r.benchmark,
                "metric":          r.metric,
                "baseline_value":  r.baseline_value,
                "current_value":   r.current_value,
                "relative_change": round(r.relative_change, 6),
                "severity":        r.severity.value,
                "p_value":         r.p_value,
            }
            for r in self.regressions
        ]
        return {
            "timestamp":         self.timestamp,
            "commit":            self.commit,
            "overall_severity":  self.overall_severity.value,
            "exit_code":         self.exit_code(),
            "benchmark_results": self.benchmark_results,
            "regressions":       regressions,
        }

    def format_text(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Benchmark Report  {self.timestamp}  commit={self.commit}",
            f"Overall severity:  {self.overall_severity.value.upper()}",
            "",
        ]
        sev_counts: dict[str, int] = {}
        for r in self.regressions:
            if r.severity != RegressionSeverity.NONE:
                key = r.severity.value
                sev_counts[key] = sev_counts.get(key, 0) + 1
        if not sev_counts:
            lines.append("  No regressions detected.")
        else:
            lines.append("  Regressions:")
            for r in self.regressions:
                if r.severity == RegressionSeverity.NONE:
                    continue
                pval = f" p={r.p_value:.3f}" if r.p_value is not None else ""
                lines.append(
                    f"    [{r.severity.value.upper():8s}] {r.benchmark}.{r.metric}  "
                    f"{r.baseline_value:.4g} → {r.current_value:.4g} "
                    f"({r.relative_change:+.1%}){pval}"
                )
        lines.append("")
        lines.append("  Benchmark summary:")
        for br in self.benchmark_results:
            s = br.get("summary", {})
            lines.append(
                f"    {br.get('name', '?'):<30} "
                f"solved={s.get('solved', '?')}/{s.get('n', '?')} "
                f"fit={s.get('median_fitness', float('nan')):.4g}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------

def _load_config(path: str | Path) -> dict:
    path = Path(path)
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except ImportError:
        raise ImportError(
            "run_benchmark_suite() requires PyYAML.  "
            "Install it with:  pip install pyyaml"
        )


def _current_commit() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_benchmark_suite(
    path: str | Path = "benchmarks/benchmarks.yaml",
    *,
    update_baseline: bool = False,
    baseline_dir: str | Path | None = None,
    history_dir: str | Path | None = None,
    verbose: bool = True,
) -> BenchmarkReport:
    """Run all benchmarks defined in *path* and detect regressions.

    Args:
        path:             Path to ``benchmarks.yaml``.
        update_baseline:  When True, save current results as new baseline.
        baseline_dir:     Override the baseline directory from the YAML config.
        history_dir:      Override the history directory from the YAML config.
        verbose:          Print progress to stdout.

    Returns:
        :class:`BenchmarkReport` with per-benchmark results and regression findings.
    """
    from benchmarks.default_bench import _run_case, Case

    cfg = _load_config(path)
    root = Path(path).parent

    bl_dir = Path(baseline_dir) if baseline_dir else root / cfg.get("baseline_dir", "baseline")
    hi_dir = Path(history_dir)  if history_dir  else root / cfg.get("history_dir",  "history")

    baseline_store = BaselineStore(bl_dir)
    history_store  = HistoryStore(hi_dir)
    detector       = RegressionDetector()

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    commit    = _current_commit()

    benchmark_results: list[dict] = []
    all_regressions:   list[RegressionResult] = []

    for bm_cfg in cfg.get("benchmarks", []):
        name     = bm_cfg["name"]
        example  = bm_cfg.get("example", name)
        seeds    = bm_cfg.get("seeds", [0, 1, 2])
        max_iter = bm_cfg.get("max_iterations", 5000)
        timeout  = bm_cfg.get("timeout_s", 30.0)

        case = Case(
            example=example,
            seeds=seeds,
            max_iterations=max_iter,
            population=bm_cfg.get("population"),
            target_species=bm_cfg.get("target_species"),
            max_nodes=bm_cfg.get("max_nodes"),
            max_connections=bm_cfg.get("max_connections"),
            n_initial_hidden=bm_cfg.get("n_initial_hidden"),
            timeout_s=timeout,
            lamarck_steps=bm_cfg.get("lamarck_steps"),
            fitness_shaping=bm_cfg.get("fitness_shaping"),
        )

        if verbose:
            print(f"  Running {name} ({example}, seeds={seeds})…", flush=True)

        t0 = time.perf_counter()
        result = _run_case(case)
        elapsed = time.perf_counter() - t0

        summary = result["summary"]
        fitnesses  = [r["best_fitness"] for r in result["runs"]]
        iters_sols = [r["iterations"] for r in result["runs"] if r.get("solved")]

        current_data = {
            "name":            name,
            "timestamp":       timestamp,
            "commit":          commit,
            "fitnesses":       fitnesses,
            "iterations":      iters_sols,
            "success_rate":    summary.get("success_rate", 0.0),
            "median_fitness":  summary.get("median_fitness", float("nan")),
            "mean_fitness":    summary.get("mean_fitness", float("nan")),
            "solved":          summary.get("solved", 0),
            "n":               summary.get("n", len(seeds)),
            "elapsed_s":       round(elapsed, 2),
        }

        bm_result = {"name": name, "example": example, "summary": summary, "runs": result["runs"]}
        benchmark_results.append(bm_result)

        # Regression check
        baseline = baseline_store.load(name)
        if baseline is not None:
            regressions = detector.compare(name, baseline, current_data)
            all_regressions.extend(regressions)
            if verbose and any(r.severity != RegressionSeverity.NONE for r in regressions):
                for r in regressions:
                    if r.severity != RegressionSeverity.NONE:
                        print(
                            f"    [{r.severity.value.upper()}] {r.metric}: "
                            f"{r.baseline_value:.4g} → {r.current_value:.4g} "
                            f"({r.relative_change:+.1%})"
                        )
        elif verbose:
            print(f"    (no baseline for '{name}', skipping regression check)")

        # Update baseline if requested
        if update_baseline:
            baseline_store.save(name, current_data)
            if verbose:
                print(f"    Baseline updated: {bl_dir / (name + '.json')}")

        # Append to history
        history_entry = {
            k: v for k, v in current_data.items()
            if k not in ("fitnesses", "iterations", "runs")
        }
        history_store.append(name, history_entry)

        if verbose:
            print(
                f"    solved={summary.get('solved')}/{summary.get('n')} "
                f"fit={summary.get('median_fitness', float('nan')):.4g} "
                f"({elapsed:.1f}s)"
            )

    overall = RegressionSeverity.NONE
    for r in all_regressions:
        overall = _worst(overall, r.severity)

    return BenchmarkReport(
        timestamp=timestamp,
        commit=commit,
        benchmark_results=benchmark_results,
        regressions=all_regressions,
        overall_severity=overall,
    )
