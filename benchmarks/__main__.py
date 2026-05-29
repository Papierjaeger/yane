"""CLI entry point for the YANE regression benchmark suite.

Usage::

    # CI mode — JSON report on stdout, exit code 0/1/2
    python -m yane.benchmarks --ci

    # Update baselines (save current run as new reference)
    python -m yane.benchmarks --update-baseline

    # Human-readable report, no regression check
    python -m yane.benchmarks

    # Custom config file
    python -m yane.benchmarks --config path/to/benchmarks.yaml

    # Show trend plot for one benchmark
    python -m yane.benchmarks --trend xor

Exit codes
----------
0 — NONE or MINOR regressions only (or no baseline available)
1 — MAJOR regression detected
2 — CRITICAL regression detected
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_DEFAULT_CONFIG = Path(__file__).parent / "benchmarks.yaml"


def _cmd_run(args: argparse.Namespace) -> int:
    from benchmarks.regression import run_benchmark_suite, RegressionSeverity

    report = run_benchmark_suite(
        path=args.config,
        update_baseline=args.update_baseline,
        verbose=not args.ci,
    )

    if args.ci:
        # Machine-readable JSON to stdout
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print()
        print(report.format_text())

    return report.exit_code()


def _cmd_trend(args: argparse.Namespace) -> int:
    from benchmarks.regression import HistoryStore
    import yaml  # type: ignore[import-untyped]

    cfg_path = Path(args.config)
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        return 1

    root = cfg_path.parent
    hi_dir = root / cfg.get("history_dir", "history")
    store  = HistoryStore(hi_dir)

    name    = args.trend
    entries = store.load(name)
    if not entries:
        print(f"No history found for '{name}' in {hi_dir}", file=sys.stderr)
        return 1

    from benchmarks.regression import TrendReport
    trend = TrendReport(name=name, entries=entries)
    metric = args.metric or "median_fitness"
    print(trend.ascii_plot(metric=metric, width=70))
    print(f"\n  {len(entries)} entries  last: {entries[-1].get('timestamp', '?')}")
    return 0


def _cmd_list_baselines(args: argparse.Namespace) -> int:
    from benchmarks.regression import BaselineStore
    import yaml  # type: ignore[import-untyped]

    cfg_path = Path(args.config)
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        return 1

    root   = cfg_path.parent
    bl_dir = root / cfg.get("baseline_dir", "baseline")
    store  = BaselineStore(bl_dir)
    names  = store.list_names()

    if not names:
        print(f"No baselines found in {bl_dir}")
        return 0

    print(f"Baselines in {bl_dir}:")
    for n in names:
        data = store.load(n)
        ts   = (data or {}).get("timestamp", "?")
        print(f"  {n:<30} {ts}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m yane.benchmarks",
        description="YANE regression benchmark suite",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=_DEFAULT_CONFIG,
        metavar="PATH",
        help=f"Path to benchmarks.yaml (default: {_DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: emit JSON report on stdout, use exit codes 0/1/2",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Save current results as new baseline after running",
    )
    parser.add_argument(
        "--trend",
        metavar="NAME",
        help="Show ASCII trend plot for a benchmark by name and exit",
    )
    parser.add_argument(
        "--metric",
        default=None,
        metavar="METRIC",
        help="Metric to plot with --trend (default: median_fitness)",
    )
    parser.add_argument(
        "--list-baselines",
        action="store_true",
        help="List saved baselines and exit",
    )

    args = parser.parse_args(argv)

    if args.list_baselines:
        return _cmd_list_baselines(args)
    if args.trend:
        return _cmd_trend(args)
    return _cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
