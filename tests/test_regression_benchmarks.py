"""Tests for the regression benchmark suite (benchmarks/regression.py).

All tests use synthetic data — no actual NeuroEvolution training runs.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


# ---------------------------------------------------------------------------
# RegressionSeverity + _severity_from_relative_change
# ---------------------------------------------------------------------------

class TestRegressionSeverity(unittest.TestCase):
    def test_none_for_improvement(self):
        from benchmarks.regression import _severity_from_relative_change, RegressionSeverity
        self.assertEqual(_severity_from_relative_change(0.10), RegressionSeverity.NONE)

    def test_none_for_zero_change(self):
        from benchmarks.regression import _severity_from_relative_change, RegressionSeverity
        self.assertEqual(_severity_from_relative_change(0.0), RegressionSeverity.NONE)

    def test_minor_for_small_degradation(self):
        from benchmarks.regression import _severity_from_relative_change, RegressionSeverity
        # -4% is below the NONE threshold of -5%
        self.assertEqual(_severity_from_relative_change(-0.04), RegressionSeverity.NONE)
        # -6% → MINOR
        self.assertEqual(_severity_from_relative_change(-0.06), RegressionSeverity.MINOR)

    def test_major_for_medium_degradation(self):
        from benchmarks.regression import _severity_from_relative_change, RegressionSeverity
        self.assertEqual(_severity_from_relative_change(-0.15), RegressionSeverity.MAJOR)

    def test_critical_for_large_degradation(self):
        from benchmarks.regression import _severity_from_relative_change, RegressionSeverity
        self.assertEqual(_severity_from_relative_change(-0.25), RegressionSeverity.CRITICAL)


# ---------------------------------------------------------------------------
# RegressionDetector
# ---------------------------------------------------------------------------

class TestRegressionDetector(unittest.TestCase):
    def _det(self):
        from benchmarks.regression import RegressionDetector
        return RegressionDetector()

    def test_no_regression_when_fitness_improves(self):
        from benchmarks.regression import RegressionSeverity
        det = self._det()
        baseline = {"fitnesses": [-1.0, -1.0, -1.0], "success_rate": 0.0}
        current  = {"fitnesses": [-0.5, -0.5, -0.5], "success_rate": 0.0}
        results  = det.compare("xor", baseline, current)
        for r in results:
            self.assertEqual(r.severity, RegressionSeverity.NONE,
                             f"Expected NONE for {r.metric}, got {r.severity}")

    def test_critical_regression_on_large_fitness_drop(self):
        from benchmarks.regression import RegressionSeverity
        det = self._det()
        baseline = {"fitnesses": [-0.05, -0.05, -0.05], "success_rate": 1.0}
        current  = {"fitnesses": [-2.00, -2.00, -2.00], "success_rate": 0.0}
        results  = det.compare("xor", baseline, current)
        severities = {r.metric: r.severity for r in results}
        self.assertEqual(severities.get("median_fitness"), RegressionSeverity.CRITICAL)
        self.assertEqual(severities.get("success_rate"),   RegressionSeverity.CRITICAL)

    def test_major_regression_on_moderate_fitness_drop(self):
        from benchmarks.regression import RegressionSeverity
        det = self._det()
        # fitness drops from -1 to -1.15 → 15% worse → MAJOR
        baseline = {"fitnesses": [-1.0] * 5, "success_rate": 0.8}
        current  = {"fitnesses": [-1.15] * 5, "success_rate": 0.8}
        results  = det.compare("bench", baseline, current)
        fit_result = next(r for r in results if r.metric == "median_fitness")
        self.assertEqual(fit_result.severity, RegressionSeverity.MAJOR)

    def test_returns_empty_on_no_shared_metrics(self):
        det = self._det()
        results = det.compare("x", {}, {})
        self.assertEqual(results, [])

    def test_success_rate_drop_triggers_critical(self):
        from benchmarks.regression import RegressionSeverity
        det = self._det()
        baseline = {"success_rate": 1.0}
        current  = {"success_rate": 0.6}   # drop of 0.4 > CRITICAL_SUCCESS_DROP
        results  = det.compare("bench", baseline, current)
        sr = next((r for r in results if r.metric == "success_rate"), None)
        self.assertIsNotNone(sr)
        self.assertEqual(sr.severity, RegressionSeverity.CRITICAL)

    def test_convergence_regression_detected(self):
        from benchmarks.regression import RegressionSeverity
        det = self._det()
        # Solving takes 3× longer → CRITICAL
        baseline = {"iterations": [100, 120, 110]}
        current  = {"iterations": [350, 370, 360]}
        results  = det.compare("bench", baseline, current)
        itr = next((r for r in results if r.metric == "median_iterations"), None)
        self.assertIsNotNone(itr)
        self.assertIn(itr.severity,
                      [RegressionSeverity.MAJOR, RegressionSeverity.CRITICAL])

    def test_relative_change_zero_base(self):
        from benchmarks.regression import RegressionDetector
        det = RegressionDetector()
        self.assertEqual(det._relative_change(0.0, 0.0), 0.0)
        # Non-zero current with near-zero base
        self.assertNotEqual(det._relative_change(0.0, 1.0), 0.0)


# ---------------------------------------------------------------------------
# BaselineStore
# ---------------------------------------------------------------------------

class TestBaselineStore(unittest.TestCase):
    def test_save_and_load(self):
        from benchmarks.regression import BaselineStore
        with tempfile.TemporaryDirectory() as tmp:
            store = BaselineStore(tmp)
            data = {"fitnesses": [-0.05, -0.05], "success_rate": 1.0}
            store.save("xor", data)
            loaded = store.load("xor")
            self.assertIsNotNone(loaded)
            self.assertAlmostEqual(loaded["success_rate"], 1.0)
            self.assertEqual(loaded["fitnesses"], [-0.05, -0.05])

    def test_load_missing_returns_none(self):
        from benchmarks.regression import BaselineStore
        with tempfile.TemporaryDirectory() as tmp:
            store = BaselineStore(tmp)
            self.assertIsNone(store.load("nonexistent"))

    def test_list_names(self):
        from benchmarks.regression import BaselineStore
        with tempfile.TemporaryDirectory() as tmp:
            store = BaselineStore(tmp)
            store.save("alpha", {"x": 1})
            store.save("beta",  {"x": 2})
            names = store.list_names()
            self.assertIn("alpha", names)
            self.assertIn("beta",  names)

    def test_update_overwrites(self):
        from benchmarks.regression import BaselineStore
        with tempfile.TemporaryDirectory() as tmp:
            store = BaselineStore(tmp)
            store.save("xor", {"success_rate": 0.5})
            store.save("xor", {"success_rate": 0.9})
            loaded = store.load("xor")
            self.assertAlmostEqual(loaded["success_rate"], 0.9)

    def test_directory_created_on_save(self):
        from benchmarks.regression import BaselineStore
        with tempfile.TemporaryDirectory() as tmp:
            subdir = Path(tmp) / "new" / "subdir"
            store  = BaselineStore(subdir)
            store.save("bench", {"x": 1})
            self.assertTrue(subdir.exists())


# ---------------------------------------------------------------------------
# HistoryStore
# ---------------------------------------------------------------------------

class TestHistoryStore(unittest.TestCase):
    def test_append_and_load(self):
        from benchmarks.regression import HistoryStore
        with tempfile.TemporaryDirectory() as tmp:
            store = HistoryStore(tmp)
            store.append("xor", {"median_fitness": -0.05, "timestamp": "2024-01-01"})
            store.append("xor", {"median_fitness": -0.03, "timestamp": "2024-01-02"})
            entries = store.load("xor")
            self.assertEqual(len(entries), 2)
            self.assertAlmostEqual(entries[1]["median_fitness"], -0.03)

    def test_load_missing_returns_empty_list(self):
        from benchmarks.regression import HistoryStore
        with tempfile.TemporaryDirectory() as tmp:
            store = HistoryStore(tmp)
            self.assertEqual(store.load("missing"), [])

    def test_multiple_benchmarks_independent(self):
        from benchmarks.regression import HistoryStore
        with tempfile.TemporaryDirectory() as tmp:
            store = HistoryStore(tmp)
            store.append("xor",  {"v": 1})
            store.append("cart", {"v": 2})
            self.assertEqual(len(store.load("xor")),  1)
            self.assertEqual(len(store.load("cart")), 1)
            self.assertEqual(store.load("xor")[0]["v"],  1)
            self.assertEqual(store.load("cart")[0]["v"], 2)


# ---------------------------------------------------------------------------
# TrendReport
# ---------------------------------------------------------------------------

class TestTrendReport(unittest.TestCase):
    def test_ascii_plot_non_empty(self):
        from benchmarks.regression import TrendReport
        entries = [{"median_fitness": -1.0 + i * 0.1} for i in range(10)]
        report  = TrendReport(name="xor", entries=entries)
        plot    = report.ascii_plot()
        self.assertIn("xor", plot)
        self.assertGreater(len(plot), 5)

    def test_ascii_plot_no_data(self):
        from benchmarks.regression import TrendReport
        report = TrendReport(name="xor", entries=[])
        plot   = report.ascii_plot()
        self.assertIn("no data", plot)

    def test_ascii_plot_constant_values(self):
        from benchmarks.regression import TrendReport
        entries = [{"median_fitness": -0.5}] * 5
        report  = TrendReport(name="xor", entries=entries)
        plot    = report.ascii_plot()
        self.assertIn("xor", plot)

    def test_trend_direction_symbol(self):
        from benchmarks.regression import TrendReport
        improving = TrendReport("x", [{"v": float(i)} for i in range(5)])
        self.assertIn("↑", improving.ascii_plot(metric="v"))
        declining = TrendReport("x", [{"v": float(5 - i)} for i in range(5)])
        self.assertIn("↓", declining.ascii_plot(metric="v"))


# ---------------------------------------------------------------------------
# BenchmarkReport exit codes
# ---------------------------------------------------------------------------

class TestBenchmarkReportExitCode(unittest.TestCase):
    def _report(self, severity_name: str):
        from benchmarks.regression import BenchmarkReport, RegressionSeverity
        sev = RegressionSeverity[severity_name.upper()]
        return BenchmarkReport(
            timestamp="t", commit="abc",
            benchmark_results=[], regressions=[],
            overall_severity=sev,
        )

    def test_exit_0_for_none(self):
        self.assertEqual(self._report("none").exit_code(), 0)

    def test_exit_0_for_minor(self):
        self.assertEqual(self._report("minor").exit_code(), 0)

    def test_exit_1_for_major(self):
        self.assertEqual(self._report("major").exit_code(), 1)

    def test_exit_2_for_critical(self):
        self.assertEqual(self._report("critical").exit_code(), 2)

    def test_to_dict_contains_required_fields(self):
        from benchmarks.regression import BenchmarkReport, RegressionSeverity
        report = BenchmarkReport(
            timestamp="2024-01-01T00:00:00",
            commit="abcdef",
            benchmark_results=[],
            regressions=[],
            overall_severity=RegressionSeverity.NONE,
        )
        d = report.to_dict()
        for key in ("timestamp", "commit", "overall_severity", "exit_code",
                    "benchmark_results", "regressions"):
            self.assertIn(key, d, f"Missing key: {key}")
        self.assertEqual(d["exit_code"], 0)

    def test_format_text_non_empty(self):
        from benchmarks.regression import BenchmarkReport, RegressionSeverity
        report = BenchmarkReport(
            timestamp="t", commit="abc",
            benchmark_results=[{"name": "xor", "example": "XOR",
                                 "summary": {"solved": 4, "n": 5,
                                             "median_fitness": -0.01}, "runs": []}],
            regressions=[],
            overall_severity=RegressionSeverity.NONE,
        )
        text = report.format_text()
        self.assertIn("NONE", text)
        self.assertIn("xor", text)


# ---------------------------------------------------------------------------
# CLI exit codes (via main())
# ---------------------------------------------------------------------------

class TestCLIExitCodes(unittest.TestCase):
    def _make_fake_suite(self, severity: str, tmp: str) -> Path:
        """Write minimal YAML + baseline such that a run produces *severity*."""
        # We won't actually run benchmarks — we test just the exit-code logic.
        return Path(tmp) / "benchmarks.yaml"

    def test_exit_code_from_benchmark_report(self):
        """Verify exit codes without invoking a real training run."""
        from benchmarks.regression import BenchmarkReport, RegressionSeverity

        for sev_name, expected_code in [("none", 0), ("minor", 0),
                                         ("major", 1), ("critical", 2)]:
            report = BenchmarkReport(
                timestamp="t", commit="x",
                benchmark_results=[], regressions=[],
                overall_severity=RegressionSeverity[sev_name.upper()],
            )
            self.assertEqual(report.exit_code(), expected_code, sev_name)


if __name__ == "__main__":
    unittest.main()
