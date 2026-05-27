"""Tests for structured logging (util/logger.py and train() integration)."""

import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path

from yane import NeuroEvolution
from yane.util import logger as logmod


def _dummy_fitness(genome):
    total = 0.0
    for node in genome.nodes:
        for conn in node.connections:
            if conn.enabled:
                total += conn.weight
    return total


class TestSetupLogging(unittest.TestCase):
    """Tests for setup_logging() and helpers."""

    def setUp(self):
        # Save original state
        self._orig_root = logmod.log_root
        self._orig_max = logmod.max_log_dirs
        self._tmp = Path(tempfile.mkdtemp())
        logmod.log_root = self._tmp / "logs"

    def tearDown(self):
        logmod.log_root = self._orig_root
        logmod.max_log_dirs = self._orig_max
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_setup_logging_creates_directory_structure(self):
        """setup_logging creates logs/<name>/<timestamp>/ with run.log."""
        run_dir = logmod.setup_logging("test_cat")
        self.assertTrue(run_dir.exists())
        self.assertIn("test_cat", str(run_dir))
        self.assertTrue((run_dir / "run.log").exists())

    def test_setup_logging_returns_absolute_path(self):
        """Returned path must be absolute."""
        run_dir = logmod.setup_logging("test_cat")
        self.assertTrue(run_dir.is_absolute())

    def test_write_json_creates_valid_json(self):
        """write_json writes parseable JSON."""
        run_dir = logmod.setup_logging("test_cat")
        data = {"key": "value", "list": [1, 2, 3], "nested": {"a": 1}}
        logmod.write_json(run_dir / "config.json", data)

        with open(run_dir / "config.json") as f:
            loaded = json.load(f)
        self.assertEqual(loaded, data)

    def test_write_json_is_atomic(self):
        """write_json should not leave .tmp files on success."""
        run_dir = logmod.setup_logging("test_cat")
        logmod.write_json(run_dir / "test.json", {"x": 1})
        self.assertFalse((run_dir / "test.json.tmp").exists())

    def test_write_csv_header_and_rows(self):
        """write_csv writes header on first call, appends rows."""
        run_dir = logmod.setup_logging("test_cat")
        csv_path = run_dir / "test.csv"
        logmod.write_csv(csv_path, "a,b,c", "1,2,3")
        logmod.write_csv(csv_path, "a,b,c", "4,5,6")

        lines = csv_path.read_text().strip().split("\n")
        self.assertEqual(lines, ["a,b,c", "1,2,3", "4,5,6"])

    def test_auto_cleanup_removes_old_runs(self):
        """When max_log_dirs is small, old runs are cleaned up."""
        logmod.max_log_dirs = 3
        for i in range(5):
            logmod.setup_logging("cleanup_test")

        cat_dir = logmod.log_root / "cleanup_test"
        dirs = sorted([d for d in cat_dir.iterdir() if d.is_dir()],
                       key=lambda d: d.name)
        self.assertLessEqual(len(dirs), 3,
                             f"Should have ≤3 dirs after cleanup, got {len(dirs)}")

    def test_get_logger_returns_same_instance(self):
        """get_logger() returns the same logger instance."""
        a = logmod.get_logger()
        b = logmod.get_logger()
        self.assertIs(a, b)

    def test_log_info_warning_error_dont_raise(self):
        """Convenience logging functions should not raise."""
        logmod.setup_logging("test_cat")
        logmod.log_info("info %d", 1)
        logmod.log_warning("warn %s", "x")
        logmod.log_error("error")


class TestTrainLogging(unittest.TestCase):
    """Tests that train() produces the expected log artefacts."""

    def setUp(self):
        self._orig_root = logmod.log_root
        self._tmp = Path(tempfile.mkdtemp())
        logmod.log_root = self._tmp / "logs"

    def tearDown(self):
        logmod.log_root = self._orig_root
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_train_creates_run_log(self):
        """train() creates run.log in the log directory."""
        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane.set_max_iterations(10)
        yane.train(_dummy_fitness, run_name="test_train")

        log_dir = yane._log_run_dir
        self.assertIsNotNone(log_dir)
        self.assertTrue(log_dir.exists())
        run_log = log_dir / "run.log"
        self.assertTrue(run_log.exists())
        self.assertGreater(run_log.stat().st_size, 0)

    def test_train_creates_config_json(self):
        """train() writes config.json with all settings."""
        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane.set_max_iterations(10)
        yane.train(_dummy_fitness, run_name="test_train")

        config_path = yane._log_run_dir / "config.json"
        self.assertTrue(config_path.exists())
        config = json.loads(config_path.read_text())
        self.assertIn("population_size", config)
        self.assertIn("max_iterations", config)
        self.assertIn("lamarck_mode", config)

    def test_train_creates_fitness_history_csv(self):
        """train() writes fitness_history.csv with header and data rows."""
        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane.set_max_iterations(50)
        yane.train(_dummy_fitness, run_name="test_train")

        csv_path = yane._log_run_dir / "fitness_history.csv"
        self.assertTrue(csv_path.exists())
        lines = csv_path.read_text().strip().split("\n")
        self.assertGreater(len(lines), 1, "CSV should have header + data rows")
        self.assertIn("iteration", lines[0])
        self.assertIn("best_fitness", lines[0])
        self.assertIn("iqr_fitness", lines[0])

    def test_train_saves_best_genome(self):
        """train() pickles the best genome."""
        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane.set_max_iterations(10)
        yane.train(_dummy_fitness, run_name="test_train")

        pkl_path = yane._log_run_dir / "best_genome.pkl"
        self.assertTrue(pkl_path.exists())
        loaded = pickle.loads(pkl_path.read_bytes())
        self.assertIsNotNone(loaded.fitness)

    def test_train_uses_run_name(self):
        """train() uses explicit run_name for the log category."""
        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane.set_max_iterations(5)
        yane.train(_dummy_fitness, run_name="custom_experiment")

        log_dir_str = str(yane._log_run_dir)
        self.assertIn("custom_experiment", log_dir_str)

    def test_train_derives_name_from_fitness_fn(self):
        """train() derives category name from fitness_fn.__name__."""
        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane.set_max_iterations(5)
        # _dummy_fitness has __name__ == '_dummy_fitness'
        yane.train(_dummy_fitness)
        log_dir_str = str(yane._log_run_dir)
        self.assertIn("_dummy_fitness", log_dir_str)

    def test_train_falls_back_to_training(self):
        """train() uses 'training' when fitness_fn has no __name__."""
        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane.set_max_iterations(5)
        yane.train(lambda g: 0.0)
        log_dir_str = str(yane._log_run_dir)
        self.assertIn("training", log_dir_str)


# ---------------------------------------------------------------------------
# Run report (Generationsreport / Run-Postmortem)
# ---------------------------------------------------------------------------

class TestRunReport(unittest.TestCase):
    """Tests for export_run_report() and set_report_autosave()."""
    _orig_log_root = logmod.log_root

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        logmod.log_root = Path(self._tmp)

    def tearDown(self):
        logmod.log_root = self._orig_log_root

    def _train_briefly(self, n_iter=50):
        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane.set_max_iterations(n_iter)
        yane.train(_dummy_fitness, run_name="report_test")
        return yane

    def test_export_html(self):
        yane = self._train_briefly()
        out = Path(self._tmp) / "report.html"
        iters = yane.train(_dummy_fitness, run_name="html_test")
        content = yane.export_run_report(
            str(out), fmt="html",
            stop_reason="target_reached",
            iterations=iters,
        )
        self.assertTrue(out.exists())
        self.assertIn("<svg", content)
        self.assertIn("Run Report", content)
        self.assertIn("Best Genome", content)
        self.assertIn("Configuration", content)

    def test_export_json(self):
        yane = self._train_briefly()
        out = Path(self._tmp) / "report.json"
        iters = yane.train(_dummy_fitness, run_name="json_test")
        content = yane.export_run_report(
            str(out), fmt="json",
            stop_reason="manual",
            iterations=iters,
        )
        self.assertTrue(out.exists())
        import json
        data = json.loads(content)
        self.assertIn("run_name", data)
        self.assertIn("best_fitness", data)
        self.assertIn("fitness_history", data)
        self.assertIn("config", data)
        self.assertIn("stop_reason", data)

    def test_export_markdown(self):
        yane = self._train_briefly()
        out = Path(self._tmp) / "report.md"
        iters = yane.train(_dummy_fitness, run_name="md_test")
        content = yane.export_run_report(
            str(out), fmt="md",
            stop_reason="target_reached",
            iterations=iters,
        )
        self.assertTrue(out.exists())
        self.assertIn("Run Report", content)
        self.assertIn("Best Genome", content)
        self.assertIn("Fitness Progress", content)

    def test_invalid_format_raises(self):
        yane = self._train_briefly()
        out = Path(self._tmp) / "bad.txt"
        with self.assertRaises(ValueError):
            yane.export_run_report(str(out), fmt="txt")

    def test_set_report_autosave(self):
        """set_report_autosave causes a report to be written after train()."""
        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane.set_max_iterations(30)
        yane.set_report_autosave("{name}_{date}_report.html")
        yane.train(_dummy_fitness, run_name="autosave_test")
        # The autosave template without path separators places the file in
        # the run log directory.
        log_dir = yane._log_run_dir
        self.assertIsNotNone(log_dir)
        # Find the report in the log directory.
        found = list(log_dir.glob("*report*"))
        self.assertGreaterEqual(len(found), 1,
            msg=f"Expected report in {log_dir}, found {list(log_dir.iterdir())}")

    def test_report_contains_best_genome_details(self):
        yane = self._train_briefly(60)
        out = Path(self._tmp) / "detail.html"
        iters = yane.train(_dummy_fitness, run_name="detail_test")
        content = yane.export_run_report(
            str(out), fmt="html",
            stop_reason="target_reached",
            iterations=iters,
        )
        # Should contain node and connection tables
        self.assertIn("<th>ID</th>", content)
        self.assertIn("<th>Weight</th>", content)
        best = yane.get_best()
        # Fitness is formatted with :.6f in the report
        self.assertIn(f"{best.fitness:.6f}", content)

    def test_report_valid_svg(self):
        """HTML report contains valid SVG with fitness data."""
        yane = self._train_briefly(80)
        out = Path(self._tmp) / "svg.html"
        iters = yane.train(_dummy_fitness, run_name="svg_test")
        content = yane.export_run_report(
            str(out), fmt="html",
            stop_reason="target_reached",
            iterations=iters,
        )
        # Must contain an SVG with polyline
        self.assertIn("<polyline", content)
        self.assertIn('stroke="#2196F3"', content)


if __name__ == "__main__":
    unittest.main()
