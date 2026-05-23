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


if __name__ == "__main__":
    unittest.main()
