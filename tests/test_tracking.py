"""Tests for the experiment-tracking backend system."""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Mock backend — verifies the dispatch contract
# ---------------------------------------------------------------------------

class MockBackend:
    """Minimal TrackingBackend implementation for testing."""

    def __init__(self):
        self.inits: list[dict] = []
        self.configs: list[dict] = []
        self.metrics: list[tuple[dict, int]] = []
        self.artifacts: list[tuple[str, str]] = []
        self.finishes: int = 0

    def init(self, config: dict) -> None:
        self.inits.append(dict(config))

    def log_config(self, config: dict) -> None:
        self.configs.append(dict(config))

    def log_metrics(self, metrics: dict, step: int) -> None:
        self.metrics.append((dict(metrics), step))

    def log_artifact(self, path: str, artifact_type: str = "model") -> None:
        self.artifacts.append((path, artifact_type))

    def finish(self) -> None:
        self.finishes += 1


# ---------------------------------------------------------------------------
# Protocol + _scalar_metrics
# ---------------------------------------------------------------------------

class TestTrackingProtocol(unittest.TestCase):
    def test_mock_backend_satisfies_protocol(self):
        from yane.evolution.tracking import TrackingBackend
        backend = MockBackend()
        self.assertIsInstance(backend, TrackingBackend)

    def test_scalar_metrics_extracts_numbers_only(self):
        from yane.evolution.tracking import _scalar_metrics
        mem = {
            "max_fitness": 1.5,
            "species_count": 3,
            "labels": ["a", "b"],   # should be excluded
            "cells": {"x": 1},      # should be excluded
            "nan_val": float("nan"), # should be excluded
            "enabled": True,        # bool → int
        }
        result = _scalar_metrics(mem)
        self.assertIn("max_fitness", result)
        self.assertIn("species_count", result)
        self.assertIn("enabled", result)
        self.assertNotIn("labels", result)
        self.assertNotIn("cells", result)
        self.assertNotIn("nan_val", result)
        self.assertEqual(result["enabled"], 1)
        self.assertEqual(result["max_fitness"], 1.5)


# ---------------------------------------------------------------------------
# Integration via NeuroEvolution
# ---------------------------------------------------------------------------

class TestTrackingIntegration(unittest.TestCase):
    def _run_short(self, *backends):
        from yane import NeuroEvolution
        yane = NeuroEvolution(seed=1)
        yane.configure(2, 1, n_initial_hidden=0)
        yane.set_population_size(5)
        yane.set_max_iterations(15)
        for b in backends:
            yane.set_tracking_backend(b)
        yane.train(lambda g: sum(g.forward([0.5, 0.5])))
        return yane

    def test_init_called_once_per_run(self):
        b = MockBackend()
        self._run_short(b)
        self.assertEqual(len(b.inits), 1)

    def test_log_config_called_once(self):
        b = MockBackend()
        self._run_short(b)
        self.assertEqual(len(b.configs), 1)
        cfg = b.configs[0]
        self.assertIn("n_inputs", cfg)
        self.assertIn("population_size", cfg)

    def test_log_metrics_called_multiple_times(self):
        b = MockBackend()
        self._run_short(b)
        # At least one heartbeat should have fired (every 100 iters,
        # or iteration 0 is always logged)
        self.assertGreater(len(b.metrics), 0)

    def test_metrics_contain_key_fields(self):
        b = MockBackend()
        self._run_short(b)
        for metrics, step in b.metrics:
            self.assertIn("max_fitness", metrics)
            self.assertIsInstance(step, int)

    def test_finish_called_once(self):
        b = MockBackend()
        self._run_short(b)
        self.assertEqual(b.finishes, 1)

    def test_multiple_backends_all_receive_same_metrics(self):
        b1 = MockBackend()
        b2 = MockBackend()
        self._run_short(b1, b2)
        self.assertEqual(len(b1.inits), 1)
        self.assertEqual(len(b2.inits), 1)
        self.assertEqual(len(b1.metrics), len(b2.metrics))
        if b1.metrics:
            self.assertEqual(b1.metrics[-1][0], b2.metrics[-1][0])

    def test_clear_backends_with_no_args(self):
        from yane import NeuroEvolution
        b = MockBackend()
        yane = NeuroEvolution(seed=1)
        yane.configure(2, 1)
        yane.set_tracking_backend(b)
        self.assertEqual(len(yane._tracking_backends), 1)
        yane.set_tracking_backend()   # clear
        self.assertEqual(len(yane._tracking_backends), 0)

    def test_backend_error_does_not_abort_training(self):
        """A backend that raises should not propagate to training."""
        class BrokenBackend:
            def init(self, config): raise RuntimeError("boom")
            def log_config(self, config): raise RuntimeError("boom")
            def log_metrics(self, metrics, step): raise RuntimeError("boom")
            def log_artifact(self, path, artifact_type="model"): pass
            def finish(self): raise RuntimeError("boom")

        from yane import NeuroEvolution
        yane = NeuroEvolution(seed=1)
        yane.configure(2, 1)
        yane.set_population_size(5)
        yane.set_max_iterations(10)
        yane.set_tracking_backend(BrokenBackend())
        # Should complete without raising
        iters = yane.train(lambda g: sum(g.forward([0.5, 0.5])))
        self.assertGreater(iters, 0)

    def test_finish_not_called_on_backend_after_clear(self):
        from yane import NeuroEvolution
        b = MockBackend()
        yane = NeuroEvolution(seed=1)
        yane.configure(2, 1)
        yane.set_population_size(5)
        yane.set_max_iterations(10)
        yane.set_tracking_backend(b)
        yane.set_tracking_backend()  # clear before training
        yane.train(lambda g: sum(g.forward([0.5, 0.5])))
        self.assertEqual(b.finishes, 0)  # never registered for training


# ---------------------------------------------------------------------------
# WandbBackend (requires wandb; tested with disabled mode)
# ---------------------------------------------------------------------------

class TestWandbBackend(unittest.TestCase):
    def _has_wandb(self):
        try:
            import wandb  # noqa: F401
            return True
        except ImportError:
            return False

    def test_import_error_when_wandb_missing(self):
        from yane.evolution.tracking import WandbBackend
        backend = WandbBackend()
        # Temporarily hide wandb from import
        real_wandb = sys.modules.pop("wandb", None)
        try:
            with self.assertRaises(ImportError) as ctx:
                backend._import()
            self.assertIn("wandb", str(ctx.exception).lower())
        finally:
            if real_wandb is not None:
                sys.modules["wandb"] = real_wandb

    @unittest.skipUnless(True, "always run — uses mock")
    def test_wandb_backend_with_mock(self):
        """Test WandbBackend against a mocked wandb module."""
        from yane.evolution.tracking import WandbBackend

        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            backend = WandbBackend(project="test-proj", mode="disabled")
            backend.init({"n_inputs": 2})
            mock_wandb.init.assert_called_once()
            call_kwargs = mock_wandb.init.call_args
            self.assertEqual(call_kwargs.kwargs.get("project"), "test-proj")
            self.assertEqual(call_kwargs.kwargs.get("mode"), "disabled")

            backend.log_config({"n_inputs": 2, "pop": 50})
            mock_run.config.update.assert_called_once()

            backend.log_metrics({"max_fitness": 0.9, "species": 3}, step=5)
            mock_run.log.assert_called_once()
            logged = mock_run.log.call_args[0][0]
            self.assertIn("yane/max_fitness", logged)
            self.assertEqual(logged["yane/max_fitness"], 0.9)

            backend.finish()
            mock_run.finish.assert_called_once()


# ---------------------------------------------------------------------------
# MlflowBackend (requires mlflow; tested with mock)
# ---------------------------------------------------------------------------

class TestMlflowBackend(unittest.TestCase):
    def test_import_error_when_mlflow_missing(self):
        from yane.evolution.tracking import MlflowBackend
        backend = MlflowBackend()
        real_mlflow = sys.modules.pop("mlflow", None)
        try:
            with self.assertRaises(ImportError) as ctx:
                backend._import()
            self.assertIn("mlflow", str(ctx.exception).lower())
        finally:
            if real_mlflow is not None:
                sys.modules["mlflow"] = real_mlflow

    def test_mlflow_backend_with_mock(self):
        from yane.evolution.tracking import MlflowBackend

        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value = mock_run

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            backend = MlflowBackend(
                experiment_name="test-exp",
                run_name="run-1",
            )
            backend.init({"n_inputs": 2})
            mock_mlflow.set_experiment.assert_called_with("test-exp")
            mock_mlflow.start_run.assert_called_once()

            backend.log_config({"n_inputs": 2, "pop": 50})
            mock_mlflow.log_params.assert_called_once()
            params = mock_mlflow.log_params.call_args[0][0]
            self.assertEqual(params["n_inputs"], "2")

            backend.log_metrics({"max_fitness": 0.9}, step=3)
            mock_mlflow.log_metrics.assert_called_once()
            logged_metrics = mock_mlflow.log_metrics.call_args[0][0]
            self.assertIn("max_fitness", logged_metrics)

            backend.finish()
            mock_mlflow.end_run.assert_called_once()

    def test_mlflow_with_local_tracking(self):
        """MlflowBackend with a real local mlruns/ dir (skipped if mlflow absent)."""
        try:
            import mlflow  # noqa: F401
        except ImportError:
            self.skipTest("mlflow not installed")

        import tempfile
        from yane.evolution.tracking import MlflowBackend
        from yane import NeuroEvolution

        with tempfile.TemporaryDirectory() as tmp:
            uri = f"file://{tmp}"
            b = MlflowBackend(experiment_name="yane-test", tracking_uri=uri)
            yane = NeuroEvolution(seed=2)
            yane.configure(2, 1)
            yane.set_population_size(5)
            yane.set_max_iterations(10)
            yane.set_tracking_backend(b)
            yane.train(lambda g: sum(g.forward([0.5, 0.5])))
            # Verify at least one run was written
            import mlflow
            mlflow.set_tracking_uri(uri)
            runs = mlflow.search_runs(experiment_names=["yane-test"])
            self.assertGreater(len(runs), 0)


if __name__ == "__main__":
    unittest.main()
