import json
import pickle
import tempfile
import unittest
from pathlib import Path

from yane import NeuroEvolution
from yane.evolution import checkpoint

FIXTURES = Path(__file__).parent / "fixtures"


def _trained_yane(**kwargs):
    yane = NeuroEvolution(**kwargs)
    yane.configure(2, 1)
    g = yane.next_genome()
    yane.submit_fitness(1.0)
    return yane


class TestCheckpointMigration(unittest.TestCase):
    def test_v1_payload_is_migrated(self):
        yane = NeuroEvolution()
        yane.configure(1, 1)
        payload = {
            "version": 1,
            "config": yane._config_dict(),
            "population": yane.population,
            "tracker": yane._tracker,
        }

        migrated = checkpoint.migrate(payload)

        self.assertEqual(migrated["version"], checkpoint.VERSION)
        self.assertIn("normalizer", migrated)
        self.assertIn("n_early_stopped", migrated)

    def test_write_creates_metadata_sidecar(self):
        yane = NeuroEvolution()
        yane.configure(1, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.pkl"
            checkpoint.write(path, {
                "version": checkpoint.VERSION,
                "config": yane._config_dict(),
                "population": yane.population,
                "tracker": yane._tracker,
            })

            self.assertTrue(path.exists())
            self.assertTrue(path.with_suffix(".pkl.json").exists())
            loaded = checkpoint.read(path)
            self.assertEqual(loaded["version"], checkpoint.VERSION)

    def test_read_accepts_v1_pickle(self):
        yane = NeuroEvolution()
        yane.configure(1, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.pkl"
            path.write_bytes(pickle.dumps({
                "version": 1,
                "config": yane._config_dict(),
                "population": yane.population,
                "tracker": yane._tracker,
            }))

            loaded = checkpoint.read(path)

            self.assertEqual(loaded["version"], checkpoint.VERSION)


class TestCheckpointFixtures(unittest.TestCase):
    """Regression tests using committed fixture files to catch migration regressions."""

    def test_fixture_v1_migrates_to_current(self):
        path = FIXTURES / "checkpoint_v1.pkl"
        loaded = checkpoint.read(path)
        self.assertEqual(loaded["version"], checkpoint.VERSION)
        self.assertIn("normalizer", loaded)
        self.assertIn("n_early_stopped", loaded)
        from yane.evolution.population import Population
        self.assertIsInstance(loaded["population"], Population)

    def test_fixture_v2_loads_without_migration(self):
        path = FIXTURES / "checkpoint_v2.pkl"
        loaded = checkpoint.read(path)
        self.assertEqual(loaded["version"], checkpoint.VERSION)
        from yane.evolution.population import Population
        self.assertIsInstance(loaded["population"], Population)

    def test_fixture_v2_has_metadata_sidecar(self):
        meta_path = FIXTURES / "checkpoint_v2.pkl.json"
        self.assertTrue(meta_path.exists(), "v2 fixture should have a .json sidecar")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertIn("version", meta)
        self.assertIn("created_at", meta)
        self.assertIn("requires_reattach", meta)


class TestCheckpointAdaptiveState(unittest.TestCase):
    """AdaptiveController and OperatorScheduler state survives checkpoint round-trips."""

    def test_adaptive_ctrl_disabled_round_trip(self):
        yane = _trained_yane()
        yane.set_adaptive_control(enabled=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.pkl"
            yane.save_checkpoint(path)
            yane2 = NeuroEvolution()
            yane2.load_checkpoint(path)
            self.assertFalse(yane2._adaptive_ctrl_enabled)

    def test_adaptive_ctrl_enabled_round_trip(self):
        yane = _trained_yane()
        yane.set_adaptive_control(enabled=True)
        # run one tick to build up tick_count
        ctrl = yane.get_adaptive_controller()
        ctrl.tick(yane.population)
        original_tick = ctrl._tick_count
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.pkl"
            yane.save_checkpoint(path)
            yane2 = NeuroEvolution()
            yane2.load_checkpoint(path)
            self.assertTrue(yane2._adaptive_ctrl_enabled)
            self.assertEqual(yane2.get_adaptive_controller()._tick_count, original_tick)

    def test_operator_scheduler_disabled_round_trip(self):
        yane = _trained_yane()
        yane.set_operator_scheduler(enabled=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.pkl"
            yane.save_checkpoint(path)
            yane2 = NeuroEvolution()
            yane2.load_checkpoint(path)
            self.assertFalse(yane2._operator_scheduler_enabled)

    def test_operator_scheduler_enabled_round_trip(self):
        yane = _trained_yane()
        yane.set_operator_scheduler(enabled=True)
        sched = yane.get_operator_scheduler()
        sched._tick_count = 42
        sched.global_weights["add_node"] = 2.0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.pkl"
            yane.save_checkpoint(path)
            yane2 = NeuroEvolution()
            yane2.load_checkpoint(path)
            self.assertTrue(yane2._operator_scheduler_enabled)
            s2 = yane2.get_operator_scheduler()
            self.assertEqual(s2._tick_count, 42)
            self.assertAlmostEqual(s2.global_weights["add_node"], 2.0)

    def test_checkpoint_without_adaptive_keys_loads_as_disabled(self):
        """Old checkpoints without adaptive_ctrl keys default to disabled."""
        yane = _trained_yane()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.pkl"
            payload = {
                "version": checkpoint.VERSION,
                "config": yane._config_dict(),
                "population": yane.population,
                "tracker": yane._tracker,
            }
            path.write_bytes(pickle.dumps(payload))
            yane2 = NeuroEvolution()
            yane2.load_checkpoint(path)
            self.assertFalse(yane2._adaptive_ctrl_enabled)
            self.assertFalse(yane2._operator_scheduler_enabled)

    def test_operator_scheduler_rewired_to_population_after_load(self):
        """After loading, the population's _operator_scheduler matches the restored object."""
        yane = _trained_yane()
        yane.set_operator_scheduler(enabled=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.pkl"
            yane.save_checkpoint(path)
            yane2 = NeuroEvolution()
            yane2.load_checkpoint(path)
            self.assertIs(yane2.population._operator_scheduler,
                          yane2.get_operator_scheduler())


class TestCheckpointPolicy(unittest.TestCase):

    def test_auto_checkpoint_interval_and_best_tracking(self):
        from yane.util import logger as logmod
        orig_root = logmod.log_root
        with tempfile.TemporaryDirectory() as tmp:
            logmod.log_root = Path(tmp) / "logs"
            try:
                yane = NeuroEvolution()
                yane.set_population_size(5)
                yane.configure(1, 1)
                yane.set_checkpoint_policy(
                    interval=2,
                    keep_best=True,
                    max_keep=2,
                    path_template="ckpt_{kind}_{iteration}.pkl",
                )
                yane.set_max_iterations(5)
                yane.train(lambda _g: 1.0)

                info = yane.population_memory_info()
                self.assertEqual(info["last_auto_checkpoint_iteration"], 4)
                self.assertLessEqual(info["rolling_checkpoint_count"], 2)
                best_path = yane.get_best_checkpoint_path()
                self.assertIsNotNone(best_path)
                self.assertTrue(Path(best_path).exists())
            finally:
                logmod.log_root = orig_root

    def test_auto_checkpoint_retention_removes_old_rollovers(self):
        from yane.util import logger as logmod
        orig_root = logmod.log_root
        with tempfile.TemporaryDirectory() as tmp:
            logmod.log_root = Path(tmp) / "logs"
            try:
                yane = NeuroEvolution()
                yane.set_population_size(5)
                yane.configure(1, 1)
                yane.set_checkpoint_policy(
                    interval=1,
                    keep_best=False,
                    max_keep=2,
                    path_template="roll_{iteration}.pkl",
                )
                yane.set_max_iterations(5)
                yane.train(lambda _g: 0.0)
                paths = [Path(p) for p in yane._checkpoint_paths]
                self.assertEqual(len(paths), 2)
                self.assertTrue(all(p.exists() for p in paths))
                self.assertFalse((yane._log_run_dir / "roll_1.pkl").exists())
            finally:
                logmod.log_root = orig_root


if __name__ == "__main__":
    unittest.main()
