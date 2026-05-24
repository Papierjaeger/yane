import pickle
import tempfile
import unittest
from pathlib import Path

from yane import NeuroEvolution
from yane.evolution import checkpoint


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


if __name__ == "__main__":
    unittest.main()
