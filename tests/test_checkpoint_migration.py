import pickle
import tempfile
import unittest
from pathlib import Path

from yane import NeuroEvolution
from yane.evolution import checkpoint


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


if __name__ == "__main__":
    unittest.main()
