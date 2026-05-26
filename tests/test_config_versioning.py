"""Tests for config versioning (config hash, compatibility check)."""
from __future__ import annotations
import unittest
import pytest

from yane.evolution.checkpoint import _config_hash, check_compatibility


@pytest.mark.ci
class TestConfigHash(unittest.TestCase):

    def test_hash_is_deterministic(self):
        cfg = {"n_inputs": 2, "n_outputs": 1, "pop_size": 100}
        h1 = _config_hash(cfg)
        h2 = _config_hash(cfg)
        self.assertEqual(h1, h2)

    def test_hash_changes_on_modification(self):
        cfg1 = {"n_inputs": 2, "n_outputs": 1}
        cfg2 = {"n_inputs": 3, "n_outputs": 1}
        self.assertNotEqual(_config_hash(cfg1), _config_hash(cfg2))

    def test_hash_is_hex(self):
        h = _config_hash({"a": 1})
        self.assertEqual(len(h), 16)
        int(h, 16)  # should not raise

    def test_compatibility_exact(self):
        cfg = {"n_inputs": 2, "n_outputs": 1}
        h = _config_hash(cfg)
        result = check_compatibility(h, cfg)
        self.assertEqual(result, "EXACT")

    def test_metadata_contains_hash(self):
        import json
        from pathlib import Path
        import tempfile
        from yane import NeuroEvolution
        from yane.evolution.checkpoint import _metadata_for

        ne = NeuroEvolution()
        ne.set_population_size(10)
        ne.configure(2, 1)
        ne.set_max_iterations(10)
        def _eval(g): return 1.0
        ne.train(_eval)

        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            ne.save_checkpoint(f.name)
            meta_path = Path(f.name + ".json")
            self.assertTrue(meta_path.exists())
            meta = json.loads(meta_path.read_text())
            self.assertIn("config_hash", meta)
            self.assertEqual(len(meta["config_hash"]), 16)


if __name__ == "__main__":
    unittest.main()
