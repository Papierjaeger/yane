"""Tests for config versioning (config hash, compatibility check)."""
from __future__ import annotations
import unittest
import pytest

from yane.evolution.checkpoint import (
    CompatibilityLevel,
    _config_hash,
    check_compatibility,
    compatibility_report,
)


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

    def test_compatibility_diff_compatible(self):
        stored = {"n_inputs": 2, "n_outputs": 1, "seed": 1}
        current = {"n_inputs": 2, "n_outputs": 1, "seed": 2}
        report = compatibility_report(stored, current)
        self.assertEqual(report["level"], CompatibilityLevel.COMPATIBLE.value)
        self.assertEqual(report["diff"][0]["path"], "seed")

    def test_compatibility_diff_breaking(self):
        stored = {"n_inputs": 2, "n_outputs": 1}
        current = {"n_inputs": 3, "n_outputs": 1}
        report = compatibility_report(stored, current)
        self.assertEqual(report["level"], CompatibilityLevel.BREAKING.value)
        self.assertEqual(report["diff"][0]["level"], CompatibilityLevel.BREAKING.value)

    def test_check_compatibility_with_stored_config(self):
        stored = {"n_inputs": 2, "n_outputs": 1}
        current = {"n_inputs": 3, "n_outputs": 1}
        result = check_compatibility(_config_hash(stored), current, stored)
        self.assertEqual(result, "BREAKING")

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

    def test_payload_contains_config_hash(self):
        import tempfile
        from yane import NeuroEvolution
        from yane.evolution import checkpoint

        ne = NeuroEvolution()
        ne.set_population_size(10)
        ne.configure(2, 1)
        ne.set_max_iterations(10)

        def _eval(g):
            return 1.0

        ne.train(_eval)
        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            ne.save_checkpoint(f.name)
            payload = checkpoint.read(f.name)
        self.assertEqual(payload["config_hash"], _config_hash(payload["config"]))

    def test_load_checkpoint_blocks_breaking_current_config(self):
        import tempfile
        from yane import NeuroEvolution

        ne = NeuroEvolution()
        ne.set_population_size(10)
        ne.configure(2, 1)
        ne.set_max_iterations(10)

        def _eval(g):
            return 1.0

        ne.train(_eval)
        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            ne.save_checkpoint(f.name)
            target = NeuroEvolution()
            target.configure(3, 1)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                target.load_checkpoint(f.name)

    def test_checkpoint_cli_diff(self):
        import io
        import tempfile
        from contextlib import redirect_stdout
        from pathlib import Path
        from yane import NeuroEvolution
        from yane.checkpoint import main

        def _save(path: Path, n_inputs: int) -> None:
            ne = NeuroEvolution()
            ne.set_population_size(10)
            ne.configure(n_inputs, 1)
            ne.set_max_iterations(10)
            ne.train(lambda _g: 1.0)
            ne.save_checkpoint(path)

        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "old.pkl"
            new = Path(tmp) / "new.pkl"
            _save(old, 2)
            _save(new, 3)
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(["--diff", str(old), str(new)])
        self.assertEqual(code, 1)
        self.assertIn('"level": "BREAKING"', out.getvalue())


if __name__ == "__main__":
    unittest.main()
