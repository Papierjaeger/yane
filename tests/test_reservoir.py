"""Tests für Evolutionary Reservoir Computing (evolution/reservoir.py).

Akzeptanzkriterien:
  1. Reservoir-State ist deterministisch bei gleichem Seed
  2. spectral_radius < 1 → Echo-State-Property garantiert
  3. Ridge-Readout löst XOR-nahen Task ohne Evolution
  4. Readout-Gewichte können evolviert werden (Mutation, copy)
  5. Checkpoint-Roundtrip erhält fixiertes Reservoir + Readout
"""
from __future__ import annotations

import math
import pickle
import tempfile
import unittest
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Acceptance criterion 1: Determinismus
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestDeterminism(unittest.TestCase):

    def test_same_seed_same_reservoir(self):
        """Two ReservoirGenomes with the same seed must have identical W."""
        from yane.evolution.reservoir import ReservoirGenome
        r1 = ReservoirGenome(n_inputs=2, n_reservoir=20, n_outputs=1, seed=42)
        r2 = ReservoirGenome(n_inputs=2, n_reservoir=20, n_outputs=1, seed=42)
        for i in range(20):
            for j in range(20):
                self.assertAlmostEqual(r1._W[i][j], r2._W[i][j])

    def test_different_seed_different_reservoir(self):
        """Different seeds should produce different W (very likely)."""
        from yane.evolution.reservoir import ReservoirGenome
        r1 = ReservoirGenome(n_inputs=2, n_reservoir=20, n_outputs=1, seed=1)
        r2 = ReservoirGenome(n_inputs=2, n_reservoir=20, n_outputs=1, seed=2)
        any_diff = any(
            abs(r1._W[i][j] - r2._W[i][j]) > 1e-9
            for i in range(20) for j in range(20)
        )
        self.assertTrue(any_diff, "Different seeds should produce different W")

    def test_same_inputs_same_state_sequence(self):
        """Same inputs + same seed → identical state sequence."""
        from yane.evolution.reservoir import ReservoirGenome
        r1 = ReservoirGenome(n_inputs=2, n_reservoir=10, n_outputs=1, seed=7)
        r2 = ReservoirGenome(n_inputs=2, n_reservoir=10, n_outputs=1, seed=7)
        inputs = [[float(t % 2), float((t + 1) % 2)] for t in range(20)]
        states1 = r1.collect_states(inputs, washout=0)
        states2 = r2.collect_states(inputs, washout=0)
        for t in range(len(states1)):
            for i in range(10):
                self.assertAlmostEqual(states1[t][i], states2[t][i], places=10)


# ---------------------------------------------------------------------------
# Acceptance criterion 2: Echo-State-Property
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestEchoStateProperty(unittest.TestCase):

    def test_spectral_radius_below_one(self):
        """Reservoir spectral radius must be < 1."""
        from yane.evolution.reservoir import ReservoirGenome
        r = ReservoirGenome(n_inputs=2, n_reservoir=30, n_outputs=1,
                            spectral_radius=0.9, seed=0)
        sr = r.actual_spectral_radius
        self.assertLess(sr, 1.0,
                        f"spectral_radius {sr:.4f} must be < 1 for Echo-State-Property")

    def test_spectral_radius_approximately_correct(self):
        """Spectral radius should be close to the requested value."""
        from yane.evolution.reservoir import ReservoirGenome
        for target_sr in [0.5, 0.9, 0.95]:
            r = ReservoirGenome(n_inputs=2, n_reservoir=50, n_outputs=1,
                                spectral_radius=target_sr, seed=0)
            actual = r.actual_spectral_radius
            self.assertAlmostEqual(actual, target_sr, delta=0.15,
                                   msg=f"target={target_sr} got {actual:.3f}")

    def test_state_bounded_after_many_steps(self):
        """With spectral_radius < 1, reservoir states should remain bounded."""
        from yane.evolution.reservoir import ReservoirGenome
        import math
        r = ReservoirGenome(n_inputs=1, n_reservoir=20, n_outputs=1,
                            spectral_radius=0.9, seed=0)
        for _ in range(1000):
            r.forward([1.0])
        for v in r._state:
            self.assertFalse(math.isnan(v))
            self.assertFalse(math.isinf(v))
            self.assertLess(abs(v), 100.0, "State should be bounded after many steps")

    def test_reset_clears_state(self):
        """reset() must set all reservoir states to zero."""
        from yane.evolution.reservoir import ReservoirGenome
        r = ReservoirGenome(n_inputs=1, n_reservoir=10, n_outputs=1, seed=0)
        r.forward([1.0])
        self.assertFalse(all(v == 0.0 for v in r._state))
        r.reset()
        self.assertTrue(all(v == 0.0 for v in r._state))


# ---------------------------------------------------------------------------
# Acceptance criterion 3: Ridge-Readout löst XOR-nahen Task
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestRidgeReadout(unittest.TestCase):

    def _xor_like_dataset(self):
        """Generate XOR-like training data (linear boundary challenge)."""
        inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] * 25
        targets = [[0.0], [1.0], [1.0], [0.0]] * 25
        return inputs, targets

    def test_ridge_readout_reduces_mse(self):
        """After Ridge training, MSE should be low for a simple task."""
        from yane.evolution.reservoir import ReservoirGenome, train_ridge_readout
        r = ReservoirGenome(n_inputs=2, n_reservoir=50, n_outputs=1,
                            spectral_radius=0.9, seed=42)
        inputs, targets = self._xor_like_dataset()
        result = train_ridge_readout(r, inputs, targets, lambda_ridge=1e-3, washout=10)
        self.assertLess(result.train_mse, 0.5,
                        f"Ridge MSE {result.train_mse:.4f} should be < 0.5 for XOR-like task")

    def test_ridge_result_has_n_samples(self):
        from yane.evolution.reservoir import ReservoirGenome, train_ridge_readout
        r = ReservoirGenome(n_inputs=2, n_reservoir=30, n_outputs=1, seed=0)
        inputs = [[float(i % 2), float((i + 1) % 2)] for i in range(50)]
        targets = [[float((i + 1) % 2)] for i in range(50)]
        result = train_ridge_readout(r, inputs, targets, washout=5)
        self.assertEqual(result.n_samples, 45)  # 50 - 5 washout

    def test_readout_weights_updated_after_ridge(self):
        """W_out should change after Ridge training."""
        from yane.evolution.reservoir import ReservoirGenome, train_ridge_readout
        r = ReservoirGenome(n_inputs=2, n_reservoir=20, n_outputs=1, seed=1)
        original_readout = list(r.readout_flat)
        inputs = [[float(i % 2), 0.5] for i in range(40)]
        targets = [[float(i % 2)] for i in range(40)]
        train_ridge_readout(r, inputs, targets, washout=5)
        self.assertFalse(r.readout_flat == original_readout,
                         "Readout weights must change after Ridge training")

    def test_ridge_fit_gives_correct_output_direction(self):
        """For a simple linear task, Ridge readout should produce increasing output."""
        from yane.evolution.reservoir import ReservoirGenome, train_ridge_readout
        r = ReservoirGenome(n_inputs=1, n_reservoir=50, n_outputs=1,
                            spectral_radius=0.8, seed=5)
        # Task: output ≈ input (linear mapping)
        inputs = [[float(i) / 20] for i in range(100)]
        targets = [[float(i) / 20] for i in range(100)]
        train_ridge_readout(r, inputs, targets, washout=10)
        r.reset()
        out_low = r.forward([0.1])[0]
        r.reset()
        out_high = r.forward([0.9])[0]
        # Higher input should give higher output (correct direction)
        self.assertGreater(out_high, out_low,
                           "Output should increase with input after linear Ridge training")


# ---------------------------------------------------------------------------
# Acceptance criterion 4: Readout-Evolution
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestReadoutEvolution(unittest.TestCase):

    def test_mutate_readout_changes_weights(self):
        from yane.evolution.reservoir import ReservoirGenome
        import random
        r = ReservoirGenome(n_inputs=2, n_reservoir=20, n_outputs=1, seed=0)
        original = list(r.readout_flat)
        r.mutate_readout(sigma=1.0, rng=random.Random(42))
        self.assertFalse(r.readout_flat == original)

    def test_copy_preserves_reservoir_and_readout(self):
        from yane.evolution.reservoir import ReservoirGenome, train_ridge_readout
        r = ReservoirGenome(n_inputs=2, n_reservoir=10, n_outputs=1, seed=0)
        inputs = [[float(i % 2), 0.5] for i in range(20)]
        targets = [[float(i % 2)] for i in range(20)]
        train_ridge_readout(r, inputs, targets, washout=2)
        rc = r.copy()
        # Reservoir must be identical
        for i in range(10):
            for j in range(10):
                self.assertAlmostEqual(rc._W[i][j], r._W[i][j])
        # Readout must be identical
        for a, b in zip(rc.readout_flat, r.readout_flat):
            self.assertAlmostEqual(a, b)

    def test_copy_readout_is_independent(self):
        from yane.evolution.reservoir import ReservoirGenome
        r = ReservoirGenome(n_inputs=2, n_reservoir=5, n_outputs=1, seed=0)
        r.readout_flat = [0.1, 0.2, 0.3, 0.4, 0.5]
        rc = r.copy()
        rc.readout_flat[0] = 999.0
        self.assertAlmostEqual(r.readout_flat[0], 0.1)


# ---------------------------------------------------------------------------
# Acceptance criterion 5: Checkpoint
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestCheckpoint(unittest.TestCase):

    def test_pickle_roundtrip(self):
        from yane.evolution.reservoir import ReservoirGenome
        r = ReservoirGenome(n_inputs=2, n_reservoir=15, n_outputs=1, seed=3)
        data = pickle.dumps(r)
        r2 = pickle.loads(data)
        # Reservoir must survive roundtrip
        for i in range(15):
            self.assertAlmostEqual(r._W[i][0], r2._W[i][0])
        self.assertEqual(r.n_reservoir, r2.n_reservoir)
        self.assertAlmostEqual(r.spectral_radius, r2.spectral_radius)

    def test_save_load_roundtrip(self):
        from yane.evolution.reservoir import ReservoirGenome
        r = ReservoirGenome(n_inputs=2, n_reservoir=10, n_outputs=1, seed=4)
        r.readout_flat = [float(i) * 0.1 for i in range(10)]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "r.pkl"
            r.save(path)
            r2 = ReservoirGenome.load(path)
        self.assertEqual(r2.n_reservoir, 10)
        for a, b in zip(r.readout_flat, r2.readout_flat):
            self.assertAlmostEqual(a, b)


# ---------------------------------------------------------------------------
# NeuroEvolution integration
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestNeuroEvolutionIntegration(unittest.TestCase):

    def test_configure_reservoir_returns_reservoir_genome(self):
        import yane
        from yane.evolution.reservoir import ReservoirGenome
        ne = yane.NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=5)
        r = ne.configure_reservoir(n_reservoir=20, spectral_radius=0.9)
        self.assertIsInstance(r, ReservoirGenome)
        self.assertEqual(r.n_reservoir, 20)

    def test_train_reservoir_returns_result(self):
        import yane
        from yane.evolution.reservoir import ReservoirTrainResult
        ne = yane.NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1)
        ne.configure_reservoir(n_reservoir=30, seed=0)
        inputs = [[float(i % 2), float((i + 1) % 2)] for i in range(50)]
        targets = [[float((i + 1) % 2)] for i in range(50)]
        result = ne.train_reservoir(inputs, targets, washout=5)
        self.assertIsInstance(result, ReservoirTrainResult)
        self.assertGreaterEqual(result.n_samples, 1)

    def test_configure_reservoir_without_configure_raises(self):
        import yane
        ne = yane.NeuroEvolution()
        with self.assertRaises(RuntimeError):
            ne.configure_reservoir(n_reservoir=10)

    def test_yane_exports(self):
        import yane
        self.assertTrue(hasattr(yane, "ReservoirGenome"))
        self.assertTrue(hasattr(yane, "ReservoirTrainResult"))
        self.assertTrue(hasattr(yane, "train_ridge_readout"))


if __name__ == "__main__":
    unittest.main()
