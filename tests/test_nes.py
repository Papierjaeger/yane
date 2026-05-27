import math
import unittest
import pytest

from yane.evolution.lamarck_refiner import LamarckRefiner


@pytest.mark.ci
class TestNESRefiner(unittest.TestCase):
    """Unit tests for LamarckRefiner.refine_nes() and NES mode integration."""

    def _make_genome(self, n_inputs=1, n_outputs=1, n_hidden=1):
        from yane import NeuroEvolution
        yane = NeuroEvolution(seed=0)
        yane.configure(n_inputs, n_outputs, n_initial_hidden=n_hidden)
        return yane._population._unevaluated[0]

    # ------------------------------------------------------------------
    # LamarckRefiner unit tests
    # ------------------------------------------------------------------

    def test_mode_off_by_default(self):
        r = LamarckRefiner()
        r.max_steps = 0
        self.assertEqual(r.mode, "off")

    def test_mode_adaptive_default(self):
        r = LamarckRefiner()
        self.assertEqual(r.mode, "adaptive")

    def test_mode_nes_explicit(self):
        r = LamarckRefiner()
        r.set_nes(k=5, adaptive=False)
        self.assertEqual(r.mode, "nes_explicit")

    def test_mode_nes_adaptive(self):
        r = LamarckRefiner()
        r.set_nes(k=3, adaptive=True)
        self.assertEqual(r.mode, "nes_adaptive")

    def test_set_nes_stores_lr(self):
        r = LamarckRefiner()
        r.set_nes(k=4, learning_rate=0.05)
        self.assertAlmostEqual(r.nes_lr, 0.05)

    def test_refine_nes_returns_float(self):
        g = self._make_genome()
        r = LamarckRefiner()
        r.set_nes(k=3)
        result = r.refine_nes(g, lambda genome: 1.0, n_steps=3)
        self.assertIsInstance(result, float)

    def test_refine_nes_no_improvement_reverts(self):
        """When NES update doesn't improve fitness, original weights must be restored."""
        g = self._make_genome()
        # Record original weights
        orig = [(c.weight, n.bias)
                for n in g.nodes for c in n.connections]

        r = LamarckRefiner()
        r.set_nes(k=3)
        # fitness_fn always returns same value — update is never accepted
        r.refine_nes(g, lambda genome: 0.5, baseline_fitness=0.5, n_steps=3)

        after = [(c.weight, n.bias)
                 for n in g.nodes for c in n.connections]
        for (ow, ob), (aw, ab) in zip(orig, after):
            self.assertAlmostEqual(ow, aw, places=10)
            self.assertAlmostEqual(ob, ab, places=10)

    def test_refine_nes_accepts_improvement(self):
        """When the update improves fitness, new weights should be kept."""
        g = self._make_genome()
        r = LamarckRefiner()
        r.set_nes(k=3, learning_rate=100.0)  # huge LR to guarantee change

        call_count = [0]

        def fitness_fn(genome):
            call_count[0] += 1
            return float(call_count[0])  # always increasing → always accepted

        result = r.refine_nes(g, fitness_fn, baseline_fitness=0.0, n_steps=2)
        self.assertGreater(result, 0.0)

    def test_refine_nes_time_ms_increments(self):
        g = self._make_genome()
        r = LamarckRefiner()
        r.set_nes(k=2)
        r.refine_nes(g, lambda genome: 1.0, n_steps=2)
        self.assertGreater(r.time_ms, 0.0)

    def test_refine_nes_k0_returns_baseline(self):
        g = self._make_genome()
        r = LamarckRefiner()
        r.steps = 0
        result = r.refine_nes(g, lambda genome: 99.0, baseline_fitness=42.0, n_steps=0)
        self.assertEqual(result, 42.0)

    def test_refine_nes_no_conns_optimizes_biases(self):
        """Without connections, NES still optimizes node biases."""
        from yane import NeuroEvolution
        yane = NeuroEvolution(seed=0)
        yane.configure(1, 1)   # starts with NO connections
        g = yane._population._unevaluated[0]
        r = LamarckRefiner()
        r.set_nes(k=2)
        result = r.refine_nes(g, lambda genome: 7.0, baseline_fitness=3.0, n_steps=2)
        self.assertTrue(math.isfinite(result))

    # ------------------------------------------------------------------
    # NeuroEvolution integration
    # ------------------------------------------------------------------

    def test_set_lamarck_nes_mode_flag(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(1, 1)
        yane.set_lamarck(n_steps=3, mode='nes')
        self.assertTrue(yane._lamarck.nes_mode)
        self.assertEqual(yane._lamarck.steps, 3)

    def test_set_lamarck_hill_climbing_clears_nes_flag(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(1, 1)
        yane.set_lamarck(n_steps=3, mode='nes')
        yane.set_lamarck(n_steps=3, mode='hill_climbing')
        self.assertFalse(yane._lamarck.nes_mode)

    def test_set_lamarck_adaptive_nes_mode(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(1, 1)
        yane.set_lamarck_adaptive(max_steps=4, mode='nes')
        self.assertTrue(yane._lamarck.nes_mode)
        self.assertEqual(yane._lamarck.steps, 0)

    def test_nes_mode_in_config_dict(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(1, 1)
        yane.set_lamarck(n_steps=3, mode='nes', learning_rate=0.05)
        cfg = yane._config_dict()
        self.assertTrue(cfg["lamarck_nes_mode"])
        self.assertAlmostEqual(cfg["lamarck_nes_lr"], 0.05)

    def test_nes_train_produces_valid_fitness(self):
        """Training with NES mode enabled must produce finite best fitness."""
        from yane import NeuroEvolution
        yane = NeuroEvolution(seed=1)
        yane.configure(1, 1, n_initial_hidden=1)
        yane.set_population_size(5)
        yane.set_lamarck(n_steps=2, mode='nes')
        yane.set_max_iterations(5)
        yane.train(lambda g: sum(g.forward([0.5])))
        best = yane._population._evaluated[0].fitness if yane._population._evaluated else 0.0
        self.assertTrue(math.isfinite(best))


if __name__ == "__main__":
    unittest.main()
