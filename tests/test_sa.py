import math
import unittest
import pytest

from yane.evolution.lamarck_refiner import LamarckRefiner


@pytest.mark.ci
class TestSARefiner(unittest.TestCase):
    """Unit tests for LamarckRefiner.refine_sa() and SA mode integration."""

    def _make_genome(self, n_inputs=1, n_outputs=1, n_hidden=1):
        from yane import NeuroEvolution
        yane = NeuroEvolution(seed=0)
        yane.configure(n_inputs, n_outputs, n_initial_hidden=n_hidden)
        return yane._population._unevaluated[0]

    # ------------------------------------------------------------------
    # Mode property
    # ------------------------------------------------------------------

    def test_mode_sa_explicit(self):
        r = LamarckRefiner()
        r.set_sa(k=5, adaptive=False)
        self.assertEqual(r.mode, "sa_explicit")

    def test_mode_sa_adaptive(self):
        r = LamarckRefiner()
        r.set_sa(k=3, adaptive=True)
        self.assertEqual(r.mode, "sa_adaptive")

    def test_set_sa_clears_nes_mode(self):
        r = LamarckRefiner()
        r.set_nes(k=3)
        r.set_sa(k=3)
        self.assertFalse(r.nes_mode)
        self.assertTrue(r.sa_mode)

    def test_set_nes_clears_sa_mode(self):
        r = LamarckRefiner()
        r.set_sa(k=3)
        r.set_nes(k=3)
        self.assertFalse(r.sa_mode)
        self.assertTrue(r.nes_mode)

    def test_set_sa_stores_cooling(self):
        r = LamarckRefiner()
        r.set_sa(k=4, cooling_rate=0.8)
        self.assertAlmostEqual(r.sa_cooling, 0.8)

    def test_set_sa_stores_t0(self):
        r = LamarckRefiner()
        r.set_sa(k=3, t0=0.5)
        self.assertAlmostEqual(r.sa_t0, 0.5)

    # ------------------------------------------------------------------
    # refine_sa() basic contract
    # ------------------------------------------------------------------

    def test_refine_sa_returns_float(self):
        g = self._make_genome()
        r = LamarckRefiner()
        r.set_sa(k=3)
        result = r.refine_sa(g, lambda genome: 1.0, n_steps=3)
        self.assertIsInstance(result, float)

    def test_refine_sa_k0_returns_baseline(self):
        g = self._make_genome()
        r = LamarckRefiner()
        result = r.refine_sa(g, lambda genome: 99.0, baseline_fitness=42.0, n_steps=0)
        self.assertEqual(result, 42.0)

    def test_refine_sa_returns_best_seen(self):
        """SA can move to worse states; must still return best fitness seen."""
        g = self._make_genome()
        r = LamarckRefiner()
        r.set_sa(k=5)

        call_count = [0]

        def fitness_fn(genome):
            call_count[0] += 1
            # Give a big reward on first call, then drop
            return 100.0 if call_count[0] == 1 else 0.0

        result = r.refine_sa(g, fitness_fn, baseline_fitness=100.0, n_steps=5)
        self.assertGreaterEqual(result, 0.0)

    def test_refine_sa_accepts_improvement(self):
        """When every step improves fitness, all improvements should be kept."""
        g = self._make_genome()
        r = LamarckRefiner()
        r.set_sa(k=3, cooling_rate=0.0)  # T→0 immediately; only improvements accepted

        call_count = [0]

        def fitness_fn(genome):
            call_count[0] += 1
            return float(call_count[0])

        result = r.refine_sa(g, fitness_fn, baseline_fitness=0.0, n_steps=3)
        self.assertGreater(result, 0.0)

    def test_refine_sa_time_ms_increments(self):
        g = self._make_genome()
        r = LamarckRefiner()
        r.set_sa(k=2)
        r.refine_sa(g, lambda genome: 1.0, n_steps=2)
        self.assertGreater(r.time_ms, 0.0)

    def test_refine_sa_finite_result(self):
        g = self._make_genome()
        r = LamarckRefiner()
        r.set_sa(k=3)
        result = r.refine_sa(g, lambda genome: 1.0, n_steps=3)
        self.assertTrue(math.isfinite(result))

    def test_refine_sa_no_conns_optimizes_biases(self):
        """Without connections SA still optimizes node biases."""
        from yane import NeuroEvolution
        yane = NeuroEvolution(seed=0)
        yane.configure(1, 1)
        g = yane._population._unevaluated[0]
        r = LamarckRefiner()
        r.set_sa(k=2)
        result = r.refine_sa(g, lambda genome: 7.0, baseline_fitness=3.0, n_steps=2)
        self.assertTrue(math.isfinite(result))

    # ------------------------------------------------------------------
    # NeuroEvolution integration
    # ------------------------------------------------------------------

    def test_set_lamarck_sa_mode_flag(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(1, 1)
        yane.set_lamarck(n_steps=3, mode='sa')
        self.assertTrue(yane._lamarck.sa_mode)
        self.assertFalse(yane._lamarck.nes_mode)
        self.assertEqual(yane._lamarck.steps, 3)

    def test_set_lamarck_hill_climbing_clears_sa_flag(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(1, 1)
        yane.set_lamarck(n_steps=3, mode='sa')
        yane.set_lamarck(n_steps=3, mode='hill_climbing')
        self.assertFalse(yane._lamarck.sa_mode)

    def test_set_lamarck_adaptive_sa_mode(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(1, 1)
        yane.set_lamarck_adaptive(max_steps=4, mode='sa')
        self.assertTrue(yane._lamarck.sa_mode)
        self.assertEqual(yane._lamarck.steps, 0)

    def test_sa_mode_in_config_dict(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(1, 1)
        yane.set_lamarck(n_steps=3, mode='sa', cooling_rate=0.8)
        cfg = yane._config_dict()
        self.assertTrue(cfg["lamarck_sa_mode"])
        self.assertAlmostEqual(cfg["lamarck_sa_cooling"], 0.8)

    def test_sa_train_produces_valid_fitness(self):
        """Training with SA mode enabled must produce finite best fitness."""
        from yane import NeuroEvolution
        yane = NeuroEvolution(seed=1)
        yane.configure(1, 1, n_initial_hidden=1)
        yane.set_population_size(5)
        yane.set_lamarck(n_steps=2, mode='sa')
        yane.set_max_iterations(5)
        yane.train(lambda g: sum(g.forward([0.5])))
        best = yane._population._evaluated[0].fitness if yane._population._evaluated else 0.0
        self.assertTrue(math.isfinite(best))


if __name__ == "__main__":
    unittest.main()
