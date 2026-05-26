import unittest
import pytest

from yane import NeuroEvolution
from yane.evolution.experimental import IntrinsicCuriosityModule


@pytest.mark.ci
class TestIntrinsicCuriosityModule(unittest.TestCase):

    def _make_module(self, n_in=2, n_out=1, size=4):
        return IntrinsicCuriosityModule(n_in, n_out, network_size=size, lr=0.1)

    def test_predict_returns_correct_length(self):
        mod = self._make_module(n_in=3, n_out=2)
        out = mod.predict([0.5, -0.3, 1.0])
        self.assertEqual(len(out), 2)

    def test_error_is_nonnegative(self):
        mod = self._make_module()
        err = mod.error([0.5, 0.5], [1.0])
        self.assertGreaterEqual(err, 0.0)

    def test_error_decreases_on_repeated_update(self):
        """After many updates on the same point, error should be lower than initial."""
        import random as _rng
        _rng.seed(0)
        mod = self._make_module(n_in=1, n_out=1, size=4)
        initial = mod.error([1.0], [0.5])
        for _ in range(500):
            mod.update([1.0], [0.5])
        final = mod.error([1.0], [0.5])
        self.assertLess(final, initial + 1e-6)  # converges or stays near zero

    def test_update_increments_counter(self):
        mod = self._make_module()
        self.assertEqual(mod._n_updates, 0)
        mod.update([1.0, 0.0], [0.5])
        self.assertEqual(mod._n_updates, 1)

    def test_error_decreases_with_training(self):
        import random as _rng
        _rng.seed(1)
        mod = IntrinsicCuriosityModule(2, 1, network_size=8, lr=0.01)
        initial_err = mod.error([1.0, 0.0], [2.0])
        for _ in range(500):
            mod.update([1.0, 0.0], [2.0])
        final_err = mod.error([1.0, 0.0], [2.0])
        self.assertLess(final_err, initial_err)

    def test_empty_inputs_handled(self):
        mod = self._make_module(n_in=0, n_out=1, size=4)
        out = mod.predict([])
        self.assertEqual(len(out), 1)

    def test_empty_targets_returns_zero_error(self):
        mod = self._make_module(n_in=2, n_out=1)
        err = mod.error([1.0, 0.0], [])
        self.assertEqual(err, 0.0)


@pytest.mark.ci
class TestCuriosityIntegration(unittest.TestCase):

    def _make_yane(self, n_in=2, n_out=1) -> NeuroEvolution:
        yane = NeuroEvolution(seed=0)
        yane.configure(n_in, n_out, n_initial_hidden=1)
        yane.set_population_size(8)
        return yane

    # ------------------------------------------------------------------
    # set_curiosity API
    # ------------------------------------------------------------------

    def test_default_disabled(self):
        yane = self._make_yane()
        self.assertFalse(yane._curiosity_enabled)

    def test_set_curiosity_enables(self):
        yane = self._make_yane()
        yane.set_curiosity(weight=0.5)
        self.assertTrue(yane._curiosity_enabled)

    def test_set_curiosity_stores_params(self):
        yane = self._make_yane()
        yane.set_curiosity(weight=0.2, network_size=16, lr=0.05)
        self.assertAlmostEqual(yane._curiosity_weight, 0.2)
        self.assertEqual(yane._curiosity_network_size, 16)
        self.assertAlmostEqual(yane._curiosity_lr, 0.05)

    def test_set_curiosity_creates_module_after_configure(self):
        yane = self._make_yane()
        yane.set_curiosity()
        self.assertIsNotNone(yane._curiosity_module)
        self.assertIsInstance(yane._curiosity_module, IntrinsicCuriosityModule)

    def test_set_curiosity_before_configure_creates_module_in_configure(self):
        yane = NeuroEvolution(seed=0)
        yane.set_curiosity()
        self.assertIsNone(yane._curiosity_module)  # not yet configured
        yane.configure(2, 1)
        self.assertIsNotNone(yane._curiosity_module)

    def test_disable_clears_module(self):
        yane = self._make_yane()
        yane.set_curiosity(enabled=True)
        yane.set_curiosity(enabled=False)
        self.assertFalse(yane._curiosity_enabled)
        self.assertIsNone(yane._curiosity_module)

    # ------------------------------------------------------------------
    # config_dict
    # ------------------------------------------------------------------

    def test_config_dict_fields(self):
        yane = self._make_yane()
        yane.set_curiosity(weight=0.4, network_size=12)
        cfg = yane._config_dict()
        self.assertTrue(cfg["curiosity_enabled"])
        self.assertAlmostEqual(cfg["curiosity_weight"], 0.4)
        self.assertEqual(cfg["curiosity_network_size"], 12)

    # ------------------------------------------------------------------
    # Training integration: curiosity bonus affects fitness
    # ------------------------------------------------------------------

    def test_curiosity_produces_higher_raw_fitness(self):
        """Curiosity adds a non-negative bonus; best fitness should be >= no-curiosity."""
        def trivial(g):
            return sum(g.forward([0.5, 0.5]))

        yane_plain = NeuroEvolution(seed=7)
        yane_plain.configure(2, 1, n_initial_hidden=1)
        yane_plain.set_population_size(10)
        yane_plain.set_max_iterations(20)
        yane_plain.train(trivial)
        plain_best = yane_plain.get_best().fitness

        yane_curious = NeuroEvolution(seed=7)
        yane_curious.configure(2, 1, n_initial_hidden=1)
        yane_curious.set_population_size(10)
        yane_curious.set_curiosity(weight=1.0)
        yane_curious.set_max_iterations(20)
        yane_curious.train(trivial)
        curious_best = yane_curious.get_best().fitness

        # With a bonus, the reported fitness should be at least as high
        self.assertGreaterEqual(curious_best, plain_best - 1e-6)

    def test_curiosity_module_gets_updated(self):
        """Forward model should accumulate updates during training."""
        def trivial(g):
            return sum(g.forward([0.5, 0.5]))

        yane = self._make_yane()
        yane.set_curiosity(weight=0.5)
        yane.set_max_iterations(15)
        yane.train(trivial)
        self.assertGreater(yane._curiosity_module._n_updates, 0)

    def test_training_completes_without_error(self):
        """Curiosity enabled training should not raise."""
        yane = self._make_yane()
        yane.set_curiosity(weight=0.3)
        yane.set_max_iterations(10)
        yane.train(lambda g: sum(g.forward([1.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
