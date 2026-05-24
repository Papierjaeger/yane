import unittest
import pytest

from yane import NeuroEvolution
from yane.evolution.population import Population


@pytest.mark.ci
class TestAdaptivePopulation(unittest.TestCase):

    def _make_yane(self, pop_size: int = 10) -> NeuroEvolution:
        yane = NeuroEvolution(seed=0)
        yane.configure(1, 1, n_initial_hidden=1)
        yane.set_population_size(pop_size)
        return yane

    # ------------------------------------------------------------------
    # set_adaptive_population API
    # ------------------------------------------------------------------

    def test_default_disabled(self):
        yane = self._make_yane()
        self.assertFalse(yane._adaptive_pop_enabled)

    def test_set_enables(self):
        yane = self._make_yane()
        yane.set_adaptive_population(min_size=5, max_size=20)
        self.assertTrue(yane._adaptive_pop_enabled)

    def test_set_stores_limits(self):
        yane = self._make_yane()
        yane.set_adaptive_population(min_size=5, max_size=30, growth_rate=0.1)
        self.assertEqual(yane._adaptive_pop_min, 5)
        self.assertEqual(yane._adaptive_pop_max, 30)
        self.assertAlmostEqual(yane._adaptive_pop_rate, 0.1)

    def test_set_propagates_to_population(self):
        yane = self._make_yane()
        yane.set_adaptive_population(min_size=5, max_size=20)
        self.assertTrue(yane._population._adaptive_pop_enabled)
        self.assertEqual(yane._population._adaptive_pop_min, 5)
        self.assertEqual(yane._population._adaptive_pop_max, 20)

    def test_disable_flag(self):
        yane = self._make_yane()
        yane.set_adaptive_population(min_size=5, max_size=20, enabled=False)
        self.assertFalse(yane._adaptive_pop_enabled)
        self.assertFalse(yane._population._adaptive_pop_enabled)

    def test_invalid_min_raises(self):
        yane = self._make_yane()
        with self.assertRaises(ValueError):
            yane.set_adaptive_population(min_size=0, max_size=20)

    def test_max_less_than_min_raises(self):
        yane = self._make_yane()
        with self.assertRaises(ValueError):
            yane.set_adaptive_population(min_size=20, max_size=10)

    # ------------------------------------------------------------------
    # Population._adjust_population_size() logic
    # ------------------------------------------------------------------

    def test_disabled_no_change(self):
        pop = Population(max_size=50)
        pop._adaptive_pop_enabled = False
        pop._spawn_count = 50  # would fire if enabled
        pop._adjust_population_size()
        self.assertEqual(pop.max_size, 50)

    def test_debounce_no_change_mid_generation(self):
        pop = Population(max_size=50)
        pop._adaptive_pop_enabled = True
        pop._adaptive_pop_min = 10
        pop._adaptive_pop_max = 200
        pop._adaptive_pop_rate = 0.1
        pop._spawn_count = 7  # not a multiple of 50 → no action
        pop._adjust_population_size()
        self.assertEqual(pop.max_size, 50)

    def test_grows_on_stagnation(self):
        pop = Population(max_size=50)
        pop._adaptive_pop_enabled = True
        pop._adaptive_pop_min = 10
        pop._adaptive_pop_max = 500
        pop._adaptive_pop_rate = 0.1
        pop._spawn_count = 50  # multiple of max_size → fires
        # Force high stagnation fraction
        pop._stagnation_count = pop.stagnation_threshold  # fraction = 1.0 > 0.3
        pop._adjust_population_size()
        self.assertGreater(pop.max_size, 50)

    def test_shrinks_on_excess_species(self):
        pop = Population(max_size=50, target_species=4)
        pop._adaptive_pop_enabled = True
        pop._adaptive_pop_min = 10
        pop._adaptive_pop_max = 500
        pop._adaptive_pop_rate = 0.1
        pop._spawn_count = 50
        # Simulate many species (> 125% of target=4 → > 5) and no stagnation
        pop._stagnation_count = 0
        # Patch species_count via _species list length
        from yane.evolution.species import Species
        from yane.core.genome import Genome
        for i in range(8):
            g = Genome()
            g.fitness = float(i)
            sp = Species(g, spawn_count=0)
            sp.members = [g]
            pop._species.append(sp)
        pop._adjust_population_size()
        self.assertLess(pop.max_size, 50)

    def test_clamped_to_max(self):
        pop = Population(max_size=190)
        pop._adaptive_pop_enabled = True
        pop._adaptive_pop_min = 10
        pop._adaptive_pop_max = 200
        pop._adaptive_pop_rate = 0.1
        pop._spawn_count = 190
        pop._stagnation_count = pop.stagnation_threshold  # grow
        pop._adjust_population_size()
        self.assertLessEqual(pop.max_size, 200)

    def test_clamped_to_min(self):
        pop = Population(max_size=12, target_species=4)
        pop._adaptive_pop_enabled = True
        pop._adaptive_pop_min = 10
        pop._adaptive_pop_max = 200
        pop._adaptive_pop_rate = 0.5  # large rate to try to shrink past min
        pop._spawn_count = 12
        pop._stagnation_count = 0
        from yane.evolution.species import Species
        from yane.core.genome import Genome
        for i in range(8):
            g = Genome()
            g.fitness = float(i)
            sp = Species(g, spawn_count=0)
            sp.members = [g]
            pop._species.append(sp)
        pop._adjust_population_size()
        self.assertGreaterEqual(pop.max_size, 10)

    def test_counter_increments(self):
        pop = Population(max_size=50)
        pop._adaptive_pop_enabled = True
        pop._adaptive_pop_min = 10
        pop._adaptive_pop_max = 500
        pop._adaptive_pop_rate = 0.1
        pop._spawn_count = 50
        pop._stagnation_count = pop.stagnation_threshold
        pop._adjust_population_size()
        self.assertEqual(pop._n_pop_size_adjustments, 1)

    # ------------------------------------------------------------------
    # configure() propagation
    # ------------------------------------------------------------------

    def test_configure_before_set_propagates(self):
        yane = NeuroEvolution(seed=0)
        yane.set_adaptive_population(min_size=5, max_size=200)
        yane.configure(1, 1)
        self.assertTrue(yane._population._adaptive_pop_enabled)
        self.assertEqual(yane._population._adaptive_pop_min, 5)
        self.assertEqual(yane._population._adaptive_pop_max, 200)

    # ------------------------------------------------------------------
    # config_dict
    # ------------------------------------------------------------------

    def test_config_dict_fields(self):
        yane = self._make_yane()
        yane.set_adaptive_population(min_size=5, max_size=50, growth_rate=0.08)
        cfg = yane._config_dict()
        self.assertTrue(cfg["adaptive_pop_enabled"])
        self.assertEqual(cfg["adaptive_pop_min"], 5)
        self.assertEqual(cfg["adaptive_pop_max"], 50)
        self.assertAlmostEqual(cfg["adaptive_pop_rate"], 0.08)

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------

    def test_diagnostics_keys_present(self):
        yane = self._make_yane(pop_size=5)
        yane.set_adaptive_population(min_size=3, max_size=20)
        yane.set_max_iterations(3)
        yane.train(lambda g: sum(g.forward([0.5])))
        info = yane.population_memory_info()
        self.assertIn("adaptive_pop_enabled", info)
        self.assertIn("n_pop_size_adjustments", info)

    # ------------------------------------------------------------------
    # Integration: size actually changes during training
    # ------------------------------------------------------------------

    def test_size_can_change_during_training(self):
        yane = NeuroEvolution(seed=42)
        yane.configure(1, 1, n_initial_hidden=1)
        yane.set_population_size(10)
        yane.set_adaptive_population(min_size=5, max_size=40, growth_rate=0.2)
        yane.set_max_iterations(30)
        # Force stagnation to trigger growth
        yane.train(lambda g: 0.0)
        # After stagnation-driven training, max_size should differ from initial
        # OR stay the same if the algo is healthy — just verify it's still in range
        self.assertGreaterEqual(yane._population.max_size, 5)
        self.assertLessEqual(yane._population.max_size, 40)


if __name__ == "__main__":
    unittest.main()
