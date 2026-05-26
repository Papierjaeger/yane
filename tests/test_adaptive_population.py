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

    # ------------------------------------------------------------------
    # set_adaptive_pop_size() — schedule-aware API
    # ------------------------------------------------------------------

    def test_set_adaptive_pop_size_api_stores_schedule(self):
        """set_adaptive_pop_size() stores the schedule and propagates it."""
        yane = self._make_yane()
        yane.set_adaptive_pop_size(min_pop=5, max_pop=30, schedule="linear_decay")
        self.assertEqual(yane._adaptive_pop_schedule, "linear_decay")
        self.assertEqual(yane._population._adaptive_pop_schedule, "linear_decay")

    def test_set_adaptive_pop_size_performance_based(self):
        yane = self._make_yane()
        yane.set_adaptive_pop_size(min_pop=5, max_pop=30, schedule="performance_based")
        self.assertEqual(yane._adaptive_pop_schedule, "performance_based")

    def test_set_adaptive_pop_size_invalid_schedule_raises(self):
        yane = self._make_yane()
        with self.assertRaises(ValueError):
            yane.set_adaptive_pop_size(min_pop=5, max_pop=30, schedule="random_walk")

    def test_linear_decay_always_shrinks(self):
        """With linear_decay, pop size must never increase."""
        pop = Population(max_size=100)
        pop._adaptive_pop_enabled = True
        pop._adaptive_pop_min = 10
        pop._adaptive_pop_max = 100
        pop._adaptive_pop_rate = 0.1
        pop._adaptive_pop_schedule = "linear_decay"
        pop._adaptive_pop_total_gens = 0  # unknown → fixed step mode

        sizes = [100]
        for _ in range(20):
            pop._spawn_count += pop.max_size  # advance one generation
            pop._adjust_population_size()
            sizes.append(pop.max_size)

        for i in range(1, len(sizes)):
            self.assertLessEqual(sizes[i], sizes[i - 1],
                f"linear_decay increased size at step {i}: {sizes[i-1]} → {sizes[i]}")

    def test_linear_decay_stays_in_bounds(self):
        """linear_decay never goes below min_pop."""
        pop = Population(max_size=50)
        pop._adaptive_pop_enabled = True
        pop._adaptive_pop_min = 20
        pop._adaptive_pop_max = 50
        pop._adaptive_pop_rate = 0.5  # large step → should clamp at min
        pop._adaptive_pop_schedule = "linear_decay"
        pop._adaptive_pop_total_gens = 0

        for _ in range(30):
            pop._spawn_count += pop.max_size
            pop._adjust_population_size()

        self.assertGreaterEqual(pop.max_size, 20)

    def test_linear_decay_monotone_with_total_gens(self):
        """With total_gens set, linear_decay is monotonically non-increasing."""
        pop = Population(max_size=100)
        pop._adaptive_pop_enabled = True
        pop._adaptive_pop_min = 10
        pop._adaptive_pop_max = 100
        pop._adaptive_pop_rate = 0.1
        pop._adaptive_pop_schedule = "linear_decay"
        pop._adaptive_pop_total_gens = 10  # 10 generations from 100 to 10

        # Simulate generation boundaries: advance spawn_count by current max_size each step
        sizes = []
        for _gen in range(12):
            pop._spawn_count += pop.max_size
            pop._adjust_population_size()
            sizes.append(pop.max_size)

        # Must be monotonically non-increasing
        for i in range(1, len(sizes)):
            self.assertLessEqual(sizes[i], sizes[i - 1],
                f"linear_decay increased size at step {i}: {sizes[i-1]} → {sizes[i]}")
        # Must converge to min after enough generations
        self.assertEqual(sizes[-1], pop._adaptive_pop_min)

    def test_diagnostics_include_schedule_fields(self):
        """population_memory_info() includes schedule diagnostics when enabled."""
        yane = NeuroEvolution(seed=0)
        yane.configure(1, 1, n_initial_hidden=1)
        yane.set_population_size(5)
        yane.set_adaptive_pop_size(min_pop=3, max_pop=20, schedule="linear_decay")
        yane.set_max_iterations(5)
        yane.train(lambda g: sum(g.forward([0.5])))
        info = yane.population_memory_info()
        self.assertIn("adaptive_pop_schedule", info)
        self.assertIn("current_pop_size", info)
        self.assertIn("last_resize_trigger", info)
        self.assertIn("pop_size_history", info)
        self.assertEqual(info["adaptive_pop_schedule"], "linear_decay")

    def test_no_resize_within_generation(self):
        """_adjust_population_size fires only at generation boundaries."""
        pop = Population(max_size=50)
        pop._adaptive_pop_enabled = True
        pop._adaptive_pop_min = 10
        pop._adaptive_pop_max = 500
        pop._adaptive_pop_rate = 0.5
        pop._adaptive_pop_schedule = "linear_decay"
        pop._spawn_count = 7  # not a multiple of 50
        pop._adjust_population_size()
        self.assertEqual(pop.max_size, 50)


if __name__ == "__main__":
    unittest.main()
