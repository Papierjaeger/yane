import unittest
from yane.evolution.population import Population
from yane.core.genome import Genome


class TestPopulation(unittest.TestCase):

    def _make_population(self, max_size=10):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane._population.max_size = max_size
        return yane._population

    def test_initial_size(self):
        pop = self._make_population()
        self.assertEqual(pop.size, 1)
        self.assertEqual(pop.unevaluated_count, 1)
        self.assertEqual(pop.evaluated_count, 0)

    def test_submit_moves_genome(self):
        pop = self._make_population()
        g = pop.select_for_evaluation()
        pop.submit(g, -1.0)
        self.assertEqual(pop.evaluated_count, 1)
        self.assertEqual(pop.unevaluated_count, 0)

    def test_double_submit_raises_error(self):
        pop = self._make_population()
        g = pop.select_for_evaluation()
        pop.submit(g, -1.0)
        with self.assertRaises(ValueError):
            pop.submit(g, -0.5)   # second submit must raise
        self.assertEqual(pop.evaluated_count, 1)

    def test_population_stays_bounded(self):
        pop = self._make_population(max_size=5)
        for i in range(20):
            g = pop.select_for_evaluation()
            pop.submit(g, float(-i))
            self.assertLessEqual(pop.size, 6,  # max_size + 1 unevaluated at most
                msg=f"Population exceeded max_size at iteration {i}: {pop.size}")

    def test_get_best_returns_highest_fitness(self):
        pop = self._make_population(max_size=10)
        best_fitness = -999.0
        for i in range(5):
            g = pop.select_for_evaluation()
            f = float(-i)
            if f > best_fitness:
                best_fitness = f
            pop.submit(g, f)
        self.assertAlmostEqual(pop.get_best().fitness, best_fitness)

    def test_get_best_raises_before_evaluation(self):
        pop = self._make_population()
        with self.assertRaises(RuntimeError):
            pop.get_best()

    def test_shrink_to(self):
        pop = self._make_population(max_size=20)
        for i in range(10):
            g = pop.select_for_evaluation()
            pop.submit(g, float(-i))
        pop.shrink_to(3)
        self.assertEqual(len(pop._evaluated), 3)

    def test_shrink_keeps_best(self):
        pop = self._make_population(max_size=20)
        for i in range(10):
            g = pop.select_for_evaluation()
            pop.submit(g, float(i))
        pop.shrink_to(3)
        fitnesses = [g.fitness for g in pop._evaluated]
        self.assertEqual(sorted(fitnesses, reverse=True), fitnesses)

    def test_submit_records_eval_time_and_efficiency_score(self):
        pop = self._make_population(max_size=10)
        g1 = pop.select_for_evaluation()
        pop.submit(g1, 1.0, elapsed_ms=10.0)
        g2 = pop.select_for_evaluation()
        pop.submit(g2, 1.0, elapsed_ms=30.0)
        self.assertEqual(g1.eval_time_ms, 10.0)
        self.assertEqual(g2.eval_time_ms, 30.0)
        self.assertAlmostEqual(g1.efficiency_score, 1.0)
        self.assertAlmostEqual(g2.efficiency_score, 0.0)

    def test_efficiency_weight_fades_during_stagnation(self):
        pop = self._make_population(max_size=10)
        self.assertAlmostEqual(pop.efficiency_weight, 0.5)
        pop._stagnation_count = pop.stagnation_threshold
        self.assertAlmostEqual(pop.efficiency_weight, 0.0)

    def test_efficiency_affects_selection_score_not_raw_fitness(self):
        pop = self._make_population(max_size=10)
        g1 = pop.select_for_evaluation()
        pop.submit(g1, 1.0, elapsed_ms=10.0)
        g2 = pop.select_for_evaluation()
        pop.submit(g2, 1.0, elapsed_ms=30.0)
        pop._unevaluated.clear()
        pop._spawn_offspring()
        self.assertEqual(g1.fitness, 1.0)
        self.assertEqual(g2.fitness, 1.0)
        self.assertGreater(g1.selection_score, g2.selection_score)


    def test_offspring_counters_start_at_zero(self):
        pop = self._make_population(max_size=10)
        self.assertEqual(pop._n_crossover, 0)
        self.assertEqual(pop._n_mutation_only, 0)
        self.assertEqual(pop._n_diversity_injection, 0)

    def test_total_submitted_increments(self):
        pop = self._make_population(max_size=10)
        self.assertEqual(pop._total_submitted, 0)
        g = pop.select_for_evaluation()
        pop.submit(g, 1.0)
        self.assertEqual(pop._total_submitted, 1)
        g2 = pop.select_for_evaluation()
        pop.submit(g2, 2.0)
        self.assertEqual(pop._total_submitted, 2)

    def test_best_topology_history_records_improvement(self):
        pop = self._make_population(max_size=10)
        self.assertEqual(len(pop._best_topology_history), 0)
        g1 = pop.select_for_evaluation()
        pop.submit(g1, 1.0)
        self.assertEqual(len(pop._best_topology_history), 1)
        entry = pop._best_topology_history[0]
        self.assertEqual(len(entry), 4)  # (total_submitted, n_nodes, n_conn, fitness)
        self.assertAlmostEqual(entry[3], 1.0)

    def test_best_topology_history_only_grows_on_improvement(self):
        pop = self._make_population(max_size=10)
        g1 = pop.select_for_evaluation()
        pop.submit(g1, 5.0)
        prev_len = len(pop._best_topology_history)
        g2 = pop.select_for_evaluation()
        pop.submit(g2, 3.0)  # worse — no new entry
        self.assertEqual(len(pop._best_topology_history), prev_len)
        g3 = pop.select_for_evaluation()
        pop.submit(g3, 7.0)  # better — new entry
        self.assertEqual(len(pop._best_topology_history), prev_len + 1)

    def test_mutation_only_counter_increments(self):
        pop = self._make_population(max_size=10)
        # Evaluate enough genomes to trigger spawn
        for i in range(5):
            g = pop.select_for_evaluation()
            pop.submit(g, float(i))
        before = pop._n_mutation_only + pop._n_crossover + pop._n_diversity_injection
        # Force a spawn cycle
        pop._unevaluated.clear()
        pop._spawn_offspring()
        after = pop._n_mutation_only + pop._n_crossover + pop._n_diversity_injection
        # At least one offspring was created
        self.assertGreater(after, before)

    def test_elite_count_default_is_one(self):
        pop = self._make_population(max_size=10)
        self.assertEqual(pop.elite_count, 1)
        self.assertEqual(pop.species_elite_count, 1)

    def test_global_best_survives_many_spawn_cycles(self):
        pop = self._make_population(max_size=5)
        # Seed the population
        for i in range(5):
            g = pop.select_for_evaluation()
            pop.submit(g, float(i))
        best = pop.get_best()
        best_fitness = best.fitness
        # Run many spawn/prune cycles
        for _ in range(30):
            g = pop.select_for_evaluation()
            pop.submit(g, -999.0)   # always worse than best
        # Original best must still be present
        self.assertIn(best, pop._evaluated)
        self.assertAlmostEqual(best.fitness, best_fitness)

    def test_elite_count_zero_disables_global_protection(self):
        pop = self._make_population(max_size=3)
        pop.elite_count = 0
        pop.species_elite_count = 0
        for i in range(3):
            g = pop.select_for_evaluation()
            pop.submit(g, float(i))
        # With elite_count=0 the best can be pruned when the population is over limit
        # Just verify it doesn't crash and stays bounded
        for _ in range(10):
            g = pop.select_for_evaluation()
            pop.submit(g, -1.0)
        self.assertLessEqual(pop.size, pop.max_size + 1)

    def test_elite_count_three_protects_top_three(self):
        pop = self._make_population(max_size=10)
        pop.elite_count = 3
        pop.species_elite_count = 0
        fitnesses = [10.0, 8.0, 6.0, 4.0, 2.0]
        genomes = []
        for f in fitnesses:
            g = pop.select_for_evaluation()
            pop.submit(g, f)
            genomes.append((g, f))
        # Force many prune cycles with bad offspring
        for _ in range(20):
            g = pop.select_for_evaluation()
            pop.submit(g, -999.0)
        evaluated_fitnesses = sorted([g.fitness for g in pop._evaluated], reverse=True)
        # Top-3 fitnesses must still be present
        for f in [10.0, 8.0, 6.0]:
            self.assertIn(f, evaluated_fitnesses,
                msg=f"Elite fitness {f} should still be in population")

    def test_species_elite_protects_single_member_species(self):
        pop = self._make_population(max_size=5)
        pop.species_elite_count = 1
        # Submit enough genomes to fill pop
        for i in range(5):
            g = pop.select_for_evaluation()
            pop.submit(g, float(i))
        # Manually inject a species with one member
        from yane.evolution.population import Species
        unique_genome = pop._evaluated[-1]
        sp = Species(unique_genome)
        sp.members = [unique_genome]
        pop._species = [sp]
        # Now prune — the single-member champion should be protected
        original_size = len(pop._evaluated)
        # Add a bad genome to force pruning
        g = pop.select_for_evaluation()
        pop.submit(g, -999.0)
        self.assertIn(unique_genome, pop._evaluated,
            "Single-member species champion must be protected from pruning")

    def test_diversity_injection_counter_increments(self):
        pop = self._make_population(max_size=5)
        for i in range(3):
            g = pop.select_for_evaluation()
            pop.submit(g, float(i))
        before = pop._n_diversity_injection
        pop._inject_fresh_genome()
        self.assertEqual(pop._n_diversity_injection, before + 1)
        pop._inject_structural_diversity()
        self.assertEqual(pop._n_diversity_injection, before + 2)

    def test_jump_rate_counts_new_bests(self):
        pop = self._make_population(max_size=10)
        # Submit ascending fitnesses — every genome beats the previous best.
        for i in range(5):
            g = pop.select_for_evaluation()
            pop.submit(g, float(i))
        self.assertEqual(pop._n_new_best, 5)
        # Submit a worse genome — should NOT increment.
        g = pop.select_for_evaluation()
        pop.submit(g, -999.0)
        self.assertEqual(pop._n_new_best, 5)
        self.assertAlmostEqual(pop._total_submitted, 6)
        # jump_rate = 5/6
        expected_rate = 5 / 6
        import yane
        ne = yane.NeuroEvolution()
        ne.configure(2, 1)
        ne._population = pop
        info = ne.population_memory_info()
        self.assertAlmostEqual(info["jump_rate"], expected_rate, places=5)


if __name__ == "__main__":
    unittest.main()
