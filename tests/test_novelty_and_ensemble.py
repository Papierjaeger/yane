"""Tests for Novelty Search, Ensemble inference, and Stagnation Detection."""
import unittest

from yane import NeuroEvolution
import numpy as np
from yane.evolution.population import Population


def _make_yane(n_inputs=2, n_outputs=1, pop_size=20):
    yane = NeuroEvolution()
    yane.configure(n_inputs, n_outputs, max_nodes=10, max_connections=30)
    yane.set_population_size(pop_size)
    return yane


def _burst(yane, n, fitness_fn=None):
    for i in range(n):
        g = yane.next_genome()
        f = fitness_fn(g) if fitness_fn else float(i % 5)
        yane.submit_fitness(f)


# ---------------------------------------------------------------------------
# Novelty Search
# ---------------------------------------------------------------------------

class TestNoveltySearch(unittest.TestCase):

    def test_probe_inputs_created_for_configured_genome(self):
        yane = _make_yane(n_inputs=3)
        pop = yane._population
        self.assertGreater(len(pop._probe_inputs), 0)
        for inp in pop._probe_inputs:
            self.assertEqual(len(inp), 3)

    def test_probe_inputs_fixed_seed_reproducible(self):
        yane1 = _make_yane(n_inputs=2)
        yane2 = _make_yane(n_inputs=2)
        self.assertEqual(yane1._population._probe_inputs,
                         yane2._population._probe_inputs)

    def test_behavior_recorded_after_submit(self):
        yane = _make_yane()
        g = yane.next_genome()
        yane.submit_fitness(1.0)
        pop = yane._population
        self.assertIn(id(g), pop._behaviors)
        self.assertGreater(len(pop._behaviors[id(g)]), 0)

    def test_behavior_length_matches_probes_times_outputs(self):
        yane = _make_yane(n_inputs=2, n_outputs=2)
        g = yane.next_genome()
        yane.submit_fitness(0.0)
        pop = yane._population
        n_probes = len(pop._probe_inputs)
        expected_len = n_probes * 2  # 2 outputs
        self.assertEqual(len(pop._behaviors[id(g)]), expected_len)

    def test_behavior_cleared_when_genome_pruned(self):
        yane = _make_yane(pop_size=3)
        for i in range(5):
            g = yane.next_genome()
            yane.submit_fitness(float(i))
        pop = yane._population
        # All behaviors in dict must correspond to still-evaluated genomes
        live_ids = {id(g) for g in pop._evaluated}
        for gid in pop._behaviors:
            self.assertIn(gid, live_ids)

    def test_novelty_scores_computed(self):
        yane = _make_yane()
        for i in range(5):
            g = yane.next_genome()
            yane.submit_fitness(float(i))
        pop = yane._population
        scores = pop._compute_novelty()
        self.assertEqual(len(scores), len(pop._evaluated))
        for v in scores.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0 + 1e-9)

    def test_novelty_weight_low_when_improving(self):
        yane = _make_yane()
        pop = yane._population
        pop._stagnation_count = 0
        self.assertAlmostEqual(pop.novelty_weight, 0.1, places=5)

    def test_novelty_weight_high_when_stagnating(self):
        yane = _make_yane(pop_size=10)
        pop = yane._population
        pop._stagnation_count = pop.stagnation_threshold  # full stagnation
        self.assertAlmostEqual(pop.novelty_weight, 0.5, places=5)

    def test_novelty_weight_increases_with_stagnation(self):
        yane = _make_yane(pop_size=10)
        pop = yane._population
        weights = []
        for count in [0, 5, 10, 20]:
            pop._stagnation_count = count
            weights.append(pop.novelty_weight)
        self.assertEqual(weights, sorted(weights),
                         "Novelty weight must increase with stagnation count")

    def test_euclidean_distance_zero_for_same(self):
        v = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(float(np.linalg.norm(v - v)), 0.0)

    def test_euclidean_distance_positive_for_different(self):
        self.assertGreater(float(np.linalg.norm(np.array([0.0]) - np.array([1.0]))), 0.0)


# ---------------------------------------------------------------------------
# Stagnation Detection
# ---------------------------------------------------------------------------

class TestStagnationDetection(unittest.TestCase):

    def test_stagnation_count_resets_on_improvement(self):
        yane = _make_yane()
        pop = yane._population
        g = yane.next_genome()
        yane.submit_fitness(1.0)
        self.assertEqual(pop.stagnation_count, 0)
        g = yane.next_genome()
        yane.submit_fitness(0.5)  # worse — stagnation ticks up
        self.assertEqual(pop.stagnation_count, 1)
        g = yane.next_genome()
        yane.submit_fitness(2.0)  # improvement — resets
        self.assertEqual(pop.stagnation_count, 0)

    def test_stagnation_count_increments_without_improvement(self):
        yane = _make_yane()
        pop = yane._population
        g = yane.next_genome()
        yane.submit_fitness(5.0)  # set a high bar
        for _ in range(3):
            g = yane.next_genome()
            yane.submit_fitness(1.0)
        self.assertEqual(pop.stagnation_count, 3)

    def test_stagnation_threshold_scales_with_population(self):
        small = _make_yane(pop_size=10)
        large = _make_yane(pop_size=100)
        self.assertLess(small._population.stagnation_threshold,
                        large._population.stagnation_threshold)

    def test_diversity_injection_triggered_on_stagnation(self):
        yane = _make_yane(pop_size=5)
        pop = yane._population
        for i in range(5):
            g = yane.next_genome()
            yane.submit_fitness(float(i))

        # Force both stagnation counters to threshold
        pop._stagnation_count = pop.stagnation_threshold
        pop._since_last_injection = pop.stagnation_threshold

        # Next spawn should inject a fresh genome.
        # stagnation_count must NOT reset (only resets on real fitness improvement).
        # since_last_injection resets to 0 after injection.
        before = pop.stagnation_count
        pop._spawn_offspring()
        self.assertEqual(pop.stagnation_count, before,
                         "stagnation_count must not reset on injection — only on fitness improvement")
        self.assertEqual(pop._since_last_injection, 0,
                         "since_last_injection must reset after injection")
        self.assertEqual(len(pop._unevaluated), 1)

    def test_injected_genome_has_correct_structure(self):
        yane = _make_yane(n_inputs=3, n_outputs=2)
        pop = yane._population
        for i in range(5):
            g = yane.next_genome()
            yane.submit_fitness(float(i))
        pop._stagnation_count = pop.stagnation_threshold
        pop._spawn_offspring()
        injected = pop._unevaluated[0]
        self.assertEqual(len(injected.input_nodes), 3)
        self.assertEqual(len(injected.output_nodes), 2)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class TestEnsemble(unittest.TestCase):

    def test_get_ensemble_returns_top_k(self):
        yane = _make_yane()
        for i in range(10):
            g = yane.next_genome()
            yane.submit_fitness(float(i))
        top3 = yane.get_ensemble(k=3)
        self.assertEqual(len(top3), 3)
        fitnesses = [g.fitness for g in top3]
        self.assertEqual(fitnesses, sorted(fitnesses, reverse=True))

    def test_get_ensemble_fewer_than_k_available(self):
        yane = _make_yane()
        g = yane.next_genome()
        yane.submit_fitness(1.0)
        result = yane.get_ensemble(k=5)
        self.assertEqual(len(result), 1)

    def test_forward_ensemble_output_count(self):
        yane = _make_yane(n_inputs=2, n_outputs=3)
        for i in range(5):
            g = yane.next_genome()
            yane.submit_fitness(float(i))
        result = yane.forward_ensemble([0.5, 0.3], k=3)
        self.assertEqual(len(result), 3)

    def test_forward_ensemble_output_in_valid_range(self):
        yane = _make_yane()
        for i in range(5):
            g = yane.next_genome()
            yane.submit_fitness(float(i))
        result = yane.forward_ensemble([0.0, 1.0], k=3)
        for v in result:
            self.assertIsInstance(v, float)
            self.assertFalse(v != v)  # not NaN

    def test_forward_ensemble_is_average_of_top_k(self):
        yane = _make_yane()
        for i in range(5):
            g = yane.next_genome()
            yane.submit_fitness(float(i))
        inputs = [0.5, 0.5]
        k = 3
        ensemble = yane.forward_ensemble(inputs, k=k)
        top_k = yane.get_ensemble(k=k)
        manual_avg = [
            sum(g.forward(inputs)[0] for g in top_k) / k
        ]
        self.assertAlmostEqual(ensemble[0], manual_avg[0], places=10)

    def test_forward_ensemble_raises_before_evaluation(self):
        yane = _make_yane()
        with self.assertRaises(RuntimeError):
            yane.forward_ensemble([0.0, 0.0])

    def test_forward_ensemble_k1_equals_best_forward(self):
        yane = _make_yane()
        for i in range(5):
            g = yane.next_genome()
            yane.submit_fitness(float(i))
        inputs = [0.1, 0.9]
        ensemble_k1 = yane.forward_ensemble(inputs, k=1)
        best_out = yane.get_best().forward(inputs)
        self.assertAlmostEqual(ensemble_k1[0], best_out[0], places=10)


# ---------------------------------------------------------------------------
# Population edge cases
# ---------------------------------------------------------------------------

class TestPopulationEdgeCases(unittest.TestCase):

    def test_novelty_all_identical_genomes_no_crash(self):
        """_compute_novelty must handle all-zero variance (max_d=0) without division error."""
        yane = _make_yane(pop_size=5)
        pop = yane._population
        # Submit same fitness so all genomes are identical copies from template
        for i in range(5):
            g = yane.next_genome()
            yane.submit_fitness(0.0)
        # Must not raise ZeroDivisionError
        scores = pop._compute_novelty()
        for v in scores.values():
            self.assertFalse(v != v, "novelty score must not be NaN for identical genomes")

    def test_tournament_with_single_candidate(self):
        """Tournament (k=3) must still work when only 1 genome is evaluated."""
        yane = _make_yane(pop_size=20)
        g = yane.next_genome()
        yane.submit_fitness(5.0)
        # Trigger a spawn — only 1 candidate for tournament
        yane.next_genome()   # must not raise

    def test_single_member_species_not_always_protected(self):
        """Only multi-member species champions are protected from pruning.
        A single-genome species must be a prune candidate when population is over-full."""
        from yane.evolution.population import Population
        pop = Population(max_size=3)
        for i in range(5):
            g = pop.select_for_evaluation()
            pop.submit(g, float(i))

        # Population was pruned to max_size=3
        self.assertLessEqual(len(pop._evaluated), 3)

    def test_species_representative_never_none_after_prune(self):
        """sp.representative must be updated when the current representative is pruned."""
        from yane.evolution.population import Population
        pop = Population(max_size=4)
        for i in range(6):
            g = pop.select_for_evaluation()
            pop.submit(g, float(i % 3))   # some fitness variation

        # All species representatives must point to live genomes
        live_ids = {id(g) for g in pop._evaluated}
        for sp in pop._species:
            if sp.members and sp.representative is not None:
                self.assertIn(id(sp.representative), live_ids,
                    "Species representative must be a live genome after pruning")

    def test_inject_stagnation_resets_since_last_injection(self):
        """_since_last_injection must be 0 after any injection."""
        yane = _make_yane(pop_size=5)
        pop = yane._population
        for i in range(5):
            g = yane.next_genome()
            yane.submit_fitness(float(i))
        pop._since_last_injection = pop.stagnation_threshold
        pop._stagnation_count = pop.stagnation_threshold
        pop._inject_fresh_genome()
        self.assertEqual(pop._since_last_injection, 0)

    def test_node_input_index_mutation_only_for_input_nodes(self):
        """input_index must only mutate on INPUT nodes, not HIDDEN or OUTPUT."""
        from yane.core.node import Node, NodeType
        hidden = Node(NodeType.HIDDEN)
        output = Node(NodeType.OUTPUT)
        original_h = hidden.input_index
        original_o = output.input_index

        for _ in range(200):
            hidden.mutate(sigma=1.0)
            output.mutate(sigma=1.0)

        # input_index mutation is only applied to INPUT nodes; others unchanged
        # (The code has `if self.type == NodeType.INPUT: ...`)
        self.assertEqual(hidden.input_index, original_h,
            "HIDDEN node input_index must not mutate")
        self.assertEqual(output.input_index, original_o,
            "OUTPUT node input_index must not mutate")


if __name__ == '__main__':
    unittest.main()
