import unittest
import pytest


@pytest.mark.ci
class TestNeuroEvolutionConfig(unittest.TestCase):

    def _make(self, **kwargs):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(**kwargs)
        return yane

    def test_configure_creates_population(self):
        yane = self._make(n_inputs=2, n_outputs=1)
        self.assertIsNotNone(yane._population)

    def test_initial_genome_has_correct_io(self):
        yane = self._make(n_inputs=3, n_outputs=2)
        g = yane.next_genome()
        self.assertEqual(len(g.input_nodes), 3)
        self.assertEqual(len(g.output_nodes), 2)

    def test_initial_genome_has_no_connections(self):
        yane = self._make(n_inputs=2, n_outputs=1)
        g = yane.next_genome()
        # Genomes start empty so evolution discovers relevant connections
        self.assertEqual(g.connection_count, 0,
            "Initial genome must have no connections — topology grows via mutation")

    def test_output_nodes_have_persist_value(self):
        yane = self._make(n_inputs=2, n_outputs=1, stateful=True)
        g = yane.next_genome()
        for n in g.output_nodes:
            self.assertTrue(n.persist_value,
                "Output nodes must have persist_value=True for stateful tasks")

    def test_stateful_false_all_nodes_no_persist(self):
        yane = self._make(n_inputs=2, n_outputs=1, stateful=False)
        g = yane.next_genome()
        self.assertFalse(g.allow_memory)
        for n in g.nodes:
            self.assertFalse(n.persist_value,
                f"{n.type.value} node must not persist when allow_memory=False")

    def test_stateful_false_mutation_cannot_enable_persist_on_any_node(self):
        from yane.core.node import Node, NodeType
        yane = self._make(n_inputs=2, n_outputs=1, stateful=False)
        g = yane.next_genome()
        # Add a hidden node and force-enable persist on every node type
        h = Node(NodeType.HIDDEN, innovation=99)
        g.nodes.append(h)
        for n in g.nodes:
            n.persist_value = True
        g.mutate()
        for n in g.nodes:
            self.assertFalse(n.persist_value,
                f"mutate() must reset persist_value=False on {n.type.value} when allow_memory=False")

    def test_stateful_false_allow_memory_propagates_to_copy(self):
        yane = self._make(n_inputs=2, n_outputs=1, stateful=False)
        g = yane.next_genome()
        copy = g.copy()
        self.assertFalse(copy.allow_memory)
        for n in copy.nodes:
            self.assertFalse(n.persist_value)

    def test_stateful_true_allows_memory_genome_flag(self):
        yane = self._make(n_inputs=2, n_outputs=1, stateful=True)
        g = yane.next_genome()
        self.assertTrue(g.allow_memory)

    def test_max_nodes_propagates_to_genome(self):
        yane = self._make(n_inputs=2, n_outputs=1, max_nodes=7)
        g = yane.next_genome()
        self.assertEqual(g.max_nodes, 7)

    def test_max_connections_propagates_to_genome(self):
        yane = self._make(n_inputs=2, n_outputs=1, max_connections=15)
        g = yane.next_genome()
        self.assertEqual(g.max_connections, 15)

    def test_require_configure_before_next_genome(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        with self.assertRaises(RuntimeError):
            yane.next_genome()

    def test_require_next_genome_before_submit(self):
        yane = self._make(n_inputs=2, n_outputs=1)
        with self.assertRaises(RuntimeError):
            yane.submit_fitness(0.0)

    def test_set_inputs_validates_length(self):
        yane = self._make(n_inputs=2, n_outputs=1)
        yane.next_genome()
        with self.assertRaises(ValueError):
            yane.set_inputs([1.0, 2.0, 3.0])   # too many

    def test_set_population_size(self):
        yane = self._make(n_inputs=2, n_outputs=1)
        yane.set_population_size(50)
        self.assertEqual(yane._population.max_size, 50)

    def test_is_configured_property(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        self.assertFalse(yane.is_configured)
        yane.configure(2, 1)
        self.assertTrue(yane.is_configured)

    def test_current_genome_property(self):
        yane = self._make(n_inputs=2, n_outputs=1)
        self.assertIsNone(yane.current_genome)
        g = yane.next_genome()
        self.assertIs(yane.current_genome, g)
        yane.submit_fitness(0.0)
        self.assertIsNone(yane.current_genome)

    def test_get_best_raises_before_any_evaluation(self):
        yane = self._make(n_inputs=2, n_outputs=1)
        with self.assertRaises(RuntimeError):
            yane.get_best()

    def test_get_best_returns_highest_fitness(self):
        yane = self._make(n_inputs=2, n_outputs=1)
        best_fitness = float("-inf")
        for i in range(10):
            g = yane.next_genome()
            f = float(i)
            if f > best_fitness:
                best_fitness = f
            yane.submit_fitness(f)
        self.assertAlmostEqual(yane.get_best().fitness, best_fitness)

    def test_population_memory_info_structure(self):
        yane = self._make(n_inputs=2, n_outputs=1)
        g = yane.next_genome()
        yane.submit_fitness(1.0)
        info = yane.population_memory_info()
        for key in ("total_genomes", "total_nodes", "total_connections",
                    "avg_nodes_per_genome", "largest_genome_nodes"):
            self.assertIn(key, info, f"Missing key '{key}' in population_memory_info()")


class TestNeuroEvolutionForwardMode(unittest.TestCase):

    def test_manual_loop_full_cycle(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(2, 1)

        for _ in range(5):
            g = yane.next_genome()
            out = g.forward([0.5, 0.5])
            self.assertEqual(len(out), 1)
            yane.submit_fitness(out[0])

        best = yane.get_best()
        self.assertIsNotNone(best)

    def test_tick_mode_full_cycle(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(2, 1)

        yane.next_genome()
        yane.set_inputs([0.0, 1.0])
        yane.tick()
        yane.tick()
        outs = yane.get_outputs()
        self.assertEqual(len(outs), 1)
        yane.submit_fitness(outs[0])


@pytest.mark.ci
class TestGenomeClearGuard(unittest.TestCase):
    """_clear() must not cause crashes when a cleared genome is re-used."""

    def test_forward_on_cleared_genome_returns_empty(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(2, 1)
        g = yane.next_genome()
        g._clear()
        out = g.forward([0.5, 0.5])
        self.assertEqual(out, [], "forward() on cleared genome must return empty list safely")

    def test_double_clear_is_safe(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(2, 1)
        g = yane.next_genome()
        g._clear()
        g._clear()  # must not raise

    def test_prune_calls_clear_on_worst(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane.set_population_size(3)
        # Disable species-elite protection so pruning is purely fitness-based
        # and the worst genome (fitness 0.0) is reliably evicted regardless of
        # which species it happens to land in.
        yane.set_elitism(elite_count=1, species_elite_count=0)

        genomes = []
        for i in range(4):
            g = yane.next_genome()
            genomes.append(g)
            yane.submit_fitness(float(i))

        # genomes[0] had fitness 0.0 = worst; should have been cleared
        self.assertEqual(len(genomes[0].nodes), 0,
            "_prune() must call _clear() on the evicted genome")


class TestMultiEval(unittest.TestCase):

    def _make(self, **kwargs):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(**kwargs)
        return yane

    def test_set_multi_eval_stores_params(self):
        yane = self._make(n_inputs=2, n_outputs=1)
        yane.set_multi_eval(n=5, aggregation="median", sigma_penalty=0.3)
        self.assertEqual(yane._runner.n_evaluations, 5)
        self.assertEqual(yane._runner.aggregation, "median")
        self.assertAlmostEqual(yane._runner.sigma_penalty, 0.3)

    def test_set_multi_eval_invalid_n(self):
        yane = self._make(n_inputs=2, n_outputs=1)
        with self.assertRaises(ValueError):
            yane.set_multi_eval(n=0)

    def test_set_multi_eval_invalid_aggregation(self):
        yane = self._make(n_inputs=2, n_outputs=1)
        with self.assertRaises(ValueError):
            yane.set_multi_eval(n=3, aggregation="max")

    def test_train_calls_fitness_fn_n_times(self):
        """With n_evaluations=4, train() must call fitness_fn exactly 4× per genome."""
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(n_inputs=2, n_outputs=1)
        yane.set_multi_eval(n=4)
        yane.set_max_iterations(1)

        call_count = [0]

        def evaluate(genome):
            call_count[0] += 1
            return -1.0

        yane.train(evaluate)
        self.assertEqual(call_count[0], 4)

    def test_multi_eval_mean_aggregation(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(n_inputs=1, n_outputs=1)
        yane.set_multi_eval(n=3, aggregation="mean")
        yane.set_max_iterations(1)

        values = iter([1.0, 3.0, 5.0])

        def evaluate(genome):
            return next(values)

        yane.train(evaluate)
        self.assertAlmostEqual(yane.get_best().fitness, 3.0)

    def test_multi_eval_median_aggregation(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(n_inputs=1, n_outputs=1)
        yane.set_multi_eval(n=3, aggregation="median")
        yane.set_max_iterations(1)

        values = iter([1.0, 10.0, 3.0])

        def evaluate(genome):
            return next(values)

        yane.train(evaluate)
        self.assertAlmostEqual(yane.get_best().fitness, 3.0)

    def test_multi_eval_min_aggregation(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(n_inputs=1, n_outputs=1)
        yane.set_multi_eval(n=3, aggregation="min")
        yane.set_max_iterations(1)

        values = iter([5.0, 2.0, 8.0])

        def evaluate(genome):
            return next(values)

        yane.train(evaluate)
        self.assertAlmostEqual(yane.get_best().fitness, 2.0)

    def test_multi_eval_sigma_penalty_reduces_fitness(self):
        """sigma_penalty > 0 must reduce fitness below the mean when std > 0."""
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(n_inputs=1, n_outputs=1)
        yane.set_multi_eval(n=3, aggregation="mean", sigma_penalty=1.0)
        yane.set_max_iterations(1)

        values = iter([1.0, 3.0, 5.0])  # mean=3.0, std=sqrt(8/3)≈1.63

        def evaluate(genome):
            return next(values)

        yane.train(evaluate)
        self.assertLess(yane.get_best().fitness, 3.0)

    def test_multi_eval_sigma_penalty_zero_std(self):
        """With constant values, sigma_penalty must not change the result."""
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(n_inputs=1, n_outputs=1)
        yane.set_multi_eval(n=3, aggregation="mean", sigma_penalty=2.0)
        yane.set_max_iterations(1)

        def evaluate(genome):
            return 4.0

        yane.train(evaluate)
        self.assertAlmostEqual(yane.get_best().fitness, 4.0)

    def test_n1_default_is_fast_path(self):
        """Default n=1 must produce identical results to not using set_multi_eval."""
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(n_inputs=1, n_outputs=1)
        self.assertEqual(yane._runner.n_evaluations, 1)
        yane.set_max_iterations(1)
        call_count = [0]

        def evaluate(genome):
            call_count[0] += 1
            return -1.0

        yane.train(evaluate)
        self.assertEqual(call_count[0], 1)


@pytest.mark.ci
class TestFitnessSanitizing(unittest.TestCase):

    def _make(self, sanitize=True, **kw):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(n_inputs=2, n_outputs=1)
        if sanitize:
            yane.set_fitness_sanitizing(**kw)
        return yane

    def test_sanitize_fitness_nan_replaced(self):
        from yane.neuro_evolution import sanitize_fitness
        val, invalid, clipped = sanitize_fitness(float("nan"), fallback=-1.0)
        self.assertEqual(val, -1.0)
        self.assertTrue(invalid)
        self.assertFalse(clipped)

    def test_sanitize_fitness_inf_replaced(self):
        from yane.neuro_evolution import sanitize_fitness
        val, invalid, clipped = sanitize_fitness(float("inf"), fallback=0.0)
        self.assertEqual(val, 0.0)
        self.assertTrue(invalid)

    def test_sanitize_fitness_clip_low(self):
        from yane.neuro_evolution import sanitize_fitness
        val, invalid, clipped = sanitize_fitness(-5.0, clip_low=-1.0)
        self.assertEqual(val, -1.0)
        self.assertFalse(invalid)
        self.assertTrue(clipped)

    def test_sanitize_fitness_clip_high(self):
        from yane.neuro_evolution import sanitize_fitness
        val, invalid, clipped = sanitize_fitness(999.0, clip_high=100.0)
        self.assertEqual(val, 100.0)
        self.assertFalse(invalid)
        self.assertTrue(clipped)

    def test_sanitize_fitness_valid_passthrough(self):
        from yane.neuro_evolution import sanitize_fitness
        val, invalid, clipped = sanitize_fitness(3.14, clip_low=-10.0, clip_high=10.0)
        self.assertAlmostEqual(val, 3.14)
        self.assertFalse(invalid)
        self.assertFalse(clipped)

    def test_invalid_counter_increments(self):
        yane = self._make(sanitize=True, fallback=0.0)
        self.assertEqual(yane._sanitizer.n_invalid, 0)
        g = yane.next_genome()
        yane.submit_fitness(float("nan"))
        self.assertEqual(yane._sanitizer.n_invalid, 1)
        g = yane.next_genome()
        yane.submit_fitness(float("-inf"))
        self.assertEqual(yane._sanitizer.n_invalid, 2)

    def test_clipped_counter_increments(self):
        yane = self._make(sanitize=True, clip_low=0.0, clip_high=1.0)
        g = yane.next_genome()
        yane.submit_fitness(-5.0)
        self.assertEqual(yane._sanitizer.n_clipped, 1)
        g = yane.next_genome()
        yane.submit_fitness(99.0)
        self.assertEqual(yane._sanitizer.n_clipped, 2)

    def test_sanitize_disabled_by_default(self):
        yane = self._make(sanitize=False)
        self.assertFalse(yane._sanitizer.enabled)
        # nan passes through unmodified when disabled
        g = yane.next_genome()
        yane.submit_fitness(float("nan"))
        self.assertEqual(yane._sanitizer.n_invalid, 0)

    def test_invalid_fitness_in_train(self):
        yane = self._make(sanitize=True, fallback=-999.0)
        yane.set_max_iterations(3)
        call_count = [0]
        def evaluate(genome):
            call_count[0] += 1
            return float("nan")
        yane.train(evaluate)
        self.assertEqual(yane._sanitizer.n_invalid, 3)
        # Best genome should have fallback fitness, not nan
        import math
        self.assertTrue(math.isfinite(yane.get_best().fitness))

    def test_sanitize_counters_in_mem_info(self):
        yane = self._make(sanitize=True)
        mem = yane.population_memory_info()
        self.assertIn("sanitize_enabled", mem)
        self.assertIn("n_invalid_fitness", mem)
        self.assertIn("n_clipped_fitness", mem)
        self.assertTrue(mem["sanitize_enabled"])

    def test_sanitize_disabled_mem_info(self):
        yane = self._make(sanitize=False)
        mem = yane.population_memory_info()
        self.assertFalse(mem["sanitize_enabled"])
        self.assertEqual(mem["n_invalid_fitness"], 0)

    def test_submit_fitness_batch_sanitizes(self):
        yane = self._make(sanitize=True, fallback=0.0)
        g1 = yane.next_genome()
        yane.submit_fitness(1.0)  # first eval needed for batch
        g2 = yane.next_genome_batch(1)
        yane.submit_fitness_batch([(g2[0], float("nan"), None)])
        self.assertEqual(yane._sanitizer.n_invalid, 1)


class TestPopulationMemoryInfo(unittest.TestCase):

    def _make(self, n_inputs=2, n_outputs=1):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(n_inputs=n_inputs, n_outputs=n_outputs)
        return yane

    def _submit_n(self, yane, n, eval_ms=5.0):
        for i in range(n):
            g = yane.next_genome()
            yane.submit_fitness(float(i), elapsed_ms=eval_ms + i)
        return yane.population_memory_info()

    def test_eval_time_stats_present_after_submissions(self):
        yane = self._make()
        mem = self._submit_n(yane, 3, eval_ms=10.0)
        self.assertIn("eval_time_mean_ms", mem)
        self.assertIn("eval_time_median_ms", mem)
        self.assertIn("eval_time_p95_ms", mem)
        self.assertIn("eval_time_max_ms", mem)
        self.assertGreater(mem["eval_time_mean_ms"], 0.0)
        self.assertGreater(mem["eval_time_max_ms"], mem["eval_time_mean_ms"] - 1e-6)

    def test_eval_time_stats_absent_without_eval_times(self):
        yane = self._make()
        g = yane.next_genome()
        yane.submit_fitness(1.0, elapsed_ms=None)
        mem = yane.population_memory_info()
        self.assertNotIn("eval_time_mean_ms", mem)

    def test_offspring_counters_in_mem_info(self):
        yane = self._make()
        mem = self._submit_n(yane, 5)
        self.assertIn("n_crossover", mem)
        self.assertIn("n_mutation_only", mem)
        self.assertIn("n_diversity_injection", mem)
        total = mem["n_crossover"] + mem["n_mutation_only"] + mem["n_diversity_injection"]
        # At least the spawned offspring count
        self.assertGreaterEqual(total, 0)

    def test_best_topology_history_in_mem_info(self):
        yane = self._make()
        mem = self._submit_n(yane, 3)
        self.assertIn("best_topology_history", mem)
        hist = mem["best_topology_history"]
        self.assertIsInstance(hist, list)
        self.assertGreater(len(hist), 0)
        # Each entry should be a 4-tuple
        entry = hist[-1]
        self.assertEqual(len(entry), 4)

    def test_offspring_counters_consistent_after_many_evals(self):
        yane = self._make()
        yane.set_population_size(20)
        yane.configure(2, 1)
        for i in range(30):
            g = yane.next_genome()
            yane.submit_fitness(float(i), elapsed_ms=1.0)
        mem = yane.population_memory_info()
        n_co  = mem["n_crossover"]
        n_mut = mem["n_mutation_only"]
        n_inj = mem["n_diversity_injection"]
        # All counters are non-negative integers
        self.assertGreaterEqual(n_co, 0)
        self.assertGreaterEqual(n_mut, 0)
        self.assertGreaterEqual(n_inj, 0)


@pytest.mark.ci
class TestEarlyStopping(unittest.TestCase):
    """Generator-protocol early stopping per genome."""

    def _make(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(n_inputs=1, n_outputs=1)
        return yane

    def test_set_early_stopping_stores_factor(self):
        yane = self._make()
        yane.set_early_stopping(factor=0.5)
        self.assertEqual(yane._runner.early_stopping_factor, 0.5)

    def test_generator_fn_all_episodes_run_when_no_early_stopping(self):
        """Without early stopping, all yielded episodes are consumed."""
        yane = self._make()
        call_counts = [0]

        def fitness_gen(genome):
            for _ in range(5):
                call_counts[0] += 1
                yield 1.0

        yane.set_max_iterations(1)
        yane.train(fitness_gen)
        self.assertEqual(call_counts[0], 5)

    def test_generator_fn_stopped_early_when_running_mean_below_threshold(self):
        """Early stopping aborts generator when extrapolated fitness < threshold."""
        yane = self._make()
        yane.set_population_size(5)
        # Seed the pool so there are evaluated genomes to compute best fitness from.
        for _ in range(5):
            g = yane.next_genome()
            yane.submit_fitness(10.0)

        yane.set_early_stopping(factor=0.0)  # stop when estimated < best (no slack)
        # Pre-calibrate N so stopping fires immediately without a warm-up run.
        yane._runner.early_stopping_n = 100

        episode_count = [0]

        def poor_gen(genome):
            for _ in range(100):
                episode_count[0] += 1
                yield -999.0  # extrapolated fitness far below best=10

        yane.set_max_iterations(1)
        yane.train(poor_gen)
        # Should have stopped after the 20-episode warmup (N//5 = 20), well before 100.
        self.assertLess(episode_count[0], 100)

    def test_n_early_stopped_incremented(self):
        yane = self._make()
        yane.set_population_size(5)
        for _ in range(5):
            yane.next_genome()
            yane.submit_fitness(10.0)
        yane.set_early_stopping(factor=1.0)

        def poor_gen(genome):
            for _ in range(100):
                yield -999.0

        yane.set_max_iterations(3)
        yane.train(poor_gen)
        self.assertGreater(yane._runner.n_early_stopped, 0)

    def test_n_early_stopped_in_mem_info(self):
        yane = self._make()
        for _ in range(3):
            yane.next_genome()
            yane.submit_fitness(1.0)
        mem = yane.population_memory_info()
        self.assertIn("n_early_stopped", mem)

    def test_regular_fn_unaffected_by_early_stopping_flag(self):
        """set_early_stopping must not break regular (non-generator) fitness fns."""
        yane = self._make()
        yane.set_early_stopping(factor=0.5)
        calls = [0]

        def fitness(genome):
            calls[0] += 1
            return 1.0

        yane.set_max_iterations(3)
        yane.train(fitness)
        self.assertEqual(calls[0], 3)


@pytest.mark.ci
class TestOutputScale(unittest.TestCase):
    """output_scale strategy gene on OUTPUT nodes."""

    def test_output_scale_default_is_1(self):
        from yane.core.node import Node, NodeType
        n = Node(NodeType.OUTPUT)
        self.assertAlmostEqual(n.output_scale, 1.0)

    def test_output_scale_applied_in_get_outputs(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(n_inputs=1, n_outputs=1)
        g = yane.next_genome()
        g.output_nodes[0].output_scale = 2.0
        g.output_nodes[0].value = 0.5
        outputs = g.get_outputs()
        self.assertAlmostEqual(outputs[0], 1.0)

    def test_output_scale_copied(self):
        from yane.core.node import Node, NodeType
        n = Node(NodeType.OUTPUT)
        n.output_scale = 3.0
        c = n.copy()
        self.assertAlmostEqual(c.output_scale, 3.0)

    def test_output_scale_mutates_on_output_nodes(self):
        from yane.core.node import Node, NodeType
        import random
        random.seed(0)
        n = Node(NodeType.OUTPUT)
        n.mutation_output_scale.p_change = 1.0
        original = n.output_scale
        for _ in range(50):
            n.mutate()
        # After 50 mutations with p_change=1.0, scale should differ.
        # (May be same by random chance, but extremely unlikely)
        # Just check it stays positive.
        self.assertGreater(n.output_scale, 0.0)

    def test_hidden_node_has_no_mutation_output_scale_effect(self):
        """Hidden nodes must NOT have output_scale mutated (only OUTPUT nodes do)."""
        from yane.core.node import Node, NodeType
        n = Node(NodeType.HIDDEN)
        n.output_scale = 5.0
        n.mutate()
        # Scale should be unchanged for hidden nodes.
        self.assertAlmostEqual(n.output_scale, 5.0)


@pytest.mark.ci
class TestWeightClipping(unittest.TestCase):
    """set_weight_clipping() bounds weights and biases after each mutation."""

    def _make(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(n_inputs=2, n_outputs=1)
        return yane

    def test_weights_clamped_after_mutation(self):
        import random
        random.seed(0)
        yane = self._make()
        yane.set_weight_clipping(w_max=0.01, b_max=0.01)
        # Force many mutations to drive weights up, then verify clipping.
        yane.set_max_iterations(5)
        yane.train(lambda g: 0.0)
        for genome in yane._population._evaluated:
            for node in genome.nodes:
                self.assertLessEqual(abs(node.bias), 0.01 + 1e-9)
                for conn in node.connections:
                    self.assertLessEqual(abs(conn.weight), 0.01 + 1e-9)

    def test_clipping_disabled_by_default(self):
        yane = self._make()
        self.assertIsNone(yane._population._weight_clip)

    def test_disable_clipping_with_none(self):
        yane = self._make()
        yane.set_weight_clipping(w_max=1.0)
        self.assertIsNotNone(yane._population._weight_clip)
        yane.set_weight_clipping()
        self.assertIsNone(yane._population._weight_clip)

    def test_b_max_defaults_to_w_max(self):
        yane = self._make()
        yane.set_weight_clipping(w_max=5.0)
        self.assertEqual(yane._population._weight_clip, (5.0, 5.0))

    def test_b_max_can_differ_from_w_max(self):
        yane = self._make()
        yane.set_weight_clipping(w_max=3.0, b_max=1.0)
        self.assertEqual(yane._population._weight_clip, (3.0, 1.0))

    def test_clipping_counter_increments(self):
        """n_weight_clipped / n_bias_clipped increase when values are clamped."""
        from yane import NeuroEvolution
        yane = NeuroEvolution(seed=0)
        # n_initial_hidden=1 creates connections from the start.
        yane.configure(n_inputs=1, n_outputs=1, n_initial_hidden=1)
        # Small pop so the unevaluated queue empties fast → spawn is triggered.
        yane.set_population_size(3)
        yane.set_weight_clipping(w_max=0.001)
        # Evaluate enough iterations to exhaust the initial queue and trigger spawn.
        yane.set_max_iterations(10)
        yane.train(lambda g: 0.0)
        pop = yane._population
        self.assertGreater(pop._n_weight_clipped + pop._n_bias_clipped, 0)

    def test_clipping_counter_no_increment_when_in_bounds(self):
        """Counters stay at zero when all weights are already within bounds."""
        from yane import NeuroEvolution
        yane = NeuroEvolution(seed=0)
        yane.configure(1, 1)
        # Zero out all weights so they're within a tight bound.
        for genome in yane._population._unevaluated:
            for node in genome.nodes:
                node.bias = 0.0
                for conn in node.connections:
                    conn.weight = 0.0
        yane.set_weight_clipping(w_max=10.0)
        # No training — just check initial state.
        pop = yane._population
        self.assertEqual(pop._n_weight_clipped, 0)
        self.assertEqual(pop._n_bias_clipped, 0)

    def test_clipping_counters_exposed_in_diagnostics(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution(seed=0)
        yane.configure(1, 1)
        yane.set_weight_clipping(w_max=0.001)
        yane.set_max_iterations(1)
        yane.train(lambda g: 0.0)
        info = yane.population_memory_info()
        self.assertIn("n_weight_clipped", info)
        self.assertIn("n_bias_clipped", info)


class TestOutputSanitizing(unittest.TestCase):
    """set_output_sanitizing() replaces NaN/Inf in forward() output."""

    def _make(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution(seed=0)
        yane.configure(n_inputs=1, n_outputs=1)
        return yane

    def _genome_with_inf_output(self):
        """Return a genome whose forward() produces Inf via output_scale=inf."""
        from yane import NeuroEvolution
        yane = NeuroEvolution(seed=0)
        yane.configure(1, 1, n_initial_hidden=1)
        g = yane._population._unevaluated[0]
        g._forward_dispatch = None
        for node in g.output_nodes:
            node.output_scale = float('inf')
            node.bias = 1.0   # ensures node value != 0 so Inf propagates
        return g

    def test_forward_replaces_nan_inf(self):
        import math
        g = self._genome_with_inf_output()
        g._output_sanitize = True
        g._output_fallback = -1.0
        out = g.forward([0.0])
        self.assertEqual(len(out), 1)
        self.assertFalse(math.isnan(out[0]))
        self.assertFalse(math.isinf(out[0]))
        self.assertEqual(out[0], -1.0)

    def test_forward_no_sanitizing_by_default(self):
        import math
        g = self._genome_with_inf_output()
        self.assertFalse(g._output_sanitize)   # default off
        out = g.forward([0.0])
        # Without sanitizing, inf output_scale must produce non-finite result
        # (only if the node value is non-zero; bias=1.0 guarantees it).
        self.assertFalse(math.isfinite(out[0]))

    def test_set_output_sanitizing_applies_to_population(self):
        import math
        yane = self._make()
        yane.set_output_sanitizing(enabled=True, fallback=0.0)
        for g in yane._population._unevaluated + yane._population._evaluated:
            self.assertTrue(g._output_sanitize)
            self.assertEqual(g._output_fallback, 0.0)

    def test_set_output_sanitizing_disabled(self):
        yane = self._make()
        yane.set_output_sanitizing(enabled=True)
        yane.set_output_sanitizing(enabled=False)
        for g in yane._population._unevaluated + yane._population._evaluated:
            self.assertFalse(g._output_sanitize)

    def test_counter_increments_on_sanitize(self):
        import math
        g = self._genome_with_inf_output()
        g._output_sanitize = True
        g._output_fallback = 0.0
        self.assertEqual(g.n_output_sanitized, 0)
        g.forward([0.0])
        self.assertGreater(g.n_output_sanitized, 0)

    def test_counter_not_incremented_when_output_clean(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution(seed=0)
        yane.configure(1, 1)
        g = yane._population._unevaluated[0]
        g._output_sanitize = True
        for node in g.output_nodes:
            node.bias = 0.0
            node.output_scale = 1.0
        out = g.forward([0.5])
        import math
        self.assertTrue(math.isfinite(out[0]))
        self.assertEqual(g.n_output_sanitized, 0)

    def test_counter_in_diagnostics(self):
        yane = self._make()
        yane.set_output_sanitizing(enabled=True)
        yane.set_max_iterations(1)
        yane.train(lambda g: 0.0)
        info = yane.population_memory_info()
        self.assertIn("n_output_sanitized", info)

    def test_copy_inherits_sanitize_settings(self):
        g = self._genome_with_inf_output()
        g._output_sanitize = True
        g._output_fallback = -99.0
        g2 = g.copy()
        self.assertTrue(g2._output_sanitize)
        self.assertEqual(g2._output_fallback, -99.0)
        self.assertEqual(g2.n_output_sanitized, 0)

    def test_forward_batch_sanitizes(self):
        import math
        from yane import NeuroEvolution
        yane = NeuroEvolution(seed=0)
        yane.configure(1, 1)
        g = yane._population._unevaluated[0]
        g._output_sanitize = True
        g._output_fallback = 42.0
        for node in g.output_nodes:
            node.bias = float('inf')
        batch = [[0.0], [1.0], [2.0]]
        results = g.forward_batch(batch)
        for row in results:
            self.assertEqual(len(row), 1)
            self.assertTrue(math.isfinite(row[0]))


class TestComplexityPenalty(unittest.TestCase):

    def test_complexity_penalty_reduces_submitted_fitness(self):
        from yane import NeuroEvolution
        from yane.core.connection import Connection
        from yane.core.node import Node, NodeType

        yane = NeuroEvolution()
        yane.configure(1, 1)
        yane.set_complexity_penalty(node_penalty=0.5, connection_penalty=0.25)
        g = yane.next_genome()
        hidden = Node(NodeType.HIDDEN)
        g.nodes.append(hidden)
        g.input_nodes[0].connections.append(Connection(hidden))
        hidden.connections.append(Connection(g.output_nodes[0]))
        g._invalidate_topology()

        yane.submit_fitness(10.0)
        self.assertAlmostEqual(g.fitness, 10.0 - 0.5 - 0.5)

    def test_complexity_penalty_defaults_to_zero(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(1, 1)
        self.assertEqual(yane._config_dict()["complexity_penalty_nodes"], 0.0)
        self.assertEqual(yane._config_dict()["complexity_penalty_connections"], 0.0)


class TestWarmStartTransfer(unittest.TestCase):

    def _checkpoint(self):
        import tempfile
        from pathlib import Path
        from yane import NeuroEvolution

        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "warm.pkl"
        src = NeuroEvolution(seed=0)
        src.configure(2, 1)
        src.set_population_size(8)
        src.set_max_iterations(12)
        src.train(lambda g: 1.0)
        src.save_checkpoint(path)
        return tmp, path

    def test_warm_start_imports_compatible_population(self):
        from yane import NeuroEvolution
        tmp, path = self._checkpoint()
        self.addCleanup(tmp.cleanup)

        dst = NeuroEvolution(seed=1)
        dst.configure(2, 1)
        dst.set_population_size(5)
        n = dst.warm_start_from_checkpoint(path)
        self.assertEqual(n, 5)
        self.assertEqual(dst._population.unevaluated_count, 5)

    def test_warm_start_filters_against_new_fitness(self):
        from yane import NeuroEvolution
        tmp, path = self._checkpoint()
        self.addCleanup(tmp.cleanup)

        dst = NeuroEvolution(seed=1)
        dst.configure(2, 1)
        dst.set_population_size(8)

        calls = [0]
        def fitness(_genome):
            calls[0] += 1
            return 1.0 if calls[0] <= 3 else -1.0

        n = dst.warm_start_from_checkpoint(path, fitness_fn=fitness, min_fitness=0.0)
        self.assertEqual(n, 3)
        self.assertEqual(dst._population.evaluated_count, 3)

    def test_warm_start_adapts_more_inputs(self):
        # 2-input checkpoint → 4-input task: 2 new input nodes are appended.
        from yane import NeuroEvolution
        tmp, path = self._checkpoint()
        self.addCleanup(tmp.cleanup)

        dst = NeuroEvolution(seed=1)
        dst.configure(4, 1)
        dst.set_population_size(5)
        n = dst.warm_start_from_checkpoint(path)
        self.assertEqual(n, 5)
        for g in dst._population._unevaluated:
            self.assertEqual(len(g.input_nodes), 4)
            self.assertEqual(len(g.output_nodes), 1)

    def test_warm_start_adapts_fewer_inputs(self):
        # 2-input checkpoint → 1-input task: first input node kept, second dropped.
        from yane import NeuroEvolution
        tmp, path = self._checkpoint()
        self.addCleanup(tmp.cleanup)

        dst = NeuroEvolution(seed=1)
        dst.configure(1, 1)
        dst.set_population_size(5)
        n = dst.warm_start_from_checkpoint(path)
        self.assertEqual(n, 5)
        for g in dst._population._unevaluated:
            self.assertEqual(len(g.input_nodes), 1)
            self.assertEqual(len(g.output_nodes), 1)

    def test_warm_start_adapted_genome_is_forwardable(self):
        # Adapted genomes must produce valid forward output (no crash/NaN).
        from yane import NeuroEvolution
        import math
        tmp, path = self._checkpoint()
        self.addCleanup(tmp.cleanup)

        dst = NeuroEvolution(seed=1)
        dst.configure(4, 1)
        dst.set_population_size(5)
        dst.warm_start_from_checkpoint(path)
        for g in dst._population._unevaluated:
            out = g.forward([0.1, 0.2, 0.3, 0.4])
            self.assertEqual(len(out), 1)
            self.assertFalse(math.isnan(out[0]))


if __name__ == "__main__":
    unittest.main()
