import unittest


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
        self.assertEqual(yane._n_evaluations, 5)
        self.assertEqual(yane._eval_aggregation, "median")
        self.assertAlmostEqual(yane._eval_sigma_penalty, 0.3)

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
        self.assertEqual(yane._n_evaluations, 1)
        yane.set_max_iterations(1)
        call_count = [0]

        def evaluate(genome):
            call_count[0] += 1
            return -1.0

        yane.train(evaluate)
        self.assertEqual(call_count[0], 1)


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
        self.assertEqual(yane._n_invalid_fitness, 0)
        g = yane.next_genome()
        yane.submit_fitness(float("nan"))
        self.assertEqual(yane._n_invalid_fitness, 1)
        g = yane.next_genome()
        yane.submit_fitness(float("-inf"))
        self.assertEqual(yane._n_invalid_fitness, 2)

    def test_clipped_counter_increments(self):
        yane = self._make(sanitize=True, clip_low=0.0, clip_high=1.0)
        g = yane.next_genome()
        yane.submit_fitness(-5.0)
        self.assertEqual(yane._n_clipped_fitness, 1)
        g = yane.next_genome()
        yane.submit_fitness(99.0)
        self.assertEqual(yane._n_clipped_fitness, 2)

    def test_sanitize_disabled_by_default(self):
        yane = self._make(sanitize=False)
        self.assertFalse(yane._sanitize)
        # nan passes through unmodified when disabled
        g = yane.next_genome()
        yane.submit_fitness(float("nan"))
        self.assertEqual(yane._n_invalid_fitness, 0)

    def test_invalid_fitness_in_train(self):
        yane = self._make(sanitize=True, fallback=-999.0)
        yane.set_max_iterations(3)
        call_count = [0]
        def evaluate(genome):
            call_count[0] += 1
            return float("nan")
        yane.train(evaluate)
        self.assertEqual(yane._n_invalid_fitness, 3)
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
        self.assertEqual(yane._n_invalid_fitness, 1)


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


if __name__ == "__main__":
    unittest.main()
