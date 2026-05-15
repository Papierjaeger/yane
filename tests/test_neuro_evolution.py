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
        yane = self._make(n_inputs=2, n_outputs=1)
        g = yane.next_genome()
        for n in g.output_nodes:
            self.assertTrue(n.persist_value,
                "Output nodes must have persist_value=True so forward() returns correct values")

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


if __name__ == "__main__":
    unittest.main()
