"""Tests for post-training pruning: set_post_training_pruning, prune_stats, rollback."""

import unittest

from yane import NeuroEvolution
from yane.core.connection import Connection
from yane.core.genome import Genome


def _make_yane(n_inputs=2, n_outputs=1, pop_size=10):
    yane = NeuroEvolution(seed=0)
    yane.set_population_size(pop_size)
    yane.configure(n_inputs, n_outputs)
    return yane


def _add_connections(genome: Genome, weights: list[float]) -> None:
    """Add enabled connections from input_nodes to first output with given weights."""
    out = genome.output_nodes[0]
    for i, w in enumerate(weights):
        src = genome.input_nodes[i % len(genome.input_nodes)]
        conn = Connection(out, innovation=1000 + i)
        conn.weight = w
        conn.enabled = True
        src.connections.append(conn)
    genome._invalidate_topology()


class TestPruneStats(unittest.TestCase):
    """Tests for Genome.prune_stats()."""

    def test_prune_stats_initial_zeros(self):
        """Before any pruning, prune_stats() returns all-zero values."""
        yane = _make_yane()
        genome = yane._population._unevaluated[0]
        stats = genome.prune_stats()
        self.assertEqual(stats["connections_removed"], 0)
        self.assertEqual(stats["nodes_removed"], 0)
        self.assertAlmostEqual(stats["fitness_delta"], 0.0)
        self.assertAlmostEqual(stats["compression_rate"], 0.0)
        self.assertFalse(stats["rolled_back"])

    def test_prune_stats_after_prune(self):
        """After prune(), stats reflect actual removals."""
        yane = _make_yane()
        genome = yane._population._unevaluated[0]
        _add_connections(genome, [0.001, 0.5])  # one below threshold, one above

        removed = genome.prune(threshold=0.01)
        stats = genome.prune_stats()
        self.assertEqual(removed, 1)
        self.assertEqual(stats["connections_removed"], 1)
        self.assertGreater(stats["compression_rate"], 0.0)
        self.assertFalse(stats["rolled_back"])

    def test_compress_stats(self):
        """compress() also updates prune_stats."""
        yane = _make_yane()
        genome = yane._population._unevaluated[0]
        _add_connections(genome, [0.1, 0.2, 0.3])
        before = genome.connection_count
        genome.compress(target_size=max(0, before - 1))
        stats = genome.prune_stats()
        self.assertGreater(stats["connections_removed"], 0)
        self.assertGreater(stats["compression_rate"], 0.0)

    def test_prune_stats_returns_copy(self):
        """prune_stats() returns an independent copy; mutations don't affect internal state."""
        yane = _make_yane()
        genome = yane._population._unevaluated[0]
        _add_connections(genome, [0.001])
        genome.prune(threshold=0.01)
        stats1 = genome.prune_stats()
        stats1["connections_removed"] = 999
        stats2 = genome.prune_stats()
        self.assertEqual(stats2["connections_removed"], 1)


class TestSetPostTrainingPruning(unittest.TestCase):
    """Tests for NeuroEvolution.set_post_training_pruning() API."""

    def test_api_stores_config(self):
        yane = _make_yane()
        yane.set_post_training_pruning(enabled=True, threshold=0.05, max_drop_frac=0.03)
        self.assertTrue(yane._post_pruning_enabled)
        self.assertAlmostEqual(yane._post_pruning_threshold, 0.05)
        self.assertAlmostEqual(yane._post_pruning_max_drop_frac, 0.03)

    def test_api_disable(self):
        yane = _make_yane()
        yane.set_post_training_pruning(enabled=True)
        yane.set_post_training_pruning(enabled=False)
        self.assertFalse(yane._post_pruning_enabled)

    def test_api_invalid_threshold(self):
        yane = _make_yane()
        with self.assertRaises(ValueError):
            yane.set_post_training_pruning(threshold=-0.1)

    def test_api_invalid_max_drop_frac(self):
        yane = _make_yane()
        with self.assertRaises(ValueError):
            yane.set_post_training_pruning(max_drop_frac=1.5)


class TestPostTrainingPruningHook(unittest.TestCase):
    """Tests for the post-training pruning hook in train()."""

    def _train_one_generation(self, yane, fitness_fn):
        yane.set_max_iterations(yane._population.max_size)
        yane.train(fitness_fn)

    def test_prune_applied_after_train(self):
        """After train(), best genome has connections below threshold removed."""
        yane = _make_yane(pop_size=5)
        yane.set_post_training_pruning(enabled=True, threshold=0.5, max_drop_frac=1.0)

        # Fitness: sum of weights — pruning should never trigger rollback
        def fitness_fn(g):
            return sum(c.weight for n in g.nodes for c in n.connections if c.enabled)

        self._train_one_generation(yane, fitness_fn)
        best = yane.get_best()
        stats = best.prune_stats()
        # All weights < 0.5 should have been removed (or 0 if none existed)
        for n in best.nodes:
            for c in n.connections:
                if c.enabled:
                    self.assertGreaterEqual(abs(c.weight), 0.5,
                        "Connection below threshold survived pruning")
        self.assertFalse(stats["rolled_back"])

    def test_rollback_when_fitness_drops_too_much(self):
        """Pruning is rolled back when fitness drop exceeds max_drop_frac."""
        yane = _make_yane(pop_size=5)
        yane.set_post_training_pruning(enabled=True, threshold=10.0, max_drop_frac=0.0)

        call_count = [0]

        def fitness_fn(g):
            call_count[0] += 1
            total = sum(c.weight for n in g.nodes for c in n.connections if c.enabled)
            return max(0.0, total)

        self._train_one_generation(yane, fitness_fn)
        best = yane.get_best()
        stats = best.prune_stats()
        # If any connections existed, rollback should have fired (threshold=10 removes all)
        if stats["connections_removed"] > 0:
            self.assertTrue(stats["rolled_back"])

    def test_no_pruning_when_disabled(self):
        """When pruning is disabled, train() does not modify connections."""
        yane = _make_yane(pop_size=5)
        # Do NOT call set_post_training_pruning — disabled by default

        def fitness_fn(g):
            return sum(c.weight for n in g.nodes for c in n.connections if c.enabled)

        self._train_one_generation(yane, fitness_fn)
        best = yane.get_best()
        stats = best.prune_stats()
        # Without explicit pruning, stats should be zero
        self.assertEqual(stats["connections_removed"], 0)
        self.assertFalse(stats["rolled_back"])

    def test_prune_stats_compression_rate_in_range(self):
        """compression_rate is in [0, 1]."""
        yane = _make_yane(pop_size=5)
        yane.set_post_training_pruning(enabled=True, threshold=0.01, max_drop_frac=1.0)

        def fitness_fn(g):
            return 1.0

        self._train_one_generation(yane, fitness_fn)
        best = yane.get_best()
        stats = best.prune_stats()
        self.assertGreaterEqual(stats["compression_rate"], 0.0)
        self.assertLessEqual(stats["compression_rate"], 1.0)


if __name__ == '__main__':
    unittest.main()
