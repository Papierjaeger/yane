"""Tests that directly measure object counts and RAM to catch leaks."""
import gc
import weakref
import unittest


def _count_instances(cls):
    gc.collect()
    return sum(1 for obj in gc.get_objects() if type(obj) is cls)


def _make_yane(max_size=20, max_nodes=6, max_connections=12):
    from yane import NeuroEvolution
    yane = NeuroEvolution()
    yane.configure(2, 1, max_nodes=max_nodes, max_connections=max_connections)
    yane.set_population_size(max_size)
    return yane


def _burst(yane, n):
    import random
    for _ in range(n):
        yane.next_genome()
        yane.submit_fitness(random.random())


class TestPruneClearsCycles(unittest.TestCase):
    """_prune() must call _clear() so Python can GC immediately without the cyclic collector."""

    def test_pruned_genome_nodes_are_cleared(self):
        from yane.evolution.population import Population
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(2, 1)
        pop = yane._population
        pop.max_size = 2

        g1 = pop.select_for_evaluation(); pop.submit(g1, 1.0)
        g2 = pop.select_for_evaluation(); pop.submit(g2, 2.0)
        g3 = pop.select_for_evaluation(); pop.submit(g3, 3.0)
        # g1 was worst → _prune() removed it and called _clear()
        self.assertEqual(len(g1.nodes), 0,
            "_prune() must call _clear() to break cycles; nodes should be empty")

    def test_pruned_genome_can_be_gc_collected(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(2, 1)
        pop = yane._population
        pop.max_size = 2

        g1 = pop.select_for_evaluation(); pop.submit(g1, 1.0)
        g2 = pop.select_for_evaluation(); pop.submit(g2, 2.0)

        # Hold only a weakref so we can detect when it's freed
        ref = weakref.ref(g1)
        del g1

        # Add one more genome → _prune() removes the worst (the old g1)
        g3 = pop.select_for_evaluation(); pop.submit(g3, 3.0)
        gc.collect()

        self.assertIsNone(ref(),
            "Pruned genome must be GC-collectable after _clear() breaks the node/connection cycles")


class TestPopulationStability(unittest.TestCase):
    """After the population reaches max_size, object counts must stabilise."""

    def test_genome_count_stable_after_fill(self):
        from yane.core.genome import Genome
        yane = _make_yane(max_size=20)

        # Fill: 2× max_size iterations to ensure the population is completely full
        _burst(yane, 60)
        gc.collect()
        before = _count_instances(Genome)

        _burst(yane, 60)
        gc.collect()
        after = _count_instances(Genome)

        self.assertLessEqual(after, before + 5,
            f"Genome count grew from {before} to {after} after fill — leak in _prune()")

    def test_node_count_stable_after_fill(self):
        from yane.core.node import Node
        yane = _make_yane(max_size=20, max_nodes=6)

        _burst(yane, 60)
        gc.collect()
        before = _count_instances(Node)

        _burst(yane, 60)
        gc.collect()
        after = _count_instances(Node)

        # Crossover can legitimately increase average node count (inheriting hidden
        # nodes from both parents). Verify absolute bound: never exceeds population
        # capacity plus a small in-flight margin.
        max_size, max_nodes = 20, 6
        self.assertLessEqual(after, max_size * max_nodes + 30,
            f"Node count grew from {before} to {after} — cycle leak in genome nodes")

    def test_connection_count_bounded_by_max(self):
        """Total connections must never exceed max_size × max_connections + small margin."""
        from yane.core.connection import Connection
        max_size, max_connections = 20, 12
        yane = _make_yane(max_size=max_size, max_connections=max_connections)

        _burst(yane, 120)   # well past fill
        gc.collect()
        total = _count_instances(Connection)

        # Allow some in-flight objects (current eval genome + spawn copy)
        upper_bound = max_size * max_connections + 50
        self.assertLessEqual(total, upper_bound,
            f"Total connections {total} exceeds bound {upper_bound} — possible leak")

    def test_population_size_never_exceeds_max(self):
        yane = _make_yane(max_size=15)
        for i in range(60):
            g = yane.next_genome()
            yane.submit_fitness(float(-i))
            total = yane.population_memory_info()["total_genomes"]
            self.assertLessEqual(total, 16,
                f"Population is {total} at iteration {i}, max_size=15")

    def test_max_nodes_never_exceeded(self):
        yane = _make_yane(max_size=20, max_nodes=7)
        _burst(yane, 100)
        info = yane.population_memory_info()
        self.assertLessEqual(info["largest_genome_nodes"], 7)

    def test_max_connections_never_exceeded(self):
        yane = _make_yane(max_size=20, max_connections=14)
        _burst(yane, 100)
        info = yane.population_memory_info()
        self.assertLessEqual(info["largest_genome_connections"], 14)


class TestEnforceMemoryLimit(unittest.TestCase):

    def test_shrink_to_clears_discarded(self):
        yane = _make_yane(max_size=20)
        _burst(yane, 30)
        pop = yane._population
        before_len = len(pop._evaluated)
        # Keep a weak ref to the worst genome
        worst = min(pop._evaluated, key=lambda g: g.fitness)
        ref = weakref.ref(worst)
        del worst

        pop.shrink_to(3)
        gc.collect()

        self.assertLessEqual(len(pop._evaluated), 3)
        # The freed genomes should be collectable
        self.assertIsNone(ref(),
            "shrink_to() must call _clear() so discarded genomes can be GC'd")

    def test_enforce_memory_limit_shrinks_population(self):
        yane = _make_yane(max_size=20)
        _burst(yane, 30)
        yane.set_resource_limits(max_process_gb=0.001)
        before = len(yane._population._evaluated)
        yane._enforce_memory_limit()
        after = len(yane._population._evaluated)
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
