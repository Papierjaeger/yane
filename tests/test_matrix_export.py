import unittest

from yane import NeuroEvolution
from yane.core.connection import Connection
from yane.core.node import Node, NodeType
from yane.evolution.matrix_export import export_matrix_genome, forward_matrix
from yane.util.activation import ActivationType


def _xor_fitness(genome):
    cases = [([0.0, 0.0], 0.0), ([0.0, 1.0], 1.0), ([1.0, 0.0], 1.0), ([1.0, 1.0], 0.0)]
    genome.reset()
    err = 0.0
    for inputs, target in cases:
        out = genome.forward(inputs)
        err += (out[0] - target) ** 2
    return -err / len(cases)


class TestMatrixExport(unittest.TestCase):
    def test_matrix_forward_matches_simple_genome(self):
        yane = NeuroEvolution()
        yane.configure(2, 1)
        g = yane.next_genome()
        g.input_nodes[0].activation = ActivationType.LINEAR
        g.input_nodes[1].activation = ActivationType.LINEAR
        g.output_nodes[0].activation = ActivationType.LINEAR
        c1 = Connection(g.output_nodes[0]); c1.weight = 0.5
        c2 = Connection(g.output_nodes[0]); c2.weight = -0.25
        g.input_nodes[0].connections.append(c1)
        g.input_nodes[1].connections.append(c2)
        g._invalidate_topology()

        exported = export_matrix_genome(g)

        self.assertEqual(forward_matrix(exported, [2.0, 4.0]), g.forward([2.0, 4.0]))

    def test_matrix_export_rejects_cyclic_genome(self):
        yane = NeuroEvolution()
        yane.configure(1, 1)
        g = yane.next_genome()
        h = Node(NodeType.HIDDEN)
        g.nodes.append(h)
        g.input_nodes[0].connections.append(Connection(h))
        h.connections.append(Connection(h))
        h.connections.append(Connection(g.output_nodes[0]))
        g._invalidate_topology()

        with self.assertRaises(ValueError):
            export_matrix_genome(g)


class TestMatrixForwardIntegration(unittest.TestCase):
    def test_set_matrix_forward_enables_flag(self):
        yane = NeuroEvolution()
        yane.configure(2, 1)
        self.assertFalse(yane._matrix_forward_enabled)
        yane.set_matrix_forward(enabled=True)
        self.assertTrue(yane._matrix_forward_enabled)
        self.assertIsNotNone(yane._matrix_cache)

    def test_set_matrix_forward_disable(self):
        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane.set_matrix_forward(True)
        yane.set_matrix_forward(False)
        self.assertFalse(yane._matrix_forward_enabled)

    def test_training_with_matrix_forward_produces_fitness(self):
        yane = NeuroEvolution()
        yane.configure(2, 1, n_initial_hidden=2)
        yane.set_matrix_forward(enabled=True)
        g = yane.next_genome()
        fitness = _xor_fitness(g)
        yane.submit_fitness(fitness)
        # verify genome.forward is NOT patched after submit
        self.assertNotIn("forward", g.__dict__)

    def test_matrix_forward_train_loop_accumulates_hits(self):
        yane = NeuroEvolution()
        yane.configure(2, 1, n_initial_hidden=2)
        yane.set_matrix_forward(enabled=True)
        yane.set_max_iterations(5)
        yane.train(_xor_fitness)
        mem = yane.population_memory_info()
        self.assertIn("matrix_forward_hits", mem)
        self.assertIn("matrix_forward_misses", mem)
        total = mem["matrix_forward_hits"] + mem["matrix_forward_misses"]
        self.assertGreater(total, 0)

    def test_matrix_forward_genome_not_patched_after_evaluation(self):
        """genome.forward must not remain patched after _run_evaluations returns."""
        yane = NeuroEvolution()
        yane.configure(2, 1, n_initial_hidden=2)
        yane.set_matrix_forward(enabled=True)
        g = yane.next_genome()
        yane.submit_fitness(_xor_fitness(g))
        # genome.forward should still be the class method, not an instance attr
        self.assertNotIn("forward", g.__dict__)

    def test_matrix_forward_diagnostics_absent_when_disabled(self):
        yane = NeuroEvolution()
        yane.configure(2, 1)
        g = yane.next_genome()
        yane.submit_fitness(1.0)
        mem = yane.population_memory_info()
        self.assertNotIn("matrix_forward_hits", mem)
        self.assertNotIn("matrix_forward_misses", mem)


if __name__ == "__main__":
    unittest.main()
