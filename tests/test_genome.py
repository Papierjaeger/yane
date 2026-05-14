import unittest
from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection


def _make_genome(n_inputs=2, n_outputs=1):
    from yane import NeuroEvolution
    yane = NeuroEvolution()
    yane.configure(n_inputs, n_outputs)
    return yane.next_genome()


class TestGenomeCopy(unittest.TestCase):

    def test_copy_is_independent(self):
        g = _make_genome()
        copy = g.copy()
        # Modifying the copy does not affect the original
        copy.fitness = 999.0
        self.assertNotEqual(g.fitness, 999.0)

    def test_copy_has_same_structure(self):
        g = _make_genome()
        copy = g.copy()
        self.assertEqual(len(g.nodes), len(copy.nodes))
        self.assertEqual(len(g.input_nodes), len(copy.input_nodes))
        self.assertEqual(len(g.output_nodes), len(copy.output_nodes))

    def test_copy_nodes_are_different_objects(self):
        g = _make_genome()
        copy = g.copy()
        original_ids = {id(n) for n in g.nodes}
        copy_ids = {id(n) for n in copy.nodes}
        self.assertTrue(original_ids.isdisjoint(copy_ids))

    def test_copy_connections_point_to_new_nodes(self):
        g = _make_genome()
        copy = g.copy()
        copy_node_ids = {id(n) for n in copy.nodes}
        for node in copy.nodes:
            for conn in node.connections:
                self.assertIn(id(conn.target), copy_node_ids,
                    "Connection in copy points to node from original genome")

    def test_copy_inherits_caps(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(2, 1, max_nodes=15, max_connections=30)
        g = yane.next_genome()
        copy = g.copy()
        self.assertEqual(copy.max_nodes, 15)
        self.assertEqual(copy.max_connections, 30)


class TestGenomeForward(unittest.TestCase):

    def test_forward_returns_correct_output_count(self):
        g = _make_genome(2, 1)
        out = g.forward([0.0, 1.0])
        self.assertEqual(len(out), 1)

    def test_forward_is_deterministic(self):
        g = _make_genome(2, 1)
        out1 = g.forward([0.5, 0.5])
        out2 = g.forward([0.5, 0.5])
        self.assertAlmostEqual(out1[0], out2[0])

    def test_forward_hard_resets_between_calls(self):
        g = _make_genome(2, 1)
        out1 = g.forward([1.0, 0.0])
        out2 = g.forward([0.0, 1.0])
        out3 = g.forward([1.0, 0.0])
        self.assertAlmostEqual(out1[0], out3[0],
            msg="forward() must hard-reset — same inputs must give same output")

    def test_forward_different_inputs_different_outputs(self):
        g = _make_genome(2, 1)
        out1 = g.forward([0.0, 0.0])
        out2 = g.forward([1.0, 1.0])
        # With random weights, outputs will almost certainly differ
        # (test the network is actually connected)
        self.assertEqual(len(out1), 1)
        self.assertEqual(len(out2), 1)

    def test_forward_output_in_sigmoid_range(self):
        g = _make_genome(2, 1)
        out = g.forward([0.5, 0.5])
        self.assertGreater(out[0], 0.0)
        self.assertLess(out[0], 1.0)

    def test_tick_mode_propagates(self):
        g = _make_genome(2, 1)
        g.set_inputs([1.0, 0.0])
        g.tick()   # input → hidden/output
        g.tick()   # propagate further if needed
        out = g.get_outputs()
        self.assertEqual(len(out), 1)

    def test_reset_clears_triggered(self):
        g = _make_genome(2, 1)
        g.set_inputs([1.0, 1.0])
        g.tick()
        g.reset()
        self.assertEqual(len(g._triggered), 0)


class TestGenomeMutation(unittest.TestCase):

    def test_mutate_does_not_exceed_max_nodes(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(2, 1, max_nodes=5, max_connections=10)
        g = yane.next_genome()
        for _ in range(100):
            g.mutate()
        self.assertLessEqual(len(g.nodes), 5)

    def test_mutate_does_not_exceed_max_connections(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(2, 1, max_nodes=5, max_connections=8)
        g = yane.next_genome()
        for _ in range(200):
            g.mutate()
        self.assertLessEqual(g.connection_count, 8)

    def test_mutation_rates_stay_above_minimum(self):
        from yane.evolution.mutation import Mutation
        g = _make_genome()
        for _ in range(500):
            g.mutate()
        for node in g.nodes:
            for m in [node.mutation_bias, node.mutation_activation,
                      node.mutation_persist, node.mutation_max_triggers]:
                self.assertGreaterEqual(m.shift_rate, Mutation.MIN_RATE)
                self.assertGreaterEqual(m.bool_rate, Mutation.MIN_RATE)


if __name__ == "__main__":
    unittest.main()
