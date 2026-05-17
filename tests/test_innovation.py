"""Tests for InnovationTracker and innovation-number propagation."""
import unittest

from yane.evolution.innovation import InnovationTracker
from yane import NeuroEvolution
from yane.evolution import smart_mutation


def _make_tracked_genome(n_inputs=2, n_outputs=1):
    yane = NeuroEvolution()
    yane.configure(n_inputs, n_outputs, max_nodes=30, max_connections=100)
    return yane.next_genome(), yane._tracker


class TestInnovationTracker(unittest.TestCase):

    def test_counter_is_unique_and_monotonic(self):
        t = InnovationTracker()
        values = [t.next() for _ in range(1000)]
        self.assertEqual(values, list(range(1000)))
        self.assertEqual(len(set(values)), 1000)

    def test_same_connection_always_same_innovation(self):
        t = InnovationTracker()
        innov_a = t.get_connection(0, 4)
        innov_b = t.get_connection(0, 4)
        self.assertEqual(innov_a, innov_b)

    def test_different_connections_different_innovations(self):
        t = InnovationTracker()
        i1 = t.get_connection(0, 4)
        i2 = t.get_connection(1, 4)
        self.assertNotEqual(i1, i2)

    def test_split_same_connection_returns_identical_triple(self):
        t = InnovationTracker()
        conn_innov = t.get_connection(0, 4)
        triple_a = t.get_split(conn_innov)
        triple_b = t.get_split(conn_innov)
        self.assertEqual(triple_a, triple_b)
        self.assertEqual(len(triple_a), 3)
        node_i, conn_in, conn_out = triple_a
        self.assertNotEqual(node_i, conn_in)
        self.assertNotEqual(conn_in, conn_out)

    def test_different_splits_get_different_innovations(self):
        t = InnovationTracker()
        c1 = t.get_connection(0, 4)
        c2 = t.get_connection(1, 4)
        t1 = t.get_split(c1)
        t2 = t.get_split(c2)
        self.assertNotEqual(t1, t2)


class TestInnovationOnGenome(unittest.TestCase):

    def test_configure_assigns_innovations_to_all_nodes(self):
        g, _ = _make_tracked_genome(3, 2)
        for node in g.nodes:
            self.assertGreaterEqual(node.innovation, 0,
                f"{node.type} node must have innovation >= 0 after configure()")

    def test_add_connection_assigns_innovation(self):
        g, tracker = _make_tracked_genome()
        smart_mutation.add_connection(g, tracker)
        tracked = [c for n in g.nodes for c in n.connections if c.innovation >= 0]
        self.assertGreater(len(tracked), 0,
            "at least one connection must have a tracked innovation after add_connection")

    def test_add_node_assigns_innovations_to_new_node_and_conns(self):
        g, tracker = _make_tracked_genome()
        smart_mutation.add_connection(g, tracker)
        before_nodes = len(g.nodes)
        smart_mutation.add_node(g, tracker)
        self.assertEqual(len(g.nodes), before_nodes + 1)
        new_node = g.nodes[-1]
        self.assertGreaterEqual(new_node.innovation, 0,
            "split node must have innovation >= 0")

    def test_copy_preserves_all_innovations(self):
        g, tracker = _make_tracked_genome()
        for _ in range(5):
            smart_mutation.add_connection(g, tracker)
        copy = g.copy()
        orig_node_innovations = {n.innovation for n in g.nodes}
        copy_node_innovations = {n.innovation for n in copy.nodes}
        self.assertEqual(orig_node_innovations, copy_node_innovations)

        orig_conn_innovations = {c.innovation for n in g.nodes for c in n.connections}
        copy_conn_innovations = {c.innovation for n in copy.nodes for c in n.connections}
        self.assertEqual(orig_conn_innovations, copy_conn_innovations)

    def test_same_connection_in_two_genomes_gets_same_innovation(self):
        """Convergent evolution: two genomes independently adding the same
        structural connection must receive the same innovation number."""
        yane = NeuroEvolution()
        yane.configure(2, 1, max_nodes=10, max_connections=50)
        tracker = yane._tracker
        g1 = yane.next_genome()
        g2 = yane.next_genome()

        # Find a pair of nodes that both genomes share (same innovation numbers)
        inp_innov = g1.input_nodes[0].innovation
        out_innov = g1.output_nodes[0].innovation

        # Both genomes add the identical logical connection
        from yane.core.connection import Connection
        innov1 = tracker.get_connection(inp_innov, out_innov)
        c1 = Connection(g1.output_nodes[0], innovation=innov1)
        g1.input_nodes[0].connections.append(c1)

        innov2 = tracker.get_connection(inp_innov, out_innov)
        c2 = Connection(g2.output_nodes[0], innovation=innov2)
        g2.input_nodes[0].connections.append(c2)

        self.assertEqual(innov1, innov2,
            "same logical connection must get same innovation in both genomes")


if __name__ == "__main__":
    unittest.main()
