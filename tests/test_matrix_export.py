import unittest

from yane import NeuroEvolution
from yane.core.connection import Connection
from yane.core.node import Node, NodeType
from yane.evolution.matrix_export import export_matrix_genome, forward_matrix
from yane.util.activation import ActivationType


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


if __name__ == "__main__":
    unittest.main()
