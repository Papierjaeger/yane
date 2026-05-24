import unittest

from yane import NeuroEvolution
from yane.core.connection import Connection
from yane.core.node import Node, NodeType
from yane.evolution.modularity import duplicate_module, hidden_modules


class TestModularity(unittest.TestCase):
    def test_hidden_modules_detect_components(self):
        yane = NeuroEvolution()
        yane.configure(1, 1)
        g = yane.next_genome()
        h1 = Node(NodeType.HIDDEN)
        h2 = Node(NodeType.HIDDEN)
        h3 = Node(NodeType.HIDDEN)
        h1.connections.append(Connection(h2))
        g.nodes.extend([h1, h2, h3])

        modules = hidden_modules(g)
        sizes = sorted(len(m) for m in modules)

        self.assertEqual(sizes, [1, 2])

    def test_duplicate_module_copies_hidden_nodes_and_connections(self):
        yane = NeuroEvolution()
        yane.configure(1, 1, max_nodes=10, max_connections=20)
        g = yane.next_genome()
        h = Node(NodeType.HIDDEN, innovation=yane._tracker.next())
        g.nodes.append(h)
        g.input_nodes[0].connections.append(Connection(h))
        h.connections.append(Connection(g.output_nodes[0]))
        g._invalidate_topology()

        before_nodes = len(g.nodes)
        before_conns = g.connection_count

        self.assertTrue(duplicate_module(g, yane._tracker, module=[h]))
        self.assertEqual(len(g.nodes), before_nodes + 1)
        self.assertEqual(g.connection_count, before_conns + 2)


if __name__ == "__main__":
    unittest.main()
