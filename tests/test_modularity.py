import unittest

from yane import NeuroEvolution
from yane.core.connection import Connection
from yane.core.node import Node, NodeType
from yane.evolution.modularity import duplicate_module, hidden_modules
from yane.evolution.modularity import ModuleLibrary, module_crossover


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

    def test_module_library_stores_and_reinserts_blueprint(self):
        yane = NeuroEvolution()
        yane.configure(1, 1, max_nodes=20, max_connections=30)
        donor = yane.next_genome()
        h = Node(NodeType.HIDDEN, innovation=yane._tracker.next())
        donor.nodes.append(h)
        donor.input_nodes[0].connections.append(Connection(h))
        h.connections.append(Connection(donor.output_nodes[0]))
        donor.fitness = 2.0
        donor._invalidate_topology()

        library = ModuleLibrary(max_modules=3)
        self.assertTrue(library.add_from_genome(donor, module=[h]))

        recipient = donor.copy()
        before_nodes = len(recipient.nodes)
        self.assertTrue(library.insert_into(recipient, yane._tracker))

        self.assertEqual(len(recipient.nodes), before_nodes + 1)
        diagnostics = library.diagnostics()
        self.assertEqual(diagnostics["module_count"], 1)
        self.assertEqual(diagnostics["n_reused"], 1)
        self.assertGreater(diagnostics["reuse_rate"], 0.0)

    def test_module_crossover_inserts_compatible_donor_module(self):
        yane = NeuroEvolution()
        yane.configure(1, 1, max_nodes=20, max_connections=30)
        recipient = yane.next_genome().copy()
        donor = recipient.copy()
        h = Node(NodeType.HIDDEN, innovation=yane._tracker.next())
        donor.nodes.append(h)
        donor.input_nodes[0].connections.append(Connection(h))
        h.connections.append(Connection(donor.output_nodes[0]))
        donor._invalidate_topology()

        before_nodes = len(recipient.nodes)
        self.assertTrue(module_crossover(recipient, donor, yane._tracker))
        self.assertGreater(len(recipient.nodes), before_nodes)

    def test_neuro_evolution_module_library_diagnostics(self):
        yane = NeuroEvolution()
        yane.configure(1, 1, max_nodes=20, max_connections=30)
        yane.set_module_library(enabled=True, insert_rate=1.0)
        genome = yane.next_genome()
        h = Node(NodeType.HIDDEN, innovation=yane._tracker.next())
        genome.nodes.append(h)
        genome.input_nodes[0].connections.append(Connection(h))
        h.connections.append(Connection(genome.output_nodes[0]))
        genome._invalidate_topology()

        yane.submit_fitness(1.0)
        info = yane.population_memory_info()

        self.assertIn("module_library", info)
        self.assertEqual(info["module_library"]["module_count"], 1)
        self.assertEqual(info["module_insert_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
