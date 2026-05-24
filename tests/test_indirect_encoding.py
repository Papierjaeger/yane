import unittest

from yane import NeuroEvolution
from yane.evolution.indirect_encoding import (
    CPPNGenome,
    build_genome_from_substrate,
    generate_connections_from_coordinates,
    generate_genome_from_cppn,
    generate_weight_pattern,
    hyperneat_substrate,
    layered_coordinates,
)


class TestIndirectEncoding(unittest.TestCase):
    def test_layered_coordinates_cover_all_nodes(self):
        yane = NeuroEvolution()
        yane.configure(2, 1, n_initial_hidden=2)
        g = yane.next_genome()

        coords = layered_coordinates(g)

        self.assertEqual(len(coords), len(g.nodes))
        self.assertTrue(all(-1.0 <= y <= 1.0 for _x, y in coords.values()))

    def test_generate_connections_uses_weight_function(self):
        yane = NeuroEvolution()
        yane.configure(2, 1, max_connections=10)
        g = yane.next_genome()

        added = generate_connections_from_coordinates(
            g,
            weight_fn=lambda x1, y1, x2, y2, d: 1.0,
            threshold=0.5,
            tracker=yane._tracker,
        )

        self.assertGreaterEqual(added, 1)
        self.assertEqual(g.connection_count, added)

    def test_cppn_genome_generates_weight_pattern(self):
        cppn = CPPNGenome()
        substrate = hyperneat_substrate(2, 1, hidden_layers=(2,))

        pattern = generate_weight_pattern(cppn, substrate, threshold=0.0)

        self.assertEqual(len(pattern), len(substrate.pairs))
        self.assertTrue(all(isinstance(weight, float) for _pair, weight in pattern))

    def test_build_genome_from_hyperneat_substrate(self):
        yane = NeuroEvolution()
        yane.configure(1, 1)
        substrate = hyperneat_substrate(2, 1, hidden_layers=(2,))
        pattern = [((src, tgt), 0.5) for src, tgt in substrate.pairs]

        genome = build_genome_from_substrate(substrate, pattern, tracker=yane._tracker)

        self.assertEqual(len(genome.input_nodes), 2)
        self.assertEqual(len(genome.output_nodes), 1)
        self.assertEqual(genome.connection_count, len(pattern))

    def test_generate_genome_from_evolvable_cppn(self):
        yane = NeuroEvolution()
        yane.configure(1, 1)
        cppn = CPPNGenome()
        substrate = hyperneat_substrate(2, 1)

        genome = generate_genome_from_cppn(cppn, substrate, threshold=0.0, tracker=yane._tracker)

        self.assertEqual(len(genome.input_nodes), 2)
        self.assertEqual(len(genome.output_nodes), 1)
        self.assertEqual(genome.connection_count, len(substrate.pairs))


if __name__ == "__main__":
    unittest.main()
