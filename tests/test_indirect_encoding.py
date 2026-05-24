import unittest

from yane import NeuroEvolution
from yane.evolution.indirect_encoding import (
    generate_connections_from_coordinates,
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


if __name__ == "__main__":
    unittest.main()
