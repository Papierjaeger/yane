"""Tests for Genome.forward_batch() vectorized inference."""
from __future__ import annotations
import unittest
import random

from yane import NeuroEvolution
from yane.core.genome import Genome
from yane.core.node import NodeType


def _make_acyclic(n_in: int = 2, n_out: int = 1) -> Genome:
    yane = NeuroEvolution()
    yane.configure(n_in, n_out)
    genome = yane.next_genome()
    yane.submit_fitness(0.0)
    # Pull a few more to get connections
    for _ in range(5):
        g = yane.next_genome()
        yane.submit_fitness(0.0)
    return yane.get_best()


class TestForwardBatch(unittest.TestCase):

    def test_empty_batch_returns_empty(self):
        g = _make_acyclic()
        self.assertEqual(g.forward_batch([]), [])

    def test_single_sample_matches_forward(self):
        random.seed(0)
        g = _make_acyclic(2, 1)
        row = [0.5, -0.3]
        expected = g.forward(row)
        result = g.forward_batch([row])
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0][0], expected[0], places=10)

    def test_batch_matches_sequential(self):
        """forward_batch must produce the same output as N sequential forward() calls."""
        random.seed(42)
        g = _make_acyclic(3, 2)
        batch = [[random.gauss(0, 1) for _ in range(3)] for _ in range(50)]
        seq = [g.forward(row) for row in batch]
        bat = g.forward_batch(batch)
        self.assertEqual(len(bat), 50)
        for i, (s, b) in enumerate(zip(seq, bat)):
            for j, (sv, bv) in enumerate(zip(s, b)):
                self.assertAlmostEqual(sv, bv, places=10,
                    msg=f"Sample {i} output {j}: seq={sv} vs batch={bv}")

    def test_output_shape(self):
        random.seed(1)
        g = _make_acyclic(2, 3)
        batch = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        result = g.forward_batch(batch)
        self.assertEqual(len(result), 3)
        for row in result:
            self.assertEqual(len(row), 3)

    def test_cyclic_falls_back(self):
        """Cyclic networks must fall back without raising."""
        yane = NeuroEvolution()
        yane.configure(2, 1)
        g = yane.next_genome()
        g._has_cycles = True  # force cyclic flag
        batch = [[0.0, 1.0]]
        result = g.forward_batch(batch)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 1)

    def test_memory_node_falls_back(self):
        """Networks with persistent hidden nodes fall back to sequential."""
        yane = NeuroEvolution()
        yane.configure(2, 1)
        g = yane.next_genome()
        from yane.core.node import Node, NodeType
        mem_node = Node(NodeType.HIDDEN)
        mem_node.persist_value = True
        g.nodes.append(mem_node)
        batch = [[0.0, 0.0], [1.0, 1.0]]
        # Should not raise
        result = g.forward_batch(batch)
        self.assertEqual(len(result), 2)

    def test_neuro_evolution_forward_batch(self):
        """NeuroEvolution.forward_batch() delegates to the best genome."""
        yane = NeuroEvolution()
        yane.configure(2, 1)
        for _ in range(5):
            yane.next_genome()
            yane.submit_fitness(1.0)
        batch = [[0.1, 0.2], [0.3, 0.4]]
        result = yane.forward_batch(batch)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 1)

    def test_numpy_array_input(self):
        """forward_batch() must accept a 2-D numpy array."""
        import numpy as np
        g = _make_acyclic(2, 1)
        arr = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        result = g.forward_batch(arr)
        self.assertEqual(len(result), 3)
        # Compare against sequential
        for i, row in enumerate(arr):
            seq = g.forward(list(row))
            self.assertAlmostEqual(result[i][0], seq[0], places=10)


if __name__ == "__main__":
    unittest.main()
