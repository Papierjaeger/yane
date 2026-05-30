"""Tests für Evolvable Attention Heads (evolution/attention.py).

Akzeptanzkriterien:
  1. Attention-Node produziert korrekte Softmax-gewichtete Outputs
  2. Crossover: Genome mit unterschiedlichen Attention-Konfigurationen kreuzbar
  3. Checkpoint-Roundtrip erhält head_dim und num_heads
  4. Tests: Attention-Mathe; Multi-Head-Konkatenation; Crossover; Checkpoint
"""
from __future__ import annotations

import math
import pickle
import unittest

import pytest

from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_genome(n_inputs: int = 4, n_outputs: int = 2) -> Genome:
    g = Genome()
    for i in range(n_inputs):
        n = Node(NodeType.INPUT, i); n.activation = ActivationType.LINEAR; n.input_index = i
        g.input_nodes.append(n); g.nodes.append(n)
    for j in range(n_outputs):
        out = Node(NodeType.OUTPUT, n_inputs + j); out.activation = ActivationType.SIGMOID; out.bias = 0.0
        g.output_nodes.append(out); g.nodes.append(out)
    for inp in g.input_nodes:
        for out in g.output_nodes:
            c = Connection(out, 100 + inp.innovation * 10 + out.innovation)
            c.weight = 0.1; inp.connections.append(c)
    g._invalidate_topology()
    return g


# ---------------------------------------------------------------------------
# Acceptance criterion 1: Attention-Mathe
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestAttentionMath(unittest.TestCase):

    def test_softmax_sums_to_one(self):
        from yane.evolution.attention import _softmax
        values = [1.0, 2.0, 3.0, -1.0]
        result = _softmax(values)
        self.assertAlmostEqual(sum(result), 1.0, places=10)

    def test_softmax_order_preserved(self):
        from yane.evolution.attention import _softmax
        values = [1.0, 3.0, 2.0]
        result = _softmax(values)
        self.assertGreater(result[1], result[2])
        self.assertGreater(result[2], result[0])

    def test_softmax_equal_inputs(self):
        from yane.evolution.attention import _softmax
        result = _softmax([0.0, 0.0, 0.0])
        for v in result:
            self.assertAlmostEqual(v, 1.0 / 3.0, places=10)

    def test_softmax_numerically_stable(self):
        """Large values shouldn't produce inf or NaN."""
        from yane.evolution.attention import _softmax
        result = _softmax([1000.0, 999.0, 1001.0])
        for v in result:
            self.assertFalse(math.isnan(v))
            self.assertFalse(math.isinf(v))

    def test_attention_forward_output_length(self):
        """AttentionBlock.forward must return num_heads * head_dim values."""
        from yane.evolution.attention import AttentionBlock
        block = AttentionBlock(n_inputs=6, head_dim=4, num_heads=3)
        result = block.forward([0.1, 0.5, -0.3, 0.8, 0.2, -0.1])
        self.assertEqual(len(result), 4 * 3)

    def test_attention_n_outputs_property(self):
        from yane.evolution.attention import AttentionBlock
        block = AttentionBlock(n_inputs=8, head_dim=6, num_heads=2)
        self.assertEqual(block.n_outputs, 12)

    def test_attention_deterministic_same_weights(self):
        """Same weights + same inputs → same output."""
        from yane.evolution.attention import AttentionBlock
        block = AttentionBlock(n_inputs=4, head_dim=4, num_heads=2, seed=42)
        inp = [0.5, -0.3, 1.0, 0.2]
        out1 = block.forward(inp)
        out2 = block.forward(inp)
        self.assertEqual(out1, out2)

    def test_attention_different_inputs_different_outputs(self):
        from yane.evolution.attention import AttentionBlock
        block = AttentionBlock(n_inputs=4, head_dim=4, num_heads=2, seed=1)
        out1 = block.forward([0.0, 0.0, 0.0, 0.0])
        out2 = block.forward([1.0, 1.0, 1.0, 1.0])
        self.assertFalse(out1 == out2, "Different inputs should yield different outputs")

    def test_multi_head_output_is_concatenation(self):
        """Output length must equal num_heads * head_dim regardless of n_inputs."""
        from yane.evolution.attention import AttentionBlock
        for n_in, hd, nh in [(2, 3, 2), (8, 4, 4), (16, 2, 8)]:
            block = AttentionBlock(n_inputs=n_in, head_dim=hd, num_heads=nh)
            result = block.forward([0.5] * n_in)
            self.assertEqual(len(result), hd * nh,
                             f"n_inputs={n_in}, head_dim={hd}, num_heads={nh}")

    def test_finite_outputs(self):
        """Attention must not produce NaN or Inf."""
        from yane.evolution.attention import AttentionBlock
        block = AttentionBlock(n_inputs=4, head_dim=4, num_heads=2, seed=99)
        result = block.forward([10.0, -10.0, 5.0, -5.0])
        for v in result:
            self.assertFalse(math.isnan(v))
            self.assertFalse(math.isinf(v))


# ---------------------------------------------------------------------------
# Acceptance criterion 2: Crossover
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestAttentionCrossover(unittest.TestCase):

    def test_crossover_produces_valid_block(self):
        from yane.evolution.attention import AttentionBlock
        a = AttentionBlock(n_inputs=4, head_dim=4, num_heads=2, seed=1)
        b = AttentionBlock(n_inputs=4, head_dim=4, num_heads=2, seed=2)
        child = a.crossover(b)
        self.assertEqual(child.head_dim, 4)
        self.assertEqual(child.num_heads, 2)
        self.assertEqual(child.n_outputs, 8)

    def test_crossover_inherits_from_parents(self):
        """Child weights must come from one of the two parents."""
        from yane.evolution.attention import AttentionBlock
        import random
        random.seed(0)
        a = AttentionBlock(n_inputs=2, head_dim=2, num_heads=1, seed=3)
        b = AttentionBlock(n_inputs=2, head_dim=2, num_heads=1, seed=7)
        child = a.crossover(b)
        # Child head 0 must be either from a or from b
        is_a = child.W_Q[0] == a.W_Q[0]
        is_b = child.W_Q[0] == b.W_Q[0]
        self.assertTrue(is_a or is_b, "Child head must come from one parent")

    def test_genome_crossover_with_different_attention_no_error(self):
        """Crossover of two genomes with attention blocks must not raise."""
        g_a = _make_genome(n_inputs=8, n_outputs=2); g_a.fitness = 10.0
        g_b = _make_genome(n_inputs=8, n_outputs=2); g_b.fitness = 5.0
        from yane.evolution.attention import AttentionBlock
        g_a.attention_block = AttentionBlock(n_inputs=8, head_dim=4, num_heads=2)
        g_b.attention_block = AttentionBlock(n_inputs=8, head_dim=4, num_heads=2)
        try:
            child = g_a.crossover(g_b)
            self.assertIsNotNone(child.attention_block)
        except Exception as e:
            self.fail(f"crossover raised: {e}")

    def test_genome_copy_preserves_attention(self):
        g = _make_genome()
        from yane.evolution.attention import AttentionBlock
        g.attention_block = AttentionBlock(n_inputs=4, head_dim=4, num_heads=2, seed=5)
        gc = g.copy()
        self.assertIsNotNone(gc.attention_block)
        self.assertEqual(gc.attention_block.head_dim, 4)
        self.assertEqual(gc.attention_block.num_heads, 2)

    def test_copy_is_independent(self):
        from yane.evolution.attention import AttentionBlock
        g = _make_genome()
        g.attention_block = AttentionBlock(n_inputs=4, head_dim=4, num_heads=2, seed=6)
        gc = g.copy()
        gc.attention_block.W_Q[0][0][0] = 999.9
        self.assertNotAlmostEqual(g.attention_block.W_Q[0][0][0], 999.9)


# ---------------------------------------------------------------------------
# Acceptance criterion 3: Checkpoint round-trip
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestAttentionCheckpoint(unittest.TestCase):

    def test_pickle_preserves_head_dim_and_num_heads(self):
        """Checkpoint must preserve head_dim and num_heads."""
        from yane.evolution.attention import AttentionBlock
        g = _make_genome()
        g.attention_block = AttentionBlock(n_inputs=4, head_dim=6, num_heads=3, seed=7)
        g2 = pickle.loads(pickle.dumps(g))
        self.assertIsNotNone(g2.attention_block)
        self.assertEqual(g2.attention_block.head_dim, 6)
        self.assertEqual(g2.attention_block.num_heads, 3)

    def test_pickle_preserves_weights(self):
        from yane.evolution.attention import AttentionBlock
        g = _make_genome()
        g.attention_block = AttentionBlock(n_inputs=4, head_dim=4, num_heads=2, seed=8)
        orig_w = g.attention_block.W_Q[0][0][0]
        g2 = pickle.loads(pickle.dumps(g))
        self.assertAlmostEqual(g2.attention_block.W_Q[0][0][0], orig_w)

    def test_pickle_genome_without_attention(self):
        g = _make_genome()
        g2 = pickle.loads(pickle.dumps(g))
        self.assertIsNone(g2.attention_block)


# ---------------------------------------------------------------------------
# NeuroEvolution integration
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestNeuroEvolutionAttention(unittest.TestCase):

    def test_set_attention_sets_flag(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_attention(enabled=True, n_inputs=4, head_dim=4, num_heads=2)
        self.assertTrue(ne._attention_enabled)

    def test_set_attention_disabled(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_attention(enabled=False)
        self.assertFalse(ne._attention_enabled)

    def test_attention_n_inputs(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_attention(n_inputs=8, head_dim=4, num_heads=3)
        self.assertEqual(ne.attention_n_inputs(), 12)

    def test_configure_assigns_attention_block(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_attention(n_inputs=4, head_dim=4, num_heads=2)
        ne.configure(n_inputs=ne.attention_n_inputs(), n_outputs=1, max_nodes=10)
        pop = ne.population
        genomes = list(pop._unevaluated) if pop else []
        if genomes:
            self.assertIsNotNone(genomes[0].attention_block)

    def test_train_with_attention_no_crash(self):
        import yane
        ne = yane.NeuroEvolution(seed=0)
        ne.set_attention(n_inputs=4, head_dim=4, num_heads=2)
        ne.configure(n_inputs=ne.attention_n_inputs(), n_outputs=1, max_nodes=10, max_connections=20)
        ne.set_max_iterations(5)
        ne.train(lambda g: sum(g.forward([0.5, 0.3, -0.2, 0.8])))

    def test_no_attention_when_disabled(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.configure(n_inputs=4, n_outputs=1, max_nodes=10)
        pop = ne.population
        genomes = list(pop._unevaluated) if pop else []
        for g in genomes:
            self.assertIsNone(g.attention_block)


if __name__ == "__main__":
    unittest.main()
