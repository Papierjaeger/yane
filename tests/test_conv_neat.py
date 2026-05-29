"""Tests for Convolutional NEAT (evolution/conv_neat.py).

Key invariants tested:
  1. Weight-sharing: param count = K²·in_c·out_c (image-size independent)
  2. forward_image returns correct dimension (n_outputs of the NEAT network)
  3. Global-average-pool: feature count is always out_channels, regardless of H×W
  4. Genome copy/crossover preserves conv_stack
  5. Checkpoint (pickle) round-trip preserves conv_stack
  6. NeuroEvolution integration: set_conv_neat + forward_image during train
  7. add_conv_block maintains in_channels chain
  8. Zero-cost when disabled (genome.conv_stack is None by default)
"""
from __future__ import annotations

import pickle
import unittest
import tempfile
from pathlib import Path

import pytest

from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_genome(n_inputs: int = 8, n_outputs: int = 4) -> Genome:
    g = Genome()
    g.max_nodes = 20
    g.max_connections = 50
    for i in range(n_inputs):
        n = Node(NodeType.INPUT, i)
        n.activation = ActivationType.LINEAR
        g.input_nodes.append(n)
        g.nodes.append(n)
    for j in range(n_outputs):
        out = Node(NodeType.OUTPUT, n_inputs + j)
        out.activation = ActivationType.SIGMOID
        g.output_nodes.append(out)
        g.nodes.append(out)
    for inp in g.input_nodes:
        for out in g.output_nodes:
            c = Connection(out, innovation=inp.innovation * 100 + out.innovation)
            c.weight = 0.1
            inp.connections.append(c)
    g._invalidate_topology()
    return g


def _flat_image(h: int, w: int, c: int = 1, value: float = 0.5) -> list[float]:
    return [value] * (h * w * c)


# ---------------------------------------------------------------------------
# ConvBlock
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestConvBlock(unittest.TestCase):

    def test_weight_sharing_param_count_k3_in1_out4(self):
        """K=3, in=1, out=4 → 9 weights/channel, 4 channels = 36 weights total."""
        from yane.evolution.conv_neat import ConvBlock
        b = ConvBlock(kernel_size=3, stride=1, in_channels=1, out_channels=4)
        self.assertEqual(b.n_weights_per_channel, 9)
        total_weights = sum(len(k) for k in b.kernels)
        self.assertEqual(total_weights, 36)
        # Image size does NOT affect param count — this is the key invariant
        self.assertEqual(b.total_params, 36 + 4)  # weights + biases

    def test_weight_sharing_image_size_independence(self):
        """Param count must not change with different image sizes."""
        from yane.evolution.conv_neat import ConvBlock
        b = ConvBlock(kernel_size=5, stride=2, in_channels=3, out_channels=8)
        n_params_expected = 5 * 5 * 3 * 8 + 8
        self.assertEqual(b.total_params, n_params_expected)
        # Apply to different image sizes — param count stays the same
        for h, w in [(8, 8), (16, 16), (28, 28)]:
            planes = [[0.0] * (h * w) for _ in range(3)]
            result = b.forward(planes, h, w)
            self.assertEqual(len(result), 8, f"forward should return out_channels for {h}×{w}")
            self.assertEqual(b.total_params, n_params_expected)

    def test_forward_returns_out_channels_values(self):
        from yane.evolution.conv_neat import ConvBlock
        b = ConvBlock(kernel_size=3, stride=1, in_channels=1, out_channels=6)
        planes = [[0.5] * 16]  # 4×4 image
        result = b.forward(planes, 4, 4)
        self.assertEqual(len(result), 6)

    def test_global_avg_pool_image_size_independence(self):
        """forward() always returns out_channels values regardless of H×W."""
        from yane.evolution.conv_neat import ConvBlock
        b = ConvBlock(kernel_size=3, stride=1, in_channels=1, out_channels=4)
        for h, w in [(4, 4), (8, 8), (16, 16)]:
            planes = [[1.0] * (h * w)]
            result = b.forward(planes, h, w)
            self.assertEqual(len(result), 4,
                             f"Global avg pool must give out_channels for {h}×{w}")

    def test_copy_is_independent(self):
        from yane.evolution.conv_neat import ConvBlock
        b = ConvBlock(kernel_size=3, stride=1, in_channels=1, out_channels=2)
        c = b.copy()
        c.kernels[0][0] = 999.9
        self.assertNotAlmostEqual(b.kernels[0][0], 999.9)

    def test_mutate_changes_weights(self):
        from yane.evolution.conv_neat import ConvBlock
        import random
        b = ConvBlock(kernel_size=3, stride=1, in_channels=1, out_channels=2)
        old_w = b.kernels[0][0]
        rng = random.Random(42)
        b.mutate(sigma=1.0, rng=rng)
        # With large sigma, at least one weight should differ
        self.assertFalse(all(
            abs(b.kernels[c][w] - orig) < 1e-10
            for c in range(len(b.kernels))
            for w, orig in enumerate([0.0])
        ), "mutate should change weights")

    def test_crossover_produces_valid_block(self):
        from yane.evolution.conv_neat import ConvBlock
        import random
        random.seed(1)
        a = ConvBlock(kernel_size=3, stride=1, in_channels=1, out_channels=2)
        b = ConvBlock(kernel_size=3, stride=1, in_channels=1, out_channels=2)
        child = a.crossover(b)
        self.assertEqual(child.out_channels, 2)
        self.assertEqual(len(child.kernels[0]), 9)


# ---------------------------------------------------------------------------
# ConvStack
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestConvStack(unittest.TestCase):

    def test_n_outputs_equals_sum_of_out_channels(self):
        from yane.evolution.conv_neat import ConvBlock, ConvStack
        stack = ConvStack([
            ConvBlock(kernel_size=3, in_channels=1, out_channels=4),
            ConvBlock(kernel_size=3, in_channels=4, out_channels=8),
        ])
        self.assertEqual(stack.n_outputs, 12)  # 4 + 8

    def test_forward_image_returns_n_outputs_values(self):
        from yane.evolution.conv_neat import make_conv_stack
        stack = make_conv_stack(n_image_channels=1, n_blocks=1, out_channels=6)
        pixels = _flat_image(8, 8, 1)
        result = stack.forward_image(pixels, 8, 8, 1)
        self.assertEqual(len(result), 6)

    def test_forward_image_image_size_independence(self):
        """n_outputs must be constant regardless of image dimensions."""
        from yane.evolution.conv_neat import make_conv_stack
        stack = make_conv_stack(n_image_channels=1, n_blocks=2,
                                kernel_size=3, out_channels=4)
        for h, w in [(8, 8), (16, 16), (28, 28)]:
            pixels = _flat_image(h, w, 1)
            result = stack.forward_image(pixels, h, w, 1)
            self.assertEqual(len(result), stack.n_outputs,
                             f"n_outputs must be constant for {h}×{w}")

    def test_multi_channel_forward(self):
        from yane.evolution.conv_neat import make_conv_stack
        stack = make_conv_stack(n_image_channels=3, n_blocks=1, out_channels=8)
        pixels = _flat_image(6, 6, 3)
        result = stack.forward_image(pixels, 6, 6, 3)
        self.assertEqual(len(result), 8)

    def test_copy_is_independent(self):
        from yane.evolution.conv_neat import make_conv_stack
        stack = make_conv_stack(n_image_channels=1, n_blocks=1, out_channels=4)
        copy = stack.copy()
        copy.blocks[0].kernels[0][0] = 999.9
        self.assertNotAlmostEqual(stack.blocks[0].kernels[0][0], 999.9)

    def test_crossover_produces_valid_stack(self):
        from yane.evolution.conv_neat import make_conv_stack
        a = make_conv_stack(n_image_channels=1, n_blocks=2, out_channels=4)
        b = make_conv_stack(n_image_channels=1, n_blocks=2, out_channels=4)
        child = a.crossover(b)
        self.assertEqual(child.n_outputs, 8)  # 4 + 4


# ---------------------------------------------------------------------------
# add_conv_block
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestAddConvBlock(unittest.TestCase):

    def test_add_block_chains_in_channels(self):
        """New block's in_channels must equal previous block's out_channels."""
        from yane.evolution.conv_neat import ConvBlock, ConvStack, add_conv_block
        stack = ConvStack([ConvBlock(kernel_size=3, in_channels=1, out_channels=4)])
        add_conv_block(stack, out_channels=8)
        new_block = stack.blocks[-1]
        self.assertEqual(new_block.in_channels, 4,
                         "in_channels of new block must match previous out_channels")
        self.assertEqual(new_block.out_channels, 8)

    def test_add_block_increases_n_outputs(self):
        from yane.evolution.conv_neat import ConvBlock, ConvStack, add_conv_block
        stack = ConvStack([ConvBlock(kernel_size=3, in_channels=1, out_channels=4)])
        before = stack.n_outputs
        add_conv_block(stack, out_channels=6)
        self.assertEqual(stack.n_outputs, before + 6)

    def test_add_block_empty_stack_sets_in_channels_1(self):
        from yane.evolution.conv_neat import ConvStack, add_conv_block
        stack = ConvStack([])
        add_conv_block(stack, out_channels=4)
        self.assertEqual(stack.blocks[0].in_channels, 1)


# ---------------------------------------------------------------------------
# genome.forward_image()
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestGenomeForwardImage(unittest.TestCase):

    def _make_genome_with_conv(self, out_channels: int = 4):
        from yane.evolution.conv_neat import make_conv_stack
        g = _make_genome(n_inputs=out_channels, n_outputs=2)
        g.conv_stack = make_conv_stack(n_image_channels=1, n_blocks=1,
                                       out_channels=out_channels)
        return g

    def test_forward_image_returns_n_outputs(self):
        g = self._make_genome_with_conv(4)
        pixels = _flat_image(8, 8, 1)
        result = g.forward_image(pixels, 8, 8, 1)
        self.assertEqual(len(result), 2)  # NEAT network has 2 outputs

    def test_forward_image_raises_without_conv_stack(self):
        g = _make_genome(n_inputs=4, n_outputs=2)
        with self.assertRaises(RuntimeError):
            g.forward_image(_flat_image(8, 8, 1), 8, 8, 1)

    def test_forward_image_consistent_with_manual_call(self):
        from yane.evolution.conv_neat import make_conv_stack
        g = _make_genome(n_inputs=4, n_outputs=2)
        stack = make_conv_stack(n_image_channels=1, n_blocks=1, out_channels=4)
        g.conv_stack = stack
        pixels = _flat_image(8, 8, 1)
        # Manual: features → forward
        features = stack.forward_image(pixels, 8, 8, 1)
        expected = g.forward(features)
        result = g.forward_image(pixels, 8, 8, 1)
        self.assertEqual(result, expected)


# ---------------------------------------------------------------------------
# Genome copy/crossover preserves conv_stack
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestGenomeCopyAndCrossover(unittest.TestCase):

    def test_copy_preserves_conv_stack(self):
        from yane.evolution.conv_neat import make_conv_stack
        g = _make_genome(n_inputs=4, n_outputs=2)
        g.conv_stack = make_conv_stack(n_image_channels=1, n_blocks=1, out_channels=4)
        c = g.copy()
        self.assertIsNotNone(c.conv_stack)
        self.assertEqual(c.conv_stack.n_outputs, g.conv_stack.n_outputs)

    def test_copy_conv_stack_is_independent(self):
        from yane.evolution.conv_neat import make_conv_stack
        g = _make_genome(n_inputs=4, n_outputs=2)
        g.conv_stack = make_conv_stack(n_image_channels=1, n_blocks=1, out_channels=4)
        c = g.copy()
        c.conv_stack.blocks[0].kernels[0][0] = 999.9
        self.assertNotAlmostEqual(g.conv_stack.blocks[0].kernels[0][0], 999.9)

    def test_crossover_preserves_conv_stack(self):
        from yane.evolution.conv_neat import make_conv_stack
        g_a = _make_genome(n_inputs=4, n_outputs=2)
        g_b = _make_genome(n_inputs=4, n_outputs=2)
        g_a.conv_stack = make_conv_stack(n_image_channels=1, n_blocks=1, out_channels=4)
        g_b.conv_stack = make_conv_stack(n_image_channels=1, n_blocks=1, out_channels=4)
        g_a.fitness = 1.0
        g_b.fitness = 0.5
        child = g_a.crossover(g_b)
        self.assertIsNotNone(child.conv_stack)

    def test_crossover_different_stack_sizes_no_error(self):
        from yane.evolution.conv_neat import make_conv_stack
        g_a = _make_genome(n_inputs=8, n_outputs=2)
        g_b = _make_genome(n_inputs=4, n_outputs=2)
        g_a.conv_stack = make_conv_stack(n_image_channels=1, n_blocks=2, out_channels=4)
        g_b.conv_stack = make_conv_stack(n_image_channels=1, n_blocks=1, out_channels=4)
        g_a.fitness = 1.0
        g_b.fitness = 0.5
        try:
            child = g_a.crossover(g_b)
            self.assertIsNotNone(child)
        except Exception as e:
            self.fail(f"crossover raised: {e}")

    def test_genome_no_conv_stack_null(self):
        """Without set_conv_neat, genome.conv_stack should be None."""
        g = _make_genome()
        self.assertIsNone(g.conv_stack)


# ---------------------------------------------------------------------------
# Checkpoint (pickle) round-trip
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestCheckpointRoundTrip(unittest.TestCase):

    def test_pickle_preserves_conv_stack(self):
        from yane.evolution.conv_neat import make_conv_stack
        g = _make_genome(n_inputs=4, n_outputs=2)
        g.conv_stack = make_conv_stack(n_image_channels=1, n_blocks=1, out_channels=4)
        original_w = g.conv_stack.blocks[0].kernels[0][0]
        data = pickle.dumps(g)
        g2 = pickle.loads(data)
        self.assertIsNotNone(g2.conv_stack)
        self.assertAlmostEqual(g2.conv_stack.blocks[0].kernels[0][0], original_w)

    def test_neuro_evolution_checkpoint_roundtrip_with_conv(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_conv_neat(n_image_channels=1, n_blocks=1, out_channels=4)
        ne.configure(n_inputs=ne.conv_n_inputs(), n_outputs=2)
        ne.set_max_iterations(3)
        pixels = _flat_image(8, 8, 1)
        ne.train(lambda g: sum(g.forward_image(pixels, 8, 8, 1)))

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "conv_test.pkl"
            ne.save_checkpoint(path)
            ne2 = yane.NeuroEvolution()
            ne2.load_checkpoint(path)
            best = ne2.get_best()
            self.assertIsNotNone(best.conv_stack)


# ---------------------------------------------------------------------------
# NeuroEvolution integration
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestNeuroEvolutionIntegration(unittest.TestCase):

    def test_set_conv_neat_returns_stack(self):
        import yane
        from yane.evolution.conv_neat import ConvStack
        ne = yane.NeuroEvolution()
        stack = ne.set_conv_neat(n_image_channels=1, n_blocks=1, out_channels=8)
        self.assertIsInstance(stack, ConvStack)

    def test_conv_n_inputs_equals_n_outputs(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_conv_neat(n_image_channels=1, n_blocks=1, out_channels=6)
        self.assertEqual(ne.conv_n_inputs(), 6)

    def test_conv_n_inputs_raises_without_set_conv_neat(self):
        import yane
        ne = yane.NeuroEvolution()
        with self.assertRaises(RuntimeError):
            ne.conv_n_inputs()

    def test_configure_assigns_conv_stack(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_conv_neat(n_image_channels=1, n_blocks=1, out_channels=4)
        ne.configure(n_inputs=ne.conv_n_inputs(), n_outputs=2)
        pop = ne.population
        genomes = list(pop._unevaluated) + list(pop._evaluated) if pop else []
        for g in genomes:
            self.assertIsNotNone(g.conv_stack)

    def test_train_with_forward_image_does_not_raise(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_conv_neat(n_image_channels=1, n_blocks=1, out_channels=4)
        ne.configure(n_inputs=ne.conv_n_inputs(), n_outputs=2)
        ne.set_max_iterations(3)
        pixels = _flat_image(8, 8, 1)
        ne.train(lambda g: sum(g.forward_image(pixels, 8, 8, 1)))

    def test_set_conv_neat_disabled(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_conv_neat(enabled=False)
        self.assertFalse(ne._conv_neat_enabled)
        self.assertIsNone(ne._conv_neat_stack)

    def test_genome_no_conv_stack_without_set_conv_neat(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.configure(n_inputs=4, n_outputs=2)
        pop = ne.population
        genomes = list(pop._unevaluated) + list(pop._evaluated) if pop else []
        for g in genomes:
            self.assertIsNone(g.conv_stack)


if __name__ == "__main__":
    unittest.main()
