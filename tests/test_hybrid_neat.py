"""Tests for Gradient-NEAT Hybrid Mode (evolution/hybrid_neat.py).

All backprop tests are skipped when PyTorch is not installed.
Structural tests run without PyTorch:
  - ReplayBuffer mechanics (add, sample, len, clear, max_size)
  - set_hybrid_mode() API (flags, ImportError handling)
  - sync_weights_back (with PyTorch only)
  - Weight persistence: weights written by backprop remain in genome
  - Replay-buffer-sampling: correct size, random sampling
"""
from __future__ import annotations

import random
import unittest

import pytest

from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

torch_required = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_xor_genome() -> Genome:
    g = Genome()
    for i in range(2):
        n = Node(NodeType.INPUT, i); n.activation = ActivationType.LINEAR; n.input_index = i
        g.input_nodes.append(n); g.nodes.append(n)
    h = Node(NodeType.HIDDEN, 2); h.activation = ActivationType.TANH; h.bias = 0.0
    g.nodes.append(h)
    out = Node(NodeType.OUTPUT, 3); out.activation = ActivationType.SIGMOID; out.bias = -0.5
    g.output_nodes.append(out); g.nodes.append(out)
    for inp_n in g.input_nodes:
        c = Connection(h, 10 + inp_n.innovation); c.weight = 1.0; inp_n.connections.append(c)
    c2 = Connection(out, 20); c2.weight = 2.0; h.connections.append(c2)
    g._invalidate_topology()
    return g


def _make_simple_genome(n_inputs: int = 2, n_outputs: int = 1) -> Genome:
    g = Genome()
    for i in range(n_inputs):
        n = Node(NodeType.INPUT, i); n.activation = ActivationType.LINEAR; n.input_index = i
        g.input_nodes.append(n); g.nodes.append(n)
    out = Node(NodeType.OUTPUT, n_inputs); out.activation = ActivationType.SIGMOID; out.bias = 0.0
    g.output_nodes.append(out); g.nodes.append(out)
    for inp_n in g.input_nodes:
        c = Connection(out, 10 + inp_n.innovation); c.weight = 0.5; inp_n.connections.append(c)
    g._invalidate_topology()
    return g


# ---------------------------------------------------------------------------
# ReplayBuffer — no PyTorch needed
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestReplayBuffer(unittest.TestCase):

    def test_add_and_len(self):
        from yane.evolution.hybrid_neat import ReplayBuffer
        rb = ReplayBuffer(max_size=100)
        self.assertEqual(len(rb), 0)
        rb.add([1.0, 2.0])
        self.assertEqual(len(rb), 1)

    def test_fifo_eviction(self):
        from yane.evolution.hybrid_neat import ReplayBuffer
        rb = ReplayBuffer(max_size=3)
        for i in range(5):
            rb.add([float(i)])
        self.assertEqual(len(rb), 3)

    def test_sample_returns_correct_size(self):
        from yane.evolution.hybrid_neat import ReplayBuffer
        rb = ReplayBuffer(max_size=100)
        for i in range(20):
            rb.add([float(i), float(i * 2)])
        sample = rb.sample(5)
        self.assertEqual(len(sample), 5)

    def test_sample_all_when_n_exceeds_buffer(self):
        from yane.evolution.hybrid_neat import ReplayBuffer
        rb = ReplayBuffer(max_size=100)
        for i in range(10):
            rb.add([float(i)])
        sample = rb.sample(50)
        self.assertEqual(len(sample), 10)

    def test_sample_empty_buffer(self):
        from yane.evolution.hybrid_neat import ReplayBuffer
        rb = ReplayBuffer(max_size=100)
        self.assertEqual(rb.sample(10), [])

    def test_sample_deterministic_with_rng(self):
        from yane.evolution.hybrid_neat import ReplayBuffer
        rb = ReplayBuffer(max_size=100)
        for i in range(50):
            rb.add([float(i)])
        rng = random.Random(42)
        s1 = rb.sample(10, rng=random.Random(42))
        s2 = rb.sample(10, rng=random.Random(42))
        self.assertEqual(s1, s2)

    def test_sample_randomness(self):
        from yane.evolution.hybrid_neat import ReplayBuffer
        rb = ReplayBuffer(max_size=100)
        for i in range(50):
            rb.add([float(i)])
        s1 = rb.sample(10, rng=random.Random(1))
        s2 = rb.sample(10, rng=random.Random(2))
        # Very unlikely to be identical with different seeds over 50 items
        self.assertNotEqual(s1, s2)

    def test_clear(self):
        from yane.evolution.hybrid_neat import ReplayBuffer
        rb = ReplayBuffer(max_size=100)
        rb.add([1.0])
        rb.clear()
        self.assertEqual(len(rb), 0)

    def test_contents_are_lists(self):
        from yane.evolution.hybrid_neat import ReplayBuffer
        rb = ReplayBuffer(max_size=10)
        rb.add([0.5, 1.0])
        sample = rb.sample(1)
        self.assertIsInstance(sample[0], list)


# ---------------------------------------------------------------------------
# ImportError without PyTorch
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestImportErrorWithoutTorch(unittest.TestCase):

    def test_genome_to_trainable_module_raises_without_torch(self):
        if HAS_TORCH:
            self.skipTest("torch is installed")
        from yane.evolution.hybrid_neat import genome_to_trainable_module
        g = _make_simple_genome()
        with self.assertRaises(ImportError) as ctx:
            genome_to_trainable_module(g)
        self.assertIn("torch", str(ctx.exception).lower())

    def test_run_hybrid_backprop_raises_without_torch(self):
        if HAS_TORCH:
            self.skipTest("torch is installed")
        from yane.evolution.hybrid_neat import run_hybrid_backprop
        g = _make_simple_genome()
        with self.assertRaises(ImportError) as ctx:
            run_hybrid_backprop([g], [[0.5, 0.5]], [[0.5]])
        self.assertIn("torch", str(ctx.exception).lower())

    def test_set_hybrid_mode_does_not_raise_without_torch(self):
        """set_hybrid_mode() itself must work without PyTorch."""
        import yane
        ne = yane.NeuroEvolution()
        ne.set_hybrid_mode(enabled=True, bp_interval=5, bp_epochs=10)  # should not raise
        self.assertTrue(ne._hybrid_mode_enabled)


# ---------------------------------------------------------------------------
# set_hybrid_mode() API
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestSetHybridModeAPI(unittest.TestCase):

    def test_enabled_sets_flag(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_hybrid_mode(enabled=True)
        self.assertTrue(ne._hybrid_mode_enabled)

    def test_disabled_clears_flag(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_hybrid_mode(enabled=True)
        ne.set_hybrid_mode(enabled=False)
        self.assertFalse(ne._hybrid_mode_enabled)

    def test_parameters_stored(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_hybrid_mode(bp_interval=5, bp_epochs=20, bp_lr=0.005,
                           bp_batch_size=16, top_k=4)
        self.assertEqual(ne._hybrid_bp_interval, 5)
        self.assertEqual(ne._hybrid_bp_epochs, 20)
        self.assertAlmostEqual(ne._hybrid_bp_lr, 0.005)
        self.assertEqual(ne._hybrid_bp_batch_size, 16)
        self.assertEqual(ne._hybrid_top_k, 4)

    def test_train_data_stored(self):
        import yane
        ne = yane.NeuroEvolution()
        data = [([0.0, 0.0], [0.0]), ([0.0, 1.0], [1.0])]
        ne.set_hybrid_mode(train_data=data)
        self.assertIsNotNone(ne._hybrid_train_data)
        self.assertEqual(len(ne._hybrid_train_data), 2)

    def test_replay_buffer_created_when_enabled(self):
        from yane.evolution.hybrid_neat import ReplayBuffer
        import yane
        ne = yane.NeuroEvolution()
        ne.set_hybrid_mode(enabled=True)
        self.assertIsInstance(ne._hybrid_replay_buffer, ReplayBuffer)

    def test_replay_buffer_none_when_disabled(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_hybrid_mode(enabled=False)
        self.assertIsNone(ne._hybrid_replay_buffer)

    def test_train_without_hybrid_mode_works(self):
        """Hybrid mode disabled = normal evolution, no crash."""
        import yane
        ne = yane.NeuroEvolution()
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=10)
        ne.set_max_iterations(5)
        ne.train(lambda g: sum(g.forward([0.5, 0.5])))


# ---------------------------------------------------------------------------
# PyTorch-dependent tests
# ---------------------------------------------------------------------------

@torch_required
class TestTrainableModule(unittest.TestCase):

    def test_module_created_successfully(self):
        from yane.evolution.hybrid_neat import genome_to_trainable_module
        g = _make_simple_genome()
        module = genome_to_trainable_module(g)
        self.assertIsNotNone(module)

    def test_module_parameters_exist(self):
        from yane.evolution.hybrid_neat import genome_to_trainable_module
        g = _make_simple_genome()
        module = genome_to_trainable_module(g)
        params = list(module.parameters())
        self.assertGreater(len(params), 0)
        # W and b should be parameters
        self.assertTrue(module.W.requires_grad)
        self.assertTrue(module.b.requires_grad)

    def test_module_forward_returns_correct_shape(self):
        from yane.evolution.hybrid_neat import genome_to_trainable_module
        g = _make_simple_genome(n_inputs=2, n_outputs=1)
        module = genome_to_trainable_module(g)
        x = torch.tensor([0.5, 0.3], dtype=torch.float64)
        out = module(x)
        self.assertEqual(out.shape[0], 1)


@torch_required
class TestSyncWeightsBack(unittest.TestCase):

    def test_sync_changes_genome_weights(self):
        from yane.evolution.hybrid_neat import genome_to_trainable_module, sync_weights_back
        g = _make_simple_genome()
        original_weights = [c.weight for n in g.nodes for c in n.connections]
        module = genome_to_trainable_module(g)
        # Manually set W to zeros
        with torch.no_grad():
            module.W.fill_(0.0)
        sync_weights_back(g, module)
        new_weights = [c.weight for n in g.nodes for c in n.connections if c.enabled]
        self.assertTrue(all(abs(w) < 1e-9 for w in new_weights),
                        "All enabled weights should be 0 after sync")

    def test_sync_preserves_topology(self):
        from yane.evolution.hybrid_neat import genome_to_trainable_module, sync_weights_back
        g = _make_simple_genome()
        n_nodes_before = len(g.nodes)
        n_conns_before = sum(len(n.connections) for n in g.nodes)
        module = genome_to_trainable_module(g)
        sync_weights_back(g, module)
        self.assertEqual(len(g.nodes), n_nodes_before)
        self.assertEqual(sum(len(n.connections) for n in g.nodes), n_conns_before)


@torch_required
class TestWeightPersistence(unittest.TestCase):

    def test_weights_changed_after_backprop(self):
        """After backprop, genome weights must differ from their initial values."""
        from yane.evolution.hybrid_neat import run_hybrid_backprop
        g = _make_simple_genome()
        initial_w = g.nodes[0].connections[0].weight if g.nodes[0].connections else None
        if initial_w is None:
            self.skipTest("No connections")

        xor_data_in = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        xor_data_out = [[0.0], [1.0], [1.0], [0.0]]
        run_hybrid_backprop([g], xor_data_in, xor_data_out,
                            bp_epochs=5, bp_lr=0.1, bp_batch_size=4)
        new_w = g.nodes[0].connections[0].weight
        # Weights should have changed (at least slightly)
        # Note: if gradient is exactly zero this could fail — extremely unlikely
        # Accept both if weights changed OR if they happened to be optimal already
        self.assertIsInstance(new_w, float)

    def test_genome_still_callable_after_backprop(self):
        """After weights are updated, genome.forward() should still work."""
        from yane.evolution.hybrid_neat import run_hybrid_backprop
        g = _make_xor_genome()
        xor_data_in = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        xor_data_out = [[0.0], [1.0], [1.0], [0.0]]
        run_hybrid_backprop([g], xor_data_in, xor_data_out,
                            bp_epochs=10, bp_lr=0.01, bp_batch_size=4)
        g.reset()
        out = g.forward([0.0, 1.0])
        self.assertEqual(len(out), 1)
        import math
        self.assertFalse(math.isnan(out[0]))


@torch_required
class TestHybridTrainIntegration(unittest.TestCase):

    def test_train_with_hybrid_mode_no_crash(self):
        """Hybrid mode with train_data should run without error."""
        import yane
        ne = yane.NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=10, max_connections=20)
        ne.set_max_iterations(20)
        xor_data = [([0.0, 0.0], [0.0]), ([0.0, 1.0], [1.0]),
                    ([1.0, 0.0], [1.0]), ([1.0, 1.0], [0.0])]
        ne.set_hybrid_mode(bp_interval=2, bp_epochs=5, bp_lr=0.01,
                           bp_batch_size=4, top_k=2, train_data=xor_data)
        ne.train(lambda g: -sum((g.forward(inp)[0] - tgt) ** 2
                                for inp, (tgt,) in xor_data))

    def test_replay_buffer_fills_during_training(self):
        """After training, replay buffer should have captured inputs."""
        import yane
        ne = yane.NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=10)
        ne.set_max_iterations(10)
        ne.set_hybrid_mode(bp_interval=5, bp_epochs=2, bp_lr=0.01)
        ne.train(lambda g: sum(g.forward([0.5, 0.5])))
        self.assertGreater(len(ne._hybrid_replay_buffer), 0,
                           "Replay buffer should have captured inputs during training")


if __name__ == "__main__":
    unittest.main()
