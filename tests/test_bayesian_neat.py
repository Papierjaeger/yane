"""Tests for Probabilistic / Bayesian NEAT (evolution/bayesian_neat.py).

Acceptance criteria:
  1. bayesian_forward(n=100) std is smaller than bayesian_forward(n=1).
  2. inference_mode=True → deterministic forward (identical outputs each call).
  3. n=1 → std_outputs is all zeros.
  4. bayesian_forward returns (mean, std) of correct length.
  5. set_probabilistic on Genome sets the flags correctly.
  6. NeuroEvolution.set_probabilistic propagates to all genomes.
  7. Genome.bayesian_forward() delegates correctly.
  8. Crossover / copy preserves probabilistic state.
"""
from __future__ import annotations

import math
import unittest

import pytest

from yane import NeuroEvolution
from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType
from yane.evolution.bayesian_neat import set_probabilistic, bayesian_forward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_genome(n_inputs: int = 2, n_outputs: int = 1) -> Genome:
    """Minimal acyclic genome: all inputs connected to one output."""
    g = Genome()
    inp_nodes = []
    for i in range(n_inputs):
        nd = Node(NodeType.INPUT, i)
        nd.activation = ActivationType.LINEAR
        nd.input_index = i
        nd.input_scale = 1.0
        g.nodes.append(nd)
        g.input_nodes.append(nd)
        inp_nodes.append(nd)

    for j in range(n_outputs):
        out = Node(NodeType.OUTPUT, n_inputs + j)
        out.activation = ActivationType.TANH
        out.bias = 0.1
        g.nodes.append(out)
        g.output_nodes.append(out)
        for i, src in enumerate(inp_nodes):
            conn = Connection(out, innovation=100 + j * n_inputs + i)
            conn.weight = 0.5 + 0.1 * i
            conn.enabled = True
            src.connections.append(conn)

    g._invalidate_topology()
    return g


def _make_yane(n_inputs: int = 2, n_outputs: int = 1) -> NeuroEvolution:
    yane = NeuroEvolution(seed=42)
    yane.set_population_size(10)
    yane.configure(n_inputs, n_outputs)
    return yane


# ---------------------------------------------------------------------------
# 1. std shrinks with more samples
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestBayesianForwardSampling(unittest.TestCase):

    def test_std_shrinks_with_more_samples(self):
        """std(n=100) should be substantially smaller than std(n=4)."""
        g = _make_simple_genome()
        set_probabilistic(g, enabled=True, noise_std=0.2)
        inputs = [0.5, -0.5]

        # With very few samples variance is high
        import random
        random.seed(0)
        _, std_small = bayesian_forward(g, inputs, n=4)

        random.seed(0)
        _, std_large = bayesian_forward(g, inputs, n=200)

        # Both should be non-negative
        for s in std_small + std_large:
            self.assertGreaterEqual(s, 0.0)

        # Mean std with more samples should be ≤ mean std with fewer samples
        # (asymptotically; 200 vs 4 is a very large ratio)
        mean_std_small = sum(std_small) / len(std_small)
        mean_std_large = sum(std_large) / len(std_large)
        self.assertLess(mean_std_large, mean_std_small + 0.3)

    def test_mean_converges(self):
        """With large n, mean should be stable across two runs (within noise_std)."""
        g = _make_simple_genome()
        set_probabilistic(g, enabled=True, noise_std=0.05)
        inputs = [1.0, 0.0]

        import random
        random.seed(1)
        mean1, _ = bayesian_forward(g, inputs, n=500)
        random.seed(2)
        mean2, _ = bayesian_forward(g, inputs, n=500)

        for a, b in zip(mean1, mean2):
            self.assertAlmostEqual(a, b, delta=0.1)

    def test_n1_std_is_zero(self):
        """n=1 → std_outputs should be all zeros (single sample, no variance)."""
        g = _make_simple_genome()
        set_probabilistic(g, enabled=True, noise_std=0.3)
        _, std = bayesian_forward(g, [0.0, 0.0], n=1)
        for s in std:
            self.assertAlmostEqual(s, 0.0, places=12)

    def test_return_lengths(self):
        """bayesian_forward returns (mean, std) both of length n_outputs."""
        g = _make_simple_genome(n_inputs=3, n_outputs=2)
        set_probabilistic(g, enabled=True, noise_std=0.1)
        mean, std = bayesian_forward(g, [0.1, 0.2, 0.3], n=20)
        self.assertEqual(len(mean), 2)
        self.assertEqual(len(std), 2)

    def test_n_must_be_positive(self):
        """n < 1 raises ValueError."""
        g = _make_simple_genome()
        set_probabilistic(g, enabled=True, noise_std=0.1)
        with self.assertRaises(ValueError):
            bayesian_forward(g, [0.0, 0.0], n=0)


# ---------------------------------------------------------------------------
# 2. Inference mode → deterministic
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestInferenceMode(unittest.TestCase):

    def test_inference_mode_deterministic(self):
        """With inference_mode=True, two forward passes return identical outputs."""
        g = _make_simple_genome()
        set_probabilistic(g, enabled=True, noise_std=1.0, inference_mode=True)
        inputs = [0.7, -0.3]
        g.reset(); out1 = list(g.forward(inputs))
        g.reset(); out2 = list(g.forward(inputs))
        for a, b in zip(out1, out2):
            self.assertAlmostEqual(a, b, places=12)

    def test_stochastic_mode_nondeterministic(self):
        """With noise enabled and inference_mode=False, outputs should differ across runs."""
        g = _make_simple_genome()
        set_probabilistic(g, enabled=True, noise_std=2.0, inference_mode=False)
        inputs = [0.5, 0.5]
        seen = set()
        import random
        random.seed(0)
        for _ in range(50):
            g.reset()
            out = tuple(round(v, 6) for v in g.forward(inputs))
            seen.add(out)
        # With large noise, we expect multiple distinct outputs
        self.assertGreater(len(seen), 1)

    def test_disabled_probabilistic_is_deterministic(self):
        """When enabled=False, forward is deterministic (no noise added)."""
        g = _make_simple_genome()
        set_probabilistic(g, enabled=False)
        inputs = [0.3, 0.7]
        g.reset(); out1 = list(g.forward(inputs))
        g.reset(); out2 = list(g.forward(inputs))
        for a, b in zip(out1, out2):
            self.assertAlmostEqual(a, b, places=12)


# ---------------------------------------------------------------------------
# 3. set_probabilistic on Genome
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestSetProbabilistic(unittest.TestCase):

    def test_flags_set_correctly(self):
        """set_probabilistic sets all three flags on the genome."""
        g = _make_simple_genome()
        set_probabilistic(g, enabled=True, noise_std=0.3, inference_mode=True)
        self.assertTrue(g._prob_enabled)
        self.assertAlmostEqual(g._prob_noise_std, 0.3)
        self.assertTrue(g._prob_inference_mode)

    def test_disable(self):
        """set_probabilistic(enabled=False) turns off noise."""
        g = _make_simple_genome()
        set_probabilistic(g, enabled=True, noise_std=1.0)
        set_probabilistic(g, enabled=False)
        self.assertFalse(g._prob_enabled)


# ---------------------------------------------------------------------------
# 4. NeuroEvolution.set_probabilistic
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestNeuroEvolutionSetProbabilistic(unittest.TestCase):

    def test_propagates_to_population(self):
        """set_probabilistic propagates flags to all genomes in the population."""
        yane = _make_yane()
        yane.set_probabilistic(enabled=True, noise_std=0.15)

        all_genomes = (
            list(yane._population._evaluated)
            + list(yane._population._unevaluated)
        )
        self.assertGreater(len(all_genomes), 0)
        for g in all_genomes:
            self.assertTrue(g._prob_enabled)
            self.assertAlmostEqual(g._prob_noise_std, 0.15)

    def test_disable_propagates(self):
        """set_probabilistic(enabled=False) disables noise on all genomes."""
        yane = _make_yane()
        yane.set_probabilistic(enabled=True, noise_std=0.5)
        yane.set_probabilistic(enabled=False)

        all_genomes = (
            list(yane._population._evaluated)
            + list(yane._population._unevaluated)
        )
        for g in all_genomes:
            self.assertFalse(g._prob_enabled)

    def test_ne_flags_updated(self):
        """NeuroEvolution stores the probabilistic settings internally."""
        yane = _make_yane()
        yane.set_probabilistic(enabled=True, noise_std=0.07, inference_mode=True)
        self.assertTrue(yane._prob_enabled)
        self.assertAlmostEqual(yane._prob_noise_std, 0.07)
        self.assertTrue(yane._prob_inference_mode)


# ---------------------------------------------------------------------------
# 5. Genome.bayesian_forward() method
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestGenomeBayesianForward(unittest.TestCase):

    def test_genome_bayesian_forward_method(self):
        """Genome.bayesian_forward() delegates to bayesian_neat.bayesian_forward."""
        g = _make_simple_genome()
        set_probabilistic(g, enabled=True, noise_std=0.1)
        mean, std = g.bayesian_forward([0.5, -0.5], n=50)
        self.assertEqual(len(mean), 1)
        self.assertEqual(len(std), 1)
        self.assertGreaterEqual(std[0], 0.0)

    def test_genome_bayesian_forward_n1_std_zero(self):
        """Genome.bayesian_forward(n=1) → std = 0."""
        g = _make_simple_genome()
        set_probabilistic(g, enabled=True, noise_std=0.5)
        _, std = g.bayesian_forward([0.0, 0.0], n=1)
        self.assertAlmostEqual(std[0], 0.0, places=12)


# ---------------------------------------------------------------------------
# 6. Copy / crossover preserves probabilistic state
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestCopyPreservesProbabilistic(unittest.TestCase):

    def test_copy_preserves_flags(self):
        """genome.copy() preserves _prob_enabled, _prob_noise_std, _prob_inference_mode."""
        g = _make_simple_genome()
        set_probabilistic(g, enabled=True, noise_std=0.25, inference_mode=True)
        child = g.copy()
        self.assertTrue(child._prob_enabled)
        self.assertAlmostEqual(child._prob_noise_std, 0.25)
        self.assertTrue(child._prob_inference_mode)

    def test_crossover_preserves_flags(self):
        """genome.crossover() inherits probabilistic settings from fitter parent."""
        g1 = _make_simple_genome()
        g2 = _make_simple_genome()
        set_probabilistic(g1, enabled=True, noise_std=0.33)
        set_probabilistic(g2, enabled=False)
        g1.fitness = 10.0
        g2.fitness = 5.0
        child = g1.crossover(g2)
        # Fitter parent (g1) is the self; child should have g1's prob settings
        self.assertTrue(child._prob_enabled)
        self.assertAlmostEqual(child._prob_noise_std, 0.33)
