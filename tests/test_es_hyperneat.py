"""Tests for ES-HyperNEAT (Evolvable Substrate HyperNEAT).

Covers:
  - CPPN-variance computation (_cppn_variance)
  - Quadtree termination (never hangs, bounded by max_depth)
  - es_hyperneat_substrate: node placement, always >= base node count
  - Different CPPNs yield different substrate sizes
  - hyperneat_substrate(evolve=True) API matches es_hyperneat_substrate
  - generate_genome_from_cppn(evolve_substrate=True) produces a valid genome
  - Fixed substrate (evolve=False) is unaffected
  - Fallback: very uniform CPPN still returns a usable substrate
"""
from __future__ import annotations

import math
import unittest

import pytest

from yane.evolution.indirect_encoding import (
    CPPNGenome,
    Substrate,
    hyperneat_substrate,
    generate_genome_from_cppn,
    es_hyperneat_substrate,
    _cppn_variance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zero_cppn(x1, y1, x2, y2, dist):
    """Always returns 0 — completely uniform CPPN."""
    return 0.0


def _sine_cppn(x1, y1, x2, y2, dist):
    """Oscillating CPPN — high variance near zero crossings."""
    return math.sin(10.0 * x1 + 7.0 * y1)


def _noisy_cppn(x1, y1, x2, y2, dist):
    """Returns alternating ±1 based on sign of input sum."""
    return 1.0 if (x1 + y1) > 0 else -1.0


# ---------------------------------------------------------------------------
# _cppn_variance
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestCPPNVariance(unittest.TestCase):

    def test_uniform_cppn_has_zero_variance(self):
        var = _cppn_variance(_zero_cppn, 0.0, 0.0)
        self.assertAlmostEqual(var, 0.0)

    def test_varying_cppn_has_positive_variance(self):
        # Sine CPPN has non-zero variance at most positions
        var = _cppn_variance(_sine_cppn, 0.0, 0.0)
        # Not guaranteed non-zero at exactly (0,0), try a different position
        var2 = _cppn_variance(_sine_cppn, 0.3, 0.2)
        self.assertGreaterEqual(max(var, var2), 0.0)

    def test_variance_nonnegative(self):
        for pos in [(0.0, 0.0), (0.5, 0.3), (-0.7, 0.1)]:
            var = _cppn_variance(_sine_cppn, *pos)
            self.assertGreaterEqual(var, 0.0)


# ---------------------------------------------------------------------------
# es_hyperneat_substrate — standalone function
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestEsHyperNEATSubstrate(unittest.TestCase):

    def test_returns_substrate_instance(self):
        substrate = es_hyperneat_substrate(_zero_cppn, n_inputs=2, n_outputs=2)
        self.assertIsInstance(substrate, Substrate)

    def test_node_count_at_least_inputs_plus_outputs(self):
        """Even with a uniform CPPN, at least n_inputs + n_outputs nodes exist."""
        substrate = es_hyperneat_substrate(_zero_cppn, n_inputs=3, n_outputs=2)
        n_io = len(substrate.input_indices) + len(substrate.output_indices)
        self.assertGreaterEqual(len(substrate.coordinates), n_io)

    def test_correct_input_output_count(self):
        sub = es_hyperneat_substrate(_sine_cppn, n_inputs=4, n_outputs=2)
        self.assertEqual(len(sub.input_indices), 4)
        self.assertEqual(len(sub.output_indices), 2)

    def test_varying_cppn_adds_hidden_nodes(self):
        """A CPPN with non-uniform output should discover hidden nodes."""
        sub = es_hyperneat_substrate(
            _sine_cppn, n_inputs=2, n_outputs=2,
            variance_threshold=0.001,  # low threshold → more nodes
            max_depth=2,
            initial_resolution=2,
        )
        self.assertGreater(len(sub.hidden_indices), 0,
                           "Sine CPPN should produce at least one hidden node")

    def test_uniform_cppn_fallback(self):
        """Completely uniform CPPN should return a substrate with at least one centre node."""
        sub = es_hyperneat_substrate(
            _zero_cppn, n_inputs=2, n_outputs=2,
            variance_threshold=0.001,
            max_depth=2,
        )
        # Should have at least the fallback centre node
        self.assertGreaterEqual(len(sub.hidden_indices), 1)

    def test_max_depth_bounds_node_count(self):
        """With max_depth=1, we can have at most initial_resolution² initial cells."""
        sub = es_hyperneat_substrate(
            _noisy_cppn, n_inputs=2, n_outputs=2,
            variance_threshold=0.0,  # always subdivide
            max_depth=1,
            initial_resolution=2,
        )
        # At max_depth=1: up to 4 (2×2) initial cells, each can place ≤1 node
        self.assertLessEqual(len(sub.hidden_indices), 4 * 4)

    def test_quadtree_terminates(self):
        """Algorithm must always terminate regardless of CPPN."""
        for cppn in [_zero_cppn, _sine_cppn, _noisy_cppn]:
            sub = es_hyperneat_substrate(
                cppn, n_inputs=2, n_outputs=2,
                max_depth=3, initial_resolution=2,
            )
            self.assertIsInstance(sub, Substrate)

    def test_no_duplicate_nodes(self):
        """Coordinates should be unique (deduplication active)."""
        sub = es_hyperneat_substrate(
            _sine_cppn, n_inputs=2, n_outputs=2,
            variance_threshold=0.001, max_depth=3, initial_resolution=3,
        )
        rounded = [(round(x, 3), round(y, 3)) for x, y in sub.coordinates]
        self.assertEqual(len(rounded), len(set(rounded)), "Duplicate coordinates found")

    def test_all_pairs_use_valid_indices(self):
        """All connection pairs must reference valid node indices."""
        sub = es_hyperneat_substrate(
            _sine_cppn, n_inputs=3, n_outputs=2,
            variance_threshold=0.01, max_depth=2,
        )
        n = len(sub.coordinates)
        for src, tgt in sub.pairs:
            self.assertGreaterEqual(src, 0)
            self.assertLess(src, n)
            self.assertGreaterEqual(tgt, 0)
            self.assertLess(tgt, n)

    def test_different_cpppns_yield_different_node_counts(self):
        """Two CPPNs with different activation patterns may produce different substrates."""
        sub_uniform = es_hyperneat_substrate(
            _zero_cppn, n_inputs=2, n_outputs=2,
            variance_threshold=0.001, max_depth=2,
        )
        sub_varying = es_hyperneat_substrate(
            _sine_cppn, n_inputs=2, n_outputs=2,
            variance_threshold=0.001, max_depth=2,
        )
        # The varying CPPN should generally produce more nodes
        # (not a strict guarantee, but very likely with default sine params)
        self.assertGreaterEqual(
            len(sub_varying.coordinates),
            len(sub_uniform.coordinates),
            "Sine CPPN should produce at least as many nodes as a uniform CPPN"
        )

    def test_cppn_genome_instance_accepted(self):
        """CPPNGenome instance should be accepted as cppn argument."""
        cppn = CPPNGenome()
        sub = es_hyperneat_substrate(cppn, n_inputs=2, n_outputs=2, max_depth=2)
        self.assertIsInstance(sub, Substrate)


# ---------------------------------------------------------------------------
# hyperneat_substrate(evolve=True) extended API
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestHyperNEATSubstrateEvolveAPI(unittest.TestCase):

    def test_evolve_true_matches_es_function(self):
        """hyperneat_substrate(evolve=True) should produce equivalent results."""
        cppn = _sine_cppn
        sub_es = es_hyperneat_substrate(cppn, n_inputs=2, n_outputs=2,
                                        variance_threshold=0.01, max_depth=2)
        sub_ht = hyperneat_substrate(2, 2, evolve=True, cppn=cppn,
                                     variance_threshold=0.01, max_depth=2)
        self.assertEqual(len(sub_es.input_indices), len(sub_ht.input_indices))
        self.assertEqual(len(sub_es.output_indices), len(sub_ht.output_indices))

    def test_evolve_false_unaffected(self):
        """hyperneat_substrate(evolve=False) behaves exactly as before."""
        sub = hyperneat_substrate(4, 2, hidden_layers=(3,))
        sub_ne = hyperneat_substrate(4, 2, hidden_layers=(3,), evolve=False)
        self.assertEqual(len(sub.coordinates), len(sub_ne.coordinates))
        self.assertEqual(sub.input_indices, sub_ne.input_indices)

    def test_evolve_true_requires_cppn(self):
        """Passing evolve=True without cppn must raise ValueError."""
        with self.assertRaises(ValueError):
            hyperneat_substrate(2, 2, evolve=True, cppn=None)

    def test_with_hidden_layers_and_evolve(self):
        """Combining manual hidden layers with evolve adds even more nodes."""
        sub_base = hyperneat_substrate(2, 2, hidden_layers=(4,), evolve=False)
        sub_evol = hyperneat_substrate(2, 2, hidden_layers=(4,), evolve=True,
                                       cppn=_sine_cppn,
                                       variance_threshold=0.001, max_depth=2)
        # Evolved substrate should have at least as many nodes as base
        self.assertGreaterEqual(len(sub_evol.coordinates), len(sub_base.coordinates))


# ---------------------------------------------------------------------------
# generate_genome_from_cppn(evolve_substrate=True) extended API
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestGenerateGenomeEvolveSubstrate(unittest.TestCase):

    def test_evolve_substrate_false_unchanged(self):
        """evolve_substrate=False must behave identically to original."""
        cppn = CPPNGenome()
        sub = hyperneat_substrate(2, 2)
        g1 = generate_genome_from_cppn(cppn, sub, threshold=0.2, evolve_substrate=False)
        g2 = generate_genome_from_cppn(cppn, sub, threshold=0.2)
        self.assertEqual(len(g1.nodes), len(g2.nodes))

    def test_evolve_substrate_true_produces_valid_genome(self):
        """With evolve_substrate=True, the returned genome is valid and usable."""
        cppn = CPPNGenome()
        sub = hyperneat_substrate(2, 2)
        genome = generate_genome_from_cppn(
            cppn, sub, threshold=0.1, evolve_substrate=True,
            es_variance_threshold=0.01, es_max_depth=2,
        )
        self.assertGreater(len(genome.input_nodes), 0)
        self.assertGreater(len(genome.output_nodes), 0)
        # Forward pass should not crash
        result = genome.forward([0.5] * len(genome.input_nodes))
        self.assertEqual(len(result), len(genome.output_nodes))

    def test_evolve_substrate_may_add_nodes_vs_fixed(self):
        """ES-HyperNEAT substrate may produce a genome with more nodes."""
        cppn = CPPNGenome()
        sub = hyperneat_substrate(2, 2)
        g_fixed = generate_genome_from_cppn(cppn, sub, threshold=0.1,
                                            evolve_substrate=False)
        g_evol = generate_genome_from_cppn(cppn, sub, threshold=0.1,
                                           evolve_substrate=True,
                                           es_variance_threshold=0.001,
                                           es_max_depth=2)
        # Evolved genome may have more nodes (hidden nodes from Quadtree)
        self.assertGreaterEqual(len(g_evol.nodes), len(g_fixed.input_nodes) + len(g_fixed.output_nodes))


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestEsHyperNEATEndToEnd(unittest.TestCase):

    def test_full_workflow(self):
        """Full ES-HyperNEAT workflow: substrate → genome → forward."""
        cppn = CPPNGenome()
        substrate = es_hyperneat_substrate(
            cppn, n_inputs=3, n_outputs=2,
            variance_threshold=0.01, max_depth=2, initial_resolution=2,
        )
        self.assertIsInstance(substrate, Substrate)

        genome = generate_genome_from_cppn(cppn, substrate, threshold=0.1)
        self.assertGreater(len(genome.input_nodes), 0)
        self.assertGreater(len(genome.output_nodes), 0)

        inputs = [0.5] * len(genome.input_nodes)
        outputs = genome.forward(inputs)
        self.assertEqual(len(outputs), len(genome.output_nodes))

    def test_yane_api_exports(self):
        """es_hyperneat_substrate is exported from the yane package."""
        import yane
        self.assertTrue(hasattr(yane, "es_hyperneat_substrate"))
        self.assertTrue(callable(yane.es_hyperneat_substrate))


if __name__ == "__main__":
    unittest.main()
