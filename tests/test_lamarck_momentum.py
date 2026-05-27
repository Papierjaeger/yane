"""Tests for Lamarck-Momentum: gradient-informed mutation direction."""

import random
import unittest

from yane import NeuroEvolution
from yane.core.genome import Genome  # noqa: F401 – type hint used in fitness signatures
from yane.evolution.lamarck_refiner import LamarckRefiner


def _make_genome() -> Genome:
    from yane import NeuroEvolution
    yane = NeuroEvolution(seed=0)
    yane.configure(2, 1, n_initial_hidden=1)
    return yane._population._unevaluated[0]


def _positive_fitness(genome: Genome) -> float:
    """Fitness proportional to sum of weights; gradient points toward higher weights."""
    return sum(c.weight for n in genome.nodes for c in n.connections if c.enabled)


class TestLamarckMomentumRefiner(unittest.TestCase):
    """Unit tests for LamarckRefiner momentum storage."""

    def test_momentum_stored_after_refine(self):
        """After refine(), genome carries _lamarck_momentum when momentum is enabled."""
        r = LamarckRefiner()
        r.set_explicit(n_steps=3, sigma=1.0)
        r.momentum_enabled = True
        r.momentum_prob = 0.5
        r.momentum_decay = 0.9

        g = _make_genome()
        r.refine(g, _positive_fitness)

        self.assertTrue(hasattr(g, '_lamarck_momentum'))
        self.assertIsInstance(g._lamarck_momentum, dict)
        self.assertEqual(g._lamarck_momentum_prob, 0.5)
        self.assertEqual(g._lamarck_momentum_decay, 0.9)

    def test_momentum_not_stored_when_disabled(self):
        """When momentum_enabled=False, genome gets no momentum attributes."""
        r = LamarckRefiner()
        r.set_explicit(n_steps=3, sigma=1.0)
        r.momentum_enabled = False

        g = _make_genome()
        r.refine(g, _positive_fitness)

        self.assertFalse(hasattr(g, '_lamarck_momentum'))

    def test_momentum_vector_reflects_delta(self):
        """Stored momentum reflects actual weight change (non-zero on improvement)."""
        r = LamarckRefiner()
        r.set_explicit(n_steps=5, sigma=0.5)
        r.momentum_enabled = True
        r.momentum_prob = 1.0
        r.momentum_decay = 1.0

        random.seed(42)
        g = _make_genome()
        conns = [c for n in g.nodes for c in n.connections if c.enabled]
        orig_weights = {c.innovation: c.weight for c in conns}

        r.refine(g, _positive_fitness)

        # At least one connection should have a non-zero delta if refinement improved
        for c in conns:
            expected_delta = c.weight - orig_weights[c.innovation]
            stored_delta = g._lamarck_momentum.get(c.innovation, 0.0)
            self.assertAlmostEqual(stored_delta, expected_delta, places=10)

    def test_momentum_nes(self):
        """NES mode also stores momentum after refinement."""
        r = LamarckRefiner()
        r.set_nes(k=2, sigma=0.3)
        r.momentum_enabled = True

        g = _make_genome()
        r.refine_nes(g, _positive_fitness)
        self.assertTrue(hasattr(g, '_lamarck_momentum'))

    def test_momentum_sa(self):
        """SA mode also stores momentum after refinement."""
        r = LamarckRefiner()
        r.set_sa(k=3, sigma=0.3)
        r.momentum_enabled = True

        g = _make_genome()
        r.refine_sa(g, _positive_fitness)
        self.assertTrue(hasattr(g, '_lamarck_momentum'))

    def test_momentum_zero_prob_no_mutation_effect(self):
        """With momentum_prob=0, mutation is unaffected by stored momentum."""
        r = LamarckRefiner()
        r.set_explicit(n_steps=3, sigma=0.5)
        r.momentum_enabled = True
        r.momentum_prob = 0.0

        random.seed(7)
        g = _make_genome()
        r.refine(g, _positive_fitness)
        # Store weights before mutate
        conns = [c for n in g.nodes for c in n.connections if c.enabled]
        w_before = [c.weight for c in conns]
        b_before = [n.bias for n in g.nodes]

        # Patch random to be deterministic and check no momentum nudge
        # We verify by setting momentum to large values and confirming mutation
        # matches expectation without the nudge.
        g._lamarck_momentum = {c.innovation: 1000.0 for c in conns}
        g._lamarck_bias_momentum = [1000.0] * len(g.nodes)
        # prob=0 means momentum block never fires — weights won't jump by 1000
        random.seed(99)
        g.mutate()
        w_after = [c.weight for c in conns]
        # No weight should have jumped by ~1000
        for w0, w1 in zip(w_before, w_after):
            self.assertLess(abs(w1 - w0), 100.0)

    def test_momentum_decay_reduces_stored_values(self):
        """Stored momentum decays each time mutate() is called."""
        r = LamarckRefiner()
        r.set_explicit(n_steps=3, sigma=0.5)
        r.momentum_enabled = True
        r.momentum_prob = 1.0
        r.momentum_decay = 0.5

        random.seed(0)
        g = _make_genome()
        r.refine(g, _positive_fitness)

        conns = [c for n in g.nodes for c in n.connections if c.enabled]
        initial_momentum = {inn: v for inn, v in g._lamarck_momentum.items()}
        g.mutate()
        for inn, v0 in initial_momentum.items():
            self.assertAlmostEqual(g._lamarck_momentum[inn], v0 * 0.5, places=10)


class TestLamarckMomentumAPI(unittest.TestCase):
    """Integration tests for NeuroEvolution.set_lamarck_momentum()."""

    def test_api_sets_refiner_fields(self):
        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane.set_lamarck_momentum(enabled=True, momentum_prob=0.4, decay=0.8)
        self.assertTrue(yane._lamarck.momentum_enabled)
        self.assertAlmostEqual(yane._lamarck.momentum_prob, 0.4)
        self.assertAlmostEqual(yane._lamarck.momentum_decay, 0.8)

    def test_api_disable(self):
        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane.set_lamarck_momentum(enabled=True)
        yane.set_lamarck_momentum(enabled=False)
        self.assertFalse(yane._lamarck.momentum_enabled)

    def test_api_invalid_prob(self):
        yane = NeuroEvolution()
        yane.configure(2, 1)
        with self.assertRaises(ValueError):
            yane.set_lamarck_momentum(momentum_prob=1.5)

    def test_api_invalid_decay(self):
        yane = NeuroEvolution()
        yane.configure(2, 1)
        with self.assertRaises(ValueError):
            yane.set_lamarck_momentum(decay=0.0)


if __name__ == '__main__':
    unittest.main()
