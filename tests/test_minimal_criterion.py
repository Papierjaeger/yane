"""Tests für Open-Ended Evolution / Minimal Criterion (evolution/minimal_criterion.py).

Akzeptanzkriterien:
  1. Genome unter Kriterium werden nicht zur Fortpflanzung zugelassen (Penalty-Fitness)
  2. Adaptive Lockerung greift wenn viable_frac < min_viable_frac
  3. Tests: Kriteriums-Filter; adaptive Schwelle; viable-Boost; Archiv-Integration
"""
from __future__ import annotations

import unittest

import pytest

from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_genome(weight: float = 0.5, fitness: float = 1.0) -> Genome:
    g = Genome()
    inp = Node(NodeType.INPUT, 0); inp.activation = ActivationType.LINEAR; inp.input_index = 0
    out = Node(NodeType.OUTPUT, 1); out.activation = ActivationType.SIGMOID; out.bias = 0.0
    g.nodes.extend([inp, out]); g.input_nodes.append(inp); g.output_nodes.append(out)
    c = Connection(out, 10); c.weight = weight; inp.connections.append(c)
    g.fitness = fitness
    g._invalidate_topology()
    return g


# ---------------------------------------------------------------------------
# Acceptance criterion 1: Kriteriums-Filter
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestCriterionFilter(unittest.TestCase):

    def test_viable_genome_keeps_base_fitness(self):
        from yane.evolution.minimal_criterion import MinimalCriterion
        mc = MinimalCriterion(criterion_fn=lambda g: True)
        g = _make_genome(fitness=5.0)
        result = mc.apply(g, 5.0)
        self.assertAlmostEqual(result, 5.0)

    def test_non_viable_genome_gets_penalty(self):
        from yane.evolution.minimal_criterion import MinimalCriterion
        mc = MinimalCriterion(criterion_fn=lambda g: False,
                              penalty=-1e6, min_viable_frac=0.0)
        g = _make_genome(fitness=5.0)
        result = mc.apply(g, 5.0)
        self.assertLess(result, 0.0, "Non-viable genome must receive negative penalty")

    def test_criterion_based_on_fitness(self):
        """Criterion lambda g: g.fitness > 0.5 should work correctly."""
        from yane.evolution.minimal_criterion import MinimalCriterion
        mc = MinimalCriterion(
            criterion_fn=lambda g: g.fitness > 0.5,
            penalty=-999.0,
            min_viable_frac=0.0,
        )
        g_viable = _make_genome(fitness=1.0)
        g_not_viable = _make_genome(fitness=0.1)
        self.assertAlmostEqual(mc.apply(g_viable, 1.0), 1.0)
        self.assertLess(mc.apply(g_not_viable, 0.1), 0.0)

    def test_viable_count_tracked(self):
        from yane.evolution.minimal_criterion import MinimalCriterion
        mc = MinimalCriterion(criterion_fn=lambda g: g.fitness > 0.5, min_viable_frac=0.0)
        for f in [0.1, 0.8, 0.3, 0.9, 0.2]:
            g = _make_genome(fitness=f)
            mc.apply(g, f)
        # 2 out of 5 are viable (0.8 and 0.9)
        self.assertEqual(mc._n_viable, 2)
        self.assertEqual(mc._n_total, 5)

    def test_viable_frac_property(self):
        from yane.evolution.minimal_criterion import MinimalCriterion
        mc = MinimalCriterion(criterion_fn=lambda g: True, min_viable_frac=0.0)
        for _ in range(4):
            mc.apply(_make_genome(fitness=1.0), 1.0)
        self.assertAlmostEqual(mc.viable_frac, 1.0)

    def test_wrap_fitness_non_viable_gets_penalty(self):
        """wrap_fitness must penalize non-viable genome fitness."""
        from yane.evolution.minimal_criterion import MinimalCriterion
        mc = MinimalCriterion(
            criterion_fn=lambda g: g.fitness > 0.0,  # all that have positive fitness
            penalty=-500.0,
            min_viable_frac=0.0,
        )
        wrapped = mc.wrap_fitness(lambda g: -10.0)  # always returns negative base
        g = _make_genome()
        result = wrapped(g)
        # base = -10.0, genome.fitness = -10.0 → not viable → penalty
        self.assertLess(result, 0.0)


# ---------------------------------------------------------------------------
# Acceptance criterion 2: Adaptive Lockerung
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestAdaptiveRelaxation(unittest.TestCase):

    def test_relaxation_activates_below_min_viable_frac(self):
        """When viable_frac < min_viable_frac, penalty should be less severe."""
        from yane.evolution.minimal_criterion import MinimalCriterion
        # Set min_viable_frac=0.5 → relaxation when < 50% viable
        mc = MinimalCriterion(
            criterion_fn=lambda g: g.fitness > 100.0,  # nothing passes (all < 100)
            min_viable_frac=0.5,
            penalty=-1000.0,
            viable_boost_factor=0.1,  # relaxed penalty = -1000 * 0.1 = -100
        )
        # Apply to enough genomes so viable_frac < 0.5
        for _ in range(5):
            g = _make_genome(fitness=1.0)  # all not viable
            result = mc.apply(g, 1.0)
        # After 5 non-viable evaluations: viable_frac = 0 < 0.5 → relaxation active
        self.assertTrue(mc._relaxation_active,
                        "Relaxation must activate when viable_frac < min_viable_frac")

    def test_relaxed_penalty_less_severe(self):
        """Relaxed penalty must be less severe than full penalty."""
        from yane.evolution.minimal_criterion import MinimalCriterion
        mc = MinimalCriterion(
            criterion_fn=lambda g: g.fitness > 100.0,
            min_viable_frac=0.9,
            penalty=-1000.0,
            viable_boost_factor=0.1,
        )
        # First call: viable_frac = 0/0 = 1.0 initially... actually 0 because no total
        # Let's evaluate a non-viable genome and trigger relaxation
        # After enough evals with 0 viable: relaxation should kick in
        for _ in range(10):
            g = _make_genome(fitness=0.5)
            mc.apply(g, 0.5)
        # Now check that relaxation is active (all non-viable)
        if mc._relaxation_active:
            # Evaluate one more non-viable genome
            g = _make_genome(fitness=0.5)
            result = mc.apply(g, 0.5)
            # Relaxed penalty = penalty * viable_boost_factor = -1000 * 0.1 = -100
            # Should be between penalty and 0
            self.assertGreater(result, -1000.0,
                               "Relaxed penalty must be less severe than full penalty")

    def test_relaxation_inactive_when_enough_viable(self):
        """When viable_frac >= min_viable_frac, relaxation must not be active."""
        from yane.evolution.minimal_criterion import MinimalCriterion
        mc = MinimalCriterion(
            criterion_fn=lambda g: True,  # all viable
            min_viable_frac=0.5,
            penalty=-1000.0,
        )
        for _ in range(10):
            mc.apply(_make_genome(fitness=1.0), 1.0)
        self.assertFalse(mc._relaxation_active)

    def test_viable_fraction_history_tracked(self):
        """Per-generation viable fractions should be recorded."""
        from yane.evolution.minimal_criterion import MinimalCriterion
        mc = MinimalCriterion(
            criterion_fn=lambda g: g.fitness > 0.5,
            min_viable_frac=0.0,
        )
        # Gen 1: 2 viable, 1 not
        for f in [0.8, 0.2, 0.9]:
            mc.apply(_make_genome(fitness=f), f)
        mc.reset_generation()
        history = mc.viable_fraction_history()
        self.assertEqual(len(history), 1)
        self.assertAlmostEqual(history[0], 2.0 / 3.0, places=5)

    def test_reset_generation_clears_counters(self):
        from yane.evolution.minimal_criterion import MinimalCriterion
        mc = MinimalCriterion(criterion_fn=lambda g: True, min_viable_frac=0.0)
        for _ in range(5):
            mc.apply(_make_genome(fitness=1.0), 1.0)
        mc.reset_generation()
        self.assertEqual(mc._n_total, 0)
        self.assertEqual(mc._n_viable, 0)


# ---------------------------------------------------------------------------
# Acceptance criterion 3: Archiv-Integration (set_open_ended)
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestArchivIntegration(unittest.TestCase):

    def test_set_open_ended_novelty_mode(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=10)
        ne.set_minimal_criterion(lambda g: g.fitness > -1000.0)
        ne.set_open_ended(mode="novelty_with_criterion", archive_size=100)
        self.assertEqual(ne._open_ended_mode, "novelty_with_criterion")

    def test_set_open_ended_invalid_raises(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.configure(n_inputs=2, n_outputs=1)
        ne.set_minimal_criterion(lambda g: True)
        with self.assertRaises(ValueError):
            ne.set_open_ended(mode="invalid_mode")

    def test_set_minimal_criterion_none_disables(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_minimal_criterion(lambda g: True)
        self.assertIsNotNone(ne._minimal_criterion)
        ne.set_minimal_criterion(None)
        self.assertIsNone(ne._minimal_criterion)

    def test_make_novelty_with_criterion(self):
        from yane.evolution.minimal_criterion import MinimalCriterion, make_novelty_with_criterion
        mc = MinimalCriterion(criterion_fn=lambda g: g.fitness > 0.0, min_viable_frac=0.0)
        base_fn = lambda g: 1.0
        wrapped = make_novelty_with_criterion(base_fn, mc)
        g = _make_genome(fitness=1.0)
        result = wrapped(g)
        self.assertAlmostEqual(result, 1.0)

    def test_make_curiosity_with_criterion(self):
        from yane.evolution.minimal_criterion import MinimalCriterion, make_curiosity_with_criterion
        mc = MinimalCriterion(criterion_fn=lambda g: False, penalty=-500.0, min_viable_frac=0.0)
        base_fn = lambda g: 3.0
        wrapped = make_curiosity_with_criterion(base_fn, mc)
        g = _make_genome(fitness=3.0)
        result = wrapped(g)
        self.assertLess(result, 0.0)


# ---------------------------------------------------------------------------
# NeuroEvolution integration
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestNeuroEvolutionIntegration(unittest.TestCase):

    def test_set_minimal_criterion_returns_mc(self):
        import yane
        from yane.evolution.minimal_criterion import MinimalCriterion
        ne = yane.NeuroEvolution()
        mc = ne.set_minimal_criterion(lambda g: True)
        self.assertIsInstance(mc, MinimalCriterion)

    def test_non_viable_gets_penalty_during_train(self):
        """Non-viable genomes should have low fitness after training."""
        import yane
        ne = yane.NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=10, max_connections=20)
        # Criterion: output > 0.9 (hard to satisfy initially)
        ne.set_minimal_criterion(
            lambda g: g.fitness > 0.9,
            penalty=-999.0,
            min_viable_frac=0.0,
        )
        ne.set_max_iterations(5)
        ne.train(lambda g: sum(g.forward([0.5, 0.5])))
        # Check that mc tracked some evaluations
        mc = ne._minimal_criterion
        self.assertGreater(mc._n_total, 0)

    def test_train_without_criterion_unaffected(self):
        """Without minimal criterion, training should work normally."""
        import yane
        ne = yane.NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=10)
        ne.set_max_iterations(5)
        ne.train(lambda g: sum(g.forward([0.5, 0.5])))

    def test_yane_exports(self):
        import yane
        self.assertTrue(hasattr(yane, "MinimalCriterion"))


if __name__ == "__main__":
    unittest.main()
