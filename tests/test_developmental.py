"""Tests für Developmental NEAT (evolution/developmental.py).

Akzeptanzkriterien:
  1. developmental_forward() fügt während Episode tatsächlich Connections hinzu
  2. Entwicklungsregeln werden korrekt vererbt und mutiert
  3. Tests: Regel-Trigger; Episoden-Lokalität; freeze; Vererbung; Checkpoint
"""
from __future__ import annotations

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

def _make_genome(n_inputs: int = 2, n_outputs: int = 1) -> Genome:
    g = Genome()
    for i in range(n_inputs):
        n = Node(NodeType.INPUT, i); n.activation = ActivationType.LINEAR; n.input_index = i
        g.input_nodes.append(n); g.nodes.append(n)
    out = Node(NodeType.OUTPUT, n_inputs); out.activation = ActivationType.SIGMOID; out.bias = 0.0
    g.output_nodes.append(out); g.nodes.append(out)
    c = Connection(out, 10); c.weight = 1.0; g.input_nodes[0].connections.append(c)
    g._invalidate_topology()
    return g


def _make_genome_with_rule(always_fire: bool = True) -> Genome:
    from yane.evolution.developmental import make_threshold_rule
    g = _make_genome()
    # Set output node value to 0.8 (above 0.5 threshold → rule fires)
    # For always_fire=True: low threshold so rule always fires
    # For always_fire=False: high threshold so rule never fires
    threshold = 0.0 if always_fire else 2.0
    rule = make_threshold_rule(
        trigger_node_idx=len(g.nodes) - 1,  # output node
        threshold=threshold,
        trigger_mode="above",
        src_idx=0,
        tgt_idx=len(g.nodes) - 1,
        weight=0.5,
        max_fires=1,
    )
    g.dev_rules = [rule]
    return g


# ---------------------------------------------------------------------------
# Acceptance criterion 1: developmental_forward() adds connections
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestDevelopmentalForwardAddsConnections(unittest.TestCase):

    def test_rule_fires_and_adds_connection(self):
        """developmental_forward must add a connection when rule triggers."""
        g = _make_genome_with_rule(always_fire=True)
        n_before = sum(len(n.connections) for n in g.nodes)
        g.developmental_forward([1.0, 1.0])
        n_after = sum(len(n.connections) for n in g.nodes)
        self.assertGreater(n_after, n_before,
                           "developmental_forward must add at least one connection")

    def test_dev_added_tracked(self):
        """Added connections must be in genome._dev_added."""
        g = _make_genome_with_rule(always_fire=True)
        self.assertEqual(len(g._dev_added), 0)
        g.developmental_forward([1.0, 1.0])
        self.assertGreater(len(g._dev_added), 0)

    def test_no_connection_added_when_rule_does_not_fire(self):
        """When trigger is not met, no connection should be added."""
        g = _make_genome_with_rule(always_fire=False)
        n_before = sum(len(n.connections) for n in g.nodes)
        g.developmental_forward([0.5, 0.5])
        n_after = sum(len(n.connections) for n in g.nodes)
        self.assertEqual(n_before, n_after)

    def test_added_connection_affects_next_forward(self):
        """Connection added by rule must be used in subsequent forward calls."""
        g = _make_genome_with_rule(always_fire=True)
        g.developmental_forward([1.0, 1.0])  # rule fires, adds connection
        conn_count_1 = sum(len(n.connections) for n in g.nodes)
        # Now forward() uses the new connection
        g.forward([1.0, 1.0])  # no error expected
        self.assertGreater(conn_count_1, 1)

    def test_max_fires_respected(self):
        """Rule with max_fires=1 must not fire more than once per episode."""
        from yane.evolution.developmental import make_threshold_rule
        g = _make_genome()
        rule = make_threshold_rule(trigger_node_idx=0, threshold=0.0,
                                   src_idx=0, tgt_idx=len(g.nodes)-1,
                                   weight=0.1, max_fires=1)
        g.dev_rules = [rule]
        g.developmental_forward([1.0, 0.0])
        dev_after_1 = len(g._dev_added)
        g.developmental_forward([1.0, 0.0])  # max_fires reached → no more
        dev_after_2 = len(g._dev_added)
        self.assertEqual(dev_after_1, dev_after_2,
                         "Rule with max_fires=1 should not fire twice in same episode")


# ---------------------------------------------------------------------------
# Episoden-Lokalität — acceptance criterion 3 (part)
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestEpisodicLocality(unittest.TestCase):

    def test_reset_removes_dev_connections(self):
        """genome.reset() must remove all episodically added connections."""
        g = _make_genome_with_rule(always_fire=True)
        n_before = sum(len(n.connections) for n in g.nodes)
        g.developmental_forward([1.0, 1.0])
        self.assertGreater(len(g._dev_added), 0)
        g.reset()
        n_after = sum(len(n.connections) for n in g.nodes)
        self.assertEqual(n_after, n_before,
                         "reset() must restore base topology")

    def test_dev_added_cleared_after_reset(self):
        g = _make_genome_with_rule(always_fire=True)
        g.developmental_forward([1.0, 1.0])
        g.reset()
        self.assertEqual(len(g._dev_added), 0)

    def test_rule_fires_again_after_reset(self):
        """After reset(), fire counter resets → rule can fire again."""
        g = _make_genome_with_rule(always_fire=True)
        g.developmental_forward([1.0, 1.0])
        g.reset()
        g.developmental_forward([1.0, 1.0])  # should fire again
        self.assertGreater(len(g._dev_added), 0)

    def test_base_genome_unaffected_across_episodes(self):
        """Two episodes should have the same base connection count."""
        from yane.evolution.developmental import make_threshold_rule
        g = _make_genome()
        rule = make_threshold_rule(threshold=0.0, max_fires=5)
        g.dev_rules = [rule]
        base_n = sum(len(n.connections) for n in g.nodes)
        for _ in range(3):
            g.developmental_forward([1.0, 1.0])
            g.reset()
        n_now = sum(len(n.connections) for n in g.nodes)
        self.assertEqual(n_now, base_n)


# ---------------------------------------------------------------------------
# Freeze — acceptance criterion 3 (part)
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestFreezeDevelopment(unittest.TestCase):

    def test_freeze_disables_rules(self):
        """freeze_development() must prevent rules from firing."""
        g = _make_genome_with_rule(always_fire=True)
        g.freeze_development()
        n_before = sum(len(n.connections) for n in g.nodes)
        g.developmental_forward([1.0, 1.0])
        n_after = sum(len(n.connections) for n in g.nodes)
        self.assertEqual(n_before, n_after,
                         "freeze_development must disable all rules")

    def test_unfreeze_after_reset(self):
        """Manually unfreezing re-enables rules."""
        g = _make_genome_with_rule(always_fire=True)
        g.freeze_development()
        g.reset()
        g._dev_frozen = False  # manually unfreeze
        g.developmental_forward([1.0, 1.0])
        self.assertGreater(len(g._dev_added), 0)

    def test_frozen_flag_default_false(self):
        g = _make_genome()
        self.assertFalse(g._dev_frozen)


# ---------------------------------------------------------------------------
# Vererbung und Mutation — acceptance criterion 2
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestRuleInheritance(unittest.TestCase):

    def test_copy_preserves_rules(self):
        g = _make_genome_with_rule()
        gc = g.copy()
        self.assertEqual(len(gc.dev_rules), len(g.dev_rules))

    def test_copy_rules_independent(self):
        from yane.evolution.developmental import ParametricRule
        g = _make_genome_with_rule()
        gc = g.copy()
        # Modify copy's rule
        if gc.dev_rules and isinstance(gc.dev_rules[0], ParametricRule):
            gc.dev_rules[0].weight = 999.0
        if g.dev_rules and isinstance(g.dev_rules[0], ParametricRule):
            self.assertNotAlmostEqual(g.dev_rules[0].weight, 999.0)

    def test_crossover_inherits_rules(self):
        g_a = _make_genome_with_rule(always_fire=True); g_a.fitness = 10.0
        g_b = _make_genome_with_rule(always_fire=False); g_b.fitness = 5.0
        child = g_a.crossover(g_b)
        self.assertGreater(len(child.dev_rules), 0)

    def test_rule_mutation_changes_threshold(self):
        from yane.evolution.developmental import make_threshold_rule
        import random
        rule = make_threshold_rule(threshold=0.5)
        original = rule.threshold
        rule.mutate(threshold_sigma=1.0, rng=random.Random(42))
        self.assertNotAlmostEqual(rule.threshold, original, places=3)

    def test_mutate_rules_on_genome(self):
        from yane.evolution.developmental import mutate_rules
        import random
        g = _make_genome_with_rule()
        from yane.evolution.developmental import ParametricRule
        if g.dev_rules and isinstance(g.dev_rules[0], ParametricRule):
            orig_w = g.dev_rules[0].weight
            mutate_rules(g, weight_sigma=1.0, rng=random.Random(0))
            self.assertNotAlmostEqual(g.dev_rules[0].weight, orig_w, places=3)


# ---------------------------------------------------------------------------
# Checkpoint — acceptance criterion 3 (part)
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestCheckpoint(unittest.TestCase):

    def test_pickle_preserves_rules(self):
        from yane.evolution.developmental import ParametricRule
        g = _make_genome_with_rule()
        data = pickle.dumps(g)
        g2 = pickle.loads(data)
        self.assertEqual(len(g2.dev_rules), len(g.dev_rules))

    def test_pickle_dev_added_cleared(self):
        """Episode-local state must not survive pickling."""
        g = _make_genome_with_rule(always_fire=True)
        g.developmental_forward([1.0, 1.0])
        self.assertGreater(len(g._dev_added), 0)
        g2 = pickle.loads(pickle.dumps(g))
        self.assertEqual(len(g2._dev_added), 0)

    def test_empty_dev_rules_after_copy(self):
        """Genome without rules: copy also has no rules."""
        g = _make_genome()
        gc = g.copy()
        self.assertEqual(gc.dev_rules, [])


# ---------------------------------------------------------------------------
# NeuroEvolution / yane exports
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestExports(unittest.TestCase):

    def test_yane_exports_developmental(self):
        import yane
        self.assertTrue(hasattr(yane, "DevelopmentalRule"))
        self.assertTrue(hasattr(yane, "ParametricRule"))
        self.assertTrue(hasattr(yane, "make_threshold_rule"))

    def test_genome_has_developmental_forward(self):
        g = _make_genome()
        self.assertTrue(callable(g.developmental_forward))

    def test_genome_has_freeze_development(self):
        g = _make_genome()
        self.assertTrue(callable(g.freeze_development))

    def test_parametric_rule_correct_trigger_mode(self):
        from yane.evolution.developmental import make_threshold_rule
        g = _make_genome()
        g.nodes[0].value = 0.8
        rule_above = make_threshold_rule(trigger_node_idx=0, threshold=0.5, trigger_mode="above")
        rule_below = make_threshold_rule(trigger_node_idx=0, threshold=0.5, trigger_mode="below")
        self.assertTrue(rule_above.should_fire(g))
        self.assertFalse(rule_below.should_fire(g))

        g.nodes[0].value = 0.2
        self.assertFalse(rule_above.should_fire(g))
        self.assertTrue(rule_below.should_fire(g))


if __name__ == "__main__":
    unittest.main()
