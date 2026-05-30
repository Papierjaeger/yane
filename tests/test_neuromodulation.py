"""Tests für Neuromodulation (evolution/neuromodulation.py).

Akzeptanzkriterien:
  1. is_modulator Flag auf Knoten (Node.__slots__, copy, crossover, checkpoint)
  2. MODULATOR-Ausgabe skaliert eingehende Verbindungen anderer Knoten
     (one-step-delayed: Gain aus Pass T wirkt auf Pass T+1)
  3. genome.reset() setzt Modulation-Gains zurück (episodenlokal)
  4. Zero-Cost wenn kein MODULATOR im Genome
  5. NeuroEvolution.set_neuromodulation() + Training ohne Crash
  6. mutate_modulator_flags kann Knoten zu MODULATORen befördern
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

def _make_genome() -> Genome:
    """2 inputs → 1 hidden → 1 output, all linear weights=1."""
    g = Genome()
    inp0 = Node(NodeType.INPUT, 0); inp0.activation = ActivationType.LINEAR; inp0.input_index = 0
    inp1 = Node(NodeType.INPUT, 1); inp1.activation = ActivationType.LINEAR; inp1.input_index = 1
    hid = Node(NodeType.HIDDEN, 2); hid.activation = ActivationType.SIGMOID; hid.bias = 0.0
    out = Node(NodeType.OUTPUT, 3); out.activation = ActivationType.SIGMOID; out.bias = 0.0
    g.nodes.extend([inp0, inp1, hid, out])
    g.input_nodes.extend([inp0, inp1])
    g.output_nodes.append(out)
    for inp in g.input_nodes:
        c = Connection(hid, 10 + inp.innovation); c.weight = 1.0; inp.connections.append(c)
    c2 = Connection(out, 20); c2.weight = 1.0; hid.connections.append(c2)
    g._invalidate_topology()
    return g


def _make_genome_with_modulator() -> Genome:
    """2 inputs: input0 → modulator → output, input1 → output directly."""
    g = Genome()
    inp0 = Node(NodeType.INPUT, 0); inp0.activation = ActivationType.LINEAR; inp0.input_index = 0
    inp1 = Node(NodeType.INPUT, 1); inp1.activation = ActivationType.LINEAR; inp1.input_index = 1
    mod = Node(NodeType.HIDDEN, 2); mod.activation = ActivationType.SIGMOID; mod.bias = 0.0
    mod.is_modulator = True
    out = Node(NodeType.OUTPUT, 3); out.activation = ActivationType.LINEAR; out.bias = 0.0
    g.nodes.extend([inp0, inp1, mod, out])
    g.input_nodes.extend([inp0, inp1])
    g.output_nodes.append(out)
    # inp0 → modulator (provides gain signal)
    c_mod = Connection(mod, 10); c_mod.weight = 1.0; inp0.connections.append(c_mod)
    # modulator → output (targets output for modulation)
    c_mod_tgt = Connection(out, 11); c_mod_tgt.weight = 0.0; mod.connections.append(c_mod_tgt)
    # inp1 → output (the connection being modulated)
    c_direct = Connection(out, 12); c_direct.weight = 1.0; inp1.connections.append(c_direct)
    g._invalidate_topology()
    return g


# ---------------------------------------------------------------------------
# Node is_modulator flag — acceptance criterion 1
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestModulatorFlag(unittest.TestCase):

    def test_is_modulator_default_false(self):
        n = Node(NodeType.HIDDEN, 0)
        self.assertFalse(n.is_modulator)

    def test_modulation_gain_default_one(self):
        n = Node(NodeType.HIDDEN, 0)
        self.assertAlmostEqual(n.modulation_gain, 1.0)

    def test_set_is_modulator(self):
        n = Node(NodeType.HIDDEN, 0)
        n.is_modulator = True
        self.assertTrue(n.is_modulator)

    def test_pickle_preserves_is_modulator(self):
        g = _make_genome()
        g.nodes[2].is_modulator = True
        g2 = pickle.loads(pickle.dumps(g))
        self.assertTrue(g2.nodes[2].is_modulator)

    def test_copy_preserves_is_modulator(self):
        g = _make_genome()
        g.nodes[2].is_modulator = True
        gc = g.copy()
        self.assertTrue(gc.nodes[2].is_modulator)

    def test_crossover_preserves_is_modulator(self):
        g_a = _make_genome(); g_a.fitness = 10.0
        g_b = _make_genome(); g_b.fitness = 5.0
        g_a.nodes[2].is_modulator = True
        child = g_a.crossover(g_b)
        # Child inherits from fitter parent (g_a) which has is_modulator=True on node 2
        self.assertTrue(child.nodes[2].is_modulator)

    def test_genome_has_modulators_false(self):
        from yane.evolution.neuromodulation import genome_has_modulators
        g = _make_genome()
        self.assertFalse(genome_has_modulators(g))

    def test_genome_has_modulators_true(self):
        from yane.evolution.neuromodulation import genome_has_modulators, make_node_modulator
        g = _make_genome()
        make_node_modulator(g, 2)
        self.assertTrue(genome_has_modulators(g))


# ---------------------------------------------------------------------------
# Modulation applies gain — acceptance criterion 2
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestModulationGainApplication(unittest.TestCase):

    def test_update_modulation_gains_sets_target_gain(self):
        """After forward, MODULATOR's value must be stored as target's gain."""
        from yane.evolution.neuromodulation import update_modulation_gains
        g = _make_genome_with_modulator()
        g.reset()
        g.forward([0.8, 1.0])  # inp0=0.8 → modulator
        mod_node = g.nodes[2]
        out_node = g.nodes[3]
        # modulator value = sigmoid(0.8 * 1.0) ≈ 0.69
        update_modulation_gains(g)
        expected_gain = max(0.0, min(2.0, mod_node.value))
        self.assertAlmostEqual(out_node.modulation_gain, expected_gain, places=5)

    def test_apply_modulation_changes_connection_weight(self):
        """apply_modulation_to_weights must modify working weight proportional to gain."""
        from yane.evolution.neuromodulation import apply_modulation_to_weights
        g = _make_genome_with_modulator()
        out_node = g.nodes[3]
        out_node.modulation_gain = 0.5  # manually set gain
        # Find connection from inp1 → out (the modulated one)
        inp1 = g.input_nodes[1]
        conn = inp1.connections[0]
        base_w = conn.weight  # = 1.0
        apply_modulation_to_weights(g)
        self.assertAlmostEqual(conn.weight, base_w * 0.5, places=5,
                               msg="Working weight must be base * gain")

    def test_gain_zero_suppresses_connection(self):
        """Gain=0 must suppress the connection completely."""
        from yane.evolution.neuromodulation import apply_modulation_to_weights
        g = _make_genome_with_modulator()
        out_node = g.nodes[3]
        out_node.modulation_gain = 0.0
        apply_modulation_to_weights(g)
        inp1 = g.input_nodes[1]
        conn = inp1.connections[0]
        self.assertAlmostEqual(conn.weight, 0.0, places=5)

    def test_neutral_gain_no_change(self):
        """Gain=1.0 must not change connection weights."""
        from yane.evolution.neuromodulation import apply_modulation_to_weights
        g = _make_genome_with_modulator()
        # modulation_gain defaults to 1.0
        inp1 = g.input_nodes[1]
        conn = inp1.connections[0]
        w_before = conn.weight
        apply_modulation_to_weights(g)
        self.assertAlmostEqual(conn.weight, w_before, places=9)


# ---------------------------------------------------------------------------
# Episode-local modulation / reset — acceptance criterion 3
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestEpisodicLocality(unittest.TestCase):

    def test_reset_restores_base_weights(self):
        from yane.evolution.neuromodulation import apply_modulation_to_weights, restore_modulation_weights
        g = _make_genome_with_modulator()
        out_node = g.nodes[3]
        out_node.modulation_gain = 0.5
        inp1 = g.input_nodes[1]
        conn = inp1.connections[0]
        original_w = conn.weight
        apply_modulation_to_weights(g)
        self.assertNotAlmostEqual(conn.weight, original_w)
        restore_modulation_weights(g)
        self.assertAlmostEqual(conn.weight, original_w,
                               msg="restore must return weight to base value")

    def test_reset_clears_modulation_gains(self):
        from yane.evolution.neuromodulation import update_modulation_gains, restore_modulation_weights
        g = _make_genome_with_modulator()
        g.reset(); g.forward([1.0, 1.0])
        update_modulation_gains(g)
        # Some gains should be != 1.0 now
        restore_modulation_weights(g)
        for node in g.nodes:
            self.assertAlmostEqual(node.modulation_gain, 1.0,
                                   msg="restore must reset all gains to neutral")


# ---------------------------------------------------------------------------
# Zero cost when disabled — acceptance criterion 4
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestZeroCostWhenDisabled(unittest.TestCase):

    def test_genome_without_modulators_no_gain_change(self):
        """apply_modulation_to_weights on a genome with no modulators must be a no-op."""
        from yane.evolution.neuromodulation import apply_modulation_to_weights, genome_has_modulators
        g = _make_genome()
        self.assertFalse(genome_has_modulators(g))
        weights_before = [(c.weight, c._base_weight)
                          for n in g.nodes for c in n.connections]
        apply_modulation_to_weights(g)
        weights_after = [(c.weight, c._base_weight)
                         for n in g.nodes for c in n.connections]
        self.assertEqual(weights_before, weights_after)


# ---------------------------------------------------------------------------
# NeuroEvolution integration — acceptance criterion 5
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestNeuroEvolutionIntegration(unittest.TestCase):

    def test_set_neuromodulation_sets_flag(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_neuromodulation(enabled=True)
        self.assertTrue(ne._neuromodulation_enabled)

    def test_set_neuromodulation_disabled(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_neuromodulation(enabled=False)
        self.assertFalse(ne._neuromodulation_enabled)

    def test_train_with_no_modulators_no_crash(self):
        """With modulation enabled but no modulator nodes → normal training."""
        import yane
        ne = yane.NeuroEvolution()
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=10, max_connections=20)
        ne.set_neuromodulation(enabled=True)
        ne.set_max_iterations(5)
        ne.train(lambda g: sum(g.forward([0.5, 0.5])))

    def test_train_with_modulator_no_crash(self):
        """Training with a MODULATOR node active must not crash."""
        import yane
        from yane.evolution.neuromodulation import make_node_modulator
        ne = yane.NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=15, max_connections=30,
                     n_initial_hidden=1)
        ne.set_neuromodulation(enabled=True)
        ne.set_max_iterations(5)
        # Mark the first hidden node as a modulator
        pop = ne.population
        for g in list(pop._unevaluated):
            hidden = [n for n in g.nodes
                      if n not in g.input_nodes and n not in g.output_nodes]
            if hidden:
                make_node_modulator(g, g.nodes.index(hidden[0]))
        ne.train(lambda g: sum(g.forward([0.5, 0.5])))

    def test_modulation_gains_restored_after_eval(self):
        """After training, modulation gains must be back to neutral (1.0)."""
        import yane
        from yane.evolution.neuromodulation import make_node_modulator
        ne = yane.NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=15, max_connections=30,
                     n_initial_hidden=1)
        ne.set_neuromodulation(enabled=True)
        ne.set_max_iterations(3)
        for g in list(ne.population._unevaluated):
            hidden = [n for n in g.nodes
                      if n not in g.input_nodes and n not in g.output_nodes]
            if hidden:
                make_node_modulator(g, g.nodes.index(hidden[0]))
        ne.train(lambda g: sum(g.forward([0.5, 0.5])))
        best = ne.get_best()
        for node in best.nodes:
            self.assertAlmostEqual(node.modulation_gain, 1.0)


# ---------------------------------------------------------------------------
# mutate_modulator_flags — acceptance criterion 6
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestMutateModulatorFlags(unittest.TestCase):

    def test_adds_modulator_with_high_prob(self):
        from yane.evolution.neuromodulation import mutate_modulator_flags
        g = _make_genome()
        # With probability 1.0 every eligible node becomes a modulator
        for _ in range(100):
            mutate_modulator_flags(g, add_prob=1.0, remove_prob=0.0)
        hidden = [n for n in g.nodes
                  if n not in g.input_nodes and n not in g.output_nodes]
        self.assertTrue(any(n.is_modulator for n in hidden),
                        "With add_prob=1.0, at least one hidden node should become modulator")

    def test_removes_modulator_with_high_prob(self):
        from yane.evolution.neuromodulation import mutate_modulator_flags
        g = _make_genome()
        g.nodes[2].is_modulator = True
        for _ in range(50):
            mutate_modulator_flags(g, add_prob=0.0, remove_prob=1.0)
        self.assertFalse(g.nodes[2].is_modulator,
                         "With remove_prob=1.0, modulator should be removed")


if __name__ == "__main__":
    unittest.main()
