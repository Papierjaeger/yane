import math
import unittest
import pytest

from yane import NeuroEvolution
from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection


def _make_simple_genome() -> Genome:
    """Tiny fully-connected 2→1 genome with known weights."""
    g = Genome()
    inp0 = Node(NodeType.INPUT, innovation=0)
    inp1 = Node(NodeType.INPUT, innovation=1)
    out = Node(NodeType.OUTPUT, innovation=2)
    g.nodes = [inp0, inp1, out]
    g.input_nodes = [inp0, inp1]
    g.output_nodes = [out]
    c0 = Connection(out, innovation=10)
    c0.weight = 2.0
    c1 = Connection(out, innovation=11)
    c1.weight = 0.05  # small weight → low gate
    inp0.connections = [c0]
    inp1.connections = [c1]
    g._invalidate_topology()
    return g


@pytest.mark.ci
class TestDARTSGates(unittest.TestCase):

    def test_update_initialises_gates(self):
        g = _make_simple_genome()
        self.assertIsNone(g._darts_gates)
        g.update_darts_gates()
        self.assertIsNotNone(g._darts_gates)
        self.assertIn(10, g._darts_gates)
        self.assertIn(11, g._darts_gates)

    def test_gate_range(self):
        g = _make_simple_genome()
        g.update_darts_gates()
        for innov, gate in g._darts_gates.items():
            self.assertGreater(gate, 0.0)
            self.assertLess(gate, 1.0)

    def test_large_weight_high_gate(self):
        g = _make_simple_genome()
        g.update_darts_gates()
        # weight=2.0 → sigmoid(4) ≈ 0.982
        gate_large = g._darts_gates[10]
        gate_small = g._darts_gates[11]
        self.assertGreater(gate_large, gate_small)

    def test_zero_weight_gate_is_half(self):
        g = _make_simple_genome()
        g.input_nodes[0].connections[0].weight = 0.0
        g.update_darts_gates()
        gate = g._darts_gates[10]
        self.assertAlmostEqual(gate, 0.5, places=5)

    def test_sigmoid_formula(self):
        g = _make_simple_genome()
        g.input_nodes[0].connections[0].weight = 1.5
        g.update_darts_gates()
        expected = 1.0 / (1.0 + math.exp(-3.0))  # sigmoid(|1.5| * 2)
        self.assertAlmostEqual(g._darts_gates[10], expected, places=8)


@pytest.mark.ci
class TestDARTSPruning(unittest.TestCase):

    def test_prune_removes_low_gate(self):
        g = _make_simple_genome()
        g.update_darts_gates()
        # c1 has weight=0.05 → gate = sigmoid(0.1) ≈ 0.525; set threshold high
        # so it gets pruned
        g._darts_gates[11] = 0.05  # force below threshold
        removed = g.prune_darts_connections(threshold=0.1)
        self.assertEqual(removed, 1)
        all_innov = [c.innovation for n in g.nodes for c in n.connections]
        self.assertNotIn(11, all_innov)
        self.assertIn(10, all_innov)

    def test_prune_keeps_high_gate(self):
        g = _make_simple_genome()
        g.update_darts_gates()
        g._darts_gates[10] = 0.9
        g._darts_gates[11] = 0.9
        removed = g.prune_darts_connections(threshold=0.1)
        self.assertEqual(removed, 0)

    def test_prune_returns_zero_without_gates(self):
        g = _make_simple_genome()
        removed = g.prune_darts_connections(threshold=0.1)
        self.assertEqual(removed, 0)

    def test_prune_invalidates_topology(self):
        g = _make_simple_genome()
        g._exec_order = []  # simulate cached topology
        g._darts_gates = {10: 0.9, 11: 0.05}
        g.prune_darts_connections(threshold=0.1)
        self.assertIsNone(g._exec_order)

    def test_copy_preserves_gates(self):
        g = _make_simple_genome()
        g.update_darts_gates()
        g._darts_gates[10] = 0.85
        copy = g.copy()
        self.assertIsNotNone(copy._darts_gates)
        self.assertAlmostEqual(copy._darts_gates[10], 0.85)

    def test_copy_is_independent(self):
        g = _make_simple_genome()
        g.update_darts_gates()
        copy = g.copy()
        copy._darts_gates[10] = 0.0
        # original unchanged
        self.assertNotEqual(g._darts_gates[10], 0.0)


@pytest.mark.ci
class TestDARTSIntegration(unittest.TestCase):

    def _make_yane(self) -> NeuroEvolution:
        yane = NeuroEvolution(seed=0)
        yane.configure(2, 1, n_initial_hidden=1)
        yane.set_population_size(8)
        return yane

    def test_default_disabled(self):
        yane = self._make_yane()
        self.assertFalse(yane._darts_enabled)

    def test_set_darts_mode_enables(self):
        yane = self._make_yane()
        yane.set_darts_mode(enabled=True)
        self.assertTrue(yane._darts_enabled)

    def test_set_darts_mode_stores_threshold(self):
        yane = self._make_yane()
        yane.set_darts_mode(prune_threshold=0.2)
        self.assertAlmostEqual(yane._darts_prune_threshold, 0.2)

    def test_invalid_threshold_raises(self):
        yane = self._make_yane()
        with self.assertRaises(ValueError):
            yane.set_darts_mode(prune_threshold=1.5)
        with self.assertRaises(ValueError):
            yane.set_darts_mode(prune_threshold=-0.1)

    def test_config_dict_fields(self):
        yane = self._make_yane()
        yane.set_darts_mode(prune_threshold=0.15)
        cfg = yane._config_dict()
        self.assertTrue(cfg["darts_enabled"])
        self.assertAlmostEqual(cfg["darts_prune_threshold"], 0.15)

    def test_training_completes_without_error(self):
        yane = self._make_yane()
        yane.set_darts_mode()
        yane.set_max_iterations(20)
        yane.train(lambda g: sum(g.forward([1.0, 0.0])))

    def test_gates_populated_after_training(self):
        yane = NeuroEvolution(seed=1)
        yane.configure(2, 1, n_initial_hidden=1)
        yane.set_population_size(8)
        yane.set_darts_mode()
        yane.set_max_iterations(20)
        yane.train(lambda g: sum(g.forward([1.0, 0.0])))
        best = yane.get_best()
        self.assertIsNotNone(best._darts_gates)
        self.assertGreater(len(best._darts_gates), 0)

    def test_post_training_prune_removes_connections(self):
        """With prune_threshold=1.0 all connections should be pruned."""
        yane = NeuroEvolution(seed=2)
        yane.configure(2, 1, n_initial_hidden=1)
        yane.set_population_size(8)
        yane.set_darts_mode(prune_threshold=1.0)  # removes everything
        yane.set_max_iterations(20)
        yane.train(lambda g: sum(g.forward([1.0, 0.0])))
        best = yane.get_best()
        remaining = sum(len(n.connections) for n in best.nodes)
        self.assertEqual(remaining, 0)

    def test_disable_flag_prevents_gate_updates(self):
        yane = NeuroEvolution(seed=3)
        yane.configure(2, 1, n_initial_hidden=1)
        yane.set_population_size(8)
        yane.set_darts_mode(enabled=False)
        yane.set_max_iterations(10)
        yane.train(lambda g: sum(g.forward([1.0, 0.0])))
        best = yane.get_best()
        # Gates should not have been populated
        self.assertIsNone(best._darts_gates)


if __name__ == "__main__":
    unittest.main()
