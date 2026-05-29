"""Tests for experimental Layer-3 features (experimental.py).

Note: InputGrouping and OutputGrouping tests have been removed — these spikes
are superseded by evolution/input_grouping.py and evolution/output_grouping.py
with dedicated test suites (test_input_grouping.py, test_output_grouping.py).
"""
from __future__ import annotations
import unittest
import pytest
from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType


@pytest.mark.ci
class TestSharedWeights(unittest.TestCase):

    def test_shared_weight_sync(self):
        from yane.evolution.experimental import SharedWeightGroup
        g = SharedWeightGroup(group_id=0, initial_weight=1.0)
        self.assertAlmostEqual(g.weight, 1.0)
        g.set_weight(5.0)
        self.assertAlmostEqual(g.weight, 5.0)


@pytest.mark.ci
class TestDARTSLite(unittest.TestCase):

    def test_gate_pruning(self):
        from yane.evolution.experimental import DARTSLite
        darts = DARTSLite(threshold=0.3)
        darts.set_gate(10, 0.1)
        darts.set_gate(11, 0.9)
        self.assertLess(darts.get_gate(10), 0.3)
        self.assertGreater(darts.get_gate(11), 0.3)


@pytest.mark.ci
class TestCuriosity(unittest.TestCase):

    def test_prediction_error(self):
        from yane.evolution.experimental import CuriosityModule
        cm = CuriosityModule(n_state=2, n_action=1)
        err = cm.error([1.0, 0.0], [0.5], [1.1, 0.1])
        self.assertGreater(err, 0.0)

    def test_update_reduces_error(self):
        from yane.evolution.experimental import CuriosityModule
        cm = CuriosityModule(n_state=2, n_action=1, lr=0.1)
        err_before = cm.error([1.0, 0.0], [0.5], [1.1, 0.1])
        for _ in range(100):
            cm.update([1.0, 0.0], [0.5], [1.1, 0.1])
        err_after = cm.error([1.0, 0.0], [0.5], [1.1, 0.1])
        self.assertLess(err_after, err_before)


@pytest.mark.ci
class TestSTDP(unittest.TestCase):

    def test_update_changes_weight(self):
        from yane.evolution.experimental import STDPRule
        rule = STDPRule()
        g = Genome()
        inp = Node(NodeType.INPUT, 0); out = Node(NodeType.OUTPUT, 1)
        conn = Connection(out, innovation=10)
        conn.weight = 0.5
        inp.connections.append(conn)
        w_before = conn.weight
        rule.update(conn, pre_time=0.0, post_time=5.0)
        self.assertNotAlmostEqual(conn.weight, w_before)


@pytest.mark.ci
class TestNeuromodulation(unittest.TestCase):

    def test_modulation_gates_plasticity(self):
        from yane.evolution.experimental import Neuromodulator
        nm = Neuromodulator(plasticity_enabled=False)
        result = nm.modulate(None, delta=1.0)
        self.assertAlmostEqual(result, 0.0)

    def test_signal_update(self):
        from yane.evolution.experimental import Neuromodulator
        nm = Neuromodulator(baseline=0.5)
        nm.update_signal(reward=1.0, lr=0.5)
        self.assertAlmostEqual(nm.signal, 0.75)


@pytest.mark.ci
class TestCoevolution(unittest.TestCase):

    def test_add_agent_and_env(self):
        from yane.evolution.experimental import CoevolutionPool
        pool = CoevolutionPool()
        pool.add_agent(Genome())
        pool.add_env(Genome())
        self.assertEqual(len(pool.agents), 1)
        self.assertEqual(len(pool.envs), 1)

    def test_pairing_creates_pairs(self):
        from yane.evolution.experimental import CoevolutionPool
        pool = CoevolutionPool()
        for _ in range(3):
            pool.add_agent(Genome())
        pool.add_env(Genome())
        pairs = pool.pair()
        self.assertEqual(len(pairs), 3)


if __name__ == "__main__":
    unittest.main()
