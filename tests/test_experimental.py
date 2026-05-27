"""Tests for experimental Layer-3 features (experimental.py)."""
from __future__ import annotations
import unittest
import pytest
from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType


@pytest.mark.ci
class TestInputGrouping(unittest.TestCase):

    def test_grouping_reduces_dimension(self):
        from yane.evolution.experimental import InputGrouping
        ig = InputGrouping(n_raw=8, n_groups=2)
        out = ig.forward([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        self.assertEqual(len(out), 2)

    def test_grouping_mean(self):
        from yane.evolution.experimental import InputGrouping
        ig = InputGrouping(n_raw=4, n_groups=2)
        ig.assignment = [0, 0, 1, 1]
        out = ig.forward([2.0, 4.0, 6.0, 8.0])
        self.assertAlmostEqual(out[0], 3.0)
        self.assertAlmostEqual(out[1], 7.0)


@pytest.mark.ci
class TestOutputGrouping(unittest.TestCase):

    def test_expands_outputs(self):
        from yane.evolution.experimental import OutputGrouping
        og = OutputGrouping(n_outputs=2, n_actions=5)
        out = og.forward([0.0, 1.0])
        self.assertEqual(len(out), 5)


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
