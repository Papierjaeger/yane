"""Tests for ensemble evaluation (EnsembleGenome, make_ensemble, export)."""
from __future__ import annotations

import unittest
import pytest

from yane.core.genome import Genome
from yane.core.connection import Connection
from yane.core.node import Node, NodeType
from yane.util.activation import ActivationType


def _make_mock_genome(weight: float, fitness: float, bias: float = 0.0) -> Genome:
    """Create a simple genome: input → output with a fixed weight."""
    g = Genome()
    g.max_nodes = 10
    g.max_connections = 10
    inp = Node(NodeType.INPUT, 0)
    inp.activation = ActivationType.LINEAR
    out = Node(NodeType.OUTPUT, 1)
    out.activation = ActivationType.LINEAR
    out.bias = bias
    g.nodes.extend([inp, out])
    g.input_nodes.append(inp)
    g.output_nodes.append(out)
    conn = Connection(out, innovation=100)
    conn.weight = weight
    inp.connections.append(conn)
    g.fitness = fitness
    g._invalidate_topology()
    return g


@pytest.mark.ci
class TestEnsembleGenome(unittest.TestCase):

    def test_ensemble_mean(self):
        from yane.evolution.ensemble import EnsembleGenome
        m1 = _make_mock_genome(2.0, fitness=1.0)
        m2 = _make_mock_genome(4.0, fitness=0.5)
        ens = EnsembleGenome([m1, m2], mode="mean")
        out = ens.forward([1.0])
        # m1: 1*2=2, m2: 1*4=4, mean = 3
        self.assertAlmostEqual(out[0], 3.0)

    def test_ensemble_vote(self):
        from yane.evolution.ensemble import EnsembleGenome
        # Three members: 2 vote class 0, 1 votes class 1
        m1 = _make_mock_genome(0.0, fitness=1.0)
        m2 = _make_mock_genome(0.0, fitness=1.0)
        m3 = _make_mock_genome(0.0, fitness=1.0)
        # Modify one to produce higher output on index 1 than index 0
        m3.output_nodes[0].bias = 5.0  # output will be higher
        m1.output_nodes[0].bias = -5.0  # output will be negative
        m2.output_nodes[0].bias = -5.0
        # Actually let's make 2-output genomes for voting to make sense
        m1 = _make_mock_genome(1.0, fitness=1.0, bias=-5.0)
        m2 = _make_mock_genome(1.0, fitness=1.0, bias=-5.0)
        m3 = _make_mock_genome(1.0, fitness=1.0, bias=5.0)
        # Need 2 outputs for vote to be meaningful
        for m in [m1, m2, m3]:
            out2 = Node(NodeType.OUTPUT, 99)
            out2.activation = ActivationType.LINEAR
            m.nodes.append(out2)
            m.output_nodes.append(out2)
            # Both outputs get same input via single connection
            conn2 = Connection(out2, innovation=101)
            conn2.weight = 1.0
            m.input_nodes[0].connections.append(conn2)
            m._invalidate_topology()

        ens = EnsembleGenome([m1, m2, m3], mode="vote")
        out = ens.forward([1.0])
        # With input=1.0 and weight=1.0 to both outputs:
        # Member1: out0=1.0+(-5.0)=-4.0, out1=1.0+0=1.0 → votes for out1
        # Member2: same → votes for out1
        # Member3: out0=1.0+5.0=6.0, out1=1.0+0=1.0 → votes for out0
        # So out1 gets 2/3 votes, out0 gets 1/3
        self.assertAlmostEqual(out[0], 1.0 / 3.0)
        self.assertAlmostEqual(out[1], 2.0 / 3.0)

    def test_ensemble_weighted(self):
        from yane.evolution.ensemble import EnsembleGenome
        m1 = _make_mock_genome(10.0, fitness=0.9)  # weight=10, fitness=0.9 → weight=0.9
        m2 = _make_mock_genome(0.0, fitness=0.1)   # weight=0, fitness=0.1 → weight=0.1
        ens = EnsembleGenome([m1, m2], mode="weighted")
        out = ens.forward([1.0])
        # weighted: 0.9*10 + 0.1*0 = 9.0
        self.assertAlmostEqual(out[0], 9.0)

    def test_ensemble_weighted_all_negative_fitnesses(self):
        from yane.evolution.ensemble import EnsembleGenome
        m1 = _make_mock_genome(1.0, fitness=-1.0)
        m2 = _make_mock_genome(3.0, fitness=-2.0)
        ens = EnsembleGenome([m1, m2], mode="weighted")
        out = ens.forward([1.0])
        self.assertNotAlmostEqual(out[0], 0.0)
        self.assertAlmostEqual(out[0], 1.0)

    def test_ensemble_empty_raises(self):
        from yane.evolution.ensemble import EnsembleGenome
        with self.assertRaises(ValueError):
            EnsembleGenome([])

    def test_ensemble_to_python_produces_valid_code(self):
        from yane.evolution.ensemble import EnsembleGenome
        m1 = _make_mock_genome(2.0, fitness=1.0)
        m2 = _make_mock_genome(4.0, fitness=0.5)
        ens = EnsembleGenome([m1, m2], mode="mean")
        code = ens.to_python("MyEnsemble")
        self.assertIn("def member0_forward(inputs)", code)
        self.assertIn("def member1_forward(inputs)", code)
        self.assertIn("def myensemble_forward(inputs)", code)
        # The code should be executable
        ns = {}
        exec(compile(code, "<test>", "exec"), ns)
        result = ns["myensemble_forward"]([1.0])
        self.assertAlmostEqual(result[0], 3.0)

    def test_make_ensemble_from_neuroevolution(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(1, 1)
        yane.set_max_iterations(30)
        def _eval(g):
            return sum(abs(c.weight) for src in g.nodes for c in src.connections)
        yane.train(_eval)
        ens = yane.make_ensemble(k=2, mode="mean")
        self.assertEqual(len(ens.members), 2)
        out = ens.forward([0.5])
        self.assertIsInstance(out, list)
        self.assertGreater(len(out), 0)

    def test_ensemble_different_modes_produce_different_results(self):
        from yane.evolution.ensemble import EnsembleGenome
        m1 = _make_mock_genome(1.0, fitness=9000.0)
        m2 = _make_mock_genome(10.0, fitness=10.0)
        ens_mean = EnsembleGenome([m1, m2], mode="mean")
        ens_weighted = EnsembleGenome([m1, m2], mode="weighted")
        out_mean = ens_mean.forward([1.0])
        out_weighted = ens_weighted.forward([1.0])
        self.assertNotAlmostEqual(out_mean[0], out_weighted[0])


if __name__ == "__main__":
    unittest.main()
