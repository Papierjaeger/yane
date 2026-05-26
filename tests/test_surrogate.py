"""Tests for fitness surrogate model."""
from __future__ import annotations
import unittest
import pytest
from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType


def _make_genome(weight: float, fitness: float) -> Genome:
    g = Genome()
    g.max_nodes = 10
    g.max_connections = 10
    inp = Node(NodeType.INPUT, 0)
    inp.activation = ActivationType.LINEAR
    out = Node(NodeType.OUTPUT, 1)
    out.activation = ActivationType.LINEAR
    g.nodes.extend([inp, out])
    g.input_nodes.append(inp)
    g.output_nodes.append(out)
    conn = Connection(out, innovation=10)
    conn.weight = weight
    inp.connections.append(conn)
    g.fitness = fitness
    g._invalidate_topology()
    return g


@pytest.mark.ci
class TestFitnessSurrogate(unittest.TestCase):

    def test_predict_returns_none_before_training(self):
        from yane.evolution.surrogate import FitnessSurrogate
        s = FitnessSurrogate()
        g = _make_genome(1.0, 0.0)
        self.assertIsNone(s.predict(g))

    def test_warmup_returns_true(self):
        from yane.evolution.surrogate import FitnessSurrogate
        s = FitnessSurrogate(warmup_evals=10)
        g = _make_genome(1.0, 0.0)
        for _ in range(10):
            self.assertTrue(s.should_evaluate(g))

    def test_train_and_predict(self):
        from yane.evolution.surrogate import FitnessSurrogate
        s = FitnessSurrogate(warmup_evals=50)
        # Create enough training data with clear pattern
        genomes = [_make_genome(w, w * 10) for w in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0, 0.01, 0.05, 0.2, 0.8]]
        for g in genomes:
            s.should_evaluate(g)
        s.train(genomes)
        # After training, prediction should work
        pred = s.predict(_make_genome(3.0, 0.0))
        self.assertIsNotNone(pred)

    def test_spearman_rho_after_training(self):
        from yane.evolution.surrogate import FitnessSurrogate
        s = FitnessSurrogate(warmup_evals=5)
        genomes = [_make_genome(w, w * 2) for w in range(1, 20)]
        for g in genomes:
            s.should_evaluate(g)
        s.train(genomes)
        rho = s.get_spearman_rho()
        self.assertGreater(rho, 0.5)  # strong correlation expected

    def test_get_diagnostics(self):
        from yane.evolution.surrogate import FitnessSurrogate
        s = FitnessSurrogate()
        diag = s.get_diagnostics()
        self.assertIn("surrogate_trained", diag)
        self.assertIn("surrogate_frac", diag)


if __name__ == "__main__":
    unittest.main()
