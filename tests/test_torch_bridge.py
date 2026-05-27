"""Tests for YANE → PyTorch bridge."""
from __future__ import annotations
import unittest
import pytest

pytest.importorskip("torch")

from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType


def _make_genome() -> Genome:
    g = Genome()
    g.max_nodes = 10; g.max_connections = 10
    in1 = Node(NodeType.INPUT, 0); in1.activation = ActivationType.LINEAR
    in2 = Node(NodeType.INPUT, 1); in2.activation = ActivationType.LINEAR
    h1 = Node(NodeType.HIDDEN, 2); h1.activation = ActivationType.RELU; h1.bias = -0.2
    out = Node(NodeType.OUTPUT, 3); out.activation = ActivationType.SIGMOID
    g.nodes.extend([in1, in2, h1, out])
    g.input_nodes.extend([in1, in2]); g.output_nodes.append(out)
    for src, tgt, w in [(in1, h1, 0.5), (in2, h1, -0.3), (h1, out, 0.8)]:
        conn = Connection(tgt, innovation=len(g.input_nodes[0].connections)+10)
        conn.weight = w; src.connections.append(conn)
    g.fitness = 0.9; g._invalidate_topology()
    return g


@pytest.mark.ci
class TestTorchBridge(unittest.TestCase):

    def test_export_creates_module(self):
        import torch
        from yane.evolution.torch_bridge import genome_to_torch_module
        g = _make_genome()
        model = genome_to_torch_module(g)
        self.assertIsInstance(model, torch.nn.Module)

    def test_forward_matches_original(self):
        import torch
        from yane.evolution.torch_bridge import genome_to_torch_module
        g = _make_genome()
        model = genome_to_torch_module(g)
        model.eval()
        with torch.no_grad():
            out_torch = model(torch.tensor([1.0, 0.5])).tolist()
        out_yane = g.forward([1.0, 0.5])
        for a, b in zip(out_torch, out_yane):
            self.assertAlmostEqual(a, b, places=5)

    def test_forward_with_torch_fallback(self):
        from yane.evolution.torch_bridge import forward_with_torch
        g = _make_genome()
        result = forward_with_torch(g, [1.0, 0.5])
        expected = g.forward([1.0, 0.5])
        for a, b in zip(result, expected):
            self.assertAlmostEqual(a, b, places=5)


if __name__ == "__main__":
    unittest.main()
