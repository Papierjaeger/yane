"""Tests for GenomeCodec protocol (PickleCodec, JsonCodec)."""
from __future__ import annotations

import unittest
import pytest

from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType


def _make_test_genome() -> Genome:
    """Create a small genome with known structure for round-trip tests."""
    g = Genome()
    g.max_nodes = 10
    g.max_connections = 10
    in1 = Node(NodeType.INPUT, 0)
    in1.activation = ActivationType.LINEAR
    in1.bias = 0.1
    h1 = Node(NodeType.HIDDEN, 1)
    h1.activation = ActivationType.RELU
    h1.bias = -0.2
    out = Node(NodeType.OUTPUT, 2)
    out.activation = ActivationType.SIGMOID
    out.bias = 0.3
    g.nodes.extend([in1, h1, out])
    g.input_nodes.append(in1)
    g.output_nodes.append(out)
    c1 = Connection(h1, innovation=10)
    c1.weight = 0.5
    in1.connections.append(c1)
    c2 = Connection(out, innovation=11)
    c2.weight = -0.8
    h1.connections.append(c2)
    g.fitness = 0.9
    g._invalidate_topology()
    return g


@pytest.mark.ci
class TestPickleCodec(unittest.TestCase):

    def test_roundtrip(self):
        from yane.evolution.codec import PickleCodec
        g = _make_test_genome()
        codec = PickleCodec()
        data = codec.encode(g)
        g2 = codec.decode(data)
        self.assertEqual(len(g2.nodes), len(g.nodes))
        self.assertEqual(g2.fitness, g.fitness)
        self.assertEqual(g2.max_nodes, g.max_nodes)

    def test_forward_after_roundtrip(self):
        from yane.evolution.codec import PickleCodec
        g = _make_test_genome()
        codec = PickleCodec()
        g2 = codec.decode(codec.encode(g))
        out1 = g.forward([1.0])
        out2 = g2.forward([1.0])
        self.assertAlmostEqual(out1[0], out2[0], places=10)


@pytest.mark.ci
class TestJsonCodec(unittest.TestCase):

    def test_roundtrip(self):
        from yane.evolution.codec import JsonCodec
        g = _make_test_genome()
        codec = JsonCodec()
        data = codec.encode(g)
        self.assertTrue(data.startswith(b"{"))
        g2 = codec.decode(data)
        self.assertEqual(len(g2.nodes), len(g.nodes))
        self.assertEqual(g2.max_nodes, g.max_nodes)

    def test_forward_after_roundtrip(self):
        from yane.evolution.codec import JsonCodec
        g = _make_test_genome()
        codec = JsonCodec()
        g2 = codec.decode(codec.encode(g))
        out1 = g.forward([1.0])
        out2 = g2.forward([1.0])
        self.assertAlmostEqual(out1[0], out2[0], places=10)

    def test_json_is_readable(self):
        from yane.evolution.codec import JsonCodec
        g = _make_test_genome()
        data = JsonCodec().encode(g)
        text = data.decode("utf-8")
        self.assertIn("sigmoid", text)
        self.assertIn("0.5", text)


@pytest.mark.ci
class TestCodecRegistry(unittest.TestCase):

    def test_register_and_get(self):
        from yane.evolution.codec import register_codec, get_codec, PickleCodec
        pc = PickleCodec()
        register_codec(pc)
        retrieved = get_codec("pickle")
        self.assertIsNotNone(retrieved)

    def test_detect_codec(self):
        from yane.evolution.codec import detect_codec, PickleCodec, JsonCodec
        g = _make_test_genome()
        pickled = PickleCodec().encode(g)
        self.assertEqual(detect_codec(pickled), "pickle")
        jsond = JsonCodec().encode(g)
        self.assertEqual(detect_codec(jsond), "json")
        with self.assertRaises(ValueError):
            detect_codec(b"not a checkpoint")

    def test_set_checkpoint_codec_api(self):
        from yane import NeuroEvolution
        ne = NeuroEvolution()
        ne.set_checkpoint_codec("json")
        self.assertEqual(ne._checkpoint_codec, "json")
        ne.set_checkpoint_codec("pickle")
        self.assertEqual(ne._checkpoint_codec, "pickle")

    def test_codec_checkpoint_roundtrip(self):
        """Save and load checkpoint with non-default codec."""
        import tempfile
        from pathlib import Path
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.set_population_size(30)
        yane.configure(2, 1)
        yane.set_max_iterations(50)
        def _eval(g): return sum(abs(c.weight) for src in g.nodes for c in src.connections)
        yane.train(_eval)
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            yane.set_checkpoint_codec("json")
            yane.save_checkpoint(f.name)
            self.assertGreater(Path(f.name).stat().st_size, 50)
            self.assertTrue(Path(f.name).read_bytes().lstrip().startswith(b"{"))
            # Load back
            yane2 = NeuroEvolution()
            yane2.set_checkpoint_codec("json")
            yane2.load_checkpoint(f.name)
            self.assertIsNotNone(yane2._population)


if __name__ == "__main__":
    from pathlib import Path
    unittest.main()
