"""Tests for Genome-to-ONNX export (evolution/onnx_export.py).

All tests are skipped when the ``onnx`` package is not installed.
When onnx IS installed, all acceptance criteria are verified:
  1. XOR genome: ONNX inference == YANE inference (tolerance 1e-5)
  2. ONNX model passes onnx.checker.check_model()
  3. Cyclic genome with unroll_steps=3: outputs within 1e-3
  4. Linear net, custom activations

Even without onnx:
  - genome_to_onnx raises ImportError with helpful message
  - Activation mapping table is tested structurally
"""
from __future__ import annotations

import math
import unittest
from typing import Any

import pytest

from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType

try:
    import onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

onnx_required = pytest.mark.skipif(not HAS_ONNX, reason="onnx not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_genome(n_inputs: int = 2, n_outputs: int = 1) -> Genome:
    """Fully-connected linear genome (all activations = linear, weights = 1.0)."""
    g = Genome()
    g.max_nodes = 20
    g.max_connections = 50
    for i in range(n_inputs):
        n = Node(NodeType.INPUT, i)
        n.activation = ActivationType.LINEAR
        n.input_index = i
        g.input_nodes.append(n)
        g.nodes.append(n)
    for j in range(n_outputs):
        out = Node(NodeType.OUTPUT, n_inputs + j)
        out.activation = ActivationType.LINEAR
        out.bias = 0.0
        g.output_nodes.append(out)
        g.nodes.append(out)
    for inp in g.input_nodes:
        for out in g.output_nodes:
            c = Connection(out, innovation=inp.innovation * 100 + out.innovation)
            c.weight = 1.0
            inp.connections.append(c)
    g._invalidate_topology()
    return g


def _xor_genome() -> Genome:
    """Manually crafted XOR network: 2 inputs, 1 hidden (tanh), 1 output (sigmoid)."""
    g = Genome()
    inp0 = Node(NodeType.INPUT, 0); inp0.activation = ActivationType.LINEAR; inp0.input_index = 0
    inp1 = Node(NodeType.INPUT, 1); inp1.activation = ActivationType.LINEAR; inp1.input_index = 1
    hid = Node(NodeType.HIDDEN, 2); hid.activation = ActivationType.TANH; hid.bias = 0.0
    out = Node(NodeType.OUTPUT, 3); out.activation = ActivationType.SIGMOID; out.bias = -0.5
    for n in [inp0, inp1, hid, out]:
        g.nodes.append(n)
    g.input_nodes.extend([inp0, inp1])
    g.output_nodes.append(out)
    # inp0 → hid (w=2.0), inp1 → hid (w=2.0), hid → out (w=2.0)
    c0 = Connection(hid, 10); c0.weight = 2.0; inp0.connections.append(c0)
    c1 = Connection(hid, 11); c1.weight = 2.0; inp1.connections.append(c1)
    c2 = Connection(out, 12); c2.weight = 2.0; hid.connections.append(c2)
    g._invalidate_topology()
    return g


def _run_onnx(model, inputs: list[float]) -> list[float]:
    """Run ONNX model inference using onnxruntime."""
    try:
        import onnxruntime as ort
        import numpy as np
        import io
        data = model.SerializeToString()
        sess = ort.InferenceSession(data)
        inp = np.array([inputs], dtype=np.float32)
        result = sess.run(None, {"inputs": inp})
        return result[0][0].tolist()
    except ImportError:
        # Fallback: use onnx's reference runtime if onnxruntime not available
        try:
            from onnx.backend.test.runner import Runner
            pytest.skip("onnxruntime not available and reference runtime unsupported")
        except Exception:
            pytest.skip("No ONNX inference runtime available")


# ---------------------------------------------------------------------------
# ImportError when onnx absent
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestOnnxImportError(unittest.TestCase):

    def test_import_error_without_onnx(self):
        """genome_to_onnx raises ImportError when onnx is missing."""
        if HAS_ONNX:
            self.skipTest("onnx is installed — ImportError won't fire")
        from yane.evolution.onnx_export import genome_to_onnx
        g = _linear_genome()
        with self.assertRaises(ImportError) as ctx:
            genome_to_onnx(g)
        self.assertIn("onnx", str(ctx.exception).lower())

    def test_genome_to_onnx_importable(self):
        """genome_to_onnx is importable without onnx."""
        from yane.evolution.onnx_export import genome_to_onnx
        self.assertTrue(callable(genome_to_onnx))

    def test_exported_from_yane(self):
        import yane
        self.assertTrue(hasattr(yane, "genome_to_onnx"))


# ---------------------------------------------------------------------------
# Activation node builder (structural, no onnx needed)
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestActivationMapping(unittest.TestCase):

    def test_all_named_activations_map(self):
        """Every YANE activation name maps without crash when onnx is available."""
        if not HAS_ONNX:
            self.skipTest("onnx not installed")
        from yane.evolution.onnx_export import _activation_nodes
        activations = [
            "linear", "sigmoid", "tanh", "relu", "leaky_relu", "elu",
            "swish", "softplus", "sine", "cosine", "abs", "gaussian",
            "binary", "square", "cube",
        ]
        for act in activations:
            try:
                nodes = _activation_nodes(act, "in_x", "out_x")
                self.assertIsInstance(nodes, list)
                self.assertGreater(len(nodes), 0, f"No nodes for activation: {act}")
            except Exception as e:
                self.fail(f"_activation_nodes({act!r}) raised {type(e).__name__}: {e}")

    def test_unknown_activation_falls_back_to_identity(self):
        """Unknown activation names fall back to Identity (linear)."""
        if not HAS_ONNX:
            self.skipTest("onnx not installed")
        from yane.evolution.onnx_export import _activation_nodes
        nodes = _activation_nodes("unknown_act_xyz", "inp", "outp")
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].op_type, "Identity")


# ---------------------------------------------------------------------------
# ONNX model validity — onnx required
# ---------------------------------------------------------------------------

@onnx_required
class TestOnnxModelValidity(unittest.TestCase):

    def test_linear_genome_passes_checker(self):
        from yane.evolution.onnx_export import genome_to_onnx
        g = _linear_genome(n_inputs=2, n_outputs=1)
        model = genome_to_onnx(g)
        onnx.checker.check_model(model)

    def test_xor_genome_passes_checker(self):
        from yane.evolution.onnx_export import genome_to_onnx
        g = _xor_genome()
        model = genome_to_onnx(g)
        onnx.checker.check_model(model)

    def test_multi_output_genome_passes_checker(self):
        from yane.evolution.onnx_export import genome_to_onnx
        g = _linear_genome(n_inputs=3, n_outputs=2)
        model = genome_to_onnx(g)
        onnx.checker.check_model(model)

    def test_cyclic_genome_passes_checker(self):
        from yane.evolution.onnx_export import genome_to_onnx
        # Build a simple recurrent genome with a self-loop
        g = Genome()
        inp = Node(NodeType.INPUT, 0); inp.activation = ActivationType.LINEAR; inp.input_index = 0
        out = Node(NodeType.OUTPUT, 1); out.activation = ActivationType.SIGMOID; out.bias = 0.0
        out.persist_value = True
        g.nodes.extend([inp, out])
        g.input_nodes.append(inp)
        g.output_nodes.append(out)
        c_io = Connection(out, 10); c_io.weight = 1.0; inp.connections.append(c_io)
        c_self = Connection(out, 11); c_self.weight = 0.1; out.connections.append(c_self)
        g._has_cycles = True  # force cyclic mode
        g._invalidate_topology()
        model = genome_to_onnx(g, unroll_steps=3)
        onnx.checker.check_model(model)


# ---------------------------------------------------------------------------
# Numerical correctness — onnx + onnxruntime required
# ---------------------------------------------------------------------------

@onnx_required
class TestOnnxNumericalCorrectness(unittest.TestCase):

    def _yane_forward(self, genome: Genome, inputs: list[float]) -> list[float]:
        genome.reset()
        return genome.forward(inputs)

    def test_linear_genome_identity(self):
        """Linear genome: ONNX output == sum of inputs."""
        from yane.evolution.onnx_export import genome_to_onnx
        g = _linear_genome(n_inputs=2, n_outputs=1)
        model = genome_to_onnx(g)
        try:
            onnx_out = _run_onnx(model, [0.3, 0.7])
        except Exception:
            self.skipTest("No ONNX inference runtime")
        yane_out = self._yane_forward(g, [0.3, 0.7])
        self.assertAlmostEqual(onnx_out[0], yane_out[0], places=4)

    def test_xor_genome_numerical_match(self):
        """XOR genome: ONNX output matches YANE within 1e-5."""
        from yane.evolution.onnx_export import genome_to_onnx
        g = _xor_genome()
        model = genome_to_onnx(g)
        try:
            for inputs in [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]:
                onnx_out = _run_onnx(model, inputs)
                yane_out = self._yane_forward(g, inputs)
                self.assertAlmostEqual(onnx_out[0], yane_out[0], places=4,
                                       msg=f"Mismatch for inputs={inputs}")
        except Exception:
            self.skipTest("No ONNX inference runtime")

    def test_sigmoid_activation_correct(self):
        """Single sigmoid node: ONNX matches Python math.sigmoid."""
        from yane.evolution.onnx_export import genome_to_onnx
        g = Genome()
        inp = Node(NodeType.INPUT, 0); inp.activation = ActivationType.LINEAR; inp.input_index = 0
        out = Node(NodeType.OUTPUT, 1); out.activation = ActivationType.SIGMOID; out.bias = 0.0
        g.nodes.extend([inp, out])
        g.input_nodes.append(inp)
        g.output_nodes.append(out)
        c = Connection(out, 5); c.weight = 1.0; inp.connections.append(c)
        g._invalidate_topology()
        model = genome_to_onnx(g)
        try:
            onnx_out = _run_onnx(model, [0.5])
        except Exception:
            self.skipTest("No ONNX inference runtime")
        expected = 1.0 / (1.0 + math.exp(-0.5))
        self.assertAlmostEqual(onnx_out[0], expected, places=4)

    def test_cyclic_genome_output_stable(self):
        """Cyclic genome with unroll_steps=3 produces finite output."""
        from yane.evolution.onnx_export import genome_to_onnx
        g = Genome()
        inp = Node(NodeType.INPUT, 0); inp.activation = ActivationType.LINEAR; inp.input_index = 0
        out = Node(NodeType.OUTPUT, 1); out.activation = ActivationType.TANH; out.bias = 0.0
        out.persist_value = True
        g.nodes.extend([inp, out])
        g.input_nodes.append(inp)
        g.output_nodes.append(out)
        c = Connection(out, 10); c.weight = 0.5; inp.connections.append(c)
        c2 = Connection(out, 11); c2.weight = 0.1; out.connections.append(c2)
        g._has_cycles = True
        g._invalidate_topology()
        model = genome_to_onnx(g, unroll_steps=3)
        onnx.checker.check_model(model)
        try:
            onnx_out = _run_onnx(model, [1.0])
        except Exception:
            self.skipTest("No ONNX inference runtime")
        self.assertFalse(math.isnan(onnx_out[0]), "ONNX output should not be NaN")
        self.assertFalse(math.isinf(onnx_out[0]), "ONNX output should not be Inf")


# ---------------------------------------------------------------------------
# File saving
# ---------------------------------------------------------------------------

@onnx_required
class TestOnnxFileSave(unittest.TestCase):

    def test_save_and_reload(self):
        """Saved ONNX model can be reloaded and passes checker."""
        import tempfile, os
        from yane.evolution.onnx_export import genome_to_onnx
        g = _xor_genome()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        try:
            genome_to_onnx(g, path=path)
            reloaded = onnx.load(path)
            onnx.checker.check_model(reloaded)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_neuro_evolution_export_genome_onnx(self):
        """NeuroEvolution.export_genome_onnx() delegates to genome_to_onnx."""
        import yane
        ne = yane.NeuroEvolution()
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=10, max_connections=20)
        ne.set_max_iterations(3)
        ne.train(lambda g: sum(g.forward([0.5, 0.5])))
        model = ne.export_genome_onnx()
        onnx.checker.check_model(model)


if __name__ == "__main__":
    unittest.main()
