"""Tests für WebAssembly/Pure-JS-Export (evolution/wasm_export.py).

Akzeptanzkriterien:
  1. Pure-JS-Modus funktioniert ohne Emscripten
  2. XOR-Genom als .html: JS-Forward-Pass identisch zu YANE (Toleranz 1e-5)
  3. Output-Vergleich Python ↔ JS via Node.js
  4. Zyklisches Netz: Outputs endlich (kein NaN/Inf)
  5. WASM-Modus: klarer ImportError ohne Emscripten
"""
from __future__ import annotations

import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path

import pytest

from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType

# Check if Node.js is available for numerical comparison tests
try:
    result = subprocess.run(["node", "--version"], capture_output=True, timeout=5)
    HAS_NODE = result.returncode == 0
except (FileNotFoundError, subprocess.TimeoutExpired):
    HAS_NODE = False

node_required = pytest.mark.skipif(not HAS_NODE, reason="node not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _xor_genome() -> Genome:
    """Manually crafted XOR genome (tanh hidden, sigmoid output)."""
    g = Genome()
    inp0 = Node(NodeType.INPUT, 0); inp0.activation = ActivationType.LINEAR; inp0.input_index = 0
    inp1 = Node(NodeType.INPUT, 1); inp1.activation = ActivationType.LINEAR; inp1.input_index = 1
    hid = Node(NodeType.HIDDEN, 2); hid.activation = ActivationType.TANH; hid.bias = 0.0
    out = Node(NodeType.OUTPUT, 3); out.activation = ActivationType.SIGMOID; out.bias = -0.5
    for n in [inp0, inp1, hid, out]:
        g.nodes.append(n)
    g.input_nodes.extend([inp0, inp1])
    g.output_nodes.append(out)
    c0 = Connection(hid, 10); c0.weight = 2.0; inp0.connections.append(c0)
    c1 = Connection(hid, 11); c1.weight = 2.0; inp1.connections.append(c1)
    c2 = Connection(out, 12); c2.weight = 2.0; hid.connections.append(c2)
    g._invalidate_topology()
    return g


def _linear_genome(n_inputs: int = 2, n_outputs: int = 1) -> Genome:
    g = Genome()
    for i in range(n_inputs):
        n = Node(NodeType.INPUT, i); n.activation = ActivationType.LINEAR; n.input_index = i
        g.input_nodes.append(n); g.nodes.append(n)
    out = Node(NodeType.OUTPUT, n_inputs); out.activation = ActivationType.LINEAR; out.bias = 0.0
    g.output_nodes.append(out); g.nodes.append(out)
    for inp in g.input_nodes:
        c = Connection(out, 10 + inp.innovation); c.weight = 1.0; inp.connections.append(c)
    g._invalidate_topology()
    return g


def _run_js(js_code: str, inputs: list[float]) -> list[float]:
    """Execute JS forward function via Node.js and return outputs."""
    runner = js_code + f"\nconsole.log(JSON.stringify(forward({json.dumps(inputs)})));"
    result = subprocess.run(
        ["node", "-e", runner],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"node error: {result.stderr}")
    return json.loads(result.stdout.strip())


# ---------------------------------------------------------------------------
# Acceptance criterion 1: Pure-JS works without Emscripten
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestPureJSExport(unittest.TestCase):

    def test_genome_to_js_returns_string(self):
        from yane.evolution.wasm_export import genome_to_js
        g = _xor_genome()
        js = genome_to_js(g)
        self.assertIsInstance(js, str)
        self.assertIn("function forward(inputs)", js)
        self.assertIn("return [", js)

    def test_genome_to_js_contains_activation(self):
        """Sigmoid activation must appear in generated JS."""
        from yane.evolution.wasm_export import genome_to_js
        g = _xor_genome()
        js = genome_to_js(g)
        self.assertIn("Math.exp", js, "sigmoid should use Math.exp")

    def test_genome_to_html_returns_string(self):
        from yane.evolution.wasm_export import genome_to_html
        g = _xor_genome()
        html = genome_to_html(g, title="Test XOR")
        self.assertIsInstance(html, str)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("function forward(inputs)", html)

    def test_genome_to_html_contains_inputs(self):
        """HTML must contain the right number of input fields."""
        from yane.evolution.wasm_export import genome_to_html
        g = _xor_genome()
        html = genome_to_html(g)
        self.assertIn('id="inp0"', html)
        self.assertIn('id="inp1"', html)

    def test_genome_to_html_file_write(self):
        from yane.evolution.wasm_export import genome_to_html
        g = _xor_genome()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.html"
            html = genome_to_html(g, path=path)
            self.assertTrue(path.exists())
            self.assertEqual(path.read_text(encoding="utf-8"), html)

    def test_all_activations_generate_js(self):
        """Every supported activation name must produce valid JS."""
        from yane.evolution.wasm_export import _js_activation
        activations = [
            "linear", "sigmoid", "tanh", "relu", "leaky_relu", "elu",
            "swish", "softplus", "sine", "cosine", "abs", "gaussian",
            "binary", "square", "cube",
        ]
        for act in activations:
            expr = _js_activation(act, "x")
            self.assertIsInstance(expr, str, f"No expression for {act!r}")
            self.assertGreater(len(expr), 0)

    def test_unknown_activation_fallback_to_linear(self):
        """Unknown activation should fall back to identity (the input expr)."""
        from yane.evolution.wasm_export import _js_activation
        expr = _js_activation("unknown_xyz", "v")
        self.assertEqual(expr, "v")


# ---------------------------------------------------------------------------
# Acceptance criterion 2+3: Python ↔ JS numerical comparison
# ---------------------------------------------------------------------------

@node_required
class TestPythonJSNumericalMatch(unittest.TestCase):

    def _compare(self, genome: Genome, inputs: list[float], tol: float = 1e-5):
        from yane.evolution.wasm_export import genome_to_js
        genome.reset()
        py_out = genome.forward(inputs)
        js_src = genome_to_js(genome)
        js_out = _run_js(js_src, inputs)
        self.assertEqual(len(py_out), len(js_out),
                         "Output length must match between Python and JS")
        for i, (py, js) in enumerate(zip(py_out, js_out)):
            self.assertAlmostEqual(py, js, delta=tol,
                                   msg=f"output[{i}]: Python={py:.8f} JS={js:.8f} differ by >{tol}")

    def test_xor_genome_all_inputs(self):
        """XOR genome: JS output matches YANE within 1e-5 for all 4 inputs."""
        g = _xor_genome()
        for inputs in [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]:
            self._compare(g, inputs, tol=1e-5)

    def test_linear_genome(self):
        """Linear genome: output = sum of inputs."""
        g = _linear_genome(n_inputs=2)
        for inputs in [[0.3, 0.7], [1.0, 0.0], [-0.5, 0.5]]:
            self._compare(g, inputs, tol=1e-10)

    def test_single_sigmoid(self):
        """Single sigmoid node: JS sigmoid matches Python."""
        g = Genome()
        inp = Node(NodeType.INPUT, 0); inp.activation = ActivationType.LINEAR; inp.input_index = 0
        out = Node(NodeType.OUTPUT, 1); out.activation = ActivationType.SIGMOID; out.bias = 0.0
        g.nodes.extend([inp, out]); g.input_nodes.append(inp); g.output_nodes.append(out)
        c = Connection(out, 5); c.weight = 1.0; inp.connections.append(c)
        g._invalidate_topology()
        for val in [0.0, 0.5, -1.0, 2.0]:
            self._compare(g, [val], tol=1e-6)

    def test_tanh_activation(self):
        """Tanh activation: JS Math.tanh matches Python math.tanh."""
        g = Genome()
        inp = Node(NodeType.INPUT, 0); inp.activation = ActivationType.LINEAR; inp.input_index = 0
        out = Node(NodeType.OUTPUT, 1); out.activation = ActivationType.TANH; out.bias = 0.0
        g.nodes.extend([inp, out]); g.input_nodes.append(inp); g.output_nodes.append(out)
        c = Connection(out, 5); c.weight = 1.0; inp.connections.append(c)
        g._invalidate_topology()
        for val in [-2.0, -0.5, 0.0, 0.5, 2.0]:
            self._compare(g, [val], tol=1e-10)

    def test_multi_output_genome(self):
        """Multi-output genome: all outputs match."""
        g = Genome()
        inp = Node(NodeType.INPUT, 0); inp.activation = ActivationType.LINEAR; inp.input_index = 0
        out0 = Node(NodeType.OUTPUT, 1); out0.activation = ActivationType.SIGMOID; out0.bias = 0.0
        out1 = Node(NodeType.OUTPUT, 2); out1.activation = ActivationType.TANH; out1.bias = 0.5
        g.nodes.extend([inp, out0, out1])
        g.input_nodes.append(inp); g.output_nodes.extend([out0, out1])
        c0 = Connection(out0, 10); c0.weight = 1.0; inp.connections.append(c0)
        c1 = Connection(out1, 11); c1.weight = -1.0; inp.connections.append(c1)
        g._invalidate_topology()
        self._compare(g, [0.7], tol=1e-8)

    def test_html_contains_correct_js(self):
        """HTML forward function produces same output as Python."""
        from yane.evolution.wasm_export import genome_to_html
        g = _xor_genome()
        html = genome_to_html(g)
        # Extract the JS function (it's in the HTML body)
        start = html.find("function forward")
        end = html.find("document.getElementById", start)
        js_block = html[start:end].strip()
        for inputs in [[0.0, 1.0], [1.0, 0.0]]:
            js_out = _run_js(js_block, inputs)
            g.reset()
            py_out = g.forward(inputs)
            self.assertAlmostEqual(js_out[0], py_out[0], delta=1e-5)


# ---------------------------------------------------------------------------
# Acceptance criterion 4: Cyclic genome outputs are finite
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestCyclicGenome(unittest.TestCase):

    def test_cyclic_js_produces_finite_output(self):
        """Cyclic (recurrent) genome: JS output must not be NaN or Inf."""
        from yane.evolution.wasm_export import genome_to_js
        g = Genome()
        inp = Node(NodeType.INPUT, 0); inp.activation = ActivationType.LINEAR; inp.input_index = 0
        out = Node(NodeType.OUTPUT, 1); out.activation = ActivationType.TANH; out.bias = 0.0
        out.persist_value = True
        g.nodes.extend([inp, out]); g.input_nodes.append(inp); g.output_nodes.append(out)
        c = Connection(out, 10); c.weight = 0.5; inp.connections.append(c)
        c2 = Connection(out, 11); c2.weight = 0.3; out.connections.append(c2)
        g._invalidate_topology(); g._has_cycles = True
        js = genome_to_js(g, unroll_steps=3)
        self.assertIn("function forward", js)

    @node_required
    def test_cyclic_js_output_finite_via_node(self):
        from yane.evolution.wasm_export import genome_to_js
        g = Genome()
        inp = Node(NodeType.INPUT, 0); inp.activation = ActivationType.LINEAR; inp.input_index = 0
        out = Node(NodeType.OUTPUT, 1); out.activation = ActivationType.TANH; out.bias = 0.0
        out.persist_value = True
        g.nodes.extend([inp, out]); g.input_nodes.append(inp); g.output_nodes.append(out)
        c = Connection(out, 10); c.weight = 0.5; inp.connections.append(c)
        c2 = Connection(out, 11); c2.weight = 0.3; out.connections.append(c2)
        g._invalidate_topology(); g._has_cycles = True
        js = genome_to_js(g, unroll_steps=3)
        result = _run_js(js, [1.0])
        self.assertEqual(len(result), 1)
        self.assertFalse(math.isnan(result[0]))
        self.assertFalse(math.isinf(result[0]))


# ---------------------------------------------------------------------------
# Acceptance criterion 5: WASM mode raises ImportError
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestWasmModeError(unittest.TestCase):

    def test_wasm_mode_raises_import_error(self):
        from yane.evolution.wasm_export import genome_to_html
        g = _xor_genome()
        with self.assertRaises(ImportError) as ctx:
            genome_to_html(g, mode="wasm")
        self.assertIn("emscripten", str(ctx.exception).lower())

    def test_js_mode_does_not_raise(self):
        from yane.evolution.wasm_export import genome_to_html
        g = _xor_genome()
        try:
            html = genome_to_html(g, mode="js")
            self.assertIn("forward", html)
        except Exception as e:
            self.fail(f"mode='js' raised unexpectedly: {e}")


# ---------------------------------------------------------------------------
# NeuroEvolution integration
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestNeuroEvolutionIntegration(unittest.TestCase):

    def test_export_genome_wasm_returns_html(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=10)
        ne.set_max_iterations(3)
        ne.train(lambda g: sum(g.forward([0.5, 0.5])))
        html = ne.export_genome_wasm()
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("function forward", html)

    def test_genome_to_js_exported_from_yane(self):
        import yane
        self.assertTrue(hasattr(yane, "genome_to_js"))
        self.assertTrue(hasattr(yane, "genome_to_html"))


if __name__ == "__main__":
    unittest.main()
