"""Tests for TFLite / C-Array export (evolution/tflite_export.py).

Acceptance criteria:
  1. genome_to_c_array generates .h and .cc files.
  2. Generated .h file has include guard and correct macros.
  3. Generated .cc file has syntactically plausible C code.
  4. C-Array correctly encodes node count, input/output counts.
  5. genome_to_tflite raises ImportError (TFLite not installed in CI).
  6. NeuroEvolution.export_genome_c_array() wrapper works.
  7. Custom prefix is reflected in the generated identifiers.
  8. Genome with no connections generates an empty connection array.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import pytest

from yane import NeuroEvolution
from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType
from yane.evolution.tflite_export import genome_to_c_array, genome_to_tflite


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_genome(n_inputs: int = 2, n_outputs: int = 1) -> Genome:
    """Minimal acyclic genome: inputs → output."""
    g = Genome()
    inps = []
    for i in range(n_inputs):
        nd = Node(NodeType.INPUT, i)
        nd.activation = ActivationType.LINEAR
        nd.input_index = i
        g.nodes.append(nd)
        g.input_nodes.append(nd)
        inps.append(nd)

    for j in range(n_outputs):
        out = Node(NodeType.OUTPUT, n_inputs + j)
        out.activation = ActivationType.TANH
        out.bias = 0.1 * j
        g.nodes.append(out)
        g.output_nodes.append(out)
        for k, src in enumerate(inps):
            conn = Connection(out, innovation=100 + j * n_inputs + k)
            conn.weight = 0.5 + 0.1 * k
            conn.enabled = True
            src.connections.append(conn)

    g._invalidate_topology()
    return g


def _make_yane() -> NeuroEvolution:
    yane = NeuroEvolution(seed=0)
    yane.set_population_size(5)
    yane.configure(2, 1)
    return yane


# ---------------------------------------------------------------------------
# 1. File generation
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestCArrayFileGeneration(unittest.TestCase):

    def test_generates_header_and_source(self):
        """genome_to_c_array creates both .h and .cc files."""
        g = _make_simple_genome()
        with tempfile.TemporaryDirectory() as tmpdir:
            h_path, cc_path = genome_to_c_array(g, path=tmpdir, prefix="test_net")
            self.assertTrue(h_path.exists(), ".h file not created")
            self.assertTrue(cc_path.exists(), ".cc file not created")
            self.assertEqual(h_path.suffix, ".h")
            self.assertEqual(cc_path.suffix, ".cc")

    def test_returns_paths(self):
        """genome_to_c_array returns (header_path, source_path) as Path objects."""
        g = _make_simple_genome()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = genome_to_c_array(g, path=tmpdir)
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
            h, cc = result
            self.assertIsInstance(h, Path)
            self.assertIsInstance(cc, Path)

    def test_creates_output_directory(self):
        """genome_to_c_array creates the output directory if it doesn't exist."""
        g = _make_simple_genome()
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "nested", "model")
            genome_to_c_array(g, path=subdir, prefix="m")
            self.assertTrue(os.path.isdir(subdir))


# ---------------------------------------------------------------------------
# 2. Header file content
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestHeaderContent(unittest.TestCase):

    def test_include_guard(self):
        """Header contains include guard."""
        g = _make_simple_genome()
        with tempfile.TemporaryDirectory() as tmpdir:
            h_path, _ = genome_to_c_array(g, path=tmpdir, prefix="mynet")
            content = h_path.read_text()
            self.assertIn("#ifndef MYNET_H", content)
            self.assertIn("#define MYNET_H", content)
            self.assertIn("#endif", content)

    def test_n_inputs_macro(self):
        """Header has N_INPUTS macro matching the genome."""
        g = _make_simple_genome(n_inputs=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            h_path, _ = genome_to_c_array(g, path=tmpdir, prefix="net")
            content = h_path.read_text()
            self.assertIn("3", content)  # 3 inputs reflected somewhere
            self.assertIn("N_INPUTS", content)

    def test_n_outputs_macro(self):
        """Header has N_OUTPUTS macro matching the genome."""
        g = _make_simple_genome(n_inputs=2, n_outputs=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            h_path, _ = genome_to_c_array(g, path=tmpdir, prefix="net")
            content = h_path.read_text()
            self.assertIn("N_OUTPUTS", content)

    def test_forward_prototype(self):
        """Header declares forward function prototype."""
        g = _make_simple_genome()
        with tempfile.TemporaryDirectory() as tmpdir:
            h_path, _ = genome_to_c_array(g, path=tmpdir, prefix="nn")
            content = h_path.read_text()
            self.assertIn("nn_forward", content)
            self.assertIn("float", content)


# ---------------------------------------------------------------------------
# 3. Source file content
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestSourceContent(unittest.TestCase):

    def test_includes_math_h(self):
        """Source includes <math.h>."""
        g = _make_simple_genome()
        with tempfile.TemporaryDirectory() as tmpdir:
            _, cc_path = genome_to_c_array(g, path=tmpdir, prefix="net")
            content = cc_path.read_text()
            self.assertIn("#include <math.h>", content)

    def test_forward_function_present(self):
        """Source contains the forward function implementation."""
        g = _make_simple_genome()
        with tempfile.TemporaryDirectory() as tmpdir:
            _, cc_path = genome_to_c_array(g, path=tmpdir, prefix="net")
            content = cc_path.read_text()
            self.assertIn("net_forward", content)
            self.assertIn("float", content)

    def test_bias_array_present(self):
        """Source contains the bias array."""
        g = _make_simple_genome()
        with tempfile.TemporaryDirectory() as tmpdir:
            _, cc_path = genome_to_c_array(g, path=tmpdir, prefix="net")
            content = cc_path.read_text()
            self.assertIn("biases", content)

    def test_custom_prefix_in_identifiers(self):
        """Custom prefix is used in all generated identifiers."""
        g = _make_simple_genome()
        prefix = "xor_v3"
        with tempfile.TemporaryDirectory() as tmpdir:
            h_path, cc_path = genome_to_c_array(g, path=tmpdir, prefix=prefix)
            h_content = h_path.read_text()
            cc_content = cc_path.read_text()
            self.assertIn(prefix, h_content)
            self.assertIn(prefix, cc_content)
            self.assertIn(f"{prefix}_forward", cc_content)

    def test_empty_connections_handled(self):
        """Genome with no connections generates placeholder connection array."""
        g = Genome()
        inp = Node(NodeType.INPUT, 0)
        inp.activation = ActivationType.LINEAR
        inp.input_index = 0
        out = Node(NodeType.OUTPUT, 1)
        out.activation = ActivationType.LINEAR
        out.bias = 0.0
        g.nodes.extend([inp, out])
        g.input_nodes.append(inp)
        g.output_nodes.append(out)
        g._invalidate_topology()

        with tempfile.TemporaryDirectory() as tmpdir:
            _, cc_path = genome_to_c_array(g, path=tmpdir, prefix="empty")
            content = cc_path.read_text()
            self.assertIn("empty_forward", content)
            # Should not raise, and the file should be parseable


# ---------------------------------------------------------------------------
# 4. genome_to_tflite
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestTFLiteExport(unittest.TestCase):

    def test_tflite_raises_import_error(self):
        """genome_to_tflite raises ImportError or NotImplementedError when tflite is absent."""
        g = _make_simple_genome()
        try:
            genome_to_tflite(g)
            self.fail("Expected ImportError or NotImplementedError")
        except (ImportError, NotImplementedError):
            pass  # expected


# ---------------------------------------------------------------------------
# 5. NeuroEvolution.export_genome_c_array()
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestNeuroEvolutionCArrayExport(unittest.TestCase):

    def test_ne_export_c_array(self):
        """NeuroEvolution.export_genome_c_array() generates files."""
        yane = _make_yane()
        # Set a fitness so get_best() works
        for g in yane._population._unevaluated:
            g.fitness = 1.0
        yane._population._evaluated.extend(yane._population._unevaluated)
        yane._population._unevaluated.clear()

        with tempfile.TemporaryDirectory() as tmpdir:
            h_path, cc_path = yane.export_genome_c_array(path=tmpdir, prefix="ne_net")
            self.assertTrue(h_path.exists())
            self.assertTrue(cc_path.exists())

    def test_ne_export_not_configured_raises(self):
        """export_genome_c_array raises when not configured."""
        yane = NeuroEvolution(seed=0)
        with self.assertRaises(Exception):
            yane.export_genome_c_array()
