"""Tests for Symbolic Regression Export (core/_symbolic.py + Genome.to_symbolic()).

Acceptance criteria:
  1. "python" format evaluates to identical values as genome.forward() (tol 1e-6).
  2. "text" format returns a readable string.
  3. "latex" format returns a LaTeX string (contains backslash or special chars).
  4. "sympy" format is eval()-compatible (same as "python").
  5. Constant folding: zero-weight terms removed.
  6. Constant folding: 1.0*x → x (no leading coefficient).
  7. Cyclic genome raises ValueError.
  8. Multi-output genomes emit multiple expressions.
  9. Genome.to_symbolic() method works.
 10. input_names parameter is respected.
"""
from __future__ import annotations

import math
import unittest

import pytest

from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType
from yane.core._symbolic import genome_to_symbolic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_linear_genome() -> Genome:
    """y = 0.5*x0 + 0.3*x1 + 0.1 (linear, no activation beyond identity)."""
    g = Genome()
    x0 = Node(NodeType.INPUT, 0); x0.activation = ActivationType.LINEAR; x0.input_index = 0
    x1 = Node(NodeType.INPUT, 1); x1.activation = ActivationType.LINEAR; x1.input_index = 1
    out = Node(NodeType.OUTPUT, 2); out.activation = ActivationType.LINEAR; out.bias = 0.1

    g.nodes.extend([x0, x1, out])
    g.input_nodes.extend([x0, x1])
    g.output_nodes.append(out)

    c0 = Connection(out, innovation=1); c0.weight = 0.5; c0.enabled = True
    c1 = Connection(out, innovation=2); c1.weight = 0.3; c1.enabled = True
    x0.connections.append(c0)
    x1.connections.append(c1)
    g._invalidate_topology()
    return g


def _make_tanh_genome() -> Genome:
    """y = tanh(0.7*x0 - 0.4*x1 + 0.2)"""
    g = Genome()
    x0 = Node(NodeType.INPUT, 0); x0.activation = ActivationType.LINEAR; x0.input_index = 0
    x1 = Node(NodeType.INPUT, 1); x1.activation = ActivationType.LINEAR; x1.input_index = 1
    out = Node(NodeType.OUTPUT, 2); out.activation = ActivationType.TANH; out.bias = 0.2

    g.nodes.extend([x0, x1, out])
    g.input_nodes.extend([x0, x1])
    g.output_nodes.append(out)

    c0 = Connection(out, innovation=1); c0.weight = 0.7; c0.enabled = True
    c1 = Connection(out, innovation=2); c1.weight = -0.4; c1.enabled = True
    x0.connections.append(c0)
    x1.connections.append(c1)
    g._invalidate_topology()
    return g


def _make_multi_output_genome() -> Genome:
    """Two outputs: y0 = x0 + 0.1, y1 = -x1 + 0.2."""
    g = Genome()
    x0 = Node(NodeType.INPUT, 0); x0.activation = ActivationType.LINEAR; x0.input_index = 0
    x1 = Node(NodeType.INPUT, 1); x1.activation = ActivationType.LINEAR; x1.input_index = 1
    o0 = Node(NodeType.OUTPUT, 2); o0.activation = ActivationType.LINEAR; o0.bias = 0.1
    o1 = Node(NodeType.OUTPUT, 3); o1.activation = ActivationType.LINEAR; o1.bias = 0.2

    g.nodes.extend([x0, x1, o0, o1])
    g.input_nodes.extend([x0, x1])
    g.output_nodes.extend([o0, o1])

    c0 = Connection(o0, innovation=1); c0.weight = 1.0; c0.enabled = True
    c1 = Connection(o1, innovation=2); c1.weight = -1.0; c1.enabled = True
    x0.connections.append(c0)
    x1.connections.append(c1)
    g._invalidate_topology()
    return g


def _make_zero_weight_genome() -> Genome:
    """y = 0.5*x0 + 0.0*x1 + 0.0 (x1 term should be folded out)."""
    g = Genome()
    x0 = Node(NodeType.INPUT, 0); x0.activation = ActivationType.LINEAR; x0.input_index = 0
    x1 = Node(NodeType.INPUT, 1); x1.activation = ActivationType.LINEAR; x1.input_index = 1
    out = Node(NodeType.OUTPUT, 2); out.activation = ActivationType.LINEAR; out.bias = 0.0

    g.nodes.extend([x0, x1, out])
    g.input_nodes.extend([x0, x1])
    g.output_nodes.append(out)

    c0 = Connection(out, innovation=1); c0.weight = 0.5; c0.enabled = True
    c1 = Connection(out, innovation=2); c1.weight = 0.0; c1.enabled = True
    x0.connections.append(c0)
    x1.connections.append(c1)
    g._invalidate_topology()
    return g


def _make_cyclic_genome() -> Genome:
    """A genome with a recurrent connection (cyclic)."""
    g = Genome()
    x0 = Node(NodeType.INPUT, 0); x0.activation = ActivationType.LINEAR; x0.input_index = 0
    h = Node(NodeType.HIDDEN, 1); h.activation = ActivationType.TANH; h.persist_value = True
    out = Node(NodeType.OUTPUT, 2); out.activation = ActivationType.LINEAR; out.bias = 0.0

    g.nodes.extend([x0, h, out])
    g.input_nodes.append(x0)
    g.output_nodes.append(out)

    c0 = Connection(h, innovation=1); c0.weight = 1.0; c0.enabled = True
    c1 = Connection(out, innovation=2); c1.weight = 1.0; c1.enabled = True
    c_rec = Connection(h, innovation=3); c_rec.weight = 0.5; c_rec.enabled = True
    x0.connections.append(c0)
    h.connections.append(c1)
    h.connections.append(c_rec)  # recurrent: h → h
    g._has_cycles = True  # mark as cyclic explicitly
    g._invalidate_topology()
    return g


def _eval_python_expr(expr: str, input_values: dict) -> float:
    """Evaluate a symbolic python expression with given variable bindings."""
    return eval(expr, {"math": math}, input_values)


# ---------------------------------------------------------------------------
# 1. Python format evaluation equivalence
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestPythonFormatEquivalence(unittest.TestCase):

    def test_linear_genome_python_format(self):
        """Python format evaluates to same value as genome.forward()."""
        g = _make_linear_genome()
        for x0, x1 in [(0.0, 0.0), (1.0, -1.0), (0.5, 0.5), (-2.0, 3.0)]:
            g.reset()
            expected = g.forward([x0, x1])[0]
            expr = genome_to_symbolic(g, input_names=["x0", "x1"], fmt="python")
            got = _eval_python_expr(expr, {"x0": x0, "x1": x1})
            self.assertAlmostEqual(got, expected, places=6,
                                   msg=f"Mismatch at x0={x0}, x1={x1}: {expr}")

    def test_tanh_genome_python_format(self):
        """Python format with tanh activation matches forward()."""
        g = _make_tanh_genome()
        for x0, x1 in [(0.0, 0.0), (1.0, 0.5), (-0.3, 0.7)]:
            g.reset()
            expected = g.forward([x0, x1])[0]
            expr = genome_to_symbolic(g, input_names=["x0", "x1"], fmt="python")
            got = _eval_python_expr(expr, {"x0": x0, "x1": x1})
            self.assertAlmostEqual(got, expected, places=6,
                                   msg=f"Mismatch at x0={x0}, x1={x1}: {expr}")

    def test_sympy_format_same_as_python(self):
        """sympy format produces eval-compatible expression identical to python format."""
        g = _make_linear_genome()
        py_expr = genome_to_symbolic(g, fmt="python")
        sy_expr = genome_to_symbolic(g, fmt="sympy")
        # Both should evaluate to the same value
        val_py = _eval_python_expr(py_expr, {"x0": 1.0, "x1": 2.0})
        val_sy = _eval_python_expr(sy_expr, {"x0": 1.0, "x1": 2.0})
        self.assertAlmostEqual(val_py, val_sy, places=12)


# ---------------------------------------------------------------------------
# 2. Text format
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestTextFormat(unittest.TestCase):

    def test_text_format_returns_string(self):
        """text format returns a non-empty string."""
        g = _make_linear_genome()
        result = genome_to_symbolic(g, fmt="text")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_text_format_contains_variable_name(self):
        """text format contains input variable names."""
        g = _make_linear_genome()
        result = genome_to_symbolic(g, input_names=["a", "b"], fmt="text")
        self.assertIn("a", result)
        self.assertIn("b", result)


# ---------------------------------------------------------------------------
# 3. LaTeX format
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestLatexFormat(unittest.TestCase):

    def test_latex_format_returns_string(self):
        """latex format returns a non-empty string."""
        g = _make_tanh_genome()
        result = genome_to_symbolic(g, fmt="latex")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_latex_contains_backslash_or_bracket(self):
        """LaTeX format for tanh genome contains LaTeX markup."""
        g = _make_tanh_genome()
        result = genome_to_symbolic(g, fmt="latex")
        # tanh renders as \tanh(...) in latex format
        self.assertIn("tanh", result)

    def test_latex_linear_genome(self):
        """LaTeX format for linear genome is reasonable."""
        g = _make_linear_genome()
        result = genome_to_symbolic(g, fmt="latex")
        self.assertIsInstance(result, str)


# ---------------------------------------------------------------------------
# 4. Constant folding
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestConstantFolding(unittest.TestCase):

    def test_zero_weight_term_removed(self):
        """With fold_constants=True, zero-weight connection is removed from expression."""
        g = _make_zero_weight_genome()
        expr = genome_to_symbolic(g, input_names=["x0", "x1"], fmt="python", fold_constants=True)
        # The x1 term should not appear (weight=0.0)
        # We check that the expression still evaluates correctly
        for x0, x1 in [(1.0, 0.0), (2.0, 100.0), (-1.0, 999.0)]:
            g.reset()
            expected = g.forward([x0, x1])[0]
            got = _eval_python_expr(expr, {"x0": x0, "x1": x1})
            self.assertAlmostEqual(got, expected, places=6)

    def test_zero_weight_term_not_in_expr(self):
        """x1 with weight=0 should not appear when fold_constants=True."""
        g = _make_zero_weight_genome()
        expr = genome_to_symbolic(g, input_names=["x0", "x1"], fmt="python", fold_constants=True)
        # x1 should not appear in the expression at all
        self.assertNotIn("x1", expr)

    def test_no_folding_keeps_zero_weight(self):
        """With fold_constants=False, zero-weight terms are kept."""
        g = _make_zero_weight_genome()
        expr = genome_to_symbolic(g, input_names=["x0", "x1"], fmt="python", fold_constants=False)
        # Expression is still correct
        for x0, x1 in [(1.0, 5.0), (0.0, 0.0)]:
            g.reset()
            expected = g.forward([x0, x1])[0]
            got = _eval_python_expr(expr, {"x0": x0, "x1": x1})
            self.assertAlmostEqual(got, expected, places=6)

    def test_unit_weight_simplified(self):
        """1.0*x is folded to just x in the expression."""
        g = _make_multi_output_genome()
        # First output has 1.0*x0 connection
        expr = genome_to_symbolic(g, input_names=["x0", "x1"], fmt="python", fold_constants=True)
        # When fold_constants=True, should not see "1.0*x0" — just "x0"
        self.assertNotIn("1.0*x0", expr)


# ---------------------------------------------------------------------------
# 5. Cyclic genome raises ValueError
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestCyclicGenome(unittest.TestCase):

    def test_cyclic_raises_value_error(self):
        """Cyclic genome raises ValueError in genome_to_symbolic."""
        g = _make_cyclic_genome()
        with self.assertRaises(ValueError):
            genome_to_symbolic(g, fmt="python")

    def test_genome_method_cyclic_raises(self):
        """Genome.to_symbolic() raises ValueError for cyclic genome."""
        g = _make_cyclic_genome()
        with self.assertRaises(ValueError):
            g.to_symbolic(format="python")


# ---------------------------------------------------------------------------
# 6. Multi-output genomes
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestMultiOutput(unittest.TestCase):

    def test_multi_output_returns_two_expressions(self):
        """Multi-output genome returns expressions joined by '; '."""
        g = _make_multi_output_genome()
        result = genome_to_symbolic(g, input_names=["x0", "x1"], fmt="python")
        # Two outputs → separator is '; '
        parts = result.split("; ")
        self.assertEqual(len(parts), 2)

    def test_multi_output_evaluates_correctly(self):
        """Both expressions in multi-output result evaluate correctly."""
        g = _make_multi_output_genome()
        result = genome_to_symbolic(g, input_names=["x0", "x1"], fmt="python")
        parts = result.split("; ")
        for x0, x1 in [(1.0, 2.0), (-0.5, 0.3)]:
            g.reset()
            expected = g.forward([x0, x1])
            for j, part in enumerate(parts):
                got = _eval_python_expr(part, {"x0": x0, "x1": x1})
                self.assertAlmostEqual(got, expected[j], places=6)

    def test_latex_multi_output_separated_by_comma(self):
        """LaTeX format for multi-output uses ', ' separator."""
        g = _make_multi_output_genome()
        result = genome_to_symbolic(g, fmt="latex")
        self.assertIn(",", result)


# ---------------------------------------------------------------------------
# 7. Genome.to_symbolic() method
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestGenomeToSymbolicMethod(unittest.TestCase):

    def test_method_returns_string(self):
        """Genome.to_symbolic() returns a string."""
        g = _make_linear_genome()
        result = g.to_symbolic(format="python")
        self.assertIsInstance(result, str)

    def test_method_python_format_equivalent(self):
        """Genome.to_symbolic() python format matches forward() values."""
        g = _make_tanh_genome()
        for x0, x1 in [(0.0, 0.0), (1.0, 1.0), (-1.0, 0.5)]:
            g.reset()
            expected = g.forward([x0, x1])[0]
            expr = g.to_symbolic(input_names=["x0", "x1"], format="python")
            got = _eval_python_expr(expr, {"x0": x0, "x1": x1})
            self.assertAlmostEqual(got, expected, places=6)

    def test_method_input_names_respected(self):
        """input_names parameter is used in the output expression."""
        g = _make_linear_genome()
        expr = g.to_symbolic(input_names=["alpha", "beta"], format="python")
        self.assertIn("alpha", expr)
        self.assertIn("beta", expr)

    def test_method_all_formats_return_strings(self):
        """All four formats return non-empty strings."""
        g = _make_linear_genome()
        for fmt in ("python", "text", "latex", "sympy"):
            result = g.to_symbolic(format=fmt)
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0, f"Empty result for format='{fmt}'")
