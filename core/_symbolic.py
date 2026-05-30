"""Symbolic regression export for YANE genomes.

Converts an acyclic genome into a closed-form symbolic expression in one of
four formats: ``"python"``, ``"text"``, ``"latex"``, or ``"sympy"``.

The python/sympy formats produce eval()-compatible strings; the latex format
produces a LaTeX math expression.  All formats evaluate numerically to the
same values as ``genome.forward()`` (within floating-point precision).

Cyclic genomes (containing recurrent / memory connections) cannot be
represented in closed form and raise :exc:`ValueError`.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome


# ---------------------------------------------------------------------------
# Activation → symbolic expression builders
# ---------------------------------------------------------------------------

def _act_python(act_name: str, inner: str) -> str:
    """Return a Python-evaluable symbolic expression for *act_name*(*inner*)."""
    m = {
        "linear":     inner,
        "sigmoid":    f"(1.0/(1.0+math.exp(-({inner}))))",
        "tanh":       f"math.tanh({inner})",
        "relu":       f"max(0.0,{inner})",
        "leaky_relu": f"({inner} if {inner}>0.0 else 0.01*({inner}))",
        "elu":        f"({inner} if {inner}>=0.0 else math.expm1({inner}))",
        "swish":      f"(({inner})/(1.0+math.exp(-({inner}))))",
        "softplus":   f"math.log1p(math.exp({inner}))",
        "sine":       f"math.sin({inner})",
        "cosine":     f"math.cos({inner})",
        "abs":        f"abs({inner})",
        "gaussian":   f"math.exp(-(({inner})**2))",
        "binary":     f"(1.0 if ({inner})>=0.5 else 0.0)",
        "square":     f"(({inner})**2)",
        "cube":       f"(({inner})**3)",
    }
    return m.get(act_name, inner)


def _act_latex(act_name: str, inner: str) -> str:
    """Return a LaTeX symbolic expression for *act_name*(*inner*)."""
    m = {
        "linear":     inner,
        "sigmoid":    rf"\sigma({inner})",
        "tanh":       rf"\tanh({inner})",
        "relu":       rf"\text{{ReLU}}({inner})",
        "leaky_relu": rf"\text{{LeakyReLU}}({inner})",
        "elu":        rf"\text{{ELU}}({inner})",
        "swish":      rf"\text{{Swish}}({inner})",
        "softplus":   rf"\ln(1+e^{{{inner}}})",
        "sine":       rf"\sin({inner})",
        "cosine":     rf"\cos({inner})",
        "abs":        rf"|{inner}|",
        "gaussian":   rf"e^{{-({inner})^2}}",
        "binary":     rf"\mathbf{{1}}[{inner}\geq 0.5]",
        "square":     rf"({inner})^2",
        "cube":       rf"({inner})^3",
    }
    return m.get(act_name, inner)


# ---------------------------------------------------------------------------
# Constant folding helpers
# ---------------------------------------------------------------------------

_ZERO_THRESHOLD = 1e-12


def _fmt_float(v: float, fmt: str) -> str:
    """Format a float for use in an expression string."""
    if fmt == "latex":
        # Use concise notation; avoid unnecessary trailing zeros
        if v == int(v) and abs(v) < 1e9:
            return str(int(v))
        return f"{v:.6g}"
    # python / text / sympy
    return repr(v)


def _weight_term(weight: float, src_expr: str, fmt: str, fold: bool) -> str | None:
    """Return ``weight * src_expr`` as a string, or None if folded to zero."""
    if fold and abs(weight) < _ZERO_THRESHOLD:
        return None  # zero weight — drop term
    if fold and abs(weight - 1.0) < _ZERO_THRESHOLD:
        return src_expr  # 1.0 * x → x
    if fold and abs(weight + 1.0) < _ZERO_THRESHOLD:
        return f"-({src_expr})"  # -1.0 * x → -(x)
    w_str = _fmt_float(weight, fmt)
    if fmt == "latex":
        return rf"{w_str} \cdot {src_expr}"
    return f"{w_str}*{src_expr}"


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def genome_to_symbolic(
    genome: "Genome",
    input_names: "list[str] | None" = None,
    fmt: str = "python",
    fold_constants: bool = True,
) -> str:
    """Convert an acyclic genome to a symbolic expression string.

    Parameters
    ----------
    genome:
        Trained acyclic genome.
    input_names:
        Names for each input variable.  Defaults to ``["x0", "x1", ...]``.
    fmt:
        Output format: ``"python"``, ``"text"``, ``"latex"``, or ``"sympy"``.
    fold_constants:
        Remove zero-weight terms and simplify 1.0*x → x.

    Returns
    -------
    str
        Symbolic expression.  For multiple outputs, expressions are
        joined by ``"; "`` (python/text/sympy) or ``", "`` (latex).

    Raises
    ------
    ValueError
        If the genome is cyclic.
    """
    # Build exec order (topological sort)
    if genome._exec_order is not None:
        exec_order = genome._exec_order
    else:
        exec_order = genome._build_exec_order()

    if exec_order is None:
        raise ValueError(
            "genome.to_symbolic() requires an acyclic genome.  "
            "Cyclic genomes cannot be represented as a closed-form expression."
        )

    # Determine format (sympy uses Python syntax)
    is_latex = fmt == "latex"
    is_python_like = fmt in ("python", "sympy", "text")

    n_inputs = len(genome.input_nodes)
    if input_names is None:
        input_names = [f"x{i}" for i in range(n_inputs)]
    elif len(input_names) < n_inputs:
        # Pad missing names
        input_names = list(input_names) + [
            f"x{i}" for i in range(len(input_names), n_inputs)
        ]

    # Map each node to its symbolic expression string
    node_id = {id(nd): i for i, nd in enumerate(genome.nodes)}
    expr: dict[int, str] = {}  # node index → symbolic expression

    # Assign input node expressions
    for nd in genome.input_nodes:
        ni = node_id[id(nd)]
        idx = getattr(nd, "input_index", ni)
        var = input_names[min(idx, len(input_names) - 1)]
        scale = getattr(nd, "input_scale", 1.0)
        if fold_constants and abs(scale - 1.0) < _ZERO_THRESHOLD:
            expr[ni] = var
        else:
            s_str = _fmt_float(scale, fmt)
            if is_latex:
                expr[ni] = rf"{s_str} \cdot {var}"
            else:
                expr[ni] = f"{s_str}*{var}"

    # Process non-input nodes in topological order
    for nd in exec_order:
        ni = node_id[id(nd)]
        # Collect weighted incoming connections
        terms: list[str] = []

        # Bias term
        bias = nd.bias
        if not (fold_constants and abs(bias) < _ZERO_THRESHOLD):
            terms.append(_fmt_float(bias, fmt))

        # Connection terms
        for src_nd in genome.nodes:
            si = node_id[id(src_nd)]
            for conn in src_nd.connections:
                if not conn.enabled:
                    continue
                tgt_i = node_id.get(id(conn.target))
                if tgt_i != ni:
                    continue
                src_expr_raw = expr.get(si)
                if src_expr_raw is None:
                    continue
                w = conn._weight
                term = _weight_term(w, src_expr_raw, fmt, fold_constants)
                if term is not None:
                    terms.append(term)

        # Build the pre-activation sum
        if not terms:
            inner = _fmt_float(0.0, fmt)
        elif len(terms) == 1:
            inner = terms[0]
        else:
            if is_latex:
                inner = " + ".join(terms)
                inner = f"({inner})"
            else:
                inner = "(" + " + ".join(terms) + ")"

        # Apply activation
        act_name = nd.activation.value
        if act_name == "linear":
            expr[ni] = inner
        elif is_latex:
            expr[ni] = _act_latex(act_name, inner)
        else:
            expr[ni] = _act_python(act_name, inner)

    # Collect output expressions
    output_exprs: list[str] = []
    for nd in genome.output_nodes:
        ni = node_id[id(nd)]
        e = expr.get(ni, "0.0")
        # Apply output_scale if set
        out_scale = getattr(nd, "output_scale", 1.0)
        if not (fold_constants and abs(out_scale - 1.0) < _ZERO_THRESHOLD):
            s_str = _fmt_float(out_scale, fmt)
            if is_latex:
                e = rf"{s_str} \cdot {e}"
            else:
                e = f"{s_str}*{e}"
        output_exprs.append(e)

    if len(output_exprs) == 1:
        return output_exprs[0]
    if is_latex:
        return ", ".join(output_exprs)
    return "; ".join(output_exprs)
