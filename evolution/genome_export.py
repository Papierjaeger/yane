"""Export trained YANE genomes to standalone Python functions or weight matrices.

Functions here are standalone utilities; they are also exposed as methods on
Genome via monkey-patching in neuro_evolution.py at import time.

Usage::

    fn_src = genome_to_python(genome)
    exec(compile(fn_src, '<genome>', 'exec'), ns := {})
    result = ns['forward'](inputs)

    weights = genome_to_numpy_weights(genome)
    # weights['W'][i][j] = weight from node i to node j; 0.0 if no connection
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome


# ---------------------------------------------------------------------------
# Activation source code — only math stdlib used
# ---------------------------------------------------------------------------

_ACTIVATION_SRC = {
    "linear":     "v",
    "sigmoid":    "(1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, v)))))",
    "tanh":       "math.tanh(v)",
    "relu":       "(v if v > 0.0 else 0.0)",
    "binary":     "(1.0 if v >= 0.5 else 0.0)",
    "leaky_relu": "(v if v > 0.0 else 0.01 * v)",
    "elu":        "(v if v >= 0.0 else (math.exp(max(-500.0, v)) - 1.0))",
    "swish":      "(_sv := max(-500.0, min(500.0, v)); _sv / (1.0 + math.exp(-_sv)))",
    "softplus":   "(v if v > 20.0 else math.log(1.0 + math.exp(max(-500.0, v))))",
    "sine":       "math.sin(v)",
    "cosine":     "math.cos(v)",
    "square":     "((min(v, 1e4) ** 2) if v >= 0 else (max(v, -1e4) ** 2))",
    "abs":        "(v if v >= 0.0 else -v)",
    "gaussian":   "(math.exp(-(min(max(v, -26.0), 26.0) ** 2)))",
    "cube":       "(min(max(v, -1e3), 1e3) ** 3)",
}

# swish uses a walrus — only available on Python 3.8+. Use a helper function instead.
_SWISH_HELPER = """\
def _swish(v):
    _sv = max(-500.0, min(500.0, v))
    return _sv / (1.0 + math.exp(-_sv))
"""


def _activation_expr(act_name: str, arg: str) -> str:
    """Return a Python expression computing activation *act_name* of *arg*."""
    if act_name == "swish":
        return f"_swish({arg})"
    raw = _ACTIVATION_SRC.get(act_name, arg)
    return f"(lambda v: {raw})({arg})"


def genome_to_python(genome: "Genome") -> str:
    """Generate a standalone Python function source for *genome*.

    The resulting source string defines a module-level ``forward(inputs)``
    function that uses only ``import math`` — no YANE dependency.

    Limitations:
    - Stateful (memory/recurrent) genomes are not supported; only the
      acyclic, non-memory path is exported. Memory nodes are treated as
      having value 0 at the start of each call.
    - Only acyclic genomes (``_has_cycles=False``) are supported; cyclic
      genomes raise ValueError.
    """
    if getattr(genome, "_has_cycles", False):
        raise ValueError(
            "genome_to_python() only supports acyclic genomes. "
            "This genome has cycles (recurrent connections)."
        )

    # Ensure topological order is computed (lazily built on first forward()).
    if genome._exec_order is None:
        exec_order = genome._build_exec_order()
        if exec_order is None:
            raise ValueError(
                "genome_to_python() requires a non-cyclic topology; "
                "no valid execution order found."
            )
        genome._exec_order = exec_order

    lines: list[str] = [
        "import math",
        "",
        _SWISH_HELPER,
        "",
        "def forward(inputs):",
        "    v = {}  # node index -> value",
    ]

    all_nodes = genome.nodes
    node_id = {id(n): i for i, n in enumerate(all_nodes)}

    # Build incoming-connection map: node_index → [(src_index, weight)]
    # Connections are stored on source nodes as outgoing; we invert here.
    incoming: dict[int, list[tuple[int, float]]] = {i: [] for i in range(len(all_nodes))}
    for src_node in all_nodes:
        src_ni = node_id[id(src_node)]
        for conn in src_node.connections:
            if not conn.enabled:
                continue
            tgt_ni = node_id.get(id(conn.target))
            if tgt_ni is not None:
                incoming[tgt_ni].append((src_ni, conn.weight))

    # Input nodes — just pass the input value through (no activation on inputs)
    for node in genome.input_nodes:
        ni = node_id[id(node)]
        idx = node.input_index
        lines.append(f"    v[{ni}] = inputs[{idx}] if {idx} < len(inputs) else 0.0")

    # Hidden + output nodes in topological order
    for node in genome._exec_order:
        ni = node_id[id(node)]
        act_name = node.activation.value
        bias = node.bias

        in_conns = incoming[ni]

        # Build the weighted-sum + bias expression
        parts = [f"{w!r} * v.get({src_ni}, 0.0)" for src_ni, w in in_conns]
        if bias != 0.0:
            sum_expr = " + ".join(parts) + f" + {bias!r}" if parts else repr(bias)
        else:
            sum_expr = " + ".join(parts) if parts else "0.0"

        # Always apply the activation function (e.g. sigmoid(0) = 0.5, not 0)
        act_expr = _activation_expr(act_name, sum_expr)
        lines.append(f"    v[{ni}] = {act_expr}")

    # Return output values
    out_indices = [node_id[id(n)] for n in genome.output_nodes]
    out_list = ", ".join(f"v.get({i}, 0.0)" for i in out_indices)
    lines.append(f"    return [{out_list}]")
    lines.append("")

    return "\n".join(lines)


def genome_to_numpy_weights(genome: "Genome") -> dict:
    """Extract the weight matrix and bias vector from *genome*.

    Returns a dict with:
        "W":      list[list[float]]  — N×N weight matrix (row=source, col=target)
        "b":      list[float]        — N bias values
        "labels": list[str]          — node type+index labels ("I0", "H0", "O0")
        "n_inputs":  int
        "n_outputs": int
    """
    nodes = genome.nodes
    n = len(nodes)
    node_id = {id(nd): i for i, nd in enumerate(nodes)}

    # Build weight matrix (source row, target col)
    W = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    labels: list[str] = []

    in_count = out_count = hid_count = 0
    for nd in nodes:
        t = nd.type.value
        if t == "input":
            labels.append(f"I{in_count}")
            in_count += 1
        elif t == "output":
            labels.append(f"O{out_count}")
            out_count += 1
        else:
            labels.append(f"H{hid_count}")
            hid_count += 1
        b[node_id[id(nd)]] = nd.bias

    for src_idx, nd in enumerate(nodes):
        for conn in nd.connections:
            if not conn.enabled:
                continue
            tgt_idx = node_id.get(id(conn.target))
            if tgt_idx is not None:
                W[src_idx][tgt_idx] += conn.weight

    return {
        "W": W,
        "b": b,
        "labels": labels,
        "n_inputs": len(genome.input_nodes),
        "n_outputs": len(genome.output_nodes),
    }
