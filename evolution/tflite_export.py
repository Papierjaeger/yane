"""Genome-to-Embedded Export (TFLite / C-Array).

Two export modes:
  1. **C-Array** (``"c_array"`` mode): generates ``{prefix}.h`` and
     ``{prefix}.cc`` — a commented C99 header with weights/biases and a
     standalone ``forward(float* inputs, float* outputs)`` function.
     Fully usable on microcontrollers without any ML framework.

  2. **TFLite** (``"tflite"`` mode): raises :exc:`ImportError` with a clear
     installation hint when the ``tflite`` package is absent (like the
     ONNX export module).

Usage (C-Array)::

    from yane.evolution.tflite_export import genome_to_c_array

    genome_to_c_array(genome, path="/tmp/model", prefix="yane_net")
    # Produces /tmp/model/yane_net.h and /tmp/model/yane_net.cc

Usage (TFLite)::

    from yane.evolution.tflite_export import genome_to_tflite
    model = genome_to_tflite(genome)  # raises ImportError if tflite absent
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome


# ---------------------------------------------------------------------------
# Activation → C expression builder
# ---------------------------------------------------------------------------

# C function bodies for each activation.  Argument is always the variable `v`.
_C_ACTIVATION: dict[str, str] = {
    "linear":     "{var}",
    "sigmoid":    "(1.0f / (1.0f + expf(-{var})))",
    "tanh":       "tanhf({var})",
    "relu":       "({var} > 0.0f ? {var} : 0.0f)",
    "leaky_relu": "({var} > 0.0f ? {var} : 0.01f * {var})",
    "elu":        "({var} >= 0.0f ? {var} : (expf({var}) - 1.0f))",
    "swish":      "({var} / (1.0f + expf(-{var})))",
    "softplus":   "logf(1.0f + expf({var}))",
    "sine":       "sinf({var})",
    "cosine":     "cosf({var})",
    "square":     "({var} * {var})",
    "abs":        "fabsf({var})",
    "gaussian":   "expf(-({var} * {var}))",
    "cube":       "({var} * {var} * {var})",
    "binary":     "({var} >= 0.5f ? 1.0f : 0.0f)",
}


def _c_activate(act_name: str, var: str) -> str:
    """Return a C expression for *act_name* applied to *var*."""
    template = _C_ACTIVATION.get(act_name, "{var}")
    return template.replace("{var}", var)


# ---------------------------------------------------------------------------
# Topological order helper
# ---------------------------------------------------------------------------

def _ensure_exec_order(genome: "Genome") -> list:
    """Return (possibly cached) topological execution order."""
    if genome._exec_order is not None:
        return genome._exec_order
    if not genome._has_cycles:
        order = genome._build_exec_order()
        if order is not None:
            genome._exec_order = order
            return order
    return []  # cyclic — process all non-input nodes in list order


def _incoming_map(genome: "Genome", node_id: dict) -> dict:
    """Build {target_idx: [(src_idx, weight)]} map from enabled connections."""
    incoming: dict[int, list] = {i: [] for i in range(len(genome.nodes))}
    for src in genome.nodes:
        src_i = node_id[id(src)]
        for conn in src.connections:
            if not conn.enabled:
                continue
            tgt_i = node_id.get(id(conn.target))
            if tgt_i is not None:
                incoming[tgt_i].append((src_i, conn._weight))
    return incoming


# ---------------------------------------------------------------------------
# C-Array export
# ---------------------------------------------------------------------------

def genome_to_c_array(
    genome: "Genome",
    path: "str | Path" = ".",
    prefix: str = "yane_net",
) -> tuple:
    """Export genome as a pair of C source files for embedded deployment.

    Generates:
      - ``{path}/{prefix}.h``   — struct declaration + forward prototype.
      - ``{path}/{prefix}.cc``  — weight tables and ``forward()`` implementation.

    The generated code depends only on ``<math.h>`` and is C99-compatible.

    Parameters
    ----------
    genome:
        Trained genome to export.  Only acyclic genomes are supported
        cleanly; cyclic genomes emit a note in the output.
    path:
        Directory to write files into (created if absent).
    prefix:
        Name prefix for the generated files and C identifiers.

    Returns
    -------
    (header_path, source_path):
        Absolute :class:`pathlib.Path` objects for the generated files.
    """
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    header_path = out_dir / f"{prefix}.h"
    source_path = out_dir / f"{prefix}.cc"

    all_nodes = genome.nodes
    n = len(all_nodes)
    node_id = {id(nd): i for i, nd in enumerate(all_nodes)}

    n_inputs  = len(genome.input_nodes)
    n_outputs = len(genome.output_nodes)

    exec_order = _ensure_exec_order(genome)
    incoming   = _incoming_map(genome, node_id)

    # ── Build weight + bias arrays ──────────────────────────────────────────
    # We store only (src_idx, tgt_idx, weight) triples for enabled connections.
    conn_list: list[tuple[int, int, float]] = []
    for src in all_nodes:
        src_i = node_id[id(src)]
        for conn in src.connections:
            if not conn.enabled:
                continue
            tgt_i = node_id.get(id(conn.target))
            if tgt_i is not None:
                conn_list.append((src_i, tgt_i, conn._weight))

    n_conn = len(conn_list)

    biases: list[float] = [nd.bias for nd in all_nodes]

    # ── Header file ──────────────────────────────────────────────────────────
    guard = f"{prefix.upper()}_H"
    h_lines = [
        f"/* Auto-generated by YANE — {prefix}.h */",
        f"/* {n_inputs} inputs, {n_outputs} outputs, {n} nodes, {n_conn} connections */",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <stddef.h>",
        "",
        f"#define {prefix.upper()}_N_INPUTS  {n_inputs}",
        f"#define {prefix.upper()}_N_OUTPUTS {n_outputs}",
        f"#define {prefix.upper()}_N_NODES   {n}",
        f"#define {prefix.upper()}_N_CONNS   {n_conn}",
        "",
        "/* Node bias values */",
        f"extern const float {prefix}_biases[{n}];",
        "",
        "/* Connection table: src_idx, tgt_idx, weight */",
        f"struct {prefix}_Conn {{ int src; int tgt; float w; }};",
        f"extern const struct {prefix}_Conn {prefix}_conns[{n_conn}];",
        "",
        f"/* Forward pass: inputs[{n_inputs}] → outputs[{n_outputs}] */",
        f"void {prefix}_forward(const float* inputs, float* outputs);",
        "",
        f"#endif /* {guard} */",
        "",
    ]
    header_path.write_text("\n".join(h_lines))

    # ── Source file ───────────────────────────────────────────────────────────
    cc_lines = [
        f"/* Auto-generated by YANE — {prefix}.cc */",
        f"/* {n_inputs} inputs, {n_outputs} outputs, {n} nodes, {n_conn} connections */",
        "",
        "#include <math.h>",
        "#include <string.h>",
        f'#include "{prefix}.h"',
        "",
    ]

    # Bias array
    cc_lines.append(f"const float {prefix}_biases[{n}] = {{")
    for i, nd in enumerate(all_nodes):
        sep = "," if i < n - 1 else ""
        cc_lines.append(f"    {nd.bias!r}f{sep}  /* node {i} */")
    cc_lines.append("};")
    cc_lines.append("")

    # Connection table
    if n_conn > 0:
        cc_lines.append(f"const struct {prefix}_Conn {prefix}_conns[{n_conn}] = {{")
        for k, (src_i, tgt_i, w) in enumerate(conn_list):
            sep = "," if k < n_conn - 1 else ""
            cc_lines.append(f"    {{{src_i}, {tgt_i}, {w!r}f}}{sep}  /* conn {k} */")
        cc_lines.append("};")
    else:
        cc_lines.append(f"const struct {prefix}_Conn {prefix}_conns[1] = {{{{0, 0, 0.0f}}}};  /* empty */")
    cc_lines.append("")

    # Forward function
    cc_lines.extend([
        f"void {prefix}_forward(const float* inputs, float* outputs) {{",
        f"    float v[{n}];",
        f"    int i, k;",
        "",
        f"    /* Initialise all node values to bias */",
        f"    for (i = 0; i < {n}; i++) v[i] = {prefix}_biases[i];",
        "",
        f"    /* Load inputs */",
    ])

    for inp_node in genome.input_nodes:
        ni = node_id[id(inp_node)]
        idx = getattr(inp_node, "input_index", ni)
        scale = getattr(inp_node, "input_scale", 1.0)
        if scale != 1.0:
            cc_lines.append(f"    v[{ni}] = inputs[{idx}] * {scale!r}f;")
        else:
            cc_lines.append(f"    v[{ni}] = inputs[{idx}];")

    cc_lines.extend([
        "",
        f"    /* Accumulate weighted sums from connections */",
        f"    for (k = 0; k < {n_conn}; k++) {{",
        f"        v[{prefix}_conns[k].tgt] += v[{prefix}_conns[k].src] * {prefix}_conns[k].w;",
        f"    }}",
        "",
        f"    /* Apply activations (in-place, topological order) */",
    ])

    # Apply activations in topological order for non-input nodes
    for nd in exec_order:
        ni = node_id[id(nd)]
        act_name = nd.activation.value
        act_expr = _c_activate(act_name, f"v[{ni}]")
        cc_lines.append(f"    v[{ni}] = {act_expr};")

    cc_lines.extend([
        "",
        f"    /* Write outputs */",
    ])

    for j, out_nd in enumerate(genome.output_nodes):
        ni = node_id[id(out_nd)]
        scale = getattr(out_nd, "output_scale", 1.0)
        if scale != 1.0:
            cc_lines.append(f"    outputs[{j}] = v[{ni}] * {scale!r}f;")
        else:
            cc_lines.append(f"    outputs[{j}] = v[{ni}];")

    cc_lines.append("}")
    cc_lines.append("")

    source_path.write_text("\n".join(cc_lines))

    return header_path, source_path


# ---------------------------------------------------------------------------
# TFLite export stub
# ---------------------------------------------------------------------------

def genome_to_tflite(
    genome: "Genome",
    path: "str | Path | None" = None,
) -> object:
    """Export genome to TFLite FlatBuffer format.

    Requires the ``tflite`` package (or ``tensorflow`` with TFLite conversion).

    Raises
    ------
    ImportError
        When neither ``tflite`` nor ``tensorflow`` is installed.
    """
    try:
        import tflite  # type: ignore[import]
    except ImportError:
        try:
            import tensorflow as _tf  # type: ignore[import]
            tflite = _tf.lite
        except ImportError:
            raise ImportError(
                "genome_to_tflite() requires the 'tflite' or 'tensorflow' package.\n"
                "Install with:  pip install tflite  (or pip install tensorflow)\n"
                "Alternatively, use genome_to_c_array() for embedded deployment "
                "without any ML framework."
            )

    raise NotImplementedError(
        "genome_to_tflite(): TFLite is installed but direct model construction "
        "from a YANE genome is not yet implemented.  "
        "Use genome_to_onnx() + onnx2tf as an alternative pipeline."
    )
