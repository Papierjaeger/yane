"""Genome-to-ONNX export for production deployment.

Converts a trained YANE :class:`~yane.core.genome.Genome` to an ONNX model
that can be run with any ONNX-compatible runtime (ONNX Runtime, TensorFlow,
CoreML, TFLite, etc.) without a YANE installation.

Requires the ``onnx`` package::

    pip install onnx

Usage::

    from yane.evolution.onnx_export import genome_to_onnx

    model = genome_to_onnx(genome, path="model.onnx")
    # Or without saving:
    model_proto = genome_to_onnx(genome)

Supported activations
---------------------
Activation functions are mapped to ONNX ops.  The following YANE activations
have direct or composed ONNX equivalents and are supported:

``linear``, ``sigmoid``, ``tanh``, ``relu``, ``leaky_relu``, ``elu``,
``swish``, ``softplus``, ``sine``, ``cosine``, ``abs``, ``gaussian``,
``binary``, ``square``, ``cube``.

Cyclic networks
---------------
ONNX requires an acyclic computation graph.  Cyclic (recurrent) genomes are
unrolled for ``unroll_steps`` iterations.  Each step feeds the previous
step's node values as the initial state for memory nodes.  Use
``unroll_steps >= 2`` to capture temporal dependencies.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome


# ---------------------------------------------------------------------------
# Activation → ONNX op builder
# ---------------------------------------------------------------------------

def _activation_nodes(act_name: str, input_name: str, output_name: str) -> list:
    """Return a list of ONNX NodeProto objects implementing *act_name*.

    Requires ``onnx`` to be importable.  All outputs are float32 scalars
    (or batched 1-D tensors, consistent with how the rest of the graph works).
    """
    try:
        from onnx import helper, TensorProto
        from onnx import numpy_helper as nph
        import numpy as np
    except ImportError:
        raise ImportError("genome_to_onnx() requires the 'onnx' package: pip install onnx")

    def _const(name: str, value: float) -> "onnx.NodeProto":
        arr = np.array([value], dtype=np.float32)
        return helper.make_node(
            "Constant", inputs=[], outputs=[name],
            value=nph.from_array(arr, name=name),
        )

    if act_name == "linear":
        return [helper.make_node("Identity", inputs=[input_name], outputs=[output_name])]

    if act_name == "sigmoid":
        return [helper.make_node("Sigmoid", inputs=[input_name], outputs=[output_name])]

    if act_name == "tanh":
        return [helper.make_node("Tanh", inputs=[input_name], outputs=[output_name])]

    if act_name == "relu":
        return [helper.make_node("Relu", inputs=[input_name], outputs=[output_name])]

    if act_name == "leaky_relu":
        return [helper.make_node("LeakyRelu", inputs=[input_name], outputs=[output_name],
                                  alpha=0.01)]

    if act_name == "elu":
        # ELU(x) = x if x >= 0 else exp(x) - 1
        return [helper.make_node("Elu", inputs=[input_name], outputs=[output_name], alpha=1.0)]

    if act_name == "swish":
        # swish(x) = x * sigmoid(x)
        sig_name = f"{output_name}_sig"
        return [
            helper.make_node("Sigmoid", inputs=[input_name], outputs=[sig_name]),
            helper.make_node("Mul", inputs=[input_name, sig_name], outputs=[output_name]),
        ]

    if act_name == "softplus":
        # softplus(x) = log(1 + exp(x))  — Softplus op available from opset 1
        return [helper.make_node("Softplus", inputs=[input_name], outputs=[output_name])]

    if act_name == "sine":
        return [helper.make_node("Sin", inputs=[input_name], outputs=[output_name])]

    if act_name == "cosine":
        return [helper.make_node("Cos", inputs=[input_name], outputs=[output_name])]

    if act_name == "abs":
        return [helper.make_node("Abs", inputs=[input_name], outputs=[output_name])]

    if act_name == "square":
        return [helper.make_node("Mul", inputs=[input_name, input_name],
                                  outputs=[output_name])]

    if act_name == "cube":
        sq_name = f"{output_name}_sq"
        return [
            helper.make_node("Mul", inputs=[input_name, input_name], outputs=[sq_name]),
            helper.make_node("Mul", inputs=[sq_name, input_name], outputs=[output_name]),
        ]

    if act_name == "gaussian":
        # gaussian(x) = exp(-(x^2))
        sq_name = f"{output_name}_sq"
        neg_name = f"{output_name}_neg"
        return [
            helper.make_node("Mul", inputs=[input_name, input_name], outputs=[sq_name]),
            helper.make_node("Neg", inputs=[sq_name], outputs=[neg_name]),
            helper.make_node("Exp", inputs=[neg_name], outputs=[output_name]),
        ]

    if act_name == "binary":
        # binary(x) = 1.0 if x >= 0.5 else 0.0
        thresh_c = f"{output_name}_thresh"
        cmp_name = f"{output_name}_cmp"
        one_c = f"{output_name}_one"
        zero_c = f"{output_name}_zero"
        return [
            _const(thresh_c, 0.5),
            helper.make_node("GreaterOrEqual", inputs=[input_name, thresh_c],
                             outputs=[cmp_name]),
            _const(one_c, 1.0),
            _const(zero_c, 0.0),
            helper.make_node("Where", inputs=[cmp_name, one_c, zero_c],
                             outputs=[output_name]),
        ]

    # Unknown activation: treat as linear
    return [helper.make_node("Identity", inputs=[input_name], outputs=[output_name])]


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def genome_to_onnx(
    genome: "Genome",
    path: "str | Path | None" = None,
    opset_version: int = 17,
    unroll_steps: int = 1,
) -> "onnx.ModelProto":
    """Export a YANE genome to an ONNX model.

    Parameters
    ----------
    genome :
        Trained genome to export.  Both acyclic and cyclic (recurrent) genomes
        are supported.
    path :
        Optional file path to save the model (``".onnx"`` extension).
        When *None*, the model is built and returned without saving.
    opset_version :
        ONNX opset version (default: 17).
    unroll_steps :
        Number of recurrent steps to unroll for cyclic genomes.  Ignored for
        acyclic genomes (where a single forward pass is used).

    Returns
    -------
    onnx.ModelProto
        The ONNX model.  Call ``onnx.checker.check_model(model)`` to validate.

    Raises
    ------
    ImportError
        When the ``onnx`` package is not installed.

    Examples
    --------
    ::

        model = genome_to_onnx(genome, path="xor.onnx", opset_version=17)
        # Run with ONNX Runtime:
        import onnxruntime as ort
        sess = ort.InferenceSession("xor.onnx")
        output = sess.run(None, {"inputs": [[0.0, 1.0]]})[0]
    """
    try:
        import onnx
        from onnx import helper, TensorProto
        from onnx import numpy_helper as nph
        import numpy as np
    except ImportError:
        raise ImportError(
            "genome_to_onnx() requires the 'onnx' package.\n"
            "Install it with:  pip install onnx"
        )

    is_cyclic = getattr(genome, "_has_cycles", False)

    if is_cyclic:
        return _build_cyclic_onnx(genome, opset_version, unroll_steps, onnx, helper, TensorProto, nph, np)
    else:
        return _build_acyclic_onnx(genome, opset_version, onnx, helper, TensorProto, nph, np, path)

    # Save
    if path is not None:
        onnx.save(model, str(path))
    return model


# ---------------------------------------------------------------------------
# Acyclic graph builder
# ---------------------------------------------------------------------------

def _build_acyclic_onnx(genome, opset_version, onnx, helper, TensorProto, nph, np, path):
    """Build ONNX graph for an acyclic genome (single forward pass)."""

    # Ensure topological order is available
    if genome._exec_order is None:
        exec_order = genome._build_exec_order()
        if exec_order is None:
            raise ValueError("genome_to_onnx(): cannot find valid execution order.")
        genome._exec_order = exec_order

    all_nodes = genome.nodes
    node_id = {id(n): i for i, n in enumerate(all_nodes)}

    # Build incoming connection map (target → [(src_id, weight)])
    incoming: dict[int, list[tuple[int, float]]] = {i: [] for i in range(len(all_nodes))}
    for src in all_nodes:
        src_ni = node_id[id(src)]
        for conn in src.connections:
            if not conn.enabled:
                continue
            tgt_ni = node_id.get(id(conn.target))
            if tgt_ni is not None:
                incoming[tgt_ni].append((src_ni, conn.weight))

    onnx_nodes: list = []
    initializers: list = []

    def _add_const(name: str, value: float) -> None:
        arr = np.array([value], dtype=np.float32)
        onnx_nodes.append(helper.make_node(
            "Constant", inputs=[], outputs=[name],
            value=nph.from_array(arr, name=name),
        ))

    def _node_val(ni: int) -> str:
        return f"x_{ni}"

    # --- Input nodes ---
    for inp_node in genome.input_nodes:
        ni = node_id[id(inp_node)]
        idx = getattr(inp_node, "input_index", ni)
        # Slice the i-th element from the flat input vector
        # input shape: [1, n_inputs] → slice at column idx
        idx_const = f"idx_{ni}"
        onnx_nodes.append(helper.make_node(
            "Constant", inputs=[], outputs=[idx_const],
            value=nph.from_array(np.array([0, idx], dtype=np.int64), name=idx_const),
        ))
        ends_const = f"ends_{ni}"
        onnx_nodes.append(helper.make_node(
            "Constant", inputs=[], outputs=[ends_const],
            value=nph.from_array(np.array([1, idx + 1], dtype=np.int64), name=ends_const),
        ))
        sliced = f"sliced_{ni}"
        onnx_nodes.append(helper.make_node(
            "Slice", inputs=["inputs", idx_const, ends_const],
            outputs=[sliced],
        ))
        flat = f"flat_{ni}"
        onnx_nodes.append(helper.make_node(
            "Flatten", inputs=[sliced], outputs=[flat],
            axis=0,
        ))
        onnx_nodes.append(helper.make_node("Identity", inputs=[flat], outputs=[_node_val(ni)]))

    # --- Non-input nodes in topological order ---
    for node in genome._exec_order:
        ni = node_id[id(node)]
        act_name = node.activation.value
        bias = float(node.bias)
        in_conns = incoming[ni]

        # Build weighted sum of inputs
        if in_conns:
            weighted: list[str] = []
            for k, (src_ni, weight) in enumerate(in_conns):
                w_const = f"w_{ni}_{k}"
                _add_const(w_const, float(weight))
                mul_out = f"mul_{ni}_{k}"
                onnx_nodes.append(helper.make_node(
                    "Mul", inputs=[_node_val(src_ni), w_const], outputs=[mul_out],
                ))
                weighted.append(mul_out)
            # Sum all weighted inputs
            if len(weighted) == 1:
                sum_name = f"sum_{ni}"
                onnx_nodes.append(helper.make_node("Identity", inputs=[weighted[0]], outputs=[sum_name]))
            else:
                # Chain Adds: ((a + b) + c) + d ...
                cur = weighted[0]
                for j, nxt in enumerate(weighted[1:]):
                    nxt_sum = f"sum_{ni}_{j}"
                    onnx_nodes.append(helper.make_node("Add", inputs=[cur, nxt], outputs=[nxt_sum]))
                    cur = nxt_sum
                sum_name = cur
        else:
            sum_name = f"zero_{ni}"
            _add_const(sum_name, 0.0)

        # Add bias
        bias_const = f"bias_{ni}"
        _add_const(bias_const, bias)
        prebias_name = f"prebias_{ni}"
        onnx_nodes.append(helper.make_node("Add", inputs=[sum_name, bias_const], outputs=[prebias_name]))

        # Apply activation
        onnx_nodes.extend(_activation_nodes(act_name, prebias_name, _node_val(ni)))

    # --- Gather output values ---
    out_ni_list = [node_id[id(n)] for n in genome.output_nodes]
    n_outputs = len(out_ni_list)
    if n_outputs == 1:
        final_output = _node_val(out_ni_list[0])
        # Reshape to [1, 1]
        shape_out = "shape_out"
        onnx_nodes.append(helper.make_node(
            "Constant", inputs=[], outputs=[shape_out],
            value=nph.from_array(np.array([1, 1], dtype=np.int64), name=shape_out),
        ))
        onnx_nodes.append(helper.make_node("Reshape", inputs=[final_output, shape_out],
                                            outputs=["outputs"]))
    else:
        out_names = [_node_val(ni) for ni in out_ni_list]
        # Unsqueeze each to [1, 1] then concat along axis 1
        unsqueezed = []
        shape_one = "shape_one"
        onnx_nodes.append(helper.make_node(
            "Constant", inputs=[], outputs=[shape_one],
            value=nph.from_array(np.array([1, 1], dtype=np.int64), name=shape_one),
        ))
        for k, name in enumerate(out_names):
            resh = f"resh_out_{k}"
            onnx_nodes.append(helper.make_node("Reshape", inputs=[name, shape_one], outputs=[resh]))
            unsqueezed.append(resh)
        onnx_nodes.append(helper.make_node("Concat", inputs=unsqueezed, outputs=["outputs"], axis=1))

    # --- Build graph ---
    n_inputs = len(genome.input_nodes)
    input_tensor = helper.make_tensor_value_info("inputs", TensorProto.FLOAT, [1, n_inputs])
    output_tensor = helper.make_tensor_value_info("outputs", TensorProto.FLOAT, [1, n_outputs])

    graph = helper.make_graph(onnx_nodes, "yane_genome", [input_tensor], [output_tensor],
                              initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset_version)])
    model.ir_version = 8

    if path is not None:
        onnx.save(model, str(path))
    return model


# ---------------------------------------------------------------------------
# Cyclic (recurrent) graph builder — time-unrolled
# ---------------------------------------------------------------------------

def _build_cyclic_onnx(genome, opset_version, unroll_steps, onnx, helper, TensorProto, nph, np, path=None):
    """Build ONNX graph for a cyclic genome via time unrolling."""

    all_nodes = genome.nodes
    node_id = {id(n): i for i, n in enumerate(all_nodes)}

    # Build incoming connection map
    incoming: dict[int, list[tuple[int, float]]] = {i: [] for i in range(len(all_nodes))}
    for src in all_nodes:
        src_ni = node_id[id(src)]
        for conn in src.connections:
            if not conn.enabled:
                continue
            tgt_ni = node_id.get(id(conn.target))
            if tgt_ni is not None:
                incoming[tgt_ni].append((src_ni, conn.weight))

    onnx_nodes: list = []

    def _add_const(name: str, value: float) -> None:
        onnx_nodes.append(helper.make_node(
            "Constant", inputs=[], outputs=[name],
            value=nph.from_array(np.array([value], dtype=np.float32), name=name),
        ))

    def _node_val(ni: int, step: int) -> str:
        return f"x_{ni}_s{step}"

    n_inputs = len(genome.input_nodes)
    n_outputs = len(genome.output_nodes)

    # --- Step 0 initialisation (all node values = 0) ---
    for ni in range(len(all_nodes)):
        _add_const(_node_val(ni, 0), 0.0)

    # Override input nodes at step 0 with actual inputs
    for inp_node in genome.input_nodes:
        ni = node_id[id(inp_node)]
        idx = getattr(inp_node, "input_index", ni)
        idx_c = f"idx_s0_{ni}"
        ends_c = f"ends_s0_{ni}"
        onnx_nodes.append(helper.make_node(
            "Constant", inputs=[], outputs=[idx_c],
            value=nph.from_array(np.array([0, idx], dtype=np.int64), name=idx_c),
        ))
        onnx_nodes.append(helper.make_node(
            "Constant", inputs=[], outputs=[ends_c],
            value=nph.from_array(np.array([1, idx + 1], dtype=np.int64), name=ends_c),
        ))
        sliced = f"sliced_s0_{ni}"
        onnx_nodes.append(helper.make_node(
            "Slice", inputs=["inputs", idx_c, ends_c], outputs=[sliced],
        ))
        flat = f"flat_s0_{ni}"
        onnx_nodes.append(helper.make_node("Flatten", inputs=[sliced], outputs=[flat], axis=0))
        onnx_nodes.append(helper.make_node(
            "Identity", inputs=[flat], outputs=[_node_val(ni, 0)],
        ))

    # --- Unroll steps ---
    for step in range(1, unroll_steps + 1):
        prev = step - 1
        # For each non-input node, compute value at this step
        non_input_ids = [node_id[id(n)] for n in all_nodes if n not in genome.input_nodes]
        for ni in non_input_ids:
            node = all_nodes[ni]
            act_name = node.activation.value
            bias = float(node.bias)
            in_conns = incoming[ni]

            if in_conns:
                weighted: list[str] = []
                for k, (src_ni, weight) in enumerate(in_conns):
                    w_c = f"w_{ni}_{k}_s{step}"
                    _add_const(w_c, float(weight))
                    mul_out = f"mul_{ni}_{k}_s{step}"
                    onnx_nodes.append(helper.make_node(
                        "Mul", inputs=[_node_val(src_ni, prev), w_c], outputs=[mul_out],
                    ))
                    weighted.append(mul_out)
                if len(weighted) == 1:
                    sum_name = f"sum_{ni}_s{step}"
                    onnx_nodes.append(helper.make_node("Identity", inputs=[weighted[0]], outputs=[sum_name]))
                else:
                    cur = weighted[0]
                    for j, nxt in enumerate(weighted[1:]):
                        nxt_sum = f"sum_{ni}_{j}_s{step}"
                        onnx_nodes.append(helper.make_node("Add", inputs=[cur, nxt], outputs=[nxt_sum]))
                        cur = nxt_sum
                    sum_name = cur
            else:
                sum_name = f"zero_{ni}_s{step}"
                _add_const(sum_name, 0.0)

            bias_c = f"bias_{ni}_s{step}"
            _add_const(bias_c, bias)
            prebias = f"prebias_{ni}_s{step}"
            onnx_nodes.append(helper.make_node("Add", inputs=[sum_name, bias_c], outputs=[prebias]))
            onnx_nodes.extend(_activation_nodes(act_name, prebias, _node_val(ni, step)))

        # Input nodes carry their input value at each step
        for inp_node in genome.input_nodes:
            ni = node_id[id(inp_node)]
            onnx_nodes.append(helper.make_node(
                "Identity", inputs=[_node_val(ni, 0)], outputs=[_node_val(ni, step)],
            ))

    # --- Gather outputs from final step ---
    final_step = unroll_steps
    out_ni_list = [node_id[id(n)] for n in genome.output_nodes]
    out_names = [_node_val(ni, final_step) for ni in out_ni_list]

    shape_one = "shape_one_cyc"
    onnx_nodes.append(helper.make_node(
        "Constant", inputs=[], outputs=[shape_one],
        value=nph.from_array(np.array([1, 1], dtype=np.int64), name=shape_one),
    ))
    unsqueezed = []
    for k, name in enumerate(out_names):
        resh = f"resh_cyc_{k}"
        onnx_nodes.append(helper.make_node("Reshape", inputs=[name, shape_one], outputs=[resh]))
        unsqueezed.append(resh)

    if len(unsqueezed) == 1:
        onnx_nodes.append(helper.make_node("Identity", inputs=[unsqueezed[0]], outputs=["outputs"]))
    else:
        onnx_nodes.append(helper.make_node("Concat", inputs=unsqueezed, outputs=["outputs"], axis=1))

    input_tensor = helper.make_tensor_value_info("inputs", TensorProto.FLOAT, [1, n_inputs])
    output_tensor = helper.make_tensor_value_info("outputs", TensorProto.FLOAT, [1, n_outputs])

    graph = helper.make_graph(onnx_nodes, "yane_genome_cyclic",
                              [input_tensor], [output_tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset_version)])
    model.ir_version = 8

    if path is not None:
        onnx.save(model, str(path))
    return model
