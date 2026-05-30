"""WebAssembly / Pure-JS Export für YANE-Genome.

Erzeugt eine standalone HTML/JS-Datei, die das trainierte Genome im Browser
ausführt — ohne Emscripten, ohne ONNX, ohne YANE-Abhängigkeit.

Implementierte Modi
-------------------
``"js"`` (Standard)
    Pure-JavaScript-Transpilation. Funktioniert in jedem Browser und
    Node.js ohne externe Abhängigkeiten.

``"wasm"``
    Erfordert Emscripten (``emcc``). Nicht verfügbar auf diesem System →
    ``ImportError`` mit klarer Installationsanweisung.

Verwendung::

    from yane.evolution.wasm_export import genome_to_js, genome_to_html

    js_src = genome_to_js(genome)
    genome_to_html(genome, path="xor.html", title="XOR Network")

Oder via NeuroEvolution::

    yane.export_genome_wasm("model.html")
    yane.export_genome_wasm("model.html", mode="js")

Unterstützte Aktivierungsfunktionen
-------------------------------------
linear, sigmoid, tanh, relu, leaky_relu, elu, swish, softplus, sine, cosine,
abs, gaussian, binary, square, cube.

Zyklische Netze
---------------
Zyklische Genome werden time-unrolled: ``unroll_steps`` Iterationen, wobei
Memory-Nodes vom Vorschritt in den nächsten übergehen. Identisch zum ONNX-Export.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome


# ---------------------------------------------------------------------------
# Activation → JavaScript expression
# ---------------------------------------------------------------------------

def _js_activation(act_name: str, expr: str) -> str:
    """Return a JavaScript expression applying *act_name* to *expr*."""
    act = act_name.lower()
    if act == "linear":
        return expr
    if act == "sigmoid":
        return f"(1.0 / (1.0 + Math.exp(-Math.max(-500, Math.min(500, {expr})))))"
    if act == "tanh":
        return f"Math.tanh({expr})"
    if act == "relu":
        return f"Math.max(0.0, {expr})"
    if act == "leaky_relu":
        return f"({expr} >= 0 ? {expr} : 0.01 * ({expr}))"
    if act == "elu":
        return f"({expr} >= 0 ? {expr} : Math.exp(Math.max(-500, {expr})) - 1.0)"
    if act == "swish":
        return f"(({expr}) / (1.0 + Math.exp(-Math.max(-500, Math.min(500, {expr})))))"
    if act == "softplus":
        return f"({expr} > 20 ? {expr} : Math.log(1.0 + Math.exp(Math.max(-500, {expr}))))"
    if act == "sine":
        return f"Math.sin({expr})"
    if act == "cosine":
        return f"Math.cos({expr})"
    if act == "abs":
        return f"Math.abs({expr})"
    if act == "gaussian":
        return f"Math.exp(-(Math.min(Math.abs({expr}), 26) ** 2))"
    if act == "binary":
        return f"({expr} >= 0.5 ? 1.0 : 0.0)"
    if act == "square":
        return f"(({expr}) * ({expr}))"
    if act == "cube":
        return f"(({expr}) * ({expr}) * ({expr}))"
    # Fallback: linear
    return expr


# ---------------------------------------------------------------------------
# Acyclic JS generation
# ---------------------------------------------------------------------------

def genome_to_js(
    genome: "Genome",
    function_name: str = "forward",
    unroll_steps: int = 1,
) -> str:
    """Generate a standalone JavaScript ``forward(inputs)`` function.

    Parameters
    ----------
    genome :
        Trained genome to export.  Both acyclic and cyclic genomes are
        supported.
    function_name :
        Name of the generated JS function (default: ``"forward"``).
    unroll_steps :
        Recurrent unroll depth for cyclic genomes (≥ 1).

    Returns
    -------
    str
        JavaScript source string defining the ``forward(inputs)`` function.

    Raises
    ------
    ValueError
        When an acyclic execution order cannot be determined.
    """
    is_cyclic = getattr(genome, "_has_cycles", False)
    if not is_cyclic and genome._exec_order is None:
        exec_order = genome._build_exec_order()
        if exec_order is None:
            is_cyclic = True
        else:
            genome._exec_order = exec_order

    if is_cyclic:
        return _genome_to_js_cyclic(genome, function_name, unroll_steps)
    else:
        return _genome_to_js_acyclic(genome, function_name)


def _genome_to_js_acyclic(genome: "Genome", fn_name: str) -> str:
    if genome._exec_order is None:
        exec_order = genome._build_exec_order()
        if exec_order is None:
            raise ValueError("genome_to_js(): no valid execution order found.")
        genome._exec_order = exec_order

    all_nodes = genome.nodes
    nid = {id(n): i for i, n in enumerate(all_nodes)}

    incoming: dict[int, list[tuple[int, float]]] = {i: [] for i in range(len(all_nodes))}
    for src in all_nodes:
        si = nid[id(src)]
        for conn in src.connections:
            if not conn.enabled:
                continue
            ti = nid.get(id(conn.target))
            if ti is not None:
                incoming[ti].append((si, conn.weight))

    lines = [f"function {fn_name}(inputs) {{", "  var v = {};"]

    for node in genome.input_nodes:
        ni = nid[id(node)]
        idx = getattr(node, "input_index", ni)
        lines.append(f"  v[{ni}] = (inputs[{idx}] !== undefined ? inputs[{idx}] : 0.0);")

    for node in genome._exec_order:
        ni = nid[id(node)]
        act_name = node.activation.value if hasattr(node.activation, "value") else str(node.activation)
        bias = float(node.bias)
        in_conns = incoming[ni]

        if in_conns:
            parts = [f"{w!r} * (v[{si}] || 0.0)" for si, w in in_conns]
            sum_expr = " + ".join(parts)
            if bias != 0.0:
                sum_expr += f" + {bias!r}"
        else:
            sum_expr = repr(bias) if bias != 0.0 else "0.0"

        act_expr = _js_activation(act_name, sum_expr)
        lines.append(f"  v[{ni}] = {act_expr};")

    out_indices = [nid[id(n)] for n in genome.output_nodes]
    out_list = ", ".join(f"(v[{i}] || 0.0)" for i in out_indices)
    lines.append(f"  return [{out_list}];")
    lines.append("}")
    return "\n".join(lines)


def _genome_to_js_cyclic(genome: "Genome", fn_name: str, unroll_steps: int) -> str:
    all_nodes = genome.nodes
    nid = {id(n): i for i, n in enumerate(all_nodes)}

    incoming: dict[int, list[tuple[int, float]]] = {i: [] for i in range(len(all_nodes))}
    for src in all_nodes:
        si = nid[id(src)]
        for conn in src.connections:
            if not conn.enabled:
                continue
            ti = nid.get(id(conn.target))
            if ti is not None:
                incoming[ti].append((si, conn.weight))

    def nv(ni: int, step: int) -> str:
        return f"v{ni}_s{step}"

    lines = [f"function {fn_name}(inputs) {{"]

    # Step 0: all nodes = 0, then set inputs
    for ni in range(len(all_nodes)):
        lines.append(f"  var {nv(ni, 0)} = 0.0;")
    for node in genome.input_nodes:
        ni = nid[id(node)]
        idx = getattr(node, "input_index", ni)
        lines.append(f"  {nv(ni, 0)} = (inputs[{idx}] !== undefined ? inputs[{idx}] : 0.0);")

    non_input_ids = [nid[id(n)] for n in all_nodes if n not in genome.input_nodes]

    for step in range(1, unroll_steps + 1):
        prev = step - 1
        for ni in non_input_ids:
            node = all_nodes[ni]
            act_name = node.activation.value if hasattr(node.activation, "value") else str(node.activation)
            bias = float(node.bias)
            in_conns = incoming[ni]
            if in_conns:
                parts = [f"{w!r} * {nv(si, prev)}" for si, w in in_conns]
                sum_expr = " + ".join(parts)
                if bias != 0.0:
                    sum_expr += f" + {bias!r}"
            else:
                sum_expr = repr(bias) if bias != 0.0 else "0.0"
            act_expr = _js_activation(act_name, sum_expr)
            lines.append(f"  var {nv(ni, step)} = {act_expr};")
        for node in genome.input_nodes:
            ni = nid[id(node)]
            lines.append(f"  var {nv(ni, step)} = {nv(ni, 0)};")

    final = unroll_steps
    out_list = ", ".join(f"{nv(nid[id(n)], final)}" for n in genome.output_nodes)
    lines.append(f"  return [{out_list}];")
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML wrapper
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    body {{ font-family: monospace; max-width: 700px; margin: 2em auto; padding: 1em; background:#1e1e2e; color:#cdd6f4; }}
    h1 {{ color:#cba6f7; }}
    label {{ display:block; margin:.4em 0 .1em; color:#a6e3a1; }}
    input[type=number] {{ width:120px; padding:.3em; background:#313244; color:#cdd6f4; border:1px solid #585b70; border-radius:4px; }}
    button {{ margin-top:1em; padding:.5em 1.5em; background:#cba6f7; color:#1e1e2e; border:none; border-radius:4px; cursor:pointer; font-size:1em; }}
    pre {{ background:#181825; padding:1em; border-radius:6px; overflow-x:auto; color:#89dceb; }}
    #result {{ margin-top:1em; font-size:1.2em; color:#f9e2af; }}
  </style>
</head>
<body>
<h1>{title}</h1>
<p>Inputs: {n_inputs} &nbsp; Outputs: {n_outputs}</p>
<form id="inputForm">
{input_fields}
  <button type="submit">Run Network</button>
</form>
<div id="result"></div>
<details>
  <summary style="cursor:pointer;color:#74c7ec">Show generated JS</summary>
  <pre id="jsSrc"></pre>
</details>
<script>
{js_src}

document.getElementById("jsSrc").textContent = {fn_name}.toString();

document.getElementById("inputForm").addEventListener("submit", function(e) {{
  e.preventDefault();
  var inputs = [{input_read}];
  var outputs = {fn_name}(inputs);
  document.getElementById("result").textContent = "Output: [" + outputs.map(function(v){{return v.toFixed(6);}}).join(", ") + "]";
}});
</script>
</body>
</html>
"""


def genome_to_html(
    genome: "Genome",
    path: "str | Path | None" = None,
    title: str = "YANE Network",
    function_name: str = "forward",
    unroll_steps: int = 1,
    mode: str = "js",
) -> str:
    """Generate a standalone HTML file with the genome's forward pass.

    Parameters
    ----------
    genome :
        Genome to export.
    path :
        File path to write the HTML (optional).
    title :
        HTML ``<title>`` and ``<h1>`` text.
    function_name :
        Name of the JS forward function.
    unroll_steps :
        Unroll depth for cyclic genomes.
    mode :
        ``"js"`` (default) for pure-JS; ``"wasm"`` raises ``ImportError``.

    Returns
    -------
    str
        The full HTML source string.

    Raises
    ------
    ImportError
        When ``mode="wasm"`` and Emscripten is not available.
    """
    if mode == "wasm":
        raise ImportError(
            "WebAssembly mode requires Emscripten (emcc).\n"
            "Install it from https://emscripten.org/ or use mode='js' instead."
        )

    js_src = genome_to_js(genome, function_name=function_name, unroll_steps=unroll_steps)
    n_inputs = len(genome.input_nodes)
    n_outputs = len(genome.output_nodes)

    input_fields = "\n".join(
        f'  <label>Input {i}: <input type="number" id="inp{i}" value="0" step="0.1"></label>'
        for i in range(n_inputs)
    )
    input_read = ", ".join(
        f'+document.getElementById("inp{i}").value' for i in range(n_inputs)
    )

    html = _HTML_TEMPLATE.format(
        title=title,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        input_fields=input_fields,
        input_read=input_read,
        js_src=js_src,
        fn_name=function_name,
    )

    if path is not None:
        Path(path).write_text(html, encoding="utf-8")
    return html
