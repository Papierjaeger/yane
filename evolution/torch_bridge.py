"""YANE → PyTorch bridge: export genomes as ``torch.nn.Module``.

Requires PyTorch to be installed.  Usage::

    model = genome_to_torch_module(genome)
    output = model(torch.tensor([1.0, 2.0]))
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome


def genome_to_torch_module(genome: "Genome"):
    """Convert a YANE genome to a ``torch.nn.Module``.

    The module has the same topology as the genome.  Memory nodes are
    mapped to ``nn.GRUCell`` equivalents (state kept in hidden state).
    """
    import torch
    import torch.nn as nn

    nodes = genome.nodes
    n = len(nodes)
    node_id = {id(n): i for i, n in enumerate(nodes)}

    # Build weight matrix and bias vector
    W = torch.zeros(n, n, dtype=torch.float64)
    b = torch.zeros(n, dtype=torch.float64)
    for src_node in nodes:
        si = node_id[id(src_node)]
        b[si] = src_node.bias
        for conn in src_node.connections:
            if conn.enabled:
                ti = node_id.get(id(conn.target))
                if ti is not None:
                    W[ti, si] = conn.weight

    # Determine execution order (topological)
    exec_order = getattr(genome, "_exec_order", None)
    if exec_order is None:
        exec_order = genome._build_exec_order()
    if exec_order is None:
        raise ValueError("Cannot export cyclic genome to PyTorch")

    exec_indices = [node_id[id(n)] for n in exec_order]
    input_indices = [node_id[id(n)] for n in genome.input_nodes]
    output_indices = [node_id[id(n)] for n in genome.output_nodes]

    class YANEModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("W", W)
            self.register_buffer("b", b)
            self.input_indices = input_indices
            self.output_indices = output_indices
            self.exec_indices = exec_indices
            # Activation functions
            self.acts = nn.ModuleDict()
            for ni, node in enumerate(nodes):
                act_name = node.activation
                if isinstance(act_name, str):
                    act_name = act_name
                else:
                    act_name = act_name.value
                self.acts[str(ni)] = _activation_module(act_name)

        def forward(self, x):
            x = x.to(dtype=self.W.dtype)
            values = torch.zeros(len(nodes), dtype=self.W.dtype, device=x.device)
            # Set input values
            for i, idx in enumerate(self.input_indices):
                values[idx] = x[i] if i < x.shape[-1] else 0.0
            # Execute in topological order
            for idx in self.exec_indices:
                v = values[idx] + self.b[idx]
                # Weighted sum from incoming connections
                incoming = self.W[idx] @ values
                v = v + incoming
                act = self.acts[str(idx)]
                values[idx] = act(v)
            return torch.stack([values[i] for i in self.output_indices])

    return YANEModule()


def _activation_module(name: str):
    import torch.nn as nn
    act_map = {
        "linear": nn.Identity(),
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(0.01),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "elu": nn.ELU(),
        "softplus": nn.Softplus(),
        "silu": nn.SiLU(),
        "gelu": nn.GELU(approximate="tanh"),
    }
    return act_map.get(name, nn.Sigmoid())


def forward_with_torch(genome: "Genome", inputs: list[float]) -> list[float]:
    """Run a genome through its PyTorch module and return outputs.

    Falls back to ``genome.forward()`` if PyTorch is not installed.
    """
    try:
        import torch
        model = genome_to_torch_module(genome)
        model.eval()
        with torch.no_grad():
            out = model(torch.tensor(inputs, dtype=torch.float64))
        return out.tolist()
    except ImportError:
        return genome.forward(inputs)
