"""Optional PyTorch fine-tuning for exported feed-forward genomes."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome


def backprop_finetune_linear_outputs(
    genome: "Genome",
    samples: Sequence[tuple[Sequence[float], Sequence[float]]],
    steps: int = 25,
    lr: float = 0.01,
) -> float:
    """Fine-tune enabled connection weights with PyTorch, then write them back.

    This intentionally supports the conservative case first: acyclic genomes
    without persistent hidden state. Unsupported activations are approximated by
    sigmoid in the exported matrix path, matching ``matrix_export.forward_matrix``.
    Returns the final mean squared error.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for backprop_finetune_linear_outputs()") from exc

    from yane.evolution.matrix_export import export_matrix_genome

    exported = export_matrix_genome(genome)
    weights = torch.tensor(exported.weights, dtype=torch.float64, requires_grad=True)
    bias = torch.tensor(exported.bias, dtype=torch.float64)
    opt = torch.optim.Adam([weights], lr=lr)

    xs = torch.tensor([s[0] for s in samples], dtype=torch.float64)
    ys = torch.tensor([s[1] for s in samples], dtype=torch.float64)
    final_loss = 0.0
    for _ in range(max(1, steps)):
        opt.zero_grad()
        preds = []
        for row in xs:
            values = torch.zeros(weights.shape[0], dtype=torch.float64)
            for idx, value in zip(exported.input_indices, row):
                values[idx] = value
            for idx in exported.exec_indices:
                v = values[idx] + bias[idx]
                act = exported.activation[idx]
                if act == "linear":
                    out = v
                elif act == "tanh":
                    out = torch.tanh(v)
                elif act == "relu":
                    out = torch.relu(v)
                else:
                    out = torch.sigmoid(v)
                values = values + weights[idx] * out
            preds.append(values[list(exported.output_indices)])
        pred = torch.stack(preds)
        loss = torch.mean((pred - ys) ** 2)
        loss.backward()
        opt.step()
        final_loss = float(loss.detach())

    matrix = weights.detach().numpy()
    node_to_idx = {node: i for i, node in enumerate(genome.nodes)}
    for src in genome.nodes:
        src_i = node_to_idx[src]
        for conn in src.connections:
            if conn.enabled:
                conn.weight = float(matrix[src_i, node_to_idx[conn.target]])
    genome._invalidate_topology()
    return final_loss
