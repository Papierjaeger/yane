"""Matrix export for compatible feed-forward genomes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from yane.util.activation import ActivationType

if TYPE_CHECKING:
    from yane.core.genome import Genome


@dataclass
class MatrixGenome:
    weights: np.ndarray
    bias: np.ndarray
    activation: tuple[str, ...]
    input_indices: tuple[int, ...]
    output_indices: tuple[int, ...]
    exec_indices: tuple[int, ...]


class MatrixForwardCache:
    """Cache matrix exports for compatible genomes and invalidate by topology."""

    def __init__(self) -> None:
        self._cache: dict[int, tuple[tuple, MatrixGenome]] = {}

    @staticmethod
    def signature(genome: "Genome") -> tuple:
        return (
            len(genome.nodes),
            genome.connection_count,
            tuple(
                (src.innovation, conn.target.innovation, conn.innovation, conn.enabled, conn.weight)
                for src in genome.nodes
                for conn in src.connections
            ),
        )

    def get(self, genome: "Genome") -> MatrixGenome:
        key = id(genome)
        sig = self.signature(genome)
        cached = self._cache.get(key)
        if cached is not None and cached[0] == sig:
            return cached[1]
        exported = export_matrix_genome(genome)
        self._cache[key] = (sig, exported)
        return exported


def is_matrix_compatible(genome: "Genome") -> bool:
    try:
        export_matrix_genome(genome)
        return True
    except ValueError:
        return False


def export_matrix_genome(genome: "Genome") -> MatrixGenome:
    """Export an acyclic genome to adjacency-matrix form.

    Raises ValueError for cyclic genomes or persistent hidden memory because
    those require recurrent state handling rather than a simple DAG matrix pass.
    """
    genome._build_exec_order()
    if genome._has_cycles or _has_enabled_cycle(genome):
        raise ValueError("Matrix export only supports acyclic genomes")
    if any(n.persist_value for n in genome.nodes if n not in genome.output_nodes):
        raise ValueError("Matrix export does not support persistent hidden nodes")

    nodes = list(genome.nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    weights = np.zeros((n, n), dtype=np.float64)
    for src in nodes:
        src_i = node_to_idx[src]
        for conn in src.connections:
            if conn.enabled:
                weights[src_i, node_to_idx[conn.target]] = conn.weight
    bias = np.array([node.bias for node in nodes], dtype=np.float64)
    activation = tuple(
        node.activation.value if isinstance(node.activation, ActivationType) else node.activation
        for node in nodes
    )
    return MatrixGenome(
        weights=weights,
        bias=bias,
        activation=activation,
        input_indices=tuple(node_to_idx[n] for n in genome.input_nodes),
        output_indices=tuple(node_to_idx[n] for n in genome.output_nodes),
        exec_indices=tuple(node_to_idx[n] for n in (genome._exec_order or [])),
    )


def _has_enabled_cycle(genome: "Genome") -> bool:
    visiting: set = set()
    visited: set = set()

    def visit(node) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for conn in node.connections:
            if conn.enabled and visit(conn.target):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in genome.nodes)


def forward_matrix(exported: MatrixGenome, inputs, xp=np) -> list[float]:
    """Run a matrix-exported genome with NumPy-like array module."""
    values = xp.zeros(exported.weights.shape[0], dtype=xp.float64)
    for idx, value in zip(exported.input_indices, inputs):
        values[idx] = float(value)
    weights = xp.asarray(exported.weights)
    bias = xp.asarray(exported.bias)
    output_set = set(exported.output_indices)
    for idx in exported.exec_indices:
        v = values[idx] + bias[idx]
        act = exported.activation[idx]
        if act == "linear":
            out = v
        elif act == "tanh":
            out = xp.tanh(v)
        elif act == "relu":
            out = xp.maximum(0.0, v)
        else:
            out = 1.0 / (1.0 + xp.exp(-v))
        values = values + weights[idx] * out
        if idx in output_set:
            values[idx] = out
    return [float(values[idx]) for idx in exported.output_indices]


def forward_matrix_gpu(exported: MatrixGenome, inputs) -> list[float]:
    """Run a matrix export with CuPy when available."""
    try:
        import cupy as cp
    except ImportError as exc:
        raise ImportError("CuPy is required for forward_matrix_gpu()") from exc
    result = forward_matrix(exported, inputs, xp=cp)
    cp.cuda.Stream.null.synchronize()
    return result


def forward_compatible_batch(
    genomes: list["Genome"],
    batch_inputs,
    cache: MatrixForwardCache | None = None,
    use_gpu: bool = False,
) -> list[list[list[float]]]:
    """Evaluate compatible genomes over a batch of inputs via matrix exports."""
    cache = cache or MatrixForwardCache()
    outputs: list[list[list[float]]] = []
    for genome in genomes:
        exported = cache.get(genome)
        run = forward_matrix_gpu if use_gpu else forward_matrix
        outputs.append([run(exported, inputs) for inputs in batch_inputs])
    return outputs
