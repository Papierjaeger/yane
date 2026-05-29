"""Gradient-NEAT Hybrid Mode — interleaving NEAT evolution with backpropagation.

Every ``bp_interval`` generations, the top-K genomes are briefly fine-tuned
with gradient descent before being handed back to NEAT.  This combines the
global search of NEAT (topology + weight exploration) with the local search
efficiency of backprop (precise weight tuning in a fixed topology).

Requires PyTorch (``pip install torch``).  When PyTorch is absent, calling
:func:`run_hybrid_backprop` raises ``ImportError`` with a clear message; the
rest of this module works without PyTorch.

Replay Buffer
-------------
During NEAT evaluation, raw input vectors are accumulated in a
:class:`ReplayBuffer`.  At each backprop phase the buffer provides a
supervised training set: the **teacher** (best genome at that moment) labels
the inputs, and the **students** (top-K genomes) are trained toward those
labels.

This makes the hybrid self-supervised: no external dataset is required,
though one can be supplied via ``train_data`` for fully-supervised tasks
(e.g. XOR).

Usage::

    yane.set_hybrid_mode(
        enabled=True,
        bp_interval=10,      # backprop every 10 generations
        bp_epochs=50,        # gradient steps per backprop phase
        bp_lr=0.01,          # Adam learning rate
        bp_batch_size=32,    # replay buffer sample size
        top_k=3,             # genomes to fine-tune
        train_data=None,     # optional list of (inputs, targets)
    )
    yane.train(fitness_fn)
"""
from __future__ import annotations

import random
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Circular buffer of raw input vectors seen during NEAT evaluation.

    Parameters
    ----------
    max_size :
        Maximum number of entries stored (FIFO eviction).
    """

    def __init__(self, max_size: int = 10_000) -> None:
        self.max_size = max_size
        self._buffer: deque[list[float]] = deque(maxlen=max_size)

    def add(self, inputs: list[float]) -> None:
        """Record one input vector."""
        self._buffer.append(list(inputs))

    def sample(self, n: int, rng: random.Random | None = None) -> list[list[float]]:
        """Return up to *n* random entries from the buffer.

        Returns all entries when ``n >= len(buffer)``.
        """
        if not self._buffer:
            return []
        n = min(n, len(self._buffer))
        pop = list(self._buffer)
        if rng is not None:
            return rng.sample(pop, n)
        return random.sample(pop, n)

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()


# ---------------------------------------------------------------------------
# Weight sync utilities
# ---------------------------------------------------------------------------

def _node_id_map(genome: "Genome") -> dict[int, int]:
    return {id(n): i for i, n in enumerate(genome.nodes)}


def genome_to_trainable_module(genome: "Genome"):
    """Convert a genome to a PyTorch module with *trainable* parameters.

    Unlike :func:`~yane.evolution.torch_bridge.genome_to_torch_module` (which
    uses ``register_buffer``), this version wraps W and b as
    ``nn.Parameter`` so that ``loss.backward()`` can compute gradients.

    Raises ``ImportError`` when PyTorch is not installed.
    Raises ``ValueError`` for cyclic genomes.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError(
            "set_hybrid_mode() requires PyTorch.\n"
            "Install it with:  pip install torch"
        )

    nodes = genome.nodes
    n = len(nodes)
    nid = _node_id_map(genome)

    W_data = torch.zeros(n, n, dtype=torch.float64)
    b_data = torch.zeros(n, dtype=torch.float64)
    for src in nodes:
        si = nid[id(src)]
        b_data[si] = src.bias
        for conn in src.connections:
            if conn.enabled:
                ti = nid.get(id(conn.target))
                if ti is not None:
                    W_data[ti, si] = conn.weight

    exec_order = getattr(genome, "_exec_order", None)
    if exec_order is None:
        exec_order = genome._build_exec_order()
    if exec_order is None:
        raise ValueError("hybrid_neat: cannot export cyclic genome to PyTorch")

    exec_indices = [nid[id(nd)] for nd in exec_order]
    in_indices = [nid[id(nd)] for nd in genome.input_nodes]
    out_indices = [nid[id(nd)] for nd in genome.output_nodes]

    from yane.evolution.torch_bridge import _activation_module  # reuse activation map

    class TrainableYANEModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.W = nn.Parameter(W_data.clone())
            self.b = nn.Parameter(b_data.clone())
            self.input_indices = in_indices
            self.output_indices = out_indices
            self.exec_indices = exec_indices
            self.acts = nn.ModuleDict()
            for ni, node in enumerate(nodes):
                act_name = node.activation
                if not isinstance(act_name, str):
                    act_name = act_name.value
                self.acts[str(ni)] = _activation_module(act_name)

        def forward(self, x):
            x = x.to(dtype=self.W.dtype)
            values = torch.zeros(len(nodes), dtype=self.W.dtype, device=x.device)
            for i, idx in enumerate(self.input_indices):
                values[idx] = x[i] if i < x.shape[-1] else 0.0
            for idx in self.exec_indices:
                incoming = self.W[idx] @ values
                v = values[idx] + self.b[idx] + incoming
                values[idx] = self.acts[str(idx)](v)
            return torch.stack([values[i] for i in self.output_indices])

    return TrainableYANEModule()


def sync_weights_back(genome: "Genome", module) -> None:
    """Copy updated weight/bias tensors from *module* back to *genome*.

    After a backprop phase, call this to propagate gradient-updated values
    into the genome's connections and node biases so NEAT sees the improved
    weights.

    Parameters
    ----------
    genome :
        The source genome whose topology is used.
    module :
        A :class:`TrainableYANEModule` whose ``.W`` and ``.b`` parameters have
        been updated by backprop.
    """
    nid = _node_id_map(genome)
    W = module.W.detach()
    b = module.b.detach()
    for src in genome.nodes:
        si = nid[id(src)]
        src.bias = float(b[si])
        for conn in src.connections:
            if conn.enabled:
                ti = nid.get(id(conn.target))
                if ti is not None:
                    conn.weight = float(W[ti, si])
    genome._invalidate_topology()


# ---------------------------------------------------------------------------
# Backprop phase
# ---------------------------------------------------------------------------

def run_hybrid_backprop(
    genomes: list["Genome"],
    inputs_batch: list[list[float]],
    targets_batch: list[list[float]],
    bp_epochs: int = 50,
    bp_lr: float = 0.01,
    bp_batch_size: int = 32,
    rng: random.Random | None = None,
) -> dict:
    """Run one backprop phase on *genomes* against (inputs, targets) pairs.

    Each genome is independently fine-tuned by gradient descent against the
    MSE loss on the provided training batch.  Updated weights are written back
    to the genomes via :func:`sync_weights_back`.

    Parameters
    ----------
    genomes :
        Top-K genomes to fine-tune.
    inputs_batch :
        List of input vectors (the "X" of the training data).
    targets_batch :
        List of target output vectors corresponding to *inputs_batch* (the "y").
    bp_epochs :
        Number of gradient steps per genome.
    bp_lr :
        Adam optimiser learning rate.
    bp_batch_size :
        Mini-batch size drawn from *inputs_batch* per gradient step.
    rng :
        Optional RNG for mini-batch sampling.

    Returns
    -------
    dict
        ``{"losses": [final_loss_genome_0, ...], "n_epochs": bp_epochs}``

    Raises
    ------
    ImportError
        When PyTorch is not installed.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError(
            "Hybrid backprop requires PyTorch.\n"
            "Install it with:  pip install torch"
        )

    if not inputs_batch or not targets_batch:
        return {"losses": [], "n_epochs": 0}

    _rng = rng or random

    # Convert to tensors once
    X_all = torch.tensor(inputs_batch, dtype=torch.float64)
    Y_all = torch.tensor(targets_batch, dtype=torch.float64)

    losses: list[float] = []
    criterion = nn.MSELoss()

    for genome in genomes:
        try:
            module = genome_to_trainable_module(genome)
        except (ValueError, Exception):
            losses.append(float("inf"))
            continue

        optim = torch.optim.Adam(module.parameters(), lr=bp_lr)
        final_loss = float("inf")

        for epoch in range(bp_epochs):
            # Sample mini-batch
            n_data = X_all.shape[0]
            if n_data <= bp_batch_size:
                X_batch, Y_batch = X_all, Y_all
            else:
                idx = sorted(_rng.sample(range(n_data), bp_batch_size))
                X_batch = X_all[idx]
                Y_batch = Y_all[idx]

            optim.zero_grad()
            preds = torch.stack([module(X_batch[i]) for i in range(X_batch.shape[0])])
            loss = criterion(preds, Y_batch)
            loss.backward()
            optim.step()
            final_loss = float(loss.detach())

        sync_weights_back(genome, module)
        losses.append(final_loss)

    return {"losses": losses, "n_epochs": bp_epochs}
