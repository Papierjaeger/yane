"""Continual / Lifelong Learning NEAT — Training ohne Catastrophic Forgetting.

Implementiert vier Modi zur Vermeidung von Katastrophenvergessenheit:

**EWC (Elastic Weight Consolidation):**
Nach Aufgabe 1 werden wichtige Gewichte identifiziert (Ankerwerte).
Bei Aufgabe 2 wird ein Penalty auf Gewichtsänderungen addiert:
``penalty = (λ/2) * Σ (w_i - w*_i)²``

**Progressive:**
Nach jeder Aufgabe werden bestehende Gewichte eingefroren;
neue Knoten/Verbindungen können nur für neue Aufgaben hinzugefügt werden.

**Memory-Replay:**
Ein Replay-Buffer speichert (Eingabe, Ausgabe)-Paare aus früheren Aufgaben.
Beim Training neuer Aufgaben wird periodisch auf alten Beispielen nachtrainiert.

**Hybrid:**
EWC + Memory-Replay kombiniert.

Integration::

    yane.set_continual_learning(mode="ewc", lambda_ewc=0.1)
    yane.task_start("xor")
    yane.train(xor_evaluator)
    yane.task_start("parity")
    yane.train(parity_evaluator)   # uses EWC penalty
    results = yane.evaluate_all_tasks()
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome


# ---------------------------------------------------------------------------
# EWC Penalty
# ---------------------------------------------------------------------------

@dataclass
class TaskAnchor:
    """Stores the best genome and metadata from a completed task."""
    name: str
    best_genome: "Genome"
    best_fitness: float
    # Weight anchor: {(src_innov, tgt_innov): weight}
    anchor_weights: dict[tuple[int, int], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.anchor_weights and self.best_genome is not None:
            self.anchor_weights = _extract_weights(self.best_genome)


def _extract_weights(genome: "Genome") -> dict[tuple[int, int], float]:
    """Extract all enabled connection weights as {(src_innov, tgt_innov): weight}."""
    weights: dict[tuple[int, int], float] = {}
    for src in genome.nodes:
        for conn in src.connections:
            if conn.enabled and conn.innovation >= 0:
                weights[(src.innovation, conn.target.innovation)] = conn.weight
    return weights


def compute_ewc_penalty(
    genome: "Genome",
    anchors: list[TaskAnchor],
    lambda_ewc: float = 0.1,
) -> float:
    """Compute EWC penalty against all previous task anchors.

    Penalty = (λ/2) * Σ_tasks Σ_weights (w_i - w*_i)²

    Returns a non-negative float.
    """
    if not anchors or lambda_ewc <= 0:
        return 0.0
    current_weights = _extract_weights(genome)
    penalty = 0.0
    for anchor in anchors:
        for key, anchor_w in anchor.anchor_weights.items():
            current_w = current_weights.get(key, 0.0)
            delta = current_w - anchor_w
            penalty += delta * delta
    return (lambda_ewc / 2.0) * penalty


def make_ewc_fitness(
    base_fitness_fn: Callable[["Genome"], float],
    anchors: list[TaskAnchor],
    lambda_ewc: float = 0.1,
) -> Callable[["Genome"], float]:
    """Wrap *base_fitness_fn* with EWC regularization.

    Returns a new fitness function that subtracts the EWC penalty from the
    base fitness.

    Parameters
    ----------
    base_fitness_fn :
        The task-specific fitness function.
    anchors :
        Previous task anchors (from ``ContinualLearner.task_anchors``).
    lambda_ewc :
        Regularization strength.
    """
    def _ewc_fn(genome: "Genome") -> float:
        base = base_fitness_fn(genome)
        penalty = compute_ewc_penalty(genome, anchors, lambda_ewc)
        return base - penalty
    return _ewc_fn


# ---------------------------------------------------------------------------
# Progressive Expansion
# ---------------------------------------------------------------------------

def freeze_genome_weights(genome: "Genome") -> None:
    """Mark all current connections as 'frozen' (base spike_rate → 0).

    Frozen connections will not be mutated by NEAT.  New connections added
    later are not frozen and can be freely evolved for the new task.
    """
    for src in genome.nodes:
        for conn in src.connections:
            # Use spike_rate=0 as frozen marker (no random re-init during mutation)
            conn.spike_rate = 0.0


def progressive_expand(genome: "Genome", n_new_nodes: int = 2) -> None:
    """Add new hidden nodes for the next task; freeze existing weights.

    The new nodes have random connections to existing output nodes, giving
    the network fresh capacity without disturbing prior learned weights.
    """
    from yane.core.node import Node, NodeType
    from yane.core.connection import Connection
    from yane.util.activation import ActivationType
    freeze_genome_weights(genome)
    max_innov = max((n.innovation for n in genome.nodes if n.innovation >= 0), default=-1)
    for i in range(n_new_nodes):
        new_node = Node(NodeType.HIDDEN, max_innov + i + 1)
        new_node.activation = ActivationType.TANH
        genome.nodes.append(new_node)
        # Connect to all output nodes with fresh random weights
        for out_node in genome.output_nodes:
            innov = max_innov + n_new_nodes + i * len(genome.output_nodes) + genome.output_nodes.index(out_node)
            conn = Connection(out_node, innovation=innov)
            conn.weight = random.gauss(0.0, 0.5)
            new_node.connections.append(conn)
    genome._invalidate_topology()


# ---------------------------------------------------------------------------
# Memory Replay Buffer
# ---------------------------------------------------------------------------

class TaskMemory:
    """Stores (inputs, outputs) pairs from a previous task for replay.

    Parameters
    ----------
    max_size :
        Maximum number of stored examples.
    """

    def __init__(self, task_name: str, max_size: int = 1000) -> None:
        self.task_name = task_name
        self._buffer: deque[tuple[list[float], list[float]]] = deque(maxlen=max_size)

    def add(self, inputs: list[float], outputs: list[float]) -> None:
        self._buffer.append((list(inputs), list(outputs)))

    def sample(
        self,
        n: int,
        rng: random.Random | None = None,
    ) -> list[tuple[list[float], list[float]]]:
        if not self._buffer:
            return []
        _rng = rng or random
        n = min(n, len(self._buffer))
        return _rng.sample(list(self._buffer), n)

    def __len__(self) -> int:
        return len(self._buffer)


def make_replay_fitness(
    base_fitness_fn: Callable[["Genome"], float],
    memories: list[TaskMemory],
    replay_weight: float = 0.5,
    replay_samples: int = 10,
) -> Callable[["Genome"], float]:
    """Wrap *base_fitness_fn* with memory-replay regularization.

    Periodically computes MSE loss on old-task examples and subtracts it
    from the base fitness.

    Parameters
    ----------
    base_fitness_fn :
        Task-specific fitness.
    memories :
        Old-task memory buffers.
    replay_weight :
        How much old-task MSE affects the current fitness (0 = none).
    replay_samples :
        Number of samples to draw from each memory per evaluation.
    """
    def _replay_fn(genome: "Genome") -> float:
        base = base_fitness_fn(genome)
        if not memories or replay_weight <= 0:
            return base
        replay_loss = 0.0
        n_checked = 0
        for mem in memories:
            for inputs, expected_out in mem.sample(replay_samples):
                genome.reset()
                try:
                    actual_out = genome.forward(inputs)
                    for a, e in zip(actual_out, expected_out):
                        replay_loss += (a - e) ** 2
                        n_checked += 1
                except Exception:
                    pass
        if n_checked > 0:
            replay_loss /= n_checked
        return base - replay_weight * replay_loss
    return _replay_fn


# ---------------------------------------------------------------------------
# ContinualLearner — orchestrates all modes
# ---------------------------------------------------------------------------

class ContinualLearner:
    """Orchestrates continual learning across multiple tasks.

    Parameters
    ----------
    mode :
        Learning mode: ``"ewc"``, ``"progressive"``, ``"replay"``, or
        ``"hybrid"`` (EWC + replay).
    lambda_ewc :
        EWC regularization strength (used for ``"ewc"`` and ``"hybrid"``).
    replay_weight :
        Replay regularization strength (used for ``"replay"`` and ``"hybrid"``).
    replay_buffer_size :
        Max examples stored per task in the replay buffer.
    n_progressive_nodes :
        New hidden nodes added per task when ``mode="progressive"``.
    """

    MODES = ("ewc", "progressive", "replay", "hybrid")

    def __init__(
        self,
        mode: str = "ewc",
        lambda_ewc: float = 0.1,
        replay_weight: float = 0.5,
        replay_buffer_size: int = 500,
        n_progressive_nodes: int = 2,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode
        self.lambda_ewc = lambda_ewc
        self.replay_weight = replay_weight
        self.replay_buffer_size = replay_buffer_size
        self.n_progressive_nodes = n_progressive_nodes

        self.task_anchors: list[TaskAnchor] = []
        self.task_memories: list[TaskMemory] = []
        self._current_task: str = "task_0"

    def start_task(self, name: str) -> None:
        """Mark the beginning of a new task."""
        self._current_task = name

    def wrap_fitness(
        self,
        fitness_fn: Callable[["Genome"], float],
        ne: "NeuroEvolution | None" = None,
    ) -> Callable[["Genome"], float]:
        """Wrap *fitness_fn* with the appropriate continual-learning regularization.

        Parameters
        ----------
        fitness_fn :
            Task-specific fitness function.
        ne :
            The :class:`~yane.neuro_evolution.NeuroEvolution` instance
            (used by ``"progressive"`` to expand genomes before the first
            evaluation).
        """
        if not self.task_anchors and not self.task_memories:
            return fitness_fn  # first task: no regularization needed

        if self.mode == "ewc":
            return make_ewc_fitness(fitness_fn, self.task_anchors, self.lambda_ewc)
        elif self.mode == "replay":
            return make_replay_fitness(fitness_fn, self.task_memories,
                                       self.replay_weight)
        elif self.mode == "hybrid":
            ewc_fn = make_ewc_fitness(fitness_fn, self.task_anchors, self.lambda_ewc)
            return make_replay_fitness(ewc_fn, self.task_memories,
                                       self.replay_weight)
        else:  # progressive
            return fitness_fn  # expansion handled separately

    def finish_task(
        self,
        best_genome: "Genome",
        best_fitness: float,
        sample_inputs: list[list[float]] | None = None,
    ) -> None:
        """Called after a task completes — anchor weights + build memory.

        Parameters
        ----------
        best_genome :
            The best genome from the completed task.
        best_fitness :
            The best fitness achieved.
        sample_inputs :
            Input vectors to evaluate on *best_genome* and store in memory.
        """
        # Anchor (for EWC and hybrid)
        if self.mode in ("ewc", "hybrid"):
            anchor = TaskAnchor(
                name=self._current_task,
                best_genome=best_genome.copy(),
                best_fitness=best_fitness,
            )
            self.task_anchors.append(anchor)

        # Memory (for replay and hybrid)
        if self.mode in ("replay", "hybrid") and sample_inputs:
            mem = TaskMemory(self._current_task, max_size=self.replay_buffer_size)
            for inputs in sample_inputs:
                best_genome.reset()
                try:
                    out = best_genome.forward(inputs)
                    mem.add(inputs, [float(v) for v in out])
                except Exception:
                    pass
            self.task_memories.append(mem)

        # Progressive: expand genome for next task
        if self.mode == "progressive":
            progressive_expand(best_genome, n_new_nodes=self.n_progressive_nodes)

    def forgetting_rate(
        self,
        task_idx: int,
        current_genome: "Genome",
        evaluator: Callable[["Genome"], float],
    ) -> float:
        """Compute how much fitness was lost on a previous task.

        Returns ``1.0 - current_fitness / original_fitness``.
        A rate of 0.0 means no forgetting; 1.0 = complete forgetting.
        """
        if task_idx >= len(self.task_anchors):
            return 0.0
        anchor = self.task_anchors[task_idx]
        if anchor.best_fitness == 0.0:
            return 0.0
        try:
            current_fit = evaluator(current_genome)
        except Exception:
            return 1.0
        return max(0.0, 1.0 - current_fit / anchor.best_fitness)
