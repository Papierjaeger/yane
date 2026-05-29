"""Population Distillation — compress a NEAT ensemble into a compact student genome.

Distillation transfers the collective knowledge of the top-K trained genomes
(the "teacher ensemble") into a single, smaller "student" genome by minimising
the mean squared error (MSE) between the student's and the ensemble's outputs
on a set of probe inputs.

The optimisation is gradient-free: the student's weights and biases are tuned
via repeated hill-climbing (Lamarckian refinement), which requires no external
ML library and works naturally within YANE's existing machinery.

Usage::

    # After training
    result = yane.distill_ensemble(k=5, target_nodes=10,
                                   distillation_steps=500)
    print(result.final_loss)
    print(result.compression_ratio)
    student = result.student

Standalone usage::

    from yane.evolution.distillation import distill_ensemble, DistillationResult
    result = distill_ensemble(teachers, student, probe_inputs,
                              distillation_steps=300)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class DistillationResult:
    """Result of an ensemble distillation run."""

    student: "Genome"
    """The distilled (student) genome."""

    final_loss: float
    """Final MSE between student and teacher ensemble on the probe inputs."""

    initial_loss: float
    """MSE before distillation started."""

    loss_history: list[float] = field(default_factory=list)
    """MSE at regular intervals (sampled, not every step)."""

    distillation_steps: int = 0
    """Total number of hill-climbing steps performed."""

    n_teachers: int = 0
    """Number of teacher genomes in the ensemble."""

    teacher_mean_nodes: float = 0.0
    """Mean node count of the teacher ensemble."""

    teacher_mean_connections: float = 0.0
    """Mean enabled connection count of the teacher ensemble."""

    @property
    def student_nodes(self) -> int:
        return len(self.student.nodes)

    @property
    def student_connections(self) -> int:
        return sum(
            sum(1 for c in n.connections if c.enabled)
            for n in self.student.nodes
        )

    @property
    def compression_ratio(self) -> float:
        """``teacher_mean_nodes / student_nodes`` — higher = more compression.

        Returns 1.0 when teacher_mean_nodes == 0 (degenerate case).
        """
        if self.teacher_mean_nodes <= 0:
            return 1.0
        return self.teacher_mean_nodes / max(1, self.student_nodes)

    @property
    def loss_is_monotone(self) -> bool:
        """True when the loss history is non-increasing (hill-climbing guarantee)."""
        if len(self.loss_history) < 2:
            return True
        return all(self.loss_history[i] >= self.loss_history[i + 1]
                   for i in range(len(self.loss_history) - 1))


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _mse(student: "Genome", teacher_outputs: list[list[float]],
         probe_inputs: list[list[float]]) -> float:
    """Compute mean squared error between student and pre-computed teacher outputs."""
    total = 0.0
    n = 0
    for inputs, target in zip(probe_inputs, teacher_outputs):
        student.reset()
        try:
            student_out = student.forward(inputs)
        except Exception:
            return float("inf")
        for s, t in zip(student_out, target):
            diff = s - t
            total += diff * diff
            n += 1
    return total / max(1, n)


def _generate_probe_inputs(
    n_inputs: int,
    n_probes: int,
    rng: random.Random,
    input_range: tuple[float, float] = (0.0, 1.0),
) -> list[list[float]]:
    lo, hi = input_range
    return [[rng.uniform(lo, hi) for _ in range(n_inputs)] for _ in range(n_probes)]


def _teacher_outputs(
    teachers: list["Genome"],
    probe_inputs: list[list[float]],
) -> list[list[float]]:
    """Compute the mean ensemble output for each probe input."""
    results: list[list[float]] = []
    for inputs in probe_inputs:
        outputs_per_teacher: list[list[float]] = []
        for t in teachers:
            t.reset()
            try:
                out = t.forward(inputs)
                outputs_per_teacher.append([float(v) for v in out])
            except Exception:
                pass
        if not outputs_per_teacher:
            results.append([0.0] * (len(teachers[0].output_nodes) if teachers else 1))
            continue
        n_out = len(outputs_per_teacher[0])
        mean_out = [
            sum(o[j] for o in outputs_per_teacher) / len(outputs_per_teacher)
            for j in range(n_out)
        ]
        results.append(mean_out)
    return results


def _make_student(
    n_inputs: int,
    n_outputs: int,
    target_nodes: int,
    rng: random.Random,
    max_connections: int | None = None,
) -> "Genome":
    """Build a small fully-connected genome for use as the student."""
    from yane.core.genome import Genome
    from yane.core.node import Node, NodeType
    from yane.core.connection import Connection
    from yane.util.activation import ActivationType

    g = Genome()
    n_hidden = max(0, target_nodes - n_inputs - n_outputs)
    g.max_nodes = target_nodes
    g.max_connections = max_connections

    for i in range(n_inputs):
        n = Node(NodeType.INPUT, i)
        n.activation = ActivationType.LINEAR
        n.input_index = i
        g.input_nodes.append(n)
        g.nodes.append(n)

    hidden_nodes = []
    for h in range(n_hidden):
        n = Node(NodeType.HIDDEN, n_inputs + h)
        n.activation = ActivationType.TANH
        n.bias = rng.gauss(0.0, 0.1)
        hidden_nodes.append(n)
        g.nodes.append(n)

    for j in range(n_outputs):
        n = Node(NodeType.OUTPUT, n_inputs + n_hidden + j)
        n.activation = ActivationType.SIGMOID
        n.bias = rng.gauss(0.0, 0.1)
        g.output_nodes.append(n)
        g.nodes.append(n)

    innov = n_inputs + n_hidden + n_outputs
    # Connect inputs → hidden → outputs (or inputs → outputs if no hidden)
    destinations = hidden_nodes + g.output_nodes
    for src in g.input_nodes:
        for dst in destinations:
            c = Connection(dst, innovation=innov)
            c.weight = rng.gauss(0.0, 0.5)
            src.connections.append(c)
            innov += 1
    for src in hidden_nodes:
        for dst in g.output_nodes:
            c = Connection(dst, innovation=innov)
            c.weight = rng.gauss(0.0, 0.5)
            src.connections.append(c)
            innov += 1

    g._invalidate_topology()
    return g


# ---------------------------------------------------------------------------
# Hill-climbing distillation loop
# ---------------------------------------------------------------------------

def _hill_climb_mse(
    student: "Genome",
    teacher_outputs: list[list[float]],
    probe_inputs: list[list[float]],
    n_steps: int,
    sigma: float,
    log_interval: int,
    rng: random.Random,
) -> tuple[float, list[float]]:
    """Optimise student weights/biases to minimise MSE vs teacher outputs.

    Uses simple hill-climbing: perturb all weights, accept if MSE decreases.
    Returns ``(final_mse, loss_history)``.

    ``loss_history`` contains the MSE sampled every ``log_interval`` steps.
    """
    conns = student.get_lamarck_connections()
    nodes = student.nodes

    current_loss = _mse(student, teacher_outputs, probe_inputs)
    history = [current_loss]

    for step in range(1, n_steps + 1):
        # Save state
        saved_w = [c.weight for c in conns]
        saved_b = [n.bias for n in nodes]

        # Perturb
        for c in conns:
            c.weight += rng.gauss(0.0, sigma)
        for n in nodes:
            if n not in student.input_nodes:
                n.bias += rng.gauss(0.0, sigma * 0.5)
        student._invalidate_topology()

        new_loss = _mse(student, teacher_outputs, probe_inputs)

        if new_loss <= current_loss:
            current_loss = new_loss  # accept
        else:
            # Revert
            for c, w in zip(conns, saved_w):
                c.weight = w
            for n, b in zip(nodes, saved_b):
                n.bias = b
            student._invalidate_topology()

        if step % log_interval == 0:
            history.append(current_loss)

    # Append final loss if not already the last entry
    if history[-1] != current_loss:
        history.append(current_loss)

    return current_loss, history


# ---------------------------------------------------------------------------
# Main distillation function
# ---------------------------------------------------------------------------

def distill_ensemble(
    teachers: list["Genome"],
    student: "Genome | None" = None,
    probe_inputs: list[list[float]] | None = None,
    distillation_steps: int = 500,
    n_probes: int = 100,
    sigma: float = 0.1,
    sigma_decay: float = 0.99,
    log_interval: int = 50,
    seed: int | None = None,
) -> DistillationResult:
    """Distil a teacher ensemble into a student genome via MSE minimisation.

    Parameters
    ----------
    teachers :
        List of teacher genomes (the ensemble).  All must share the same
        input/output count.
    student :
        Student genome to optimise.  When *None*, you must call
        ``NeuroEvolution.distill_ensemble()`` which creates the student
        from ``target_nodes``.
    probe_inputs :
        Input vectors to use for distillation.  When *None*, ``n_probes``
        random inputs in [0, 1]^n_inputs are generated.
    distillation_steps :
        Total number of hill-climbing steps.
    n_probes :
        Number of random probe inputs generated when *probe_inputs* is *None*.
    sigma :
        Initial perturbation noise scale (hill-climbing step size).
    sigma_decay :
        Multiplicative decay applied to *sigma* after each accepted step
        (annealing).  Use 1.0 to keep sigma constant.
    log_interval :
        Record MSE in ``loss_history`` every this many steps.
    seed :
        RNG seed for reproducibility.

    Returns
    -------
    DistillationResult
    """
    if not teachers:
        raise ValueError("distill_ensemble: teachers list must not be empty")
    if student is None:
        raise ValueError("distill_ensemble: provide a student genome or use "
                         "NeuroEvolution.distill_ensemble() which creates one")

    rng = random.Random(seed)

    n_inputs = len(teachers[0].input_nodes)
    if probe_inputs is None:
        probe_inputs = _generate_probe_inputs(n_inputs, n_probes, rng)

    # Pre-compute teacher outputs once
    t_outputs = _teacher_outputs(teachers, probe_inputs)

    # Metrics
    teacher_mean_nodes = sum(len(t.nodes) for t in teachers) / len(teachers)
    teacher_mean_conns = sum(
        sum(1 for n in t.nodes for c in n.connections if c.enabled)
        for t in teachers
    ) / len(teachers)

    initial_loss = _mse(student, t_outputs, probe_inputs)

    # Run hill-climbing with optional sigma decay
    if sigma_decay < 1.0:
        # Split into mini-epochs to apply decay
        epoch_size = max(1, distillation_steps // 20)
        history = [initial_loss]
        current_loss = initial_loss
        cur_sigma = sigma
        total_done = 0
        while total_done < distillation_steps:
            steps_this = min(epoch_size, distillation_steps - total_done)
            current_loss, partial_hist = _hill_climb_mse(
                student, t_outputs, probe_inputs, steps_this, cur_sigma,
                log_interval=max(1, steps_this), rng=rng,
            )
            history.append(current_loss)
            cur_sigma = max(1e-6, cur_sigma * sigma_decay)
            total_done += steps_this
    else:
        current_loss, history = _hill_climb_mse(
            student, t_outputs, probe_inputs, distillation_steps, sigma,
            log_interval=log_interval, rng=rng,
        )

    return DistillationResult(
        student=student,
        final_loss=current_loss,
        initial_loss=initial_loss,
        loss_history=history,
        distillation_steps=distillation_steps,
        n_teachers=len(teachers),
        teacher_mean_nodes=teacher_mean_nodes,
        teacher_mean_connections=teacher_mean_conns,
    )
