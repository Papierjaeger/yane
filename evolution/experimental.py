"""Experimental Layer-3 research features for YANE.

This module contains features that are experimental and may change or break.
They are isolated from the stable core per the architecture guidelines.

Contents:
- InputGrouping / OutputGrouping: evolvable aggregation layers
- SharedWeights: weight sharing across connections
- ConvNEAT: convolutional NEAT modules
- ESHyperNEAT: evolvable substrate HyperNEAT
- DARTSLite: differentiable architecture search relaxation
- CuriosityModule: intrinsic reward for exploration
- STDP: spike-timing-dependent plasticity rule
- Neuromodulation: dynamic gating of plasticity
- CoEvolution: paired environment-agent co-evolution
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable

from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType


# =====================================================================
# Input / Output Grouping — evolvable aggregation layers
# =====================================================================

class InputGrouping:
    """Evolvable input aggregation: group raw inputs before feeding to network.

    Each group applies a configurable reduction (mean, max, sum) to a subset
    of input channels.  The grouping pattern is encoded as strategy genes
    on the genome.
    """

    def __init__(self, n_raw: int, n_groups: int = 4):
        self.n_raw = n_raw
        self.n_groups = n_groups
        # Group assignment: raw_input_idx → group_idx
        self.assignment: list[int] = [i % n_groups for i in range(n_raw)]
        self.reduction: str = "mean"  # mean, max, sum

    def forward(self, raw_inputs: list[float]) -> list[float]:
        groups = [0.0] * self.n_groups
        counts = [0] * self.n_groups
        for i, v in enumerate(raw_inputs):
            g = self.assignment[i] if i < len(self.assignment) else i % self.n_groups
            groups[g] += v
            counts[g] += 1
        if self.reduction == "mean":
            return [groups[g] / max(counts[g], 1) for g in range(self.n_groups)]
        elif self.reduction == "max":
            # Recompute with max
            vals = [0.0] * self.n_groups
            for i, v in enumerate(raw_inputs):
                g = self.assignment[i] if i < len(self.assignment) else i % self.n_groups
                vals[g] = max(vals[g], v)
            return vals
        return groups  # sum


class OutputGrouping:
    """Evolvable output de-aggregation: expand few output nodes to many actions.

    Maps network outputs to action channels via duplication or interpolation.
    """

    def __init__(self, n_outputs: int, n_actions: int):
        self.n_outputs = n_outputs
        self.n_actions = n_actions

    def forward(self, outputs: list[float]) -> list[float]:
        if self.n_actions <= self.n_outputs:
            return outputs[:self.n_actions]
        # Stretch via linear interpolation
        result = []
        for i in range(self.n_actions):
            src = i * (self.n_outputs - 1) / max(self.n_actions - 1, 1)
            lo = int(src)
            hi = min(lo + 1, self.n_outputs - 1)
            frac = src - lo
            result.append(outputs[lo] * (1 - frac) + outputs[hi] * frac)
        return result


# =====================================================================
# Shared Weights — weight sharing across connections
# =====================================================================

class SharedWeightGroup:
    """A group of connections that share the same weight value.

    When the shared weight is updated (e.g. via backprop or mutation),
    ALL connections in the group are updated simultaneously.
    """

    def __init__(self, group_id: int, initial_weight: float = 0.0):
        self.group_id = group_id
        self.weight = initial_weight
        self.connections: list[Connection] = []

    def add(self, conn: Connection) -> None:
        self.connections.append(conn)
        conn.weight = self.weight

    def set_weight(self, w: float) -> None:
        self.weight = w
        for conn in self.connections:
            conn.weight = w


def apply_shared_weights(genome: Genome, groups: list[SharedWeightGroup]) -> None:
    """Apply shared weight groups to a genome's connections."""
    for group in groups:
        for conn in group.connections:
            conn.weight = group.weight


# =====================================================================
# ConvNEAT — convolutional NEAT (CoDeepNEAT-inspired)
# =====================================================================

class ConvModule:
    """A convolutional module that can be evolved by NEAT.

    The module wraps a small genome that acts as a convolution kernel:
    it processes local receptive fields and shares weights across space.
    """

    def __init__(self, kernel_size: int = 3, in_channels: int = 1):
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        # Internal genome encodes the kernel weights
        self.genome = Genome()
        # Create input nodes for each kernel element
        for i in range(kernel_size * kernel_size * in_channels):
            n = Node(NodeType.INPUT, i)
            n.activation = ActivationType.LINEAR
            self.genome.input_nodes.append(n)
            self.genome.nodes.append(n)
        # Output node
        out = Node(NodeType.OUTPUT, kernel_size * kernel_size * in_channels)
        out.activation = ActivationType.SIGMOID
        self.genome.output_nodes.append(out)
        self.genome.nodes.append(out)

    def forward(self, patch: list[float]) -> float:
        """Process a local patch and return the convolved value."""
        out = self.genome.forward(patch)
        return out[0] if out else 0.0


# =====================================================================
# ES-HyperNEAT — evolvable substrate
# =====================================================================

class Substrate:
    """A geometric substrate defining node positions for ES-HyperNEAT.

    Nodes are placed on a 2D grid; connection weights are generated by
    querying a CPPN at the source/target coordinates.
    """

    def __init__(self, width: int = 4, height: int = 4):
        self.width = width
        self.height = height
        self.nodes: list[tuple[float, float]] = [
            (x / width, y / height)
            for y in range(height) for x in range(width)
        ]

    def query_cppn(self, cppn: Genome, src: tuple[float, float],
                   tgt: tuple[float, float]) -> float:
        """Query a CPPN genome to get the weight between two substrate nodes."""
        out = cppn.forward([src[0], src[1], tgt[0], tgt[1]])
        return out[0] if out else 0.0


# =====================================================================
# DARTS-Lite — differentiable architecture search relaxation
# =====================================================================

class DARTSLite:
    """Lightweight differentiable architecture search.

    Uses continuous gating (sigmoid) on connection existence.  Gates with
    values below threshold after training are pruned.
    """

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.gates: dict[int, float] = {}  # conn_innovation → gate value

    def get_gate(self, conn_innov: int) -> float:
        return self.gates.get(conn_innov, 1.0)

    def set_gate(self, conn_innov: int, value: float) -> None:
        self.gates[conn_innov] = max(0.0, min(1.0, value))

    def apply_gates(self, genome: Genome) -> None:
        """Disable connections whose gate is below threshold."""
        for src in genome.nodes:
            for conn in src.connections:
                gate = self.gates.get(conn.innovation, 1.0)
                if gate < self.threshold:
                    conn.enabled = False

    def prune_gates(self, genome: Genome) -> int:
        """Remove gated-out connections. Returns count removed."""
        removed = 0
        for src in list(genome.nodes):
            to_remove = []
            for conn in src.connections:
                gate = self.gates.get(conn.innovation, 1.0)
                if gate < self.threshold:
                    to_remove.append(conn)
            for conn in to_remove:
                src.connections.remove(conn)
                removed += 1
        if removed:
            genome._invalidate_topology()
        return removed


# =====================================================================
# Curiosity Module — intrinsic reward
# =====================================================================

class CuriosityModule:
    """Intrinsic reward module based on prediction error.

    A small forward model predicts the next state given (state, action).
    High prediction error → high curiosity bonus.
    """

    def __init__(self, n_state: int, n_action: int, lr: float = 0.01):
        self.n_state = n_state
        self.n_action = n_action
        self.lr = lr
        # Simple linear predictor: W @ [s, a] + b
        self.W = [0.0] * (n_state + n_action) * n_state
        self.b = [0.0] * n_state

    def predict(self, state: list[float], action: list[float]) -> list[float]:
        n_in = self.n_state + self.n_action
        inp = state + action
        pred = []
        for i in range(self.n_state):
            val = self.b[i]
            for j in range(n_in):
                val += self.W[i * n_in + j] * inp[j]
            pred.append(val)
        return pred

    def error(self, state: list[float], action: list[float],
              next_state: list[float]) -> float:
        pred = self.predict(state, action)
        return sum((p - ns) ** 2 for p, ns in zip(pred, next_state)) ** 0.5

    def update(self, state: list[float], action: list[float],
               next_state: list[float]) -> None:
        """Online SGD update of prediction weights."""
        pred = self.predict(state, action)
        n_in = self.n_state + self.n_action
        inp = state + action
        for i in range(self.n_state):
            delta = pred[i] - next_state[i]
            self.b[i] -= self.lr * delta
            for j in range(n_in):
                self.W[i * n_in + j] -= self.lr * delta * inp[j]


# =====================================================================
# STDP — spike-timing-dependent plasticity
# =====================================================================

class STDPRule:
    """Simple STDP rule for weight updates based on pre/post firing order.

    Δw = A_plus * exp(-Δt / τ_plus)  if pre before post
    Δw = A_minus * exp(+Δt / τ_minus) if post before pre
    """

    def __init__(self, a_plus: float = 0.01, a_minus: float = 0.012,
                 tau_plus: float = 20.0, tau_minus: float = 20.0):
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus

    def update(self, conn: Connection, pre_time: float, post_time: float) -> None:
        dt = post_time - pre_time
        if dt > 0:
            conn.weight += self.a_plus * math.exp(-dt / self.tau_plus)
        else:
            conn.weight -= self.a_minus * math.exp(dt / self.tau_minus)


# =====================================================================
# Neuromodulation — dynamic gating of plasticity
# =====================================================================

class Neuromodulator:
    """Dynamic gating of neural plasticity via a modulatory signal.

    When the modulatory signal is high, plasticity is enabled; when low,
    weights are frozen.
    """

    def __init__(self, baseline: float = 0.5, plasticity_enabled: bool = True):
        self.signal = baseline
        self.plasticity_enabled = plasticity_enabled

    def modulate(self, conn: Connection, delta: float) -> float:
        """Return the modulated weight update."""
        if not self.plasticity_enabled:
            return 0.0
        return delta * self.signal

    def update_signal(self, reward: float, lr: float = 0.1) -> None:
        """Update modulatory signal based on reward (RL-style)."""
        self.signal += lr * (reward - self.signal)
        self.signal = max(0.0, min(1.0, self.signal))


# =====================================================================
# Co-Evolution — environment-agent pairing
# =====================================================================

class CoevolutionPair:
    """Pair an agent genome with an environment genome.

    The environment genome encodes task parameters (e.g. slope, obstacles).
    Both populations evolve simultaneously.
    """

    def __init__(self, agent: Genome, env: Genome):
        self.agent = agent
        self.env = env

    def evaluate(self, fitness_fn: Callable) -> float:
        """Evaluate the agent on the environment."""
        return fitness_fn(self.agent)


class CoevolutionPool:
    """Manages paired agent-environment evolution."""

    def __init__(self, n_agents: int = 50, n_envs: int = 20):
        self.agents: list[Genome] = []
        self.envs: list[Genome] = []
        self.history: list[dict] = []

    def add_agent(self, genome: Genome) -> None:
        self.agents.append(genome)

    def add_env(self, genome: Genome) -> None:
        self.envs.append(genome)

    def pair(self) -> list[CoevolutionPair]:
        """Create random agent-environment pairs."""
        pairs = []
        for agent in self.agents:
            env = random.choice(self.envs) if self.envs else None
            if env:
                pairs.append(CoevolutionPair(agent, env))
        return pairs
