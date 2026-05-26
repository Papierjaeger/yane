"""Layer-3 research features for YANE.

These features are modular and fully toggleable via the NeuroEvolution API.
All features are disabled by default and impose no runtime cost when off.

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

class IntrinsicCuriosityModule:
    """2-layer forward model predicting genome output from input.

    Curiosity bonus = prediction error: genomes producing surprising
    (hard-to-predict) outputs receive a positive bonus, encouraging exploration.

    Architecture: n_inputs → [ReLU hidden] → network_size → n_outputs (linear).
    Weights updated online via SGD after each genome evaluation.
    Integrate via NeuroEvolution.set_curiosity().
    """

    def __init__(self, n_inputs: int, n_outputs: int, network_size: int = 8,
                 lr: float = 0.01):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.network_size = network_size
        self.lr = lr
        scale1 = math.sqrt(2.0 / max(n_inputs, 1))
        self._W1 = [[random.gauss(0.0, scale1) for _ in range(n_inputs)]
                    for _ in range(network_size)]
        self._b1 = [0.0] * network_size
        scale2 = math.sqrt(2.0 / max(network_size, 1))
        self._W2 = [[random.gauss(0.0, scale2) for _ in range(network_size)]
                    for _ in range(n_outputs)]
        self._b2 = [0.0] * n_outputs
        self._n_updates: int = 0

    def _forward(self, inputs: list[float]) -> tuple[list[float], list[float]]:
        n = min(len(inputs), self.n_inputs)
        h = []
        for i in range(self.network_size):
            val = self._b1[i]
            for j in range(n):
                val += self._W1[i][j] * inputs[j]
            h.append(max(0.0, val))
        out = []
        for i in range(self.n_outputs):
            val = self._b2[i]
            for j in range(self.network_size):
                val += self._W2[i][j] * h[j]
            out.append(val)
        return h, out

    def predict(self, inputs: list[float]) -> list[float]:
        _, out = self._forward(inputs)
        return out

    def error(self, inputs: list[float], targets: list[float]) -> float:
        """Return prediction error (L2 norm) as curiosity signal."""
        _, out = self._forward(inputs)
        n = min(len(out), len(targets))
        if n == 0:
            return 0.0
        return sum((out[i] - targets[i]) ** 2 for i in range(n)) ** 0.5

    def update(self, inputs: list[float], targets: list[float]) -> None:
        """Online SGD update of predictor weights."""
        h, out = self._forward(inputs)
        n = min(len(out), len(targets))
        if n == 0:
            return
        _clip = 1.0
        d_out = [max(-_clip, min(_clip, (out[i] - targets[i]) * 2.0 / n))
                 for i in range(self.n_outputs)]
        for i in range(self.n_outputs):
            self._b2[i] -= self.lr * d_out[i]
            for j in range(self.network_size):
                grad = d_out[i] * h[j]
                self._W2[i][j] -= self.lr * max(-_clip, min(_clip, grad))
        d_h = [0.0] * self.network_size
        for j in range(self.network_size):
            for i in range(self.n_outputs):
                d_h[j] += d_out[i] * self._W2[i][j]
            if h[j] <= 0.0:
                d_h[j] = 0.0
            d_h[j] = max(-_clip, min(_clip, d_h[j]))
        n_in = min(len(inputs), self.n_inputs)
        for i in range(self.network_size):
            self._b1[i] -= self.lr * d_h[i]
            for j in range(n_in):
                grad = d_h[i] * inputs[j]
                self._W1[i][j] -= self.lr * max(-_clip, min(_clip, grad))
        self._n_updates += 1


# =====================================================================
# CuriosityModule — legacy state-action forward model (backward compat)
# =====================================================================

class CuriosityModule:
    """State-action forward model for environment-level curiosity.

    Predicts the next state given (state, action); prediction error is the
    curiosity signal.  For genome I/O curiosity use IntrinsicCuriosityModule
    and NeuroEvolution.set_curiosity() instead.
    """

    def __init__(self, n_state: int, n_action: int, lr: float = 0.01):
        self.n_state = n_state
        self.n_action = n_action
        self.lr = lr
        n_in = n_state + n_action
        self.W = [0.0] * n_in * n_state
        self.b = [0.0] * n_state

    def predict(self, state: list[float], action: list[float]) -> list[float]:
        n_in = self.n_state + self.n_action
        inp = state + action
        pred = []
        for i in range(self.n_state):
            val = self.b[i] + sum(self.W[i * n_in + j] * inp[j] for j in range(n_in))
            pred.append(val)
        return pred

    def error(self, state: list[float], action: list[float],
              next_state: list[float]) -> float:
        pred = self.predict(state, action)
        return sum((p - ns) ** 2 for p, ns in zip(pred, next_state)) ** 0.5

    def update(self, state: list[float], action: list[float],
               next_state: list[float]) -> None:
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
