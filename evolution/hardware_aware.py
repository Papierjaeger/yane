"""Hardware-Aware NEAT — deployment-constraint evolution.

Estimates the compute cost (FLOPs), memory footprint, and on-device latency of
a YANE genome from its topology alone (no runtime measurement required).  When
constraints are active, genomes that exceed the budget receive a fitness penalty
proportional to the violation, letting NEAT naturally prefer smaller, faster
networks without breaking the evolution loop.

Usage::

    yane.set_hardware_constraints(
        max_flops=500_000,
        max_memory_bytes=8192,
        max_latency_us=200.0,
        target_platform="cortex-m4",
    )
    yane.train(eval_fn)       # penalty applied automatically each generation

    metrics = yane.hardware_profile(genome)
    print(metrics.flops, metrics.memory_bytes, metrics.latency_us)

    front = yane.hw_pareto_front()   # non-dominated (fitness, cost) genomes
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome


# ---------------------------------------------------------------------------
# Activation FLOP costs  (integer FLOPs per node evaluation)
# ---------------------------------------------------------------------------

# Rough cost in floating-point operations.
# - linear / step / binary: trivial (multiply-by-1 or compare)
# - relu / leaky_relu: single compare + conditional multiply
# - sigmoid / tanh: exp + divide (≈ 4 FLOPs each)
# - sin / cos: look-up polynomial ≈ 6 FLOPs
# - gaussian: exp(-x²) ≈ exp + square ≈ 5 FLOPs

_ACTIVATION_FLOPS: dict[str, int] = {
    "linear":     1,
    "step":       1,
    "binary":     1,
    "relu":       2,
    "leaky_relu": 2,
    "swish":      3,
    "softplus":   4,
    "sigmoid":    4,
    "tanh":       4,
    "elu":        4,
    "gaussian":   5,
    "sine":       6,
    "cosine":     6,
    "cubic":      3,
    "square":     2,
    "abs":        2,
}
_DEFAULT_ACTIVATION_FLOPS = 3  # fallback for unknown activation types


# ---------------------------------------------------------------------------
# Platform profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlatformProfile:
    """Clock-speed and FPU characteristics of a deployment target."""

    name:             str
    mhz:              float   # effective clock speed in MHz
    cycles_per_flop:  float   # FLOPs / cycle (FPU: ~1-3; software FP: ~5-20)
    description:      str = ""

    @property
    def flops_per_second(self) -> float:
        return self.mhz * 1e6 / self.cycles_per_flop


# fmt: off
PLATFORM_PROFILES: dict[str, PlatformProfile] = {
    "cortex-m4":       PlatformProfile("cortex-m4",        168,  3.0,  "STM32F4 series (FPU)"),
    "cortex-m7":       PlatformProfile("cortex-m7",        400,  2.0,  "STM32H7 series (double-precision FPU)"),
    "esp32":           PlatformProfile("esp32",            240, 10.0,  "ESP32 Xtensa LX6 (software FP on single-core)"),
    "raspberry-pi-zero": PlatformProfile("raspberry-pi-zero", 1000, 5.0, "RPi Zero W, ARM1176 (single-core)"),
    "raspberry-pi-4":  PlatformProfile("raspberry-pi-4",  1800,  2.0,  "RPi 4, Cortex-A72 (NEON SIMD)"),
    "desktop":         PlatformProfile("desktop",          3000,  1.0,  "Modern x86-64 (AVX, out-of-order)"),
    "mobile-arm":      PlatformProfile("mobile-arm",       2500,  1.5,  "Smartphone SoC (Cortex-A78 class)"),
}
# fmt: on

_DEFAULT_PLATFORM = "desktop"


# ---------------------------------------------------------------------------
# Metrics and constraints
# ---------------------------------------------------------------------------

@dataclass
class HardwareMetrics:
    """Deployment cost metrics for a single genome."""

    flops:         int    # floating-point operations per forward pass
    memory_bytes:  int    # estimated on-device footprint in bytes
    latency_us:    float  # estimated inference latency in microseconds
    platform:      str    # platform used for latency estimate

    def __repr__(self) -> str:
        return (
            f"HardwareMetrics(flops={self.flops:,}, "
            f"memory_bytes={self.memory_bytes:,}, "
            f"latency_us={self.latency_us:.2f}, "
            f"platform={self.platform!r})"
        )


@dataclass
class HardwareConstraints:
    """Budget limits and penalty settings for hardware-aware evolution."""

    max_flops:        int | None   = None  # FLOPs per forward pass
    max_memory_bytes: int | None   = None  # bytes
    max_latency_us:   float | None = None  # microseconds
    target_platform:  str          = _DEFAULT_PLATFORM
    penalty_scale:    float        = 1.0   # multiplier on violation penalty

    # Byte sizes for the minimal C-struct deployment model
    bytes_per_node:       int = 8   # float bias + uint8 activation + padding
    bytes_per_connection: int = 8   # float weight + uint16 from_id + uint16 to_id


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_hardware_metrics(
    genome: "Genome",
    constraints: HardwareConstraints | None = None,
) -> HardwareMetrics:
    """Return FLOPs, memory, and latency for *genome*.

    All metrics are derived from topology; no runtime measurement needed.

    FLOPs model
    -----------
    - Each enabled connection contributes 1 multiplication + 1 addition = 2 FLOPs.
    - Each non-input node contributes: 1 bias addition + activation-function FLOPs.

    Memory model
    ------------
    ``n_nodes * bytes_per_node + n_connections * bytes_per_connection``
    using a minimal C-struct layout suitable for embedded deployment.

    Latency model
    -------------
    ``latency_us = flops * cycles_per_flop / (mhz * 1e6) * 1e6``
    i.e.  ``flops / flops_per_second * 1e6`` microseconds.
    """
    if constraints is None:
        constraints = HardwareConstraints()

    profile = PLATFORM_PROFILES.get(constraints.target_platform)
    if profile is None:
        raise ValueError(
            f"Unknown target_platform {constraints.target_platform!r}.  "
            f"Available: {sorted(PLATFORM_PROFILES)}"
        )

    nodes       = genome.nodes
    n_nodes     = len(nodes)
    n_inputs    = len(genome.input_nodes)
    n_non_input = max(0, n_nodes - n_inputs)

    # Count enabled connections across all nodes
    n_connections = sum(
        1 for node in nodes for conn in node.connections if conn.enabled
    )

    # FLOPs: connections + per-node activation cost
    conn_flops = n_connections * 2  # 1 mul + 1 add each
    node_flops = 0
    for node in nodes:
        if node.input_index is not None:
            continue   # input nodes: no computation beyond passthrough
        act_name = node.activation.value if hasattr(node.activation, "value") else str(node.activation)
        node_flops += 1 + _ACTIVATION_FLOPS.get(act_name, _DEFAULT_ACTIVATION_FLOPS)  # bias add + activation

    flops = conn_flops + node_flops

    # Memory
    memory_bytes = (
        n_nodes       * constraints.bytes_per_node
        + n_connections * constraints.bytes_per_connection
    )

    # Latency
    fps = profile.flops_per_second  # FLOPs per second
    latency_us = (flops / max(1.0, fps)) * 1e6

    return HardwareMetrics(
        flops=flops,
        memory_bytes=memory_bytes,
        latency_us=latency_us,
        platform=constraints.target_platform,
    )


# ---------------------------------------------------------------------------
# Penalty
# ---------------------------------------------------------------------------

def compute_penalty(
    metrics: HardwareMetrics,
    constraints: HardwareConstraints,
) -> float:
    """Return a non-negative fitness penalty for constraint violations.

    Each violated budget contributes ``(actual/limit - 1) * penalty_scale``.
    The total penalty is the sum across all active constraints and is
    subtracted from the genome's fitness in ``_finalize_fitness()``.
    """
    penalty = 0.0
    scale   = constraints.penalty_scale

    if constraints.max_flops is not None and metrics.flops > constraints.max_flops:
        excess = (metrics.flops - constraints.max_flops) / constraints.max_flops
        penalty += excess * scale

    if (constraints.max_memory_bytes is not None
            and metrics.memory_bytes > constraints.max_memory_bytes):
        excess = (metrics.memory_bytes - constraints.max_memory_bytes) / constraints.max_memory_bytes
        penalty += excess * scale

    if (constraints.max_latency_us is not None
            and metrics.latency_us > constraints.max_latency_us):
        excess = (metrics.latency_us - constraints.max_latency_us) / constraints.max_latency_us
        penalty += excess * scale

    return penalty


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------

def _hw_dominates(
    a_fitness: float,
    a_metrics: HardwareMetrics,
    b_fitness: float,
    b_metrics: HardwareMetrics,
) -> bool:
    """Return True if solution A Pareto-dominates solution B.

    In the (fitness, cost) space we maximise fitness and minimise cost
    (FLOPs, memory, latency).  A dominates B if A is no worse on every
    objective and strictly better on at least one.
    """
    if a_fitness < b_fitness:            return False
    if a_metrics.flops > b_metrics.flops:        return False
    if a_metrics.memory_bytes > b_metrics.memory_bytes: return False
    if a_metrics.latency_us > b_metrics.latency_us:    return False
    return (
        a_fitness > b_fitness
        or a_metrics.flops         < b_metrics.flops
        or a_metrics.memory_bytes  < b_metrics.memory_bytes
        or a_metrics.latency_us    < b_metrics.latency_us
    )


def hw_pareto_front(
    genomes: list["Genome"],
    constraints: HardwareConstraints | None = None,
) -> list[tuple["Genome", HardwareMetrics]]:
    """Return the non-dominated set from *genomes*.

    Each element is ``(genome, HardwareMetrics)``.  The front is sorted by
    descending fitness.

    A genome is non-dominated if no other genome in the population is
    simultaneously at least as good on fitness AND at least as cheap on
    FLOPs, memory, and latency.
    """
    if not genomes:
        return []

    if constraints is None:
        constraints = HardwareConstraints()

    scored: list[tuple[float, "Genome", HardwareMetrics]] = []
    for g in genomes:
        m = compute_hardware_metrics(g, constraints)
        scored.append((g.fitness, g, m))

    front: list[tuple["Genome", HardwareMetrics]] = []
    for i, (fi, gi, mi) in enumerate(scored):
        dominated = any(
            _hw_dominates(fj, mj, fi, mi)
            for j, (fj, gj, mj) in enumerate(scored)
            if j != i
        )
        if not dominated:
            front.append((gi, mi))

    front.sort(key=lambda t: t[0].fitness, reverse=True)
    return front
