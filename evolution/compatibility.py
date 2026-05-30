"""NEAT-style compatibility distance between two genomes.

Automatically dispatches to a fast Python path (small networks) or a
vectorised NumPy path (large networks) based on connection count.

Modular distance metrics (``DistanceMetric`` protocol + built-ins):
  TopologyDistance   — standard NEAT formula (default)
  WeightDistance     — average |Δweight| over matching gene pairs
  ActivationDistance — fraction of non-input nodes with different activation
  ChainMetric        — weighted sum of multiple metrics
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from yane.core.genome import Genome

# Below this connection count per genome, pure-Python frozenset operations are
# faster than NumPy due to array-creation overhead (~13 μs fixed cost).
# Above it, NumPy's vectorised searchsorted + abs win (crossover measured ~200).
_COMPAT_NUMPY_THRESHOLD: int = 200


# ---------------------------------------------------------------------------
# DistanceMetric protocol + built-in implementations
# ---------------------------------------------------------------------------

@runtime_checkable
class DistanceMetric(Protocol):
    """Protocol for pluggable genome-pair distance metrics."""
    def __call__(self, g1: Genome, g2: Genome) -> float: ...


class TopologyDistance:
    """NEAT compatibility distance δ = c1·E/N + c2·D/N + c3·W̄.

    Wraps the module-level ``compatibility()`` function.  When
    *only_enabled* is True, disabled connections are excluded from the
    innovation dict (useful for the ``topology_no_disabled`` speciation
    metric).
    """
    def __init__(self, only_enabled: bool = False) -> None:
        self._only_enabled = only_enabled

    def __call__(self, g1: Genome, g2: Genome) -> float:
        return compatibility(g1, g2, only_enabled=self._only_enabled)

    def __repr__(self) -> str:  # pragma: no cover
        return f"TopologyDistance(only_enabled={self._only_enabled})"


class WeightDistance:
    """Average absolute weight difference over shared (matching) gene pairs.

    Returns 0.0 when the two genomes share no connections.
    """
    def __call__(self, g1: Genome, g2: Genome) -> float:
        def _build(g: Genome):
            return {conn.innovation: conn
                    for src in g.nodes for conn in src.connections
                    if conn.innovation >= 0}
        d1 = _build(g1)
        d2 = _build(g2)
        shared = frozenset(d1) & frozenset(d2)
        if not shared:
            return 0.0
        return sum(abs(d1[k].weight - d2[k].weight) for k in shared) / len(shared)

    def __repr__(self) -> str:  # pragma: no cover
        return "WeightDistance()"


class ActivationDistance:
    """Fraction of non-input nodes whose activation function differs.

    Counts positionally (index-aligned); extra nodes in the larger genome
    all count as mismatches.
    """
    def __call__(self, g1: Genome, g2: Genome) -> float:
        from yane.core.node import NodeType
        a1 = [n.activation for n in g1.nodes if n.node_type != NodeType.INPUT]
        a2 = [n.activation for n in g2.nodes if n.node_type != NodeType.INPUT]
        n = max(len(a1), len(a2))
        if n == 0:
            return 0.0
        diffs = sum(1 for i in range(min(len(a1), len(a2))) if a1[i] != a2[i])
        diffs += abs(len(a1) - len(a2))
        return diffs / n

    def __repr__(self) -> str:  # pragma: no cover
        return "ActivationDistance()"


class ChainMetric:
    """Weighted sum of multiple ``DistanceMetric`` instances.

    Args:
        metrics:  Iterable of distance metric callables.
        weights:  Matching scalar weights (default: 1.0 for each metric).

    Example::

        metric = ChainMetric(
            [TopologyDistance(), WeightDistance()],
            weights=[0.6, 0.4],
        )
    """
    def __init__(self, metrics, weights=None) -> None:
        self._metrics = list(metrics)
        self._weights = list(weights) if weights is not None else [1.0] * len(self._metrics)
        if len(self._weights) != len(self._metrics):
            raise ValueError("len(weights) must equal len(metrics)")

    def __call__(self, g1: Genome, g2: Genome) -> float:
        return sum(w * m(g1, g2) for m, w in zip(self._metrics, self._weights))

    def __repr__(self) -> str:  # pragma: no cover
        return f"ChainMetric({self._metrics!r}, weights={self._weights!r})"


def compatibility(g1: Genome, g2: Genome, only_enabled: bool = False) -> float:
    """NEAT-style compatibility distance δ = c1·E/N + c2·D/N + c3·W̄

    Automatically switches between two implementations based on network size:

    • Python path  (N < _COMPAT_NUMPY_THRESHOLD):
      Frozenset operations + Python loops.  Fast for small NEAT networks.

    • NumPy path   (N ≥ _COMPAT_NUMPY_THRESHOLD):
      Sorted int32 arrays + np.searchsorted + vectorised abs/sum.

    Coefficients:  c1 = 1.0 (excess)   c2 = 1.0 (disjoint)   c3 = 0.4 (W̄)
    """
    def _build(g: Genome):
        return {
            conn.innovation: conn
            for src in g.nodes for conn in src.connections
            if conn.innovation >= 0 and (not only_enabled or conn.enabled)
        }

    if only_enabled:
        g1_innov = _build(g1)
        g2_innov = _build(g2)
        n1_innov = len(g1_innov)
        n2_innov = len(g2_innov)
        max_innov_g1 = max(g1_innov, default=-1)
        max_innov_g2 = max(g2_innov, default=-1)
        g1_keys = frozenset(g1_innov)
        g2_keys = frozenset(g2_innov)
        g1_sorted = None
        g2_sorted = None
    else:
        _cache1 = g1._innov_cache
        if _cache1 is None:
            _d = _build(g1)
            _n = len(_d)
            _sarr = np.array(sorted(_d), dtype=np.int32) if _n >= _COMPAT_NUMPY_THRESHOLD else None
            _cache1 = (_d, max(_d, default=-1), _n, frozenset(_d), _sarr)
            g1._innov_cache = _cache1
        g1_innov, max_innov_g1, n1_innov, g1_keys, g1_sorted = _cache1

        _cache2 = g2._innov_cache
        if _cache2 is None:
            _d = _build(g2)
            _n = len(_d)
            _sarr = np.array(sorted(_d), dtype=np.int32) if _n >= _COMPAT_NUMPY_THRESHOLD else None
            _cache2 = (_d, max(_d, default=-1), _n, frozenset(_d), _sarr)
            g2._innov_cache = _cache2
        g2_innov, max_innov_g2, n2_innov, g2_keys, g2_sorted = _cache2

    if not g1_innov and not g2_innov:
        n1, n2 = len(g1.nodes), len(g2.nodes)
        c1, c2 = g1.connection_count, g2.connection_count
        node_diff = abs(n1 - n2) / ((n1 if n1 >= n2 else n2) or 1)
        conn_diff = abs(c1 - c2) / ((c1 if c1 >= c2 else c2) or 1)
        return (node_diff + conn_diff) / 2.0

    if n1_innov < _COMPAT_NUMPY_THRESHOLD and n2_innov < _COMPAT_NUMPY_THRESHOLD:
        return _compat_python(g1_innov, g2_innov, g1_keys, g2_keys,
                              max_innov_g1, max_innov_g2, n1_innov, n2_innov)
    else:
        return _compat_numpy(g1_innov, g2_innov, g1_sorted, g2_sorted,
                             g1_keys, g2_keys,
                             max_innov_g1, max_innov_g2, n1_innov, n2_innov)


def _compat_python(g1_innov, g2_innov, g1_keys, g2_keys,
                   max1, max2, n1, n2) -> float:
    """Python path — optimised for small networks (N < _COMPAT_NUMPY_THRESHOLD)."""
    if n1 == n2 and g1_keys == g2_keys:
        if n1 == 0:
            return 0.0
        weight_diff_sum = 0.0
        for k in g1_keys:
            d = g1_innov[k].weight - g2_innov[k].weight
            weight_diff_sum += d if d >= 0.0 else -d
        return 0.4 * weight_diff_sum / n1

    smaller_max = max1 if max1 <= max2 else max2
    N = n1 if n1 >= n2 else n2
    if N == 0:
        N = 1

    # Two-pass scan: avoids allocating an intermediate matching_set frozenset.
    # Pass 1 — iterate g1: categorise each innovation as matching/excess/disjoint.
    excess = disjoint = matching = 0
    weight_diff_sum = 0.0
    for k in g1_keys:
        if k in g2_keys:
            matching += 1
            d = g1_innov[k].weight - g2_innov[k].weight
            weight_diff_sum += d if d >= 0.0 else -d
        elif k > smaller_max:
            excess += 1
        else:
            disjoint += 1

    # Pass 2 — iterate g2: count innovations absent from g1.
    for k in g2_keys:
        if k not in g1_keys:
            if k > smaller_max:
                excess += 1
            else:
                disjoint += 1

    W_bar = weight_diff_sum / matching if matching > 0 else 0.0
    return (excess / N) + (disjoint / N) + (0.4 * W_bar)


def _compat_numpy(g1_innov, g2_innov, g1_sorted, g2_sorted,
                  g1_keys, g2_keys,
                  max1, max2, n1, n2) -> float:
    """NumPy path — faster for large networks (N ≥ _COMPAT_NUMPY_THRESHOLD)."""
    if g1_sorted is None:
        g1_sorted = np.array(sorted(g1_keys), dtype=np.int32)
    if g2_sorted is None:
        g2_sorted = np.array(sorted(g2_keys), dtype=np.int32)

    g1_weights = np.fromiter((g1_innov[k].weight for k in g1_sorted),
                              dtype=np.float64, count=n1)
    g2_weights = np.fromiter((g2_innov[k].weight for k in g2_sorted),
                              dtype=np.float64, count=n2)

    if n1 == n2 and np.array_equal(g1_sorted, g2_sorted):
        if n1 == 0:
            return 0.0
        return 0.4 * float(np.abs(g1_weights - g2_weights).mean())

    smaller_max = max1 if max1 <= max2 else max2
    N = n1 if n1 >= n2 else n2
    if N == 0:
        N = 1

    idx = np.searchsorted(g2_sorted, g1_sorted)
    idx_clipped = np.clip(idx, 0, n2 - 1) if n2 > 0 else np.zeros(n1, dtype=np.intp)
    match_mask = (g1_sorted == g2_sorted[idx_clipped]) if n2 > 0 else np.zeros(n1, dtype=bool)

    matching = int(match_mask.sum())
    W_bar = 0.0
    if matching > 0:
        W_bar = float(np.abs(g1_weights[match_mask]
                             - g2_weights[idx_clipped[match_mask]]).sum()) / matching

    g1_excl = g1_sorted[~match_mask]
    if n2 > 0:
        not_in_g1 = np.ones(n2, dtype=bool)
        not_in_g1[idx_clipped[match_mask]] = False
        g2_excl = g2_sorted[not_in_g1]
    else:
        g2_excl = g2_sorted

    excess  = int((g1_excl > smaller_max).sum()) + int((g2_excl > smaller_max).sum())
    disjoint = len(g1_excl) + len(g2_excl) - excess
    return (excess / N) + (disjoint / N) + (0.4 * W_bar)


# ---------------------------------------------------------------------------
# Temporal Speciation — behaviour-based distance via DTW
# ---------------------------------------------------------------------------

def _dtw(traj1: list[list[float]], traj2: list[list[float]]) -> float:
    """Dynamic Time Warping distance between two output trajectories.

    Each trajectory is a list of output vectors (one per timestep).
    The pointwise distance is Euclidean between output vectors.

    Returns 0.0 when both trajectories are identical.

    Parameters
    ----------
    traj1, traj2 :
        Lists of equal-length output vectors.  May have different numbers
        of timesteps.

    Returns
    -------
    float
        DTW distance (≥ 0).  Equal trajectories return 0.
    """
    import math
    n, m = len(traj1), len(traj2)
    if n == 0 or m == 0:
        return 0.0

    def _dist(a: list[float], b: list[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    # DP table (n+1) × (m+1), initialised to inf
    INF = float("inf")
    prev = [INF] * (m + 1)
    prev[0] = 0.0

    for i in range(1, n + 1):
        curr = [INF] * (m + 1)
        for j in range(1, m + 1):
            cost = _dist(traj1[i - 1], traj2[j - 1])
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr

    return prev[m]


def _topology_hash(genome: "Genome") -> int:
    """Cheap structural fingerprint for trajectory cache invalidation.

    Changes whenever a connection is added, removed, enabled/disabled, or
    has its weight modified.  Does NOT require ``_invalidate_topology()``.
    """
    h = len(genome.nodes) * 1000003
    for src in genome.nodes:
        for conn in src.connections:
            if conn.enabled:
                h ^= hash((conn.innovation, round(conn.weight, 6)))
    return h


class TemporalDistance:
    """Behaviour-based speciation distance using Dynamic Time Warping.

    Two genomes are considered similar when they produce similar output
    *trajectories* over ``rollout_len`` timesteps on random inputs.
    Each genome's trajectory is cached by topology hash so identical
    structures are never re-evaluated.

    Implements the :class:`DistanceMetric` protocol; can be combined with
    topology metrics via :class:`ChainMetric`.

    Parameters
    ----------
    n_rollouts :
        Number of independent rollout trials to average over.
    rollout_len :
        Length of each rollout (timesteps per trial).
    time_weight :
        Scaling factor applied to the raw DTW distance.  Use < 1.0 to
        down-weight temporal behaviour relative to topology when combined in a
        ``ChainMetric``.
    seed :
        RNG seed for the random input sequences.  Fixed by default so the
        same inputs are used across all genome comparisons in a run.

    Example
    -------
    ::

        yane.set_compatibility_distance(TemporalDistance(
            n_rollouts=5, rollout_len=20, time_weight=0.5
        ))
        # Or combined with topology:
        from yane.evolution.compatibility import ChainMetric, TopologyDistance
        yane.set_compatibility_distance(ChainMetric(
            [TopologyDistance(), TemporalDistance()],
            weights=[0.5, 0.5],
        ))
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        rollout_len: int = 20,
        time_weight: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.n_rollouts = max(1, n_rollouts)
        self.rollout_len = max(1, rollout_len)
        self.time_weight = float(time_weight)
        self._seed = seed
        # Trajectory cache: topology_hash → averaged trajectory
        self._cache: dict[int, list[list[float]]] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def __call__(self, g1: "Genome", g2: "Genome") -> float:
        traj1 = self._get_trajectory(g1)
        traj2 = self._get_trajectory(g2)
        if not traj1 or not traj2:
            return 0.0
        return self.time_weight * _dtw(traj1, traj2)

    def invalidate_cache(self) -> None:
        """Clear the trajectory cache (e.g., between generations)."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_trajectory(self, genome: "Genome") -> list[list[float]]:
        """Return cached trajectory or compute and cache it."""
        key = _topology_hash(genome)
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]
        self._cache_misses += 1
        traj = self._compute_trajectory(genome)
        self._cache[key] = traj
        return traj

    def _compute_trajectory(self, genome: "Genome") -> list[list[float]]:
        """Run ``n_rollouts`` rollouts and return the mean trajectory."""
        import random
        n_in = len(genome.input_nodes)
        if n_in == 0:
            return []

        rng = random.Random(self._seed)
        # Pre-generate all input sequences so they're identical across genomes
        sequences: list[list[list[float]]] = [
            [[rng.uniform(-1.0, 1.0) for _ in range(n_in)] for _ in range(self.rollout_len)]
            for _ in range(self.n_rollouts)
        ]

        n_out = len(genome.output_nodes)
        summed = [[0.0] * n_out for _ in range(self.rollout_len)]

        for seq in sequences:
            genome.reset()
            for t, inputs in enumerate(seq):
                try:
                    out = genome.forward(inputs)
                    for j, v in enumerate(out[:n_out]):
                        summed[t][j] += float(v) if _is_finite(float(v)) else 0.0
                except Exception:
                    pass  # genome with errors contributes 0

        n = max(1, self.n_rollouts)
        return [[s / n for s in step] for step in summed]


def _is_finite(v: float) -> bool:
    import math
    return math.isfinite(v)
