"""NEAT-style compatibility distance between two genomes.

Automatically dispatches to a fast Python path (small networks) or a
vectorised NumPy path (large networks) based on connection count.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from yane.core.genome import Genome

# Below this connection count per genome, pure-Python frozenset operations are
# faster than NumPy due to array-creation overhead (~13 μs fixed cost).
# Above it, NumPy's vectorised searchsorted + abs win (crossover measured ~200).
_COMPAT_NUMPY_THRESHOLD: int = 200


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
