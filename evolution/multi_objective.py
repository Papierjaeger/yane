"""Pareto helpers for multi-objective fitness values."""
from __future__ import annotations

from collections.abc import Sequence


def as_objectives(value) -> tuple[float, ...] | None:
    """Return *value* as an objective tuple, or None for scalar fitness."""
    if isinstance(value, (str, bytes)):
        return None
    if isinstance(value, Sequence):
        return tuple(float(v) for v in value)
    return None


def dominates(
    a: Sequence[float],
    b: Sequence[float],
    maximize: Sequence[bool] | None = None,
) -> bool:
    """Return True when objective vector *a* Pareto-dominates *b*."""
    if len(a) != len(b):
        raise ValueError("Objective vectors must have the same length")
    if maximize is None:
        maximize = (True,) * len(a)
    if len(maximize) != len(a):
        raise ValueError("maximize must have the same length as objectives")

    better_or_equal = True
    strictly_better = False
    for av, bv, max_this in zip(a, b, maximize):
        if max_this:
            if av < bv:
                better_or_equal = False
                break
            if av > bv:
                strictly_better = True
        else:
            if av > bv:
                better_or_equal = False
                break
            if av < bv:
                strictly_better = True
    return better_or_equal and strictly_better


def non_dominated_sort(
    objectives: Sequence[Sequence[float]],
    maximize: Sequence[bool] | None = None,
) -> list[list[int]]:
    """Return NSGA-II-style fronts as lists of objective indices."""
    n = len(objectives)
    dominated_by_count = [0] * n
    dominates_list: list[list[int]] = [[] for _ in range(n)]
    fronts: list[list[int]] = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(objectives[i], objectives[j], maximize):
                dominates_list[i].append(j)
                dominated_by_count[j] += 1
            elif dominates(objectives[j], objectives[i], maximize):
                dominates_list[j].append(i)
                dominated_by_count[i] += 1
        if dominated_by_count[i] == 0:
            fronts[0].append(i)

    rank = 0
    while rank < len(fronts) and fronts[rank]:
        next_front: list[int] = []
        for i in fronts[rank]:
            for j in dominates_list[i]:
                dominated_by_count[j] -= 1
                if dominated_by_count[j] == 0:
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        rank += 1
    return fronts


def crowding_distance(
    objectives: Sequence[Sequence[float]],
    front: Sequence[int],
    maximize: Sequence[bool] | None = None,
) -> dict[int, float]:
    """Return NSGA-II crowding distances for one front."""
    if not front:
        return {}
    n_obj = len(objectives[front[0]])
    distances = {idx: 0.0 for idx in front}
    if len(front) <= 2:
        for idx in front:
            distances[idx] = float("inf")
        return distances

    if maximize is None:
        maximize = (True,) * n_obj
    for obj_i in range(n_obj):
        ordered = sorted(front, key=lambda idx: objectives[idx][obj_i])
        if maximize[obj_i]:
            ordered = list(reversed(ordered))
        distances[ordered[0]] = float("inf")
        distances[ordered[-1]] = float("inf")
        vals = [objectives[idx][obj_i] for idx in ordered]
        span = max(vals) - min(vals)
        if span <= 1e-12:
            continue
        for pos in range(1, len(ordered) - 1):
            prev_v = objectives[ordered[pos - 1]][obj_i]
            next_v = objectives[ordered[pos + 1]][obj_i]
            distances[ordered[pos]] += abs(next_v - prev_v) / span
    return distances


def pareto_scores(
    objectives: Sequence[Sequence[float]],
    maximize: Sequence[bool] | None = None,
) -> list[float]:
    """Convert objective vectors into scalar selection scores.

    Front rank dominates the score; crowding distance breaks ties inside a front
    so sparse parts of the Pareto front remain represented.
    """
    n = len(objectives)
    if n == 0:
        return []
    scores = [0.0] * n
    fronts = non_dominated_sort(objectives, maximize)
    for rank, front in enumerate(fronts):
        distances = crowding_distance(objectives, front, maximize)
        base = len(fronts) - rank
        finite = [d for d in distances.values() if d != float("inf")]
        scale = max(finite, default=1.0) or 1.0
        for idx in front:
            d = distances[idx]
            tie = 1.0 if d == float("inf") else d / scale
            scores[idx] = base + min(1.0, tie)
    return scores
