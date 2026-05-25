"""Pluggable fitness transformation layer for YANE.

A FitnessTransform is any callable:
    (fitnesses: list[float]) -> list[float]

Built-in transforms are provided as classes with a ``name`` attribute.
Combine them with ChainTransform or compose manually.

Usage::

    ne = NeuroEvolution()
    ne.set_fitness_transform(RankTransform())
    ne.set_fitness_transform(ChainTransform([RankTransform(), SigmaScaling()]))
    ne.set_fitness_transform(None)   # remove transform
"""
from __future__ import annotations

import math
from typing import Callable


class RankTransform:
    """Replace raw fitness values with rank / N (result in (0, 1])."""
    name = "rank"

    def __call__(self, fitnesses: list[float]) -> list[float]:
        n = len(fitnesses)
        if n == 0:
            return fitnesses
        order = sorted(range(n), key=lambda i: fitnesses[i])
        result = [0.0] * n
        for rank, idx in enumerate(order, 1):
            result[idx] = rank / n
        return result


class SigmaScaling:
    """Scale by (f - mean) / (2 * std); clip result to [0, max_scale]."""
    name = "sigma_scaling"

    def __init__(self, max_scale: float = 2.0) -> None:
        self.max_scale = max_scale

    def __call__(self, fitnesses: list[float]) -> list[float]:
        n = len(fitnesses)
        if n < 2:
            return fitnesses
        mean = sum(fitnesses) / n
        var = sum((f - mean) ** 2 for f in fitnesses) / n
        std = math.sqrt(var)
        if std < 1e-12:
            return [1.0] * n
        return [
            max(0.0, min(self.max_scale, 1.0 + (f - mean) / (2.0 * std)))
            for f in fitnesses
        ]


class LinearNormalize:
    """Linearly normalize fitness values to [lo, hi]."""
    name = "linear_normalize"

    def __init__(self, lo: float = 0.0, hi: float = 1.0) -> None:
        self.lo = lo
        self.hi = hi

    def __call__(self, fitnesses: list[float]) -> list[float]:
        if not fitnesses:
            return fitnesses
        f_min = min(fitnesses)
        f_max = max(fitnesses)
        span = f_max - f_min
        if span < 1e-12:
            return [self.lo] * len(fitnesses)
        r = self.hi - self.lo
        return [self.lo + r * (f - f_min) / span for f in fitnesses]


class ClipTransform:
    """Clip fitness values to [min_val, max_val]."""
    name = "clip"

    def __init__(self, min_val: float, max_val: float) -> None:
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, fitnesses: list[float]) -> list[float]:
        lo, hi = self.min_val, self.max_val
        return [max(lo, min(hi, f)) for f in fitnesses]


class ChainTransform:
    """Apply transforms in sequence (left to right)."""
    name = "chain"

    def __init__(self, transforms: list[Callable]) -> None:
        self.transforms = list(transforms)

    def __call__(self, fitnesses: list[float]) -> list[float]:
        result = fitnesses
        for t in self.transforms:
            result = t(result)
        return result
