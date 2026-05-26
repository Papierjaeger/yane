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

import dataclasses
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


# ---------------------------------------------------------------------------
# Fitness Landscape Analyzer — automatic detection and recommendations
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FitnessLandscapeReport:
    """Diagnostic report about the current fitness landscape."""

    sparsity_score: float = 0.0
    """Fraction of genomes with identical raw fitness (0 = all unique)."""

    plateau_fraction: float = 0.0
    """Fraction of genomes within 1 % of median fitness (high = plateau)."""

    skewness: float = 0.0
    """Sample skewness: > 1 = right-skewed (few high), < -1 = left-skewed."""

    cluster_separability: float = 0.0
    """Mean between-species fitness difference / within-species std (0 = no separation)."""

    recommendations: list[str] = dataclasses.field(default_factory=list)
    """Suggested actions based on the analysis."""

    applied_transform: str | None = None
    """Name of the transform that was applied (if auto-mode)."""


class FitnessLandscapeAnalyzer:
    """Analyse the fitness distribution of an evaluated population and
    recommend / apply fitness transformations.

    Usage::

        report = FitnessLandscapeAnalyzer.analyze(evaluated_genomes)
        transform = FitnessLandscapeAnalyzer.recommend_transform(report)
    """

    @staticmethod
    def analyze(evaluated: list) -> FitnessLandscapeReport:
        """Analyse the fitness landscape from a list of evaluated genomes.

        Each genome must have a ``raw_fitness`` or ``fitness`` attribute.
        """
        if not evaluated:
            return FitnessLandscapeReport()

        fitnesses = [
            float(getattr(g, "raw_fitness", getattr(g, "fitness", 0.0)))
            for g in evaluated
        ]
        n = len(fitnesses)
        report = FitnessLandscapeReport()

        # --- Sparsity: fraction of identical values ---
        unique = len(set(round(f, 12) for f in fitnesses))
        report.sparsity_score = 1.0 - (unique / n) if n > 0 else 0.0

        # --- Plateau: fraction of values within 1 % of median ---
        sorted_f = sorted(fitnesses)
        median = sorted_f[n // 2] if n > 0 else 0.0
        if abs(median) > 1e-12:
            report.plateau_fraction = sum(
                1 for f in fitnesses if abs(f - median) / abs(median) < 0.01
            ) / n
        else:
            report.plateau_fraction = sum(1 for f in fitnesses if abs(f) < 1e-12) / n

        # --- Skewness ---
        if n >= 3:
            mean = sum(fitnesses) / n
            var = sum((f - mean) ** 2 for f in fitnesses) / (n - 1)
            if var > 1e-12:
                std = math.sqrt(var)
                skew = sum((f - mean) ** 3 for f in fitnesses) / (n * std ** 3)
                report.skewness = skew
            else:
                report.skewness = 0.0

        # --- Cluster separability (requires species info) ---
        species_map: dict = {}
        for g in evaluated:
            sid = getattr(g, "_last_species_id", None)
            species_map.setdefault(sid, []).append(
                float(getattr(g, "raw_fitness", getattr(g, "fitness", 0.0)))
            )
        if len(species_map) >= 2:
            species_means = [sum(v) / len(v) for v in species_map.values()]
            species_stds = []
            for v in species_map.values():
                m = sum(v) / len(v)
                species_stds.append(math.sqrt(sum((f - m) ** 2 for f in v) / len(v)) if len(v) > 1 else 0.0)
            mean_std = sum(species_stds) / len(species_stds) if species_stds else 0.0
            if mean_std > 1e-12:
                report.cluster_separability = (
                    max(species_means) - min(species_means)
                ) / mean_std

        # --- Recommendations ---
        if abs(report.skewness) > 1.5:
            report.recommendations.append("apply RankTransform")
        if report.plateau_fraction > 0.5:
            report.recommendations.append("apply SigmaScaling")
        if report.sparsity_score > 0.8:
            report.recommendations.append("inject diversity")
        if report.cluster_separability < 1.0 and len(species_map) >= 2:
            report.recommendations.append("improve speciation separation")

        return report

    @staticmethod
    def recommend_transform(report: FitnessLandscapeReport) -> Callable | None:
        """Return a recommended FitnessTransform based on the report, or None."""
        if abs(report.skewness) > 1.5:
            return RankTransform()
        if report.plateau_fraction > 0.5:
            return SigmaScaling()
        return None
