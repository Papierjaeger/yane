"""Reusable behavior/topology descriptor and fitness component registry."""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

from yane.core.genome import Genome

if TYPE_CHECKING:
    from yane.evolution.population import Population


DescriptorFn = Callable[[Genome], tuple[float, ...]]
MetricFn = Callable[[Genome], float]


@dataclass
class FitnessComponent:
    name: str
    fn: MetricFn
    weight: float = 1.0
    maximize: bool = True


@dataclass
class FitnessWeightHistoryEntry:
    generation: int
    stagnation_count: int
    plateau_ratio: float
    reason: str
    weights: dict[str, float]
    component_variance: dict[str, float]
    contribution_share: dict[str, float]


class DescriptorRegistry:
    def __init__(self) -> None:
        self._descriptors: dict[str, DescriptorFn] = {}

    def register(self, name: str, fn: DescriptorFn) -> None:
        self._descriptors[name] = fn

    def get(self, name: str) -> DescriptorFn:
        return self._descriptors[name]

    def names(self) -> list[str]:
        return sorted(self._descriptors)


def topology_descriptor(genome: Genome) -> tuple[float, float]:
    hidden = max(0, len(genome.nodes) - len(genome.input_nodes) - len(genome.output_nodes))
    return float(hidden), float(genome.connection_count)


def timing_descriptor(genome: Genome) -> tuple[float, float]:
    return float(genome.eval_time_ms or 0.0), float(genome.efficiency_score)


def behavior_descriptor(n_probes: int = 4, lo: float = -1.0, hi: float = 1.0, seed: int = 42) -> DescriptorFn:
    rng = random.Random(seed)
    probes_cache: dict[int, list[list[float]]] = {}

    def _descriptor(genome: Genome) -> tuple[float, ...]:
        n_inputs = len(genome.input_nodes)
        probes = probes_cache.setdefault(
            n_inputs,
            [[rng.uniform(lo, hi) for _ in range(n_inputs)] for _ in range(n_probes)],
        )
        values: list[float] = []
        for probe in probes:
            genome.reset()
            values.extend(float(v) for v in genome.forward(probe))
        return tuple(values)

    return _descriptor


def scalarize_components(genome: Genome, components: list[FitnessComponent]) -> tuple[float, tuple[float, ...]]:
    values = tuple(component.fn(genome) for component in components)
    score = 0.0
    for component, value in zip(components, values):
        signed = value if component.maximize else -value
        score += component.weight * signed
    return score, values


class AdaptiveFitnessComponentWeights:
    """Adaptive scalarizer for auxiliary fitness/descriptor components.

    The task fitness remains external; this helper only manages optional
    component weights and records how they evolve. During stagnation it nudges
    weight toward components that still vary in the population and away from
    components that dominate the scalarization, while keeping every component
    above ``collapse_floor`` of the total weight.
    """

    def __init__(
        self,
        components: list[FitnessComponent],
        *,
        mode: str = "fixed",
        min_weight: float = 0.0,
        max_weight: float = 5.0,
        adaptation_rate: float = 0.25,
        collapse_floor: float = 0.05,
        history_max: int = 200,
    ) -> None:
        if mode not in {"fixed", "adaptive"}:
            raise ValueError("mode must be 'fixed' or 'adaptive'")
        if not components:
            raise ValueError("at least one fitness component is required")
        self.components = list(components)
        self.mode = mode
        self.min_weight = float(min_weight)
        self.max_weight = max(self.min_weight, float(max_weight))
        self.adaptation_rate = max(0.0, min(1.0, float(adaptation_rate)))
        self.collapse_floor = max(0.0, min(1.0, float(collapse_floor)))
        self.history_max = max(1, int(history_max))
        self.generation = 0
        self.last_reason = "init"
        self.history: list[FitnessWeightHistoryEntry] = []
        self._initial_total = sum(c.weight for c in self.components if c.weight > 0.0) or float(len(self.components))
        self._clamp_and_protect()
        self._record(0, 0.0, "init", {}, {})

    def scalarize(self, genome: Genome) -> tuple[float, tuple[float, ...]]:
        score, values = scalarize_components(genome, self.components)
        genome.fitness_component_values = values
        genome.fitness_component_weights = self.weights
        return score, values

    @property
    def weights(self) -> dict[str, float]:
        return {c.name: float(c.weight) for c in self.components}

    def tick(self, population: "Population") -> None:
        self.generation += 1
        threshold = max(1, population.stagnation_threshold)
        plateau = min(1.0, max(0.0, population.stagnation_count / threshold))
        variance, contribution = self._population_component_stats(population)

        if self.mode != "adaptive" or plateau <= 0.0:
            reason = "fixed" if self.mode == "fixed" else "adaptive:no_stagnation"
            self.last_reason = reason
            self._record(population.stagnation_count, plateau, reason, variance, contribution)
            return

        n = len(self.components)
        total_var = sum(variance.values())
        mean_share = 1.0 / n
        for component in self.components:
            var_share = variance.get(component.name, 0.0) / total_var if total_var > 0.0 else mean_share
            current_share = contribution.get(component.name, 0.0)
            target = component.weight
            if current_share > mean_share * 1.5:
                target *= max(0.5, 1.0 - self.adaptation_rate * plateau)
            if var_share > mean_share:
                target *= 1.0 + self.adaptation_rate * plateau * (var_share / mean_share - 1.0)
            component.weight = component.weight * (1.0 - self.adaptation_rate) + target * self.adaptation_rate

        self._clamp_and_protect()
        self.last_reason = "adaptive:stagnation"
        self._record(population.stagnation_count, plateau, self.last_reason, variance, contribution)

    def get_diagnostics(self) -> dict:
        return {
            "mode": self.mode,
            "generation": self.generation,
            "last_reason": self.last_reason,
            "weights": self.weights,
            "history": [
                {
                    "generation": h.generation,
                    "stagnation_count": h.stagnation_count,
                    "plateau_ratio": h.plateau_ratio,
                    "reason": h.reason,
                    "weights": dict(h.weights),
                    "component_variance": dict(h.component_variance),
                    "contribution_share": dict(h.contribution_share),
                }
                for h in self.history
            ],
        }

    def _population_component_stats(self, population: "Population") -> tuple[dict[str, float], dict[str, float]]:
        values_by_name: dict[str, list[float]] = {c.name: [] for c in self.components}
        contributions = {c.name: 0.0 for c in self.components}
        for genome in population._evaluated:
            values = getattr(genome, "fitness_component_values", None)
            if values is None or len(values) != len(self.components):
                try:
                    _, values = self.scalarize(genome)
                except Exception:
                    continue
            for component, value in zip(self.components, values):
                if not math.isfinite(float(value)):
                    continue
                signed = float(value) if component.maximize else -float(value)
                values_by_name[component.name].append(signed)
                contributions[component.name] += abs(component.weight * signed)

        variance = {
            name: (statistics.pvariance(vals) if len(vals) >= 2 else 0.0)
            for name, vals in values_by_name.items()
        }
        total_contrib = sum(contributions.values())
        share = {
            name: (value / total_contrib if total_contrib > 0.0 else 0.0)
            for name, value in contributions.items()
        }
        return variance, share

    def _clamp_and_protect(self) -> None:
        for component in self.components:
            component.weight = max(self.min_weight, min(self.max_weight, float(component.weight)))
        if (
            len(self.components) <= 1
            or self.collapse_floor <= 0.0
            or self.min_weight < 0.0
            or any(component.weight < 0.0 for component in self.components)
        ):
            return
        total = sum(max(0.0, c.weight) for c in self.components)
        if total <= 0.0:
            for component in self.components:
                component.weight = min(self.max_weight, max(self.min_weight, 1.0))
            total = sum(c.weight for c in self.components)
        floor = total * self.collapse_floor
        for component in self.components:
            component.weight = max(component.weight, floor)
        total_after = sum(c.weight for c in self.components)
        if total_after > 0.0:
            scale = self._initial_total / total_after
            for component in self.components:
                component.weight = max(self.min_weight, min(self.max_weight, component.weight * scale))

    def _record(
        self,
        stagnation_count: int,
        plateau_ratio: float,
        reason: str,
        variance: dict[str, float],
        contribution: dict[str, float],
    ) -> None:
        self.history.append(
            FitnessWeightHistoryEntry(
                generation=self.generation,
                stagnation_count=int(stagnation_count),
                plateau_ratio=float(plateau_ratio),
                reason=reason,
                weights=self.weights,
                component_variance=dict(variance),
                contribution_share=dict(contribution),
            )
        )
        if len(self.history) > self.history_max:
            self.history = self.history[-self.history_max:]


DEFAULT_DESCRIPTORS = DescriptorRegistry()
DEFAULT_DESCRIPTORS.register("topology", topology_descriptor)
DEFAULT_DESCRIPTORS.register("timing", timing_descriptor)
DEFAULT_DESCRIPTORS.register("behavior", behavior_descriptor())
