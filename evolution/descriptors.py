"""Reusable behavior/topology descriptor and fitness component registry."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from yane.core.genome import Genome


DescriptorFn = Callable[[Genome], tuple[float, ...]]
MetricFn = Callable[[Genome], float]


@dataclass
class FitnessComponent:
    name: str
    fn: MetricFn
    weight: float = 1.0
    maximize: bool = True


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


DEFAULT_DESCRIPTORS = DescriptorRegistry()
DEFAULT_DESCRIPTORS.register("topology", topology_descriptor)
DEFAULT_DESCRIPTORS.register("timing", timing_descriptor)
DEFAULT_DESCRIPTORS.register("behavior", behavior_descriptor())
