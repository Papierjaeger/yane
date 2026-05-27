"""Helpers for competitive coevolution experiments."""
from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome


@dataclass
class OpponentRecord:
    genome: "Genome"
    fitness: float
    age: int = 0


class HallOfFame:
    """Stores strong historical opponents to reduce cyclic forgetting."""

    def __init__(self, max_size: int = 50) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.max_size = max_size
        self._records: list[OpponentRecord] = []

    def add(self, genome: "Genome", fitness: float) -> None:
        self._records.append(OpponentRecord(genome.copy(), float(fitness)))
        self._records.sort(key=lambda r: r.fitness, reverse=True)
        del self._records[self.max_size:]

    def sample(self, k: int) -> list["Genome"]:
        if k <= 0 or not self._records:
            return []
        records = random.sample(self._records, min(k, len(self._records)))
        return [r.genome.copy() for r in records]

    def tick(self) -> None:
        for record in self._records:
            record.age += 1

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> tuple[OpponentRecord, ...]:
        return tuple(self._records)


def competitive_fitness(
    genome: "Genome",
    opponents: Sequence["Genome"],
    match_fn: Callable[["Genome", "Genome"], float],
    aggregation: str = "mean",
) -> float:
    """Evaluate *genome* against opponents using a match callback.

    ``match_fn(genome, opponent)`` returns the candidate's score for one match.
    Aggregation can be ``mean``, ``min`` or ``max``.
    """
    if not opponents:
        return 0.0
    scores = [float(match_fn(genome, opponent)) for opponent in opponents]
    if aggregation == "min":
        return min(scores)
    if aggregation == "max":
        return max(scores)
    if aggregation != "mean":
        raise ValueError("aggregation must be 'mean', 'min', or 'max'")
    return sum(scores) / len(scores)


def mixed_opponents(
    current_population: Sequence["Genome"],
    hall_of_fame: HallOfFame,
    k_current: int = 3,
    k_hof: int = 3,
) -> list["Genome"]:
    """Sample opponents from current candidates and historical elites."""
    opponents: list[Genome] = []
    if k_current > 0 and current_population:
        picked = random.sample(list(current_population), min(k_current, len(current_population)))
        opponents.extend(g.copy() for g in picked)
    opponents.extend(hall_of_fame.sample(k_hof))
    return opponents
