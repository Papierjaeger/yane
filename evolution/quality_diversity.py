"""Quality Diversity / MAP-Elites archive helpers."""
from __future__ import annotations

import random
from dataclasses import dataclass
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome


@dataclass
class EliteCell:
    descriptor: tuple[float, ...]
    fitness: float
    genome: "Genome"


class MAPElitesArchive:
    """Grid archive that keeps the best genome per behavior cell."""

    def __init__(
        self,
        bins: Sequence[int],
        ranges: Sequence[tuple[float, float]],
        max_cells: int | None = None,
    ) -> None:
        if len(bins) != len(ranges):
            raise ValueError("bins and ranges must have the same length")
        if not bins:
            raise ValueError("MAP-Elites requires at least one descriptor dimension")
        self.bins = tuple(int(b) for b in bins)
        self.ranges = tuple((float(lo), float(hi)) for lo, hi in ranges)
        if any(b < 1 for b in self.bins):
            raise ValueError("all bins must be >= 1")
        for lo, hi in self.ranges:
            if hi <= lo:
                raise ValueError("each descriptor range must have hi > lo")
        self.max_cells = max_cells
        self.cells: dict[tuple[int, ...], EliteCell] = {}
        self.n_additions: int = 0
        self.n_replacements: int = 0

    def cell_for(self, descriptor: Sequence[float]) -> tuple[int, ...]:
        if len(descriptor) != len(self.bins):
            raise ValueError("descriptor dimension does not match archive")
        idxs: list[int] = []
        for value, n_bins, (lo, hi) in zip(descriptor, self.bins, self.ranges):
            if value <= lo:
                idxs.append(0)
            elif value >= hi:
                idxs.append(n_bins - 1)
            else:
                rel = (float(value) - lo) / (hi - lo)
                idxs.append(min(n_bins - 1, int(rel * n_bins)))
        return tuple(idxs)

    def add(self, descriptor: Sequence[float], genome: "Genome", fitness: float) -> bool:
        cell = self.cell_for(descriptor)
        desc = tuple(float(v) for v in descriptor)
        current = self.cells.get(cell)
        if current is not None and current.fitness >= fitness:
            return False
        self.cells[cell] = EliteCell(desc, fitness, genome.copy())
        if current is None:
            self.n_additions += 1
        else:
            self.n_replacements += 1
        self._trim_if_needed()
        return True

    def sample_elite(self) -> "Genome | None":
        if not self.cells:
            return None
        return random.choice(tuple(self.cells.values())).genome.copy()

    @property
    def coverage(self) -> float:
        total = 1
        for b in self.bins:
            total *= b
        return len(self.cells) / total

    def _trim_if_needed(self) -> None:
        if self.max_cells is None or len(self.cells) <= self.max_cells:
            return
        ordered = sorted(self.cells.items(), key=lambda item: item[1].fitness)
        for key, _cell in ordered[:len(self.cells) - self.max_cells]:
            self.cells.pop(key, None)


def descriptor_from_outputs(
    probes: Sequence[Sequence[float]],
) -> Callable[["Genome"], tuple[float, ...]]:
    """Build a simple descriptor function from outputs on fixed probes."""
    def _descriptor(genome: "Genome") -> tuple[float, ...]:
        rows: list[float] = []
        for inp in probes:
            genome.reset()
            rows.extend(float(v) for v in genome.forward(list(inp)))
        return tuple(rows)
    return _descriptor
