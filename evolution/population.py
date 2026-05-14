from __future__ import annotations
import math
import random

from yane.core.genome import Genome


class Population:
    def __init__(self, max_size: int = 100, initial_genome: Genome | None = None) -> None:
        self.max_size = max_size
        self._unevaluated: set[Genome] = {initial_genome or Genome()}
        self._evaluated: list[Genome] = []

    def select_for_evaluation(self) -> Genome:
        if not self._unevaluated:
            self._spawn_offspring()
        return next(iter(self._unevaluated))

    def submit(self, genome: Genome, fitness: float) -> None:
        if genome not in self._unevaluated:
            return  # already submitted or unknown genome — ignore
        genome.fitness = fitness
        self._unevaluated.discard(genome)
        self._evaluated.append(genome)
        self._prune()

    def get_best(self) -> Genome:
        if not self._evaluated:
            raise RuntimeError("No evaluated genomes yet.")
        return max(self._evaluated, key=lambda g: g.fitness)

    @property
    def size(self) -> int:
        return len(self._evaluated) + len(self._unevaluated)

    @property
    def evaluated_count(self) -> int:
        return len(self._evaluated)

    @property
    def unevaluated_count(self) -> int:
        return len(self._unevaluated)

    def _spawn_offspring(self) -> None:
        if not self._evaluated:
            self._unevaluated.add(Genome())
            return
        k = max(2, math.ceil(len(self._evaluated) * 0.1))
        parent = self._tournament_select(k)
        child = parent.copy()
        child.mutate()
        self._unevaluated.add(child)

    def _tournament_select(self, k: int) -> Genome:
        candidates = random.sample(self._evaluated, min(k, len(self._evaluated)))
        return max(candidates, key=lambda g: g.fitness)

    def _prune(self) -> None:
        # Remove worst evaluated genomes until total size is within max_size.
        # _unevaluated is bounded naturally (one genome per spawn cycle).
        while self._evaluated and len(self._evaluated) + len(self._unevaluated) > self.max_size:
            worst = min(self._evaluated, key=lambda g: g.fitness)
            self._evaluated.remove(worst)
