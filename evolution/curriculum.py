"""Curriculum Learning support for YANE.

Usage::

    from yane.evolution.curriculum import CurriculumStage
    import yane

    def easy(genome):   return ...
    def hard(genome):   return ...

    ne = yane.NeuroEvolution()
    ne.configure(2, 1)
    ne.set_curriculum([
        CurriculumStage(easy, target_fitness=0.9, name="easy"),
        CurriculumStage(hard, target_fitness=0.95, name="hard"),
    ])
    ne.train()

Note: generator-based fitness functions (early-stopping protocol) are not
supported inside curriculum stages.  Use regular functions that return float.
"""
from __future__ import annotations
import math
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome


class CurriculumStage:
    """One stage of a curriculum.

    Args:
        fitness_fn: Regular callable ``genome -> float``.  Generator functions
            are not supported (early stopping is silently bypassed).
        target_fitness: Fitness threshold that triggers advancement to the next
            stage.  ``None`` on the last stage means "run until an external stop
            condition (min_fitness / max_iterations / converged)".
        name: Optional human-readable label for logging and GUI.
    """

    def __init__(
        self,
        fitness_fn: Callable[["Genome"], float],
        target_fitness: float | None = None,
        name: str | None = None,
    ) -> None:
        if not callable(fitness_fn):
            raise TypeError("fitness_fn must be callable")
        self.fitness_fn = fitness_fn
        self.target_fitness = target_fitness
        self.name = name or ""

    def __repr__(self) -> str:
        return (
            f"CurriculumStage(name={self.name!r}, "
            f"target_fitness={self.target_fitness})"
        )


class Curriculum:
    """Manages stage progression for curriculum-based training.

    Acts as a callable fitness function so it can be passed directly to
    ``EvaluationRunner.run()``.  Internally routes to the current stage's
    fitness function and tracks the best fitness seen on that stage.
    """

    def __init__(self, stages: list[CurriculumStage]) -> None:
        if not stages:
            raise ValueError("Curriculum requires at least one stage")
        self.stages: list[CurriculumStage] = list(stages)
        self._current: int = 0
        self.n_advances: int = 0
        self._stage_best: float = -math.inf
        self._stage_evals: int = 0

    # ------------------------------------------------------------------
    # Callable protocol — used as fitness_fn inside EvaluationRunner
    # ------------------------------------------------------------------

    def __call__(self, genome: "Genome") -> float:
        return self.stages[self._current].fitness_fn(genome)

    # ------------------------------------------------------------------
    # Stage management
    # ------------------------------------------------------------------

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self._current]

    @property
    def stage_index(self) -> int:
        return self._current

    @property
    def stage_count(self) -> int:
        return len(self.stages)

    @property
    def is_last_stage(self) -> bool:
        return self._current >= len(self.stages) - 1

    def record_fitness(self, fitness: float) -> None:
        """Update the stage-local best with a finalized fitness value."""
        self._stage_evals += 1
        if math.isfinite(fitness) and fitness > self._stage_best:
            self._stage_best = fitness

    def maybe_advance(self) -> bool:
        """Advance to the next stage if the current target is reached.

        Returns True if a stage transition occurred.
        """
        if self.is_last_stage:
            return False
        target = self.stages[self._current].target_fitness
        if target is None:
            return False
        if self._stage_best >= target:
            self._current += 1
            self.n_advances += 1
            self._stage_best = -math.inf
            self._stage_evals = 0
            return True
        return False

    def is_complete(self) -> bool:
        """Return True if we are on the last stage and its target is met."""
        if not self.is_last_stage:
            return False
        target = self.stages[self._current].target_fitness
        return target is not None and self._stage_best >= target

    def info(self) -> dict:
        """Snapshot suitable for GUI / diagnostics."""
        stage = self.stages[self._current]
        return {
            "curriculum_stage_index": self._current,
            "curriculum_stage_count": len(self.stages),
            "curriculum_stage_name": stage.name,
            "curriculum_stage_target": stage.target_fitness,
            "curriculum_stage_best": self._stage_best if math.isfinite(self._stage_best) else None,
            "curriculum_stage_evals": self._stage_evals,
            "curriculum_n_advances": self.n_advances,
        }
