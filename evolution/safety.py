"""Safety-Constrained Evolution (Safe NEAT).

Wraps fitness evaluation with safety constraints to enforce hard/soft/barrier
limits on genome behavior.

Usage::

    from yane.evolution.safety import SafetyConstraint, SafetySystem

    constraints = [
        SafetyConstraint("velocity", lambda g: g.max_vel < 1.0, mode="hard", penalty=-1000.0),
        SafetyConstraint("energy",   lambda g: g.energy < 10.0, mode="soft", penalty=0.5),
    ]
    system = SafetySystem(constraints, min_safe_frac=0.2)

    safe_fitness = system.evaluate(genome, raw_fitness)

Integration with NeuroEvolution::

    ne.set_safety_constraints(constraints, min_safe_frac=0.2)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from yane.core.genome import Genome


@dataclass
class SafetyConstraint:
    """Defines a single safety constraint for evolved genomes.

    Parameters
    ----------
    name:
        Human-readable constraint name (used in diagnostics).
    check:
        Callable ``(genome) -> bool``.  Returns *True* when the constraint
        is **satisfied** (safe), *False* when violated.
    mode:
        ``"hard"``    — violation immediately sets fitness to *penalty*.
        ``"soft"``    — fitness is reduced proportionally to the violation
                        rate (``fitness *= (1 - penalty_frac)``).
        ``"barrier"`` — logarithmic barrier: fitness reduced by
                        ``-log(1 - violation_frac) * penalty`` when close to
                        the boundary.  When fully violated the full penalty
                        is applied.
    penalty:
        For ``"hard"``: the replacement fitness value (e.g. ``-1000.0``).
        For ``"soft"``: fraction by which fitness is reduced (e.g. ``0.5``
        means 50 % reduction).
        For ``"barrier"``: barrier weight multiplier.
    """
    name: str
    check: Callable  # (Genome) -> bool: True = safe, False = violated
    mode: str = "hard"   # "hard" | "soft" | "barrier"
    penalty: float = -1e6


class ConstraintViolation(Exception):
    """Raised internally by SafetySystem when a hard constraint is violated."""
    def __init__(self, constraint: SafetyConstraint):
        self.constraint = constraint
        super().__init__(f"Hard constraint violated: {constraint.name}")


class SafetySystem:
    """Manages a collection of safety constraints and applies them to fitness.

    Parameters
    ----------
    constraints:
        List of :class:`SafetyConstraint` to enforce.
    min_safe_frac:
        Minimum fraction of the current population that must be "safe"
        (no hard-constraint violations).  When fewer than this fraction are
        safe, the best safe genomes are shielded from selection pressure so
        they cannot be displaced.  Used externally by
        ``NeuroEvolution._apply_safe_frac_protection()``.
    eval_budget:
        Maximum constraint evaluations per genome call (unused; future use).
    """

    def __init__(
        self,
        constraints: "list[SafetyConstraint]",
        min_safe_frac: float = 0.0,
    ) -> None:
        self.constraints = list(constraints)
        self.min_safe_frac = float(min_safe_frac)
        # Diagnostics
        self.n_hard_violations: int = 0
        self.n_soft_violations: int = 0
        self.n_evaluations: int = 0

    def is_safe(self, genome: "Genome") -> bool:
        """Return True iff the genome satisfies ALL hard constraints."""
        for c in self.constraints:
            if c.mode == "hard":
                try:
                    if not c.check(genome):
                        return False
                except Exception:
                    return False
        return True

    def evaluate(self, genome: "Genome", raw_fitness: float) -> float:
        """Apply all constraints and return the adjusted fitness.

        Parameters
        ----------
        genome:
            The genome to evaluate.
        raw_fitness:
            The task fitness before constraint adjustment.

        Returns
        -------
        float
            Adjusted fitness.  For hard violations this is ``constraint.penalty``.
        """
        self.n_evaluations += 1
        fitness = raw_fitness

        for c in self.constraints:
            try:
                satisfied = bool(c.check(genome))
            except Exception:
                satisfied = False

            if satisfied:
                continue  # constraint met — no penalty

            if c.mode == "hard":
                self.n_hard_violations += 1
                return float(c.penalty)

            elif c.mode == "soft":
                self.n_soft_violations += 1
                # Reduce fitness by frac of its absolute value so the penalty
                # always worsens fitness regardless of whether it is positive
                # or negative (fitness * (1 - frac) would *improve* negative
                # fitness, which is wrong for a maximisation objective).
                frac = float(c.penalty)
                frac = max(0.0, min(1.0, frac))
                fitness -= abs(fitness) * frac

            elif c.mode == "barrier":
                self.n_soft_violations += 1
                # Logarithmic barrier: subtract log(1 + |penalty|) from fitness.
                # Always worsens fitness regardless of sign.
                barrier_weight = abs(float(c.penalty))
                fitness -= math.log1p(barrier_weight)

        return fitness

    def wrap_fitness_fn(
        self,
        fitness_fn: Callable,
    ) -> Callable:
        """Return a wrapped fitness function that applies constraints.

        The wrapped function has the same signature as *fitness_fn* but
        applies all constraints to the raw fitness before returning.

        Parameters
        ----------
        fitness_fn:
            Original fitness function ``(genome) -> float``.
        """
        system = self

        def _safe_fitness(genome: "Genome") -> float:
            raw = fitness_fn(genome)
            return system.evaluate(genome, raw)

        _safe_fitness.__name__ = getattr(fitness_fn, "__name__", "safe_fitness")
        return _safe_fitness

    def protect_safe_fraction(
        self,
        population_genomes: "list[Genome]",
    ) -> "list[Genome]":
        """Identify safe genomes that should be protected from displacement.

        Returns the list of safe genomes (those satisfying all hard
        constraints).  When ``min_safe_frac > 0`` and fewer than
        ``min_safe_frac * len(population_genomes)`` are safe, this method
        returns the complete safe set so callers can apply elitism or boost
        fitness.

        Parameters
        ----------
        population_genomes:
            All genomes in the current population.

        Returns
        -------
        list[Genome]
            Genomes that are safe.  Empty if no safe genomes or no hard
            constraints exist.
        """
        if not self.constraints:
            return []
        safe = [g for g in population_genomes if self.is_safe(g)]
        return safe

    def safe_fraction(self, population_genomes: "list[Genome]") -> float:
        """Return fraction of *population_genomes* that satisfies all hard constraints."""
        if not population_genomes:
            return 1.0
        safe = sum(1 for g in population_genomes if self.is_safe(g))
        return safe / len(population_genomes)
