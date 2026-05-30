"""Open-Ended Evolution / Minimal Criterion für YANE.

Filtert Genome vor der Selektion: Nur Genome, die ein Mindest-Kriterium
erfüllen, dürfen sich fortpflanzen.  Wenn zu wenige Genome viable sind,
wird der Schwellwert adaptiv gelockert.

**Minimal Criterion:**
Ein Callable ``(genome) -> bool`` entscheidet ob ein Genom "viable" ist.
Nicht-viable Genome erhalten eine Strafe statt ihrer normalen Fitness.

**Adaptive Lockerung:**
Wenn der Anteil vibler Genome unter ``min_viable_frac`` fällt, wird
``viable_boost_factor`` auf die nicht-viablen Genome angewendet
(Fitness weniger stark bestraft) → adaptive Lockerung.

**Open-Ended Modi:**
- ``novelty_with_criterion``: Novelty-Fitness + Minimal Criterion
- ``curiosity_with_criterion``: Curiosity-Bonus + Minimal Criterion
- ``quality_diversity_with_criterion``: QD-Fitness + Minimal Criterion

Jeder Modus kombiniert die bestehende YANE-Infrastruktur mit dem
Minimal-Criterion-Filter.

Integration::

    yane.set_minimal_criterion(lambda g: g.fitness > -50.0)
    yane.train(fitness_fn)  # non-viable genomes get penalty fitness
"""
from __future__ import annotations

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome

CriterionFn = Callable[["Genome"], bool]


# ---------------------------------------------------------------------------
# MinimalCriterion
# ---------------------------------------------------------------------------

class MinimalCriterion:
    """Wraps a fitness function to enforce a viability criterion.

    Non-viable genomes receive a *penalty* fitness instead of their
    actual fitness, preventing them from being selected as parents.

    When the fraction of viable genomes falls below ``min_viable_frac``,
    the penalty is reduced (adaptive relaxation), keeping the search alive.

    Parameters
    ----------
    criterion_fn :
        ``(genome) -> bool`` — True = viable, False = not viable.
    min_viable_frac :
        Minimum fraction of the population that must be viable.
        If ``viable_frac < min_viable_frac``, adaptive relaxation kicks in.
    penalty :
        Fitness assigned to non-viable genomes when no relaxation is active.
    viable_boost_factor :
        When adaptive relaxation is active, multiply the penalty by this
        factor (< 1.0 = less severe penalty → more genomes can reproduce).
    """

    PENALTY = -1e6

    def __init__(
        self,
        criterion_fn: CriterionFn,
        min_viable_frac: float = 0.1,
        penalty: float = PENALTY,
        viable_boost_factor: float = 0.5,
    ) -> None:
        self.criterion_fn = criterion_fn
        self.min_viable_frac = min_viable_frac
        self.penalty = penalty
        self.viable_boost_factor = viable_boost_factor

        # Tracking
        self._n_total: int = 0
        self._n_viable: int = 0
        self._relaxation_active: bool = False
        self._generation_stats: list[float] = []  # history of viable_frac per gen

    @property
    def viable_frac(self) -> float:
        """Current fraction of viable genomes in this generation."""
        return self._n_viable / max(1, self._n_total)

    def reset_generation(self) -> None:
        """Reset per-generation counters.  Call at the start of each generation."""
        if self._n_total > 0:
            self._generation_stats.append(self.viable_frac)
        self._n_total = 0
        self._n_viable = 0

    def is_viable(self, genome: "Genome") -> bool:
        """Check whether *genome* passes the viability criterion."""
        try:
            return bool(self.criterion_fn(genome))
        except Exception:
            return False

    def apply(self, genome: "Genome", base_fitness: float) -> float:
        """Return adjusted fitness: base_fitness if viable, penalty if not.

        Also updates per-generation viable counters.
        """
        self._n_total += 1
        viable = self.is_viable(genome)
        if viable:
            self._n_viable += 1
            return base_fitness

        # Non-viable: apply penalty or relaxed penalty
        eff_penalty = self.penalty
        if self.viable_frac < self.min_viable_frac:
            self._relaxation_active = True
            # Relax: make penalty less severe
            eff_penalty = max(self.penalty, self.penalty * self.viable_boost_factor)
        else:
            self._relaxation_active = False

        return eff_penalty

    def wrap_fitness(
        self,
        base_fitness_fn: Callable[["Genome"], float],
    ) -> Callable[["Genome"], float]:
        """Return a wrapped fitness function that enforces the criterion.

        The genome's ``raw_fitness`` must already be set before ``criterion_fn``
        is evaluated (e.g., ``lambda g: g.fitness > -50``), so the wrapper
        first calls ``base_fitness_fn``, sets `fitness` on the genome
        temporarily, then checks the criterion.
        """
        mc = self

        def _mc_fn(genome: "Genome") -> float:
            base = base_fitness_fn(genome)
            # Temporarily expose the base fitness so criterion can inspect it
            genome.fitness = base
            return mc.apply(genome, base)

        return _mc_fn

    def viable_fraction_history(self) -> list[float]:
        """Return per-generation viable fractions."""
        return list(self._generation_stats)


# ---------------------------------------------------------------------------
# Open-Ended mode helpers
# ---------------------------------------------------------------------------

def make_novelty_with_criterion(
    base_fitness_fn: Callable[["Genome"], float],
    criterion: MinimalCriterion,
) -> Callable[["Genome"], float]:
    """Combine novelty-augmented fitness with minimal criterion.

    Novelty must be enabled on the NeuroEvolution instance separately via
    ``set_novelty_search(enabled=True)``.  This function adds the
    criterion filter on top.
    """
    def _fn(genome: "Genome") -> float:
        base = base_fitness_fn(genome)
        genome.fitness = base
        return criterion.apply(genome, base)
    return _fn


def make_curiosity_with_criterion(
    base_fitness_fn: Callable[["Genome"], float],
    criterion: MinimalCriterion,
) -> Callable[["Genome"], float]:
    """Combine curiosity-augmented fitness with minimal criterion."""
    def _fn(genome: "Genome") -> float:
        base = base_fitness_fn(genome)
        genome.fitness = base
        return criterion.apply(genome, base)
    return _fn


def make_qd_with_criterion(
    base_fitness_fn: Callable[["Genome"], float],
    criterion: MinimalCriterion,
) -> Callable[["Genome"], float]:
    """Combine quality-diversity fitness with minimal criterion."""
    def _fn(genome: "Genome") -> float:
        base = base_fitness_fn(genome)
        genome.fitness = base
        return criterion.apply(genome, base)
    return _fn
