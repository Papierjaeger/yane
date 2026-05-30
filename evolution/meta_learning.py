"""Meta-Learning NEAT — Few-Shot Adaptation.

Evolviert Genome, die sich in wenigen Episoden (Lamarck-Schritten) an neue
Aufgaben anpassen — ähnlich MAML (Model-Agnostic Meta-Learning), aber
vollständig gradientenfrei.

**Architektur:**

Outer Loop (NEAT-Evolution):
  Evolviert Population von Meta-Genomen basierend auf Post-Adaptation-Fitness.

Inner Loop (Lamarck-Adaptation, ``adaptation_steps`` Schritte):
  Für jede Evaluation: nimm eine neue Aufgabe vom Task-Sampler und passe
  das Genom mit hill-climbing an → Meta-Fitness = Fitness danach.

**Kernidee:**
Die äußere Evolution favorisiert Genome, die mit wenigen Lamarck-Schritten
maximale Fitness auf beliebigen Aufgaben erzielen — d.h. gute Inital-
gewichte für schnelle Adaptation.

**Akzeptanzkriterien:**
- Post-Adaptation-Fitness > Pre-Adaptation-Fitness (Lamarck verbessert Fitness).
- Meta-Fitness wird als Post-Adaptation-Fitness gesetzt.

Integration::

    def task_sampler():
        target = random.uniform(-1.0, 1.0)
        return lambda g: -abs(g.forward([0.5])[0] - target)

    result = yane.meta_train(
        task_sampler=task_sampler,
        meta_iterations=500,
        adaptation_steps=3,
    )
    print(result.best_meta_fitness)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome

TaskSampler = Callable[[], Callable[["Genome"], float]]


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class MetaTrainResult:
    """Result of a ``meta_train()`` run."""

    best_genome: "Genome"
    """Genome with the best meta-fitness (post-adaptation)."""

    best_meta_fitness: float
    """Best post-adaptation fitness achieved."""

    adaptation_deltas: list[float] = field(default_factory=list)
    """Per-iteration (post_fit - pre_fit) improvements (sampled)."""

    meta_iterations: int = 0
    """Total outer iterations run."""

    @property
    def mean_adaptation_delta(self) -> float:
        """Average fitness improvement per adaptation episode."""
        if not self.adaptation_deltas:
            return 0.0
        return sum(self.adaptation_deltas) / len(self.adaptation_deltas)


# ---------------------------------------------------------------------------
# MetaLearner
# ---------------------------------------------------------------------------

class MetaLearner:
    """Wraps Lamarck refinement as a meta-learning inner loop.

    Each call to :meth:`compute_meta_fitness` samples a task, evaluates the
    genome before and after *adaptation_steps* hill-climbing steps, and
    returns the post-adaptation fitness as the meta-fitness signal.

    Parameters
    ----------
    adaptation_steps :
        Lamarck hill-climbing steps per inner loop.
    lamarck_sigma :
        Noise scale for hill-climbing perturbations.
    track_deltas :
        When True, record ``(post_fit - pre_fit)`` for diagnostics.
    """

    def __init__(
        self,
        adaptation_steps: int = 3,
        lamarck_sigma: float = 0.1,
        track_deltas: bool = True,
    ) -> None:
        from yane.evolution.lamarck_refiner import LamarckRefiner
        self.adaptation_steps = adaptation_steps
        self.track_deltas = track_deltas
        self._refiner = LamarckRefiner()
        self._refiner.set_explicit(n_steps=adaptation_steps, sigma=lamarck_sigma)
        self._deltas: list[float] = []

    @property
    def adaptation_deltas(self) -> list[float]:
        return list(self._deltas)

    def compute_meta_fitness(
        self,
        genome: "Genome",
        task_sampler: TaskSampler,
    ) -> float:
        """Sample a task, adapt, and return post-adaptation fitness.

        The genome's weights are modified in-place by the Lamarck inner loop.
        For fair outer-loop evaluation, the caller should use a copy of the
        genome if the original must not be modified.

        Parameters
        ----------
        genome :
            Genome to adapt (may be modified in-place by refinement).
        task_sampler :
            Callable that returns a fitness function for a new task.

        Returns
        -------
        float
            Post-adaptation fitness.
        """
        task_fn = task_sampler()
        pre_fit = task_fn(genome)
        post_fit = self._refiner.refine(genome, task_fn, baseline_fitness=pre_fit)
        if self.track_deltas:
            self._deltas.append(post_fit - pre_fit)
        return post_fit

    def make_fitness_fn(
        self,
        task_sampler: TaskSampler,
    ) -> Callable[["Genome"], float]:
        """Return a fitness function for the NEAT outer loop.

        Each call evaluates ``compute_meta_fitness`` on a *copy* of the genome
        (so the original is not modified between generations).
        """
        def _meta_fn(genome: "Genome") -> float:
            adapted = genome.copy()
            return self.compute_meta_fitness(adapted, task_sampler)
        return _meta_fn
