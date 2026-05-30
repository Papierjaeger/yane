"""Behaviour Cloning as Warm-Start for NEAT evolution.

Supervised pre-training of a genome on expert demonstrations before the
evolutionary loop begins.  Reduces the number of generations required to
reach a target fitness by starting from a genome that already approximates
the desired behaviour.

Algorithm
---------
1. Pick a source genome (typically the freshly configured best genome or a
   provided genome).
2. Minimise the mean squared error (MSE) between genome outputs and
   demonstration targets via Lamarckian hill-climbing.
3. Optionally seed the entire population with noisy copies of the cloned
   genome so evolution starts in a promising region of weight-space.

Usage::

    from yane.evolution.behaviour_cloning import behaviour_clone

    demonstrations = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ]
    result = behaviour_clone(yane, demonstrations, n_steps=200)
    print(f"Error {result.initial_mse:.4f} → {result.final_mse:.4f}")
    result.seed_population()   # replace population with copies of cloned genome
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from yane.core.genome import Genome


@dataclass
class BehaviourCloneResult:
    """Result of a behaviour cloning run.

    Attributes
    ----------
    cloned_genome:
        The genome after cloning (weights tuned to match demonstrations).
        Always a copy — the original population is not modified unless
        :meth:`seed_population` is called.
    initial_mse:
        Mean squared error on demonstrations before cloning.
    final_mse:
        Mean squared error on demonstrations after cloning.
    n_steps_run:
        Number of Lamarckian hill-climbing steps actually performed.
    compression_ratio:
        ``initial_mse / final_mse`` (how much better cloning got).  Returns
        1.0 when initial_mse is zero (already perfect).
    _ne_ref:
        Internal reference to the NeuroEvolution instance (for seed_population).
    """
    cloned_genome: "Genome"
    initial_mse: float
    final_mse: float
    n_steps_run: int
    _ne_ref: object = field(default=None, repr=False, compare=False)

    @property
    def compression_ratio(self) -> float:
        if self.initial_mse < 1e-12:
            return 1.0
        return self.initial_mse / max(self.final_mse, 1e-12)

    def seed_population(
        self,
        n_copies: int | None = None,
        noise_sigma: float = 0.05,
        freeze_layers: "list[str] | None" = None,
    ) -> None:
        """Seed the evolution population with noisy copies of the cloned genome.

        Parameters
        ----------
        n_copies:
            Number of population slots to replace.  ``None`` = all slots.
        noise_sigma:
            Weight perturbation added to each copy (0 = identical copies).
        freeze_layers:
            Passed to :func:`seed_population_with`.
        """
        if self._ne_ref is None:
            raise RuntimeError(
                "BehaviourCloneResult has no NeuroEvolution reference.  "
                "Use NeuroEvolution.behaviour_clone() to obtain a result "
                "with seeding capability."
            )
        seed_population_with(
            self._ne_ref,
            self.cloned_genome,
            n_copies=n_copies,
            noise_sigma=noise_sigma,
            freeze_layers=freeze_layers,
        )


def _mse(genome: "Genome", demonstrations: "list[tuple]") -> float:
    """Mean squared error of *genome* on *demonstrations*."""
    total = 0.0
    for inp, tgt in demonstrations:
        genome.reset()
        out = genome.forward(inp)
        total += sum((o - t) ** 2 for o, t in zip(out, tgt))
    return total / max(len(demonstrations), 1)


def _make_bc_fitness_fn(
    demonstrations: "list[tuple]",
) -> "Callable[[Genome], float]":
    """Return a fitness function that maximises negative MSE (for LamarckRefiner)."""
    def _fit(genome: "Genome") -> float:
        return -_mse(genome, demonstrations)
    return _fit


def behaviour_clone(
    ne: object,
    demonstrations: "list[tuple[list[float], list[float]]]",
    n_steps: int = 200,
    sigma: float = 0.05,
    *,
    lamarck_steps_per_round: int | None = None,
    seed_population: bool = False,
    noise_sigma: float = 0.05,
    freeze_layers: "list[str] | None" = None,
) -> "BehaviourCloneResult":
    """Clone the best genome to match *demonstrations* via Lamarckian hill-climbing.

    Parameters
    ----------
    ne:
        :class:`~yane.neuro_evolution.NeuroEvolution` instance.  Must be
        configured (``configure()`` called).
    demonstrations:
        List of ``(inputs, targets)`` pairs, e.g.
        ``[([0.0, 1.0], [1.0]), ([1.0, 0.0], [1.0])]``.
    n_steps:
        Total number of hill-climbing evaluation steps.
    sigma:
        Weight perturbation sigma per step.
    lamarck_steps_per_round:
        Internal Lamarck steps per call to :class:`~yane.evolution.lamarck_refiner.LamarckRefiner`.
        Defaults to ``n_steps`` (single pass).
    seed_population:
        If ``True``, immediately call :meth:`BehaviourCloneResult.seed_population`
        after cloning.
    noise_sigma:
        Sigma used when seeding the population with noisy copies.
    freeze_layers:
        Connection groups to freeze when seeding (passed to
        :meth:`~yane.neuro_evolution.NeuroEvolution.load_genome_as_seed`).

    Returns
    -------
    BehaviourCloneResult
    """
    from yane.evolution.lamarck_refiner import LamarckRefiner

    ne._ensure_configured()  # type: ignore[attr-defined]
    pop = ne._population  # type: ignore[attr-defined]
    if pop._evaluated:
        genome = pop.get_best().copy()
    elif pop._unevaluated:
        genome = pop._unevaluated[0].copy()
    else:
        raise RuntimeError("Population is empty; call configure() first.")

    if not demonstrations:
        raise ValueError("demonstrations must not be empty")

    fitness_fn = _make_bc_fitness_fn(demonstrations)

    initial_mse = _mse(genome, demonstrations)

    # Configure a private LamarckRefiner for cloning
    refiner = LamarckRefiner()
    refiner.sigma = sigma
    steps = lamarck_steps_per_round if lamarck_steps_per_round is not None else n_steps
    refiner.steps = max(1, steps)

    baseline = fitness_fn(genome)
    refiner.refine(genome, fitness_fn, baseline_fitness=baseline, n_steps=n_steps)

    final_mse = _mse(genome, demonstrations)

    # Fitness annotation: negative MSE (consistent with fitness_fn)
    genome.fitness = -final_mse
    genome.raw_fitness = -final_mse

    result = BehaviourCloneResult(
        cloned_genome=genome,
        initial_mse=initial_mse,
        final_mse=final_mse,
        n_steps_run=n_steps,
        _ne_ref=ne,
    )

    if seed_population:
        result.seed_population(noise_sigma=noise_sigma, freeze_layers=freeze_layers)

    return result


def seed_population_with(
    ne: object,
    genome: "Genome",
    n_copies: int | None = None,
    noise_sigma: float = 0.05,
    freeze_layers: "list[str] | None" = None,
) -> int:
    """Seed the evolution population from the cloned genome.

    Calls :meth:`~yane.neuro_evolution.NeuroEvolution.load_genome_as_seed` to
    set *genome* as the population template, then optionally adds extra noisy
    copies to the unevaluated pool so the first training generation starts from
    a diverse neighbourhood of weight-space.

    Parameters
    ----------
    ne:
        :class:`~yane.neuro_evolution.NeuroEvolution` instance.
    genome:
        The cloned genome to use as the seed.
    n_copies:
        Number of noisy copies to add to the unevaluated pool in addition to
        the base seed.  ``None`` means no extra copies (just the seed).
    noise_sigma:
        Per-weight Gaussian noise added to each extra copy.
    freeze_layers:
        Passed to ``load_genome_as_seed`` (freezes selected weight groups).

    Returns
    -------
    int
        Total genomes in the unevaluated pool after seeding.
    """
    # Use the official API to set the genome as the population seed/template.
    ne.load_genome_as_seed(genome, freeze_layers=freeze_layers)  # type: ignore[attr-defined]
    pop = ne._population  # type: ignore[attr-defined]

    # Optionally add extra noisy copies to the unevaluated pool.
    if n_copies is not None and n_copies > 0:
        for _ in range(n_copies):
            copy = genome.copy()
            if noise_sigma > 0.0:
                for node in copy.nodes:
                    for conn in node.connections:
                        conn.weight += random.gauss(0.0, noise_sigma)
                        if conn._weight_arr is not None:
                            conn._weight_arr[conn._weight_idx] = conn._weight
                    node.bias += random.gauss(0.0, noise_sigma)
            copy.fitness = genome.fitness
            copy.raw_fitness = genome.raw_fitness
            pop._unevaluated.append(copy)

    return len(pop._unevaluated)
