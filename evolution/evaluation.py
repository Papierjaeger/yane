"""Genome evaluation runner — multi-eval, early stopping, result aggregation."""
from __future__ import annotations
import dataclasses
import inspect
import statistics
import time
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from yane.core.genome import Genome
    from yane.evolution.population import Population
    from yane.evolution.lamarck_refiner import LamarckRefiner


@dataclasses.dataclass
class EvaluationResult:
    """Structured result from evaluating a single genome.

    Returned by EvaluationRunner.run() and NeuroEvolution._run_evaluations().
    The public API (submit_fitness, submit_fitness_batch) is not affected.
    """
    genome: Genome
    fitness: float
    elapsed_ms: float
    n_lamarck_steps: int = 0
    stopped_early: bool = False
    early_stop_reason: str = ""     # "threshold", or "" if not stopped
    raw_fitnesses: list[float] = dataclasses.field(default_factory=list)


def aggregate_fitnesses(
    fitnesses: list[float],
    aggregation: str,
    sigma_penalty: float,
) -> float:
    """Combine multiple fitness values into one.

    aggregation: "mean" | "median" | "min"
    sigma_penalty: subtract sigma_penalty * std from the result (0 = no penalty).
    """
    if len(fitnesses) == 1:
        return fitnesses[0]
    if aggregation == "median":
        result = statistics.median(fitnesses)
    elif aggregation == "min":
        result = min(fitnesses)
    else:
        result = statistics.mean(fitnesses)
    if sigma_penalty > 0.0:
        result -= sigma_penalty * statistics.pstdev(fitnesses)
    return result


class EvaluationRunner:
    """Owns multi-eval and early-stopping state; runs genome evaluations."""

    def __init__(self) -> None:
        self.n_evaluations: int = 1
        self.aggregation: str = "mean"
        self.sigma_penalty: float = 0.0
        self.early_stopping_factor: float | None = None
        self.n_early_stopped: int = 0
        self.early_stopping_n: int | None = None

    def configure_multi_eval(
        self,
        n: int,
        aggregation: str,
        sigma_penalty: float,
    ) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        valid = ("mean", "median", "min")
        if aggregation not in valid:
            raise ValueError(f"aggregation must be one of {valid}, got {aggregation!r}")
        self.n_evaluations = n
        self.aggregation = aggregation
        self.sigma_penalty = max(0.0, sigma_penalty)

    def run(
        self,
        genome: Genome,
        fitness_fn: Callable[[Genome], float],
        population: Population,
        lamarck: LamarckRefiner,
    ) -> EvaluationResult:
        """Evaluate genome (possibly multiple times), then apply adaptive Lamarck.

        Supports two calling conventions for fitness_fn:

        1. Regular function — called n_evaluations times; results aggregated.
        2. Generator function — called once; each yield is one episode result.
           Early stopping aborts the generator when the running mean drops
           below the pool threshold.
        """
        # Explicit Lamarck: refine weights in-place before the main evaluation.
        # Runs outside the elapsed_ms window so efficiency-penalty sees only the
        # actual eval time, not the hill-climbing overhead.
        n_lamarck_steps = 0
        if lamarck.steps > 0:
            lamarck.refine(genome, fitness_fn, n_steps=lamarck.steps)
            n_lamarck_steps = lamarck.steps
            lamarck.n_applied += 1
            lamarck.n_steps_total += lamarck.steps
            sp = population.get_species_for_genome(genome)
            if sp is not None:
                sp.lamarck_n_applied += 1
                sp.lamarck_n_steps_total += lamarck.steps

        start = time.perf_counter()
        raw: list[float] = []
        stopped_early = False
        early_stop_reason = ""

        if inspect.isgeneratorfunction(fitness_fn):
            gen = fitness_fn(genome)
            N = self.early_stopping_n   # snapshot; None until calibrated
            cumulative = 0.0
            episode_count = 0
            try:
                for k, episode_fitness in enumerate(gen, 1):
                    raw.append(episode_fitness)
                    cumulative += episode_fitness
                    episode_count = k
                    if (self.early_stopping_factor is not None
                            and N is not None
                            and k >= max(1, N // 5)):
                        estimated = cumulative * (N / k)
                        evaluated = population._evaluated
                        if evaluated:
                            best = max(g.fitness for g in evaluated)
                            threshold = best - abs(best) * self.early_stopping_factor
                            if estimated < threshold:
                                gen.close()
                                stopped_early = True
                                early_stop_reason = "threshold"
                                self.n_early_stopped += 1
                                break
            except StopIteration:
                pass
            if not stopped_early and episode_count > 0 and self.early_stopping_n is None:
                self.early_stopping_n = episode_count
            fitness = aggregate_fitnesses(raw, self.aggregation, self.sigma_penalty) if raw else 0.0
        elif self.n_evaluations <= 1:
            fitness = fitness_fn(genome)
        else:
            for _ in range(self.n_evaluations):
                raw.append(fitness_fn(genome))
            fitness = aggregate_fitnesses(raw, self.aggregation, self.sigma_penalty)

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if lamarck.steps == 0:
            # Adaptive Lamarck fires after the baseline is known, only when
            # stagnation pressure is high.
            n_steps = lamarck.adaptive_steps(genome, fitness, population)
            if n_steps > 0:
                fitness = lamarck.refine(
                    genome, fitness_fn,
                    baseline_fitness=fitness,
                    n_steps=n_steps,
                )
                n_lamarck_steps = n_steps
                lamarck.n_applied += 1
                lamarck.n_steps_total += n_steps
                sp = population.get_species_for_genome(genome)
                if sp is not None:
                    sp.lamarck_n_applied += 1
                    sp.lamarck_n_steps_total += n_steps

        return EvaluationResult(
            genome=genome,
            fitness=fitness,
            elapsed_ms=elapsed_ms,
            n_lamarck_steps=n_lamarck_steps,
            stopped_early=stopped_early,
            early_stop_reason=early_stop_reason,
            raw_fitnesses=raw,
        )
