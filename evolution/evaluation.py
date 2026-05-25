"""Genome evaluation runner — multi-eval, early stopping, result aggregation."""
from __future__ import annotations
import dataclasses
import inspect
import statistics
import time
from collections.abc import Sequence
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
    n_fitness_calls: int = 1


def _is_vector_fitness(value) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _aggregate_scalar(values: list[float], aggregation: str, sigma_penalty: float) -> float:
    if aggregation == "median":
        result = statistics.median(values)
    elif aggregation == "min":
        result = min(values)
    else:
        result = statistics.mean(values)
    if sigma_penalty > 0.0 and len(values) > 1:
        result -= sigma_penalty * statistics.pstdev(values)
    return result


def aggregate_fitnesses(
    fitnesses: list,
    aggregation: str,
    sigma_penalty: float,
) -> float | tuple[float, ...]:
    """Combine multiple fitness values into one.

    aggregation: "mean" | "median" | "min"
    sigma_penalty: subtract sigma_penalty * std from the result (0 = no penalty).
    """
    if len(fitnesses) == 1:
        return fitnesses[0]
    if _is_vector_fitness(fitnesses[0]):
        width = len(fitnesses[0])
        rows = [tuple(float(v) for v in row) for row in fitnesses]
        if any(len(row) != width for row in rows):
            raise ValueError("All objective vectors must have the same length")
        return tuple(
            _aggregate_scalar([row[i] for row in rows], aggregation, sigma_penalty)
            for i in range(width)
        )
    return _aggregate_scalar([float(v) for v in fitnesses], aggregation, sigma_penalty)


class EvaluationRunner:
    """Owns multi-eval and early-stopping state; runs genome evaluations."""

    def __init__(self) -> None:
        self.n_evaluations: int = 1
        self.aggregation: str = "mean"
        self.sigma_penalty: float = 0.0
        self.early_stopping_factor: float | None = None
        self.n_early_stopped: int = 0
        self.early_stopping_n: int | None = None
        self.anytime_enabled: bool = False
        self.anytime_min_evals: int = 1
        self.anytime_max_evals: int = 5
        self.anytime_promotion_frac: float = 0.3
        self.anytime_aggregation: str = "mean"
        self.anytime_total_genomes: int = 0
        self.anytime_promoted: int = 0
        self.anytime_total_calls: int = 0
        self.anytime_saved_calls: int = 0
        self.anytime_promoted_variances: list[float] = []

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

    def configure_anytime_eval(
        self,
        enabled: bool = True,
        min_evals: int = 1,
        max_evals: int = 5,
        promotion_frac: float = 0.3,
        aggregation: str = "mean",
    ) -> None:
        """Configure adaptive per-genome evaluation budgeting.

        Each genome receives ``min_evals`` measurements. Only genomes whose
        provisional score is competitive with the current evaluated pool are
        promoted to ``max_evals``. In early warm-up, promotion stays permissive
        until enough evaluated genomes exist to form a quantile threshold.
        """
        if min_evals < 1:
            raise ValueError(f"min_evals must be >= 1, got {min_evals}")
        if max_evals < min_evals:
            raise ValueError(
                f"max_evals must be >= min_evals, got {max_evals} < {min_evals}"
            )
        valid = ("mean", "median", "min", "max")
        if aggregation not in valid:
            raise ValueError(f"aggregation must be one of {valid}, got {aggregation!r}")
        self.anytime_enabled = bool(enabled)
        self.anytime_min_evals = int(min_evals)
        self.anytime_max_evals = int(max_evals)
        self.anytime_promotion_frac = max(0.0, min(1.0, float(promotion_frac)))
        self.anytime_aggregation = aggregation

    def _aggregate_anytime(self, values: list[float]) -> float:
        if self.anytime_aggregation == "max":
            return max(values)
        return aggregate_fitnesses(values, self.anytime_aggregation, self.sigma_penalty)

    def _should_promote(self, provisional: float, population: Population) -> bool:
        if self.anytime_promotion_frac >= 1.0:
            return True
        if self.anytime_promotion_frac <= 0.0:
            return False
        evaluated = population._evaluated
        if len(evaluated) < max(4, int(1.0 / max(1e-9, self.anytime_promotion_frac))):
            return True
        scores = sorted(g.fitness for g in evaluated)
        cutoff_index = int((1.0 - self.anytime_promotion_frac) * (len(scores) - 1))
        threshold = scores[max(0, min(len(scores) - 1, cutoff_index))]
        return provisional >= threshold

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
        sp = population.get_species_for_genome(genome)
        species_id = id(sp) if sp is not None else None
        _lamarck_eligible = lamarck.is_eligible_for_species(species_id)

        if lamarck.steps > 0 and _lamarck_eligible:
            if lamarck.nes_mode:
                refine_fn = lamarck.refine_nes
            elif lamarck.sa_mode:
                refine_fn = lamarck.refine_sa
            elif lamarck.cma_mode:
                refine_fn = lamarck.refine_cma_es
            else:
                refine_fn = lamarck.refine
            refine_fn(genome, fitness_fn, n_steps=lamarck.steps)
            n_lamarck_steps = lamarck.steps
            lamarck.n_applied += 1
            lamarck.n_steps_total += lamarck.steps
            if sp is not None:
                sp.lamarck_n_applied += 1
                sp.lamarck_n_steps_total += lamarck.steps
            if species_id is not None:
                lamarck.record_species_stats(species_id, lamarck.steps, improved=False)

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
        elif self.anytime_enabled:
            raw = [fitness_fn(genome) for _ in range(self.anytime_min_evals)]
            provisional = self._aggregate_anytime(raw)
            promoted = self._should_promote(float(provisional), population)
            if promoted:
                extra = self.anytime_max_evals - self.anytime_min_evals
                if extra > 0:
                    raw.extend(fitness_fn(genome) for _ in range(extra))
                self.anytime_promoted += 1
                if len(raw) > 1:
                    self.anytime_promoted_variances.append(statistics.pvariance(raw))
                    if len(self.anytime_promoted_variances) > 500:
                        self.anytime_promoted_variances = self.anytime_promoted_variances[-500:]
            fitness = self._aggregate_anytime(raw)
            self.anytime_total_genomes += 1
            self.anytime_total_calls += len(raw)
            self.anytime_saved_calls += max(0, self.anytime_max_evals - len(raw))
        elif self.n_evaluations <= 1:
            fitness = fitness_fn(genome)
        else:
            for _ in range(self.n_evaluations):
                raw.append(fitness_fn(genome))
            fitness = aggregate_fitnesses(raw, self.aggregation, self.sigma_penalty)

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if lamarck.steps == 0 and _lamarck_eligible:
            # Adaptive Lamarck fires after the baseline is known, only when
            # stagnation pressure is high.
            n_steps = lamarck.adaptive_steps(genome, fitness, population)
            if n_steps > 0 and lamarck._consume_budget(n_steps):
                if lamarck.nes_mode:
                    refine_fn = lamarck.refine_nes
                elif lamarck.sa_mode:
                    refine_fn = lamarck.refine_sa
                elif lamarck.cma_mode:
                    refine_fn = lamarck.refine_cma_es
                else:
                    refine_fn = lamarck.refine
                old_fitness = fitness
                fitness = refine_fn(
                    genome, fitness_fn,
                    baseline_fitness=fitness,
                    n_steps=n_steps,
                )
                improved = fitness > old_fitness
                n_lamarck_steps = n_steps
                lamarck.n_applied += 1
                lamarck.n_steps_total += n_steps
                if sp is not None:
                    sp.lamarck_n_applied += 1
                    sp.lamarck_n_steps_total += n_steps
                if species_id is not None:
                    lamarck.record_species_stats(species_id, n_steps, improved)

        return EvaluationResult(
            genome=genome,
            fitness=fitness,
            elapsed_ms=elapsed_ms,
            n_lamarck_steps=n_lamarck_steps,
            stopped_early=stopped_early,
            early_stop_reason=early_stop_reason,
            raw_fitnesses=raw,
            n_fitness_calls=len(raw) if raw else 1,
        )
