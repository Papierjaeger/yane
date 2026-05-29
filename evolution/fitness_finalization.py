"""Final fitness shaping helpers for NeuroEvolution submission paths."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from yane.core.genome import Genome
from yane.evolution.efficiency_penalty import EfficiencyPenalty
from yane.evolution.fitness_sanitizer import FitnessSanitizer
from yane.evolution.descriptors import AdaptiveFitnessComponentWeights
from yane.evolution.multi_objective import as_objectives


@dataclass(frozen=True)
class FitnessFinalizationConfig:
    """Configuration needed to turn raw evaluator output into scalar fitness."""

    sanitizer: FitnessSanitizer
    multi_objective_enabled: bool = False
    multi_objective_weights: Sequence[float] | None = None
    fitness_component_weights: AdaptiveFitnessComponentWeights | None = None
    efficiency_penalty: EfficiencyPenalty | None = None
    complexity_penalty_nodes: float = 0.0
    complexity_penalty_connections: float = 0.0


def finalize_fitness_value(
    fitness,
    elapsed_ms: float | None,
    genome: Genome | None,
    config: FitnessFinalizationConfig,
) -> float:
    """Sanitize, scalarize and penalize a raw fitness value."""
    objectives = as_objectives(fitness) if config.multi_objective_enabled else None
    if objectives is not None:
        if config.multi_objective_weights is not None:
            if len(config.multi_objective_weights) != len(objectives):
                raise ValueError(
                    "multi-objective weights must match objective count "
                    f"({len(config.multi_objective_weights)} != {len(objectives)})"
                )
            scalar = sum(w * v for w, v in zip(config.multi_objective_weights, objectives))
        else:
            scalar = sum(objectives)
        if genome is not None:
            genome.objectives = tuple(config.sanitizer.apply(v) for v in objectives)
        fitness = scalar
    elif genome is not None:
        genome.objectives = None

    # raw_fitness = task-only score, before both the curiosity exploration bonus
    # and any fitness-component bonus.  The curiosity wrapper stores the pure
    # task score on the genome as _curiosity_task_base so we can recover it here
    # even after Lamarck has called the wrapped evaluator multiple times.
    # We delete the sentinel here so it never leaks into checkpoints.
    if genome is not None and '_curiosity_task_base' in genome.__dict__:
        raw_fitness = genome.__dict__.pop('_curiosity_task_base')
    else:
        raw_fitness = fitness  # task score before component bonus

    if genome is not None and config.fitness_component_weights is not None:
        component_score, _ = config.fitness_component_weights.scalarize(genome)
        fitness += component_score

    fitness = config.sanitizer.apply(fitness)
    if config.efficiency_penalty is not None and elapsed_ms is not None:
        fitness = config.efficiency_penalty.apply(fitness, elapsed_ms)
    if genome is not None and (
        config.complexity_penalty_nodes > 0.0
        or config.complexity_penalty_connections > 0.0
    ):
        hidden = max(0, len(genome.nodes) - len(genome.input_nodes) - len(genome.output_nodes))
        fitness -= (
            config.complexity_penalty_nodes * hidden
            + config.complexity_penalty_connections * genome.connection_count
        )
    if genome is not None:
        genome.raw_fitness = raw_fitness
    return fitness
