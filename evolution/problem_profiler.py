"""Problem Profiler — Phase 2 of P0 Meta-Adaptive Orchestration.

Analyses an evaluator function by running a small warmup set of random genomes
and computing properties of the fitness landscape: task type, difficulty, noise
level, stochasticity proxy (temporal dependency), and fitness distribution.

Usage::

    yane.configure(n_inputs=4, n_outputs=2)
    profile = yane.profile_problem(cartpole_evaluator, n_warmup=30)
    print(profile.task_type)          # "rl_continuous"
    print(profile.temporal_dependency) # > 0.5 for stochastic RL
    print(profile.noise_level)         # > 0 for stochastic tasks

Design notes:
- Only uses the fitness scalar returned by ``evaluator(genome)``.  No state
  introspection or gym wrapper required.
- Each warmup genome is evaluated TWICE so that noise (stochasticity) can be
  separated from genome-quality variance.
- ``state_dim_effective`` is currently set to ``n_inputs`` as a conservative
  estimate.  A future version may instrument the evaluator to capture states.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from yane.neuro_evolution import NeuroEvolution


# ---------------------------------------------------------------------------
# ProblemProfile
# ---------------------------------------------------------------------------

@dataclass
class ProblemProfile:
    """Compact description of a problem's fitness landscape.

    All metrics are computed from ``n_warmup`` random genomes evaluated twice.

    Attributes
    ----------
    task_type:            Heuristic task category.
    task_type_confidence: Confidence in the classification (0–1).
    alternative_types:    Other plausible task types.
    n_inputs:             Number of genome inputs.
    n_outputs:            Number of genome outputs.
    estimated_difficulty: 0 = trivial (random genomes already at target),
                          1 = very hard.
    noise_level:          Normalised average absolute difference between two
                          evaluations of the same genome.  0 = deterministic.
    reward_sparsity:      Fraction of evaluations returning ≈ 0 fitness.
    temporal_dependency:  Proxy for sequential / RL structure.
                          High when the evaluator is stochastic (noise > 0).
    state_dim_effective:  Estimated effective input dimensionality.
    fitness_mean:         Mean fitness over all warmup evaluations.
    fitness_std:          Standard deviation.
    fitness_min:          Minimum observed fitness.
    fitness_max:          Maximum observed fitness.
    """

    task_type: str
    task_type_confidence: float
    n_inputs: int
    n_outputs: int
    estimated_difficulty: float
    noise_level: float
    reward_sparsity: float
    temporal_dependency: float
    state_dim_effective: int
    fitness_mean: float
    fitness_std: float
    fitness_min: float
    fitness_max: float
    alternative_types: list[str] = field(default_factory=list)

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.__dict__.setdefault("alternative_types", [])
        self.__dict__.setdefault("task_type_confidence", 1.0)


# ---------------------------------------------------------------------------
# ProblemProfiler
# ---------------------------------------------------------------------------

class ProblemProfiler:
    """Profiles an evaluator by running a small warmup population."""

    def __init__(self, ne: "NeuroEvolution") -> None:
        self._ne = ne

    def profile(
        self,
        evaluator: Callable,
        n_warmup: int = 50,
    ) -> ProblemProfile:
        """Profile the evaluator and return a ``ProblemProfile``.

        Parameters
        ----------
        evaluator:  The fitness function ``(genome) → float``.
        n_warmup:   Number of random genomes to evaluate (each twice).
                    Larger values give more reliable metrics but take longer.
                    Minimum is 5; values below are silently clamped up.

        Returns
        -------
        ProblemProfile
        """
        ne = self._ne
        n_warmup = max(5, int(n_warmup))

        pop = ne._population
        n_inputs = ne._n_inputs
        n_outputs = ne._n_outputs
        rng = random.Random(ne._seed if ne._seed is not None else 0)

        # 1. Generate n_warmup random genomes from the population template.
        genomes = _sample_random_genomes(pop, n_warmup, rng)

        # 2. Evaluate each genome TWICE to measure noise.
        fitness_a: list[float] = []
        fitness_b: list[float] = []
        for g in genomes:
            fitness_a.append(_safe_eval(evaluator, g))
            fitness_b.append(_safe_eval(evaluator, g))

        all_fitness = fitness_a + fitness_b

        # 3. Compute individual metrics.
        mean_f = _mean(all_fitness)
        std_f = _std(all_fitness)

        noise = _noise_level(fitness_a, fitness_b)
        sparsity = _reward_sparsity(all_fitness, mean_f)
        temporal_dep = _temporal_dependency(noise)
        difficulty = _estimated_difficulty(all_fitness, ne.min_fitness)
        state_dim = n_inputs  # conservative — no env introspection in Phase 2

        task_type, confidence, alternatives = _classify_task(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            mean_fitness=mean_f,
            std_fitness=std_f,
            noise_level=noise,
            reward_sparsity=sparsity,
        )

        return ProblemProfile(
            task_type=task_type,
            task_type_confidence=confidence,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            estimated_difficulty=difficulty,
            noise_level=noise,
            reward_sparsity=sparsity,
            temporal_dependency=temporal_dep,
            state_dim_effective=state_dim,
            fitness_mean=mean_f,
            fitness_std=std_f,
            fitness_min=min(all_fitness),
            fitness_max=max(all_fitness),
            alternative_types=alternatives,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_random_genomes(pop, n: int, rng: random.Random):
    """Return *n* randomised copies of the population template."""
    from yane.evolution import smart_mutation

    template = pop._template
    tracker = pop._tracker
    n_in = len(template.input_nodes)
    n_out = len(template.output_nodes)
    min_conn = n_out
    max_conn = min(n_in * n_out + 4, max(n_in, 20))

    genomes = []
    for _ in range(n):
        g = template.copy()
        n_conn = rng.randint(min_conn, max_conn)
        for _ in range(n_conn):
            try:
                smart_mutation.add_connection(g, tracker)
            except Exception:
                pass
        # Randomise weights so genomes behave differently.
        for node in g.nodes:
            for conn in node.connections:
                conn.weight = rng.gauss(0.0, 1.0)
        genomes.append(g)
    return genomes


def _safe_eval(evaluator: Callable, genome) -> float:
    """Call evaluator, returning 0.0 on any exception."""
    try:
        result = evaluator(genome)
        if result is None or (isinstance(result, float) and math.isnan(result)):
            return 0.0
        return float(result)
    except Exception:
        return 0.0


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _noise_level(fitness_a: list[float], fitness_b: list[float]) -> float:
    """Mean absolute difference between two evaluations, normalised by |mean|.

    0.0 for a deterministic evaluator; > 0 when the evaluator is stochastic.
    """
    if not fitness_a:
        return 0.0
    diffs = [abs(a - b) for a, b in zip(fitness_a, fitness_b)]
    mean_diff = _mean(diffs)
    mean_abs = _mean([abs(f) for f in fitness_a + fitness_b])
    return mean_diff / (mean_abs + 1e-8)


def _temporal_dependency(noise: float) -> float:
    """Proxy for temporal/sequential structure.

    High noise implies stochastic, episodic (RL) evaluation → high temporal
    dependency.  Deterministic evaluators (classification, regression) have
    noise ≈ 0 → temporal_dependency ≈ 0.

    The normalisation is chosen so that moderate stochasticity (noise ≈ 0.3)
    already exceeds 0.5, clearly separating RL from deterministic tasks.
    """
    return min(1.0, noise * 2.0)


def _reward_sparsity(all_fitness: list[float], mean_f: float) -> float:
    """Fraction of evaluations returning fitness near zero."""
    if not all_fitness:
        return 0.0
    # Threshold: 2% of |mean| or 1.0 (whichever is larger)
    threshold = max(1.0, abs(mean_f) * 0.02)
    return sum(1 for f in all_fitness if abs(f) <= threshold) / len(all_fitness)


def _estimated_difficulty(all_fitness: list[float], target: float | None) -> float:
    """Estimate how hard the problem is relative to the target fitness.

    0.0 = trivial (random genomes already reach target),
    1.0 = very hard (random genomes far below target).
    """
    if not all_fitness:
        return 0.5
    mean_f = _mean(all_fitness)

    if target is not None and abs(target) > 1e-8:
        # Ratio of current mean to target.
        ratio = mean_f / target
        return max(0.0, min(1.0, 1.0 - ratio))

    # No target: use the spread of fitness values as a proxy.
    max_f = max(all_fitness)
    min_f = min(all_fitness)
    spread = max_f - min_f
    if spread < 1e-8:
        return 0.5
    # How far below the max is the mean? → difficulty estimate.
    return max(0.0, min(1.0, 1.0 - (mean_f - min_f) / spread))


def _classify_task(
    n_inputs: int,
    n_outputs: int,
    mean_fitness: float,
    std_fitness: float,
    noise_level: float,
    reward_sparsity: float,
) -> tuple[str, float, list[str]]:
    """Heuristically classify the task type.

    Returns
    -------
    (task_type, confidence, alternative_types)
    """
    candidates: list[tuple[float, str]] = []  # (score, type)

    # ── Signals ──────────────────────────────────────────────────────────────
    is_stochastic = noise_level > 0.05
    is_multioutput = n_outputs > 2
    is_negative = mean_fitness < -1.0
    is_sparse = reward_sparsity > 0.3
    is_single_output = n_outputs == 1

    # ── Score each candidate ─────────────────────────────────────────────────
    # RL continuous: many outputs, OR stochastic + multi-step environment
    rl_cont_score = 0.0
    if is_multioutput:
        rl_cont_score += 0.4
    if is_stochastic:
        rl_cont_score += 0.35
    if is_negative:
        rl_cont_score += 0.2
    if not is_single_output:
        rl_cont_score += 0.1
    candidates.append((rl_cont_score, "rl_continuous"))

    # RL discrete: 1-2 outputs, stochastic, sparse rewards
    rl_disc_score = 0.0
    if is_stochastic:
        rl_disc_score += 0.4
    if is_sparse:
        rl_disc_score += 0.2
    if not is_multioutput:
        rl_disc_score += 0.2
    if is_single_output:
        rl_disc_score -= 0.1
    candidates.append((rl_disc_score, "rl_discrete"))

    # Classification: deterministic, bounded fitness, single or multi-output
    cls_score = 0.0
    if not is_stochastic:
        cls_score += 0.5
    if 0.0 <= mean_fitness <= n_inputs * 10:
        cls_score += 0.2
    if not is_negative:
        cls_score += 0.2
    candidates.append((cls_score, "classification"))

    # Regression: deterministic, can be negative
    reg_score = 0.0
    if not is_stochastic:
        reg_score += 0.4
    if is_single_output:
        reg_score += 0.2
    if is_negative:
        reg_score += 0.2
    candidates.append((reg_score, "regression"))

    # ── Pick best ──────────────────────────────────────────────────────────
    candidates.sort(reverse=True)
    best_score, best_type = candidates[0]
    total = sum(s for s, _ in candidates if s > 0) or 1.0
    confidence = min(1.0, best_score / total)

    # Alternatives: other types with score > 0.15
    alternatives = [t for s, t in candidates[1:] if s > 0.15]

    return best_type, round(confidence, 3), alternatives
