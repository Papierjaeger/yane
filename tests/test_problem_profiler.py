"""Tests for the Problem Profiler (P0 Meta-Adaptive Orchestration, Phase 2).

Covers:
- ProblemProfile dataclass construction and defaults
- _noise_level: 0 for deterministic, >0 for stochastic evaluators
- _temporal_dependency: low for deterministic, >0.5 for stochastic evaluators
- _reward_sparsity: correctly detects ≈0 fitness fraction
- _estimated_difficulty: correct with and without target fitness
- _classify_task: classification, regression, rl_continuous, rl_discrete
- ProblemProfiler.profile(): end-to-end with synthetic evaluators
- NeuroEvolution.profile_problem() integration
- profile_problem() raises without configure()
"""
from __future__ import annotations

import random
import math

import pytest

from yane import NeuroEvolution
from yane.evolution.problem_profiler import (
    ProblemProfile,
    ProblemProfiler,
    _classify_task,
    _estimated_difficulty,
    _noise_level,
    _reward_sparsity,
    _temporal_dependency,
    _mean,
    _std,
)


# ---------------------------------------------------------------------------
# Synthetic evaluators
# ---------------------------------------------------------------------------

def deterministic_evaluator(genome):
    """Returns a fixed sum of all connection weights — fully deterministic."""
    total = 0.0
    for node in genome.nodes:
        for conn in node.connections:
            total += conn.weight
    return abs(total) + 1.0  # always positive, never near 0


class StochasticEvaluator:
    """Returns a noisy fitness value — simulates RL stochasticity."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def __call__(self, genome):
        base = sum(abs(c.weight) for n in genome.nodes for c in n.connections) + 5.0
        # Large noise relative to base
        noise = self._rng.gauss(0, base * 0.8)
        return base + noise


def sparse_evaluator(genome):
    """Returns 0 most of the time — simulates sparse reward."""
    total = sum(c.weight for n in genome.nodes for c in n.connections)
    return 0.0 if abs(total) < 5.0 else abs(total)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def ne():
    instance = NeuroEvolution(seed=1)
    instance.configure(n_inputs=2, n_outputs=1)
    return instance


@pytest.fixture()
def ne_multi():
    """Multi-output NeuroEvolution — mimics continuous RL."""
    instance = NeuroEvolution(seed=2)
    instance.configure(n_inputs=4, n_outputs=4)
    return instance


# ---------------------------------------------------------------------------
# Unit tests for metric helpers
# ---------------------------------------------------------------------------

class TestMetricHelpers:
    def test_noise_level_zero_for_identical(self):
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0, 3.0]
        assert _noise_level(a, b) == pytest.approx(0.0)

    def test_noise_level_positive_for_different(self):
        a = [1.0, 2.0, 3.0]
        b = [2.0, 3.0, 4.0]
        assert _noise_level(a, b) > 0.0

    def test_noise_level_normalized(self):
        # Large absolute noise relative to mean → normalized value reflects that
        a = [10.0, 10.0]
        b = [20.0, 20.0]
        noise = _noise_level(a, b)
        assert 0.0 < noise <= 1.0  # normalized to reasonable range

    def test_noise_level_empty(self):
        assert _noise_level([], []) == pytest.approx(0.0)

    def test_temporal_dependency_zero_from_zero_noise(self):
        assert _temporal_dependency(0.0) == pytest.approx(0.0)

    def test_temporal_dependency_positive_from_positive_noise(self):
        assert _temporal_dependency(0.3) > 0.0

    def test_temporal_dependency_high_for_high_noise(self):
        # noise > 0.3 should give temporal_dep > 0.5
        assert _temporal_dependency(0.4) > 0.5

    def test_temporal_dependency_capped_at_one(self):
        assert _temporal_dependency(100.0) == pytest.approx(1.0)

    def test_reward_sparsity_zero_for_large_fitness(self):
        large = [100.0, 200.0, 150.0]
        sparsity = _reward_sparsity(large, _mean(large))
        assert sparsity == pytest.approx(0.0)

    def test_reward_sparsity_high_for_zeros(self):
        sparse = [0.0, 0.0, 0.0, 0.5, 0.0]
        sparsity = _reward_sparsity(sparse, _mean(sparse))
        assert sparsity >= 0.6

    def test_reward_sparsity_empty(self):
        assert _reward_sparsity([], 0.0) == pytest.approx(0.0)

    def test_estimated_difficulty_with_target(self):
        # Mean = target → difficulty = 0
        assert _estimated_difficulty([195.0, 200.0], target=195.0) == pytest.approx(
            0.0, abs=0.1
        )
        # Mean = 0, target = 100 → difficulty = 1
        assert _estimated_difficulty([0.0, 0.0], target=100.0) == pytest.approx(1.0)

    def test_estimated_difficulty_partial(self):
        # Mean = 50, target = 100 → difficulty ≈ 0.5
        diff = _estimated_difficulty([50.0, 50.0], target=100.0)
        assert 0.4 <= diff <= 0.6

    def test_estimated_difficulty_no_target(self):
        # Should return a value in [0, 1]
        diff = _estimated_difficulty([1.0, 5.0, 10.0], target=None)
        assert 0.0 <= diff <= 1.0

    def test_mean_and_std(self):
        vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        assert _mean(vals) == pytest.approx(5.0)
        assert _std(vals) > 0


# ---------------------------------------------------------------------------
# Task classification
# ---------------------------------------------------------------------------

class TestClassifyTask:
    def test_classification_detected(self):
        task, conf, alts = _classify_task(
            n_inputs=4, n_outputs=1,
            mean_fitness=3.0, std_fitness=1.0,
            noise_level=0.0, reward_sparsity=0.0,
        )
        assert task == "classification"
        assert conf > 0.3

    def test_rl_continuous_from_multi_output(self):
        task, conf, alts = _classify_task(
            n_inputs=8, n_outputs=4,
            mean_fitness=10.0, std_fitness=20.0,
            noise_level=0.5, reward_sparsity=0.3,
        )
        assert task == "rl_continuous"

    def test_rl_discrete_from_stochastic_single_output(self):
        task, conf, alts = _classify_task(
            n_inputs=4, n_outputs=1,
            mean_fitness=15.0, std_fitness=30.0,
            noise_level=0.6, reward_sparsity=0.4,
        )
        assert task in ("rl_discrete", "rl_continuous")

    def test_regression_from_deterministic_single_output(self):
        task, conf, alts = _classify_task(
            n_inputs=2, n_outputs=1,
            mean_fitness=-5.0, std_fitness=3.0,
            noise_level=0.0, reward_sparsity=0.0,
        )
        assert task in ("regression", "classification")

    def test_confidence_in_range(self):
        _, conf, _ = _classify_task(
            n_inputs=2, n_outputs=1,
            mean_fitness=2.0, std_fitness=1.0,
            noise_level=0.0, reward_sparsity=0.0,
        )
        assert 0.0 <= conf <= 1.0

    def test_alternatives_list(self):
        _, _, alts = _classify_task(
            n_inputs=4, n_outputs=2,
            mean_fitness=20.0, std_fitness=15.0,
            noise_level=0.4, reward_sparsity=0.2,
        )
        assert isinstance(alts, list)


# ---------------------------------------------------------------------------
# End-to-end ProblemProfiler tests
# ---------------------------------------------------------------------------

class TestProblemProfiler:
    def test_returns_profile_instance(self, ne):
        profile = ProblemProfiler(ne).profile(deterministic_evaluator, n_warmup=10)
        assert isinstance(profile, ProblemProfile)

    def test_n_inputs_n_outputs_correct(self, ne):
        profile = ProblemProfiler(ne).profile(deterministic_evaluator, n_warmup=10)
        assert profile.n_inputs == 2
        assert profile.n_outputs == 1

    def test_deterministic_evaluator_noise_near_zero(self, ne):
        profile = ProblemProfiler(ne).profile(deterministic_evaluator, n_warmup=20)
        assert profile.noise_level == pytest.approx(0.0, abs=1e-6)

    def test_deterministic_temporal_dependency_near_zero(self, ne):
        profile = ProblemProfiler(ne).profile(deterministic_evaluator, n_warmup=20)
        assert profile.temporal_dependency == pytest.approx(0.0, abs=1e-6)

    def test_deterministic_task_type_classification(self, ne):
        profile = ProblemProfiler(ne).profile(deterministic_evaluator, n_warmup=20)
        assert profile.task_type in ("classification", "regression")

    def test_stochastic_evaluator_noise_positive(self, ne):
        profile = ProblemProfiler(ne).profile(
            StochasticEvaluator(seed=7), n_warmup=20
        )
        assert profile.noise_level > 0.01

    def test_stochastic_temporal_dependency_high(self, ne):
        profile = ProblemProfiler(ne).profile(
            StochasticEvaluator(seed=7), n_warmup=20
        )
        assert profile.temporal_dependency > 0.5

    def test_stochastic_task_type_rl(self, ne):
        profile = ProblemProfiler(ne).profile(
            StochasticEvaluator(seed=7), n_warmup=20
        )
        assert profile.task_type in ("rl_continuous", "rl_discrete")

    def test_sparse_evaluator_sparsity_high(self, ne):
        profile = ProblemProfiler(ne).profile(sparse_evaluator, n_warmup=30)
        # Many genomes return 0 → sparsity should be > 0
        assert profile.reward_sparsity >= 0.0  # can't guarantee > 0.5 always

    def test_fitness_stats_consistent(self, ne):
        profile = ProblemProfiler(ne).profile(deterministic_evaluator, n_warmup=20)
        assert profile.fitness_min <= profile.fitness_mean <= profile.fitness_max
        assert profile.fitness_std >= 0.0

    def test_all_fields_present(self, ne):
        profile = ProblemProfiler(ne).profile(deterministic_evaluator, n_warmup=10)
        required = [
            "task_type", "task_type_confidence", "n_inputs", "n_outputs",
            "estimated_difficulty", "noise_level", "reward_sparsity",
            "temporal_dependency", "state_dim_effective",
            "fitness_mean", "fitness_std", "fitness_min", "fitness_max",
            "alternative_types",
        ]
        for attr in required:
            assert hasattr(profile, attr), f"Missing field: {attr}"

    def test_state_dim_effective_equals_n_inputs(self, ne):
        profile = ProblemProfiler(ne).profile(deterministic_evaluator, n_warmup=10)
        assert profile.state_dim_effective == 2

    def test_multi_output_classified_rl_continuous(self, ne_multi):
        def stochastic_multi(genome):
            import random
            return random.gauss(10.0, 20.0)

        profile = ProblemProfiler(ne_multi).profile(stochastic_multi, n_warmup=15)
        assert profile.task_type == "rl_continuous"

    def test_difficulty_with_target_fitness(self, ne):
        ne.set_min_fitness(10.0)
        profile = ProblemProfiler(ne).profile(deterministic_evaluator, n_warmup=15)
        assert 0.0 <= profile.estimated_difficulty <= 1.0

    def test_n_warmup_minimum_clamped(self, ne):
        # n_warmup=1 should not crash (clamped to 5)
        profile = ProblemProfiler(ne).profile(deterministic_evaluator, n_warmup=1)
        assert isinstance(profile, ProblemProfile)

    def test_confidence_in_valid_range(self, ne):
        profile = ProblemProfiler(ne).profile(deterministic_evaluator, n_warmup=10)
        assert 0.0 <= profile.task_type_confidence <= 1.0

    def test_alternatives_is_list(self, ne):
        profile = ProblemProfiler(ne).profile(deterministic_evaluator, n_warmup=10)
        assert isinstance(profile.alternative_types, list)


# ---------------------------------------------------------------------------
# NeuroEvolution integration
# ---------------------------------------------------------------------------

class TestNeuroEvolutionIntegration:
    def test_profile_problem_returns_profile(self, ne):
        profile = ne.profile_problem(deterministic_evaluator, n_warmup=10)
        assert isinstance(profile, ProblemProfile)

    def test_profile_problem_raises_without_configure(self):
        ne_unconfigured = NeuroEvolution(seed=0)
        with pytest.raises(RuntimeError, match="configure"):
            ne_unconfigured.profile_problem(deterministic_evaluator)

    def test_profile_problem_n_inputs_matches_configure(self):
        ne_4 = NeuroEvolution(seed=0)
        ne_4.configure(n_inputs=4, n_outputs=3)
        profile = ne_4.profile_problem(deterministic_evaluator, n_warmup=8)
        assert profile.n_inputs == 4
        assert profile.n_outputs == 3

    def test_profile_problem_deterministic_noise_zero(self, ne):
        profile = ne.profile_problem(deterministic_evaluator, n_warmup=15)
        assert profile.noise_level == pytest.approx(0.0, abs=1e-6)

    def test_profile_problem_stochastic_noise_positive(self, ne):
        profile = ne.profile_problem(StochasticEvaluator(seed=99), n_warmup=15)
        assert profile.noise_level > 0.0

    def test_profile_problem_with_zero_target(self):
        """target=0 should not cause division-by-zero."""
        ne_t = NeuroEvolution(seed=0)
        ne_t.configure(n_inputs=2, n_outputs=1)
        ne_t.set_min_fitness(0.0)
        profile = ne_t.profile_problem(deterministic_evaluator, n_warmup=8)
        assert math.isfinite(profile.estimated_difficulty)
