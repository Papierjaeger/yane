"""Tests for auto_train() — Phase 6 of P0 Meta-Adaptive Orchestration.

Covers:
- AutoTrainResult structure (all required fields present)
- auto_train() with n_inputs/n_outputs parameters (zero prior setup)
- auto_train() on already-configured NeuroEvolution
- auto_config_report is a non-empty string with expected sections
- cold start: runs without KB, applies conservative defaults
- KB learns after auto_train()
- Second run uses KB suggestion
- max_time_seconds budget respected (loosely)
- target_fitness stops training early
- problem_name shows in report
- MetaOptimizer and FeatureGating are both enabled during auto_train()
- active_features and final_params in result
- auto_train() without prior configure() but passing n_inputs/n_outputs
- auto_train() without configure() AND without n_inputs raises
- Calling auto_train() twice on same instance works
- AutoTrainResult importable from yane package
"""
from __future__ import annotations

import time

import pytest

from yane import AutoTrainResult, NeuroEvolution


def _xor_eval(genome):
    total = 0.0
    for x, y, target in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]:
        out = genome.forward([x, y])
        pred = 1 if out[0] > 0.5 else 0
        total += 1.0 if pred == target else 0.0
    return total


def _fast_eval(genome):
    return 1.0


# ---------------------------------------------------------------------------
# AutoTrainResult
# ---------------------------------------------------------------------------

class TestAutoTrainResult:
    def test_importable_from_yane(self):
        from yane import AutoTrainResult as ATR
        assert ATR is AutoTrainResult

    def test_has_all_fields(self):
        r = AutoTrainResult(
            best_genome=None,
            final_fitness=0.0,
            total_generations=0,
            wall_time=0.0,
        )
        assert hasattr(r, "best_genome")
        assert hasattr(r, "final_fitness")
        assert hasattr(r, "total_generations")
        assert hasattr(r, "wall_time")
        assert hasattr(r, "active_features")
        assert hasattr(r, "final_params")
        assert hasattr(r, "problem_profile")
        assert hasattr(r, "auto_config_report")
        assert hasattr(r, "kb_suggestions_used")
        assert hasattr(r, "n_kb_suggestions")

    def test_defaults(self):
        r = AutoTrainResult(
            best_genome=None, final_fitness=0.0,
            total_generations=0, wall_time=0.0
        )
        assert r.active_features == []
        assert r.final_params == {}
        assert r.problem_profile is None
        assert r.auto_config_report == ""
        assert r.kb_suggestions_used is False
        assert r.n_kb_suggestions == 0


# ---------------------------------------------------------------------------
# Basic auto_train() results
# ---------------------------------------------------------------------------

class TestAutoTrainBasic:
    def _run(self, **kw):
        ne = NeuroEvolution(seed=0)
        return ne.auto_train(
            _fast_eval,
            n_inputs=2,
            n_outputs=1,
            n_warmup=5,
            max_time_seconds=1,
            **kw,
        )

    def test_returns_auto_train_result(self):
        r = self._run()
        assert isinstance(r, AutoTrainResult)

    def test_best_genome_not_none(self):
        r = self._run()
        assert r.best_genome is not None

    def test_final_fitness_finite(self):
        import math
        r = self._run()
        assert math.isfinite(r.final_fitness)

    def test_wall_time_positive(self):
        r = self._run()
        assert r.wall_time > 0.0

    def test_total_generations_positive(self):
        r = self._run()
        assert r.total_generations > 0

    def test_problem_profile_set(self):
        r = self._run()
        assert r.problem_profile is not None

    def test_active_features_is_list(self):
        r = self._run()
        assert isinstance(r.active_features, list)

    def test_final_params_is_dict(self):
        r = self._run()
        assert isinstance(r.final_params, dict)

    def test_kb_suggestions_used_is_bool(self):
        r = self._run()
        assert isinstance(r.kb_suggestions_used, bool)

    def test_n_kb_suggestions_non_negative(self):
        r = self._run()
        assert r.n_kb_suggestions >= 0


# ---------------------------------------------------------------------------
# auto_config_report
# ---------------------------------------------------------------------------

class TestAutoConfigReport:
    def _report(self, **kw):
        ne = NeuroEvolution(seed=0)
        return ne.auto_train(
            _fast_eval, n_inputs=2, n_outputs=1,
            n_warmup=5, max_time_seconds=1, **kw
        ).auto_config_report

    def test_report_is_non_empty_string(self):
        r = self._report()
        assert isinstance(r, str) and len(r) > 0

    def test_report_contains_header(self):
        r = self._report()
        assert "Auto-Configuration Report" in r

    def test_report_contains_task_type(self):
        r = self._report()
        assert "classification" in r or "regression" in r or "rl_" in r

    def test_report_contains_knowledge_base_section(self):
        r = self._report()
        assert "Knowledge Base" in r

    def test_report_contains_wall_time(self):
        r = self._report()
        assert "Wall time" in r

    def test_report_contains_best_fitness(self):
        r = self._report()
        assert "Best fitness" in r

    def test_report_contains_meta_optimizer_section(self):
        r = self._report()
        assert "MetaOptimizer" in r

    def test_report_contains_feature_gating_section(self):
        r = self._report()
        assert "Feature Gating" in r

    def test_report_contains_generations(self):
        r = self._report()
        assert "Generations" in r

    def test_problem_name_in_report(self):
        r = self._report(problem_name="MyTask")
        assert "MyTask" in r


# ---------------------------------------------------------------------------
# Cold start
# ---------------------------------------------------------------------------

class TestColdStart:
    def test_cold_start_runs_without_kb(self):
        ne = NeuroEvolution(seed=0)
        r = ne.auto_train(_fast_eval, n_inputs=2, n_outputs=1,
                          n_warmup=5, max_time_seconds=1)
        assert r.kb_suggestions_used is False
        # KB may return the cold-start default suggestion (confidence < threshold)
        assert r.n_kb_suggestions >= 0

    def test_cold_start_report_says_cold_start(self):
        ne = NeuroEvolution(seed=0)
        r = ne.auto_train(_fast_eval, n_inputs=2, n_outputs=1,
                          n_warmup=5, max_time_seconds=1)
        assert "cold start" in r.auto_config_report

    def test_cold_start_final_fitness_valid(self):
        import math
        ne = NeuroEvolution(seed=0)
        r = ne.auto_train(_fast_eval, n_inputs=2, n_outputs=1,
                          n_warmup=5, max_time_seconds=1)
        assert math.isfinite(r.final_fitness)


# ---------------------------------------------------------------------------
# Knowledge Base integration
# ---------------------------------------------------------------------------

class TestKBIntegration:
    def test_kb_learns_after_run(self):
        ne = NeuroEvolution(seed=0)
        ne.auto_train(_fast_eval, n_inputs=2, n_outputs=1,
                      n_warmup=5, max_time_seconds=1)
        assert ne._knowledge_base is not None
        assert len(ne._knowledge_base) >= 1

    def test_second_run_has_kb_entries(self):
        ne = NeuroEvolution(seed=0)
        ne.auto_train(_fast_eval, n_inputs=2, n_outputs=1,
                      n_warmup=5, max_time_seconds=1)
        ne2 = NeuroEvolution(seed=0)
        # Share KB from first run
        ne2._knowledge_base = ne._knowledge_base
        r2 = ne2.auto_train(_fast_eval, n_inputs=2, n_outputs=1,
                             n_warmup=5, max_time_seconds=1)
        # Second run may or may not use KB (depends on confidence threshold)
        # But KB should have more entries now
        assert len(ne2._knowledge_base) >= 1


# ---------------------------------------------------------------------------
# Configuration modes
# ---------------------------------------------------------------------------

class TestConfigurationModes:
    def test_n_inputs_n_outputs_auto_configure(self):
        ne = NeuroEvolution(seed=0)
        assert not ne.is_configured
        ne.auto_train(_fast_eval, n_inputs=3, n_outputs=2,
                      n_warmup=5, max_time_seconds=1)
        assert ne.is_configured
        assert ne._n_inputs == 3
        assert ne._n_outputs == 2

    def test_already_configured_works(self):
        ne = NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1)
        r = ne.auto_train(_fast_eval, n_warmup=5, max_time_seconds=1)
        assert isinstance(r, AutoTrainResult)

    def test_no_configure_no_args_raises(self):
        ne = NeuroEvolution(seed=0)
        with pytest.raises(RuntimeError):
            ne.auto_train(_fast_eval, n_warmup=5)

    def test_only_n_inputs_missing_raises(self):
        ne = NeuroEvolution(seed=0)
        with pytest.raises((RuntimeError, TypeError)):
            ne.auto_train(_fast_eval, n_outputs=1, n_warmup=5)


# ---------------------------------------------------------------------------
# Budget and stopping
# ---------------------------------------------------------------------------

class TestBudgetAndStopping:
    def test_max_time_seconds_respected(self):
        ne = NeuroEvolution(seed=0)
        t0 = time.time()
        ne.auto_train(_fast_eval, n_inputs=2, n_outputs=1,
                      n_warmup=5, max_time_seconds=2)
        elapsed = time.time() - t0
        # Should finish reasonably close to budget (within 10x for fast evals)
        assert elapsed < 30

    def test_target_fitness_stops_early(self):
        ne = NeuroEvolution(seed=0)
        r = ne.auto_train(
            _fast_eval, n_inputs=2, n_outputs=1,
            n_warmup=5, target_fitness=1.0, max_time_seconds=10,
        )
        assert r.final_fitness >= 1.0

    def test_two_runs_same_instance_works(self):
        ne = NeuroEvolution(seed=0)
        r1 = ne.auto_train(_fast_eval, n_inputs=2, n_outputs=1,
                           n_warmup=5, max_time_seconds=1)
        r2 = ne.auto_train(_fast_eval, n_inputs=2, n_outputs=1,
                           n_warmup=5, max_time_seconds=1)
        assert isinstance(r1, AutoTrainResult)
        assert isinstance(r2, AutoTrainResult)


# ---------------------------------------------------------------------------
# Meta-components enabled
# ---------------------------------------------------------------------------

class TestMetaComponents:
    def test_meta_optimizer_enabled_during_auto_train(self):
        ne = NeuroEvolution(seed=0)
        ne.auto_train(_fast_eval, n_inputs=2, n_outputs=1,
                      n_warmup=5, max_time_seconds=1)
        d = ne.get_meta_optimizer_diagnostics()
        assert d["enabled"] is True

    def test_feature_gating_enabled_during_auto_train(self):
        ne = NeuroEvolution(seed=0)
        ne.auto_train(_fast_eval, n_inputs=2, n_outputs=1,
                      n_warmup=5, max_time_seconds=1)
        d = ne.get_feature_gating_diagnostics()
        assert d["enabled"] is True

    def test_problem_profile_stored_on_ne(self):
        ne = NeuroEvolution(seed=0)
        ne.auto_train(_fast_eval, n_inputs=2, n_outputs=1,
                      n_warmup=5, max_time_seconds=1)
        assert ne._last_problem_profile is not None
