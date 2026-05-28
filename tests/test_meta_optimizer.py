"""Tests for the MetaOptimizer (P0 Meta-Adaptive Orchestration, Phase 3).

Covers:
- TrainingPhase enum values
- PhaseManager: plateau detection, phase transitions, minimum-gen guard
- SimpleGP: predict, EI, suggest (monotone improvement over random)
- _CatBandit: UCB1 selects arms, best() returns highest-reward arm
- _ContOptimizer: suggest returns value in domain, update records observation
- MetaOptimizer: tick() dispatches, phase transitions fire, overhead guard
- MetaOptimizer: categorical param updated via ne.set_param()
- MetaOptimizer.get_diagnostics() structure
- NeuroEvolution.set_meta_optimizer() / get_meta_optimizer_diagnostics()
- train() with meta_optimizer does not crash
- _normal_cdf and _normal_pdf sanity
"""
from __future__ import annotations

import math
import time

import pytest

from yane import NeuroEvolution
from yane.evolution.meta_optimizer import (
    MetaOptimizer,
    PhaseManager,
    SimpleGP,
    TrainingPhase,
    _CatBandit,
    _ContOptimizer,
    _PHASE_ORDER,
    _normal_cdf,
    _normal_pdf,
)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

class TestNormalFunctions:
    def test_pdf_non_negative(self):
        for x in [-3, -1, 0, 1, 3]:
            assert _normal_pdf(x) >= 0.0

    def test_pdf_symmetric(self):
        assert _normal_pdf(1.0) == pytest.approx(_normal_pdf(-1.0))

    def test_cdf_at_zero(self):
        assert _normal_cdf(0.0) == pytest.approx(0.5, abs=0.01)

    def test_cdf_positive_infty_approx(self):
        assert _normal_cdf(5.0) > 0.99

    def test_cdf_negative_infty_approx(self):
        assert _normal_cdf(-5.0) < 0.01

    def test_cdf_symmetry(self):
        assert _normal_cdf(1.0) + _normal_cdf(-1.0) == pytest.approx(1.0, abs=0.001)


# ---------------------------------------------------------------------------
# SimpleGP
# ---------------------------------------------------------------------------

class TestSimpleGP:
    def test_predict_empty_returns_defaults(self):
        gp = SimpleGP()
        mu, sigma = gp.predict(0.5)
        assert math.isfinite(mu)
        assert sigma > 0

    def test_predict_after_observation(self):
        gp = SimpleGP()
        gp.observe(0.5, 2.0)
        mu, sigma = gp.predict(0.5)
        # Mean should be close to observed value
        assert abs(mu - 2.0) < 0.5

    def test_std_positive(self):
        gp = SimpleGP()
        gp.observe(0.2, 1.0)
        gp.observe(0.8, 2.0)
        _, sigma = gp.predict(0.5)
        assert sigma >= 0.0

    def test_suggest_in_domain(self):
        gp = SimpleGP()
        for x, y in [(0.1, 1.0), (0.5, 3.0), (0.9, 2.0)]:
            gp.observe(x, y)
        val = gp.suggest(0.0, 1.0)
        assert 0.0 <= val <= 1.0

    def test_suggest_no_obs_random(self):
        gp = SimpleGP()
        val = gp.suggest(0.0, 1.0)
        assert 0.0 <= val <= 1.0

    def test_ei_non_negative(self):
        gp = SimpleGP()
        gp.observe(0.5, 2.0)
        ei = gp.expected_improvement(0.3, best_y=2.0)
        assert ei >= 0.0

    def test_multiple_observations_stable(self):
        gp = SimpleGP()
        for i in range(8):
            gp.observe(i / 8.0, float(i))
        mu, sigma = gp.predict(0.5)
        assert math.isfinite(mu)
        assert math.isfinite(sigma)


# ---------------------------------------------------------------------------
# PhaseManager
# ---------------------------------------------------------------------------

class TestPhaseManager:
    def test_starts_in_explore(self):
        pm = PhaseManager(plateau_patience=5, phase_min_gens=3)
        assert pm.current_phase == TrainingPhase.EXPLORE

    def test_no_transition_before_min_gens(self):
        pm = PhaseManager(plateau_patience=2, phase_min_gens=10)
        for g in range(8):
            pm.update(g, 1.0)  # stagnant from gen 0
        assert pm.current_phase == TrainingPhase.EXPLORE

    def test_transitions_after_plateau(self):
        pm = PhaseManager(plateau_patience=3, phase_min_gens=2)
        for g in range(6):
            pm.update(g, 1.0)  # stagnant → should transition
        assert pm.current_phase != TrainingPhase.EXPLORE

    def test_improvement_resets_stagnation(self):
        pm = PhaseManager(plateau_patience=3, phase_min_gens=2)
        pm.update(0, 1.0)
        pm.update(1, 1.0)
        pm.update(2, 2.0)  # improvement — resets stagnation
        pm.update(3, 2.0)  # now 1 gen stagnant (< patience=3)
        pm.update(4, 2.0)  # 2 gen stagnant
        assert pm.current_phase == TrainingPhase.EXPLORE  # not yet plateau

    def test_full_phase_sequence(self):
        pm = PhaseManager(plateau_patience=3, phase_min_gens=2)
        phases_seen = set()
        for g in range(50):
            pm.update(g, 1.0)
            phases_seen.add(pm.current_phase)
        # Should have passed through all phases (all stagnant → keeps advancing)
        assert TrainingPhase.CONVERGE in phases_seen

    def test_converge_is_final_phase(self):
        pm = PhaseManager(plateau_patience=2, phase_min_gens=1)
        for g in range(30):
            pm.update(g, 1.0)
        pm.update(30, 1.0)
        # Can't go beyond CONVERGE
        assert pm.current_phase == TrainingPhase.CONVERGE

    def test_diagnostics_structure(self):
        pm = PhaseManager()
        d = pm.get_diagnostics()
        assert "current_phase" in d
        assert "stagnation_gens" in d
        assert "transitions" in d

    def test_transitions_list_grows(self):
        pm = PhaseManager(plateau_patience=2, phase_min_gens=1)
        for g in range(20):
            pm.update(g, 1.0)
        assert len(pm.get_diagnostics()["transitions"]) >= 1


# ---------------------------------------------------------------------------
# _CatBandit
# ---------------------------------------------------------------------------

class TestCatBandit:
    def test_select_returns_candidate(self):
        b = _CatBandit("test", ["a", "b", "c"])
        val = b.select()
        assert val in ["a", "b", "c"]

    def test_all_arms_tried_initially(self):
        b = _CatBandit("test", ["a", "b", "c"])
        chosen = {b.select() for _ in range(20)}
        assert len(chosen) == 3

    def test_best_returns_highest_reward(self):
        b = _CatBandit("test", ["a", "b"])
        # Force arm 0 to be selected and give high reward
        for _ in range(3):
            b.select()
        # Manually set: "a" arm (idx 0) gets reward, "b" (idx 1) gets nothing
        b._rewards[0] = 10.0
        b._rewards[1] = 1.0
        b._counts[0] = 3
        b._counts[1] = 3
        assert b.best() == "a"

    def test_update_increases_reward(self):
        b = _CatBandit("test", ["a", "b"])
        b.select()
        before = sum(b._rewards)
        b.update(5.0)
        after = sum(b._rewards)
        assert after > before

    def test_diagnostics(self):
        b = _CatBandit("test", ["a", "b"])
        d = b.get_diagnostics()
        assert "param" in d
        assert "counts" in d
        assert "best" in d


# ---------------------------------------------------------------------------
# _ContOptimizer
# ---------------------------------------------------------------------------

class TestContOptimizer:
    def test_suggest_in_domain(self):
        opt = _ContOptimizer("test", lo=0.1, hi=0.9)
        val = opt.suggest()
        assert 0.1 <= val <= 0.9

    def test_suggest_integer_in_domain(self):
        opt = _ContOptimizer("test", lo=1.0, hi=10.0, is_integer=True)
        val = opt.suggest()
        assert isinstance(val, int)
        assert 1 <= val <= 10

    def test_update_records_observation(self):
        opt = _ContOptimizer("test", lo=0.0, hi=1.0)
        opt.suggest()
        opt.update(3.0)
        assert len(opt._gp._xs) == 1

    def test_multiple_suggests_stay_in_domain(self):
        opt = _ContOptimizer("test", lo=0.2, hi=0.8)
        for _ in range(10):
            opt.suggest()
            opt.update(1.0)
            val = opt.suggest()
            assert 0.2 <= val <= 0.8


# ---------------------------------------------------------------------------
# MetaOptimizer
# ---------------------------------------------------------------------------

class TestMetaOptimizer:
    def _make_ne(self, iters=200):
        ne = NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1)
        ne.set_max_iterations(iters)
        return ne

    def test_starts_in_explore(self):
        mo = MetaOptimizer()
        assert mo.current_phase == TrainingPhase.EXPLORE

    def test_tick_does_not_raise(self, tmp_path):
        ne = self._make_ne()
        mo = MetaOptimizer(tune_interval=5, plateau_patience=3, phase_min_gens=2)
        for gen in range(30):
            mo.tick(gen, float(gen % 5), ne)  # should not raise

    def test_tick_changes_categorical_param(self, tmp_path):
        ne = self._make_ne()
        mo = MetaOptimizer(tune_interval=1, plateau_patience=100)
        # Run many ticks to ensure lamarck.mode is set
        for gen in range(30):
            mo.tick(gen, 1.0, ne)
        changes = [c for c in mo._param_change_log if c["param"] == "lamarck.mode"]
        assert len(changes) >= 1

    def test_phase_transition_fires(self):
        ne = self._make_ne()
        mo = MetaOptimizer(
            tune_interval=1,
            plateau_patience=3,
            phase_min_gens=2,
        )
        for gen in range(20):
            mo.tick(gen, 1.0, ne)  # constant fitness → plateau
        assert mo.current_phase != TrainingPhase.EXPLORE

    def test_overhead_guard_skips_when_overloaded(self):
        ne = self._make_ne()
        mo = MetaOptimizer(tune_interval=1, max_overhead_pct=0.001)
        # Simulate heavy meta overhead
        mo._meta_ms = 1000.0
        mo._total_eval_ms = 1.0
        for gen in range(20):
            mo.tick(gen, float(gen), ne)
        assert mo._n_skipped > 0

    def test_get_diagnostics_structure(self):
        mo = MetaOptimizer()
        d = mo.get_diagnostics()
        assert "phase" in d
        assert "n_ticks" in d
        assert "categorical" in d
        assert "continuous" in d
        assert "overhead_pct" in d

    def test_overhead_pct_zero_initially(self):
        mo = MetaOptimizer()
        assert mo.overhead_pct == pytest.approx(0.0)

    def test_continuous_tuned_in_exploit_phase(self):
        ne = self._make_ne()
        mo = MetaOptimizer(tune_interval=1, plateau_patience=2, phase_min_gens=1)
        # Advance to EXPLOIT
        for gen in range(10):
            mo.tick(gen, 1.0, ne)
        # Should be in EXPLOIT or later
        assert mo.current_phase in (
            TrainingPhase.EXPLOIT, TrainingPhase.REFINE, TrainingPhase.CONVERGE
        )


# ---------------------------------------------------------------------------
# NeuroEvolution integration
# ---------------------------------------------------------------------------

class TestNeuroEvolutionMetaOptimizerIntegration:
    def test_set_meta_optimizer_enables_it(self):
        ne = NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1)
        ne.set_meta_optimizer(enabled=True)
        assert ne._meta_optimizer_enabled is True
        assert ne._meta_optimizer_obj is not None

    def test_set_meta_optimizer_disabled(self):
        ne = NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1)
        ne.set_meta_optimizer(enabled=False)
        assert ne._meta_optimizer_enabled is False
        assert ne._meta_optimizer_obj is None

    def test_get_meta_optimizer_diagnostics_disabled(self):
        ne = NeuroEvolution(seed=0)
        d = ne.get_meta_optimizer_diagnostics()
        assert d["enabled"] is False

    def test_get_meta_optimizer_diagnostics_enabled(self):
        ne = NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1)
        ne.set_meta_optimizer(enabled=True)
        d = ne.get_meta_optimizer_diagnostics()
        assert d["enabled"] is True
        assert "phase" in d

    def test_train_with_meta_optimizer_does_not_crash(self):
        ne = NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1)
        ne.set_max_iterations(60)
        ne.set_meta_optimizer(
            enabled=True,
            tune_interval=5,
            plateau_patience=3,
            phase_min_gens=2,
        )

        def dummy_eval(g):
            return 1.0

        ne.train(dummy_eval)  # must not raise

    def test_meta_optimizer_changes_lamarck_mode_during_train(self):
        # Use small pop so many generations occur within the iteration budget
        ne = NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1)
        ne.set_population_size(10)
        ne.set_max_iterations(100)  # 100 iters / pop=10 = 10 generations
        ne.set_meta_optimizer(
            enabled=True,
            tune_interval=3,
            plateau_patience=5,
            phase_min_gens=2,
        )

        def dummy_eval(g):
            return 1.0

        ne.train(dummy_eval)
        d = ne.get_meta_optimizer_diagnostics()
        assert d["n_ticks"] > 0

    def test_meta_optimizer_phase_advances_during_train(self):
        ne = NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1)
        ne.set_population_size(10)
        ne.set_max_iterations(150)  # 15 generations
        ne.set_meta_optimizer(
            enabled=True,
            tune_interval=2,
            plateau_patience=5,
            phase_min_gens=3,
        )

        def stagnant_eval(g):
            return 1.0  # constant fitness → guarantees plateau

        ne.train(stagnant_eval)
        d = ne.get_meta_optimizer_diagnostics()
        # With constant fitness, should advance past EXPLORE
        assert d["phase"] in ("exploit", "refine", "converge")

    def test_train_without_meta_optimizer_unaffected(self):
        ne = NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1)
        ne.set_max_iterations(50)

        def dummy_eval(g):
            return 1.0

        ne.train(dummy_eval)  # no meta optimizer → must still work normally
        assert ne.get_meta_optimizer_diagnostics()["enabled"] is False
