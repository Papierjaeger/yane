"""Tests for FeatureGate (P0 Meta-Adaptive Orchestration, Phase 5).

Covers:
- FeatureRecord: UCB1 score, update
- FeatureGate registration and initial state
- Tick triggers first test at generation 0
- Test-interval respected (no early starts)
- INACTIVE → TESTING → ACTIVE when fitness improves
- INACTIVE → TESTING → INACTIVE when fitness flat
- enable_fn / disable_fn called at correct transitions
- degradation_level increases on stagnation; DISABLED at >= 0.8
- degrade_fn called on each degradation step
- max_concurrent not exceeded (active + testing combined)
- Auto-reactivation after reactivation_delay
- Explicit reactivate()
- Error-resilience: enable_fn / disable_fn exceptions don't crash tick
- get_diagnostics structure
- get_active_features
- change_log records transitions
- NeuroEvolution.set_auto_features / get_feature_gating_diagnostics
- train() with feature gating doesn't crash
"""
from __future__ import annotations

import pytest

from yane import NeuroEvolution
from yane.evolution.feature_gating import FeatureGate, FeatureRecord, FeatureStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fg(**kw) -> FeatureGate:
    """Defaults tuned for fast tests: short intervals, large reactivation guard."""
    defaults = dict(
        test_interval=1,
        test_duration=1,
        impact_threshold=0.0,
        degradation_patience=2,
        reactivation_delay=1000,   # large so tests don't see surprise reactivation
        max_concurrent=3,
    )
    defaults.update(kw)
    return FeatureGate(**defaults)


def _register(fg, name, enable_calls=None, disable_calls=None, degrade_calls=None):
    """Register a feature that logs calls into the supplied lists."""
    def _en(ne):
        if enable_calls is not None:
            enable_calls.append(name)

    def _dis(ne):
        if disable_calls is not None:
            disable_calls.append(name)

    def _deg(ne, level):
        if degrade_calls is not None:
            degrade_calls.append((name, level))

    fg.register(name, enable_fn=_en, disable_fn=_dis, degrade_fn=_deg)


NE = None  # stand-in NeuroEvolution reference for unit tests


def _activate(fg, name, gen_start=0):
    """Register *name* and drive it to ACTIVE state; returns last gen used."""
    _register(fg, name)
    fg.tick(gen_start, 0.0, NE)          # start test; fitness_before=0.0
    fg.tick(gen_start + 1, 1.0, NE)      # finish: impact=1.0>0 → ACTIVE
    assert fg._features[name].status == FeatureStatus.ACTIVE
    return gen_start + 1


def _disable(fg, name, gen_start=0, patience=1):
    """Register *name*, activate it, then drive it to DISABLED. Returns last gen used."""
    g = _activate(fg, name, gen_start)
    # Stagnate until DISABLED (4 rounds at step 0.2 per patience block)
    while fg._features[name].status != FeatureStatus.DISABLED and g < gen_start + 200:
        g += 1
        fg.tick(g, 1.0, NE)   # constant fitness → stagnation
    assert fg._features[name].status == FeatureStatus.DISABLED
    return g


# ---------------------------------------------------------------------------
# FeatureRecord
# ---------------------------------------------------------------------------

class TestFeatureRecord:
    def test_initial_status_inactive(self):
        rec = FeatureRecord("x", lambda ne: None, lambda ne: None)
        assert rec.status == FeatureStatus.INACTIVE

    def test_ucb1_score_unexplored_is_infinite(self):
        rec = FeatureRecord("x", lambda ne: None, lambda ne: None)
        assert rec.ucb1_score(10) == float("inf")

    def test_ucb1_score_after_update(self):
        rec = FeatureRecord("x", lambda ne: None, lambda ne: None)
        rec.update_ucb1(1.0)
        score = rec.ucb1_score(total_selections=5, c=2.0)
        assert score > 0.0

    def test_update_ucb1_increments_trials(self):
        rec = FeatureRecord("x", lambda ne: None, lambda ne: None)
        rec.update_ucb1(0.5)
        rec.update_ucb1(0.5)
        assert rec.n_trials == 2
        assert rec.total_reward == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestFeatureGateRegistration:
    def test_register_adds_feature(self):
        fg = _make_fg()
        _register(fg, "feat1")
        assert "feat1" in fg._features

    def test_register_duplicate_raises(self):
        fg = _make_fg()
        _register(fg, "feat1")
        with pytest.raises(ValueError):
            _register(fg, "feat1")

    def test_register_multiple_features(self):
        fg = _make_fg()
        for name in ["a", "b", "c"]:
            _register(fg, name)
        assert len(fg._features) == 3

    def test_initial_status_all_inactive(self):
        fg = _make_fg()
        for name in ["a", "b"]:
            _register(fg, name)
        for rec in fg._features.values():
            assert rec.status == FeatureStatus.INACTIVE


# ---------------------------------------------------------------------------
# Test-interval and first-tick selection
# ---------------------------------------------------------------------------

class TestFeatureGateSelection:
    def test_tick_starts_first_test_immediately(self):
        fg = _make_fg(test_interval=1)
        _register(fg, "feat1")
        fg.tick(0, 0.0, NE)
        assert fg._features["feat1"].status == FeatureStatus.TESTING

    def test_test_interval_respected(self):
        # _last_test_gen resets on test-finish (gen 1) → next test at 1+5=6
        fg = _make_fg(test_interval=5, test_duration=1)
        _register(fg, "feat1")
        fg.tick(0, 0.0, NE)   # gen 0 → starts test
        fg.tick(1, 0.0, NE)   # finish test → INACTIVE, _last_test_gen=1
        for g in range(2, 6):
            fg.tick(g, 0.0, NE)  # (g-1) < 5 → no new test
        assert fg._features["feat1"].status == FeatureStatus.INACTIVE
        # gen 6: (6-1)=5 >= 5 → starts new test
        fg.tick(6, 0.0, NE)
        assert fg._features["feat1"].status == FeatureStatus.TESTING

    def test_ucb1_explores_all_features(self):
        fg = _make_fg(test_interval=1, test_duration=1, max_concurrent=5)
        for name in ["a", "b", "c"]:
            _register(fg, name)
        selected = set()
        for g in range(60):
            fg.tick(g, float(g % 2), NE)  # alternate fitness
            for n, r in fg._features.items():
                if r.status != FeatureStatus.INACTIVE:
                    selected.add(n)
        assert selected == {"a", "b", "c"}

    def test_no_candidate_when_all_occupied(self):
        fg = _make_fg(test_interval=1, test_duration=10, max_concurrent=1)
        _register(fg, "feat1")
        _register(fg, "feat2")
        fg.tick(0, 0.0, NE)
        # Only one should be TESTING (max_concurrent=1)
        testing = [n for n, r in fg._features.items() if r.status == FeatureStatus.TESTING]
        assert len(testing) == 1


# ---------------------------------------------------------------------------
# Test lifecycle
# ---------------------------------------------------------------------------

class TestFeatureGateTestLifecycle:
    def test_high_impact_feature_becomes_active(self):
        fg = _make_fg()
        _register(fg, "feat1")
        fg.tick(0, 0.0, NE)     # start test, fitness_before=0.0
        fg.tick(1, 5.0, NE)     # finish: impact=5.0 > 0.0 → ACTIVE
        assert fg._features["feat1"].status == FeatureStatus.ACTIVE

    def test_no_impact_feature_stays_inactive(self):
        fg = _make_fg()
        _register(fg, "feat1")
        fg.tick(0, 5.0, NE)     # start test, fitness_before=5.0
        fg.tick(1, 5.0, NE)     # finish: impact=0.0 not > 0.0 → INACTIVE
        # _last_test_gen reset to gen 1; (1-1)=0 < 1 → no new test
        assert fg._features["feat1"].status == FeatureStatus.INACTIVE

    def test_enable_fn_called_when_test_starts(self):
        fg = _make_fg()
        en = []
        _register(fg, "feat1", enable_calls=en)
        fg.tick(0, 0.0, NE)
        assert en == ["feat1"]

    def test_disable_fn_called_when_no_impact(self):
        fg = _make_fg()
        dis = []
        _register(fg, "feat1", disable_calls=dis)
        fg.tick(0, 5.0, NE)
        fg.tick(1, 5.0, NE)    # no impact → disable called
        assert "feat1" in dis

    def test_disable_fn_not_called_when_active(self):
        fg = _make_fg()
        dis = []
        _register(fg, "feat1", disable_calls=dis)
        fg.tick(0, 0.0, NE)
        fg.tick(1, 5.0, NE)    # positive impact → ACTIVE, disable NOT called
        assert dis == []

    def test_test_duration_respected(self):
        fg = _make_fg(test_interval=1, test_duration=3)
        _register(fg, "feat1")
        fg.tick(0, 0.0, NE)     # start test; end_gen=3
        fg.tick(1, 5.0, NE)     # gen 1 < 3 → still testing
        assert fg._features["feat1"].status == FeatureStatus.TESTING
        fg.tick(2, 5.0, NE)     # gen 2 < 3 → still testing
        assert fg._features["feat1"].status == FeatureStatus.TESTING
        fg.tick(3, 5.0, NE)     # gen 3 >= 3 → finished → ACTIVE
        assert fg._features["feat1"].status == FeatureStatus.ACTIVE

    def test_ucb1_reward_positive_for_active(self):
        fg = _make_fg()
        _register(fg, "feat1")
        fg.tick(0, 0.0, NE)
        fg.tick(1, 5.0, NE)     # ACTIVE
        rec = fg._features["feat1"]
        assert rec.n_trials == 1
        assert rec.total_reward > 0.0

    def test_ucb1_reward_zero_for_no_impact(self):
        fg = _make_fg()
        _register(fg, "feat1")
        fg.tick(0, 5.0, NE)
        fg.tick(1, 5.0, NE)     # no impact
        rec = fg._features["feat1"]
        assert rec.n_trials == 1
        assert rec.total_reward == 0.0


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------

class TestFeatureGateDegradation:
    def test_degradation_level_increases_on_stagnation(self):
        fg = _make_fg(degradation_patience=2)
        _activate(fg, "feat1")
        # 2 gens stagnant → degradation_level += 0.2
        fg.tick(2, 1.0, NE)
        fg.tick(3, 1.0, NE)    # stagnation_count=2 >= 2 → level=0.2
        assert fg._features["feat1"].degradation_level == pytest.approx(0.2)

    def test_degradation_resets_on_improvement(self):
        fg = _make_fg(degradation_patience=3)
        _activate(fg, "feat1")
        fg.tick(2, 1.0, NE)    # stagnation_count=1
        fg.tick(3, 1.0, NE)    # stagnation_count=2
        fg.tick(4, 2.0, NE)    # improvement → stagnation_count reset to 0
        assert fg._features["feat1"].stagnation_count == 0
        assert fg._features["feat1"].degradation_level == pytest.approx(0.0)

    def test_degradation_reaches_08_disables(self):
        fg = _make_fg(degradation_patience=1, reactivation_delay=1000)
        _activate(fg, "feat1")
        # 4 patience-1 rounds: 0→0.2→0.4→0.6→0.8 → DISABLED
        for g in range(2, 15):
            fg.tick(g, 1.0, NE)    # constant fitness
        assert fg._features["feat1"].status == FeatureStatus.DISABLED

    def test_degrade_fn_called_on_each_step(self):
        fg = _make_fg(degradation_patience=1, reactivation_delay=1000)
        deg = []
        _register(fg, "feat1", degrade_calls=deg)
        fg.tick(0, 0.0, NE)
        fg.tick(1, 1.0, NE)    # ACTIVE
        for g in range(2, 10):
            fg.tick(g, 1.0, NE)
        assert len(deg) >= 1
        for name, level in deg:
            assert name == "feat1"
            assert 0.0 < level <= 1.0

    def test_disable_fn_called_on_disabled(self):
        fg = _make_fg(degradation_patience=1, reactivation_delay=1000)
        dis = []
        _register(fg, "feat1", disable_calls=dis)
        fg.tick(0, 0.0, NE)
        fg.tick(1, 1.0, NE)    # ACTIVE; disable not called yet
        assert dis == []
        for g in range(2, 15):
            fg.tick(g, 1.0, NE)
        assert fg._features["feat1"].status == FeatureStatus.DISABLED
        assert "feat1" in dis

    def test_degradation_level_step_is_02(self):
        fg = _make_fg(degradation_patience=2, reactivation_delay=1000)
        _activate(fg, "feat1")
        fg.tick(2, 1.0, NE)
        fg.tick(3, 1.0, NE)    # first round: 0.0 → 0.2
        assert fg._features["feat1"].degradation_level == pytest.approx(0.2)
        fg.tick(4, 1.0, NE)
        fg.tick(5, 1.0, NE)    # second round: 0.2 → 0.4
        assert fg._features["feat1"].degradation_level == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# max_concurrent
# ---------------------------------------------------------------------------

class TestFeatureGateMaxConcurrent:
    def test_active_plus_testing_capped_at_max_concurrent(self):
        fg = _make_fg(max_concurrent=2, test_interval=1, test_duration=100)
        for name in ["a", "b", "c", "d"]:
            _register(fg, name)
        for g in range(10):
            fg.tick(g, float(g), NE)
        n_occ = sum(
            1 for r in fg._features.values()
            if r.status in (FeatureStatus.ACTIVE, FeatureStatus.TESTING)
        )
        assert n_occ <= 2

    def test_testing_counts_toward_max_concurrent(self):
        fg = _make_fg(max_concurrent=1, test_interval=1, test_duration=10)
        _register(fg, "a")
        _register(fg, "b")
        fg.tick(0, 0.0, NE)    # starts test for one feature (max=1)
        testing = [n for n, r in fg._features.items() if r.status == FeatureStatus.TESTING]
        assert len(testing) == 1


# ---------------------------------------------------------------------------
# Reactivation
# ---------------------------------------------------------------------------

class TestFeatureGateReactivation:
    def test_disabled_reactivates_after_delay(self):
        # Use large test_interval to prevent immediate re-test after reactivation
        fg = FeatureGate(
            test_interval=1000,
            test_duration=1,
            degradation_patience=1,
            reactivation_delay=3,
            impact_threshold=0.0,
        )
        # _disable internally calls _activate → _register; don't pre-register
        g = _disable(fg, "feat1", patience=1)
        disabled_gen = fg._features["feat1"].disabled_at_gen
        # Advance past the delay; _check_reactivation also resets _last_test_gen
        for gg in range(g + 1, disabled_gen + 5):
            fg.tick(gg, 1.0, NE)
        # After reactivation, _last_test_gen is reset → no immediate new test
        assert fg._features["feat1"].status == FeatureStatus.INACTIVE

    def test_reactivate_explicit_disabled(self):
        fg = _make_fg(degradation_patience=1, reactivation_delay=1000)
        _disable(fg, "feat1", patience=1)
        fg.reactivate("feat1")
        assert fg._features["feat1"].status == FeatureStatus.INACTIVE
        assert fg._features["feat1"].degradation_level == pytest.approx(0.0)

    def test_reactivate_active_is_noop(self):
        fg = _make_fg()
        _register(fg, "feat1")
        fg.tick(0, 0.0, NE)
        fg.tick(1, 5.0, NE)    # ACTIVE
        fg.reactivate("feat1")   # silently does nothing
        assert fg._features["feat1"].status == FeatureStatus.ACTIVE

    def test_reactivate_unknown_raises(self):
        fg = _make_fg()
        with pytest.raises(ValueError):
            fg.reactivate("nonexistent")

    def test_disabled_before_delay_stays_disabled(self):
        fg = FeatureGate(
            test_interval=1000,
            test_duration=1,
            degradation_patience=1,
            reactivation_delay=100,
            impact_threshold=0.0,
        )
        g = _disable(fg, "feat1", patience=1)
        disabled_gen = fg._features["feat1"].disabled_at_gen
        fg.tick(disabled_gen + 1, 1.0, NE)
        assert fg._features["feat1"].status == FeatureStatus.DISABLED


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class TestFeatureGateDiagnostics:
    def test_get_diagnostics_structure(self):
        fg = _make_fg()
        _register(fg, "feat1")
        d = fg.get_diagnostics()
        assert "enabled" in d
        assert "n_active" in d
        assert "n_testing" in d
        assert "max_concurrent" in d
        assert "features" in d
        assert "change_log" in d

    def test_get_diagnostics_enabled_true(self):
        fg = _make_fg()
        assert fg.get_diagnostics()["enabled"] is True

    def test_get_active_features_empty_initially(self):
        fg = _make_fg()
        _register(fg, "feat1")
        assert fg.get_active_features() == []

    def test_get_active_features_after_activation(self):
        fg = _make_fg()
        _register(fg, "feat1")
        fg.tick(0, 0.0, NE)
        fg.tick(1, 5.0, NE)    # ACTIVE
        assert "feat1" in fg.get_active_features()

    def test_change_log_grows_on_transitions(self):
        fg = _make_fg()
        _register(fg, "feat1")
        fg.tick(0, 0.0, NE)    # testing
        fg.tick(1, 5.0, NE)    # active
        assert len(fg._change_log) >= 2

    def test_change_log_entry_structure(self):
        fg = _make_fg()
        _register(fg, "feat1")
        fg.tick(0, 0.0, NE)
        entry = fg._change_log[0]
        assert "generation" in entry
        assert "feature" in entry
        assert "to" in entry
        assert "reason" in entry

    def test_feature_diagnostics_per_feature(self):
        fg = _make_fg()
        _register(fg, "feat1")
        d = fg.get_diagnostics()
        feat_d = d["features"]["feat1"]
        assert "status" in feat_d
        assert "degradation_level" in feat_d
        assert "n_trials" in feat_d


# ---------------------------------------------------------------------------
# Error resilience
# ---------------------------------------------------------------------------

class TestFeatureGateErrors:
    def test_no_features_registered_tick_noop(self):
        fg = _make_fg()
        fg.tick(0, 0.0, NE)    # must not raise

    def test_enable_fn_exception_doesnt_crash_tick(self):
        fg = _make_fg()

        def bad_enable(ne):
            raise RuntimeError("boom")

        fg.register("feat1", enable_fn=bad_enable, disable_fn=lambda ne: None)
        fg.tick(0, 0.0, NE)    # must not raise; feature is TESTING
        assert fg._features["feat1"].status == FeatureStatus.TESTING

    def test_disable_fn_exception_doesnt_crash_tick(self):
        fg = _make_fg()

        def bad_disable(ne):
            raise RuntimeError("boom")

        fg.register("feat1", enable_fn=lambda ne: None, disable_fn=bad_disable)
        fg.tick(0, 0.0, NE)    # start test
        fg.tick(1, 0.0, NE)    # no impact → disable called → exception swallowed
        assert fg._features["feat1"].status == FeatureStatus.INACTIVE

    def test_degrade_fn_exception_doesnt_crash_tick(self):
        fg = _make_fg(degradation_patience=1, reactivation_delay=1000)

        def bad_degrade(ne, level):
            raise RuntimeError("boom")

        fg.register("feat1", enable_fn=lambda ne: None, disable_fn=lambda ne: None,
                    degrade_fn=bad_degrade)
        fg.tick(0, 0.0, NE)
        fg.tick(1, 1.0, NE)    # ACTIVE
        for g in range(2, 10):
            fg.tick(g, 1.0, NE)   # must not raise despite degrade_fn raising


# ---------------------------------------------------------------------------
# NeuroEvolution integration
# ---------------------------------------------------------------------------

class TestNeuroEvolutionFeatureGating:
    def _make_ne(self, iters=60):
        ne = NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1)
        ne.set_max_iterations(iters)
        return ne

    def test_set_auto_features_enables(self):
        ne = self._make_ne()
        ne.set_auto_features(enabled=True, test_interval=5)
        assert ne._feature_gate_enabled is True
        assert ne._feature_gate is not None

    def test_set_auto_features_disabled(self):
        ne = self._make_ne()
        ne.set_auto_features(enabled=False)
        assert ne._feature_gate_enabled is False
        assert ne._feature_gate is None

    def test_get_feature_gating_diagnostics_disabled(self):
        ne = self._make_ne()
        d = ne.get_feature_gating_diagnostics()
        assert d["enabled"] is False

    def test_get_feature_gating_diagnostics_enabled(self):
        ne = self._make_ne()
        ne.set_auto_features(enabled=True)
        d = ne.get_feature_gating_diagnostics()
        assert d["enabled"] is True
        assert "n_active" in d
        assert "features" in d

    def test_train_with_feature_gating_no_crash(self):
        ne = self._make_ne()
        ne.set_auto_features(
            enabled=True,
            test_interval=5,
            test_duration=3,
            max_concurrent=2,
        )
        ne.train(lambda g: 1.0)    # must not raise

    def test_train_without_feature_gating_unaffected(self):
        ne = self._make_ne()
        ne.train(lambda g: 1.0)
        assert ne.get_feature_gating_diagnostics()["enabled"] is False

    def test_feature_gating_registered_features_present(self):
        ne = self._make_ne()
        ne.set_auto_features(enabled=True)
        d = ne.get_feature_gating_diagnostics()
        features = d["features"]
        assert "curiosity" in features
        assert "darts" in features
        assert "shared_weights" in features

    def test_feature_gating_ticks_during_train(self):
        ne = NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1)
        ne.set_population_size(10)
        ne.set_max_iterations(120)  # 12 generations
        ne.set_auto_features(
            enabled=True,
            test_interval=3,
            test_duration=2,
            max_concurrent=3,
        )
        ne.train(lambda g: 1.0)
        d = ne.get_feature_gating_diagnostics()
        # At least one transition (first test starts at gen 0)
        assert len(d["change_log"]) >= 1

    def test_feature_gating_and_meta_optimizer_both_run(self):
        ne = NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1)
        ne.set_population_size(10)
        ne.set_max_iterations(100)
        ne.set_meta_optimizer(enabled=True, tune_interval=3, plateau_patience=5)
        ne.set_auto_features(enabled=True, test_interval=5)
        ne.train(lambda g: 1.0)   # must not raise
        assert ne.get_meta_optimizer_diagnostics()["enabled"] is True
        assert ne.get_feature_gating_diagnostics()["enabled"] is True
