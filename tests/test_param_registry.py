"""Tests for the Unified Parameter Registry (P0 Meta-Adaptive Orchestration, Phase 1).

Covers:
- ParamSpec construction and current_value tracking
- ParamRegistry registration, get_spec, get_current, list_names
- Validation: categorical, continuous, integer, boolean domains
- set_param() dispatch on a real NeuroEvolution instance
- get_param_space() structure and completeness (≥30 params)
- Change-log recording
- Fitness-impact recording
- Registry-Roundtrip: set_param() equivalent to direct set_*() call
- Pickle-safety of ParamSpec
"""
from __future__ import annotations

import pickle

import pytest

from yane import NeuroEvolution
from yane.evolution.param_registry import (
    ParamRegistry,
    ParamSpec,
    _validate,
    build_default_registry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def ne():
    """Minimal unconfigured NeuroEvolution instance."""
    return NeuroEvolution(seed=0)


@pytest.fixture()
def ne_configured():
    """NeuroEvolution instance with a tiny population."""
    instance = NeuroEvolution(seed=0)
    instance.configure(n_inputs=2, n_outputs=1)
    return instance


@pytest.fixture()
def registry(ne):
    return ne.get_param_registry()


# ---------------------------------------------------------------------------
# ParamSpec basics
# ---------------------------------------------------------------------------

class TestParamSpec:
    def test_default_becomes_current(self):
        spec = ParamSpec(
            name="test.x",
            type="integer",
            domain=(1, 10),
            default=5,
            stage="both",
            subsystem="test",
        )
        assert spec.current_value == 5

    def test_current_value_settable(self):
        spec = ParamSpec(
            name="test.x",
            type="categorical",
            domain=["a", "b"],
            default="a",
            stage="both",
            subsystem="test",
        )
        spec.current_value = "b"
        assert spec.current_value == "b"

    def test_impact_history_starts_empty(self):
        spec = ParamSpec(
            name="test.x",
            type="boolean",
            domain=[True, False],
            default=True,
            stage="both",
            subsystem="test",
        )
        assert spec.impact_history == []

    def test_pickle_roundtrip(self):
        spec = ParamSpec(
            name="test.x",
            type="continuous",
            domain=(0.0, 1.0),
            default=0.5,
            stage="both",
            subsystem="test",
            description="A test param.",
        )
        spec.current_value = 0.7
        spec.impact_history.append(0.12)
        loaded = pickle.loads(pickle.dumps(spec))
        assert loaded.current_value == 0.7
        assert loaded.impact_history == [0.12]
        assert loaded.description == "A test param."


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def _spec(self, type_, domain, default):
        return ParamSpec(
            name="t.p", type=type_, domain=domain, default=default,
            stage="both", subsystem="t",
        )

    def test_categorical_valid(self):
        spec = self._spec("categorical", ["a", "b"], "a")
        _validate(spec, "b")  # should not raise

    def test_categorical_invalid(self):
        spec = self._spec("categorical", ["a", "b"], "a")
        with pytest.raises(ValueError, match="not in allowed values"):
            _validate(spec, "c")

    def test_continuous_valid(self):
        spec = self._spec("continuous", (0.0, 1.0), 0.5)
        _validate(spec, 0.0)
        _validate(spec, 1.0)
        _validate(spec, 0.3)

    def test_continuous_out_of_range(self):
        spec = self._spec("continuous", (0.0, 1.0), 0.5)
        with pytest.raises(ValueError, match="outside"):
            _validate(spec, 1.5)

    def test_integer_valid(self):
        spec = self._spec("integer", (1, 100), 5)
        _validate(spec, 50)

    def test_integer_out_of_range(self):
        spec = self._spec("integer", (1, 100), 5)
        with pytest.raises(ValueError, match="outside"):
            _validate(spec, 0)

    def test_integer_wrong_type(self):
        spec = self._spec("integer", (1, 100), 5)
        with pytest.raises(TypeError, match="expected int"):
            _validate(spec, 5.0)

    def test_boolean_valid(self):
        spec = self._spec("boolean", [True, False], True)
        _validate(spec, False)

    def test_boolean_wrong_type(self):
        spec = self._spec("boolean", [True, False], True)
        with pytest.raises(TypeError, match="expected bool"):
            _validate(spec, 1)

    def test_unknown_type_raises(self):
        spec = self._spec("unknown_type", [], None)
        with pytest.raises(ValueError, match="Unknown param type"):
            _validate(spec, None)


# ---------------------------------------------------------------------------
# ParamRegistry
# ---------------------------------------------------------------------------

class TestParamRegistry:
    def _make_reg(self):
        reg = ParamRegistry()
        reg.register(
            ParamSpec("foo.bar", "integer", (0, 10), 5, "both", "foo"),
            dispatcher=None,
        )
        reg.register(
            ParamSpec("foo.mode", "categorical", ["a", "b"], "a", "both", "foo"),
            dispatcher=None,
        )
        return reg

    def test_register_and_get_spec(self):
        reg = self._make_reg()
        spec = reg.get_spec("foo.bar")
        assert spec is not None
        assert spec.name == "foo.bar"

    def test_get_spec_unknown_returns_none(self):
        reg = self._make_reg()
        assert reg.get_spec("does.not.exist") is None

    def test_get_current_returns_default(self):
        reg = self._make_reg()
        assert reg.get_current("foo.bar") == 5

    def test_get_current_unknown_returns_none(self):
        reg = self._make_reg()
        assert reg.get_current("x.y") is None

    def test_list_names_sorted(self):
        reg = self._make_reg()
        names = reg.list_names()
        assert names == sorted(names)
        assert "foo.bar" in names
        assert "foo.mode" in names

    def test_set_value_updates_current(self):
        reg = self._make_reg()
        reg.set_value("foo.bar", 7)
        assert reg.get_current("foo.bar") == 7

    def test_set_value_unknown_raises(self):
        reg = self._make_reg()
        with pytest.raises(KeyError):
            reg.set_value("x.y", 1)

    def test_set_value_invalid_raises(self):
        reg = self._make_reg()
        with pytest.raises(ValueError):
            reg.set_value("foo.bar", 999)  # out of range

    def test_change_log_recorded(self):
        reg = self._make_reg()
        reg.set_value("foo.bar", 7)
        log = reg.get_change_log()
        assert len(log) == 1
        assert log[-1]["name"] == "foo.bar"
        assert log[-1]["new"] == 7
        assert log[-1]["old"] == 5

    def test_dispatch_calls_dispatcher(self):
        reg = self._make_reg()
        called = []
        reg.register(
            ParamSpec("bar.x", "integer", (0, 10), 0, "both", "bar"),
            dispatcher=lambda ne, v: called.append(v),
        )
        reg.dispatch(object(), "bar.x", 3)
        assert called == [3]

    def test_dispatch_records_change(self):
        reg = self._make_reg()
        called = []
        reg.register(
            ParamSpec("baz.y", "integer", (0, 10), 0, "both", "baz"),
            dispatcher=lambda ne, v: called.append(v),
        )
        reg.dispatch(object(), "baz.y", 4)
        assert reg.get_current("baz.y") == 4

    def test_dispatch_unknown_raises(self):
        reg = self._make_reg()
        with pytest.raises(KeyError):
            reg.dispatch(object(), "no.such", 1)

    def test_record_fitness_impact(self):
        reg = self._make_reg()
        reg.record_fitness_impact("foo.bar", 0.05)
        reg.record_fitness_impact("foo.bar", -0.02)
        spec = reg.get_spec("foo.bar")
        assert spec.impact_history == [0.05, -0.02]

    def test_record_fitness_impact_unknown_ignored(self):
        reg = self._make_reg()
        reg.record_fitness_impact("no.such", 1.0)  # should not raise

    def test_get_param_space_structure(self):
        reg = self._make_reg()
        space = reg.get_param_space()
        assert "foo.bar" in space
        entry = space["foo.bar"]
        assert set(entry.keys()) >= {
            "type", "domain", "default", "current", "stage",
            "subsystem", "description", "impact_history",
        }

    def test_pickle_roundtrip(self):
        reg = self._make_reg()
        reg.set_value("foo.bar", 9)
        loaded: ParamRegistry = pickle.loads(pickle.dumps(reg))
        assert loaded.get_current("foo.bar") == 9


# ---------------------------------------------------------------------------
# Integration with NeuroEvolution
# ---------------------------------------------------------------------------

class TestNeuroEvolutionIntegration:
    def test_get_param_registry_returns_registry(self, ne):
        reg = ne.get_param_registry()
        assert isinstance(reg, ParamRegistry)

    def test_second_call_returns_same_registry(self, ne):
        reg1 = ne.get_param_registry()
        reg2 = ne.get_param_registry()
        assert reg1 is reg2

    def test_get_param_space_has_enough_params(self, ne):
        space = ne.get_param_space()
        assert len(space) >= 30, (
            f"Expected ≥30 parameters in registry, got {len(space)}"
        )

    def test_get_param_space_subsystems_present(self, ne):
        space = ne.get_param_space()
        names = list(space.keys())
        subsystems_seen = {space[n]["subsystem"] for n in names}
        for expected in ("lamarck", "pop", "speciation", "anytime", "recovery"):
            assert expected in subsystems_seen, (
                f"Expected subsystem {expected!r} in registry"
            )

    def test_get_param_space_all_have_required_keys(self, ne):
        space = ne.get_param_space()
        required = {"type", "domain", "default", "current", "stage", "subsystem"}
        for name, info in space.items():
            missing = required - set(info.keys())
            assert not missing, f"Param {name!r} missing keys: {missing}"

    def test_set_param_lamarck_mode(self, ne):
        ne.set_param("lamarck.mode", "cma_es")
        reg = ne.get_param_registry()
        assert reg.get_current("lamarck.mode") == "cma_es"
        # Verify it actually applied: the lamarck refiner should now be in CMA-ES mode
        assert ne._lamarck.cma_mode is True

    def test_set_param_lamarck_mode_equivalent_to_direct(self, ne):
        ne2 = NeuroEvolution(seed=0)
        ne.set_param("lamarck.mode", "nes")
        ne2.set_lamarck(mode="nes")
        assert ne._lamarck.nes_mode == ne2._lamarck.nes_mode

    def test_set_param_pop_size(self, ne):
        ne.configure(n_inputs=2, n_outputs=1)
        ne.set_param("pop.size", 50)
        assert ne.get_param_registry().get_current("pop.size") == 50
        # Population size is set on the population
        assert ne._population_size == 50

    def test_set_param_anytime_enabled(self, ne):
        ne.set_param("anytime.enabled", True)
        assert ne.get_param_registry().get_current("anytime.enabled") is True
        assert ne._runner.anytime_enabled is True

    def test_set_param_anytime_disabled(self, ne):
        ne.set_param("anytime.enabled", False)
        assert ne._runner.anytime_enabled is False

    def test_set_param_recovery_cooldown(self, ne):
        ne.set_param("recovery.cooldown", 30)
        assert ne._recovery_cooldown == 30
        assert ne.get_param_registry().get_current("recovery.cooldown") == 30

    def test_set_param_invalid_domain_raises(self, ne):
        with pytest.raises(ValueError):
            ne.set_param("lamarck.mode", "nonexistent_mode")

    def test_set_param_unknown_name_raises(self, ne):
        with pytest.raises(KeyError):
            ne.set_param("does.not.exist", 42)

    def test_set_param_wrong_type_raises(self, ne):
        with pytest.raises(TypeError):
            ne.set_param("pop.size", 100.5)  # float, not int

    def test_set_param_records_change(self, ne):
        ne.set_param("lamarck.n_steps", 10)
        log = ne.get_param_registry().get_change_log()
        assert any(entry["name"] == "lamarck.n_steps" for entry in log)

    def test_set_param_speciation_target(self, ne):
        ne.set_param("speciation.target_n", 8)
        assert ne.get_param_registry().get_current("speciation.target_n") == 8

    def test_set_param_crossover_enabled(self, ne_configured):
        ne_configured.set_param("crossover.enabled", False)
        assert ne_configured._population._crossover_enabled is False

    def test_set_param_novelty_disabled(self, ne_configured):
        ne_configured.set_param("novelty.enabled", False)
        assert ne_configured._population._novelty_enabled is False

    def test_set_param_curiosity_weight(self, ne):
        ne.set_param("curiosity.weight", 0.5)
        assert ne._curiosity_weight == 0.5

    def test_set_param_darts_enabled(self, ne):
        ne.set_param("darts.enabled", True)
        assert ne._darts_enabled is True

    def test_set_param_shared_weights_enabled(self, ne):
        ne.set_param("shared_weights.enabled", True)
        assert ne._shared_weights_enabled is True

    def test_set_param_weight_blend_alpha(self, ne):
        ne.set_param("crossover.weight_blend_alpha", 0.9)
        assert ne._weight_blend_alpha == pytest.approx(0.9)

    def test_get_param_space_returns_all_names(self, ne):
        space = ne.get_param_space()
        expected_names = [
            "lamarck.mode", "lamarck.n_steps", "lamarck.sigma",
            "lamarck.enabled", "pop.size", "speciation.target_n",
            "anytime.enabled", "anytime.promotion_frac",
            "recovery.enabled", "recovery.cooldown",
            "novelty.enabled", "crossover.enabled",
            "curiosity.enabled", "darts.enabled", "shared_weights.enabled",
            "pruning.enabled",
        ]
        for name in expected_names:
            assert name in space, f"Expected {name!r} in param space"

    def test_registry_fitness_impact_via_ne(self, ne):
        reg = ne.get_param_registry()
        reg.record_fitness_impact("lamarck.mode", 0.08)
        space = ne.get_param_space()
        assert 0.08 in space["lamarck.mode"]["impact_history"]

    def test_build_default_registry_standalone(self, ne):
        """build_default_registry can be called independently."""
        reg = build_default_registry(ne)
        assert isinstance(reg, ParamRegistry)
        assert len(reg.list_names()) >= 30
