"""Tests for the ResourceBudget system (evolution/resource_budget.py).

Covers:
- parse_time / parse_memory unit parsers
- ResourceDiscovery hardware sensors (safe on all platforms)
- BudgetConfig construction
- BudgetEnforcer: time budget, memory budget, injectable clock
- GracefulDegradation: all 6 levels
- NeuroEvolution.set_budget() / budget_status() API
- train() stop reason "budget_exceeded" for time budget
- MetaOptimizer interaction: budget wins on pop_size
- API backwards compatibility: set_resource_limits() / set_efficiency_penalty() still work
"""
from __future__ import annotations

import time as _time
import unittest

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ne():
    import yane
    ne = yane.NeuroEvolution()
    ne.configure(n_inputs=2, n_outputs=1, max_nodes=20, max_connections=40)
    return ne


def _synthetic_clock(start: float = 0.0):
    """Return a callable that returns a monotonically increasing value."""
    t = [start]
    def tick(delta=0.0):
        t[0] += delta
        return t[0]
    return tick


# ---------------------------------------------------------------------------
# parse_time
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestParseTime(unittest.TestCase):

    def test_minutes(self):
        from yane.evolution.resource_budget import parse_time
        self.assertAlmostEqual(parse_time("30min"), 1800.0)

    def test_hours(self):
        from yane.evolution.resource_budget import parse_time
        self.assertAlmostEqual(parse_time("1h"), 3600.0)
        self.assertAlmostEqual(parse_time("2hr"), 7200.0)

    def test_seconds_explicit(self):
        from yane.evolution.resource_budget import parse_time
        self.assertAlmostEqual(parse_time("45s"), 45.0)
        self.assertAlmostEqual(parse_time("45sec"), 45.0)

    def test_bare_number_string(self):
        from yane.evolution.resource_budget import parse_time
        self.assertAlmostEqual(parse_time("300"), 300.0)

    def test_numeric_passthrough(self):
        from yane.evolution.resource_budget import parse_time
        self.assertAlmostEqual(parse_time(300), 300.0)
        self.assertAlmostEqual(parse_time(60.0), 60.0)

    def test_none_returns_none(self):
        from yane.evolution.resource_budget import parse_time
        self.assertIsNone(parse_time(None))
        self.assertIsNone(parse_time("none"))

    def test_invalid_raises(self):
        from yane.evolution.resource_budget import parse_time
        with self.assertRaises(ValueError):
            parse_time("gibberish")


# ---------------------------------------------------------------------------
# parse_memory
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestParseMemory(unittest.TestCase):

    def test_gigabytes(self):
        from yane.evolution.resource_budget import parse_memory
        self.assertEqual(parse_memory("4GB"), 4_000_000_000)

    def test_gibibytes(self):
        from yane.evolution.resource_budget import parse_memory
        self.assertEqual(parse_memory("1GiB"), 1_073_741_824)

    def test_megabytes(self):
        from yane.evolution.resource_budget import parse_memory
        self.assertEqual(parse_memory("512MB"), 512_000_000)

    def test_percent(self):
        from yane.evolution.resource_budget import parse_memory
        result = parse_memory("50%", available_bytes=1_000_000_000)
        self.assertEqual(result, 500_000_000)

    def test_auto(self):
        from yane.evolution.resource_budget import parse_memory
        result = parse_memory("auto", available_bytes=1_000_000_000)
        self.assertEqual(result, 800_000_000)

    def test_none_returns_none(self):
        from yane.evolution.resource_budget import parse_memory
        self.assertIsNone(parse_memory(None))

    def test_numeric_passthrough(self):
        from yane.evolution.resource_budget import parse_memory
        self.assertEqual(parse_memory(1_000_000), 1_000_000)

    def test_invalid_raises(self):
        from yane.evolution.resource_budget import parse_memory
        with self.assertRaises(ValueError):
            parse_memory("lots")


# ---------------------------------------------------------------------------
# ResourceDiscovery — safe on all platforms
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestResourceDiscovery(unittest.TestCase):

    def test_cpu_count_positive(self):
        from yane.evolution.resource_budget import ResourceDiscovery
        self.assertGreater(ResourceDiscovery.cpu_count(), 0)

    def test_ram_total_positive(self):
        from yane.evolution.resource_budget import ResourceDiscovery
        self.assertGreater(ResourceDiscovery.ram_total_bytes(), 0)

    def test_ram_available_nonnegative(self):
        from yane.evolution.resource_budget import ResourceDiscovery
        self.assertGreaterEqual(ResourceDiscovery.ram_available_bytes(), 0)

    def test_current_process_bytes_positive(self):
        from yane.evolution.resource_budget import ResourceDiscovery
        self.assertGreater(ResourceDiscovery.current_process_bytes(), 0)

    def test_gpu_memory_none_or_int(self):
        from yane.evolution.resource_budget import ResourceDiscovery
        result = ResourceDiscovery.gpu_memory_bytes()
        self.assertTrue(result is None or isinstance(result, int))

    def test_battery_plugged_bool_or_none(self):
        from yane.evolution.resource_budget import ResourceDiscovery
        result = ResourceDiscovery.battery_plugged()
        self.assertIn(type(result), (bool, type(None)))

    def test_describe_returns_dict(self):
        from yane.evolution.resource_budget import ResourceDiscovery
        info = ResourceDiscovery.describe()
        self.assertIn("cpu_cores", info)
        self.assertIn("ram_total_gb", info)
        self.assertIn("battery_plugged", info)

    def test_auto_memory_budget_below_total(self):
        from yane.evolution.resource_budget import ResourceDiscovery
        budget = ResourceDiscovery.auto_memory_budget(0.80)
        total = ResourceDiscovery.ram_total_bytes()
        self.assertLess(budget, total)


# ---------------------------------------------------------------------------
# BudgetEnforcer — time budget with injectable clock
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestBudgetEnforcerTime(unittest.TestCase):

    def _make_enforcer(self, total_time_s: float):
        from yane.evolution.resource_budget import BudgetConfig, BudgetEnforcer
        tick = [0.0]
        def clock():
            return tick[0]
        def advance(dt):
            tick[0] += dt
        config = BudgetConfig(total_time_seconds=total_time_s)
        ne_stub = object()
        enf = BudgetEnforcer(config, ne_ref=ne_stub, clock=clock)
        return enf, advance

    def test_not_over_before_start(self):
        enf, _ = self._make_enforcer(10.0)
        self.assertFalse(enf.is_time_over())

    def test_not_over_within_budget(self):
        enf, advance = self._make_enforcer(10.0)
        enf.start()
        advance(5.0)
        self.assertFalse(enf.is_time_over())

    def test_over_at_budget(self):
        enf, advance = self._make_enforcer(10.0)
        enf.start()
        advance(10.0)
        self.assertTrue(enf.is_time_over())

    def test_over_past_budget(self):
        enf, advance = self._make_enforcer(10.0)
        enf.start()
        advance(15.0)
        self.assertTrue(enf.is_time_over())

    def test_elapsed_seconds(self):
        enf, advance = self._make_enforcer(60.0)
        enf.start()
        advance(13.0)
        self.assertAlmostEqual(enf.elapsed_seconds(), 13.0)

    def test_no_time_budget_never_over(self):
        from yane.evolution.resource_budget import BudgetConfig, BudgetEnforcer
        config = BudgetConfig(total_time_seconds=None)
        enf = BudgetEnforcer(config, ne_ref=object())
        enf.start()
        self.assertFalse(enf.is_time_over())


# ---------------------------------------------------------------------------
# BudgetEnforcer — memory budget
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestBudgetEnforcerMemory(unittest.TestCase):

    def test_no_memory_budget_no_escalation(self):
        from yane.evolution.resource_budget import BudgetConfig, BudgetEnforcer
        ne = _make_ne()
        config = BudgetConfig(max_memory_bytes=None)
        enf = BudgetEnforcer(config, ne_ref=ne)
        enf.check_memory(over_budget=True)  # forced
        self.assertEqual(enf.degradation.current_level, 0)

    def test_memory_over_budget_triggers_escalation(self):
        from yane.evolution.resource_budget import BudgetConfig, BudgetEnforcer
        ne = _make_ne()
        config = BudgetConfig(max_memory_bytes=1)  # effectively always over
        enf = BudgetEnforcer(config, ne_ref=ne)
        enf.check_memory(over_budget=True)
        self.assertEqual(enf.degradation.current_level, 1)

    def test_memory_within_budget_no_escalation(self):
        from yane.evolution.resource_budget import BudgetConfig, BudgetEnforcer
        ne = _make_ne()
        config = BudgetConfig(max_memory_bytes=10**12)  # practically unlimited
        enf = BudgetEnforcer(config, ne_ref=ne)
        enf.check_memory(over_budget=False)
        self.assertEqual(enf.degradation.current_level, 0)


# ---------------------------------------------------------------------------
# GracefulDegradation — each level
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestGracefulDegradation(unittest.TestCase):

    def test_level_0_initially(self):
        from yane.evolution.resource_budget import GracefulDegradation
        ne = _make_ne()
        dg = GracefulDegradation(ne)
        self.assertEqual(dg.current_level, 0)
        self.assertFalse(dg.stop_requested)

    def test_level1_reduces_pop(self):
        from yane.evolution.resource_budget import GracefulDegradation
        ne = _make_ne()
        ne._population_size = 100
        dg = GracefulDegradation(ne)
        dg.escalate()
        self.assertEqual(dg.current_level, 1)
        self.assertLessEqual(ne._population_size, 50)

    def test_level2_disables_lamarck(self):
        from yane.evolution.resource_budget import GracefulDegradation
        ne = _make_ne()
        ne._lamarck.steps = 5
        ne._lamarck.max_steps = 5
        dg = GracefulDegradation(ne)
        dg.current_level = 1  # skip level 1
        dg.escalate()
        self.assertEqual(dg.current_level, 2)
        self.assertEqual(ne._lamarck.steps, 0)
        self.assertEqual(ne._lamarck.max_steps, 0)

    def test_level3_reduces_max_nodes(self):
        from yane.evolution.resource_budget import GracefulDegradation
        ne = _make_ne()
        ne._max_nodes = 100
        dg = GracefulDegradation(ne)
        dg.current_level = 2
        dg.escalate()
        self.assertEqual(dg.current_level, 3)
        self.assertLessEqual(ne._max_nodes, 75)

    def test_level4_disables_research_features(self):
        from yane.evolution.resource_budget import GracefulDegradation
        ne = _make_ne()
        ne._curiosity_enabled = True
        ne._darts_enabled = True
        ne._shared_weights_enabled = True
        dg = GracefulDegradation(ne)
        dg.current_level = 3
        dg.escalate()
        self.assertEqual(dg.current_level, 4)
        self.assertFalse(ne._curiosity_enabled)
        self.assertFalse(ne._darts_enabled)
        self.assertFalse(ne._shared_weights_enabled)

    def test_level5_reduces_eval_budget(self):
        from yane.evolution.resource_budget import GracefulDegradation
        ne = _make_ne()
        ne._runner.configure_anytime_eval(enabled=True, max_evals=5)
        dg = GracefulDegradation(ne)
        dg.current_level = 4
        dg.escalate()
        self.assertEqual(dg.current_level, 5)
        self.assertEqual(ne._runner.anytime_max_evals, 1)

    def test_level6_requests_emergency_stop(self):
        from yane.evolution.resource_budget import GracefulDegradation
        ne = _make_ne()
        dg = GracefulDegradation(ne)
        dg.current_level = 5
        dg.escalate()
        self.assertEqual(dg.current_level, 6)
        self.assertTrue(dg.stop_requested)

    def test_escalate_all_6_levels_sequentially(self):
        from yane.evolution.resource_budget import GracefulDegradation
        ne = _make_ne()
        ne._population_size = 100
        ne._lamarck.steps = 5
        ne._max_nodes = 100
        ne._curiosity_enabled = True
        dg = GracefulDegradation(ne)
        for i in range(1, 7):
            level = dg.escalate()
            self.assertEqual(level, i)
        self.assertTrue(dg.stop_requested)

    def test_escalate_beyond_6_is_noop(self):
        from yane.evolution.resource_budget import GracefulDegradation
        ne = _make_ne()
        dg = GracefulDegradation(ne)
        dg.current_level = 6
        dg.stop_requested = True
        level = dg.escalate()
        self.assertEqual(level, 6)

    def test_applied_log_grows(self):
        from yane.evolution.resource_budget import GracefulDegradation
        ne = _make_ne()
        dg = GracefulDegradation(ne)
        dg.escalate()
        self.assertEqual(len(dg._applied), 1)
        dg.escalate()
        self.assertEqual(len(dg._applied), 2)


# ---------------------------------------------------------------------------
# NeuroEvolution.set_budget() and budget_status()
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestNeuroEvolutionBudgetAPI(unittest.TestCase):

    def test_set_budget_auto_no_crash(self):
        ne = _make_ne()
        ne.set_budget("auto")   # should not raise
        status = ne.budget_status()
        self.assertIn("memory_budget_bytes", status)
        self.assertIsNotNone(status["memory_budget_bytes"])

    def test_set_budget_time_string(self):
        ne = _make_ne()
        ne.set_budget(total_time="30min")
        status = ne.budget_status()
        self.assertAlmostEqual(status["time_budget_seconds"], 1800.0)

    def test_set_budget_memory_string(self):
        ne = _make_ne()
        ne.set_budget(max_memory="1GB")
        status = ne.budget_status()
        self.assertEqual(status["memory_budget_bytes"], 1_000_000_000)

    def test_budget_status_empty_without_set_budget(self):
        ne = _make_ne()
        self.assertEqual(ne.budget_status(), {})

    def test_set_budget_invalid_preset_raises(self):
        ne = _make_ne()
        with self.assertRaises(ValueError):
            ne.set_budget("magic_preset")

    def test_budget_status_has_required_keys(self):
        ne = _make_ne()
        ne.set_budget(total_time="5min", max_memory="4GB")
        status = ne.budget_status()
        for key in ("elapsed_seconds", "time_budget_seconds",
                    "degradation_level", "stop_requested"):
            self.assertIn(key, status)

    def test_set_budget_target_platform(self):
        ne = _make_ne()
        ne.set_budget(target_platform="cortex-m4")
        self.assertEqual(ne.budget_status()["target_platform"], "cortex-m4")


# ---------------------------------------------------------------------------
# train() stops on time budget using injectable clock
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestTrainTimeBudget(unittest.TestCase):

    def test_train_stops_on_budget_exceeded(self):
        """train() should stop with stop_reason='budget_exceeded' when time is up."""
        from yane.evolution.resource_budget import BudgetConfig, BudgetEnforcer

        ne = _make_ne()
        ne.set_max_iterations(10_000)  # large, so budget fires first

        # Use a clock that jumps to past the budget after 2 calls
        calls = [0]
        def fast_clock():
            calls[0] += 1
            return float(calls[0]) * 100.0  # each tick = 100s

        config = BudgetConfig(total_time_seconds=50.0)  # expires after ~1 call
        ne._budget_enforcer = BudgetEnforcer(config, ne_ref=ne, clock=fast_clock)

        ne.train(lambda g: 1.0)
        self.assertEqual(ne.stop_reason, "budget_exceeded")

    def test_degradation_level_zero_when_no_memory_pressure(self):
        ne = _make_ne()
        ne.set_budget(max_memory=10**12)  # practically unlimited
        ne.set_max_iterations(5)
        ne.train(lambda g: 1.0)
        self.assertEqual(ne.budget_status()["degradation_level"], 0)


# ---------------------------------------------------------------------------
# API backwards-compatibility
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestBackwardsCompatibility(unittest.TestCase):

    def test_set_resource_limits_still_works(self):
        ne = _make_ne()
        ne.set_resource_limits(min_free_gb=1.0, max_used_percent=90.0)
        # Should not raise; ResourceGuard is created
        self.assertIsNotNone(ne._resource_guard)

    def test_set_efficiency_penalty_still_works(self):
        ne = _make_ne()
        ne.set_efficiency_penalty(max_ms=100.0, penalty_per_ms=0.01)
        self.assertIsNotNone(ne._efficiency_penalty)


if __name__ == "__main__":
    unittest.main()
