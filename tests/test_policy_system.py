"""Tests for the Adaptive Policy System."""
from __future__ import annotations
import unittest
import pytest

from yane.evolution.policy import PolicyRegistry, TrainingContext, Action


class _SimplePolicy:
    name = "simple"

    def __init__(self):
        self.observe_count = 0
        self.last_ctx = None

    def observe(self, ctx):
        self.observe_count += 1
        self.last_ctx = ctx

    def decide(self, ctx):
        if self.observe_count == 1:
            return Action("init", priority=5, conflict_group="init", payload={"val": 42})
        return None

    def apply(self, ctx, action):
        self.applied = action


class _ConflictPolicy:
    name = "conflict"

    def observe(self, ctx):
        pass

    def decide(self, ctx):
        return Action("conflict_action", priority=10, conflict_group="init")

    def apply(self, ctx, action):
        self.applied = action


@pytest.mark.ci
class TestPolicyRegistry(unittest.TestCase):

    def test_register_and_tick(self):
        reg = PolicyRegistry()
        p = _SimplePolicy()
        reg.register(p)
        reg.tick(TrainingContext(generation=1))
        self.assertEqual(p.observe_count, 1)
        self.assertIsNotNone(getattr(p, "applied", None))

    def test_disabled_policy_does_not_fire(self):
        reg = PolicyRegistry()
        p = _SimplePolicy()
        reg.register(p, enabled=False)
        reg.tick(TrainingContext())
        self.assertEqual(p.observe_count, 0)

    def test_order_respected(self):
        reg = PolicyRegistry()
        calls = []
        class P1:
            name = "p1"
            def observe(self, ctx): calls.append("p1_obs")
            def decide(self, ctx): return Action("p1", priority=1)
            def apply(self, ctx, a): calls.append("p1_app")
        class P2:
            name = "p2"
            def observe(self, ctx): calls.append("p2_obs")
            def decide(self, ctx): return Action("p2", priority=1)
            def apply(self, ctx, a): calls.append("p2_app")
        p1, p2 = P1(), P2()
        reg.register(p1)
        reg.register(p2)
        reg.set_order(["p2", "p1"])
        reg.tick(TrainingContext())
        self.assertEqual(calls, ["p2_obs", "p1_obs", "p2_app", "p1_app"])

    def test_conflict_resolution_higher_priority_wins(self):
        reg = PolicyRegistry()
        p1 = _SimplePolicy()
        p2 = _ConflictPolicy()
        reg.register(p1)
        reg.register(p2)
        reg.tick(TrainingContext())
        # Both proposed actions have conflict_group="init", priority 10 > 5
        self.assertEqual(p2.applied.name, "conflict_action")
        self.assertFalse(hasattr(p1, "applied") and p1.applied is not None)

    def test_get_diagnostics(self):
        reg = PolicyRegistry()
        p = _SimplePolicy()
        reg.register(p)
        diag = reg.get_diagnostics()
        self.assertIn("active_policies", diag)
        self.assertIn("policy_order", diag)

    def test_register_policy_api(self):
        from yane import NeuroEvolution
        ne = NeuroEvolution()
        p = _SimplePolicy()
        ne.register_policy(p)
        self.assertTrue(ne._policy_tick_enabled)
        diag = ne.get_policy_diagnostics()
        self.assertIn("active_policies", diag)

    def test_policy_integration_during_train(self):
        """Policy tick runs during train() without errors."""
        from yane import NeuroEvolution
        ne = NeuroEvolution()
        ne.set_population_size(30)
        ne.configure(2, 1)
        ne.set_max_iterations(60)
        p = _SimplePolicy()
        ne.register_policy(p)
        def _eval(g): return sum(abs(c.weight) for src in g.nodes for c in src.connections)
        ne.train(_eval)
        self.assertGreater(p.observe_count, 0)


# --- Edge cases ---

class TestPolicyEdgeCases(unittest.TestCase):

    def test_best_value_empty(self):
        from yane.evolution.online_tuning import UCB1Bandit
        bandit = UCB1Bandit("test", [1, 2])
        self.assertEqual(bandit.best_value(), 1)

    def test_update_before_select_no_error(self):
        from yane.evolution.online_tuning import UCB1Bandit
        bandit = UCB1Bandit("test", [1])
        bandit.update(10.0)  # should not raise

    def test_get_diagnostics_empty(self):
        from yane.evolution.policy import PolicyRegistry
        reg = PolicyRegistry()
        diag = reg.get_diagnostics()
        self.assertEqual(diag["active_policies"], [])

    def test_island_invalid_params(self):
        from yane.evolution.islands import IslandModel
        with self.assertRaises(ValueError):
            IslandModel(n_islands=0)

    def test_surrogate_invalid_frac(self):
        from yane.evolution.surrogate import FitnessSurrogate
        s = FitnessSurrogate(surrogate_frac=1.5)
        self.assertAlmostEqual(s.surrogate_frac, 0.95)


if __name__ == "__main__":
    unittest.main()
