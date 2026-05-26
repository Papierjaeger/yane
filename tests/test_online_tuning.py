"""Tests for online hyperparameter tuning (UCB1 bandit)."""
from __future__ import annotations

import unittest
import pytest


@pytest.mark.ci
class TestUCB1Bandit(unittest.TestCase):

    def test_select_returns_candidate(self):
        from yane.evolution.online_tuning import UCB1Bandit
        bandit = UCB1Bandit("test", [1, 2, 3])
        val = bandit.select(iteration=0, max_iterations=100)
        self.assertIn(val, [1, 2, 3])

    def test_exploration_selects_all_candidates(self):
        from yane.evolution.online_tuning import UCB1Bandit
        bandit = UCB1Bandit("test", [10, 20, 30], exploration_fraction=1.0)
        seen = set()
        for i in range(100):
            seen.add(bandit.select(iteration=i, max_iterations=100))
        self.assertEqual(seen, {10, 20, 30})

    def test_exploitation_prefers_high_reward(self):
        from yane.evolution.online_tuning import UCB1Bandit
        bandit = UCB1Bandit("test", [0, 1], exploration_fraction=0.0)
        # Arm 0 gets more reward
        bandit.counts = [10, 10]
        bandit.rewards = [100.0, 0.0]
        selections = {0: 0, 1: 0}
        for _ in range(50):
            val = bandit.select(iteration=100, max_iterations=100)
            idx = bandit.candidates.index(val)
            selections[idx] += 1
        # Arm 0 should be selected significantly more often
        self.assertGreater(selections[0], selections[1])

    def test_update_records_reward(self):
        from yane.evolution.online_tuning import UCB1Bandit
        bandit = UCB1Bandit("test", [5, 10], exploration_fraction=0.0)
        bandit.select(iteration=0, max_iterations=100)
        bandit.update(50.0)
        self.assertAlmostEqual(bandit.rewards[bandit._last_idx], 50.0)

    def test_best_value(self):
        from yane.evolution.online_tuning import UCB1Bandit
        bandit = UCB1Bandit("test", [1, 2], exploration_fraction=0.0)
        bandit.counts = [10, 10]
        bandit.rewards = [0.0, 100.0]
        self.assertEqual(bandit.best_value(), 2)

    def test_get_diagnostics(self):
        from yane.evolution.online_tuning import UCB1Bandit
        bandit = UCB1Bandit("rate", [0.1, 0.5], exploration_fraction=0.5)
        bandit.select(iteration=0, max_iterations=100)
        diag = bandit.get_diagnostics()
        self.assertIn("bandit_rate_counts", diag)
        self.assertIn("bandit_rate_candidates", diag)

    def test_set_online_tuning_api(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.set_online_tuning(True, params=["mutation_rate"])
        self.assertTrue(yane._online_tuning_enabled)
        self.assertIn("mutation_rate", yane._online_tuning_bandits)
        yane.set_online_tuning(False)
        self.assertFalse(yane._online_tuning_enabled)

    def test_online_tuning_integration(self):
        """Bandit tuning runs during train() without errors."""
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.set_population_size(50)
        yane.configure(2, 1)
        yane.set_max_iterations(120)  # ≥ 2 generations with pop_size=50
        yane.set_online_tuning(True, params=["mutation_rate", "n_lamarck_steps"])
        def _eval(g): return sum(abs(c.weight) for src in g.nodes for c in src.connections)
        yane.train(_eval)
        # Bandits should have been active
        for bandit in yane._online_tuning_bandits.values():
            self.assertGreater(sum(bandit.counts), 0,
                f"Bandit {bandit.param_name} was never used")


if __name__ == "__main__":
    unittest.main()
