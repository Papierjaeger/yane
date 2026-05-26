"""Tests for hyperparameter search."""
from __future__ import annotations
import unittest
import pytest


@pytest.mark.ci
class TestHyperparameterSearch(unittest.TestCase):

    def test_product_dict(self):
        from yane.evolution.hyperparameter_search import _product_dict
        grid = {"a": [1, 2], "b": ["x", "y"]}
        result = _product_dict(grid)
        self.assertEqual(len(result), 4)
        self.assertIn({"a": 1, "b": "x"}, result)

    def test_hyperparameter_search_runs(self):
        from yane.evolution.hyperparameter_search import hyperparameter_search
        grid = {"pop_size": [20], "target_species": [2]}
        results = hyperparameter_search(
            grid, n_seeds=1, budget_iterations=30,
            n_inputs=2, n_outputs=1,
        )
        self.assertGreater(len(results), 0)
        self.assertIn("config", results[0])
        self.assertIn("median_fitness", results[0])
        self.assertIn("seeds", results[0])

    def test_results_sorted_by_fitness(self):
        from yane.evolution.hyperparameter_search import hyperparameter_search
        grid = {"pop_size": [20, 30], "target_species": [2]}
        results = hyperparameter_search(
            grid, n_seeds=1, budget_iterations=20,
            n_inputs=2, n_outputs=1,
        )
        self.assertEqual(len(results), 2)
        # Check it's sorted descending
        self.assertGreaterEqual(
            results[0]["median_fitness"], results[1]["median_fitness"]
        )


if __name__ == "__main__":
    unittest.main()
