import unittest

from yane import NeuroEvolution
from yane.evolution.evaluation import aggregate_fitnesses
from yane.evolution.multi_objective import dominates, non_dominated_sort, pareto_scores


class TestParetoHelpers(unittest.TestCase):
    def test_dominates_respects_maximize_flags(self):
        self.assertTrue(dominates((2.0, 1.0), (1.0, 2.0), maximize=(True, False)))
        self.assertFalse(dominates((2.0, 3.0), (1.0, 2.0), maximize=(True, False)))

    def test_non_dominated_sort_fronts(self):
        fronts = non_dominated_sort(
            [(3.0, 0.0), (2.0, 1.0), (1.0, 2.0), (0.0, 0.0)],
            maximize=(True, True),
        )
        self.assertEqual(set(fronts[0]), {0, 1, 2})
        self.assertEqual(fronts[1], [3])

    def test_pareto_scores_prefer_first_front(self):
        scores = pareto_scores(
            [(3.0, 0.0), (2.0, 1.0), (0.0, 0.0)],
            maximize=(True, True),
        )
        self.assertGreater(scores[0], scores[2])
        self.assertGreater(scores[1], scores[2])


class TestNeuroEvolutionMultiObjective(unittest.TestCase):
    def test_vector_fitness_is_stored_and_scalarized(self):
        yane = NeuroEvolution()
        yane.set_multi_objective(weights=(1.0, -0.5), maximize=(True, False))
        yane.configure(1, 1)

        g = yane.next_genome()
        yane.submit_fitness((10.0, 4.0))

        self.assertEqual(g.objectives, (10.0, 4.0))
        self.assertEqual(g.fitness, 8.0)
        info = yane.population_memory_info()
        self.assertTrue(info["multi_objective_enabled"])
        self.assertEqual(info["multi_objective_maximize"], (True, False))

    def test_weight_count_must_match_objective_count(self):
        yane = NeuroEvolution()
        yane.set_multi_objective(weights=(1.0,), maximize=(True, True))
        yane.configure(1, 1)
        yane.next_genome()
        with self.assertRaises(ValueError):
            yane.submit_fitness((1.0, 2.0))

    def test_multi_eval_aggregates_objective_vectors(self):
        yane = NeuroEvolution()
        yane.set_multi_objective(weights=(1.0, -1.0), maximize=(True, False))
        yane.configure(1, 1)
        yane.set_multi_eval(n=3, aggregation="mean", sigma_penalty=0.0)
        g = yane.next_genome()

        values = iter([(10.0, 1.0), (12.0, 3.0), (14.0, 5.0)])
        result = yane._run_evaluations(g, lambda _g: next(values))

        self.assertEqual(result.fitness, (12.0, 3.0))
        yane.submit_fitness(result.fitness, result.elapsed_ms)
        self.assertEqual(g.objectives, (12.0, 3.0))
        self.assertEqual(g.fitness, 9.0)

    def test_aggregate_fitnesses_vector_median(self):
        result = aggregate_fitnesses(
            [(1.0, 10.0), (3.0, 8.0), (5.0, 6.0)],
            "median",
            0.0,
        )
        self.assertEqual(result, (3.0, 8.0))


if __name__ == "__main__":
    unittest.main()
