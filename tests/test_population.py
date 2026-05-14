import unittest
from yane.evolution.population import Population
from yane.core.genome import Genome


class TestPopulation(unittest.TestCase):

    def _make_population(self, max_size=10):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane._population.max_size = max_size
        return yane._population

    def test_initial_size(self):
        pop = self._make_population()
        self.assertEqual(pop.size, 1)
        self.assertEqual(pop.unevaluated_count, 1)
        self.assertEqual(pop.evaluated_count, 0)

    def test_submit_moves_genome(self):
        pop = self._make_population()
        g = pop.select_for_evaluation()
        pop.submit(g, -1.0)
        self.assertEqual(pop.evaluated_count, 1)
        self.assertEqual(pop.unevaluated_count, 0)

    def test_double_submit_is_noop(self):
        pop = self._make_population()
        g = pop.select_for_evaluation()
        pop.submit(g, -1.0)
        pop.submit(g, -0.5)   # second submit must be ignored
        self.assertEqual(pop.evaluated_count, 1)

    def test_population_stays_bounded(self):
        pop = self._make_population(max_size=5)
        for i in range(20):
            g = pop.select_for_evaluation()
            pop.submit(g, float(-i))
            self.assertLessEqual(pop.size, 6,  # max_size + 1 unevaluated at most
                msg=f"Population exceeded max_size at iteration {i}: {pop.size}")

    def test_get_best_returns_highest_fitness(self):
        pop = self._make_population(max_size=10)
        best_fitness = -999.0
        for i in range(5):
            g = pop.select_for_evaluation()
            f = float(-i)
            if f > best_fitness:
                best_fitness = f
            pop.submit(g, f)
        self.assertAlmostEqual(pop.get_best().fitness, best_fitness)

    def test_get_best_raises_before_evaluation(self):
        pop = self._make_population()
        with self.assertRaises(RuntimeError):
            pop.get_best()

    def test_shrink_to(self):
        pop = self._make_population(max_size=20)
        for i in range(10):
            g = pop.select_for_evaluation()
            pop.submit(g, float(-i))
        pop.shrink_to(3)
        self.assertEqual(len(pop._evaluated), 3)

    def test_shrink_keeps_best(self):
        pop = self._make_population(max_size=20)
        for i in range(10):
            g = pop.select_for_evaluation()
            pop.submit(g, float(i))
        pop.shrink_to(3)
        fitnesses = [g.fitness for g in pop._evaluated]
        self.assertEqual(sorted(fitnesses, reverse=True), fitnesses)


if __name__ == "__main__":
    unittest.main()
