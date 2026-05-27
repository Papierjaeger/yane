import unittest

from yane import NeuroEvolution
from yane.evolution.coevolution import HallOfFame, competitive_fitness, mixed_opponents


def _genome_with_fitness(value):
    yane = NeuroEvolution()
    yane.configure(1, 1)
    g = yane.next_genome()
    g.fitness = value
    return g


class TestCoevolutionHelpers(unittest.TestCase):
    def test_hall_of_fame_keeps_best_records(self):
        hof = HallOfFame(max_size=2)
        hof.add(_genome_with_fitness(1.0), 1.0)
        hof.add(_genome_with_fitness(3.0), 3.0)
        hof.add(_genome_with_fitness(2.0), 2.0)

        self.assertEqual(len(hof), 2)
        self.assertEqual([r.fitness for r in hof.records], [3.0, 2.0])

    def test_competitive_fitness_aggregation(self):
        g = _genome_with_fitness(0.0)
        opponents = [_genome_with_fitness(1.0), _genome_with_fitness(3.0)]

        def match(_g, opponent):
            return opponent.fitness

        self.assertEqual(competitive_fitness(g, opponents, match), 2.0)
        self.assertEqual(competitive_fitness(g, opponents, match, aggregation="min"), 1.0)
        self.assertEqual(competitive_fitness(g, opponents, match, aggregation="max"), 3.0)

    def test_mixed_opponents_samples_current_and_hof(self):
        current = [_genome_with_fitness(1.0), _genome_with_fitness(2.0)]
        hof = HallOfFame(max_size=3)
        hof.add(_genome_with_fitness(5.0), 5.0)

        opponents = mixed_opponents(current, hof, k_current=1, k_hof=1)

        self.assertEqual(len(opponents), 2)
        self.assertTrue(all(o is not current[0] for o in opponents))


if __name__ == "__main__":
    unittest.main()
