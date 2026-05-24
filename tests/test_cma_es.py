import unittest

from yane import NeuroEvolution


class TestCMAESLamarck(unittest.TestCase):
    def test_set_lamarck_cma_es_mode(self):
        yane = NeuroEvolution()
        yane.set_lamarck(n_steps=2, mode="cma_es", cma_population=4)

        self.assertTrue(yane._lamarck.cma_mode)
        self.assertFalse(yane._lamarck.nes_mode)
        self.assertFalse(yane._lamarck.sa_mode)
        self.assertEqual(yane._lamarck.cma_population, 4)
        self.assertEqual(yane._lamarck.mode, "cma_es_explicit")

    def test_cma_es_refine_preserves_or_improves(self):
        yane = NeuroEvolution(seed=1)
        yane.configure(1, 1)
        g = yane.next_genome()

        def fitness(genome):
            return -sum(abs(n.bias) for n in genome.nodes)

        baseline = fitness(g)
        improved = yane._lamarck.refine_cma_es(g, fitness, baseline_fitness=baseline, n_steps=1)

        self.assertGreaterEqual(improved, baseline)


if __name__ == "__main__":
    unittest.main()
