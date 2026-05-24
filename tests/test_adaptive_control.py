import unittest

from yane import NeuroEvolution
from yane.core.genome import Genome
from yane.evolution.population import Population
from yane.evolution.species import Species


class TestAdaptiveInterspeciesCrossover(unittest.TestCase):
    def test_neuro_evolution_configures_adaptive_interspecies_crossover(self):
        yane = NeuroEvolution()
        yane.set_adaptive_interspecies_crossover(min_rate=0.01, max_rate=0.15)
        yane.configure(2, 1)

        info = yane.population_memory_info()

        self.assertEqual(info["interspecies_crossover_mode"], "adaptive")
        self.assertAlmostEqual(info["interspecies_crossover_min"], 0.01)
        self.assertAlmostEqual(info["interspecies_crossover_max"], 0.15)

    def test_adaptive_interspecies_rate_ramps_with_stagnation(self):
        pop = Population(max_size=10, initial_genome=Genome(), target_species=2)
        g1 = Genome()
        g2 = Genome()
        sp1 = Species(g1)
        sp2 = Species(g2)
        sp1.add(g1)
        sp2.add(g2)
        sp2.stagnation_count = pop.stagnation_threshold
        pop._species = [sp1, sp2]
        pop.configure_interspecies_crossover(
            mode="adaptive",
            min_rate=0.01,
            max_rate=0.2,
        )

        rate = pop._adaptive_interspecies_rate()

        self.assertAlmostEqual(rate, 0.2)
        self.assertEqual(pop._interspecies_crossover_last_reason, "adaptive:species_stagnation")

    def test_adaptive_interspecies_rate_zero_with_single_species(self):
        pop = Population(max_size=10, initial_genome=Genome(), target_species=2)
        g = Genome()
        sp = Species(g)
        sp.add(g)
        pop._species = [sp]
        pop.configure_interspecies_crossover(
            mode="adaptive",
            min_rate=0.01,
            max_rate=0.2,
        )

        self.assertEqual(pop._adaptive_interspecies_rate(), 0.0)
        self.assertEqual(pop._interspecies_crossover_last_reason, "adaptive:single_species")


class TestAdaptiveLamarckModes(unittest.TestCase):
    def test_adaptive_lamarck_optimizer_uses_requested_max_steps(self):
        yane = NeuroEvolution()

        yane.set_lamarck_adaptive(max_steps=1, mode="nes")
        self.assertEqual(yane._lamarck.mode, "nes_adaptive")
        self.assertEqual(yane._lamarck.max_steps, 1)

        yane.set_lamarck_adaptive(max_steps=2, mode="sa")
        self.assertEqual(yane._lamarck.mode, "sa_adaptive")
        self.assertEqual(yane._lamarck.max_steps, 2)

        yane.set_lamarck_adaptive(max_steps=4, mode="cma_es")
        self.assertEqual(yane._lamarck.mode, "cma_es_adaptive")
        self.assertEqual(yane._lamarck.max_steps, 4)


if __name__ == "__main__":
    unittest.main()
