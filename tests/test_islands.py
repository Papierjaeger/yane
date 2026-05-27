"""Tests for multi-population island model."""
from __future__ import annotations
import unittest
import pytest


@pytest.mark.ci
class TestIslandModel(unittest.TestCase):

    def _make_islands(self, n=3, pop_size=30):
        from yane import NeuroEvolution
        from yane.evolution.islands import IslandModel
        yane = NeuroEvolution()
        yane.set_population_size(pop_size)
        yane.configure(2, 1)
        kw = {
            "max_size": pop_size,
            "initial_genome": yane._population._template.copy(),
            "tracker": yane._tracker,
            "target_species": 5,
        }
        return IslandModel(n_islands=n, island_kwargs=[kw] * n,
                           migration_interval=3, migration_count=2)

    def test_creates_islands(self):
        im = self._make_islands(4)
        self.assertEqual(len(im.islands), 4)

    def test_get_best_across_all(self):
        im = self._make_islands(2, pop_size=10)
        # Add a genome to the first island
        g = im.islands[0]._template.copy()
        g.fitness = 5.0
        g.raw_fitness = 5.0
        im.islands[0]._evaluated.append(g)
        best = im.get_best_across_all()
        self.assertIsNotNone(best)
        self.assertAlmostEqual(best.fitness, 5.0)

    def test_migration_does_not_raise(self):
        im = self._make_islands(2, pop_size=10)
        # Add some genomes to each island
        from yane.core.genome import Genome
        for idx in range(2):
            for _ in range(5):
                g = im.islands[idx]._template.copy()
                g.fitness = float(idx + 1) * 10.0
                g.raw_fitness = g.fitness
                im.islands[idx]._evaluated.append(g)
        # Tick enough to trigger migration (interval=3, so at ticks 3,6,9)
        for _ in range(12):
            im.tick()
        # Should have at least 1 migration event
        self.assertGreaterEqual(len(im._migration_events), 1)

    def test_diagnostics(self):
        im = self._make_islands(2, pop_size=10)
        diag = im.get_diagnostics()
        self.assertEqual(diag["n_islands"], 2)
        self.assertIn("island_best_fitness", diag)

    def test_set_island_model_api(self):
        from yane import NeuroEvolution
        ne = NeuroEvolution()
        ne.set_population_size(30)
        ne.configure(2, 1)
        im = ne.set_island_model(n_islands=3, migration_interval=5)
        self.assertEqual(im.n_islands, 3)
        diag = ne.get_island_diagnostics()
        self.assertEqual(diag["n_islands"], 3)

    def test_neuroevolution_trains_islands(self):
        from yane import NeuroEvolution
        ne = NeuroEvolution()
        ne.set_population_size(4)
        ne.configure(1, 1)
        ne.set_island_model(n_islands=2, migration_interval=1, migration_count=1)
        ne.set_max_iterations(8)
        ne.train(lambda g: sum(g.forward([0.0])))
        diag = ne.get_island_diagnostics()
        self.assertGreaterEqual(sum(v is not None for v in diag["island_best_fitness"]), 2)
        self.assertGreaterEqual(diag["total_migrations"], 1)


if __name__ == "__main__":
    unittest.main()
