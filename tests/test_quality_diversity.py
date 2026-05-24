import unittest
import tempfile
from pathlib import Path

from yane import NeuroEvolution
from yane.evolution.quality_diversity import MAPElitesArchive


class TestMAPElitesArchive(unittest.TestCase):
    def test_archive_keeps_best_per_cell(self):
        yane = NeuroEvolution()
        yane.configure(1, 1)
        g1 = yane.next_genome()
        g2 = g1.copy()
        archive = MAPElitesArchive(bins=(2,), ranges=((0.0, 1.0),))

        self.assertTrue(archive.add((0.25,), g1, 1.0))
        self.assertFalse(archive.add((0.25,), g2, 0.5))
        self.assertTrue(archive.add((0.25,), g2, 2.0))
        self.assertEqual(len(archive.cells), 1)
        self.assertEqual(next(iter(archive.cells.values())).fitness, 2.0)

    def test_cell_clamps_to_range(self):
        archive = MAPElitesArchive(bins=(4,), ranges=((0.0, 1.0),))
        self.assertEqual(archive.cell_for((-10.0,)), (0,))
        self.assertEqual(archive.cell_for((10.0,)), (3,))

    def test_archive_exports_json_and_csv(self):
        yane = NeuroEvolution()
        yane.configure(1, 1)
        g = yane.next_genome()
        archive = MAPElitesArchive(bins=(2,), ranges=((0.0, 1.0),))
        archive.add((0.5,), g, 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "archive.json"
            csv_path = Path(tmp) / "archive.csv"
            archive.export_json(json_path)
            archive.export_csv(csv_path)
            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())


class TestQualityDiversityIntegration(unittest.TestCase):
    def test_submit_updates_archive_and_diagnostics(self):
        yane = NeuroEvolution()
        yane.configure(1, 1)
        yane.set_quality_diversity(
            descriptor_fn=lambda g: (float(g.connection_count),),
            bins=(4,),
            ranges=((0.0, 4.0),),
        )

        g = yane.next_genome()
        yane.submit_fitness(1.0)

        archive = yane.get_quality_diversity_archive()
        self.assertIsNotNone(archive)
        self.assertEqual(len(archive.cells), 1)
        info = yane.population_memory_info()
        self.assertTrue(info["quality_diversity_enabled"])
        self.assertEqual(info["quality_diversity_cells"], 1)
        self.assertEqual(info["n_quality_diversity_updates"], 1)


if __name__ == "__main__":
    unittest.main()
