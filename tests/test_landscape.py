"""Tests for fitness landscape visualization (GenomeDescriptor, PCA)."""
from __future__ import annotations

import unittest
import pytest
import tempfile
from pathlib import Path

from yane.core.genome import Genome
from yane.core.connection import Connection
from yane.core.node import Node, NodeType
from yane.util.activation import ActivationType


def _make_genome(weight: float = 1.0, n_conns: int = 1) -> Genome:
    """Create a simple genome for testing."""
    g = Genome()
    g.max_nodes = 20
    g.max_connections = 20
    inp = Node(NodeType.INPUT, 0)
    inp.activation = ActivationType.LINEAR
    out = Node(NodeType.OUTPUT, 1)
    out.activation = ActivationType.LINEAR
    g.nodes.extend([inp, out])
    g.input_nodes.append(inp)
    g.output_nodes.append(out)
    for i in range(n_conns):
        conn = Connection(out, innovation=100 + i)
        conn.weight = weight * (i + 1)
        inp.connections.append(conn)
    g.fitness = weight
    g._invalidate_topology()
    # Use a simple object as species marker for testing
    g._last_species_id = object()
    return g


@pytest.mark.ci
class TestGenomeDescriptor(unittest.TestCase):

    def test_descriptor_length(self):
        from yane.evolution.landscape import genome_descriptor_vector
        g = _make_genome()
        vec = genome_descriptor_vector(g)
        self.assertEqual(len(vec), 12)

    def test_descriptor_identical_genomes(self):
        from yane.evolution.landscape import genome_descriptor_vector
        g1 = _make_genome(2.0)
        g2 = _make_genome(2.0)
        v1 = genome_descriptor_vector(g1)
        v2 = genome_descriptor_vector(g2)
        # First 11 elements describe topology/weights and should be identical
        # Element 11 is a species hash that may differ between object() instances
        self.assertEqual(v1[:11], v2[:11])

    def test_descriptor_different_weights(self):
        from yane.evolution.landscape import genome_descriptor_vector
        g1 = _make_genome(1.0)
        g2 = _make_genome(10.0)
        v1 = genome_descriptor_vector(g1)
        v2 = genome_descriptor_vector(g2)
        self.assertNotEqual(v1, v2)

    def test_descriptor_different_topology(self):
        from yane.evolution.landscape import genome_descriptor_vector
        g1 = _make_genome(1.0, n_conns=1)
        g2 = _make_genome(1.0, n_conns=3)
        v1 = genome_descriptor_vector(g1)
        v2 = genome_descriptor_vector(g2)
        self.assertNotEqual(v1[1], v2[1])  # n_connections


@pytest.mark.ci
class TestPopulationPCA(unittest.TestCase):

    def test_pca_empty(self):
        from yane.evolution.landscape import population_pca
        result = population_pca([])
        self.assertEqual(result["x"], [])
        self.assertEqual(result["y"], [])

    def test_pca_single_genome(self):
        from yane.evolution.landscape import population_pca
        result = population_pca([_make_genome()])
        self.assertEqual(len(result["x"]), 1)
        self.assertEqual(len(result["y"]), 1)

    def test_pca_projection_shape(self):
        from yane.evolution.landscape import population_pca
        genomes = [_make_genome(w) for w in [0.5, 1.0, 2.0, 4.0]]
        result = population_pca(genomes)
        self.assertEqual(len(result["x"]), 4)
        self.assertEqual(len(result["y"]), 4)
        self.assertEqual(len(result["fitness"]), 4)
        self.assertEqual(len(result["species_id"]), 4)
        self.assertEqual(len(result["explained_var"]), 2)

    def test_pca_explained_var_non_negative(self):
        from yane.evolution.landscape import population_pca
        genomes = [_make_genome(w) for w in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]]
        result = population_pca(genomes)
        for v in result["explained_var"]:
            self.assertGreaterEqual(v, 0.0)

    def test_landscape_pca_api(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.set_population_size(30)
        yane.configure(2, 1)
        yane.set_max_iterations(100)
        def _eval(g):
            return sum(abs(c.weight) for src in g.nodes for c in src.connections)
        yane.train(_eval)
        result = yane.landscape_pca()
        self.assertIn("x", result)
        self.assertIn("y", result)
        if result:
            self.assertEqual(len(result["x"]), len(result["y"]))

    def test_export_landscape_csv(self):
        from yane.evolution.landscape import export_landscape_csv, population_pca
        result = population_pca([_make_genome(w) for w in [0.5, 1.0, 2.0]])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "landscape.csv"
            export_landscape_csv(result, path)
            text = path.read_text(encoding="utf-8")
        self.assertIn("explained_var_pc1", text)
        self.assertIn("index,x,y,fitness,species_id", text)
        self.assertEqual(len([line for line in text.splitlines() if line and not line.startswith("#")]), 4)

    def test_export_landscape_png(self):
        from yane.evolution.landscape import export_landscape_png, population_pca
        result = population_pca([_make_genome(w) for w in [0.5, 1.0, 2.0]])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "landscape.png"
            export_landscape_png(result, path, width=160, height=140)
            data = path.read_bytes()
        self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreater(len(data), 100)

    def test_landscape_export_api(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.set_population_size(20)
        yane.configure(2, 1)
        yane.set_max_iterations(60)

        def _eval(g):
            return sum(abs(c.weight) for src in g.nodes for c in src.connections)

        yane.train(_eval)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "landscape.csv"
            png_path = Path(tmp) / "landscape.png"
            yane.export_landscape_csv(str(csv_path))
            yane.export_landscape_png(str(png_path), width=160, height=140)
            self.assertTrue(csv_path.exists())
            self.assertTrue(png_path.read_bytes().startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
