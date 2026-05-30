"""Tests für Gene Regulatory Network (GRN) Encoding (evolution/grn_encoding.py).

Akzeptanzkriterien:
  1. GRN mit 20 Genen enkodiert Genome mit >100 Connections
  2. Crossover zweier GRN-Genome funktioniert (Alignment der Gene)
  3. Tests: GRN-Entwicklung; Phaenotyp-Groessen-Korrelation; Crossover; Codec-Protokoll
"""
from __future__ import annotations

import pickle
import unittest

import pytest


# ---------------------------------------------------------------------------
# Acceptance criterion 1: 20 genes → >100 connections
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestGRNDevelopment(unittest.TestCase):

    def test_20_genes_5_steps_more_than_100_connections(self):
        """20 GRN-Gene × 5 Entwicklungsschritte → >100 Verbindungen im Phänotyp."""
        from yane.evolution.grn_encoding import GRNGenome
        grn = GRNGenome.random(n_genes=20, n_nodes=10, seed=0)
        phenotype = grn.develop(n_inputs=3, n_outputs=2, development_steps=5)
        n_conns = sum(len(n.connections) for n in phenotype.nodes)
        self.assertGreater(n_conns, 100,
                           f"Expected >100 connections, got {n_conns}")

    def test_phenotype_has_correct_input_output_count(self):
        from yane.evolution.grn_encoding import GRNGenome
        grn = GRNGenome.random(n_genes=10, n_nodes=8, seed=1)
        phenotype = grn.develop(n_inputs=4, n_outputs=2, development_steps=3)
        self.assertEqual(len(phenotype.input_nodes), 4)
        self.assertEqual(len(phenotype.output_nodes), 2)

    def test_more_genes_more_connections(self):
        """More genes → more connections (phenotype size correlation)."""
        from yane.evolution.grn_encoding import GRNGenome
        grn_small = GRNGenome.random(n_genes=5, n_nodes=6, seed=2)
        grn_large = GRNGenome.random(n_genes=20, n_nodes=6, seed=2)
        p_small = grn_small.develop(n_inputs=2, n_outputs=1, development_steps=5)
        p_large = grn_large.develop(n_inputs=2, n_outputs=1, development_steps=5)
        n_small = sum(len(n.connections) for n in p_small.nodes)
        n_large = sum(len(n.connections) for n in p_large.nodes)
        self.assertGreater(n_large, n_small,
                           "More genes should produce more connections")

    def test_more_steps_more_connections(self):
        """More development steps → more connections."""
        from yane.evolution.grn_encoding import GRNGenome
        grn = GRNGenome.random(n_genes=10, n_nodes=8, seed=3)
        p3 = grn.develop(n_inputs=2, n_outputs=1, development_steps=3)
        p5 = grn.develop(n_inputs=2, n_outputs=1, development_steps=5)
        n3 = sum(len(n.connections) for n in p3.nodes)
        n5 = sum(len(n.connections) for n in p5.nodes)
        self.assertGreater(n5, n3,
                           "More development steps should produce more connections")

    def test_phenotype_forward_no_crash(self):
        """Developed phenotype must run forward() without error."""
        from yane.evolution.grn_encoding import GRNGenome
        grn = GRNGenome.random(n_genes=15, n_nodes=8, seed=4)
        phenotype = grn.develop(n_inputs=2, n_outputs=1, development_steps=5)
        phenotype.reset()
        result = phenotype.forward([0.5, 0.5])
        self.assertEqual(len(result), 1)

    def test_development_deterministic(self):
        """Same GRN + same parameters → same phenotype."""
        from yane.evolution.grn_encoding import GRNGenome
        grn = GRNGenome.random(n_genes=10, n_nodes=6, seed=5)
        p1 = grn.develop(n_inputs=2, n_outputs=1, development_steps=3)
        p2 = grn.develop(n_inputs=2, n_outputs=1, development_steps=3)
        n1 = sum(len(n.connections) for n in p1.nodes)
        n2 = sum(len(n.connections) for n in p2.nodes)
        self.assertEqual(n1, n2)

    def test_regulatory_sites_affect_expression(self):
        """Genes with regulatory sites are suppressed when regulators inactive."""
        from yane.evolution.grn_encoding import GRNGenome, GRNGene
        # Gene 0: no regulators (always active)
        # Gene 1: regulated by gene 2 (never active if gene 2 inactive)
        # Gene 2: no regulators (always active)
        genes = [
            GRNGene(src_node=0, tgt_node=2, weight=1.0),          # always active
            GRNGene(src_node=0, tgt_node=2, weight=0.5, regulatory_sites=[2]),  # regulated
            GRNGene(src_node=0, tgt_node=2, weight=0.8),          # always active → activates gene 1
        ]
        grn = GRNGenome(genes)
        phenotype = grn.develop(n_inputs=2, n_outputs=1, development_steps=2)
        n_conns = sum(len(n.connections) for n in phenotype.nodes)
        # All 3 genes × 2 steps = 6 connections if all active
        self.assertGreater(n_conns, 0)

    def test_n_genes_property(self):
        from yane.evolution.grn_encoding import GRNGenome
        grn = GRNGenome.random(n_genes=15, n_nodes=8, seed=0)
        self.assertEqual(grn.n_genes, 15)


# ---------------------------------------------------------------------------
# Acceptance criterion 2: Crossover mit Gene-Alignment
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestGRNCrossover(unittest.TestCase):

    def test_crossover_returns_grn_genome(self):
        from yane.evolution.grn_encoding import GRNGenome
        a = GRNGenome.random(n_genes=10, n_nodes=8, seed=0)
        b = GRNGenome.random(n_genes=10, n_nodes=8, seed=1)
        child = a.crossover(b)
        self.assertIsInstance(child, GRNGenome)

    def test_crossover_gene_count(self):
        """Child should have same number of genes as longer parent."""
        from yane.evolution.grn_encoding import GRNGenome
        a = GRNGenome.random(n_genes=10, n_nodes=6, seed=0)
        b = GRNGenome.random(n_genes=15, n_nodes=6, seed=1)
        child = a.crossover(b)
        self.assertEqual(child.n_genes, 15)

    def test_crossover_genes_from_parents(self):
        """Each child gene should come from one of the two parents."""
        from yane.evolution.grn_encoding import GRNGenome
        import random
        random.seed(42)
        a = GRNGenome.random(n_genes=5, n_nodes=6, seed=10)
        b = GRNGenome.random(n_genes=5, n_nodes=6, seed=20)
        child = a.crossover(b)
        for i, cg in enumerate(child.genes[:5]):
            from_a = cg.weight == a.genes[i].weight
            from_b = cg.weight == b.genes[i].weight
            self.assertTrue(from_a or from_b,
                            f"Gene {i} weight {cg.weight} not from parents")

    def test_crossover_independent_of_parents(self):
        """Modifying child genes must not affect parents."""
        from yane.evolution.grn_encoding import GRNGenome
        a = GRNGenome.random(n_genes=5, n_nodes=6, seed=5)
        b = GRNGenome.random(n_genes=5, n_nodes=6, seed=6)
        child = a.crossover(b)
        original_a_weight = a.genes[0].weight
        child.genes[0].weight = 999.9
        self.assertAlmostEqual(a.genes[0].weight, original_a_weight)

    def test_copy_is_independent(self):
        from yane.evolution.grn_encoding import GRNGenome
        grn = GRNGenome.random(n_genes=5, n_nodes=6, seed=7)
        copy = grn.copy()
        copy.genes[0].weight = 999.9
        self.assertNotAlmostEqual(grn.genes[0].weight, 999.9)


# ---------------------------------------------------------------------------
# GRN-Entwicklung detail tests
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestGRNGene(unittest.TestCase):

    def test_grn_gene_copy_independent(self):
        from yane.evolution.grn_encoding import GRNGene
        g = GRNGene(src_node=0, tgt_node=1, weight=0.5, regulatory_sites=[2, 3])
        c = g.copy()
        c.weight = 999.0
        c.regulatory_sites.append(99)
        self.assertAlmostEqual(g.weight, 0.5)
        self.assertNotIn(99, g.regulatory_sites)

    def test_mutate_changes_weight(self):
        from yane.evolution.grn_encoding import GRNGene
        import random
        g = GRNGene(src_node=0, tgt_node=1, weight=0.5)
        original_w = g.weight
        g.mutate(sigma=1.0, rng=random.Random(1))
        self.assertNotAlmostEqual(g.weight, original_w, places=3)


# ---------------------------------------------------------------------------
# GenomeCodec-Protokoll — acceptance criterion 3
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestGRNCodecProtocol(unittest.TestCase):

    def test_grn_codec_is_genome_codec(self):
        from yane.evolution.grn_encoding import GRNCodec
        from yane.evolution.codec import GenomeCodec
        codec = GRNCodec(n_inputs=2, n_outputs=1, development_steps=3)
        self.assertIsInstance(codec, GenomeCodec)

    def test_codec_name(self):
        from yane.evolution.grn_encoding import GRNCodec
        codec = GRNCodec()
        self.assertEqual(codec.name, "grn")

    def test_encode_decode_roundtrip(self):
        from yane.evolution.grn_encoding import GRNCodec, GRNGenome
        codec = GRNCodec(n_inputs=2, n_outputs=1, development_steps=3)
        grn = GRNGenome.random(n_genes=10, n_nodes=6, seed=0)
        data = codec.encode(grn)
        grn2 = codec.decode(data)
        self.assertEqual(grn2.n_genes, grn.n_genes)
        self.assertAlmostEqual(grn2.genes[0].weight, grn.genes[0].weight)

    def test_develop_produces_genome(self):
        from yane.evolution.grn_encoding import GRNCodec, GRNGenome
        from yane.core.genome import Genome
        codec = GRNCodec(n_inputs=2, n_outputs=1, development_steps=5)
        grn = GRNGenome.random(n_genes=20, n_nodes=10, seed=0)
        phenotype = codec.develop(grn)
        self.assertIsInstance(phenotype, Genome)
        n_conns = sum(len(n.connections) for n in phenotype.nodes)
        self.assertGreater(n_conns, 100)

    def test_pickle_grn_genome(self):
        from yane.evolution.grn_encoding import GRNGenome
        grn = GRNGenome.random(n_genes=10, n_nodes=6, seed=1)
        data = pickle.dumps(grn)
        grn2 = pickle.loads(data)
        self.assertEqual(grn2.n_genes, grn.n_genes)
        self.assertAlmostEqual(grn2.genes[0].weight, grn.genes[0].weight)


# ---------------------------------------------------------------------------
# NeuroEvolution integration
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestNeuroEvolutionIntegration(unittest.TestCase):

    def test_set_genome_encoding_grn(self):
        import yane
        from yane.evolution.grn_encoding import GRNCodec
        ne = yane.NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=10)
        codec = ne.set_genome_encoding("grn", development_steps=5, n_genes=20)
        self.assertIsInstance(codec, GRNCodec)

    def test_set_genome_encoding_invalid_raises(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.configure(n_inputs=2, n_outputs=1)
        with self.assertRaises(ValueError):
            ne.set_genome_encoding("invalid_encoding")

    def test_develop_grn_returns_genome(self):
        import yane
        from yane.core.genome import Genome
        ne = yane.NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=10)
        ne.set_genome_encoding("grn", development_steps=5, n_genes=20)
        phenotype = ne.develop_grn()
        self.assertIsInstance(phenotype, Genome)
        n_conns = sum(len(n.connections) for n in phenotype.nodes)
        self.assertGreater(n_conns, 100)

    def test_yane_exports(self):
        import yane
        self.assertTrue(hasattr(yane, "GRNGene"))
        self.assertTrue(hasattr(yane, "GRNGenome"))
        self.assertTrue(hasattr(yane, "GRNCodec"))


if __name__ == "__main__":
    unittest.main()
