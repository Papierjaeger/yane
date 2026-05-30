"""Tests for Genome-Phylogenie / Stammbaum der Innovationen (evolution/phylogeny.py).

Acceptance criteria:
  1. PhylogenyTree.record() stores nodes correctly.
  2. ancestry() returns ancestor chain oldest-first.
  3. descendants() returns all descendants.
  4. depth() returns correct depth (root=0).
  5. mrca() finds most recent common ancestor.
  6. innovation_attribution() attributes fitness delta per innovation.
  7. to_dict() returns JSON-serialisable structure with all nodes.
  8. to_dot() returns valid DOT string.
  9. Disabled tree records nothing (zero-cost).
 10. max_size limit causes oldest roots to be pruned.
 11. NeuroEvolution.enable_phylogeny() returns PhylogenyTree.
 12. After train(), phylogeny tree has recorded genomes.
 13. root_ids() returns genomes with no parent.
 14. best_fitness_in_lineage() returns max fitness along ancestry.
"""
from __future__ import annotations

import json
import unittest

import pytest

from yane import NeuroEvolution
from yane.evolution.phylogeny import PhylogenyTree, PhylogenyNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tree() -> PhylogenyTree:
    """Build a small 5-node tree:
         1
        / \\
       2   3
       |
       4
       |
       5
    """
    tree = PhylogenyTree()
    tree.enable()
    tree.record(1, None,  fitness=1.0, generation=0)
    tree.record(2, 1,     fitness=1.5, generation=1, innovations=[10])
    tree.record(3, 1,     fitness=1.2, generation=1, innovations=[11])
    tree.record(4, 2,     fitness=2.0, generation=2, innovations=[20, 21])
    tree.record(5, 4,     fitness=1.8, generation=3)
    return tree


def _make_yane() -> NeuroEvolution:
    yane = NeuroEvolution(seed=0)
    yane.set_population_size(6)
    yane.configure(2, 1)
    return yane


# ---------------------------------------------------------------------------
# 1. Basic recording
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestPhylogenyRecord(unittest.TestCase):

    def test_record_stores_node(self):
        """record() stores a PhylogenyNode."""
        tree = PhylogenyTree()
        tree.enable()
        tree.record(1, None, fitness=0.5, generation=0)
        nd = tree.get_node(1)
        self.assertIsInstance(nd, PhylogenyNode)
        self.assertEqual(nd.genome_id, 1)
        self.assertIsNone(nd.parent_id)
        self.assertAlmostEqual(nd.fitness, 0.5)
        self.assertEqual(nd.generation, 0)

    def test_record_with_parent(self):
        """record() with parent stores correct parent_id."""
        tree = PhylogenyTree()
        tree.enable()
        tree.record(1, None, fitness=1.0, generation=0)
        tree.record(2, 1, fitness=1.5, generation=1, innovations=[10])
        nd = tree.get_node(2)
        self.assertEqual(nd.parent_id, 1)
        self.assertEqual(nd.innovations, [10])

    def test_record_fitness_delta(self):
        """record() computes correct fitness_delta."""
        tree = PhylogenyTree()
        tree.enable()
        tree.record(1, None, fitness=1.0, generation=0)
        tree.record(2, 1, fitness=1.8, generation=1)
        nd = tree.get_node(2)
        self.assertAlmostEqual(nd.fitness_delta, 0.8, places=6)

    def test_record_root_has_zero_delta(self):
        """Root node has fitness_delta = 0.0."""
        tree = PhylogenyTree()
        tree.enable()
        tree.record(1, None, fitness=2.0, generation=0)
        self.assertAlmostEqual(tree.get_node(1).fitness_delta, 0.0)

    def test_disabled_records_nothing(self):
        """Disabled tree does not record."""
        tree = PhylogenyTree()
        tree.record(1, None, fitness=1.0, generation=0)
        self.assertIsNone(tree.get_node(1))
        self.assertEqual(tree.size, 0)

    def test_enable_disable_toggle(self):
        """enable/disable toggle works."""
        tree = PhylogenyTree()
        tree.enable()
        tree.record(1, None, fitness=1.0, generation=0)
        tree.disable()
        tree.record(2, 1, fitness=1.5, generation=1)
        self.assertEqual(tree.size, 1)  # only genome 1 was recorded


# ---------------------------------------------------------------------------
# 2. ancestry() / depth()
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestAncestry(unittest.TestCase):

    def test_ancestry_root_is_empty(self):
        """Root node has empty ancestry."""
        tree = _make_tree()
        self.assertEqual(tree.ancestry(1), [])

    def test_ancestry_depth_1(self):
        """Direct child's ancestry is [parent]."""
        tree = _make_tree()
        self.assertEqual(tree.ancestry(2), [1])
        self.assertEqual(tree.ancestry(3), [1])

    def test_ancestry_depth_3(self):
        """Deep node: ancestry is [root, ..., direct_parent], oldest first."""
        tree = _make_tree()
        self.assertEqual(tree.ancestry(5), [1, 2, 4])

    def test_depth_root(self):
        """Root depth is 0."""
        tree = _make_tree()
        self.assertEqual(tree.depth(1), 0)

    def test_depth_deep(self):
        """Depth of node 5 is 3."""
        tree = _make_tree()
        self.assertEqual(tree.depth(5), 3)

    def test_ancestry_unknown_genome(self):
        """Unknown genome_id returns empty list."""
        tree = _make_tree()
        self.assertEqual(tree.ancestry(999), [])


# ---------------------------------------------------------------------------
# 3. descendants()
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestDescendants(unittest.TestCase):

    def test_descendants_root(self):
        """Root node has all other nodes as descendants."""
        tree = _make_tree()
        desc = tree.descendants(1)
        self.assertIn(2, desc)
        self.assertIn(3, desc)
        self.assertIn(4, desc)
        self.assertIn(5, desc)

    def test_descendants_leaf(self):
        """Leaf node has no descendants."""
        tree = _make_tree()
        self.assertEqual(tree.descendants(5), [])
        self.assertEqual(tree.descendants(3), [])

    def test_descendants_count(self):
        """Root (5-node tree) has exactly 4 descendants."""
        tree = _make_tree()
        self.assertEqual(len(tree.descendants(1)), 4)


# ---------------------------------------------------------------------------
# 4. mrca()
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestMRCA(unittest.TestCase):

    def test_mrca_same_node(self):
        """MRCA of a node with itself is itself."""
        tree = _make_tree()
        self.assertEqual(tree.mrca(2, 2), 2)

    def test_mrca_siblings(self):
        """MRCA of two siblings is their common parent."""
        tree = _make_tree()
        self.assertEqual(tree.mrca(2, 3), 1)

    def test_mrca_ancestor_descendant(self):
        """MRCA of ancestor and descendant is the ancestor."""
        tree = _make_tree()
        result = tree.mrca(1, 5)
        self.assertEqual(result, 1)

    def test_mrca_no_common(self):
        """MRCA of two disconnected genomes is None."""
        tree = PhylogenyTree()
        tree.enable()
        tree.record(1, None, fitness=1.0, generation=0)
        tree.record(2, None, fitness=1.5, generation=0)  # separate root
        self.assertIsNone(tree.mrca(1, 2))


# ---------------------------------------------------------------------------
# 5. innovation_attribution()
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestInnovationAttribution(unittest.TestCase):

    def test_attribution_per_innovation(self):
        """Fitness delta is split equally among innovations."""
        tree = _make_tree()
        attr = tree.innovation_attribution(4)
        # Genome 4 has innovations [20, 21], delta = 2.0 - 1.5 = 0.5
        self.assertIn(20, attr)
        self.assertIn(21, attr)
        self.assertAlmostEqual(attr[20], 0.25, places=6)
        self.assertAlmostEqual(attr[21], 0.25, places=6)

    def test_attribution_no_innovations(self):
        """Genome with no innovations returns empty dict."""
        tree = _make_tree()
        attr = tree.innovation_attribution(5)  # genome 5 has no innovations
        self.assertEqual(attr, {})

    def test_attribution_single_innovation(self):
        """Single innovation gets the full fitness delta."""
        tree = _make_tree()
        attr = tree.innovation_attribution(2)  # innovation [10], delta = 1.5 - 1.0 = 0.5
        self.assertAlmostEqual(attr[10], 0.5, places=6)

    def test_attribution_unknown_genome(self):
        """Unknown genome returns empty dict."""
        tree = _make_tree()
        self.assertEqual(tree.innovation_attribution(999), {})


# ---------------------------------------------------------------------------
# 6. to_dict()
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestToDict(unittest.TestCase):

    def test_to_dict_is_serialisable(self):
        """to_dict() result is JSON-serialisable."""
        tree = _make_tree()
        d = tree.to_dict()
        json.dumps(d)  # should not raise

    def test_to_dict_contains_all_nodes(self):
        """to_dict() has entries for all recorded genomes."""
        tree = _make_tree()
        d = tree.to_dict()
        for gid in [1, 2, 3, 4, 5]:
            self.assertIn(str(gid), d["nodes"])

    def test_to_dict_roots(self):
        """to_dict() roots list contains the root genome."""
        tree = _make_tree()
        d = tree.to_dict()
        self.assertIn(1, d["roots"])

    def test_to_dict_depth_field(self):
        """to_dict() nodes have a depth field."""
        tree = _make_tree()
        d = tree.to_dict()
        self.assertEqual(d["nodes"]["5"]["depth"], 3)
        self.assertEqual(d["nodes"]["1"]["depth"], 0)

    def test_to_json_returns_string(self):
        """to_json() returns a JSON string."""
        tree = _make_tree()
        s = tree.to_json()
        self.assertIsInstance(s, str)
        parsed = json.loads(s)
        self.assertIn("nodes", parsed)


# ---------------------------------------------------------------------------
# 7. to_dot()
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestToDot(unittest.TestCase):

    def test_to_dot_returns_string(self):
        """to_dot() returns a non-empty string."""
        tree = _make_tree()
        dot = tree.to_dot()
        self.assertIsInstance(dot, str)
        self.assertTrue(len(dot) > 0)

    def test_to_dot_starts_with_digraph(self):
        """to_dot() starts with 'digraph'."""
        tree = _make_tree()
        dot = tree.to_dot()
        self.assertTrue(dot.strip().startswith("digraph"))

    def test_to_dot_contains_node_ids(self):
        """to_dot() contains node identifiers for all genomes."""
        tree = _make_tree()
        dot = tree.to_dot()
        for gid in [1, 2, 3, 4, 5]:
            self.assertIn(f"n{gid}", dot)

    def test_to_dot_contains_arrows(self):
        """to_dot() contains edge arrows between nodes."""
        tree = _make_tree()
        dot = tree.to_dot()
        self.assertIn("->", dot)

    def test_to_dot_max_nodes_limits_output(self):
        """to_dot(max_nodes=2) limits the visible nodes to 2."""
        tree = _make_tree()
        dot = tree.to_dot(max_nodes=2)
        # Count node definition lines (contain style=filled)
        import re
        node_defs = re.findall(r'style=filled', dot)
        self.assertLessEqual(len(node_defs), 2)


# ---------------------------------------------------------------------------
# 8. root_ids() and best_fitness_in_lineage()
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestRootAndLineageFitness(unittest.TestCase):

    def test_root_ids(self):
        """root_ids() returns only genomes with no parent."""
        tree = _make_tree()
        self.assertEqual(tree.root_ids(), [1])

    def test_best_fitness_in_lineage(self):
        """best_fitness_in_lineage() returns max fitness along the ancestry."""
        tree = _make_tree()
        # Lineage of 5: [1(1.0), 2(1.5), 4(2.0), 5(1.8)]
        best = tree.best_fitness_in_lineage(5)
        self.assertAlmostEqual(best, 2.0)

    def test_best_fitness_root(self):
        """best_fitness_in_lineage() for root is its own fitness."""
        tree = _make_tree()
        self.assertAlmostEqual(tree.best_fitness_in_lineage(1), 1.0)


# ---------------------------------------------------------------------------
# 9. max_size pruning
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestMaxSizePruning(unittest.TestCase):

    def test_max_size_limits_tree(self):
        """Tree with max_size=3 prunes oldest roots when exceeded."""
        tree = PhylogenyTree(max_size=3)
        tree.enable()
        tree.record(1, None, fitness=1.0, generation=0)
        tree.record(2, None, fitness=1.1, generation=0)
        tree.record(3, None, fitness=1.2, generation=0)
        tree.record(4, None, fitness=1.3, generation=0)  # triggers trim
        self.assertLessEqual(tree.size, 3)


# ---------------------------------------------------------------------------
# 10. NeuroEvolution integration
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestNeuroEvolutionPhylogeny(unittest.TestCase):

    def test_enable_phylogeny_returns_tree(self):
        """enable_phylogeny() returns a PhylogenyTree."""
        yane = _make_yane()
        tree = yane.enable_phylogeny()
        self.assertIsInstance(tree, PhylogenyTree)
        self.assertTrue(tree.is_enabled)

    def test_get_phylogeny_before_enable_is_none(self):
        """get_phylogeny() returns None before enable_phylogeny() is called."""
        yane = _make_yane()
        self.assertIsNone(yane.get_phylogeny())

    def test_disable_phylogeny(self):
        """disable_phylogeny() disables recording."""
        yane = _make_yane()
        yane.enable_phylogeny()
        yane.disable_phylogeny()
        self.assertFalse(yane.get_phylogeny().is_enabled)

    def test_train_records_genomes(self):
        """After train(), the phylogeny tree has recorded at least some genomes."""
        yane = _make_yane()
        yane.enable_phylogeny()
        yane.set_max_iterations(20)
        yane.train(lambda g: sum(g.forward([0.5, 0.5])))
        tree = yane.get_phylogeny()
        self.assertGreater(tree.size, 0)

    def test_phylogeny_tree_has_root(self):
        """After training, the phylogeny tree has at least one root."""
        yane = _make_yane()
        yane.enable_phylogeny()
        yane.set_max_iterations(15)
        yane.train(lambda g: sum(g.forward([0.5, 0.5])))
        tree = yane.get_phylogeny()
        self.assertGreater(len(tree.root_ids()), 0)

    def test_max_size_respected_in_training(self):
        """max_size is respected during training."""
        yane = _make_yane()
        yane.enable_phylogeny(max_size=5)
        yane.set_max_iterations(30)
        yane.train(lambda g: sum(g.forward([0.5, 0.5])))
        tree = yane.get_phylogeny()
        self.assertLessEqual(tree.size, 5)

    def test_to_dict_after_training(self):
        """to_dict() is JSON-serialisable after training."""
        yane = _make_yane()
        yane.enable_phylogeny()
        yane.set_max_iterations(15)
        yane.train(lambda g: sum(g.forward([0.5, 0.5])))
        import json
        d = yane.get_phylogeny().to_dict()
        json.dumps(d)  # should not raise

    def test_to_dot_after_training(self):
        """to_dot() returns valid DOT after training."""
        yane = _make_yane()
        yane.enable_phylogeny()
        yane.set_max_iterations(15)
        yane.train(lambda g: sum(g.forward([0.5, 0.5])))
        dot = yane.get_phylogeny().to_dot()
        self.assertTrue(dot.strip().startswith("digraph"))
