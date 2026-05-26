import unittest
import pytest

from yane import NeuroEvolution
from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection


def _make_genome_with_groups() -> tuple[Genome, Connection, Connection]:
    """2→1 genome where both input connections are in the same group."""
    g = Genome()
    inp0 = Node(NodeType.INPUT, innovation=0)
    inp1 = Node(NodeType.INPUT, innovation=1)
    out = Node(NodeType.OUTPUT, innovation=2)
    g.nodes = [inp0, inp1, out]
    g.input_nodes = [inp0, inp1]
    g.output_nodes = [out]
    c0 = Connection(out, innovation=10)
    c0.weight = 0.5
    c1 = Connection(out, innovation=11)
    c1.weight = 0.7
    inp0.connections = [c0]
    inp1.connections = [c1]
    g._invalidate_topology()
    g.set_weight_group(c0, "group_a")
    g.set_weight_group(c1, "group_a")
    return g, c0, c1


@pytest.mark.ci
class TestConnectionWeightGroup(unittest.TestCase):

    def test_slot_exists(self):
        c = Connection(Node(NodeType.OUTPUT, 0), innovation=1)
        self.assertIsNone(c.weight_group)

    def test_set_group(self):
        c = Connection(Node(NodeType.OUTPUT, 0), innovation=1)
        c.weight_group = "g1"
        self.assertEqual(c.weight_group, "g1")

    def test_copy_preserves_group(self):
        tgt = Node(NodeType.OUTPUT, 0)
        src = Node(NodeType.INPUT, 1)
        c = Connection(tgt, innovation=5)
        c.weight = 0.3
        c.weight_group = "grp"
        node_map = {tgt: tgt, src: src}
        c2 = c.copy(node_map)
        self.assertEqual(c2.weight_group, "grp")

    def test_pickle_roundtrip(self):
        import pickle
        tgt = Node(NodeType.OUTPUT, 0)
        c = Connection(tgt, innovation=3)
        c.weight = 1.5
        c.weight_group = "w_group"
        c2 = pickle.loads(pickle.dumps(c))
        self.assertEqual(c2.weight_group, "w_group")
        self.assertAlmostEqual(c2.weight, 1.5)

    def test_pickle_backward_compat(self):
        """Old pickles without weight_group should default to None."""
        import pickle
        tgt = Node(NodeType.OUTPUT, 0)
        c = Connection(tgt, innovation=3)
        c.weight = 1.0
        state = c.__getstate__()
        del state['weight_group']
        c2 = Connection.__new__(Connection)
        c2.__setstate__(state)
        self.assertIsNone(c2.weight_group)


@pytest.mark.ci
class TestGenomeSharedWeights(unittest.TestCase):

    def test_set_weight_group_assigns(self):
        g, c0, c1 = _make_genome_with_groups()
        self.assertEqual(c0.weight_group, "group_a")
        self.assertEqual(c1.weight_group, "group_a")

    def test_set_weight_group_initialises_dict(self):
        g, c0, c1 = _make_genome_with_groups()
        self.assertIn("group_a", g.weight_groups)

    def test_group_weight_set_from_first_connection(self):
        """Group weight is initialised to the first connection's weight."""
        g = Genome()
        out = Node(NodeType.OUTPUT, 0)
        inp = Node(NodeType.INPUT, 1)
        g.nodes = [inp, out]
        g.input_nodes = [inp]
        g.output_nodes = [out]
        c = Connection(out, innovation=5)
        c.weight = 0.42
        inp.connections = [c]
        g.set_weight_group(c, "g")
        self.assertAlmostEqual(g.weight_groups["g"], 0.42)

    def test_sync_shared_weights_propagates(self):
        g, c0, c1 = _make_genome_with_groups()
        g.weight_groups["group_a"] = 0.99
        g.sync_shared_weights()
        self.assertAlmostEqual(c0.weight, 0.99)
        self.assertAlmostEqual(c1.weight, 0.99)

    def test_get_lamarck_connections_deduplicates(self):
        g, c0, c1 = _make_genome_with_groups()
        lc = g.get_lamarck_connections()
        # Both c0 and c1 are in group_a; only one should appear
        self.assertEqual(len(lc), 1)

    def test_get_lamarck_connections_includes_ungrouped(self):
        g = Genome()
        inp = Node(NodeType.INPUT, 0)
        out = Node(NodeType.OUTPUT, 1)
        g.nodes = [inp, out]
        g.input_nodes = [inp]
        g.output_nodes = [out]
        c_grp = Connection(out, innovation=10)
        c_grp.weight = 0.5
        c_free = Connection(out, innovation=11)
        c_free.weight = 0.3
        inp.connections = [c_grp, c_free]
        g.set_weight_group(c_grp, "g1")
        lc = g.get_lamarck_connections()
        self.assertEqual(len(lc), 2)
        self.assertIn(c_free, lc)

    def test_get_lamarck_connections_skips_disabled(self):
        g, c0, c1 = _make_genome_with_groups()
        c0.enabled = False
        c1.enabled = False
        lc = g.get_lamarck_connections()
        self.assertEqual(len(lc), 0)

    def test_copy_preserves_weight_groups(self):
        g, c0, c1 = _make_genome_with_groups()
        g.weight_groups["group_a"] = 0.77
        copy = g.copy()
        self.assertIn("group_a", copy.weight_groups)
        self.assertAlmostEqual(copy.weight_groups["group_a"], 0.77)

    def test_copy_is_independent(self):
        g, c0, c1 = _make_genome_with_groups()
        copy = g.copy()
        copy.weight_groups["group_a"] = 999.0
        self.assertNotEqual(g.weight_groups.get("group_a"), 999.0)

    def test_members_cache_invalidated_on_group_change(self):
        g, c0, c1 = _make_genome_with_groups()
        g.sync_shared_weights()  # populate cache
        self.assertIn("group_a", g._weight_group_members)
        # Re-assign should clear cache
        g.set_weight_group(c0, "group_b")
        self.assertNotIn("group_b", g._weight_group_members)

    def test_sync_groups_from_reps_updates_all(self):
        g, c0, c1 = _make_genome_with_groups()
        c0.weight = 1.23  # representative
        g._sync_groups_from_reps([c0])
        self.assertAlmostEqual(g.weight_groups["group_a"], 1.23)
        self.assertAlmostEqual(c1.weight, 1.23)

    def test_mutate_syncs_grouped_connections(self):
        """After genome.mutate(), both connections in the same group must have equal weight."""
        g, c0, c1 = _make_genome_with_groups()
        for _ in range(20):
            g.mutate()
            self.assertAlmostEqual(c0.weight, c1.weight, places=10,
                msg=f"Group members out of sync after mutate: {c0.weight} vs {c1.weight}")

    def test_genome_setstate_backward_compat(self):
        """Genomes pickled without weight_groups attributes should load cleanly."""
        import pickle
        g = Genome()
        state = g.__getstate__()
        del state['weight_groups']
        del state['_weight_group_members']
        g2 = Genome.__new__(Genome)
        g2.__setstate__(state)
        self.assertEqual(g2.weight_groups, {})
        self.assertEqual(g2._weight_group_members, {})


@pytest.mark.ci
class TestSharedWeightsIntegration(unittest.TestCase):

    def _make_yane(self) -> NeuroEvolution:
        yane = NeuroEvolution(seed=0)
        yane.configure(2, 1, n_initial_hidden=1)
        yane.set_population_size(8)
        return yane

    def test_default_disabled(self):
        yane = self._make_yane()
        self.assertFalse(yane._shared_weights_enabled)

    def test_set_shared_weights_enables(self):
        yane = self._make_yane()
        yane.set_shared_weights(enabled=True)
        self.assertTrue(yane._shared_weights_enabled)

    def test_disable_flag(self):
        yane = self._make_yane()
        yane.set_shared_weights(enabled=False)
        self.assertFalse(yane._shared_weights_enabled)

    def test_config_dict_field(self):
        yane = self._make_yane()
        yane.set_shared_weights(enabled=True)
        cfg = yane._config_dict()
        self.assertTrue(cfg["shared_weights_enabled"])

    def test_training_with_grouped_connections(self):
        """Training should complete without error when genome has grouped connections."""
        yane = self._make_yane()
        yane.set_shared_weights()
        yane.set_max_iterations(15)

        def fitness_fn(g):
            best_g = g  # assign a group on first eval
            all_conns = [c for n in g.nodes for c in n.connections if c.innovation >= 0]
            if all_conns and not g.weight_groups:
                g.set_weight_group(all_conns[0], "shared")
                if len(all_conns) > 1:
                    g.set_weight_group(all_conns[1], "shared")
            return sum(g.forward([1.0, 0.0]))

        yane.train(fitness_fn)

    def test_lamarck_preserves_group_sync(self):
        """After Lamarck refinement, group members must have equal weights."""
        g, c0, c1 = _make_genome_with_groups()
        from yane.evolution.lamarck_refiner import LamarckRefiner
        refiner = LamarckRefiner()
        refiner.steps = 5

        def fitness(genome):
            return sum(genome.forward([1.0, 0.0]))

        g._invalidate_topology()
        refiner.refine(g, fitness)
        self.assertAlmostEqual(c0.weight, c1.weight, places=10)


if __name__ == "__main__":
    unittest.main()
