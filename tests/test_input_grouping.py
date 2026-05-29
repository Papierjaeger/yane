"""Tests for Evolvable Input Aggregation Layer (evolution/input_grouping.py).

Covers all four acceptance criteria:
  1. transform() produces the correct output dimension
  2. split_group() (via apply_split_to_genome) adds an input node to the genome
  3. Crossover of two genomes with different groupers runs without error
  4. Checkpoint round-trip preserves groups

Also covers: aggregation types, mutation operators, crossover semantics,
NeuroEvolution integration, and zero-cost-when-disabled invariant.
"""
from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import pytest

from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_genome(n_inputs: int = 4, n_outputs: int = 1) -> Genome:
    g = Genome()
    g.max_nodes = 20
    g.max_connections = 40
    for i in range(n_inputs):
        n = Node(NodeType.INPUT, i)
        n.activation = ActivationType.LINEAR
        g.input_nodes.append(n)
        g.nodes.append(n)
    for j in range(n_outputs):
        n = Node(NodeType.OUTPUT, n_inputs + j)
        n.activation = ActivationType.SIGMOID
        g.output_nodes.append(n)
        g.nodes.append(n)
    # Fully connect
    for inp in g.input_nodes:
        for out in g.output_nodes:
            c = Connection(out, innovation=len(inp.connections) + out.innovation * 100)
            c.weight = 0.5
            inp.connections.append(c)
    g._invalidate_topology()
    return g


def _grouper(n_raw: int, n_groups: int | None = None):
    from yane.evolution.input_grouping import InputGrouper, InputGroup
    if n_groups is None:
        return InputGrouper(n_raw=n_raw)
    per = max(1, n_raw // n_groups)
    groups = []
    for k in range(n_groups):
        members = list(range(k * per, min((k + 1) * per, n_raw))) or [k % n_raw]
        groups.append(InputGroup(members=members))
    return InputGrouper(n_raw=n_raw, initial_groups=groups)


# ---------------------------------------------------------------------------
# AggType / InputGroup basics
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestInputGroup(unittest.TestCase):

    def test_copy_is_independent(self):
        from yane.evolution.input_grouping import InputGroup, AggType
        g = InputGroup(members=[0, 1, 2], aggregation=AggType.MAX)
        c = g.copy()
        c.members.append(9)
        self.assertNotIn(9, g.members)

    def test_all_agg_types_accepted(self):
        from yane.evolution.input_grouping import AggType
        for agg in AggType:
            self.assertIsInstance(agg.value, str)


# ---------------------------------------------------------------------------
# InputGrouper.transform() — acceptance criterion 1
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestInputGrouperTransform(unittest.TestCase):

    def test_output_dimension_equals_n_groups(self):
        ig = _grouper(n_raw=8, n_groups=4)
        raw = [float(i) for i in range(8)]
        out = ig.transform(raw)
        self.assertEqual(len(out), 4, "transform() must return exactly n_groups values")

    def test_identity_grouper_n_outputs_equals_n_raw(self):
        from yane.evolution.input_grouping import InputGrouper
        ig = InputGrouper(n_raw=6)
        self.assertEqual(ig.n_outputs, 6)

    def test_mean_aggregation(self):
        from yane.evolution.input_grouping import InputGrouper, InputGroup, AggType
        ig = InputGrouper(n_raw=4, initial_groups=[
            InputGroup(members=[0, 1], aggregation=AggType.MEAN),
            InputGroup(members=[2, 3], aggregation=AggType.MEAN),
        ])
        out = ig.transform([2.0, 4.0, 1.0, 3.0])
        self.assertAlmostEqual(out[0], 3.0)   # mean(2, 4)
        self.assertAlmostEqual(out[1], 2.0)   # mean(1, 3)

    def test_max_aggregation(self):
        from yane.evolution.input_grouping import InputGrouper, InputGroup, AggType
        ig = InputGrouper(n_raw=3, initial_groups=[
            InputGroup(members=[0, 1, 2], aggregation=AggType.MAX),
        ])
        self.assertAlmostEqual(ig.transform([1.0, 5.0, 3.0])[0], 5.0)

    def test_sum_aggregation(self):
        from yane.evolution.input_grouping import InputGrouper, InputGroup, AggType
        ig = InputGrouper(n_raw=2, initial_groups=[
            InputGroup(members=[0, 1], aggregation=AggType.SUM),
        ])
        self.assertAlmostEqual(ig.transform([2.0, 3.0])[0], 5.0)

    def test_weighted_sum_aggregation(self):
        from yane.evolution.input_grouping import InputGrouper, InputGroup, AggType
        ig = InputGrouper(n_raw=2, initial_groups=[
            InputGroup(members=[0, 1], aggregation=AggType.WEIGHTED_SUM, weights=[2.0, 0.5]),
        ])
        self.assertAlmostEqual(ig.transform([1.0, 4.0])[0], 2.0 * 1.0 + 0.5 * 4.0)

    def test_disabled_group_skipped(self):
        from yane.evolution.input_grouping import InputGrouper, InputGroup, AggType
        ig = InputGrouper(n_raw=3, initial_groups=[
            InputGroup(members=[0], aggregation=AggType.MEAN, enabled=True),
            InputGroup(members=[1], aggregation=AggType.MEAN, enabled=False),
            InputGroup(members=[2], aggregation=AggType.MEAN, enabled=True),
        ])
        out = ig.transform([10.0, 20.0, 30.0])
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0], 10.0)
        self.assertAlmostEqual(out[1], 30.0)

    def test_out_of_range_members_skipped(self):
        from yane.evolution.input_grouping import InputGrouper, InputGroup
        ig = InputGrouper(n_raw=2, initial_groups=[
            InputGroup(members=[0, 99]),  # index 99 is out of range
        ])
        out = ig.transform([5.0, 3.0])
        self.assertAlmostEqual(out[0], 5.0)  # only member 0 is valid


# ---------------------------------------------------------------------------
# split_group() — acceptance criterion 2
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestSplitGroup(unittest.TestCase):

    def test_split_group_increases_n_outputs(self):
        ig = _grouper(n_raw=4, n_groups=2)
        before = ig.n_outputs
        ig.split_group(0)
        self.assertEqual(ig.n_outputs, before + 1)

    def test_apply_split_to_genome_adds_input_node(self):
        from yane.evolution.input_grouping import apply_split_to_genome
        g = _make_genome(n_inputs=2)
        ig = _grouper(n_raw=4, n_groups=2)
        g.grouper = ig
        before = len(g.input_nodes)
        apply_split_to_genome(g, 0)
        self.assertEqual(len(g.input_nodes), before + 1,
                         "apply_split_to_genome must add exactly one input node")

    def test_split_single_member_group_duplicates_member(self):
        from yane.evolution.input_grouping import InputGrouper, InputGroup
        ig = InputGrouper(n_raw=3, initial_groups=[
            InputGroup(members=[0]),
            InputGroup(members=[1, 2]),
        ])
        ig.split_group(0)  # group 0 has only one member
        # Both resulting groups should have at least one member
        new_g = ig.groups[-1]
        self.assertTrue(len(new_g.members) >= 1)

    def test_split_returns_new_group_index(self):
        ig = _grouper(n_raw=6, n_groups=3)
        new_idx = ig.split_group(1)
        self.assertEqual(new_idx, len(ig.groups) - 1)


# ---------------------------------------------------------------------------
# merge_groups()
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestMergeGroups(unittest.TestCase):

    def test_merge_decreases_n_outputs(self):
        ig = _grouper(n_raw=4, n_groups=4)
        before = ig.n_outputs
        ig.merge_groups(0, 1)
        self.assertEqual(ig.n_outputs, before - 1)

    def test_merged_group_contains_both_member_sets(self):
        from yane.evolution.input_grouping import InputGrouper, InputGroup
        ig = InputGrouper(n_raw=4, initial_groups=[
            InputGroup(members=[0, 1]),
            InputGroup(members=[2, 3]),
        ])
        ig.merge_groups(0, 1)
        merged = ig.groups[0]
        for m in [0, 1, 2, 3]:
            self.assertIn(m, merged.members)

    def test_merge_same_index_is_noop(self):
        ig = _grouper(n_raw=4, n_groups=2)
        before = ig.n_outputs
        ig.merge_groups(0, 0)
        self.assertEqual(ig.n_outputs, before)

    def test_apply_merge_to_genome_removes_input_node(self):
        from yane.evolution.input_grouping import apply_merge_to_genome
        g = _make_genome(n_inputs=4)
        ig = _grouper(n_raw=8, n_groups=4)
        g.grouper = ig
        before = len(g.input_nodes)
        apply_merge_to_genome(g, 0, 1)
        self.assertEqual(len(g.input_nodes), before - 1)


# ---------------------------------------------------------------------------
# Other mutation operators
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestOtherMutations(unittest.TestCase):

    def test_add_input_to_group(self):
        from yane.evolution.input_grouping import InputGrouper, InputGroup
        ig = InputGrouper(n_raw=4, initial_groups=[InputGroup(members=[0])])
        ig.add_input_to_group(2, 0)
        self.assertIn(2, ig.groups[0].members)

    def test_add_input_to_group_noop_if_already_member(self):
        from yane.evolution.input_grouping import InputGrouper, InputGroup
        ig = InputGrouper(n_raw=4, initial_groups=[InputGroup(members=[0, 1])])
        ig.add_input_to_group(0, 0)
        self.assertEqual(ig.groups[0].members.count(0), 1)

    def test_remove_input_from_group(self):
        from yane.evolution.input_grouping import InputGrouper, InputGroup
        ig = InputGrouper(n_raw=4, initial_groups=[InputGroup(members=[0, 1, 2])])
        ig.remove_input_from_group(1, 0)
        self.assertNotIn(1, ig.groups[0].members)

    def test_remove_last_member_is_noop(self):
        from yane.evolution.input_grouping import InputGrouper, InputGroup
        ig = InputGrouper(n_raw=4, initial_groups=[InputGroup(members=[0])])
        ig.remove_input_from_group(0, 0)
        self.assertEqual(len(ig.groups[0].members), 1)

    def test_change_aggregation(self):
        from yane.evolution.input_grouping import InputGrouper, InputGroup, AggType
        ig = InputGrouper(n_raw=2, initial_groups=[InputGroup(members=[0, 1])])
        ig.change_aggregation(0, AggType.MAX)
        self.assertEqual(ig.groups[0].aggregation, AggType.MAX)

    def test_create_group(self):
        from yane.evolution.input_grouping import InputGrouper, AggType
        ig = InputGrouper(n_raw=4)
        before = ig.n_outputs
        ig.create_group([0, 2], AggType.SUM)
        self.assertEqual(ig.n_outputs, before + 1)


# ---------------------------------------------------------------------------
# Crossover — acceptance criterion 3
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestInputGrouperCrossover(unittest.TestCase):

    def test_crossover_same_size(self):
        ig_a = _grouper(n_raw=4, n_groups=2)
        ig_b = _grouper(n_raw=4, n_groups=2)
        child = ig_a.crossover(ig_b)
        self.assertEqual(child.n_outputs, 2)

    def test_crossover_different_sizes_no_error(self):
        """Crossover between groupers with different n_groups must not raise."""
        ig_a = _grouper(n_raw=8, n_groups=4)
        ig_b = _grouper(n_raw=8, n_groups=2)
        try:
            child = ig_a.crossover(ig_b)
            self.assertIsNotNone(child)
        except Exception as e:
            self.fail(f"crossover raised: {e}")

    def test_genome_crossover_with_different_groupers_no_error(self):
        """Crossover of two genomes with different groupers must not raise."""
        g_a = _make_genome(n_inputs=4)
        g_b = _make_genome(n_inputs=4)
        g_a.grouper = _grouper(n_raw=8, n_groups=4)
        g_b.grouper = _grouper(n_raw=8, n_groups=2)
        g_a.fitness = 10.0
        g_b.fitness = 5.0
        try:
            child = g_a.crossover(g_b)
            self.assertIsNotNone(child)
        except Exception as e:
            self.fail(f"genome.crossover raised: {e}")

    def test_genome_crossover_preserves_grouper(self):
        g_a = _make_genome(n_inputs=4)
        g_a.grouper = _grouper(n_raw=8, n_groups=4)
        g_a.fitness = 10.0
        g_b = _make_genome(n_inputs=4)
        g_b.fitness = 5.0  # no grouper
        child = g_a.crossover(g_b)
        # Fitter parent (g_a) has grouper → child should too
        self.assertIsNotNone(child.grouper)

    def test_genome_crossover_none_grouper(self):
        """Both parents without groupers → child has no grouper."""
        g_a = _make_genome(n_inputs=2)
        g_b = _make_genome(n_inputs=2)
        g_a.fitness = 1.0
        g_b.fitness = 0.5
        child = g_a.crossover(g_b)
        self.assertIsNone(child.grouper)


# ---------------------------------------------------------------------------
# Checkpoint round-trip — acceptance criterion 4
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestCheckpointRoundTrip(unittest.TestCase):

    def test_pickle_roundtrip_preserves_grouper(self):
        from yane.evolution.input_grouping import InputGroup, AggType
        g = _make_genome(n_inputs=3)
        from yane.evolution.input_grouping import InputGrouper
        g.grouper = InputGrouper(n_raw=6, initial_groups=[
            InputGroup(members=[0, 1], aggregation=AggType.MEAN),
            InputGroup(members=[2, 3], aggregation=AggType.MAX),
            InputGroup(members=[4, 5], aggregation=AggType.SUM),
        ])
        data = pickle.dumps(g)
        g2 = pickle.loads(data)
        self.assertIsNotNone(g2.grouper)
        self.assertEqual(g2.grouper.n_outputs, 3)
        self.assertEqual(g2.grouper.groups[1].aggregation.value, "max")

    def test_copy_preserves_grouper(self):
        g = _make_genome(n_inputs=2)
        g.grouper = _grouper(n_raw=4, n_groups=2)
        c = g.copy()
        self.assertIsNotNone(c.grouper)
        self.assertEqual(c.grouper.n_outputs, g.grouper.n_outputs)

    def test_copy_grouper_is_independent(self):
        g = _make_genome(n_inputs=2)
        g.grouper = _grouper(n_raw=4, n_groups=2)
        c = g.copy()
        c.grouper.groups[0].members.append(99)
        self.assertNotIn(99, g.grouper.groups[0].members)

    def test_neuro_evolution_checkpoint_roundtrip(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_input_grouping(n_groups=2, n_raw=4)
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=10, max_connections=20)
        ne.set_max_iterations(3)
        ne.train(lambda g: sum(g.forward([1.0, 2.0, 3.0, 4.0])))

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.pkl"
            ne.save_checkpoint(path)
            ne2 = yane.NeuroEvolution()
            ne2.load_checkpoint(path)
            best = ne2.get_best()
            self.assertIsNotNone(best.grouper)
            self.assertEqual(best.grouper.n_outputs, 2)


# ---------------------------------------------------------------------------
# NeuroEvolution integration
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestNeuroEvolutionIntegration(unittest.TestCase):

    def test_set_input_grouping_toggles_flag(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_input_grouping(enabled=True, n_groups=2, n_raw=4)
        self.assertTrue(ne._input_grouping_enabled)

    def test_set_input_grouping_disabled_flag(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_input_grouping(enabled=False)
        self.assertFalse(ne._input_grouping_enabled)

    def test_configure_assigns_grouper_to_initial_genome(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_input_grouping(n_groups=2, n_raw=4)
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=10, max_connections=20)
        pop = ne.population
        genomes = list(pop._unevaluated) + list(pop._evaluated) if pop else []
        if genomes:
            self.assertIsNotNone(genomes[0].grouper)

    def test_train_with_input_grouping_does_not_raise(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_input_grouping(n_groups=2, n_raw=4)
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=10, max_connections=20)
        ne.set_max_iterations(3)
        # fitness_fn receives 4 raw inputs, grouper maps → 2 inputs for the network
        ne.train(lambda g: sum(g.forward([1.0, 2.0, 3.0, 4.0])))

    def test_grouper_not_active_when_disabled(self):
        """Without set_input_grouping, genome.grouper should be None (zero cost)."""
        import yane
        ne = yane.NeuroEvolution()
        ne.configure(n_inputs=4, n_outputs=1, max_nodes=10, max_connections=20)
        pop = ne.population
        genomes = list(pop._unevaluated) + list(pop._evaluated) if pop else []
        for g in genomes:
            self.assertIsNone(g.grouper)

    def test_get_input_grouping_diagnostics_when_disabled(self):
        import yane
        ne = yane.NeuroEvolution()
        d = ne.get_input_grouping_diagnostics()
        self.assertFalse(d["enabled"])


if __name__ == "__main__":
    unittest.main()
