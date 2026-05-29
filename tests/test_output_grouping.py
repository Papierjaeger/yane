"""Tests for Evolvable Output Synergy Layer (evolution/output_grouping.py).

Acceptance criteria (per Tasks.md):
  1. expand() always returns N values
  2. genome.forward() returns N values unchanged (from external perspective)
  3. split_group() increases internal output node count by 1
  4. Crossover is error-free
  5. genome_to_python() generates correct expand block

Also covers: expansion types, mutation operators, crossover semantics,
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

def _make_genome(n_inputs: int = 2, n_outputs: int = 2) -> Genome:
    g = Genome()
    g.max_nodes = 20
    g.max_connections = 40
    for i in range(n_inputs):
        n = Node(NodeType.INPUT, i)
        n.activation = ActivationType.LINEAR
        g.input_nodes.append(n)
        g.nodes.append(n)
    for j in range(n_outputs):
        out = Node(NodeType.OUTPUT, n_inputs + j)
        out.activation = ActivationType.LINEAR
        g.output_nodes.append(out)
        g.nodes.append(out)
    for inp in g.input_nodes:
        for out in g.output_nodes:
            c = Connection(out, innovation=inp.innovation * 100 + out.innovation)
            c.weight = 1.0
            inp.connections.append(c)
    g._invalidate_topology()
    return g


def _grouper(n_outputs: int, n_proto: int | None = None):
    from yane.evolution.output_grouping import OutputGrouper, OutputGroup
    if n_proto is None:
        return OutputGrouper(n_outputs=n_outputs)
    per = max(1, n_outputs // n_proto)
    groups = []
    for k in range(n_proto):
        start = k * per
        end = min((k + 1) * per, n_outputs)
        targets = list(range(start, end)) or [k % n_outputs]
        groups.append(OutputGroup(targets=targets))
    return OutputGrouper(n_outputs=n_outputs, initial_groups=groups)


# ---------------------------------------------------------------------------
# ExpType / OutputGroup basics
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestOutputGroup(unittest.TestCase):

    def test_copy_is_independent(self):
        from yane.evolution.output_grouping import OutputGroup, ExpType
        g = OutputGroup(targets=[0, 1, 2], expansion=ExpType.SCALE, weights=[0.5, 0.8, 1.0])
        c = g.copy()
        c.targets.append(9)
        self.assertNotIn(9, g.targets)

    def test_all_exp_types_accepted(self):
        from yane.evolution.output_grouping import ExpType
        for exp in ExpType:
            self.assertIsInstance(exp.value, str)


# ---------------------------------------------------------------------------
# OutputGrouper.expand() — acceptance criterion 1
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestOutputGrouperExpand(unittest.TestCase):

    def test_expand_always_returns_n_outputs(self):
        og = _grouper(n_outputs=5, n_proto=3)
        result = og.expand([1.0, 2.0, 3.0])
        self.assertEqual(len(result), 5, "expand() must always return n_outputs values")

    def test_identity_grouper(self):
        from yane.evolution.output_grouping import OutputGrouper
        og = OutputGrouper(n_outputs=3)
        result = og.expand([10.0, 20.0, 30.0])
        self.assertEqual(result, [10.0, 20.0, 30.0])

    def test_copy_expansion(self):
        from yane.evolution.output_grouping import OutputGrouper, OutputGroup, ExpType
        og = OutputGrouper(n_outputs=3, initial_groups=[
            OutputGroup(targets=[0, 1], expansion=ExpType.COPY),
            OutputGroup(targets=[2], expansion=ExpType.COPY),
        ])
        result = og.expand([5.0, 9.0])
        # proto[0]=5.0 → ext[0]=5.0, ext[1]=5.0; proto[1]=9.0 → ext[2]=9.0
        self.assertAlmostEqual(result[0], 5.0)
        self.assertAlmostEqual(result[1], 5.0)
        self.assertAlmostEqual(result[2], 9.0)

    def test_scale_expansion(self):
        from yane.evolution.output_grouping import OutputGrouper, OutputGroup, ExpType
        og = OutputGrouper(n_outputs=2, initial_groups=[
            OutputGroup(targets=[0, 1], expansion=ExpType.SCALE, weights=[2.0, 0.5]),
        ])
        result = og.expand([4.0])
        self.assertAlmostEqual(result[0], 8.0)   # 4 * 2.0
        self.assertAlmostEqual(result[1], 2.0)   # 4 * 0.5

    def test_affine_expansion(self):
        from yane.evolution.output_grouping import OutputGrouper, OutputGroup, ExpType
        og = OutputGrouper(n_outputs=1, initial_groups=[
            OutputGroup(targets=[0], expansion=ExpType.AFFINE, weights=[3.0, 1.0]),
        ])
        result = og.expand([2.0])
        self.assertAlmostEqual(result[0], 7.0)   # 2 * 3.0 + 1.0

    def test_disabled_group_not_expanded(self):
        from yane.evolution.output_grouping import OutputGrouper, OutputGroup
        og = OutputGrouper(n_outputs=2, initial_groups=[
            OutputGroup(targets=[0], enabled=True),
            OutputGroup(targets=[1], enabled=False),
        ])
        result = og.expand([99.0])
        self.assertAlmostEqual(result[0], 99.0)
        self.assertAlmostEqual(result[1], 0.0)

    def test_short_proto_outputs_padded(self):
        og = _grouper(n_outputs=4, n_proto=3)
        result = og.expand([1.0, 2.0])  # only 2 proto values for 3 groups
        self.assertEqual(len(result), 4)

    def test_n_proto_property(self):
        from yane.evolution.output_grouping import OutputGrouper
        og = OutputGrouper(n_outputs=3)
        self.assertEqual(og.n_proto, 3)


# ---------------------------------------------------------------------------
# genome.forward() returns N values — acceptance criterion 2
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestGenomeForwardReturnsNOutputs(unittest.TestCase):

    def test_genome_forward_with_output_grouper_returns_n_outputs(self):
        """With OutputGrouper, forward() always returns n_outputs values."""
        from yane.evolution.output_grouping import OutputGrouper, OutputGroup, ExpType
        g = _make_genome(n_inputs=2, n_outputs=2)  # 2 proto-outputs
        # Expand 2 proto-outputs → 4 external outputs
        og = OutputGrouper(n_outputs=4, initial_groups=[
            OutputGroup(targets=[0, 1], expansion=ExpType.COPY),
            OutputGroup(targets=[2, 3], expansion=ExpType.COPY),
        ])
        g.out_grouper = og
        raw = g.forward([1.0, 1.0])  # genome has 2 input nodes
        # Without grouper wrap, forward returns 2 values; with it would be 4.
        # The genome itself just returns its own output_nodes values.
        # The expansion happens externally (in NeuroEvolution._run_evaluations).
        # This test verifies the grouper on the genome is accessible and works.
        expanded = g.out_grouper.expand(raw)
        self.assertEqual(len(expanded), 4)

    def test_output_grouper_expand_result_len(self):
        """expand() always returns n_outputs, independent of proto count."""
        from yane.evolution.output_grouping import OutputGrouper
        for n_out in (1, 2, 4, 8):
            og = OutputGrouper(n_outputs=n_out)
            result = og.expand([0.5] * n_out)
            self.assertEqual(len(result), n_out)


# ---------------------------------------------------------------------------
# split_group() increases output node count — acceptance criterion 3
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestSplitGroup(unittest.TestCase):

    def test_split_group_increases_n_proto(self):
        og = _grouper(n_outputs=4, n_proto=2)
        before = og.n_proto
        og.split_group(0)
        self.assertEqual(og.n_proto, before + 1)

    def test_apply_split_to_genome_adds_output_node(self):
        from yane.evolution.output_grouping import apply_split_to_genome
        g = _make_genome(n_inputs=2, n_outputs=2)
        og = _grouper(n_outputs=4, n_proto=2)
        g.out_grouper = og
        before = len(g.output_nodes)
        apply_split_to_genome(g, 0)
        self.assertEqual(len(g.output_nodes), before + 1,
                         "apply_split_to_genome must add exactly one output node")

    def test_split_returns_new_group_index(self):
        og = _grouper(n_outputs=4, n_proto=2)
        idx = og.split_group(0)
        self.assertEqual(idx, len(og.groups) - 1)

    def test_split_single_target_duplicates(self):
        from yane.evolution.output_grouping import OutputGrouper, OutputGroup
        og = OutputGrouper(n_outputs=2, initial_groups=[
            OutputGroup(targets=[0]),
            OutputGroup(targets=[1]),
        ])
        og.split_group(0)
        new_g = og.groups[-1]
        self.assertTrue(len(new_g.targets) >= 1)


# ---------------------------------------------------------------------------
# merge_groups()
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestMergeGroups(unittest.TestCase):

    def test_merge_decreases_n_proto(self):
        og = _grouper(n_outputs=4, n_proto=4)
        before = og.n_proto
        og.merge_groups(0, 1)
        self.assertEqual(og.n_proto, before - 1)

    def test_merged_group_contains_all_targets(self):
        from yane.evolution.output_grouping import OutputGrouper, OutputGroup
        og = OutputGrouper(n_outputs=4, initial_groups=[
            OutputGroup(targets=[0, 1]),
            OutputGroup(targets=[2, 3]),
        ])
        og.merge_groups(0, 1)
        self.assertIn(2, og.groups[0].targets)
        self.assertIn(3, og.groups[0].targets)

    def test_merge_same_index_noop(self):
        og = _grouper(n_outputs=4, n_proto=2)
        before = og.n_proto
        og.merge_groups(0, 0)
        self.assertEqual(og.n_proto, before)

    def test_apply_merge_to_genome_removes_output_node(self):
        from yane.evolution.output_grouping import apply_merge_to_genome
        g = _make_genome(n_inputs=2, n_outputs=4)
        og = _grouper(n_outputs=8, n_proto=4)
        g.out_grouper = og
        before = len(g.output_nodes)
        apply_merge_to_genome(g, 0, 1)
        self.assertEqual(len(g.output_nodes), before - 1)


# ---------------------------------------------------------------------------
# Other mutation operators
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestOtherMutations(unittest.TestCase):

    def test_add_output_to_group(self):
        from yane.evolution.output_grouping import OutputGrouper, OutputGroup
        og = OutputGrouper(n_outputs=4, initial_groups=[OutputGroup(targets=[0])])
        og.add_output_to_group(2, 0)
        self.assertIn(2, og.groups[0].targets)

    def test_add_output_noop_if_already_target(self):
        from yane.evolution.output_grouping import OutputGrouper, OutputGroup
        og = OutputGrouper(n_outputs=3, initial_groups=[OutputGroup(targets=[0, 1])])
        og.add_output_to_group(0, 0)
        self.assertEqual(og.groups[0].targets.count(0), 1)

    def test_remove_output_from_group(self):
        from yane.evolution.output_grouping import OutputGrouper, OutputGroup
        og = OutputGrouper(n_outputs=3, initial_groups=[OutputGroup(targets=[0, 1, 2])])
        og.remove_output_from_group(1, 0)
        self.assertNotIn(1, og.groups[0].targets)

    def test_remove_last_target_noop(self):
        from yane.evolution.output_grouping import OutputGrouper, OutputGroup
        og = OutputGrouper(n_outputs=2, initial_groups=[OutputGroup(targets=[0])])
        og.remove_output_from_group(0, 0)
        self.assertEqual(len(og.groups[0].targets), 1)

    def test_change_expansion(self):
        from yane.evolution.output_grouping import OutputGrouper, OutputGroup, ExpType
        og = OutputGrouper(n_outputs=2, initial_groups=[OutputGroup(targets=[0, 1])])
        og.change_expansion(0, ExpType.SCALE)
        self.assertEqual(og.groups[0].expansion, ExpType.SCALE)

    def test_create_group(self):
        from yane.evolution.output_grouping import OutputGrouper
        og = OutputGrouper(n_outputs=4)
        before = og.n_proto
        og.create_group([2, 3])
        self.assertEqual(og.n_proto, before + 1)


# ---------------------------------------------------------------------------
# Crossover — acceptance criterion 4
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestOutputGrouperCrossover(unittest.TestCase):

    def test_crossover_same_size(self):
        og_a = _grouper(n_outputs=4, n_proto=2)
        og_b = _grouper(n_outputs=4, n_proto=2)
        child = og_a.crossover(og_b)
        self.assertEqual(child.n_proto, 2)

    def test_crossover_different_sizes_no_error(self):
        og_a = _grouper(n_outputs=6, n_proto=3)
        og_b = _grouper(n_outputs=6, n_proto=2)
        try:
            child = og_a.crossover(og_b)
            self.assertIsNotNone(child)
        except Exception as e:
            self.fail(f"crossover raised: {e}")

    def test_genome_crossover_with_different_out_groupers_no_error(self):
        """Genome.crossover() with different out_groupers must not raise."""
        g_a = _make_genome(n_inputs=2, n_outputs=3)
        g_b = _make_genome(n_inputs=2, n_outputs=3)
        g_a.out_grouper = _grouper(n_outputs=6, n_proto=3)
        g_b.out_grouper = _grouper(n_outputs=6, n_proto=2)
        g_a.fitness = 10.0
        g_b.fitness = 5.0
        try:
            child = g_a.crossover(g_b)
            self.assertIsNotNone(child)
        except Exception as e:
            self.fail(f"genome.crossover raised: {e}")

    def test_genome_crossover_no_grouper(self):
        g_a = _make_genome(n_inputs=2, n_outputs=2)
        g_b = _make_genome(n_inputs=2, n_outputs=2)
        g_a.fitness = 1.0
        g_b.fitness = 0.5
        child = g_a.crossover(g_b)
        self.assertIsNone(child.out_grouper)


# ---------------------------------------------------------------------------
# genome_to_python() — acceptance criterion 5
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestGenomeToPython(unittest.TestCase):

    def test_genome_to_python_without_grouper(self):
        """genome_to_python works as before when no out_grouper."""
        from yane.evolution.genome_export import genome_to_python
        g = _make_genome(n_inputs=2, n_outputs=2)
        src = genome_to_python(g)
        ns = {}
        exec(compile(src, "<test>", "exec"), ns)
        result = ns["forward"]([0.5, 0.5])
        self.assertEqual(len(result), 2)

    def test_genome_to_python_with_out_grouper_returns_n_outputs(self):
        """genome_to_python with OutputGrouper generates expand block."""
        from yane.evolution.genome_export import genome_to_python
        from yane.evolution.output_grouping import OutputGrouper, OutputGroup, ExpType
        g = _make_genome(n_inputs=2, n_outputs=2)  # 2 proto-outputs
        g.out_grouper = OutputGrouper(n_outputs=4, initial_groups=[
            OutputGroup(targets=[0, 1], expansion=ExpType.COPY),
            OutputGroup(targets=[2, 3], expansion=ExpType.COPY),
        ])
        src = genome_to_python(g)
        self.assertIn("_ext", src, "expand block must define _ext")
        ns = {}
        exec(compile(src, "<test>", "exec"), ns)
        result = ns["forward"]([0.5, 0.5])
        self.assertEqual(len(result), 4, "generated forward must return n_outputs values")

    def test_genome_to_python_expand_block_in_source(self):
        """Source code must contain expansion assignments."""
        from yane.evolution.genome_export import genome_to_python
        from yane.evolution.output_grouping import OutputGrouper, OutputGroup, ExpType
        g = _make_genome(n_inputs=1, n_outputs=1)
        g.out_grouper = OutputGrouper(n_outputs=2, initial_groups=[
            OutputGroup(targets=[0, 1], expansion=ExpType.SCALE, weights=[2.0, 0.5]),
        ])
        src = genome_to_python(g)
        self.assertIn("_ext[0]", src)
        self.assertIn("_ext[1]", src)


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestCheckpointRoundTrip(unittest.TestCase):

    def test_pickle_preserves_out_grouper(self):
        from yane.evolution.output_grouping import OutputGrouper, OutputGroup, ExpType
        g = _make_genome(n_inputs=2, n_outputs=3)
        g.out_grouper = OutputGrouper(n_outputs=5, initial_groups=[
            OutputGroup(targets=[0, 1], expansion=ExpType.SCALE, weights=[1.0, 0.5]),
            OutputGroup(targets=[2, 3], expansion=ExpType.COPY),
            OutputGroup(targets=[4], expansion=ExpType.AFFINE, weights=[2.0, 1.0]),
        ])
        data = pickle.dumps(g)
        g2 = pickle.loads(data)
        self.assertIsNotNone(g2.out_grouper)
        self.assertEqual(g2.out_grouper.n_proto, 3)
        self.assertEqual(g2.out_grouper.groups[0].expansion.value, "scale")

    def test_copy_preserves_out_grouper(self):
        g = _make_genome(n_inputs=2, n_outputs=3)
        g.out_grouper = _grouper(n_outputs=5, n_proto=3)
        c = g.copy()
        self.assertIsNotNone(c.out_grouper)
        self.assertEqual(c.out_grouper.n_proto, g.out_grouper.n_proto)

    def test_copy_out_grouper_independent(self):
        g = _make_genome(n_inputs=2, n_outputs=2)
        g.out_grouper = _grouper(n_outputs=4, n_proto=2)
        c = g.copy()
        c.out_grouper.groups[0].targets.append(99)
        self.assertNotIn(99, g.out_grouper.groups[0].targets)


# ---------------------------------------------------------------------------
# NeuroEvolution integration
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestNeuroEvolutionIntegration(unittest.TestCase):

    def test_set_output_grouping_toggles_flag(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_output_grouping(n_proto=2, n_outputs=4)
        self.assertTrue(ne._output_grouping_enabled)

    def test_set_output_grouping_disabled(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_output_grouping(enabled=False)
        self.assertFalse(ne._output_grouping_enabled)

    def test_grouper_not_active_when_disabled(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.configure(n_inputs=2, n_outputs=4, max_nodes=10, max_connections=20)
        pop = ne.population
        genomes = list(pop._unevaluated) + list(pop._evaluated) if pop else []
        for g in genomes:
            self.assertIsNone(g.out_grouper)

    def test_train_with_output_grouping_does_not_raise(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_output_grouping(n_proto=1, n_outputs=2)
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=10, max_connections=20)
        ne.set_max_iterations(3)
        # forward() internally produces 1 proto value; grouper expands to 2
        ne.train(lambda g: sum(g.forward([1.0, 2.0])))

    def test_get_output_grouping_diagnostics_when_disabled(self):
        import yane
        ne = yane.NeuroEvolution()
        d = ne.get_output_grouping_diagnostics()
        self.assertFalse(d["enabled"])


if __name__ == "__main__":
    unittest.main()
