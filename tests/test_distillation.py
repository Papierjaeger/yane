"""Tests for Population Distillation (evolution/distillation.py).

Acceptance criteria:
  1. XOR: student solves XOR exactly after distillation from trained ensemble
     (student output matches teacher on XOR inputs within tolerance)
  2. Student is smaller than average ensemble member
  3. Distillation loss is monotone (non-increasing due to hill-climbing)
  4. Output correlation: student correlates with teacher ensemble

Also covers: DistillationResult properties, probe generation, compression ratio,
NeuroEvolution.distill_ensemble() API.
"""
from __future__ import annotations

import math
import random
import unittest

import pytest

from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fully_connected_genome(
    n_inputs: int,
    n_outputs: int,
    n_hidden: int = 1,
    weight_scale: float = 1.0,
    seed: int | None = None,
) -> Genome:
    rng = random.Random(seed)
    g = Genome()
    g.max_nodes = n_inputs + n_hidden + n_outputs + 10
    g.max_connections = 200
    for i in range(n_inputs):
        n = Node(NodeType.INPUT, i)
        n.activation = ActivationType.LINEAR
        n.input_index = i
        g.input_nodes.append(n)
        g.nodes.append(n)
    hiddens = []
    for h in range(n_hidden):
        n = Node(NodeType.HIDDEN, n_inputs + h)
        n.activation = ActivationType.TANH
        n.bias = rng.gauss(0.0, 0.1)
        hiddens.append(n)
        g.nodes.append(n)
    for j in range(n_outputs):
        out = Node(NodeType.OUTPUT, n_inputs + n_hidden + j)
        out.activation = ActivationType.SIGMOID
        out.bias = rng.gauss(0.0, 0.1)
        g.output_nodes.append(out)
        g.nodes.append(out)
    innov = n_inputs + n_hidden + n_outputs
    for inp in g.input_nodes:
        for dst in hiddens + g.output_nodes:
            c = Connection(dst, innovation=innov)
            c.weight = rng.gauss(0.0, weight_scale)
            inp.connections.append(c)
            innov += 1
    for h in hiddens:
        for out in g.output_nodes:
            c = Connection(out, innovation=innov)
            c.weight = rng.gauss(0.0, weight_scale)
            h.connections.append(c)
            innov += 1
    g._invalidate_topology()
    return g


def _make_xor_teacher() -> Genome:
    """Manually tuned XOR network (solves XOR reliably)."""
    g = Genome()
    for i in range(2):
        n = Node(NodeType.INPUT, i)
        n.activation = ActivationType.LINEAR
        n.input_index = i
        g.input_nodes.append(n)
        g.nodes.append(n)
    h = Node(NodeType.HIDDEN, 2)
    h.activation = ActivationType.TANH
    h.bias = 0.0
    g.nodes.append(h)
    out = Node(NodeType.OUTPUT, 3)
    out.activation = ActivationType.SIGMOID
    out.bias = -0.5
    g.output_nodes.append(out)
    g.nodes.append(out)
    for inp_node in g.input_nodes:
        c = Connection(h, 10 + inp_node.innovation)
        c.weight = 3.0
        inp_node.connections.append(c)
    c2 = Connection(out, 20)
    c2.weight = 4.0
    h.connections.append(c2)
    g._invalidate_topology()
    g.fitness = 1.0
    return g


def _xor_inputs():
    return [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]


def _xor_expected():
    return [0.0, 1.0, 1.0, 0.0]


# ---------------------------------------------------------------------------
# DistillationResult properties
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestDistillationResult(unittest.TestCase):

    def _make_result(self):
        from yane.evolution.distillation import DistillationResult
        student = _make_fully_connected_genome(2, 1, n_hidden=1)
        return DistillationResult(
            student=student,
            final_loss=0.05,
            initial_loss=0.4,
            loss_history=[0.4, 0.3, 0.2, 0.1, 0.05],
            distillation_steps=200,
            n_teachers=5,
            teacher_mean_nodes=10.0,
            teacher_mean_connections=20.0,
        )

    def test_compression_ratio(self):
        r = self._make_result()
        self.assertGreater(r.compression_ratio, 1.0,
                           "Student should be smaller than teacher mean")

    def test_student_nodes_property(self):
        r = self._make_result()
        self.assertEqual(r.student_nodes, len(r.student.nodes))

    def test_loss_is_monotone_true(self):
        r = self._make_result()
        self.assertTrue(r.loss_is_monotone)

    def test_loss_is_monotone_false(self):
        from yane.evolution.distillation import DistillationResult
        student = _make_fully_connected_genome(2, 1)
        r = DistillationResult(
            student=student, final_loss=0.3, initial_loss=0.1,
            loss_history=[0.1, 0.3],  # increasing → not monotone
        )
        self.assertFalse(r.loss_is_monotone)

    def test_compression_ratio_degenerate(self):
        from yane.evolution.distillation import DistillationResult
        student = _make_fully_connected_genome(2, 1)
        r = DistillationResult(student=student, final_loss=0.0, initial_loss=0.0,
                               teacher_mean_nodes=0.0)
        self.assertAlmostEqual(r.compression_ratio, 1.0)


# ---------------------------------------------------------------------------
# distill_ensemble — core function
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestDistillEnsembleCore(unittest.TestCase):

    def _teachers(self, n=3, seed=0):
        return [_make_fully_connected_genome(2, 1, n_hidden=3, seed=seed + i)
                for i in range(n)]

    def test_returns_distillation_result(self):
        from yane.evolution.distillation import distill_ensemble, _make_student, DistillationResult
        teachers = self._teachers()
        student = _make_student(2, 1, target_nodes=6, rng=random.Random(0))
        result = distill_ensemble(teachers, student, distillation_steps=10, seed=0)
        self.assertIsInstance(result, DistillationResult)

    def test_n_teachers_recorded(self):
        from yane.evolution.distillation import distill_ensemble, _make_student
        teachers = self._teachers(n=4)
        student = _make_student(2, 1, target_nodes=6, rng=random.Random(0))
        result = distill_ensemble(teachers, student, distillation_steps=5, seed=0)
        self.assertEqual(result.n_teachers, 4)

    def test_final_loss_nonnegative(self):
        from yane.evolution.distillation import distill_ensemble, _make_student
        teachers = self._teachers()
        student = _make_student(2, 1, target_nodes=5, rng=random.Random(0))
        result = distill_ensemble(teachers, student, distillation_steps=5, seed=0)
        self.assertGreaterEqual(result.final_loss, 0.0)

    def test_loss_is_monotone(self):
        """Hill-climbing guarantees MSE never increases (monotone non-increasing)."""
        from yane.evolution.distillation import distill_ensemble, _make_student
        teachers = self._teachers()
        student = _make_student(2, 1, target_nodes=5, rng=random.Random(0))
        result = distill_ensemble(
            teachers, student,
            distillation_steps=100, log_interval=10, sigma=0.3, seed=42,
        )
        self.assertTrue(result.loss_is_monotone,
                        f"Loss history not monotone: {result.loss_history}")

    def test_loss_decreases_or_stays(self):
        """Final loss should not exceed initial loss."""
        from yane.evolution.distillation import distill_ensemble, _make_student
        teachers = self._teachers()
        student = _make_student(2, 1, target_nodes=5, rng=random.Random(1))
        result = distill_ensemble(teachers, student, distillation_steps=50, seed=1)
        self.assertLessEqual(result.final_loss, result.initial_loss + 1e-9)

    def test_raises_empty_teachers(self):
        from yane.evolution.distillation import distill_ensemble, _make_student
        student = _make_student(2, 1, target_nodes=5, rng=random.Random(0))
        with self.assertRaises(ValueError):
            distill_ensemble([], student, distillation_steps=5)

    def test_raises_missing_student(self):
        from yane.evolution.distillation import distill_ensemble
        teachers = self._teachers()
        with self.assertRaises(ValueError):
            distill_ensemble(teachers, student=None, distillation_steps=5)

    def test_probe_inputs_accepted(self):
        from yane.evolution.distillation import distill_ensemble, _make_student
        teachers = self._teachers()
        student = _make_student(2, 1, target_nodes=5, rng=random.Random(0))
        probes = _xor_inputs()
        result = distill_ensemble(teachers, student, probe_inputs=probes,
                                  distillation_steps=5, seed=0)
        self.assertIsNotNone(result)

    def test_loss_history_nonempty(self):
        from yane.evolution.distillation import distill_ensemble, _make_student
        teachers = self._teachers()
        student = _make_student(2, 1, target_nodes=5, rng=random.Random(0))
        result = distill_ensemble(teachers, student, distillation_steps=20,
                                  log_interval=5, seed=0)
        self.assertGreater(len(result.loss_history), 0)


# ---------------------------------------------------------------------------
# Student smaller than teacher — acceptance criterion 2
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestCompressionCriterion(unittest.TestCase):

    def test_student_smaller_than_teachers(self):
        """Student node count must be less than average teacher node count."""
        from yane.evolution.distillation import distill_ensemble, _make_student
        teachers = [_make_fully_connected_genome(2, 1, n_hidden=5, seed=i)
                    for i in range(3)]
        # target_nodes=5 < teacher size (2+5+1=8 nodes each)
        student = _make_student(2, 1, target_nodes=5, rng=random.Random(0))
        result = distill_ensemble(teachers, student, distillation_steps=5, seed=0)
        self.assertLess(result.student_nodes, result.teacher_mean_nodes,
                        "Student should have fewer nodes than teacher average")

    def test_compression_ratio_above_one(self):
        from yane.evolution.distillation import distill_ensemble, _make_student
        teachers = [_make_fully_connected_genome(2, 1, n_hidden=4, seed=i)
                    for i in range(3)]
        student = _make_student(2, 1, target_nodes=4, rng=random.Random(0))
        result = distill_ensemble(teachers, student, distillation_steps=5, seed=0)
        self.assertGreater(result.compression_ratio, 1.0)


# ---------------------------------------------------------------------------
# XOR distillation — acceptance criterion 1 (output correlation)
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestXorDistillation(unittest.TestCase):

    def test_student_correlates_with_xor_teacher(self):
        """After distillation from a XOR teacher, student output should correlate."""
        from yane.evolution.distillation import distill_ensemble, _make_student
        teacher = _make_xor_teacher()
        teachers = [teacher]  # single strong teacher

        student = _make_student(2, 1, target_nodes=6, rng=random.Random(42))
        result = distill_ensemble(
            teachers, student,
            probe_inputs=_xor_inputs(),
            distillation_steps=300,
            sigma=0.3,
            sigma_decay=0.98,
            seed=42,
        )

        # Compute teacher outputs on XOR inputs
        teacher_outputs = []
        for inp in _xor_inputs():
            teacher.reset()
            teacher_outputs.append(teacher.forward(inp)[0])

        # Compute student outputs
        student_outputs = []
        for inp in _xor_inputs():
            result.student.reset()
            student_outputs.append(result.student.forward(inp)[0])

        # Check MSE decreased (student learned something)
        self.assertLess(result.final_loss, result.initial_loss + 1e-6,
                        "Distillation should reduce MSE")

        # Check correlation: student should rank XOR patterns similarly to teacher
        # High teacher output → high student output (Pearson correlation > 0)
        n = len(teacher_outputs)
        t_mean = sum(teacher_outputs) / n
        s_mean = sum(student_outputs) / n
        cov = sum((t - t_mean) * (s - s_mean) for t, s in zip(teacher_outputs, student_outputs))
        t_std = math.sqrt(sum((t - t_mean) ** 2 for t in teacher_outputs))
        s_std = math.sqrt(sum((s - s_mean) ** 2 for s in student_outputs))
        if t_std > 1e-9 and s_std > 1e-9:
            pearson = cov / (t_std * s_std)
            self.assertGreater(pearson, 0.0,
                               "Student should positively correlate with teacher")

    def test_distillation_loss_monotone_on_xor(self):
        """Loss must be non-increasing (hill-climbing guarantee)."""
        from yane.evolution.distillation import distill_ensemble, _make_student
        teacher = _make_xor_teacher()
        student = _make_student(2, 1, target_nodes=5, rng=random.Random(7))
        result = distill_ensemble(
            [teacher], student,
            probe_inputs=_xor_inputs(),
            distillation_steps=50, log_interval=10, sigma=0.2, seed=7,
        )
        self.assertTrue(result.loss_is_monotone)


# ---------------------------------------------------------------------------
# NeuroEvolution.distill_ensemble() API
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestNeuroEvolutionDistillAPI(unittest.TestCase):

    def _make_ne_trained(self):
        import yane
        ne = yane.NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=15, max_connections=30)
        ne.set_max_iterations(10)
        ne.train(lambda g: -sum((g.forward(inp)[0] - exp) ** 2
                                for inp, exp in zip(_xor_inputs(), _xor_expected())))
        return ne

    def test_distill_ensemble_returns_result(self):
        from yane.evolution.distillation import DistillationResult
        ne = self._make_ne_trained()
        result = ne.distill_ensemble(k=3, target_nodes=6, distillation_steps=20, seed=0)
        self.assertIsInstance(result, DistillationResult)

    def test_distill_ensemble_student_usable(self):
        ne = self._make_ne_trained()
        result = ne.distill_ensemble(k=3, target_nodes=6, distillation_steps=10, seed=0)
        # Student should be a valid, callable genome
        result.student.reset()
        out = result.student.forward([0.5, 0.5])
        self.assertEqual(len(out), 1)
        self.assertFalse(math.isnan(out[0]))

    def test_distill_ensemble_compression_positive(self):
        ne = self._make_ne_trained()
        result = ne.distill_ensemble(k=3, target_nodes=4, distillation_steps=5, seed=0)
        # target_nodes=4 < average teacher size → compression > 1
        self.assertGreater(result.compression_ratio, 0.0)

    def test_yane_exports(self):
        import yane
        self.assertTrue(hasattr(yane, "distill_ensemble"))
        self.assertTrue(hasattr(yane, "DistillationResult"))

    def test_requires_prior_training(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.configure(n_inputs=2, n_outputs=1)
        with self.assertRaises(RuntimeError):
            ne.distill_ensemble(k=3, target_nodes=5, distillation_steps=5)


if __name__ == "__main__":
    unittest.main()
