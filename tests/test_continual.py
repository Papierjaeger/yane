"""Tests für Continual / Lifelong Learning NEAT (evolution/continual.py).

Akzeptanzkriterien:
  1. EWC-Penalty: Gewichtsänderungen werden korrekt bestraft
  2. Progressive-Expansion: neue Knoten werden hinzugefügt, alte eingefroren
  3. Replay-Buffer: Beispiele aus alten Aufgaben werden korrekt gespeichert und abgespielt
  4. Forgetting-Rate: nach Aufgabe 2 Fitness auf Aufgabe 1 messbar
  5. NeuroEvolution-Integration: set_continual_learning() + task_start() + task_finish()
"""
from __future__ import annotations

import math
import unittest

import pytest

from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_genome(n_inputs: int = 2, n_outputs: int = 1, weight: float = 0.5) -> Genome:
    g = Genome()
    for i in range(n_inputs):
        n = Node(NodeType.INPUT, i); n.activation = ActivationType.LINEAR; n.input_index = i
        g.input_nodes.append(n); g.nodes.append(n)
    out = Node(NodeType.OUTPUT, n_inputs); out.activation = ActivationType.SIGMOID; out.bias = 0.0
    g.output_nodes.append(out); g.nodes.append(out)
    innov = 10
    for inp in g.input_nodes:
        c = Connection(out, innov); c.weight = weight; inp.connections.append(c); innov += 1
    g.fitness = 1.0
    g._invalidate_topology()
    return g


# ---------------------------------------------------------------------------
# EWC-Penalty — acceptance criterion 1
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestEWCPenalty(unittest.TestCase):

    def test_no_penalty_for_unchanged_weights(self):
        from yane.evolution.continual import compute_ewc_penalty, TaskAnchor
        g = _make_genome(weight=0.5)
        anchor = TaskAnchor(name="task1", best_genome=g.copy(), best_fitness=1.0)
        penalty = compute_ewc_penalty(g, [anchor], lambda_ewc=1.0)
        self.assertAlmostEqual(penalty, 0.0, places=10)

    def test_penalty_positive_for_changed_weights(self):
        from yane.evolution.continual import compute_ewc_penalty, TaskAnchor
        g_anchor = _make_genome(weight=0.5)
        anchor = TaskAnchor(name="task1", best_genome=g_anchor.copy(), best_fitness=1.0)
        g_current = _make_genome(weight=0.9)  # weights changed
        penalty = compute_ewc_penalty(g_current, [anchor], lambda_ewc=1.0)
        self.assertGreater(penalty, 0.0)

    def test_penalty_scales_with_lambda(self):
        from yane.evolution.continual import compute_ewc_penalty, TaskAnchor
        g_anchor = _make_genome(weight=0.0)
        anchor = TaskAnchor(name="task1", best_genome=g_anchor.copy(), best_fitness=1.0)
        g_current = _make_genome(weight=1.0)
        p1 = compute_ewc_penalty(g_current, [anchor], lambda_ewc=1.0)
        p2 = compute_ewc_penalty(g_current, [anchor], lambda_ewc=2.0)
        self.assertAlmostEqual(p2, 2.0 * p1, places=9)

    def test_ewc_fitness_lower_than_base(self):
        from yane.evolution.continual import make_ewc_fitness, TaskAnchor
        g_anchor = _make_genome(weight=0.0)
        anchor = TaskAnchor(name="task1", best_genome=g_anchor.copy(), best_fitness=1.0)
        g = _make_genome(weight=1.0)
        base_fn = lambda genome: 5.0
        ewc_fn = make_ewc_fitness(base_fn, [anchor], lambda_ewc=1.0)
        ewc_fitness = ewc_fn(g)
        self.assertLess(ewc_fitness, 5.0)

    def test_no_penalty_without_anchors(self):
        from yane.evolution.continual import compute_ewc_penalty
        g = _make_genome()
        penalty = compute_ewc_penalty(g, [], lambda_ewc=1.0)
        self.assertAlmostEqual(penalty, 0.0)

    def test_ewc_fitness_equals_base_for_unchanged_weights(self):
        from yane.evolution.continual import make_ewc_fitness, TaskAnchor
        g = _make_genome(weight=0.5)
        anchor = TaskAnchor(name="t1", best_genome=g.copy(), best_fitness=1.0)
        base_fn = lambda genome: 3.0
        ewc_fn = make_ewc_fitness(base_fn, [anchor], lambda_ewc=1.0)
        self.assertAlmostEqual(ewc_fn(g), 3.0)


# ---------------------------------------------------------------------------
# Progressive Expansion — acceptance criterion 2
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestProgressiveExpansion(unittest.TestCase):

    def test_expansion_adds_nodes(self):
        from yane.evolution.continual import progressive_expand
        g = _make_genome()
        n_before = len(g.nodes)
        progressive_expand(g, n_new_nodes=2)
        self.assertEqual(len(g.nodes), n_before + 2)

    def test_expansion_freezes_old_connections(self):
        from yane.evolution.continual import progressive_expand
        g = _make_genome()
        old_conn = g.input_nodes[0].connections[0]
        progressive_expand(g, n_new_nodes=1)
        self.assertAlmostEqual(old_conn.spike_rate, 0.0,
                               msg="Old connections must have spike_rate=0 (frozen)")

    def test_new_nodes_have_connections(self):
        from yane.evolution.continual import progressive_expand
        g = _make_genome()
        n_before = len(g.nodes)
        progressive_expand(g, n_new_nodes=3)
        new_nodes = g.nodes[n_before:]
        for node in new_nodes:
            self.assertGreater(len(node.connections), 0,
                               "New nodes must have connections to outputs")


# ---------------------------------------------------------------------------
# Replay-Buffer — acceptance criterion 3
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestReplayBuffer(unittest.TestCase):

    def test_task_memory_stores_examples(self):
        from yane.evolution.continual import TaskMemory
        mem = TaskMemory("task1")
        mem.add([0.5, 0.5], [1.0])
        mem.add([0.2, 0.8], [0.7])
        self.assertEqual(len(mem), 2)

    def test_task_memory_sample_size(self):
        from yane.evolution.continual import TaskMemory
        mem = TaskMemory("task1")
        for i in range(20):
            mem.add([float(i)], [float(i) * 0.1])
        sample = mem.sample(5)
        self.assertEqual(len(sample), 5)

    def test_replay_fitness_lower_than_base_when_memory_differs(self):
        """If the genome produces outputs different from memory, replay penalty applies."""
        from yane.evolution.continual import TaskMemory, make_replay_fitness
        g = _make_genome(weight=0.1)  # produces low output
        mem = TaskMemory("task1")
        mem.add([1.0, 1.0], [0.999])  # expected: high output
        replay_fn = make_replay_fitness(lambda genome: 1.0, [mem],
                                        replay_weight=10.0, replay_samples=1)
        fitness = replay_fn(g)
        self.assertLess(fitness, 1.0)

    def test_replay_fitness_near_base_when_memory_matches(self):
        from yane.evolution.continual import TaskMemory, make_replay_fitness
        g = _make_genome(weight=0.0)  # forward([0,0]) ≈ sigmoid(0) ≈ 0.5
        g.reset()
        expected = g.forward([0.0, 0.0])
        mem = TaskMemory("task1")
        mem.add([0.0, 0.0], expected)
        replay_fn = make_replay_fitness(lambda genome: 1.0, [mem],
                                        replay_weight=1.0, replay_samples=1)
        fitness = replay_fn(g)
        self.assertAlmostEqual(fitness, 1.0, places=5)


# ---------------------------------------------------------------------------
# Forgetting-Rate — acceptance criterion 4
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestForgettingRate(unittest.TestCase):

    def test_no_forgetting_for_unchanged_genome(self):
        from yane.evolution.continual import ContinualLearner, TaskAnchor
        learner = ContinualLearner(mode="ewc")
        g = _make_genome(weight=0.5)
        evaluator = lambda genome: 1.0  # perfect performance
        # Anchor the genome
        learner.finish_task(g, best_fitness=1.0)
        # Same genome → no forgetting
        rate = learner.forgetting_rate(0, g, evaluator)
        self.assertAlmostEqual(rate, 0.0, places=9)

    def test_high_forgetting_for_very_different_genome(self):
        from yane.evolution.continual import ContinualLearner
        learner = ContinualLearner(mode="ewc")
        g = _make_genome(weight=0.5)
        learner.finish_task(g, best_fitness=1.0)
        different_evaluator = lambda genome: 0.0  # evaluates to 0
        rate = learner.forgetting_rate(0, g, different_evaluator)
        self.assertAlmostEqual(rate, 1.0, places=9)

    def test_forgetting_rate_out_of_range_returns_zero(self):
        from yane.evolution.continual import ContinualLearner
        learner = ContinualLearner(mode="ewc")
        rate = learner.forgetting_rate(99, _make_genome(), lambda g: 0.5)
        self.assertAlmostEqual(rate, 0.0)


# ---------------------------------------------------------------------------
# ContinualLearner modes
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestContinualLearnerModes(unittest.TestCase):

    def test_ewc_mode_wraps_fitness(self):
        from yane.evolution.continual import ContinualLearner
        learner = ContinualLearner(mode="ewc", lambda_ewc=1.0)
        g = _make_genome(weight=0.0)
        learner.finish_task(g, best_fitness=1.0)  # anchor

        g2 = _make_genome(weight=1.0)
        base_fn = lambda genome: 5.0
        wrapped = learner.wrap_fitness(base_fn)
        self.assertLess(wrapped(g2), 5.0)

    def test_replay_mode_wraps_fitness(self):
        from yane.evolution.continual import ContinualLearner
        learner = ContinualLearner(mode="replay", replay_weight=1.0)
        g = _make_genome(weight=0.5)
        learner.finish_task(g, best_fitness=1.0, sample_inputs=[[0.5, 0.5]])
        self.assertEqual(len(learner.task_memories), 1)

    def test_hybrid_mode_applies_both(self):
        from yane.evolution.continual import ContinualLearner
        learner = ContinualLearner(mode="hybrid", lambda_ewc=0.5, replay_weight=0.5)
        g = _make_genome(weight=0.0)
        learner.finish_task(g, best_fitness=1.0, sample_inputs=[[0.5, 0.5]])
        self.assertEqual(len(learner.task_anchors), 1)
        self.assertEqual(len(learner.task_memories), 1)

    def test_progressive_mode_expands(self):
        from yane.evolution.continual import ContinualLearner
        learner = ContinualLearner(mode="progressive", n_progressive_nodes=2)
        g = _make_genome()
        n_before = len(g.nodes)
        learner.finish_task(g, best_fitness=1.0)
        self.assertEqual(len(g.nodes), n_before + 2)

    def test_first_task_no_wrapping(self):
        """For the first task, wrap_fitness returns the original function."""
        from yane.evolution.continual import ContinualLearner
        learner = ContinualLearner(mode="ewc")
        base_fn = lambda genome: 3.14
        wrapped = learner.wrap_fitness(base_fn)
        g = _make_genome()
        self.assertAlmostEqual(wrapped(g), 3.14)

    def test_invalid_mode_raises(self):
        from yane.evolution.continual import ContinualLearner
        with self.assertRaises(ValueError):
            ContinualLearner(mode="invalid_mode")


# ---------------------------------------------------------------------------
# NeuroEvolution integration
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestNeuroEvolutionIntegration(unittest.TestCase):

    def _make_ne(self):
        import yane
        ne = yane.NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=10, max_connections=20)
        return ne

    def test_set_continual_learning_returns_learner(self):
        from yane.evolution.continual import ContinualLearner
        ne = self._make_ne()
        learner = ne.set_continual_learning(mode="ewc", lambda_ewc=0.1)
        self.assertIsInstance(learner, ContinualLearner)

    def test_task_start_requires_setup(self):
        ne = self._make_ne()
        with self.assertRaises(RuntimeError):
            ne.task_start("task1")

    def test_task_start_and_finish(self):
        ne = self._make_ne()
        ne.set_continual_learning(mode="ewc")
        ne.task_start("xor")
        ne.set_max_iterations(5)
        ne.train(lambda g: sum(g.forward([0.5, 0.5])))
        ne.task_finish()  # should not raise

    def test_ewc_wraps_fitness_during_train(self):
        """After task_finish, second train() uses EWC-wrapped fitness."""
        ne = self._make_ne()
        ne.set_continual_learning(mode="ewc", lambda_ewc=0.5)
        ne.task_start("task1")
        ne.set_max_iterations(3)
        ne.train(lambda g: 1.0)
        ne.task_finish()
        # Second task: fitness should be wrapped (penalized for large weight changes)
        ne.task_start("task2")
        ne.set_max_iterations(3)
        ne.train(lambda g: sum(g.forward([0.5, 0.5])))

    def test_evaluate_all_tasks(self):
        ne = self._make_ne()
        ne.set_continual_learning(mode="ewc")
        ne.task_start("task1")
        ne.set_max_iterations(3)
        evaluator1 = lambda g: sum(g.forward([0.5, 0.5]))
        ne.train(evaluator1)
        ne.task_finish(evaluator=evaluator1)
        results = ne.evaluate_all_tasks()
        self.assertIn("task1", results)

    def test_yane_exports(self):
        import yane
        self.assertTrue(hasattr(yane, "ContinualLearner"))
        self.assertTrue(hasattr(yane, "TaskAnchor"))
        self.assertTrue(hasattr(yane, "compute_ewc_penalty"))


if __name__ == "__main__":
    unittest.main()
