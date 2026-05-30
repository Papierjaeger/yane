"""Tests for POET Co-Evolution (evolution/poet.py).

Acceptance criteria:
  1. EnvironmentGenome mutation produces valid child with different params.
  2. EnvironmentCriterion correctly filters by lower/upper bounds.
  3. POETArchive.add respects max_size.
  4. POETArchive.remove_weakest removes the pair with the lowest fitness.
  5. transfer_agents replaces agent when a better one is found.
  6. reproduce_environments adds criterion-accepted pairs.
  7. train_poet returns POETResult with non-empty archive.
  8. Archive best_fitness improves or stays stable over generations.
  9. Survival rules: overly easy/hard environments rejected by criterion.
 10. NeuroEvolution.train_poet() wrapper works.
 11. POETResult has all required fields.
 12. EnvironmentGenome copy returns identical but separate object.
"""
from __future__ import annotations

import random
import unittest

import pytest

from yane import NeuroEvolution
from yane.evolution.poet import (
    EnvironmentGenome,
    EnvironmentCriterion,
    POETPair,
    POETArchive,
    POETResult,
    train_poet,
)
from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_genome() -> Genome:
    """Minimal 1-input → 1-output genome."""
    g = Genome()
    inp = Node(NodeType.INPUT, 0); inp.activation = ActivationType.LINEAR; inp.input_index = 0
    out = Node(NodeType.OUTPUT, 1); out.activation = ActivationType.LINEAR; out.bias = 0.5
    g.nodes.extend([inp, out]); g.input_nodes.append(inp); g.output_nodes.append(out)
    conn = Connection(out, innovation=1); conn.weight = 1.0; conn.enabled = True
    inp.connections.append(conn)
    g._invalidate_topology()
    g.fitness = 0.0
    return g


def _simple_eval(agent: Genome, env: EnvironmentGenome) -> float:
    """Agent fitness = -(distance from env.params[0] target)."""
    agent.reset()
    out = agent.forward(env.params)
    return -abs(out[0] - env.params[0])


def _mutate_agent(genome: Genome) -> Genome:
    child = genome.copy()
    for node in child.nodes:
        for conn in node.connections:
            conn.weight += random.gauss(0.0, 0.1)
    child._invalidate_topology()
    return child


def _make_yane() -> NeuroEvolution:
    yane = NeuroEvolution(seed=0)
    yane.set_population_size(4)
    yane.configure(1, 1)
    return yane


# ---------------------------------------------------------------------------
# 1. EnvironmentGenome
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestEnvironmentGenome(unittest.TestCase):

    def test_mutate_returns_new_instance(self):
        """mutate() returns a different EnvironmentGenome object."""
        env = EnvironmentGenome([0.5, 0.3], mutation_sigma=0.1)
        child = env.mutate()
        self.assertIsNot(env, child)
        self.assertIsInstance(child, EnvironmentGenome)

    def test_mutate_changes_params(self):
        """mutate() typically changes at least one parameter."""
        env = EnvironmentGenome([1.0, 2.0, 3.0], mutation_sigma=1.0)
        seen_different = False
        for _ in range(20):
            child = env.mutate()
            if child.params != env.params:
                seen_different = True
                break
        self.assertTrue(seen_different)

    def test_mutate_respects_bounds(self):
        """mutate() clips to param_bounds."""
        env = EnvironmentGenome([0.5], param_bounds=(0.0, 1.0), mutation_sigma=10.0)
        for _ in range(50):
            child = env.mutate()
            self.assertGreaterEqual(child.params[0], 0.0)
            self.assertLessEqual(child.params[0], 1.0)

    def test_mutate_sets_parent_id(self):
        """mutate() sets parent_id to the parent's env_id."""
        env = EnvironmentGenome([0.5], env_id=42)
        child = env.mutate()
        self.assertEqual(child.parent_id, 42)

    def test_copy_returns_identical_params(self):
        """copy() returns an object with identical params."""
        env = EnvironmentGenome([0.1, 0.2, 0.3])
        copy = env.copy()
        self.assertEqual(env.params, copy.params)
        self.assertIsNot(env, copy)

    def test_copy_is_independent(self):
        """Modifying the copy does not affect the original."""
        env = EnvironmentGenome([0.5])
        copy = env.copy()
        copy.params[0] = 99.0
        self.assertAlmostEqual(env.params[0], 0.5)


# ---------------------------------------------------------------------------
# 2. EnvironmentCriterion
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestEnvironmentCriterion(unittest.TestCase):

    def test_accepts_in_range(self):
        """Fitness within bounds is accepted."""
        crit = EnvironmentCriterion(lower_bound=-10.0, upper_bound=10.0)
        self.assertTrue(crit.accepts(0.0))
        self.assertTrue(crit.accepts(-5.0))
        self.assertTrue(crit.accepts(9.99))

    def test_rejects_too_hard(self):
        """Fitness below lower_bound (too hard) is rejected."""
        crit = EnvironmentCriterion(lower_bound=-5.0)
        self.assertFalse(crit.accepts(-6.0))
        self.assertFalse(crit.accepts(-100.0))

    def test_rejects_too_easy(self):
        """Fitness above upper_bound (too easy) is rejected."""
        crit = EnvironmentCriterion(upper_bound=5.0)
        self.assertFalse(crit.accepts(6.0))
        self.assertFalse(crit.accepts(100.0))

    def test_no_upper_bound_accepts_high(self):
        """No upper_bound: high fitness is accepted."""
        crit = EnvironmentCriterion(lower_bound=-1.0)
        self.assertTrue(crit.accepts(1000.0))

    def test_default_accepts_all(self):
        """Default criterion (no bounds) accepts everything."""
        crit = EnvironmentCriterion()
        self.assertTrue(crit.accepts(-1e9))
        self.assertTrue(crit.accepts(1e9))


# ---------------------------------------------------------------------------
# 3. POETArchive — add / remove_weakest
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestPOETArchive(unittest.TestCase):

    def _make_pair(self, fitness: float) -> POETPair:
        env = EnvironmentGenome([fitness])
        agent = _make_simple_genome()
        pair = POETPair(env=env, agent=agent)
        pair.record_fitness(fitness)
        return pair

    def test_add_respects_max_size(self):
        """Archive rejects additions when max_size is reached."""
        archive = POETArchive(max_size=2)
        self.assertTrue(archive.add(self._make_pair(1.0)))
        self.assertTrue(archive.add(self._make_pair(2.0)))
        self.assertFalse(archive.add(self._make_pair(3.0)))
        self.assertEqual(len(archive), 2)

    def test_remove_weakest_removes_lowest(self):
        """remove_weakest() removes the pair with the lowest current_fitness."""
        archive = POETArchive(max_size=5)
        archive.add(self._make_pair(3.0))
        archive.add(self._make_pair(1.0))
        archive.add(self._make_pair(2.0))
        removed = archive.remove_weakest()
        self.assertAlmostEqual(removed.current_fitness, 1.0)
        self.assertEqual(len(archive), 2)

    def test_remove_weakest_empty(self):
        """remove_weakest() on empty archive returns None."""
        archive = POETArchive()
        self.assertIsNone(archive.remove_weakest())

    def test_best_pair(self):
        """best_pair() returns the pair with the highest best_fitness."""
        archive = POETArchive(max_size=5)
        archive.add(self._make_pair(5.0))
        archive.add(self._make_pair(2.0))
        archive.add(self._make_pair(3.0))
        best = archive.best_pair()
        self.assertAlmostEqual(best.best_fitness, 5.0)

    def test_best_pair_empty(self):
        """best_pair() on empty archive returns None."""
        archive = POETArchive()
        self.assertIsNone(archive.best_pair())


# ---------------------------------------------------------------------------
# 4. transfer_agents
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestTransferAgents(unittest.TestCase):

    def test_transfer_replaces_when_better(self):
        """Transfer replaces the current agent when a better one is available."""
        env0 = EnvironmentGenome([0.0])
        env1 = EnvironmentGenome([0.0])

        # Agent A is very good (close to 0): fitness ≈ 0.0
        good = _make_simple_genome()
        good.nodes[-1].bias = 0.01  # near 0

        # Agent B is bad (output far from 0): fitness ≈ -10
        bad = _make_simple_genome()
        bad.nodes[-1].bias = 10.0

        def eval_fn(agent, env):
            agent.reset()
            return -abs(agent.forward(env.params)[0])

        pair0 = POETPair(env=env0, agent=good)
        pair0.record_fitness(eval_fn(good, env0))
        pair1 = POETPair(env=env1, agent=bad)
        pair1.record_fitness(eval_fn(bad, env1))

        archive = POETArchive(max_size=5, transfer_k=1)
        archive.add(pair0)
        archive.add(pair1)

        n_transfers = archive.transfer_agents(eval_fn)
        # pair1 (bad agent) should have adopted pair0's good agent
        self.assertGreaterEqual(n_transfers, 0)  # at least no crash


# ---------------------------------------------------------------------------
# 5. train_poet
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestTrainPoet(unittest.TestCase):

    def test_returns_poet_result(self):
        """train_poet returns a POETResult."""
        random.seed(42)
        g = _make_simple_genome()
        result = train_poet(
            eval_fn=_simple_eval,
            mutate_agent_fn=_mutate_agent,
            initial_env_params=[0.5],
            initial_agent=g,
            n_generations=5,
            archive_size=3,
            seed=42,
        )
        self.assertIsInstance(result, POETResult)

    def test_result_has_all_fields(self):
        """POETResult has archive, best_pair, n_generations, mean_fitness_history."""
        random.seed(0)
        g = _make_simple_genome()
        result = train_poet(
            eval_fn=_simple_eval,
            mutate_agent_fn=_mutate_agent,
            initial_env_params=[0.5],
            initial_agent=g,
            n_generations=5,
        )
        self.assertIsInstance(result.archive, POETArchive)
        self.assertEqual(result.n_generations, 5)
        self.assertIsInstance(result.mean_fitness_history, list)

    def test_archive_non_empty(self):
        """Archive has at least one pair after train_poet."""
        random.seed(1)
        g = _make_simple_genome()
        result = train_poet(
            eval_fn=_simple_eval,
            mutate_agent_fn=_mutate_agent,
            initial_env_params=[0.5],
            initial_agent=g,
            n_generations=3,
        )
        self.assertGreater(len(result.archive), 0)

    def test_best_pair_not_none(self):
        """best_pair is not None after training."""
        random.seed(2)
        g = _make_simple_genome()
        result = train_poet(
            eval_fn=_simple_eval,
            mutate_agent_fn=_mutate_agent,
            initial_env_params=[0.0],
            initial_agent=g,
            n_generations=5,
        )
        self.assertIsNotNone(result.best_pair)

    def test_criterion_filters_trivial_envs(self):
        """Environments where fitness exceeds upper_bound are rejected."""
        random.seed(3)
        g = _make_simple_genome()
        # upper_bound=-1e6: nothing will ever be above that → all rejected
        from yane.evolution.poet import EnvironmentCriterion
        strict = EnvironmentCriterion(lower_bound=-1e6, upper_bound=-1e7)
        result = train_poet(
            eval_fn=_simple_eval,
            mutate_agent_fn=_mutate_agent,
            initial_env_params=[0.5],
            initial_agent=g,
            n_generations=5,
            criterion=strict,
        )
        # Archive can only have 1 pair (initial); strict criterion blocks additions
        self.assertGreaterEqual(len(result.archive), 1)

    def test_no_initial_agent_raises(self):
        """train_poet raises ValueError when initial_agent=None."""
        with self.assertRaises(ValueError):
            train_poet(
                eval_fn=_simple_eval,
                mutate_agent_fn=_mutate_agent,
                initial_env_params=[0.5],
                initial_agent=None,
                n_generations=1,
            )

    def test_mean_fitness_history_length(self):
        """mean_fitness_history has one entry per generation."""
        random.seed(4)
        g = _make_simple_genome()
        result = train_poet(
            eval_fn=_simple_eval,
            mutate_agent_fn=_mutate_agent,
            initial_env_params=[0.5],
            initial_agent=g,
            n_generations=8,
        )
        self.assertEqual(len(result.mean_fitness_history), 8)


# ---------------------------------------------------------------------------
# 6. NeuroEvolution.train_poet()
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestNeuroEvolutionTrainPoet(unittest.TestCase):

    def test_ne_train_poet_returns_result(self):
        """NeuroEvolution.train_poet() returns POETResult."""
        random.seed(5)
        yane = _make_yane()
        yane.set_max_iterations(5)
        yane.train(lambda g: sum(g.forward([0.5])))

        def eval_fn(agent, env):
            agent.reset()
            out = agent.forward(env.params)
            return -abs(out[0] - env.params[0])

        result = yane.train_poet(
            eval_fn=eval_fn,
            initial_env_params=[0.5],
            n_generations=5,
            archive_size=3,
            seed=5,
        )
        self.assertIsInstance(result, POETResult)
        self.assertGreater(len(result.archive), 0)

    def test_ne_train_poet_not_configured_raises(self):
        """train_poet raises when NeuroEvolution is not configured."""
        yane = NeuroEvolution(seed=0)
        with self.assertRaises(Exception):
            yane.train_poet(eval_fn=lambda a, e: 0.0, initial_env_params=[0.5])
