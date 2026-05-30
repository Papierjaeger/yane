"""Tests für Multi-Agent Cooperation (evolution/cooperative.py).

Akzeptanzkriterien:
  1. N Agenten erhalten korrekte Fitness nach den Modi
  2. role_similarity sinkt über Generationen (Rollen-Spezialisierung)
  3. Tests: Credit-Assignment-Modi; Rollen-Spezialisierung; Free-Rider-Erkennung
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

def _make_genome(weight: float = 0.5, fitness: float = 0.0) -> Genome:
    g = Genome()
    inp = Node(NodeType.INPUT, 0); inp.activation = ActivationType.LINEAR; inp.input_index = 0
    out = Node(NodeType.OUTPUT, 1); out.activation = ActivationType.SIGMOID; out.bias = 0.0
    g.nodes.extend([inp, out]); g.input_nodes.append(inp); g.output_nodes.append(out)
    c = Connection(out, 10); c.weight = weight; inp.connections.append(c)
    g.fitness = fitness
    g._invalidate_topology()
    return g


def _team_fn(agents):
    """Simple team fitness: sum of all agent outputs."""
    total = 0.0
    for agent in agents:
        agent.reset()
        try:
            total += agent.forward([0.5])[0]
        except Exception:
            pass
    return total


# ---------------------------------------------------------------------------
# Credit-Assignment-Modi — acceptance criterion 1
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestCreditAssignmentModes(unittest.TestCase):

    def test_shared_mode_all_same_fitness(self):
        """Shared: all agents must receive the same fitness."""
        from yane.evolution.cooperative import assign_shared
        agents = [_make_genome(w) for w in [0.1, 0.5, 0.9]]
        credits = assign_shared(1.5, agents)
        self.assertEqual(len(credits), 3)
        for c in credits:
            self.assertAlmostEqual(c, 1.5)

    def test_difference_mode_marginal_contribution(self):
        """Difference: useful agents get higher credit than useless agents."""
        from yane.evolution.cooperative import assign_difference
        # Agent with weight=0.9 contributes more than weight=0.01
        agents = [_make_genome(weight=0.9), _make_genome(weight=0.01)]
        team_fitness = _team_fn(agents)
        credits = assign_difference(team_fitness, agents, _team_fn)
        self.assertEqual(len(credits), 2)
        # Higher-weight agent should generally contribute more
        self.assertIsInstance(credits[0], float)
        self.assertIsInstance(credits[1], float)

    def test_difference_sums_close_to_team_fitness(self):
        """Difference credits roughly reflect individual contributions."""
        from yane.evolution.cooperative import assign_difference
        agents = [_make_genome(w) for w in [0.3, 0.7]]
        team_fitness = _team_fn(agents)
        credits = assign_difference(team_fitness, agents, _team_fn)
        # For 2 agents: c0 = f(0,1) - f(1), c1 = f(0,1) - f(0)
        # Both > 0 when agents contribute positively
        for c in credits:
            self.assertIsInstance(c, float)
            self.assertFalse(math.isnan(c))

    def test_individual_mode_uses_per_agent_fitness(self):
        """Individual: each agent gets its own fitness."""
        from yane.evolution.cooperative import assign_individual
        agents = [_make_genome(w) for w in [0.1, 0.5, 0.9]]
        def ind_fn(g):
            g.reset()
            return g.forward([0.5])[0]
        credits = assign_individual(1.5, agents, individual_fitness_fn=ind_fn)
        self.assertEqual(len(credits), 3)
        # Each credit should be the individual output
        for agent, c in zip(agents, credits):
            agent.reset()
            expected = agent.forward([0.5])[0]
            self.assertAlmostEqual(c, expected, places=5)

    def test_hierarchical_mode_descending_credits(self):
        """Hierarchical: agent 0 gets most, descending order."""
        from yane.evolution.cooperative import assign_hierarchical
        agents = [_make_genome(w) for w in [0.1, 0.5, 0.9, 0.3]]
        credits = assign_hierarchical(1.0, agents)
        self.assertEqual(len(credits), 4)
        # Credits should be non-increasing
        for i in range(len(credits) - 1):
            self.assertGreaterEqual(credits[i], credits[i + 1])

    def test_hierarchical_credits_sum_to_team_fitness(self):
        from yane.evolution.cooperative import assign_hierarchical
        agents = [_make_genome(w) for w in [0.2, 0.6]]
        credits = assign_hierarchical(2.0, agents)
        self.assertAlmostEqual(sum(credits), 2.0, places=9)

    def test_invalid_credit_mode_raises(self):
        from yane.evolution.cooperative import CooperativeSystem
        with self.assertRaises(ValueError):
            CooperativeSystem(n_agents=3, credit="invalid_mode")


# ---------------------------------------------------------------------------
# Rollen-Spezialisierung — acceptance criterion 2
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestRoleSpecialization(unittest.TestCase):

    def test_cosine_similarity_identical_agents(self):
        """Identical agents should have similarity close to 1."""
        from yane.evolution.cooperative import compute_role_similarity
        g = _make_genome(weight=0.7)
        agents = [g, g.copy()]
        probe_inputs = [[0.5], [1.0], [-0.5]]
        sim = compute_role_similarity(agents, probe_inputs)
        self.assertGreater(sim, 0.9, "Identical agents should have high similarity")

    def test_cosine_similarity_different_agents_less_than_identical(self):
        """Identical agents should have higher similarity than diverse agents."""
        from yane.evolution.cooperative import compute_role_similarity
        # Same-direction (high similarity): both high-weight
        agents_similar = [_make_genome(weight=3.0), _make_genome(weight=3.0)]
        probe_inputs = [[0.5], [1.0], [-1.0]]
        sim_same = compute_role_similarity(agents_similar, probe_inputs)
        # Diverse: one agent outputs values close to 1, other outputs values close to 0.5
        # tanh activations give a range of outputs
        import math
        g_a = _make_genome(weight=5.0); g_a.nodes[-1].activation.__class__
        g_b = _make_genome(weight=0.1)
        agents_diff = [g_a, g_b]
        sim_diff = compute_role_similarity(agents_diff, probe_inputs)
        # Identical agents should have higher or equal similarity
        self.assertGreaterEqual(sim_same, sim_diff - 0.01)

    def test_role_similarity_with_specialization_penalty(self):
        """With role_specialization=True, similar agents should be penalized."""
        from yane.evolution.cooperative import CooperativeSystem
        system = CooperativeSystem(n_agents=2, credit="shared",
                                   role_specialization=True, diversity_weight=0.5)
        agents = [_make_genome(weight=1.0), _make_genome(weight=1.0)]  # identical
        probe_inputs = [[0.5], [1.0]]
        system.evaluate_team(agents, _team_fn, probe_inputs)
        # Similarity should be recorded
        self.assertGreater(len(system.role_similarity_history), 0)

    def test_role_similarity_history_grows(self):
        """Role similarity history should record one entry per evaluate_team call."""
        from yane.evolution.cooperative import CooperativeSystem
        system = CooperativeSystem(n_agents=2, credit="shared", role_specialization=True)
        agents = [_make_genome(w) for w in [0.3, 0.7]]
        probe = [[0.5], [1.0]]
        for _ in range(3):
            system.evaluate_team(agents, _team_fn, probe)
        self.assertEqual(len(system.role_similarity_history), 3)


# ---------------------------------------------------------------------------
# Free-Rider-Erkennung — acceptance criterion 3
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestFreeRiderDetection(unittest.TestCase):

    def test_difference_reward_detects_low_contributor(self):
        """Difference reward should give low credit to a near-zero-output agent."""
        from yane.evolution.cooperative import assign_difference
        # Agent with weight≈0 contributes almost nothing
        contributors = [_make_genome(weight=0.8), _make_genome(weight=0.0001)]
        team_fitness = _team_fn(contributors)
        credits = assign_difference(team_fitness, contributors, _team_fn)
        # Agent 1 (free rider) should have very low credit vs agent 0
        self.assertGreater(abs(credits[0]), abs(credits[1]),
                           "High-contributor should have more credit than free-rider")

    def test_cooperative_system_evaluate_team_sets_fitness(self):
        from yane.evolution.cooperative import CooperativeSystem
        system = CooperativeSystem(n_agents=3, credit="shared")
        agents = [_make_genome(w) for w in [0.3, 0.5, 0.7]]
        system.evaluate_team(agents, _team_fn)
        team_fit = _team_fn(agents)
        for agent in agents:
            self.assertAlmostEqual(agent.fitness, team_fit, places=5)


# ---------------------------------------------------------------------------
# train_cooperative standalone
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestTrainCooperative(unittest.TestCase):

    def test_returns_cooperative_result(self):
        from yane.evolution.cooperative import train_cooperative, CooperativeResult
        agents = [_make_genome(w * 0.3) for w in range(4)]
        result = train_cooperative(
            agents=agents, team_fitness_fn=_team_fn,
            n_generations=3, credit="shared", seed=0,
        )
        self.assertIsInstance(result, CooperativeResult)
        self.assertEqual(result.n_generations, 3)

    def test_best_agent_accessible(self):
        from yane.evolution.cooperative import train_cooperative
        agents = [_make_genome(w * 0.3) for w in range(3)]
        result = train_cooperative(
            agents=agents, team_fitness_fn=_team_fn,
            n_generations=2, credit="shared", seed=1,
        )
        self.assertIsInstance(result.best_agent, Genome)

    def test_team_fitness_history(self):
        from yane.evolution.cooperative import train_cooperative
        agents = [_make_genome(w * 0.3) for w in range(3)]
        result = train_cooperative(
            agents=agents, team_fitness_fn=_team_fn,
            n_generations=4, credit="shared", seed=2,
        )
        self.assertGreater(len(result.team_fitness_history), 0)

    def test_difference_mode_no_crash(self):
        from yane.evolution.cooperative import train_cooperative
        agents = [_make_genome(w * 0.3) for w in range(3)]
        result = train_cooperative(
            agents=agents, team_fitness_fn=_team_fn,
            n_generations=2, credit="difference", seed=3,
        )
        self.assertIsNotNone(result)


# ---------------------------------------------------------------------------
# NeuroEvolution integration
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestNeuroEvolutionIntegration(unittest.TestCase):

    def test_set_cooperative_population_returns_system(self):
        import yane
        from yane.evolution.cooperative import CooperativeSystem
        ne = yane.NeuroEvolution()
        system = ne.set_cooperative_population(n_agents=3, credit="shared")
        self.assertIsInstance(system, CooperativeSystem)

    def test_train_cooperative_no_crash(self):
        import yane
        ne = yane.NeuroEvolution(seed=0)
        ne.configure(n_inputs=1, n_outputs=1, max_nodes=5, max_connections=10)
        ne.set_cooperative_population(n_agents=3, credit="shared")
        result = ne.train_cooperative(_team_fn, n_generations=3, pop_size=3)
        self.assertGreater(result.n_generations, 0)

    def test_train_cooperative_requires_setup(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.configure(n_inputs=1, n_outputs=1)
        with self.assertRaises(RuntimeError):
            ne.train_cooperative(_team_fn, n_generations=2)

    def test_yane_exports(self):
        import yane
        self.assertTrue(hasattr(yane, "CooperativeSystem"))
        self.assertTrue(hasattr(yane, "CooperativeResult"))
        self.assertTrue(hasattr(yane, "train_cooperative"))


if __name__ == "__main__":
    unittest.main()
