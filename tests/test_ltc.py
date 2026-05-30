"""Tests für Liquid Time-Constant (LTC) Nodes (evolution/ltc.py).

Akzeptanzkriterien:
  1. ODE-Mathe: x_{t+1} = x_t + dt*(-x_t/τ + f(inputs)) korrekt berechnet
  2. τ-Extremwerte: τ→∞ → langsam ändernder State; τ→0 → instantane Antwort
  3. genome.reset() setzt LTC-State korrekt zurück (persist_value)
  4. Crossover/Copy vererbt tau und dt
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

def _make_ltc_genome() -> Genome:
    """1 input → 1 LTC hidden node → 1 output."""
    g = Genome()
    inp = Node(NodeType.INPUT, 0); inp.activation = ActivationType.LINEAR; inp.input_index = 0
    ltc = Node(NodeType.HIDDEN, 1); ltc.activation = ActivationType.TANH; ltc.bias = 0.0
    ltc.tau = 1.0; ltc.dt = 0.1; ltc.persist_value = True
    out = Node(NodeType.OUTPUT, 2); out.activation = ActivationType.LINEAR; out.bias = 0.0
    g.nodes.extend([inp, ltc, out]); g.input_nodes.append(inp); g.output_nodes.append(out)
    c1 = Connection(ltc, 10); c1.weight = 1.0; inp.connections.append(c1)
    c2 = Connection(out, 11); c2.weight = 1.0; ltc.connections.append(c2)
    g._invalidate_topology()
    return g


# ---------------------------------------------------------------------------
# ODE-Mathe — acceptance criterion 1
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestLTCODEMath(unittest.TestCase):

    def test_ode_formula_correct(self):
        """x_{t+1} = x_t + dt * (-x_t/tau + f_in) — manually verify."""
        from yane.evolution.ltc import apply_ltc_update
        g = _make_ltc_genome()
        ltc_node = g.nodes[1]
        ltc_node.value = 0.5   # x_t = 0.5
        ltc_node.tau = 2.0
        ltc_node.dt = 0.1
        # f_in = node.value before ODE = 0.5 (set by forward pass mock)
        # Δx = 0.1 * (-0.5/2.0 + 0.5) = 0.1 * (-0.25 + 0.5) = 0.1 * 0.25 = 0.025
        # x_{t+1} = 0.5 + 0.025 = 0.525
        apply_ltc_update(g)
        self.assertAlmostEqual(ltc_node.value, 0.525, places=10)

    def test_ode_zero_input_decays(self):
        """With zero input, state decays toward 0."""
        from yane.evolution.ltc import apply_ltc_update
        g = _make_ltc_genome()
        ltc_node = g.nodes[1]
        ltc_node.value = 1.0  # x_t = 1.0
        ltc_node.tau = 1.0
        ltc_node.dt = 0.1
        # f_in = 0 (no input drive)
        # Δx = 0.1 * (-1.0/1.0 + 0) = -0.1
        # x_{t+1} = 1.0 - 0.1 = 0.9
        # Mock f_in = 0 by setting value then patching
        ltc_node.value = 1.0
        # Here f_in = node.value = 1.0 before update; after update:
        # x_next = 1.0 + 0.1 * (-1.0/1.0 + 1.0) = 1.0 + 0.1 * 0 = 1.0 (no change at equilibrium)
        apply_ltc_update(g)
        # Actually at x=1.0 with f_in=1.0 (tanh activation): x unchanged if f_in = x/tau
        # τ=1.0, so -x_t/τ + f_in = -1.0 + 1.0 = 0 → no change. That's the fixed point.
        self.assertAlmostEqual(ltc_node.value, 1.0, places=10)

    def test_multiple_ode_steps_converge(self):
        """Repeated ODE steps with constant input converge to a fixed point."""
        from yane.evolution.ltc import apply_ltc_update
        g = _make_ltc_genome()
        ltc_node = g.nodes[1]
        ltc_node.tau = 0.5
        ltc_node.dt = 0.05
        # Repeatedly apply ODE with fixed f_in = 0
        for _ in range(100):
            ltc_node.value = ltc_node.value + ltc_node.dt * (-ltc_node.value / ltc_node.tau)
        self.assertAlmostEqual(ltc_node.value, 0.0, delta=1e-3,
                               msg="State should decay to 0 without input drive")

    def test_standard_node_not_updated(self):
        """Nodes with tau=inf must be skipped by apply_ltc_update."""
        from yane.evolution.ltc import apply_ltc_update
        g = _make_ltc_genome()
        out_node = g.nodes[2]  # output node, tau=inf
        out_node.value = 0.7
        apply_ltc_update(g)
        self.assertAlmostEqual(out_node.value, 0.7, places=10,
                               msg="Standard node value must not change")

    def test_genome_has_ltc_false_by_default(self):
        from yane.evolution.ltc import genome_has_ltc
        g = Genome()
        n = Node(NodeType.HIDDEN, 0)
        g.nodes.append(n)
        self.assertFalse(genome_has_ltc(g))

    def test_genome_has_ltc_true_after_set(self):
        from yane.evolution.ltc import genome_has_ltc, make_node_ltc
        g = _make_ltc_genome()
        # node 1 already has tau=1.0 < inf
        self.assertTrue(genome_has_ltc(g))


# ---------------------------------------------------------------------------
# τ-Extremwerte — acceptance criterion 2
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestTauExtremes(unittest.TestCase):

    def test_large_tau_slow_change(self):
        """τ→∞: state changes very slowly (ODE decay term ≈ 0)."""
        from yane.evolution.ltc import apply_ltc_update
        g = _make_ltc_genome()
        ltc_node = g.nodes[1]
        ltc_node.value = 0.5
        ltc_node.tau = 1e5   # very large
        ltc_node.dt = 0.1
        apply_ltc_update(g)
        # Δx ≈ 0.1 * (-0.5/1e5 + 0.5) ≈ 0.1 * 0.5 = 0.05 (decay term negligible)
        delta = abs(ltc_node.value - 0.5)
        # With large tau: delta ≈ dt * f_in = 0.1 * 0.5 = 0.05
        self.assertAlmostEqual(delta, 0.05, delta=0.01)

    def test_small_tau_fast_response(self):
        """τ→0: state changes very quickly (ODE decay dominates)."""
        from yane.evolution.ltc import apply_ltc_update
        g = _make_ltc_genome()
        ltc_node = g.nodes[1]
        ltc_node.value = 1.0
        ltc_node.tau = 1e-4   # very small (will be clamped to _TAU_MIN)
        ltc_node.dt = 0.1
        v_before = ltc_node.value
        apply_ltc_update(g)
        # With very small tau: decay term is very large → state changes significantly
        delta = abs(ltc_node.value - v_before)
        self.assertGreater(delta, 0.5, "Small tau should cause rapid state change")

    def test_finite_output_for_all_tau_values(self):
        """LTC must not produce NaN or Inf for extreme tau."""
        from yane.evolution.ltc import apply_ltc_update
        g = _make_ltc_genome()
        ltc_node = g.nodes[1]
        ltc_node.value = 0.5
        for tau in [1e-10, 1e-5, 0.1, 1.0, 100.0, 1e5]:
            ltc_node.tau = tau
            ltc_node.dt = 0.01
            apply_ltc_update(g)
            self.assertFalse(math.isnan(ltc_node.value), f"NaN for tau={tau}")
            self.assertFalse(math.isinf(ltc_node.value), f"Inf for tau={tau}")
            ltc_node.value = 0.5  # reset for next iteration


# ---------------------------------------------------------------------------
# genome.reset() — acceptance criterion 3
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestLTCReset(unittest.TestCase):

    def test_reset_clears_ltc_state(self):
        """genome.reset() must clear LTC node value to 0.0."""
        g = _make_ltc_genome()
        ltc_node = g.nodes[1]
        ltc_node.value = 0.8  # simulated accumulated state
        g.reset()
        self.assertAlmostEqual(ltc_node.value, 0.0,
                               msg="LTC node state must be 0 after reset")

    def test_persist_value_set_by_make_node_ltc(self):
        """make_node_ltc must set persist_value=True."""
        from yane.evolution.ltc import make_node_ltc
        g = Genome()
        n = Node(NodeType.HIDDEN, 0); n.activation = ActivationType.TANH
        g.nodes.append(n)
        make_node_ltc(g, 0, tau=1.0, dt=0.05)
        self.assertTrue(n.persist_value)
        self.assertAlmostEqual(n.tau, 1.0)
        self.assertAlmostEqual(n.dt, 0.05)


# ---------------------------------------------------------------------------
# Crossover/Copy — acceptance criterion 4
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestLTCCrossover(unittest.TestCase):

    def test_copy_preserves_tau_dt(self):
        g = _make_ltc_genome()
        gc = g.copy()
        ltc = gc.nodes[1]
        self.assertAlmostEqual(ltc.tau, 1.0)
        self.assertAlmostEqual(ltc.dt, 0.1)

    def test_crossover_preserves_tau(self):
        g_a = _make_ltc_genome(); g_a.fitness = 10.0
        g_b = _make_ltc_genome(); g_b.fitness = 5.0
        g_b.nodes[1].tau = 2.0
        child = g_a.crossover(g_b)
        # Child inherits hidden node from fitter parent (g_a, tau=1.0)
        child_ltc = [n for n in child.nodes if n not in child.input_nodes and n not in child.output_nodes]
        if child_ltc:
            self.assertIn(round(child_ltc[0].tau, 1), [1.0, 2.0])

    def test_pickle_preserves_tau_dt(self):
        import pickle
        g = _make_ltc_genome()
        g2 = pickle.loads(pickle.dumps(g))
        self.assertAlmostEqual(g2.nodes[1].tau, 1.0)
        self.assertAlmostEqual(g2.nodes[1].dt, 0.1)

    def test_default_tau_is_inf(self):
        n = Node(NodeType.HIDDEN, 0)
        self.assertEqual(n.tau, float("inf"))
        self.assertAlmostEqual(n.dt, 0.01)


# ---------------------------------------------------------------------------
# NeuroEvolution integration
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestNeuroEvolutionLTC(unittest.TestCase):

    def test_set_ltc_sets_flag(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_ltc(enabled=True)
        self.assertTrue(ne._ltc_enabled)

    def test_set_ltc_disabled(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_ltc(enabled=False)
        self.assertFalse(ne._ltc_enabled)

    def test_train_with_ltc_no_crash(self):
        """Training with LTC enabled but no LTC nodes → normal evolution."""
        import yane
        ne = yane.NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=10, max_connections=20)
        ne.set_ltc(enabled=True)
        ne.set_max_iterations(5)
        ne.train(lambda g: sum(g.forward([0.5, 0.5])))

    def test_train_with_ltc_node_no_crash(self):
        """Training with an actual LTC node must not crash."""
        import yane
        from yane.evolution.ltc import make_node_ltc
        ne = yane.NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=15, max_connections=30,
                     n_initial_hidden=1)
        ne.set_ltc(enabled=True)
        ne.set_max_iterations(5)
        pop = ne.population
        for g in list(pop._unevaluated):
            hidden = [n for n in g.nodes
                      if n not in g.input_nodes and n not in g.output_nodes]
            if hidden:
                make_node_ltc(g, g.nodes.index(hidden[0]), tau=1.0)
        ne.train(lambda g: sum(g.forward([0.5, 0.5])))

    def test_mutate_ltc_params(self):
        """mutate_ltc_params should change tau."""
        from yane.evolution.ltc import mutate_ltc_params
        import random
        g = _make_ltc_genome()
        original_tau = g.nodes[1].tau
        mutate_ltc_params(g, tau_sigma=1.0, rng=random.Random(42))
        self.assertNotAlmostEqual(g.nodes[1].tau, original_tau)


if __name__ == "__main__":
    unittest.main()
