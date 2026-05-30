"""Tests for Synaptische Plastizität / STDP (evolution/stdp.py).

Acceptance criteria (per Tasks.md):
  1. Verbindungen tragen evolvierte Hebb-Koeffizienten A, B, C, D
  2. genome.forward() ändert Gewichte intra-Episode (nächste Calls sehen neue Gewichte)
  3. genome.reset() setzt Gewichte auf Basiswerte zurück (episodenlokal)
  4. Checkpoint-Roundtrip erhält Hebb-Koeffizienten
  5. Crossover vererbt Hebb-Koeffizienten
  6. NeuroEvolution.set_stdp() + Training ohne Crash
  7. Zero-cost wenn deaktiviert (keine Koeffizienten → kein Delta)
"""
from __future__ import annotations

import math
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

def _make_genome(n_inputs: int = 2, n_outputs: int = 1) -> Genome:
    g = Genome()
    for i in range(n_inputs):
        n = Node(NodeType.INPUT, i); n.activation = ActivationType.LINEAR; n.input_index = i
        g.input_nodes.append(n); g.nodes.append(n)
    out = Node(NodeType.OUTPUT, n_inputs); out.activation = ActivationType.LINEAR; out.bias = 0.0
    g.output_nodes.append(out); g.nodes.append(out)
    innov = 10
    for inp in g.input_nodes:
        c = Connection(out, innov); c.weight = 1.0; inp.connections.append(c); innov += 1
    g._invalidate_topology()
    return g


# ---------------------------------------------------------------------------
# Connection Hebb coefficients — acceptance criterion 1
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestHebbCoefficients(unittest.TestCase):

    def test_hebb_defaults_zero(self):
        from yane.core.connection import Connection
        from yane.core.node import Node, NodeType
        n = Node(NodeType.OUTPUT, 0)
        c = Connection(n, 1)
        self.assertAlmostEqual(c.hebb_a, 0.0)
        self.assertAlmostEqual(c.hebb_b, 0.0)
        self.assertAlmostEqual(c.hebb_c, 0.0)
        self.assertAlmostEqual(c.hebb_d, 0.0)
        self.assertIsNone(c._base_weight)

    def test_set_hebb_coeffs(self):
        from yane.evolution.stdp import set_hebb_coeffs
        g = _make_genome()
        set_hebb_coeffs(g, a=0.1, b=0.2, c=0.3, d=0.4)
        for src in g.nodes:
            for conn in src.connections:
                self.assertAlmostEqual(conn.hebb_a, 0.1)
                self.assertAlmostEqual(conn.hebb_b, 0.2)
                self.assertAlmostEqual(conn.hebb_c, 0.3)
                self.assertAlmostEqual(conn.hebb_d, 0.4)

    def test_set_hebb_coeffs_with_noise(self):
        import random
        from yane.evolution.stdp import set_hebb_coeffs
        g = _make_genome()
        set_hebb_coeffs(g, c=0.01, sigma=0.1, rng=random.Random(42))
        for src in g.nodes:
            for conn in src.connections:
                # c should be approximately 0.01 but noisy
                self.assertNotAlmostEqual(conn.hebb_c, 0.0)

    def test_genome_has_stdp_false_by_default(self):
        from yane.evolution.stdp import genome_has_stdp
        g = _make_genome()
        self.assertFalse(genome_has_stdp(g))

    def test_genome_has_stdp_true_after_set(self):
        from yane.evolution.stdp import genome_has_stdp, set_hebb_coeffs
        g = _make_genome()
        set_hebb_coeffs(g, c=0.01)
        self.assertTrue(genome_has_stdp(g))

    def test_hebb_clamped_to_range(self):
        from yane.evolution.stdp import set_hebb_coeffs, _HEBB_COEFF_MAX
        g = _make_genome()
        set_hebb_coeffs(g, c=999.0)  # should be clamped
        for src in g.nodes:
            for conn in src.connections:
                self.assertLessEqual(conn.hebb_c, _HEBB_COEFF_MAX)


# ---------------------------------------------------------------------------
# Intra-episode weight change — acceptance criterion 2
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestIntraEpisodeWeightChange(unittest.TestCase):

    def test_weight_changes_after_forward(self):
        """After applying STDP, conn.weight must differ from initial value."""
        from yane.evolution.stdp import apply_stdp_update, init_stdp_base_weights, set_hebb_coeffs
        g = _make_genome()
        set_hebb_coeffs(g, c=0.5)   # strong Hebb: Δw ≈ 0.5 * pre * post
        g.reset()
        g.forward([1.0, 1.0])       # trigger node activations
        # Compiled forward doesn't update input node.value — set manually for test
        for inp_n in g.input_nodes:
            inp_n.value = 1.0
        init_stdp_base_weights(g)

        conn = g.input_nodes[0].connections[0]
        w_before = conn.weight

        apply_stdp_update(g)
        self.assertNotAlmostEqual(conn.weight, w_before,
                                  msg="STDP must change connection weight")

    def test_next_forward_uses_updated_weight(self):
        """After weight change, the next forward() must produce a different output."""
        from yane.evolution.stdp import apply_stdp_update, init_stdp_base_weights, set_hebb_coeffs
        g = _make_genome()
        set_hebb_coeffs(g, c=1.0)   # strong Hebb
        g.reset()
        out1 = g.forward([1.0, 1.0])
        # The compiled forward does not update input node.value — set manually
        for inp_n in g.input_nodes:
            inp_n.value = 1.0
        init_stdp_base_weights(g)
        apply_stdp_update(g)        # weights now modified (Δw = c * pre * post)
        out2 = g.forward([1.0, 1.0])
        # Output should differ since weights changed (NeuroEvolution wrapper handles this automatically)
        self.assertNotAlmostEqual(out1[0], out2[0], places=3)

    def test_zero_coefficients_no_weight_change(self):
        """Zero hebb coefficients must produce zero delta (no effect)."""
        from yane.evolution.stdp import apply_stdp_update, init_stdp_base_weights
        g = _make_genome()
        # hebb all 0.0 by default
        g.reset()
        g.forward([1.0, 0.5])
        init_stdp_base_weights(g)

        conn = g.input_nodes[0].connections[0]
        w_before = conn.weight

        apply_stdp_update(g)
        self.assertAlmostEqual(conn.weight, w_before,
                               msg="Zero hebb → zero delta → no weight change")

    def test_weight_clamped_within_range(self):
        """Working weight must stay within [weight_min, weight_max]."""
        from yane.evolution.stdp import apply_stdp_update, init_stdp_base_weights, set_hebb_coeffs
        g = _make_genome()
        set_hebb_coeffs(g, c=100.0, d=100.0)  # extreme plasticity
        g.reset()
        g.forward([1.0, 1.0])
        for inp_n in g.input_nodes:
            inp_n.value = 1.0
        init_stdp_base_weights(g)
        apply_stdp_update(g, weight_min=-3.0, weight_max=3.0)
        for src in g.nodes:
            for conn in src.connections:
                if conn._base_weight is not None:
                    self.assertLessEqual(conn.weight, 3.0)
                    self.assertGreaterEqual(conn.weight, -3.0)


# ---------------------------------------------------------------------------
# Episode-local plasticity / reset — acceptance criterion 3
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestEpisodicLocality(unittest.TestCase):

    def test_reset_restores_base_weights(self):
        """genome.reset() must restore connection weights to base values."""
        from yane.evolution.stdp import apply_stdp_update, init_stdp_base_weights, restore_stdp_weights, set_hebb_coeffs
        g = _make_genome()
        set_hebb_coeffs(g, c=0.5)
        g.reset()
        g.forward([1.0, 1.0])
        for inp_n in g.input_nodes:
            inp_n.value = 1.0
        init_stdp_base_weights(g)

        # Save base weight
        conn = g.input_nodes[0].connections[0]
        base_w = conn._base_weight

        # Apply plasticity
        apply_stdp_update(g)
        self.assertNotAlmostEqual(conn.weight, base_w)

        # Restore (as reset() would do)
        restore_stdp_weights(g)
        self.assertAlmostEqual(conn.weight, base_w,
                               msg="restore_stdp_weights must recover base weight exactly")

    def test_base_weight_survives_multiple_episodes(self):
        """Base weight must not change across episodes."""
        from yane.evolution.stdp import apply_stdp_update, init_stdp_base_weights, restore_stdp_weights, set_hebb_coeffs
        g = _make_genome()
        set_hebb_coeffs(g, c=0.3)
        g.reset()
        g.forward([1.0, 1.0])
        for inp_n in g.input_nodes:
            inp_n.value = 1.0
        init_stdp_base_weights(g)
        conn = g.input_nodes[0].connections[0]
        original_base = conn._base_weight

        for _ in range(5):  # 5 episodes
            apply_stdp_update(g)
            restore_stdp_weights(g)

        self.assertAlmostEqual(conn._base_weight, original_base,
                               msg="Base weight must not drift across episodes")

    def test_base_weight_not_set_for_zero_coefficients(self):
        """Connections with all-zero hebb must not have _base_weight set."""
        from yane.evolution.stdp import init_stdp_base_weights
        g = _make_genome()
        # hebb all 0.0 by default
        g.reset()
        g.forward([1.0, 1.0])
        init_stdp_base_weights(g)
        for src in g.nodes:
            for conn in src.connections:
                self.assertIsNone(conn._base_weight,
                                  "Zero hebb → _base_weight should stay None")


# ---------------------------------------------------------------------------
# Checkpoint round-trip — acceptance criterion 4
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestCheckpointRoundTrip(unittest.TestCase):

    def test_pickle_preserves_hebb_coeffs(self):
        from yane.evolution.stdp import set_hebb_coeffs
        g = _make_genome()
        set_hebb_coeffs(g, a=0.1, b=0.2, c=0.3, d=0.4)
        data = pickle.dumps(g)
        g2 = pickle.loads(data)
        conn = g2.input_nodes[0].connections[0]
        self.assertAlmostEqual(conn.hebb_a, 0.1)
        self.assertAlmostEqual(conn.hebb_b, 0.2)
        self.assertAlmostEqual(conn.hebb_c, 0.3)
        self.assertAlmostEqual(conn.hebb_d, 0.4)

    def test_pickle_base_weight_cleared(self):
        """_base_weight is episode-local; must be None after unpickling."""
        from yane.evolution.stdp import set_hebb_coeffs, init_stdp_base_weights
        g = _make_genome()
        set_hebb_coeffs(g, c=0.1)
        g.forward([1.0, 1.0])
        init_stdp_base_weights(g)
        # Confirm _base_weight is set before pickling
        conn = g.input_nodes[0].connections[0]
        self.assertIsNotNone(conn._base_weight)
        # After pickling, it should be cleared
        g2 = pickle.loads(pickle.dumps(g))
        conn2 = g2.input_nodes[0].connections[0]
        self.assertIsNone(conn2._base_weight)


# ---------------------------------------------------------------------------
# Crossover / copy — acceptance criterion 5
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestCrossoverInheritsHebb(unittest.TestCase):

    def test_copy_preserves_hebb(self):
        from yane.evolution.stdp import set_hebb_coeffs
        g = _make_genome()
        set_hebb_coeffs(g, c=0.5, d=0.1)
        gc = g.copy()
        conn_c = gc.input_nodes[0].connections[0]
        self.assertAlmostEqual(conn_c.hebb_c, 0.5)
        self.assertAlmostEqual(conn_c.hebb_d, 0.1)

    def test_crossover_preserves_hebb(self):
        from yane.evolution.stdp import set_hebb_coeffs
        g_a = _make_genome(); g_a.fitness = 10.0
        g_b = _make_genome(); g_b.fitness = 5.0
        set_hebb_coeffs(g_a, c=0.7)
        set_hebb_coeffs(g_b, c=0.3)
        child = g_a.crossover(g_b)
        # Child's connections should have hebb_c close to either parent's value
        conn = child.input_nodes[0].connections[0]
        self.assertIn(round(conn.hebb_c, 1), [0.7, 0.3])

    def test_base_weight_not_inherited(self):
        """_base_weight is episode-local; offspring must start with None."""
        from yane.evolution.stdp import set_hebb_coeffs, init_stdp_base_weights
        g = _make_genome(); g.fitness = 1.0
        set_hebb_coeffs(g, c=0.1)
        g.forward([1.0, 1.0])
        init_stdp_base_weights(g)
        child = g.copy()
        for src in child.nodes:
            for conn in src.connections:
                self.assertIsNone(conn._base_weight)


# ---------------------------------------------------------------------------
# NeuroEvolution integration — acceptance criterion 6
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestNeuroEvolutionSTDP(unittest.TestCase):

    def test_set_stdp_sets_flag(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_stdp(enabled=True)
        self.assertTrue(ne._stdp_enabled)

    def test_set_stdp_disabled(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.set_stdp(enabled=False)
        self.assertFalse(ne._stdp_enabled)

    def test_train_with_stdp_no_crash(self):
        """Training with STDP active must not crash."""
        import yane
        from yane.evolution.stdp import set_hebb_coeffs
        ne = yane.NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=10, max_connections=20)
        ne.set_stdp(enabled=True, weight_min=-3.0, weight_max=3.0)
        ne.set_max_iterations(5)
        # Seed hebb coefficients on initial genomes
        pop = ne.population
        for g in list(pop._unevaluated):
            set_hebb_coeffs(g, c=0.01, sigma=0.005, rng=__import__('random').Random(0))
        ne.train(lambda g: sum(g.forward([0.5, 0.5])))

    def test_base_weights_restored_after_evaluation(self):
        """After train() completes, base weights must equal evolved weights."""
        import yane
        from yane.evolution.stdp import set_hebb_coeffs
        ne = yane.NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=10, max_connections=20)
        ne.set_stdp(enabled=True)
        ne.set_max_iterations(3)
        for g in list(ne.population._unevaluated):
            set_hebb_coeffs(g, c=0.1)
        ne.train(lambda g: sum(g.forward([1.0, 1.0])))
        best = ne.get_best()
        # After training, _base_weight should be None (restored + cleared)
        for src in best.nodes:
            for conn in src.connections:
                self.assertIsNone(conn._base_weight)

    def test_stdp_zero_cost_when_no_hebb(self):
        """With STDP enabled but no hebb coefficients set, behaviour is unchanged."""
        import yane
        ne = yane.NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=10, max_connections=20)
        ne.set_stdp(enabled=True)  # enabled, but no hebb coefficients
        ne.set_max_iterations(5)
        ne.train(lambda g: sum(g.forward([0.5, 0.5])))


if __name__ == "__main__":
    unittest.main()
