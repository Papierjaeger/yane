"""Tests for Safety-Constrained Evolution / Safe NEAT (evolution/safety.py).

Acceptance criteria:
  1. Hard constraint violation → fitness = penalty, stops immediately.
  2. Soft constraint always worsens fitness (positive AND negative raw fitness).
  3. Barrier mode always worsens fitness (positive AND negative raw fitness).
  4. SafetySystem.is_safe() correctly identifies violating genomes.
  5. safe_fraction() returns correct fraction.
  6. protect_safe_fraction() returns the safe genome set.
  7. wrap_fitness_fn wraps correctly.
  8. NeuroEvolution.set_safety_constraints() stores and applies the system.
  9. NeuroEvolution._finalize_fitness applies safety constraints automatically.
 10. Constraint check exception treated as violation (hard) or non-critical (soft).
"""
from __future__ import annotations

import unittest

import pytest

from yane import NeuroEvolution
from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType
from yane.evolution.safety import SafetyConstraint, SafetySystem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_genome() -> Genome:
    """Minimal genome: 1 input → 1 output."""
    g = Genome()
    inp = Node(NodeType.INPUT, 0)
    inp.activation = ActivationType.LINEAR
    inp.input_index = 0
    g.nodes.append(inp)
    g.input_nodes.append(inp)

    out = Node(NodeType.OUTPUT, 1)
    out.activation = ActivationType.LINEAR
    out.bias = 0.0
    g.nodes.append(out)
    g.output_nodes.append(out)

    conn = Connection(out, innovation=1)
    conn.weight = 1.0
    conn.enabled = True
    inp.connections.append(conn)
    g._invalidate_topology()
    return g


def _make_yane() -> NeuroEvolution:
    yane = NeuroEvolution(seed=0)
    yane.set_population_size(5)
    yane.configure(2, 1)
    return yane


# ---------------------------------------------------------------------------
# 1. Hard constraint
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestHardConstraint(unittest.TestCase):

    def test_hard_violation_returns_penalty(self):
        """Hard constraint violated → fitness becomes exactly penalty."""
        g = _make_simple_genome()
        c = SafetyConstraint("always_violated", check=lambda g: False,
                              mode="hard", penalty=-999.0)
        sys = SafetySystem([c])
        result = sys.evaluate(g, raw_fitness=100.0)
        self.assertAlmostEqual(result, -999.0)

    def test_hard_satisfied_returns_raw(self):
        """Hard constraint satisfied → fitness unchanged."""
        g = _make_simple_genome()
        c = SafetyConstraint("always_safe", check=lambda g: True,
                              mode="hard", penalty=-999.0)
        sys = SafetySystem([c])
        result = sys.evaluate(g, raw_fitness=50.0)
        self.assertAlmostEqual(result, 50.0)

    def test_hard_violation_increments_counter(self):
        """n_hard_violations increments on each violation."""
        g = _make_simple_genome()
        c = SafetyConstraint("bad", check=lambda g: False, mode="hard")
        sys = SafetySystem([c])
        sys.evaluate(g, 1.0)
        sys.evaluate(g, 2.0)
        self.assertEqual(sys.n_hard_violations, 2)

    def test_hard_stops_further_processing(self):
        """After a hard violation, subsequent soft constraints are not applied."""
        g = _make_simple_genome()
        applied = []

        def soft_check(g):
            applied.append(True)
            return False

        hard = SafetyConstraint("hard", check=lambda g: False, mode="hard", penalty=-100.0)
        soft = SafetyConstraint("soft", check=soft_check, mode="soft", penalty=0.5)
        sys = SafetySystem([hard, soft])
        result = sys.evaluate(g, raw_fitness=50.0)
        self.assertAlmostEqual(result, -100.0)
        self.assertEqual(len(applied), 0)  # soft never reached

    def test_hard_exception_treated_as_violation(self):
        """If check raises, it is treated as a violation for hard constraints."""
        g = _make_simple_genome()
        def bad_check(g):
            raise RuntimeError("check failed")
        c = SafetyConstraint("bad", check=bad_check, mode="hard", penalty=-500.0)
        sys = SafetySystem([c])
        result = sys.evaluate(g, raw_fitness=10.0)
        self.assertAlmostEqual(result, -500.0)


# ---------------------------------------------------------------------------
# 2. Soft constraint — always worsens fitness
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestSoftConstraint(unittest.TestCase):

    def test_soft_reduces_positive_fitness(self):
        """50% soft penalty reduces positive fitness."""
        g = _make_simple_genome()
        c = SafetyConstraint("soft", check=lambda g: False, mode="soft", penalty=0.5)
        sys = SafetySystem([c])
        raw = 100.0
        result = sys.evaluate(g, raw_fitness=raw)
        self.assertLess(result, raw)

    def test_soft_worsens_negative_fitness(self):
        """Soft penalty must make negative fitness more negative (worse)."""
        g = _make_simple_genome()
        c = SafetyConstraint("soft", check=lambda g: False, mode="soft", penalty=0.5)
        sys = SafetySystem([c])
        raw = -100.0
        result = sys.evaluate(g, raw_fitness=raw)
        # More negative = worse for maximisation
        self.assertLess(result, raw)

    def test_soft_zero_penalty_no_change(self):
        """Penalty=0.0 leaves fitness unchanged."""
        g = _make_simple_genome()
        c = SafetyConstraint("soft", check=lambda g: False, mode="soft", penalty=0.0)
        sys = SafetySystem([c])
        result = sys.evaluate(g, raw_fitness=42.0)
        self.assertAlmostEqual(result, 42.0)

    def test_soft_satisfied_no_change(self):
        """Soft constraint satisfied → fitness unchanged."""
        g = _make_simple_genome()
        c = SafetyConstraint("soft", check=lambda g: True, mode="soft", penalty=0.5)
        sys = SafetySystem([c])
        result = sys.evaluate(g, raw_fitness=80.0)
        self.assertAlmostEqual(result, 80.0)

    def test_soft_increments_counter(self):
        """n_soft_violations increments correctly."""
        g = _make_simple_genome()
        c = SafetyConstraint("soft", check=lambda g: False, mode="soft", penalty=0.3)
        sys = SafetySystem([c])
        sys.evaluate(g, 1.0)
        sys.evaluate(g, 2.0)
        self.assertEqual(sys.n_soft_violations, 2)


# ---------------------------------------------------------------------------
# 3. Barrier constraint
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestBarrierConstraint(unittest.TestCase):

    def test_barrier_worsens_positive_fitness(self):
        """Barrier constraint reduces positive fitness."""
        g = _make_simple_genome()
        c = SafetyConstraint("barrier", check=lambda g: False, mode="barrier", penalty=1.0)
        sys = SafetySystem([c])
        raw = 50.0
        result = sys.evaluate(g, raw_fitness=raw)
        self.assertLess(result, raw)

    def test_barrier_worsens_negative_fitness(self):
        """Barrier constraint makes negative fitness more negative."""
        g = _make_simple_genome()
        c = SafetyConstraint("barrier", check=lambda g: False, mode="barrier", penalty=2.0)
        sys = SafetySystem([c])
        raw = -50.0
        result = sys.evaluate(g, raw_fitness=raw)
        self.assertLess(result, raw)

    def test_barrier_satisfied_no_change(self):
        """Barrier constraint satisfied → fitness unchanged."""
        g = _make_simple_genome()
        c = SafetyConstraint("barrier", check=lambda g: True, mode="barrier", penalty=5.0)
        sys = SafetySystem([c])
        result = sys.evaluate(g, raw_fitness=30.0)
        self.assertAlmostEqual(result, 30.0)


# ---------------------------------------------------------------------------
# 4. is_safe / safe_fraction / protect_safe_fraction
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestSafeFraction(unittest.TestCase):

    def test_is_safe_all_hard_satisfied(self):
        """is_safe returns True when all hard constraints are met."""
        g = _make_simple_genome()
        c = SafetyConstraint("safe", check=lambda g: True, mode="hard")
        sys = SafetySystem([c])
        self.assertTrue(sys.is_safe(g))

    def test_is_safe_hard_violated(self):
        """is_safe returns False when any hard constraint is violated."""
        g = _make_simple_genome()
        c = SafetyConstraint("violated", check=lambda g: False, mode="hard")
        sys = SafetySystem([c])
        self.assertFalse(sys.is_safe(g))

    def test_safe_fraction_empty_population(self):
        """safe_fraction returns 1.0 for empty population."""
        sys = SafetySystem([SafetyConstraint("c", check=lambda g: False, mode="hard")])
        self.assertAlmostEqual(sys.safe_fraction([]), 1.0)

    def test_safe_fraction_mixed(self):
        """safe_fraction counts correctly with mixed safe/unsafe genomes."""
        always_safe = SafetyConstraint("ok", check=lambda g: getattr(g, "_is_safe", True), mode="hard")
        sys = SafetySystem([always_safe])

        g1 = _make_simple_genome(); g1._is_safe = True
        g2 = _make_simple_genome(); g2._is_safe = False
        g3 = _make_simple_genome(); g3._is_safe = True

        frac = sys.safe_fraction([g1, g2, g3])
        self.assertAlmostEqual(frac, 2 / 3, places=6)

    def test_protect_safe_fraction_returns_safe_set(self):
        """protect_safe_fraction returns only safe genomes."""
        always_safe = SafetyConstraint("ok", check=lambda g: getattr(g, "_is_safe", True), mode="hard")
        sys = SafetySystem([always_safe], min_safe_frac=0.2)

        g1 = _make_simple_genome(); g1._is_safe = True
        g2 = _make_simple_genome(); g2._is_safe = False
        safe = sys.protect_safe_fraction([g1, g2])
        self.assertIn(g1, safe)
        self.assertNotIn(g2, safe)

    def test_protect_returns_empty_without_constraints(self):
        """protect_safe_fraction returns [] when no constraints are configured."""
        sys = SafetySystem([])
        g = _make_simple_genome()
        self.assertEqual(sys.protect_safe_fraction([g]), [])


# ---------------------------------------------------------------------------
# 5. wrap_fitness_fn
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestWrapFitnessFn(unittest.TestCase):

    def test_wrap_applies_hard_constraint(self):
        """wrap_fitness_fn wraps correctly and applies hard constraint."""
        g = _make_simple_genome()
        c = SafetyConstraint("bad", check=lambda g: False, mode="hard", penalty=-77.0)
        sys = SafetySystem([c])
        base_fn = lambda g: 100.0
        safe_fn = sys.wrap_fitness_fn(base_fn)
        result = safe_fn(g)
        self.assertAlmostEqual(result, -77.0)

    def test_wrap_passes_through_safe_fitness(self):
        """wrap_fitness_fn passes through fitness when constraints are met."""
        g = _make_simple_genome()
        c = SafetyConstraint("ok", check=lambda g: True, mode="hard")
        sys = SafetySystem([c])
        safe_fn = sys.wrap_fitness_fn(lambda g: 55.0)
        self.assertAlmostEqual(safe_fn(g), 55.0)


# ---------------------------------------------------------------------------
# 6. NeuroEvolution.set_safety_constraints integration
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestNeuroEvolutionSafety(unittest.TestCase):

    def test_set_safety_constraints_returns_system(self):
        """set_safety_constraints returns the SafetySystem."""
        yane = _make_yane()
        c = SafetyConstraint("c", check=lambda g: True, mode="hard")
        result = yane.set_safety_constraints([c])
        self.assertIsInstance(result, SafetySystem)

    def test_set_none_disables(self):
        """set_safety_constraints(None) disables safety."""
        yane = _make_yane()
        c = SafetyConstraint("c", check=lambda g: False, mode="hard", penalty=-99.0)
        yane.set_safety_constraints([c])
        yane.set_safety_constraints(None)
        self.assertIsNone(yane._safety_system)

    def test_finalize_fitness_applies_hard_constraint(self):
        """_finalize_fitness applies hard constraint when safety is configured."""
        yane = _make_yane()
        g = yane._population._unevaluated[0]
        c = SafetyConstraint("bad", check=lambda g: False, mode="hard", penalty=-42.0)
        yane.set_safety_constraints([c])
        result = yane._finalize_fitness(100.0, None, g)
        self.assertAlmostEqual(result, -42.0)

    def test_finalize_fitness_no_system_unchanged(self):
        """_finalize_fitness leaves fitness unchanged when no safety system set."""
        yane = _make_yane()
        g = yane._population._unevaluated[0]
        yane._safety_system = None
        result = yane._finalize_fitness(77.0, None, g)
        self.assertAlmostEqual(result, 77.0)

    def test_finalize_fitness_none_genome_no_crash(self):
        """_finalize_fitness with genome=None skips safety (no crash)."""
        yane = _make_yane()
        c = SafetyConstraint("bad", check=lambda g: False, mode="hard", penalty=-1.0)
        yane.set_safety_constraints([c])
        result = yane._finalize_fitness(50.0, None, None)
        # genome=None → safety skipped → raw fitness unchanged
        self.assertAlmostEqual(result, 50.0)

    def test_safety_in_train_loop(self):
        """Safety constraint reduces fitness during training."""
        yane = _make_yane()
        violations = []

        def check_fn(g):
            violations.append(1)
            return False  # always violated

        c = SafetyConstraint("always", check=check_fn, mode="soft", penalty=0.3)
        yane.set_safety_constraints([c])
        yane.set_max_iterations(20)
        yane.train(lambda g: 10.0)
        # Safety check should have been called during training
        self.assertGreater(len(violations), 0)
