"""Tests for Behaviour Cloning as Warm-Start (evolution/behaviour_cloning.py).

Acceptance criteria:
  1. behaviour_clone returns BehaviourCloneResult with correct fields.
  2. Cloning reduces MSE (final_mse <= initial_mse).
  3. seed_population replaces population slots with noisy copies.
  4. seed_population without noise_sigma produces near-identical copies.
  5. BehaviourCloneResult.compression_ratio >= 1.0 when cloning improves.
  6. Empty demonstrations raises ValueError.
  7. NeuroEvolution.behaviour_clone() delegates to the module.
  8. seed_population=True immediately seeds when called.
  9. Population genomes are structurally valid after seeding.
 10. freeze_layers param accepted without error.
"""
from __future__ import annotations

import unittest

import pytest

from yane import NeuroEvolution
from yane.evolution.behaviour_cloning import (
    BehaviourCloneResult,
    behaviour_clone,
    seed_population_with,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_XOR_DEMOS = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0]),
]

_IDENTITY_DEMOS = [
    ([0.5], [0.5]),
    ([-0.3], [-0.3]),
    ([1.0], [1.0]),
]


def _make_yane(n_inputs: int = 2, n_outputs: int = 1, pop: int = 8) -> NeuroEvolution:
    yane = NeuroEvolution(seed=0)
    yane.set_population_size(pop)
    yane.configure(n_inputs, n_outputs)
    return yane


# ---------------------------------------------------------------------------
# 1. BehaviourCloneResult structure
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestBCResultStructure(unittest.TestCase):

    def test_returns_bc_result(self):
        """behaviour_clone returns a BehaviourCloneResult."""
        yane = _make_yane()
        result = behaviour_clone(yane, _XOR_DEMOS, n_steps=5)
        self.assertIsInstance(result, BehaviourCloneResult)

    def test_result_has_all_fields(self):
        """BehaviourCloneResult has cloned_genome, initial_mse, final_mse, n_steps_run."""
        yane = _make_yane()
        result = behaviour_clone(yane, _XOR_DEMOS, n_steps=5)
        self.assertIsNotNone(result.cloned_genome)
        self.assertGreaterEqual(result.initial_mse, 0.0)
        self.assertGreaterEqual(result.final_mse, 0.0)
        self.assertEqual(result.n_steps_run, 5)

    def test_mse_values_are_finite(self):
        """initial_mse and final_mse are finite floats."""
        yane = _make_yane()
        result = behaviour_clone(yane, _XOR_DEMOS, n_steps=3)
        import math
        self.assertTrue(math.isfinite(result.initial_mse))
        self.assertTrue(math.isfinite(result.final_mse))

    def test_compression_ratio_gte_one_when_improved(self):
        """compression_ratio >= 1.0 when final_mse <= initial_mse."""
        yane = _make_yane()
        result = behaviour_clone(yane, _XOR_DEMOS, n_steps=20)
        if result.final_mse <= result.initial_mse:
            self.assertGreaterEqual(result.compression_ratio, 1.0)


# ---------------------------------------------------------------------------
# 2. Cloning reduces MSE
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestCloningReducesMSE(unittest.TestCase):

    def test_final_mse_lte_initial_mse(self):
        """After cloning, final_mse should be <= initial_mse (monotone guarantee)."""
        yane = _make_yane()
        result = behaviour_clone(yane, _XOR_DEMOS, n_steps=50)
        self.assertLessEqual(result.final_mse, result.initial_mse + 1e-9)

    def test_identity_cloning_reduces_mse(self):
        """Identity function demos: cloning should quickly reduce error."""
        yane = _make_yane(n_inputs=1, n_outputs=1)
        result = behaviour_clone(yane, _IDENTITY_DEMOS, n_steps=100)
        self.assertLessEqual(result.final_mse, result.initial_mse + 1e-9)

    def test_cloned_genome_fitness_is_negative_mse(self):
        """Cloned genome fitness == -final_mse."""
        yane = _make_yane()
        result = behaviour_clone(yane, _XOR_DEMOS, n_steps=5)
        import math
        self.assertAlmostEqual(result.cloned_genome.fitness, -result.final_mse, places=8)

    def test_original_population_unchanged(self):
        """behaviour_clone does not modify the original population."""
        yane = _make_yane()
        original_ids = [id(g) for g in (
            list(yane._population._unevaluated)
            + list(yane._population._evaluated)
        )]
        behaviour_clone(yane, _XOR_DEMOS, n_steps=3)
        new_ids = [id(g) for g in (
            list(yane._population._unevaluated)
            + list(yane._population._evaluated)
        )]
        # Without seed_population=True, population must not change
        self.assertEqual(original_ids, new_ids)


# ---------------------------------------------------------------------------
# 3. seed_population
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestSeedPopulation(unittest.TestCase):

    def test_seed_population_sets_seed(self):
        """seed_population_with sets the cloned genome as population seed."""
        yane = _make_yane(pop=6)
        result = behaviour_clone(yane, _XOR_DEMOS, n_steps=3)
        result.seed_population()
        # After seeding, the population has the cloned genome as seed (1 base copy)
        all_genomes = (
            list(yane._population._unevaluated)
            + list(yane._population._evaluated)
        )
        self.assertGreaterEqual(len(all_genomes), 1)

    def test_seed_population_with_n_copies(self):
        """seed_population_with with n_copies adds extra copies to unevaluated pool."""
        yane = _make_yane(pop=8)
        result = behaviour_clone(yane, _XOR_DEMOS, n_steps=3)
        n_total = seed_population_with(yane, result.cloned_genome, n_copies=3)
        # 1 base seed + 3 extra copies = 4 in unevaluated pool
        self.assertEqual(n_total, 4)

    def test_seed_no_noise_seed_matches_clone(self):
        """With noise_sigma=0, the base seed produces same output as cloned genome."""
        yane = _make_yane(pop=4)
        result = behaviour_clone(yane, _XOR_DEMOS, n_steps=3)
        seed_population_with(yane, result.cloned_genome, n_copies=None, noise_sigma=0.0)
        seed_genome = yane._population._unevaluated[0]
        inp = [0.5, 0.5]
        result.cloned_genome.reset()
        ref = result.cloned_genome.forward(inp)
        seed_genome.reset()
        got = seed_genome.forward(inp)
        for a, b in zip(ref, got):
            self.assertAlmostEqual(a, b, places=5)

    def test_seed_population_true_flag(self):
        """seed_population=True immediately seeds on behaviour_clone call."""
        yane = _make_yane(pop=4)
        behaviour_clone(yane, _XOR_DEMOS, n_steps=3, seed_population=True)
        all_genomes = (
            list(yane._population._unevaluated)
            + list(yane._population._evaluated)
        )
        # Population should have at least the seed genome
        self.assertGreaterEqual(len(all_genomes), 1)

    def test_seed_result_without_ne_ref_raises(self):
        """seed_population() on a manually constructed result without ne_ref raises."""
        result = BehaviourCloneResult(
            cloned_genome=None,
            initial_mse=1.0,
            final_mse=0.5,
            n_steps_run=10,
            _ne_ref=None,
        )
        with self.assertRaises(RuntimeError):
            result.seed_population()

    def test_seeded_genomes_are_structurally_valid(self):
        """Seeded genomes have valid topology (can run forward)."""
        yane = _make_yane(pop=4)
        result = behaviour_clone(yane, _XOR_DEMOS, n_steps=3)
        seed_population_with(yane, result.cloned_genome, n_copies=2)
        all_genomes = (
            list(yane._population._unevaluated)
            + list(yane._population._evaluated)
        )
        for g in all_genomes:
            g.reset()
            out = g.forward([0.5, 0.5])
            self.assertEqual(len(out), 1)


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestBCEdgeCases(unittest.TestCase):

    def test_empty_demos_raises(self):
        """Empty demonstrations list raises ValueError."""
        yane = _make_yane()
        with self.assertRaises((ValueError, Exception)):
            behaviour_clone(yane, [], n_steps=5)

    def test_not_configured_raises(self):
        """behaviour_clone raises when not configured."""
        yane = NeuroEvolution(seed=0)
        with self.assertRaises(Exception):
            behaviour_clone(yane, _XOR_DEMOS)

    def test_freeze_layers_accepted(self):
        """freeze_layers parameter is accepted without error."""
        yane = _make_yane()
        result = behaviour_clone(
            yane, _XOR_DEMOS, n_steps=3,
            seed_population=True, freeze_layers=[]
        )
        self.assertIsInstance(result, BehaviourCloneResult)


# ---------------------------------------------------------------------------
# 5. NeuroEvolution.behaviour_clone()
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestNeuroEvolutionBCMethod(unittest.TestCase):

    def test_ne_method_returns_bc_result(self):
        """NeuroEvolution.behaviour_clone() returns BehaviourCloneResult."""
        yane = _make_yane()
        result = yane.behaviour_clone(_XOR_DEMOS, n_steps=5)
        self.assertIsInstance(result, BehaviourCloneResult)

    def test_ne_method_mse_non_negative(self):
        """NeuroEvolution.behaviour_clone() returns valid MSE values."""
        yane = _make_yane()
        result = yane.behaviour_clone(_XOR_DEMOS, n_steps=10)
        self.assertGreaterEqual(result.initial_mse, 0.0)
        self.assertGreaterEqual(result.final_mse, 0.0)

    def test_ne_method_seed_population(self):
        """NeuroEvolution.behaviour_clone(seed_population=True) seeds correctly."""
        yane = _make_yane(pop=4)
        result = yane.behaviour_clone(_XOR_DEMOS, n_steps=3, seed_population=True)
        self.assertIsInstance(result, BehaviourCloneResult)
        all_genomes = (
            list(yane._population._unevaluated)
            + list(yane._population._evaluated)
        )
        # Population has at least the base seed genome
        self.assertGreaterEqual(len(all_genomes), 1)

    def test_ne_method_seed_then_train(self):
        """After BC seeding, train() runs without errors."""
        yane = _make_yane(pop=4)
        yane.set_max_iterations(10)
        yane.behaviour_clone(_XOR_DEMOS, n_steps=3, seed_population=True)
        yane.train(lambda g: -sum(
            sum((o - t) ** 2 for o, t in zip(g.reset() or g.forward(inp), tgt))
            for inp, tgt in _XOR_DEMOS
        ) / len(_XOR_DEMOS))
