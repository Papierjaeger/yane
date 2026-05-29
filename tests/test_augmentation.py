"""Tests for evolutionary data augmentation (evolution/augmentation.py)."""
from __future__ import annotations

import random
import statistics
import unittest


# ---------------------------------------------------------------------------
# AugmentationGene — individual transforms
# ---------------------------------------------------------------------------

class TestAugmentationGene(unittest.TestCase):
    def _rng(self, seed=0):
        return random.Random(seed)

    def test_gaussian_noise_changes_inputs(self):
        from yane.evolution.augmentation import AugmentationGene
        gene = AugmentationGene("gaussian_noise", probability=1.0, magnitude=0.5)
        inputs = [0.0, 0.5, 1.0]
        rng = self._rng()
        output = gene.apply(inputs, rng)
        # Noise should change at least one value
        self.assertFalse(output == inputs, "gaussian_noise must change inputs")

    def test_gaussian_noise_zero_prob_leaves_unchanged(self):
        from yane.evolution.augmentation import AugmentationGene
        gene = AugmentationGene("gaussian_noise", probability=0.0, magnitude=1.0)
        inputs = [1.0, 2.0, 3.0]
        output = gene.apply(inputs, self._rng())
        self.assertEqual(output, inputs)

    def test_dropout_noise_zeros_some_elements(self):
        from yane.evolution.augmentation import AugmentationGene
        gene = AugmentationGene("dropout_noise", probability=1.0, magnitude=1.0)
        inputs = [1.0] * 20
        rng = self._rng(42)
        output = gene.apply(inputs, rng)
        self.assertIn(0.0, output, "dropout_noise should zero at least one element")

    def test_scaling_changes_magnitude(self):
        from yane.evolution.augmentation import AugmentationGene
        gene = AugmentationGene("scaling", probability=1.0, magnitude=0.5)
        inputs = [1.0, 2.0, 3.0]
        output = gene.apply(inputs, self._rng(1))
        self.assertFalse(output == inputs, "scaling must change values")

    def test_translation_shifts_all_uniformly(self):
        from yane.evolution.augmentation import AugmentationGene
        gene = AugmentationGene("translation", probability=1.0, magnitude=0.5)
        inputs = [1.0, 2.0, 3.0]
        output = gene.apply(inputs, self._rng(2))
        diffs = [o - i for o, i in zip(output, inputs)]
        # All shifts should be the same (uniform offset)
        self.assertAlmostEqual(max(diffs) - min(diffs), 0.0, places=10)

    def test_cutout_zeros_contiguous_block(self):
        from yane.evolution.augmentation import AugmentationGene
        gene = AugmentationGene("cutout", probability=1.0, magnitude=0.5)
        inputs = [1.0] * 10
        output = gene.apply(inputs, self._rng(3))
        zeros = [i for i, v in enumerate(output) if v == 0.0]
        self.assertGreater(len(zeros), 0, "cutout should zero at least one element")
        # Zeros must be contiguous
        if len(zeros) > 1:
            for a, b in zip(zeros, zeros[1:]):
                self.assertEqual(b - a, 1, "zeroed elements must be contiguous")

    def test_apply_preserves_length(self):
        from yane.evolution.augmentation import AugmentationGene, AUGMENTATION_TYPES
        rng = self._rng()
        for aug_type in AUGMENTATION_TYPES:
            gene = AugmentationGene(aug_type, 1.0, 0.3)
            inputs = [0.5] * 8
            output = gene.apply(inputs, rng)
            self.assertEqual(len(output), len(inputs), f"length changed for {aug_type}")

    def test_mutation_changes_parameters(self):
        from yane.evolution.augmentation import AugmentationGene
        rng = self._rng(42)
        gene = AugmentationGene("gaussian_noise", probability=0.5, magnitude=0.2)
        mutated = gene.mutate(rng, sigma=0.3)
        # Same type, different parameters (with high probability)
        self.assertEqual(mutated.aug_type, gene.aug_type)
        changed = (
            abs(mutated.probability - gene.probability) > 1e-9
            or abs(mutated.magnitude  - gene.magnitude)  > 1e-9
        )
        self.assertTrue(changed, "Mutation should change at least one parameter")

    def test_mutation_stays_in_range(self):
        from yane.evolution.augmentation import AugmentationGene
        rng = self._rng(0)
        gene = AugmentationGene("scaling", probability=0.9, magnitude=0.9)
        for _ in range(50):
            mutated = gene.mutate(rng, sigma=1.0)  # large sigma to test clamping
            self.assertGreaterEqual(mutated.probability, 0.0)
            self.assertLessEqual(mutated.probability,    1.0)
            self.assertGreaterEqual(mutated.magnitude,   0.0)
            self.assertLessEqual(mutated.magnitude,      1.0)

    def test_serialization_roundtrip(self):
        from yane.evolution.augmentation import AugmentationGene
        gene = AugmentationGene("dropout_noise", 0.7, 0.3)
        d = gene.to_dict()
        gene2 = AugmentationGene.from_dict(d)
        self.assertEqual(gene2.aug_type,    gene.aug_type)
        self.assertAlmostEqual(gene2.probability, gene.probability)
        self.assertAlmostEqual(gene2.magnitude,   gene.magnitude)


# ---------------------------------------------------------------------------
# AugmentationPipeline
# ---------------------------------------------------------------------------

class TestAugmentationPipeline(unittest.TestCase):
    def _rng(self, seed=0):
        return random.Random(seed)

    def _pipeline(self, types=None, p=0.9, mag=0.3):
        from yane.evolution.augmentation import AugmentationGene, AugmentationPipeline
        types = types or ["gaussian_noise", "scaling"]
        return AugmentationPipeline([
            AugmentationGene(t, p, mag) for t in types
        ])

    def test_apply_changes_inputs(self):
        from yane.evolution.augmentation import AugmentationGene, AugmentationPipeline
        pipeline = AugmentationPipeline([
            AugmentationGene("gaussian_noise", 1.0, 0.5),
        ])
        inputs = [0.5, 0.5, 0.5]
        output = pipeline.apply(inputs, self._rng())
        self.assertFalse(output == inputs)

    def test_apply_preserves_length(self):
        pipeline = self._pipeline()
        inputs = [0.1, 0.2, 0.3, 0.4, 0.5]
        output = pipeline.apply(inputs, self._rng())
        self.assertEqual(len(output), len(inputs))

    def test_empty_pipeline_passthrough(self):
        from yane.evolution.augmentation import AugmentationPipeline
        pipeline = AugmentationPipeline([])
        inputs = [1.0, 2.0, 3.0]
        output = pipeline.apply(inputs, self._rng())
        self.assertEqual(output, inputs)

    def test_crossover_produces_valid_pipeline(self):
        from yane.evolution.augmentation import AugmentationGene, AugmentationPipeline
        rng = self._rng(42)
        p1 = AugmentationPipeline([
            AugmentationGene("gaussian_noise", 0.8, 0.2),
            AugmentationGene("scaling",        0.6, 0.3),
        ])
        p2 = AugmentationPipeline([
            AugmentationGene("dropout_noise",  0.5, 0.4),
            AugmentationGene("translation",    0.7, 0.1),
            AugmentationGene("cutout",         0.4, 0.5),
        ])
        child = p1.crossover(p2, rng)
        # Length should be max(2, 3)=3
        self.assertEqual(len(child.genes), 3)
        # All genes come from one of the parents
        parent_types = {g.aug_type for g in p1.genes} | {g.aug_type for g in p2.genes}
        for g in child.genes:
            self.assertIn(g.aug_type, parent_types)

    def test_crossover_equal_length_parents(self):
        from yane.evolution.augmentation import AugmentationGene, AugmentationPipeline
        rng = self._rng(1)
        p1 = AugmentationPipeline([AugmentationGene("gaussian_noise", 0.5, 0.2)])
        p2 = AugmentationPipeline([AugmentationGene("scaling", 0.8, 0.4)])
        child = p1.crossover(p2, rng)
        self.assertEqual(len(child.genes), 1)

    def test_mutation_changes_parameters(self):
        from yane.evolution.augmentation import AugmentationGene, AugmentationPipeline
        rng = self._rng(99)
        orig = AugmentationPipeline([AugmentationGene("scaling", 0.5, 0.3)])
        mutated = orig.mutate(rng, sigma=0.5)
        g_orig    = orig.genes[0]
        g_mutated = mutated.genes[0]
        changed = (
            abs(g_mutated.probability - g_orig.probability) > 1e-9
            or abs(g_mutated.magnitude  - g_orig.magnitude)  > 1e-9
        )
        self.assertTrue(changed)

    def test_serialization_roundtrip(self):
        from yane.evolution.augmentation import AugmentationGene, AugmentationPipeline
        genes = [
            AugmentationGene("gaussian_noise", 0.7, 0.2),
            AugmentationGene("cutout",         0.4, 0.5),
        ]
        p = AugmentationPipeline(genes)
        d = p.to_dict()
        p2 = AugmentationPipeline.from_dict(d)
        self.assertEqual(len(p2.genes), 2)
        self.assertEqual(p2.genes[0].aug_type, "gaussian_noise")
        self.assertEqual(p2.genes[1].aug_type, "cutout")

    def test_ucb1_favours_unselected(self):
        from yane.evolution.augmentation import AugmentationGene, AugmentationPipeline
        p = AugmentationPipeline([AugmentationGene("scaling", 0.5, 0.3)])
        # Unselected pipeline has infinite score
        self.assertEqual(p.ucb1_score(10), float("inf"))
        # After one selection
        p._n_selections = 1
        p._reward_sum = 0.5
        self.assertLess(p.ucb1_score(10), float("inf"))


# ---------------------------------------------------------------------------
# AugmentationPool
# ---------------------------------------------------------------------------

class TestAugmentationPool(unittest.TestCase):
    def _pool(self, size=4, seed=0):
        from yane.evolution.augmentation import AugmentationPool
        return AugmentationPool(
            augmentation_space=["gaussian_noise", "dropout_noise"],
            population_size=size,
            pipeline_length=2,
            evolution_interval=5,
            seed=seed,
        )

    def test_pool_initialises_with_correct_size(self):
        pool = self._pool(size=6)
        self.assertEqual(len(pool._pipelines), 6)

    def test_select_returns_pipeline(self):
        from yane.evolution.augmentation import AugmentationPipeline
        pool = self._pool()
        p = pool.select()
        self.assertIsInstance(p, AugmentationPipeline)

    def test_select_increments_total_selections(self):
        pool = self._pool()
        pool.select()
        pool.select()
        self.assertEqual(pool._total_selections, 2)

    def test_update_reward_accumulates(self):
        pool = self._pool()
        pool.select()
        pool.update_reward(1.0)
        pool.update_reward(0.5)
        self.assertAlmostEqual(pool._active._reward_sum, 1.5)

    def test_evolve_preserves_pool_size(self):
        pool = self._pool(size=8)
        for _ in range(8):
            pool.select()
            pool.update_reward(random.random())
        pool.evolve()
        self.assertEqual(len(pool._pipelines), 8)

    def test_best_pipeline_returns_highest_mean_reward(self):
        pool = self._pool(size=4)
        # Give the first pipeline a very high reward
        first = pool._pipelines[0]
        first._n_selections = 3
        first._reward_sum = 9.0
        best = pool.best_pipeline()
        self.assertEqual(best, first)

    def test_invalid_augmentation_space_raises(self):
        from yane.evolution.augmentation import AugmentationPool
        with self.assertRaises(ValueError):
            AugmentationPool(augmentation_space=["nonexistent_type"])

    def test_pipeline_parameters_change_after_evolve(self):
        """Acceptance criterion: parameters visibly change after evolution."""
        pool = self._pool(size=6, seed=42)
        initial_params = [
            (g.probability, g.magnitude)
            for p in pool._pipelines
            for g in p.genes
        ]
        # Reward all pipelines then evolve
        for _ in range(6):
            pool.select()
            pool.update_reward(0.1)
        pool.evolve()
        evolved_params = [
            (g.probability, g.magnitude)
            for p in pool._pipelines
            for g in p.genes
        ]
        # At least some parameters should have changed
        changed = any(a != b for a, b in zip(initial_params, evolved_params))
        self.assertTrue(changed, "Parameters must change after evolution")

    def test_random_pipeline_has_valid_types(self):
        from yane.evolution.augmentation import AugmentationPipeline, AUGMENTATION_TYPES
        rng = random.Random(7)
        p = AugmentationPipeline.random(AUGMENTATION_TYPES, 3, rng)
        for g in p.genes:
            self.assertIn(g.aug_type, AUGMENTATION_TYPES)
        self.assertEqual(len(p.genes), 3)


# ---------------------------------------------------------------------------
# Integration via NeuroEvolution
# ---------------------------------------------------------------------------

class TestNeuroEvolutionIntegration(unittest.TestCase):
    def _make_yane(self, seed=0):
        from yane import NeuroEvolution
        yane = NeuroEvolution(seed=seed)
        yane.configure(2, 1, n_initial_hidden=1)
        yane.set_population_size(5)
        yane.set_max_iterations(10)
        return yane

    def test_set_evolutionary_augmentation_creates_pool(self):
        from yane.evolution.augmentation import AugmentationPool
        yane = self._make_yane()
        yane.set_evolutionary_augmentation(
            augmentation_space=["gaussian_noise", "scaling"],
            population_augmentations=4,
        )
        self.assertIsNotNone(yane._aug_pool)
        self.assertIsInstance(yane._aug_pool, AugmentationPool)

    def test_disabled_augmentation_clears_pool(self):
        yane = self._make_yane()
        yane.set_evolutionary_augmentation(enabled=True)
        yane.set_evolutionary_augmentation(enabled=False)
        self.assertIsNone(yane._aug_pool)

    def test_training_runs_without_error(self):
        yane = self._make_yane()
        yane.set_evolutionary_augmentation(
            augmentation_space=["gaussian_noise", "dropout_noise"],
            population_augmentations=3,
            pipeline_length=2,
            evolution_interval=3,
        )
        iters = yane.train(lambda g: float(sum(g.forward([0.5, 0.5]))))
        self.assertGreater(iters, 0)

    def test_augmentation_actually_changes_forward_output(self):
        """With 100% probability noise, augmented forward differs from plain."""
        from yane import NeuroEvolution
        from yane.evolution.augmentation import AugmentationPool, AugmentationGene, AugmentationPipeline
        import random as _random

        yane = NeuroEvolution(seed=42)
        yane.configure(2, 1, n_initial_hidden=1)
        genome = yane.next_genome()

        plain_outputs = []
        augmented_outputs = []

        # Plain evaluation
        for _ in range(5):
            genome.reset()
            plain_outputs.append(genome.forward([0.5, 0.5])[0])

        # Set up augmentation pool with guaranteed-high noise
        pool = AugmentationPool(
            augmentation_space=["gaussian_noise"],
            population_size=2,
            pipeline_length=1,
            seed=1,
        )
        # Force a high-magnitude pipeline
        pool._pipelines[0] = AugmentationPipeline([
            AugmentationGene("gaussian_noise", probability=1.0, magnitude=2.0)
        ])
        pool.select()
        yane._aug_pool = pool

        # Evaluate with augmentation active (use _run_evaluations)
        seen = set()
        for _ in range(5):
            result = yane._run_evaluations(genome, lambda g: float(sum(g.forward([0.5, 0.5]))))
            seen.add(round(result.fitness, 6))

        # With high noise, we should see multiple different fitness values
        self.assertGreater(len(seen), 1, "Augmented evaluation should produce varied fitness")

    def test_get_augmentation_diagnostics_when_disabled(self):
        yane = self._make_yane()
        d = yane.get_augmentation_diagnostics()
        self.assertFalse(d["enabled"])

    def test_get_augmentation_diagnostics_when_enabled(self):
        yane = self._make_yane()
        yane.set_evolutionary_augmentation(population_augmentations=4)
        yane.train(lambda g: float(sum(g.forward([0.5, 0.5]))))
        d = yane.get_augmentation_diagnostics()
        self.assertTrue(d["enabled"])
        self.assertIn("population_size", d)
        self.assertIn("best_genes", d)


if __name__ == "__main__":
    unittest.main()
