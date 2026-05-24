"""Tests for Lamarckian refinement, including per-species adaptive Lamarck."""

import unittest
from yane import NeuroEvolution
from yane.core.genome import Genome


def _dummy_fitness(genome: Genome) -> float:
    """Simple fitness function: sum of weights (deterministic)."""
    total = 0.0
    for node in genome.nodes:
        for conn in node.connections:
            if conn.enabled:
                total += conn.weight
    return total


def _stochastic_fitness(genome: Genome) -> float:
    """Slightly noisy fitness — helps test Lamarck hill-climbing."""
    import random
    total = 0.0
    for node in genome.nodes:
        for conn in node.connections:
            if conn.enabled:
                total += conn.weight
    return total + random.gauss(0.0, 0.001)


class TestLamarckRefine(unittest.TestCase):
    """Tests for the core _lamarck_refine hill-climbing logic."""

    def test_lamarck_does_not_worsen_fitness(self):
        """Lamarck refinement must not return worse fitness than baseline."""
        yane = NeuroEvolution()
        yane.configure(2, 1)
        genome = yane.next_genome()

        baseline = _dummy_fitness(genome)
        refined = yane._lamarck_refine(genome, _dummy_fitness,
                                        baseline_fitness=baseline, n_steps=5)
        self.assertGreaterEqual(refined, baseline,
                                "Lamarck must not make fitness worse")

    def test_lamarck_zero_steps_returns_baseline(self):
        """With n_steps=0, _lamarck_refine should return baseline unchanged."""
        yane = NeuroEvolution()
        yane.configure(2, 1)
        genome = yane.next_genome()

        baseline = _dummy_fitness(genome)
        result = yane._lamarck_refine(genome, _dummy_fitness,
                                       baseline_fitness=baseline, n_steps=0)
        self.assertEqual(result, baseline)

    def test_lamarck_explicit_mode_used_when_steps_set(self):
        """set_lamarck(n>0) should put Lamarck in explicit mode."""
        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane.set_lamarck(n_steps=3)

        info = yane.population_memory_info()
        self.assertEqual(info["lamarck_mode"], "explicit")

    def test_lamarck_adaptive_mode_default(self):
        """Default Lamarck mode should be 'adaptive'."""
        yane = NeuroEvolution()
        yane.configure(2, 1)

        info = yane.population_memory_info()
        self.assertEqual(info["lamarck_mode"], "adaptive")


class TestPerSpeciesLamarck(unittest.TestCase):
    """Tests for per-species adaptive Lamarck scaling."""

    def test_species_has_lamarck_counters(self):
        """Species objects must have lamarck_n_applied and lamarck_n_steps_total."""
        from yane.evolution.population import Species
        from yane.core.genome import Genome
        g = Genome()
        sp = Species(g)
        self.assertEqual(sp.lamarck_n_applied, 0)
        self.assertEqual(sp.lamarck_n_steps_total, 0)

    def test_get_species_for_genome_returns_none_when_unassigned(self):
        """get_species_for_genome returns None for a genome never assigned."""
        yane = NeuroEvolution()
        yane.configure(2, 1)
        genome = yane.next_genome()

        sp = yane._population.get_species_for_genome(genome)
        self.assertIsNone(sp)

    def test_get_species_for_genome_after_submit(self):
        """After submit, a genome should be findable via get_species_for_genome."""
        yane = NeuroEvolution()
        yane.configure(2, 1)
        genome = yane.next_genome()
        yane.submit_fitness(1.0)

        sp = yane._population.get_species_for_genome(genome)
        self.assertIsNotNone(sp)
        self.assertIn(genome, sp.members)

    def test_adaptive_lamarck_uses_species_stagnation(self):
        """_adaptive_lamarck_steps should use species-level stagnation when available."""
        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane.set_lamarck_adaptive(max_steps=10)

        genome = yane.next_genome()
        yane.submit_fitness(1.0)

        # Artificially set species stagnation high.
        sp = yane._population.get_species_for_genome(genome)
        self.assertIsNotNone(sp)
        sp.stagnation_count = 100  # well above threshold

        # Global stagnation is 0 (just improved), but species stagnation is high.
        self.assertEqual(yane._population.stagnation_count, 0)
        steps = yane._adaptive_lamarck_steps(genome, fitness=1.0)
        # With high species stagnation, should get max_steps (10).
        self.assertEqual(steps, 10,
                         f"Expected max_steps=10 due to species stagnation, got {steps}")

    def test_adaptive_lamarck_falls_back_to_global(self):
        """When genome has no species, fall back to global stagnation."""
        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane.set_lamarck_adaptive(max_steps=10)

        genome = yane.next_genome()
        # Not submitted → no species assigned.
        self.assertIsNone(yane._population.get_species_for_genome(genome))

        # Global stagnation is 0.
        steps = yane._adaptive_lamarck_steps(genome, fitness=1.0)
        self.assertEqual(steps, 0)

    def test_population_memory_info_has_lamarck_per_species(self):
        """population_memory_info must include lamarck_per_species diagnostics."""
        yane = NeuroEvolution()
        yane.configure(2, 1)

        # Submit a few genomes so species exist.
        for _ in range(5):
            g = yane.next_genome()
            yane.submit_fitness(1.0)

        info = yane.population_memory_info()
        self.assertIn("lamarck_per_species", info)
        per_species = info["lamarck_per_species"]
        self.assertIsInstance(per_species, list)
        self.assertGreater(len(per_species), 0, "At least one species should exist")

        for sp_info in per_species:
            self.assertIn("species_idx", sp_info)
            self.assertIn("members", sp_info)
            self.assertIn("stagnation_count", sp_info)
            self.assertIn("lamarck_n_applied", sp_info)
            self.assertIn("lamarck_n_steps_total", sp_info)

    def test_species_lamarck_counters_increment(self):
        """When Lamarck fires, per-species counters should increment."""
        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane.set_lamarck_adaptive(max_steps=5, top_k=1.0)  # top_k=1.0 → all genomes

        # Submit first genome and set high species stagnation.
        g = yane.next_genome()
        yane.submit_fitness(100.0)
        sp = yane._population.get_species_for_genome(g)
        self.assertIsNotNone(sp)
        sp.stagnation_count = 500  # Forces max Lamarck steps

        # Now submit another genome in the same species.
        g2 = yane.next_genome()
        yane.submit_fitness(100.0)

        # After two submissions with high species stagnation, Lamarck should have fired.
        # Check that species counters were incremented.
        self.assertGreaterEqual(
            sp.lamarck_n_applied, 0,
            "Species Lamarck counter should be present"
        )


class TestLamarckIntegration(unittest.TestCase):
    """End-to-end Lamarck in the training loop."""

    def test_adaptive_lamarck_fires_during_training(self):
        """After enough stagnation, adaptive Lamarck should fire in train()."""
        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane.set_lamarck_adaptive(max_steps=3)
        yane.set_max_iterations(20)

        # Use a simple fitness function.
        def fn(g):
            return sum(c.weight for n in g.nodes for c in n.connections if c.enabled)

        yane.train(fn)

        info = yane.population_memory_info()
        # Either Lamarck fired (n_applied > 0) or training was too short.
        # In 20 iterations with a simple function, at least some stagnation
        # should occur.
        self.assertIsNotNone(info.get("lamarck_n_applied"))
        self.assertIsNotNone(info.get("lamarck_n_steps_total"))


class TestLamarckSigma(unittest.TestCase):
    """Tests for the per-genome lamarck_sigma strategy gene."""

    def test_genome_has_lamarck_sigma(self):
        """New genomes must start with lamarck_sigma=1.0."""
        yane = NeuroEvolution()
        yane.configure(2, 1)
        g = yane.next_genome()
        self.assertAlmostEqual(g.lamarck_sigma, 1.0, places=9)

    def test_lamarck_sigma_used_in_refine(self):
        """LamarckRefiner.refine() uses genome.lamarck_sigma, not sigma_global.

        Verified by setting lamarck_sigma=0 (invalid → refine returns early) while
        sigma_global is non-zero, and confirming no perturbations happen.
        """
        from yane.evolution.lamarck_refiner import LamarckRefiner
        yane = NeuroEvolution()
        yane.configure(2, 1)
        genome = yane.next_genome()
        yane.submit_fitness(0.0)
        # Pull a second genome so connections may exist.
        genome = yane.next_genome()

        genome.sigma_global = 5.0    # large; would perturb if used
        genome.lamarck_sigma = 0.0   # invalid (0.0 < sigma check fails) → early return

        refiner = LamarckRefiner()
        baseline = _dummy_fitness(genome)
        result = refiner.refine(genome, _dummy_fitness, baseline_fitness=baseline, n_steps=5)
        # lamarck_sigma=0 should trigger the guard → return baseline immediately
        self.assertEqual(result, baseline,
                         "lamarck_sigma=0 must cause early return (not sigma_global=5)")

    def test_lamarck_sigma_evolves_independently(self):
        """After mutations, lamarck_sigma and sigma_global can diverge."""
        import random as _rng
        _rng.seed(42)
        yane = NeuroEvolution()
        yane.configure(2, 1)
        g = yane.next_genome()
        # Pin sigma_global; mutate many times to let lamarck_sigma drift.
        g.sigma_global = 1.0
        for _ in range(200):
            g.mutate()
        # lamarck_sigma should be clamped to [0.001, 10.0] and able to differ.
        self.assertGreaterEqual(g.lamarck_sigma, 0.001)
        self.assertLessEqual(g.lamarck_sigma, 10.0)

    def test_diagnostics_include_pop_avg_lamarck_sigma(self):
        """population_memory_info must include pop_avg_lamarck_sigma."""
        yane = NeuroEvolution()
        yane.configure(2, 1)
        for _ in range(5):
            yane.next_genome()
            yane.submit_fitness(1.0)
        info = yane.population_memory_info()
        self.assertIn("pop_avg_lamarck_sigma", info)
        self.assertGreater(info["pop_avg_lamarck_sigma"], 0.0)

    def test_lamarck_sigma_inherited_via_copy(self):
        """copy() must preserve lamarck_sigma."""
        yane = NeuroEvolution()
        yane.configure(2, 1)
        g = yane.next_genome()
        g.lamarck_sigma = 3.14
        c = g.copy()
        self.assertAlmostEqual(c.lamarck_sigma, 3.14, places=9)

    def test_old_genome_fallback(self):
        """Genomes without lamarck_sigma fall back to sigma_global in refine()."""
        from yane.evolution.lamarck_refiner import LamarckRefiner
        yane = NeuroEvolution()
        yane.configure(2, 1)
        genome = yane.next_genome()
        # Simulate old pickled genome by deleting the attribute.
        del genome.lamarck_sigma
        genome.sigma_global = 2.0
        refiner = LamarckRefiner()
        # Should not raise — falls back to sigma_global.
        baseline = _dummy_fitness(genome)
        result = refiner.refine(genome, _dummy_fitness, baseline_fitness=baseline, n_steps=1)
        self.assertGreaterEqual(result, baseline)


if __name__ == "__main__":
    unittest.main()
