"""Smoke tests for ablation toggles: novelty, speciation, crossover, diversity injection."""
import unittest

import pytest

from yane import NeuroEvolution


def _make_yane(pop_size=20):
    yane = NeuroEvolution()
    yane.configure(2, 1, max_nodes=10, max_connections=20)
    yane.set_population_size(pop_size)
    return yane


def _run(yane, n=60):
    for i in range(n):
        g = yane.next_genome()
        yane.submit_fitness(float(i % 5))


@pytest.mark.ci
class TestAblationNovelty(unittest.TestCase):

    def test_novelty_weight_zero_when_disabled(self):
        yane = _make_yane()
        yane.set_novelty_search(False)
        _run(yane, 30)
        self.assertEqual(yane._population.novelty_weight, 0.0)

    def test_novelty_enabled_returns_nonzero(self):
        yane = _make_yane()
        yane.set_novelty_search(True)
        _run(yane, 30)
        self.assertGreater(yane._population.novelty_weight, 0.0)

    def test_diagnostics_flag_novelty(self):
        yane = _make_yane()
        yane.set_novelty_search(False)
        _run(yane, 10)
        info = yane.population_memory_info()
        self.assertFalse(info["ablation_novelty_enabled"])

    def test_training_completes_without_novelty(self):
        yane = _make_yane()
        yane.set_novelty_search(False)
        _run(yane, 40)
        self.assertIsNotNone(yane.get_best())


@pytest.mark.ci
class TestAblationSpeciation(unittest.TestCase):

    def test_always_one_species_when_disabled(self):
        yane = _make_yane()
        yane.set_speciation(False)
        _run(yane, 60)
        self.assertEqual(yane._population.species_count, 1)

    def test_diagnostics_flag_speciation(self):
        yane = _make_yane()
        yane.set_speciation(False)
        _run(yane, 10)
        info = yane.population_memory_info()
        self.assertFalse(info["ablation_speciation_enabled"])

    def test_training_completes_without_speciation(self):
        yane = _make_yane()
        yane.set_speciation(False)
        _run(yane, 40)
        self.assertIsNotNone(yane.get_best())


@pytest.mark.ci
class TestAblationCrossover(unittest.TestCase):

    def test_no_crossover_offspring_when_disabled(self):
        yane = _make_yane()
        yane.set_crossover(False)
        _run(yane, 60)
        self.assertEqual(yane._population._n_crossover, 0)

    def test_mutation_only_offspring_nonzero_when_disabled(self):
        yane = _make_yane()
        yane.set_crossover(False)
        _run(yane, 60)
        self.assertGreater(yane._population._n_mutation_only, 0)

    def test_diagnostics_flag_crossover(self):
        yane = _make_yane()
        yane.set_crossover(False)
        _run(yane, 10)
        info = yane.population_memory_info()
        self.assertFalse(info["ablation_crossover_enabled"])

    def test_training_completes_without_crossover(self):
        yane = _make_yane()
        yane.set_crossover(False)
        _run(yane, 40)
        self.assertIsNotNone(yane.get_best())


@pytest.mark.ci
class TestAblationDiversityInjection(unittest.TestCase):

    def test_no_injections_when_disabled(self):
        yane = _make_yane(pop_size=10)
        yane.set_diversity_injection(False)
        _run(yane, 60)
        self.assertEqual(yane._population._n_diversity_injection, 0)

    def test_diagnostics_flag_diversity_injection(self):
        yane = _make_yane()
        yane.set_diversity_injection(False)
        _run(yane, 10)
        info = yane.population_memory_info()
        self.assertFalse(info["ablation_diversity_injection_enabled"])

    def test_training_completes_without_injection(self):
        yane = _make_yane()
        yane.set_diversity_injection(False)
        _run(yane, 40)
        self.assertIsNotNone(yane.get_best())
