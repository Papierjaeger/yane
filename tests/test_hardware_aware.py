"""Tests for hardware-aware NEAT (evolution/hardware_aware.py)."""
from __future__ import annotations

import math
import unittest


def _make_genome(n_inputs: int = 2, n_outputs: int = 1, n_hidden: int = 1):
    """Return a configured genome via NeuroEvolution (n_hidden>=1 so connections exist)."""
    from yane import NeuroEvolution
    yane = NeuroEvolution(seed=0)
    yane.configure(n_inputs, n_outputs, n_initial_hidden=max(1, n_hidden),
                   max_nodes=50, max_connections=200)
    return yane.next_genome()


# ---------------------------------------------------------------------------
# FLOP counting
# ---------------------------------------------------------------------------

class TestFlopCounting(unittest.TestCase):
    def test_minimal_genome_has_positive_flops(self):
        from yane.evolution.hardware_aware import compute_hardware_metrics
        genome = _make_genome(2, 1)
        metrics = compute_hardware_metrics(genome)
        self.assertGreater(metrics.flops, 0)

    def test_flops_increase_with_connections(self):
        """A genome with more connections should cost more FLOPs."""
        from yane.evolution.hardware_aware import compute_hardware_metrics
        g_small = _make_genome(2, 1, n_hidden=0)
        g_large = _make_genome(2, 1, n_hidden=4)
        m_small = compute_hardware_metrics(g_small)
        m_large = compute_hardware_metrics(g_large)
        # Hidden nodes add connections → more FLOPs
        self.assertGreaterEqual(m_large.flops, m_small.flops)

    def test_flops_at_least_2_per_connection(self):
        """Each enabled connection contributes ≥ 2 FLOPs (mul + add)."""
        from yane.evolution.hardware_aware import compute_hardware_metrics
        genome = _make_genome(2, 1)
        metrics = compute_hardware_metrics(genome)
        n_connections = genome.connection_count
        self.assertGreaterEqual(metrics.flops, n_connections * 2)

    def test_flops_formula_manual(self):
        """Verify the formula on a known topology."""
        from yane.evolution.hardware_aware import compute_hardware_metrics, _ACTIVATION_FLOPS
        from yane.core.genome import Genome
        # Build genome via yane and inspect
        genome = _make_genome(2, 1, n_hidden=1)
        metrics = compute_hardware_metrics(genome)
        # Manual: conn_flops = n_connections * 2
        n_conn = genome.connection_count
        conn_flops = n_conn * 2
        # node_flops = sum over non-input nodes of (1 + act_flops)
        node_flops = 0
        for node in genome.nodes:
            if node.input_index is not None:
                continue
            act_name = node.activation.value
            node_flops += 1 + _ACTIVATION_FLOPS.get(act_name, 3)
        expected = conn_flops + node_flops
        self.assertEqual(metrics.flops, expected)


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------

class TestMemoryEstimation(unittest.TestCase):
    def test_memory_positive(self):
        from yane.evolution.hardware_aware import compute_hardware_metrics
        genome = _make_genome(2, 1)
        metrics = compute_hardware_metrics(genome)
        self.assertGreater(metrics.memory_bytes, 0)

    def test_memory_formula(self):
        from yane.evolution.hardware_aware import compute_hardware_metrics, HardwareConstraints
        genome = _make_genome(2, 1)
        c = HardwareConstraints(bytes_per_node=8, bytes_per_connection=8)
        metrics = compute_hardware_metrics(genome, c)
        n_nodes = len(genome.nodes)
        n_conns = genome.connection_count
        expected = n_nodes * 8 + n_conns * 8
        self.assertEqual(metrics.memory_bytes, expected)

    def test_memory_increases_with_nodes(self):
        from yane.evolution.hardware_aware import compute_hardware_metrics
        g_small = _make_genome(2, 1, n_hidden=0)
        g_large = _make_genome(2, 1, n_hidden=5)
        m_small = compute_hardware_metrics(g_small)
        m_large = compute_hardware_metrics(g_large)
        self.assertGreater(m_large.memory_bytes, m_small.memory_bytes)

    def test_custom_byte_sizes(self):
        from yane.evolution.hardware_aware import compute_hardware_metrics, HardwareConstraints
        genome = _make_genome(2, 1)
        c = HardwareConstraints(bytes_per_node=16, bytes_per_connection=12)
        metrics = compute_hardware_metrics(genome, c)
        n_nodes = len(genome.nodes)
        n_conns = genome.connection_count
        self.assertEqual(metrics.memory_bytes, n_nodes * 16 + n_conns * 12)


# ---------------------------------------------------------------------------
# Platform profiles
# ---------------------------------------------------------------------------

class TestPlatformProfiles(unittest.TestCase):
    def test_all_profiles_load(self):
        from yane.evolution.hardware_aware import PLATFORM_PROFILES, compute_hardware_metrics
        genome = _make_genome(2, 1)
        for name in PLATFORM_PROFILES:
            from yane.evolution.hardware_aware import HardwareConstraints
            c = HardwareConstraints(target_platform=name)
            m = compute_hardware_metrics(genome, c)
            self.assertGreater(m.latency_us, 0, f"latency must be positive for {name}")
            self.assertEqual(m.platform, name)

    def test_faster_platform_gives_lower_latency(self):
        """Desktop should be faster than Cortex-M4 on the same genome."""
        from yane.evolution.hardware_aware import compute_hardware_metrics, HardwareConstraints
        genome = _make_genome(2, 1, n_hidden=2)
        m_m4     = compute_hardware_metrics(genome, HardwareConstraints(target_platform="cortex-m4"))
        m_desktop = compute_hardware_metrics(genome, HardwareConstraints(target_platform="desktop"))
        self.assertGreater(m_m4.latency_us, m_desktop.latency_us)

    def test_unknown_platform_raises(self):
        from yane.evolution.hardware_aware import compute_hardware_metrics, HardwareConstraints
        genome = _make_genome(2, 1)
        with self.assertRaises(ValueError):
            compute_hardware_metrics(genome, HardwareConstraints(target_platform="nonexistent"))

    def test_latency_scales_with_flops(self):
        """A genome with more FLOPs should have higher latency on the same platform."""
        from yane.evolution.hardware_aware import compute_hardware_metrics
        g_small = _make_genome(2, 1, n_hidden=0)
        g_large = _make_genome(4, 2, n_hidden=5)
        m_small = compute_hardware_metrics(g_small)
        m_large = compute_hardware_metrics(g_large)
        if m_large.flops > m_small.flops:
            self.assertGreater(m_large.latency_us, m_small.latency_us)


# ---------------------------------------------------------------------------
# Penalty calculation
# ---------------------------------------------------------------------------

class TestPenaltyCalculation(unittest.TestCase):
    def test_no_penalty_when_within_limits(self):
        from yane.evolution.hardware_aware import (
            compute_hardware_metrics, compute_penalty, HardwareConstraints,
        )
        genome = _make_genome(2, 1)
        c = HardwareConstraints(
            max_flops=10_000_000,       # very generous
            max_memory_bytes=1_000_000,
            max_latency_us=1_000_000.0,
        )
        metrics = compute_hardware_metrics(genome, c)
        self.assertEqual(compute_penalty(metrics, c), 0.0)

    def test_flop_violation_proportional(self):
        from yane.evolution.hardware_aware import compute_penalty, HardwareMetrics, HardwareConstraints
        c = HardwareConstraints(max_flops=1000, penalty_scale=1.0)
        metrics = HardwareMetrics(flops=2000, memory_bytes=0, latency_us=0.0, platform="desktop")
        # excess = (2000 - 1000) / 1000 = 1.0
        self.assertAlmostEqual(compute_penalty(metrics, c), 1.0)

    def test_memory_violation_proportional(self):
        from yane.evolution.hardware_aware import compute_penalty, HardwareMetrics, HardwareConstraints
        c = HardwareConstraints(max_memory_bytes=100, penalty_scale=2.0)
        metrics = HardwareMetrics(flops=0, memory_bytes=150, latency_us=0.0, platform="desktop")
        # excess = (150 - 100) / 100 = 0.5; penalty = 0.5 * 2.0 = 1.0
        self.assertAlmostEqual(compute_penalty(metrics, c), 1.0)

    def test_latency_violation(self):
        from yane.evolution.hardware_aware import compute_penalty, HardwareMetrics, HardwareConstraints
        c = HardwareConstraints(max_latency_us=100.0, penalty_scale=1.0)
        metrics = HardwareMetrics(flops=0, memory_bytes=0, latency_us=150.0, platform="desktop")
        self.assertAlmostEqual(compute_penalty(metrics, c), 0.5)

    def test_multiple_violations_sum(self):
        from yane.evolution.hardware_aware import compute_penalty, HardwareMetrics, HardwareConstraints
        c = HardwareConstraints(
            max_flops=100, max_memory_bytes=100, penalty_scale=1.0
        )
        # Each: 100% over → each penalty = 1.0, total = 2.0
        metrics = HardwareMetrics(flops=200, memory_bytes=200, latency_us=0.0, platform="desktop")
        self.assertAlmostEqual(compute_penalty(metrics, c), 2.0)

    def test_penalty_scale_applied(self):
        from yane.evolution.hardware_aware import compute_penalty, HardwareMetrics, HardwareConstraints
        c1 = HardwareConstraints(max_flops=100, penalty_scale=1.0)
        c2 = HardwareConstraints(max_flops=100, penalty_scale=3.0)
        metrics = HardwareMetrics(flops=200, memory_bytes=0, latency_us=0.0, platform="desktop")
        self.assertAlmostEqual(compute_penalty(metrics, c2), compute_penalty(metrics, c1) * 3.0)


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------

class TestParetoFront(unittest.TestCase):
    def _genomes_with_fitnesses(self, n: int):
        """Create n evaluated genomes with assigned fitnesses."""
        from yane import NeuroEvolution
        yane = NeuroEvolution(seed=0)
        yane.configure(2, 1, n_initial_hidden=0)
        yane.set_population_size(n)
        yane.set_max_iterations(n)
        yane.train(lambda g: float(sum(g.forward([0.5, 0.5]))))
        return yane._population._evaluated

    def test_pareto_front_non_empty(self):
        from yane.evolution.hardware_aware import hw_pareto_front, HardwareConstraints
        genomes = self._genomes_with_fitnesses(10)
        front = hw_pareto_front(list(genomes))
        self.assertGreater(len(front), 0)

    def test_all_front_members_are_non_dominated(self):
        """Each member of the front must not be dominated by any other genome."""
        from yane.evolution.hardware_aware import (
            hw_pareto_front, _hw_dominates, compute_hardware_metrics
        )
        genomes = self._genomes_with_fitnesses(10)
        front = hw_pareto_front(list(genomes))
        for g, m in front:
            dominated = any(
                _hw_dominates(h.fitness, compute_hardware_metrics(h), g.fitness, m)
                for h in genomes
            )
            self.assertFalse(dominated, f"Genome in front is actually dominated")

    def test_pareto_front_empty_input(self):
        from yane.evolution.hardware_aware import hw_pareto_front
        self.assertEqual(hw_pareto_front([]), [])

    def test_single_genome_is_always_on_front(self):
        from yane.evolution.hardware_aware import hw_pareto_front
        genomes = self._genomes_with_fitnesses(3)
        one = [list(genomes)[0]]
        front = hw_pareto_front(one)
        self.assertEqual(len(front), 1)

    def test_front_sorted_by_fitness_descending(self):
        from yane.evolution.hardware_aware import hw_pareto_front
        genomes = self._genomes_with_fitnesses(15)
        front = hw_pareto_front(list(genomes))
        fitnesses = [g.fitness for g, _ in front]
        self.assertEqual(fitnesses, sorted(fitnesses, reverse=True))


# ---------------------------------------------------------------------------
# Integration via NeuroEvolution API
# ---------------------------------------------------------------------------

class TestNeuroEvolutionIntegration(unittest.TestCase):
    def test_set_hardware_constraints_stores_config(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane.set_hardware_constraints(
            max_flops=100_000,
            max_memory_bytes=4096,
            target_platform="cortex-m4",
        )
        self.assertIsNotNone(yane._hw_constraints)
        self.assertEqual(yane._hw_constraints.target_platform, "cortex-m4")

    def test_hardware_profile_returns_metrics(self):
        from yane import NeuroEvolution
        from yane.evolution.hardware_aware import HardwareMetrics
        yane = NeuroEvolution()
        yane.configure(2, 1, n_initial_hidden=1)
        genome = yane.next_genome()
        metrics = yane.hardware_profile(genome)
        self.assertIsInstance(metrics, HardwareMetrics)
        self.assertGreater(metrics.flops, 0)

    def test_hardware_profile_with_custom_platform(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(2, 1, n_initial_hidden=2)  # ensure connections exist
        genome = yane.next_genome()
        m_m4      = yane.hardware_profile(genome, target_platform="cortex-m4")
        m_desktop = yane.hardware_profile(genome, target_platform="desktop")
        self.assertGreater(m_m4.latency_us, m_desktop.latency_us)

    def test_hw_pareto_front_raises_without_constraints(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane.set_population_size(5)
        yane.set_max_iterations(5)
        yane.train(lambda g: float(sum(g.forward([0.5, 0.5]))))
        with self.assertRaises(RuntimeError):
            yane.hw_pareto_front()

    def test_hw_pareto_front_after_training(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution(seed=0)
        yane.configure(2, 1, n_initial_hidden=1)
        yane.set_population_size(10)
        yane.set_max_iterations(10)
        yane.set_hardware_constraints(max_flops=10_000, target_platform="cortex-m4")
        yane.train(lambda g: float(sum(g.forward([0.5, 0.5]))))
        front = yane.hw_pareto_front()
        self.assertGreater(len(front), 0)

    def test_penalty_applied_during_training(self):
        """Tight FLOP limit should depress fitness relative to unconstrained."""
        from yane import NeuroEvolution
        # Unconstrained training
        yane1 = NeuroEvolution(seed=1)
        yane1.configure(2, 1, n_initial_hidden=2)
        yane1.set_population_size(10)
        yane1.set_max_iterations(10)
        yane1.train(lambda g: 1.0)  # constant evaluator
        best1 = yane1.get_best().fitness

        # With a very tight constraint — every genome gets penalized
        yane2 = NeuroEvolution(seed=1)
        yane2.configure(2, 1, n_initial_hidden=2)
        yane2.set_population_size(10)
        yane2.set_max_iterations(10)
        yane2.set_hardware_constraints(max_flops=1, penalty_scale=10.0)
        yane2.train(lambda g: 1.0)
        best2 = yane2.get_best().fitness

        self.assertGreater(best1, best2,
                           "Constrained training should produce lower fitness")


if __name__ == "__main__":
    unittest.main()
