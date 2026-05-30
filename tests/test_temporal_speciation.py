"""Tests für Temporal Speciation / TemporalDistance (evolution/compatibility.py).

Akzeptanzkriterien:
  1. DTW-Distanz zwischen identischen Trajektorien ist 0
  2. Caching: Trajektorie wird nicht neu berechnet wenn Topologie unveraendert
  3. Caching-Invalidierung: neue Topologie → Cache-Miss
  4. DTW-Mathe korrekt
  5. Protokoll-Kompatibilität: TemporalDistance ist ein DistanceMetric
  6. Kombinierbar mit ChainMetric
  7. NeuroEvolution.set_compatibility_distance() akzeptiert TemporalDistance
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

def _make_simple_genome(n_inputs: int = 2, n_outputs: int = 1, weight: float = 0.5) -> Genome:
    g = Genome()
    for i in range(n_inputs):
        n = Node(NodeType.INPUT, i); n.activation = ActivationType.LINEAR; n.input_index = i
        g.input_nodes.append(n); g.nodes.append(n)
    out = Node(NodeType.OUTPUT, n_inputs); out.activation = ActivationType.SIGMOID; out.bias = 0.0
    g.output_nodes.append(out); g.nodes.append(out)
    innov = 10
    for inp in g.input_nodes:
        c = Connection(out, innov); c.weight = weight; inp.connections.append(c); innov += 1
    g._invalidate_topology()
    return g


# ---------------------------------------------------------------------------
# DTW-Mathe — acceptance criterion 4
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestDTWMath(unittest.TestCase):

    def test_identical_trajectories_distance_zero(self):
        """DTW between identical trajectories must be 0."""
        from yane.evolution.compatibility import _dtw
        traj = [[0.5], [0.3], [0.7], [0.2]]
        self.assertAlmostEqual(_dtw(traj, traj), 0.0)

    def test_empty_trajectory_returns_zero(self):
        from yane.evolution.compatibility import _dtw
        self.assertAlmostEqual(_dtw([], [[0.5]]), 0.0)
        self.assertAlmostEqual(_dtw([[0.5]], []), 0.0)

    def test_single_step_euclidean(self):
        """Single step: DTW = Euclidean distance between the two output vectors."""
        from yane.evolution.compatibility import _dtw
        a = [[1.0, 0.0]]
        b = [[0.0, 1.0]]
        expected = math.sqrt(2.0)
        self.assertAlmostEqual(_dtw(a, b), expected, places=10)

    def test_dtw_symmetric(self):
        """DTW(a, b) == DTW(b, a)."""
        from yane.evolution.compatibility import _dtw
        a = [[0.1], [0.5], [0.9]]
        b = [[0.9], [0.5], [0.1]]
        self.assertAlmostEqual(_dtw(a, b), _dtw(b, a), places=10)

    def test_dtw_nonnegative(self):
        from yane.evolution.compatibility import _dtw
        a = [[0.3], [0.6], [0.1]]
        b = [[0.7], [0.2], [0.8]]
        self.assertGreaterEqual(_dtw(a, b), 0.0)

    def test_dtw_different_lengths(self):
        """DTW handles trajectories of different lengths."""
        from yane.evolution.compatibility import _dtw
        a = [[1.0], [1.0], [1.0]]
        b = [[1.0], [1.0], [1.0], [1.0], [1.0]]
        # Identical values at each step → DTW = 0
        self.assertAlmostEqual(_dtw(a, b), 0.0)

    def test_dtw_larger_for_dissimilar(self):
        """More dissimilar trajectories should yield larger DTW."""
        from yane.evolution.compatibility import _dtw
        close = [[0.5], [0.5], [0.5]]
        similar = [[0.6], [0.6], [0.6]]
        different = [[0.0], [0.0], [0.0]]
        d_close = _dtw(close, similar)
        d_far = _dtw(close, different)
        self.assertLess(d_close, d_far)


# ---------------------------------------------------------------------------
# Acceptance criterion 1: DTW zwischen identischen Trajektorien = 0
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestTemporalDistanceIdentical(unittest.TestCase):

    def test_identical_genomes_distance_zero(self):
        """Two identical genomes must produce distance 0."""
        from yane.evolution.compatibility import TemporalDistance
        g1 = _make_simple_genome(weight=0.5)
        td = TemporalDistance(n_rollouts=3, rollout_len=10, seed=42)
        d = td(g1, g1)
        self.assertAlmostEqual(d, 0.0, places=10,
                               msg="Distance between identical genomes must be 0")

    def test_copy_genome_distance_zero(self):
        """A copy of a genome must have distance 0 to the original."""
        from yane.evolution.compatibility import TemporalDistance
        g1 = _make_simple_genome(weight=0.7)
        g2 = g1.copy()
        td = TemporalDistance(n_rollouts=3, rollout_len=10, seed=42)
        d = td(g1, g2)
        self.assertAlmostEqual(d, 0.0, places=10)

    def test_different_genomes_nonzero_distance(self):
        """Different genomes should generally have distance > 0."""
        from yane.evolution.compatibility import TemporalDistance
        g1 = _make_simple_genome(weight=0.1)
        g2 = _make_simple_genome(weight=0.9)
        td = TemporalDistance(n_rollouts=3, rollout_len=10, seed=42)
        d = td(g1, g2)
        self.assertGreater(d, 0.0,
                           "Genomes with very different weights should have distance > 0")


# ---------------------------------------------------------------------------
# Acceptance criterion 2+3: Caching
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestCaching(unittest.TestCase):

    def test_cache_hit_same_topology(self):
        """Second call with same genome must use cached trajectory."""
        from yane.evolution.compatibility import TemporalDistance
        g1 = _make_simple_genome(weight=0.5)
        g2 = _make_simple_genome(weight=0.3)
        td = TemporalDistance(n_rollouts=2, rollout_len=5, seed=1)
        td(g1, g2)  # first call computes both trajectories
        misses_after_first = td._cache_misses
        hits_after_first = td._cache_hits
        td(g1, g2)  # second call: both should hit cache
        self.assertGreater(td._cache_hits, hits_after_first,
                           "Second call should produce cache hits")
        self.assertEqual(td._cache_misses, misses_after_first,
                         "Second call should produce no new cache misses")

    def test_cache_miss_after_topology_change(self):
        """After modifying a connection weight, the cache must miss."""
        from yane.evolution.compatibility import TemporalDistance
        g1 = _make_simple_genome(weight=0.5)
        g2 = _make_simple_genome(weight=0.3)
        td = TemporalDistance(n_rollouts=2, rollout_len=5, seed=1)
        td(g1, g2)  # populate cache
        misses_before = td._cache_misses
        # Change g1's topology
        g1.input_nodes[0].connections[0].weight = 0.99
        td(g1, g2)  # g1's key changed → cache miss
        self.assertGreater(td._cache_misses, misses_before,
                           "Weight change should cause cache miss for g1")

    def test_invalidate_cache_clears_all(self):
        from yane.evolution.compatibility import TemporalDistance
        g1 = _make_simple_genome()
        g2 = _make_simple_genome()
        td = TemporalDistance(n_rollouts=2, rollout_len=5, seed=2)
        td(g1, g2)
        self.assertGreater(len(td._cache), 0)
        td.invalidate_cache()
        self.assertEqual(len(td._cache), 0)

    def test_topology_hash_same_for_identical_genome(self):
        from yane.evolution.compatibility import _topology_hash
        g = _make_simple_genome(weight=0.5)
        h1 = _topology_hash(g)
        h2 = _topology_hash(g)
        self.assertEqual(h1, h2)

    def test_topology_hash_different_after_weight_change(self):
        from yane.evolution.compatibility import _topology_hash
        g = _make_simple_genome(weight=0.5)
        h1 = _topology_hash(g)
        g.input_nodes[0].connections[0].weight = 99.9
        h2 = _topology_hash(g)
        self.assertNotEqual(h1, h2)


# ---------------------------------------------------------------------------
# Acceptance criterion 5: DistanceMetric protocol
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestProtocolCompatibility(unittest.TestCase):

    def test_temporal_distance_is_distance_metric(self):
        from yane.evolution.compatibility import TemporalDistance, DistanceMetric
        td = TemporalDistance()
        self.assertIsInstance(td, DistanceMetric)

    def test_temporal_distance_callable(self):
        from yane.evolution.compatibility import TemporalDistance
        td = TemporalDistance(n_rollouts=2, rollout_len=5, seed=0)
        g1 = _make_simple_genome()
        g2 = _make_simple_genome()
        result = td(g1, g2)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)

    def test_time_weight_scales_distance(self):
        """time_weight=0 must return 0; time_weight=2 doubles the distance."""
        from yane.evolution.compatibility import TemporalDistance
        g1 = _make_simple_genome(weight=0.1)
        g2 = _make_simple_genome(weight=0.9)
        td1 = TemporalDistance(n_rollouts=2, rollout_len=5, time_weight=1.0, seed=3)
        td2 = TemporalDistance(n_rollouts=2, rollout_len=5, time_weight=2.0, seed=3)
        d1 = td1(g1, g2)
        d2 = td2(g1, g2)
        self.assertAlmostEqual(d2, 2.0 * d1, places=9)

    def test_result_is_finite(self):
        from yane.evolution.compatibility import TemporalDistance
        g1 = _make_simple_genome()
        g2 = _make_simple_genome()
        td = TemporalDistance(n_rollouts=3, rollout_len=15)
        d = td(g1, g2)
        self.assertFalse(math.isnan(d))
        self.assertFalse(math.isinf(d))


# ---------------------------------------------------------------------------
# Acceptance criterion 6: ChainMetric combination
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestChainMetricCombination(unittest.TestCase):

    def test_chain_with_topology_distance(self):
        from yane.evolution.compatibility import TemporalDistance, ChainMetric, TopologyDistance
        g1 = _make_simple_genome(weight=0.3)
        g2 = _make_simple_genome(weight=0.8)
        metric = ChainMetric(
            [TopologyDistance(), TemporalDistance(n_rollouts=2, rollout_len=5, seed=99)],
            weights=[0.5, 0.5],
        )
        d = metric(g1, g2)
        self.assertGreaterEqual(d, 0.0)
        self.assertFalse(math.isnan(d))

    def test_chain_zero_weight_on_temporal(self):
        """Weight=0 for TemporalDistance should give same result as TopologyDistance alone."""
        from yane.evolution.compatibility import TemporalDistance, ChainMetric, TopologyDistance
        g1 = _make_simple_genome(weight=0.3)
        g2 = _make_simple_genome(weight=0.8)
        topo = TopologyDistance()
        chain = ChainMetric(
            [topo, TemporalDistance(n_rollouts=2, rollout_len=5)],
            weights=[1.0, 0.0],
        )
        self.assertAlmostEqual(chain(g1, g2), topo(g1, g2), places=10)


# ---------------------------------------------------------------------------
# NeuroEvolution integration
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestNeuroEvolutionIntegration(unittest.TestCase):

    def test_set_compatibility_distance_accepts_temporal(self):
        import yane
        from yane.evolution.compatibility import TemporalDistance
        ne = yane.NeuroEvolution()
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=10, max_connections=20)
        td = TemporalDistance(n_rollouts=2, rollout_len=5, seed=0)
        ne.set_compatibility_distance(td)  # should not raise

    def test_temporal_distance_exported_from_yane(self):
        import yane
        self.assertTrue(hasattr(yane, "TemporalDistance"))
        self.assertTrue(hasattr(yane, "ChainMetric"))

    def test_train_with_temporal_distance_no_crash(self):
        import yane
        from yane.evolution.compatibility import TemporalDistance
        ne = yane.NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=10, max_connections=20)
        ne.set_population_size(10)
        ne.set_compatibility_distance(TemporalDistance(n_rollouts=2, rollout_len=5, seed=0))
        ne.set_max_iterations(15)
        ne.train(lambda g: sum(g.forward([0.5, 0.5])))


if __name__ == "__main__":
    unittest.main()
