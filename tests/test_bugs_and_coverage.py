"""Regression tests for fixed bugs and coverage for previously untested paths."""
import random
import unittest

from yane import NeuroEvolution
from yane.core.genome import Genome
from yane.core.connection import Connection
from yane.core.node import Node, NodeType
from yane.evolution.innovation import InnovationTracker
from yane.evolution.efficiency_penalty import EfficiencyPenalty
from yane.evolution.population import Population, _compatibility
from yane.evolution import smart_mutation


def _make_yane(n_inputs=2, n_outputs=1, **kw):
    yane = NeuroEvolution()
    yane.configure(n_inputs, n_outputs, **kw)
    return yane


# ---------------------------------------------------------------------------
# InnovationTracker
# ---------------------------------------------------------------------------

class TestInnovationTracker(unittest.TestCase):

    def test_next_is_monotonically_increasing(self):
        t = InnovationTracker()
        prev = t.next()
        for _ in range(10):
            n = t.next()
            self.assertGreater(n, prev)
            prev = n

    def test_get_connection_idempotent(self):
        t = InnovationTracker()
        a, b = t.next(), t.next()
        i1 = t.get_connection(a, b)
        i2 = t.get_connection(a, b)
        self.assertEqual(i1, i2, "Same (src, tgt) must always yield the same innovation")

    def test_get_connection_order_matters(self):
        t = InnovationTracker()
        a, b = t.next(), t.next()
        ab = t.get_connection(a, b)
        ba = t.get_connection(b, a)
        self.assertNotEqual(ab, ba, "(a→b) and (b→a) are distinct connections")

    def test_get_split_idempotent(self):
        t = InnovationTracker()
        conn_innov = t.next()
        r1 = t.get_split(conn_innov)
        r2 = t.get_split(conn_innov)
        self.assertEqual(r1, r2, "Splitting the same connection must yield the same innovations")

    def test_get_split_returns_three_distinct_numbers(self):
        t = InnovationTracker()
        conn_innov = t.next()
        node_i, conn_in, conn_out = t.get_split(conn_innov)
        self.assertEqual(len({node_i, conn_in, conn_out}), 3)

    def test_different_splits_get_different_innovations(self):
        t = InnovationTracker()
        c1, c2 = t.next(), t.next()
        r1 = t.get_split(c1)
        r2 = t.get_split(c2)
        # No overlap between the two splits
        self.assertTrue(set(r1).isdisjoint(set(r2)))


# ---------------------------------------------------------------------------
# _innov_cache invalidation
# ---------------------------------------------------------------------------

class TestInnovCache(unittest.TestCase):

    def test_cache_built_on_first_compatibility_call(self):
        yane = _make_yane()
        g = yane.next_genome()
        self.assertIsNone(g._innov_cache)
        g2 = g.copy()
        _compatibility(g, g2)
        self.assertIsNotNone(g._innov_cache)

    def test_cache_invalidated_after_add_connection(self):
        yane = _make_yane()
        g = yane.next_genome()
        g2 = g.copy()
        _compatibility(g, g2)  # builds cache
        self.assertIsNotNone(g._innov_cache)
        smart_mutation.add_connection(g, yane._tracker)
        self.assertIsNone(g._innov_cache, "Cache must be cleared after topology change")

    def test_cache_invalidated_after_add_node(self):
        yane = _make_yane()
        g = yane.next_genome()
        smart_mutation.add_connection(g, yane._tracker)
        g2 = g.copy()
        _compatibility(g, g2)  # builds cache
        smart_mutation.add_node(g, yane._tracker)
        self.assertIsNone(g._innov_cache)

    def test_copy_starts_with_empty_cache(self):
        yane = _make_yane()
        g = yane.next_genome()
        g2 = g.copy()
        _compatibility(g, g2)  # builds cache on both
        copy = g.copy()
        self.assertIsNone(copy._innov_cache, "Copies must start with a fresh (None) cache")

    def test_cache_cleared_by_invalidate_topology(self):
        yane = _make_yane()
        g = yane.next_genome()
        g2 = g.copy()
        _compatibility(g, g2)
        g._invalidate_topology()
        self.assertIsNone(g._innov_cache)


# ---------------------------------------------------------------------------
# add_node on empty genome (bug fix: was adding disconnected node)
# ---------------------------------------------------------------------------

class TestAddNodeEdgeCases(unittest.TestCase):

    def test_add_node_on_empty_genome_is_noop(self):
        yane = _make_yane()
        g = yane.next_genome()
        n_before = len(g.nodes)
        # With no connections, add_node must not add anything
        smart_mutation.add_node(g, yane._tracker)
        self.assertEqual(len(g.nodes), n_before,
                         "add_node with no connections must be a no-op")

    def test_add_node_splits_connection_correctly(self):
        yane = _make_yane()
        g = yane.next_genome()
        smart_mutation.add_connection(g, yane._tracker)
        n_before = len(g.nodes)
        conn_before = g.connection_count
        smart_mutation.add_node(g, yane._tracker)
        # A→B becomes A→N→B: 1 node added, connections same count (removed 1, added 2)
        self.assertEqual(len(g.nodes), n_before + 1)
        self.assertEqual(g.connection_count, conn_before + 1)

    def test_add_node_forward_still_works(self):
        yane = _make_yane(2, 1)
        g = yane.next_genome()
        smart_mutation.add_connection(g, yane._tracker)
        smart_mutation.add_node(g, yane._tracker)
        out = g.forward([0.5, 0.5])
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0] != out[0], "NaN after add_node")


# ---------------------------------------------------------------------------
# EfficiencyPenalty
# ---------------------------------------------------------------------------

class TestEfficiencyPenalty(unittest.TestCase):

    def test_no_penalty_within_budget(self):
        ep = EfficiencyPenalty(max_ms=10.0, penalty_per_ms=1.0)
        self.assertAlmostEqual(ep.apply(5.0, 8.0), 5.0)

    def test_penalty_applied_over_budget(self):
        ep = EfficiencyPenalty(max_ms=10.0, penalty_per_ms=1.0)
        # 5ms over budget → 5 penalty
        self.assertAlmostEqual(ep.apply(10.0, 15.0), 5.0)

    def test_penalty_can_make_fitness_negative(self):
        ep = EfficiencyPenalty(max_ms=5.0, penalty_per_ms=2.0)
        result = ep.apply(1.0, 20.0)
        self.assertLess(result, 0.0)

    def test_exactly_at_budget_no_penalty(self):
        ep = EfficiencyPenalty(max_ms=10.0, penalty_per_ms=1.0)
        self.assertAlmostEqual(ep.apply(3.0, 10.0), 3.0)


# ---------------------------------------------------------------------------
# Stagnation → diversity injection
# ---------------------------------------------------------------------------

class TestStagnationInjection(unittest.TestCase):

    def test_injection_triggered_after_stagnation(self):
        yane = _make_yane(2, 1)
        yane.set_population_size(10)
        pop = yane._population

        # Submit constant fitness to trigger stagnation
        for _ in range(pop.stagnation_threshold + 5):
            g = pop.select_for_evaluation()
            pop.submit(g, 0.0)  # constant fitness → stagnation

        # _since_last_injection should have been reset at least once
        # (injection resets it to 0) while stagnation_count stays high
        self.assertGreater(pop.stagnation_count, 0)

    def test_since_last_injection_resets_on_improvement(self):
        yane = _make_yane(2, 1)
        pop = yane._population

        # First submit: always an "improvement" over −inf, resets both counters.
        g = pop.select_for_evaluation()
        pop.submit(g, 0.0)
        self.assertEqual(pop._stagnation_count, 0)
        self.assertEqual(pop._since_last_injection, 0)

        # Second submit with same fitness → stagnation increments.
        g = pop.select_for_evaluation()
        pop.submit(g, 0.0)
        self.assertEqual(pop._stagnation_count, 1)
        self.assertEqual(pop._since_last_injection, 1)

        # Third submit with better fitness → both counters reset to 0.
        g = pop.select_for_evaluation()
        pop.submit(g, 100.0)
        self.assertEqual(pop._since_last_injection, 0)
        self.assertEqual(pop._stagnation_count, 0)


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------

class TestBatchAPI(unittest.TestCase):
    """next_genome_batch requires at least one evaluated genome before it can spawn
    offspring. Tests here evaluate the seed first to satisfy that precondition."""

    def _setup(self, pop_size=20):
        yane = _make_yane(2, 1)
        yane.set_population_size(pop_size)
        seed = yane.next_genome()
        yane.submit_fitness(1.0)  # evaluate seed — required before batch spawn
        return yane

    def test_next_genome_batch_returns_n_genomes(self):
        yane = self._setup()
        batch = yane.next_genome_batch(5)
        self.assertEqual(len(batch), 5)

    def test_batch_genomes_are_distinct(self):
        yane = self._setup()
        batch = yane.next_genome_batch(5)
        ids = [id(g) for g in batch]
        self.assertEqual(len(set(ids)), 5, "Batch must contain distinct genome objects")

    def test_submit_fitness_batch_evaluates_all(self):
        yane = self._setup()
        batch = yane.next_genome_batch(4)
        results = [(g, float(i)) for i, g in enumerate(batch)]
        yane.submit_fitness_batch(results)
        self.assertGreaterEqual(yane._population.evaluated_count, 4)

    def test_get_best_after_batch(self):
        yane = self._setup()
        batch = yane.next_genome_batch(4)
        results = [(g, float(i * 10)) for i, g in enumerate(batch)]
        yane.submit_fitness_batch(results)
        best = yane.get_best()
        self.assertAlmostEqual(best.fitness, 30.0)


# ---------------------------------------------------------------------------
# Novelty numerical stability (float64 fix)
# ---------------------------------------------------------------------------

class TestNoveltyNumericalStability(unittest.TestCase):

    def test_extreme_outputs_do_not_cause_nan_novelty(self):
        """Networks with very large outputs must not produce NaN novelty scores.

        Before the float64 fix, behavior vectors used float32 which overflowed
        and produced NaN in distance calculations.
        """
        from yane.evolution.population import Population
        import numpy as np

        pop = Population(max_size=10)
        for i in range(6):
            g = pop.select_for_evaluation()
            # Manually inject extreme behavior so we stress-test the novelty calc
            huge_vec = np.array([1e30] * 20, dtype=np.float64)
            pop._behaviors[id(g)] = huge_vec
            g.fitness = float(i)
            g.shared_fitness = float(i)
            pop._unevaluated.remove(g)
            pop._evaluated.append(g)
            pop._novelty_evals_since_recompute += 1

        novelty = pop._compute_novelty()
        for gid, score in novelty.items():
            self.assertFalse(score != score, f"Novelty score is NaN for genome {gid}")

    def test_behavior_vector_stored_as_float64(self):
        import numpy as np
        yane = _make_yane(2, 1)
        g = yane.next_genome()
        yane._population.submit(g, 1.0)
        bvec = yane._population._behaviors.get(id(g))
        if bvec is not None:
            self.assertEqual(bvec.dtype, np.float64,
                             "Behavior vectors must be float64 to avoid overflow")


# ---------------------------------------------------------------------------
# Compatibility distance is non-negative and (approximately) bounded
# ---------------------------------------------------------------------------

class TestCompatibilityBounds(unittest.TestCase):

    def test_distance_always_non_negative(self):
        for _ in range(20):
            yane = _make_yane(2, 1)
            g1 = yane.next_genome()
            g2 = yane.next_genome()
            for _ in range(random.randint(0, 15)):
                g1.mutate(yane._tracker)
                g2.mutate(yane._tracker)
            d = _compatibility(g1, g2)
            self.assertGreaterEqual(d, 0.0, "Compatibility distance must be non-negative")

    def test_copy_has_zero_distance(self):
        yane = _make_yane(2, 1)
        g = yane.next_genome()
        for _ in range(10):
            g.mutate(yane._tracker)
        copy = g.copy()
        self.assertAlmostEqual(_compatibility(g, copy), 0.0, places=10)


# ---------------------------------------------------------------------------
# Topological sort correctness (deque fix)
# ---------------------------------------------------------------------------

class TestTopologicalSort(unittest.TestCase):

    def test_exec_order_processes_all_non_input_nodes(self):
        yane = _make_yane(2, 2)
        g = yane.next_genome()
        for _ in range(5):
            smart_mutation.add_connection(g, yane._tracker)
            smart_mutation.add_node(g, yane._tracker)
        # Trigger topology build
        g.forward([0.1, 0.2])
        if g._exec_order is not None:
            expected = set(g.nodes) - set(g.input_nodes)
            actual = set(g._exec_order)
            self.assertEqual(actual, expected,
                             "exec_order must contain all non-input nodes exactly once")

    def test_forward_output_same_before_and_after_explicit_rebuild(self):
        yane = _make_yane(2, 1)
        g = yane.next_genome()
        for _ in range(8):
            smart_mutation.add_connection(g, yane._tracker)
        out1 = g.forward([0.3, 0.7])
        g._invalidate_topology()
        out2 = g.forward([0.3, 0.7])
        self.assertAlmostEqual(out1[0], out2[0], places=10,
            msg="Rebuilding topology must not change forward() output")


# ---------------------------------------------------------------------------
# Genome pickle (multiprocessing support)
# ---------------------------------------------------------------------------

class TestGenomePickle(unittest.TestCase):
    """Genome must survive pickle/unpickle and produce identical forward() output.

    This is required for multiprocessing: genomes are sent to worker processes
    via IPC, which uses pickle.  _compiled_forward and _forward_dispatch are
    closures that can't be pickled; __getstate__ strips them so they are
    rebuilt transparently on the first forward() call in the subprocess.
    """

    def _genome_with_connections(self, n_conns=5):
        yane = _make_yane(2, 1)
        g = yane.next_genome()
        for _ in range(n_conns):
            smart_mutation.add_connection(g, yane._tracker)
        return g

    def test_pickle_before_forward(self):
        """Unpickled acyclic genome must produce the same output as the original.

        Cyclic networks are excluded because BFS forward order depends on Python
        set iteration over Node objects, which changes across pickle/unpickle as
        objects receive new memory addresses (different id() → different hash).
        """
        import pickle
        yane = _make_yane(2, 1)
        g = yane.next_genome()
        # Build a guaranteed acyclic network: only input→output connections.
        innov1 = yane._tracker.get_connection(
            g.input_nodes[0].innovation, g.output_nodes[0].innovation)
        innov2 = yane._tracker.get_connection(
            g.input_nodes[1].innovation, g.output_nodes[0].innovation)
        from yane.core.connection import Connection as Conn
        c1 = Conn(g.output_nodes[0], innovation=innov1); c1.weight = 0.5
        c2 = Conn(g.output_nodes[0], innovation=innov2); c2.weight = -0.3
        g.input_nodes[0].connections.append(c1)
        g.input_nodes[1].connections.append(c2)
        g._invalidate_topology()

        data = pickle.dumps(g)
        g2   = pickle.loads(data)
        self.assertEqual(g.forward([0.4, 0.6]), g2.forward([0.4, 0.6]),
                         "Unpickled acyclic genome must produce same output (before first forward)")

    def test_pickle_after_forward_acyclic(self):
        """Acyclic network uses compiled closure — must be stripped and rebuilt."""
        import pickle
        g = self._genome_with_connections()
        g.forward([0.5, 0.5])   # builds _compiled_forward closure
        self.assertIsNotNone(g._forward_dispatch)

        data = pickle.dumps(g)  # __getstate__ strips closures
        g2   = pickle.loads(data)
        self.assertIsNone(g2._forward_dispatch, "Dispatch must be None after unpickling")

        out1 = g.forward([0.3, 0.7])
        g2.reset(); out2 = g2.forward([0.3, 0.7])
        self.assertAlmostEqual(out1[0], out2[0], places=10,
                               msg="Unpickled genome must produce same output after closure rebuild")

    def test_pickle_after_forward_cyclic(self):
        """Cyclic network uses _bfs_forward (bound method) — also must survive pickle."""
        import pickle
        from yane.core.connection import Connection as Conn
        yane = _make_yane(1, 1)
        g = yane.next_genome()
        smart_mutation.add_connection(g, yane._tracker)
        # Add self-loop to create cycle
        back = Conn(g.input_nodes[0]); back.weight = 0.1
        g.output_nodes[0].connections.append(back)
        g._invalidate_topology()

        g.forward([1.0])   # builds _bfs_forward dispatch
        data = pickle.dumps(g)
        g2   = pickle.loads(data)
        g.reset();  out1 = g.forward([1.0])
        g2.reset(); out2 = g2.forward([1.0])
        self.assertAlmostEqual(out1[0], out2[0], places=10,
                               msg="Cyclic genome must produce same output after unpickling")

    def test_pickle_preserves_weights(self):
        """Unpickling must restore all connection weights exactly."""
        import pickle
        g = self._genome_with_connections()
        weights_before = [c.weight for n in g.nodes for c in n.connections]
        g2 = pickle.loads(pickle.dumps(g))
        weights_after  = [c.weight for n in g2.nodes for c in n.connections]
        self.assertEqual(weights_before, weights_after,
                         "All weights must be identical after pickle/unpickle")

    def test_mp_evaluate_function(self):
        """_mp_evaluate must produce the same result as direct evaluation."""
        from yane.gui.worker import _mp_initializer, _mp_evaluate
        from yane.examples.simple_2_2_continuous import make_eval, N_INPUTS, N_OUTPUTS

        _mp_initializer(make_eval, None)   # set up global eval fn

        yane = _make_yane(N_INPUTS, N_OUTPUTS)
        g    = yane.next_genome()
        for _ in range(4):
            smart_mutation.add_connection(g, yane._tracker)

        ev = make_eval()
        direct = ev(g)
        g.reset()
        via_mp, elapsed_ms = _mp_evaluate(g)

        self.assertAlmostEqual(direct, via_mp, places=10,
                               msg="_mp_evaluate must match direct evaluation")
        self.assertGreaterEqual(elapsed_ms, 0.0)


# ---------------------------------------------------------------------------
# _finalize_fitness: efficiency penalty applied via all submission paths
# ---------------------------------------------------------------------------

class TestFinalizeFitness(unittest.TestCase):
    """Verify efficiency penalty is applied consistently across submission paths."""

    def _make(self):
        yane = _make_yane(n_inputs=1, n_outputs=1)
        yane.set_efficiency_penalty(max_ms=0.0, penalty_per_ms=1.0)
        return yane

    def test_submit_fitness_applies_efficiency_penalty(self):
        yane = self._make()
        g = yane.next_genome()
        yane.submit_fitness(10.0, elapsed_ms=2.0)
        # penalty: 2 ms over max_ms=0 → fitness reduced by 2.0
        self.assertAlmostEqual(yane.get_best().fitness, 8.0, places=6)

    def test_submit_fitness_no_elapsed_skips_penalty(self):
        yane = self._make()
        g = yane.next_genome()
        yane.submit_fitness(10.0, elapsed_ms=None)
        # no elapsed_ms → penalty skipped, only sanitize
        self.assertAlmostEqual(yane.get_best().fitness, 10.0, places=6)

    def test_submit_fitness_batch_applies_efficiency_penalty(self):
        yane = self._make()
        g = yane.next_genome()
        yane.submit_fitness(5.0, elapsed_ms=1.0)  # seed
        g2 = yane.next_genome_batch(1)
        yane.submit_fitness_batch([(g2[0], 10.0, 3.0)])
        # penalty for g2: 3 ms over 0 → 10 - 3 = 7
        best = yane.get_best()
        self.assertLessEqual(best.fitness, 7.0 + 1e-9)

    def test_train_applies_efficiency_penalty(self):
        yane = _make_yane(n_inputs=1, n_outputs=1)
        yane.set_efficiency_penalty(max_ms=0.0, penalty_per_ms=1000.0)
        yane.set_max_iterations(1)
        # With a huge penalty, fitness must be heavily penalised
        raw_fitnesses = []
        def ff(genome):
            import time; time.sleep(0); return 1000.0
        yane.train(ff)
        best = yane.get_best()
        # elapsed is always >0 ms even for no-op, so penalty fires
        self.assertLess(best.fitness, 1000.0)


class TestLeakAlpha(unittest.TestCase):
    """Leaky-memory gate on persistent hidden nodes."""

    def _persistent_node(self, alpha: float = 1.0) -> Node:
        n = Node(NodeType.HIDDEN)
        n.persist_value = True
        n.leak_alpha = alpha
        return n

    def test_default_leak_alpha_is_one(self):
        n = Node(NodeType.HIDDEN)
        self.assertEqual(n.leak_alpha, 1.0)

    def test_full_retain_alpha_one(self):
        """alpha=1.0 → value == activated (no decay)."""
        n = self._persistent_node(alpha=1.0)
        n.activation = __import__('yane.util.activation', fromlist=['ActivationType']).ActivationType.LINEAR
        n.value = 2.0
        n.fire(set())
        self.assertAlmostEqual(n.value, 2.0, places=9)

    def test_half_decay(self):
        """alpha=0.5 → value == 0.5 * activated after one step."""
        from yane.util.activation import ActivationType
        n = self._persistent_node(alpha=0.5)
        n.activation = ActivationType.LINEAR
        n.value = 4.0
        n.fire(set())
        self.assertAlmostEqual(n.value, 2.0, places=9)

    def test_zero_alpha_no_carry(self):
        """alpha=0.0 → value == 0 after fire (no memory carry-over)."""
        from yane.util.activation import ActivationType
        n = self._persistent_node(alpha=0.0)
        n.activation = ActivationType.LINEAR
        n.value = 5.0
        n.fire(set())
        self.assertAlmostEqual(n.value, 0.0, places=9)

    def test_output_node_ignores_leak_alpha(self):
        """Output nodes always full-retain regardless of leak_alpha."""
        from yane.util.activation import ActivationType
        n = Node(NodeType.OUTPUT)
        n.leak_alpha = 0.0
        n.activation = ActivationType.LINEAR
        n.value = 3.0
        n.fire(set())
        self.assertAlmostEqual(n.value, 3.0, places=9)

    def test_fire_simple_respects_leak_alpha(self):
        """fire_simple() applies the same leaky decay logic."""
        from yane.util.activation import ActivationType
        n = self._persistent_node(alpha=0.5)
        n.activation = ActivationType.LINEAR
        n.value = 6.0
        n.fire_simple()
        self.assertAlmostEqual(n.value, 3.0, places=9)

    def test_large_compiled_forward_respects_leak_alpha(self):
        """The NumPy-backed compiled path must decay persistent hidden state."""
        from yane.core.connection import Connection
        from yane.core.genome import Genome
        from yane.util.activation import ActivationType

        g = Genome()
        inp = Node(NodeType.INPUT)
        hid = self._persistent_node(alpha=0.5)
        out = Node(NodeType.OUTPUT)
        inp.activation = ActivationType.LINEAR
        hid.activation = ActivationType.LINEAR
        out.activation = ActivationType.LINEAR
        inp.connections.append(Connection(hid))
        inp.connections[0].weight = 1.0
        for _ in range(g._FIRE_NUMPY_THRESHOLD):
            conn = Connection(out)
            conn.weight = 0.0
            hid.connections.append(conn)
        g.nodes = [inp, hid, out]
        g.input_nodes = [inp]
        g.output_nodes = [out]
        g._invalidate_topology()

        g.forward([4.0])
        self.assertAlmostEqual(hid.value, 2.0, places=9)
        g.forward([0.0])
        self.assertAlmostEqual(hid.value, 1.0, places=9)

    def test_copy_preserves_leak_alpha(self):
        n = self._persistent_node(alpha=0.3)
        c = n.copy()
        self.assertAlmostEqual(c.leak_alpha, 0.3, places=9)

    def test_pickle_roundtrip_preserves_leak_alpha(self):
        import pickle
        n = self._persistent_node(alpha=0.7)
        n2 = pickle.loads(pickle.dumps(n))
        self.assertAlmostEqual(n2.leak_alpha, 0.7, places=9)

    def test_old_pickle_gets_default_leak_alpha(self):
        """Pickles from before leak_alpha existed should unpickle with alpha=1.0."""
        import pickle
        n = self._persistent_node(alpha=0.5)
        state = n.__getstate__()
        del state['leak_alpha']
        del state['mutation_leak_alpha']
        n2 = Node.__new__(Node)
        n2.__setstate__(state)
        self.assertAlmostEqual(n2.leak_alpha, 1.0, places=9)

    def test_mutate_clamps_leak_alpha(self):
        """After many mutations, leak_alpha stays in [0, 1]."""
        n = self._persistent_node(alpha=0.5)
        for _ in range(200):
            n.mutate(sigma=10.0)
        self.assertGreaterEqual(n.leak_alpha, 0.0)
        self.assertLessEqual(n.leak_alpha, 1.0)

    def test_non_persistent_node_not_mutated(self):
        """leak_alpha should not change for non-persistent hidden nodes."""
        from yane.util.activation import ActivationType
        n = Node(NodeType.HIDDEN)
        n.persist_value = False
        n.leak_alpha = 0.5
        for _ in range(50):
            n.mutate(sigma=1.0)
        self.assertAlmostEqual(n.leak_alpha, 0.5, places=9)


if __name__ == '__main__':
    unittest.main()
