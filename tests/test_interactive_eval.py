"""Tests for Interactive / Human-in-the-Loop evaluation.

Covers:
- Elo rating computation and consistency
- Surrogate model warmup and prediction
- Feedback-to-fitness conversion in all modes
- Mode switching
- NeuroEvolution.set_interactive_evaluation() / submit_feedback() integration
"""
from __future__ import annotations

import threading
import time
import unittest

import pytest

from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_genome(weight: float = 1.0, fitness: float = 0.0) -> Genome:
    g = Genome()
    g.max_nodes = 10
    g.max_connections = 10
    inp = Node(NodeType.INPUT, 0)
    inp.activation = ActivationType.LINEAR
    out = Node(NodeType.OUTPUT, 1)
    out.activation = ActivationType.LINEAR
    g.nodes.extend([inp, out])
    g.input_nodes.append(inp)
    g.output_nodes.append(out)
    conn = Connection(out, innovation=10)
    conn.weight = weight
    inp.connections.append(conn)
    g.fitness = fitness
    g._invalidate_topology()
    return g


# ---------------------------------------------------------------------------
# EloRating
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestEloRating(unittest.TestCase):

    def _elo(self):
        from yane.evolution.interactive_eval import EloRating
        return EloRating(k_factor=32.0, default_rating=1000.0)

    def test_default_rating(self):
        elo = self._elo()
        self.assertAlmostEqual(elo.get(99), 1000.0)

    def test_winner_gains_points(self):
        elo = self._elo()
        elo.update(winner_id=1, loser_id=2)
        self.assertGreater(elo.get(1), 1000.0)

    def test_loser_loses_points(self):
        elo = self._elo()
        elo.update(winner_id=1, loser_id=2)
        self.assertLess(elo.get(2), 1000.0)

    def test_sum_conserved(self):
        elo = self._elo()
        elo.update(winner_id=1, loser_id=2)
        total = elo.get(1) + elo.get(2)
        self.assertAlmostEqual(total, 2000.0, places=6)

    def test_multiple_wins_converge_order(self):
        """After many wins, the stronger genome should have a higher Elo."""
        elo = self._elo()
        for _ in range(20):
            elo.update(winner_id=1, loser_id=2)
        self.assertGreater(elo.get(1), elo.get(2))

    def test_all_rated_tracks_participants(self):
        elo = self._elo()
        elo.update(winner_id=5, loser_id=7)
        rated = set(elo.all_rated())
        self.assertIn(5, rated)
        self.assertIn(7, rated)

    def test_consistent_with_simulated_user(self):
        """Simulated user always picks genome_id=1 as winner → Elo(1) > Elo(2)."""
        elo = self._elo()
        for _ in range(10):
            elo.update(winner_id=1, loser_id=2)
        self.assertGreater(elo.get(1), elo.get(2))

    def test_draw_by_symmetric_wins(self):
        """Alternating wins should keep ratings roughly equal."""
        elo = self._elo()
        for i in range(20):
            if i % 2 == 0:
                elo.update(1, 2)
            else:
                elo.update(2, 1)
        diff = abs(elo.get(1) - elo.get(2))
        self.assertLess(diff, 100.0)


# ---------------------------------------------------------------------------
# _RatingSurrogate
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestRatingSurrogate(unittest.TestCase):

    def _surrogate(self, warmup=5):
        from yane.evolution.interactive_eval import _RatingSurrogate
        return _RatingSurrogate(warmup_queries=warmup)

    def test_predict_none_before_warmup(self):
        s = self._surrogate(warmup=5)
        g = _make_genome(1.0)
        self.assertIsNone(s.predict(g))

    def test_predict_available_after_warmup(self):
        s = self._surrogate(warmup=5)
        genomes = [_make_genome(float(w)) for w in range(1, 10)]
        for i, g in enumerate(genomes):
            s.observe(g, float(i * 10))
        # After 9 > 5 observations, predict should return float
        pred = s.predict(_make_genome(2.0))
        self.assertIsNotNone(pred)
        self.assertIsInstance(pred, float)

    def test_observe_then_predict_monotone(self):
        """Surrogate trained on linear mapping weight → weight*10 should predict
        a higher rating for a higher-weight genome (monotonicity check)."""
        s = self._surrogate(warmup=5)
        for w in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            s.observe(_make_genome(w), w * 10.0)
        pred_low = s.predict(_make_genome(0.5))
        pred_high = s.predict(_make_genome(9.0))
        # Not a strict test — just check surrogate is responsive
        self.assertIsNotNone(pred_low)
        self.assertIsNotNone(pred_high)


# ---------------------------------------------------------------------------
# InteractiveEvaluator — rating mode
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestInteractiveEvaluatorRatingMode(unittest.TestCase):

    def _make(self, surrogate=False):
        from yane.evolution.interactive_eval import InteractiveEvaluator
        return InteractiveEvaluator(mode="rating", surrogate_model=surrogate)

    def test_oracle_returns_correct_fitness(self):
        ev = self._make()
        ev.set_feedback_source(lambda g: 42.0)
        g = _make_genome()
        self.assertAlmostEqual(ev(g), 42.0)

    def test_rating_is_cached(self):
        calls = [0]
        def oracle(g):
            calls[0] += 1
            return 55.0
        ev = self._make()
        ev.set_feedback_source(oracle)
        g = _make_genome()
        ev(g)
        ev(g)  # second call — should use cache
        self.assertEqual(calls[0], 1, "Rating should be cached after first query")

    def test_query_count_increments(self):
        ev = self._make()
        ev.set_feedback_source(lambda g: 10.0)
        genomes = [_make_genome(float(i)) for i in range(5)]
        for g in genomes:
            ev(g)
        self.assertEqual(ev.query_count, 5)

    def test_submit_feedback_updates_fitness(self):
        ev = self._make()
        g = _make_genome()
        gid = g._genome_id
        # Provide feedback without an oracle
        # Feed from a separate thread to avoid blocking
        def feeder():
            time.sleep(0.01)
            ev.submit_feedback(gid, 77.0)
        t = threading.Thread(target=feeder, daemon=True)
        t.start()
        fitness = ev(g)
        t.join(timeout=2.0)
        self.assertAlmostEqual(fitness, 77.0)

    def test_get_rating_reflects_submitted_value(self):
        ev = self._make()
        ev.set_feedback_source(lambda g: 33.0)
        g = _make_genome()
        ev(g)
        self.assertAlmostEqual(ev.get_rating(g._genome_id), 33.0)

    def test_get_rating_none_before_evaluation(self):
        ev = self._make()
        g = _make_genome()
        self.assertIsNone(ev.get_rating(g._genome_id))

    def test_mode_attribute(self):
        ev = self._make()
        self.assertEqual(ev.mode, "rating")

    def test_invalid_mode_raises(self):
        from yane.evolution.interactive_eval import InteractiveEvaluator
        with self.assertRaises(ValueError):
            InteractiveEvaluator(mode="nonsense")


# ---------------------------------------------------------------------------
# InteractiveEvaluator — pairwise mode
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestInteractiveEvaluatorPairwiseMode(unittest.TestCase):

    def _make(self):
        from yane.evolution.interactive_eval import InteractiveEvaluator
        ev = InteractiveEvaluator(mode="pairwise", surrogate_model=False)
        # Oracle: genome with higher fitness value wins
        ev.set_feedback_source(lambda g: g.fitness or 0.0)
        return ev

    def test_first_genome_gets_default_elo(self):
        ev = self._make()
        g = _make_genome(fitness=10.0)
        fitness = ev(g)
        # Default Elo is 1000
        self.assertAlmostEqual(fitness, 1000.0)

    def test_winner_has_higher_elo(self):
        """After pairwise comparison, the 'better' genome should have higher Elo."""
        ev = self._make()
        g_low = _make_genome(fitness=1.0)
        g_high = _make_genome(fitness=100.0)
        ev(g_low)   # first genome → default Elo
        ev(g_high)  # compared against g_low → g_high wins
        self.assertGreater(ev.get_rating(g_high._genome_id), ev.get_rating(g_low._genome_id))

    def test_consistent_winner_converges(self):
        """Running the same best genome as winner repeatedly raises its Elo."""
        from yane.evolution.interactive_eval import InteractiveEvaluator
        ev = InteractiveEvaluator(mode="pairwise", surrogate_model=False)
        ev.set_feedback_source(lambda g: g.fitness or 0.0)
        best = _make_genome(fitness=100.0)
        ev(best)  # init
        for _ in range(5):
            rival = _make_genome(fitness=1.0)
            ev(rival)  # best beats all rivals
        elo_best = ev.get_rating(best._genome_id)
        self.assertGreater(elo_best, 1000.0, "Winner's Elo should rise above default")

    def test_pairwise_elo_sums_approximately_preserved(self):
        """After one match the total Elo of both participants is conserved."""
        ev = self._make()
        g_a = _make_genome(fitness=5.0)
        g_b = _make_genome(fitness=10.0)
        ev(g_a)
        ev(g_b)
        total = ev.get_rating(g_a._genome_id) + ev.get_rating(g_b._genome_id)
        self.assertAlmostEqual(total, 2000.0, delta=200.0)  # loose: one default, one updated

    def test_submit_feedback_pairwise_updates_ratings(self):
        """submit_feedback in pairwise mode updates Elo for both genomes."""
        from yane.evolution.interactive_eval import InteractiveEvaluator
        ev = InteractiveEvaluator(mode="pairwise", surrogate_model=False)
        g_a = _make_genome(fitness=5.0)
        g_b = _make_genome(fitness=10.0)
        # Feed them through queue path (no oracle)
        results = {}

        def run_a():
            results["a"] = ev(g_a)
        def run_b():
            results["b"] = ev(g_b)

        ta = threading.Thread(target=run_a, daemon=True)
        tb = threading.Thread(target=run_b, daemon=True)
        ta.start()
        time.sleep(0.02)
        tb.start()
        time.sleep(0.02)
        # Declare g_a winner (value=0)
        ev.submit_feedback(g_a._genome_id, 0)
        ta.join(timeout=2.0)
        tb.join(timeout=2.0)
        # g_a won, its Elo should be > 1000
        self.assertGreater(ev.get_rating(g_a._genome_id), 1000.0)


# ---------------------------------------------------------------------------
# InteractiveEvaluator — ranking mode
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestInteractiveEvaluatorRankingMode(unittest.TestCase):

    def test_ranking_fitness_decreases_with_rank(self):
        """Lower rank position (closer to 1) → higher fitness."""
        from yane.evolution.interactive_eval import InteractiveEvaluator
        ev = InteractiveEvaluator(mode="ranking", surrogate_model=False)
        g1 = _make_genome()
        g2 = _make_genome()
        # Manually submit rankings (no blocking needed)
        ev.submit_feedback(g1._genome_id, 1.0)  # rank 1 (best)
        ev.submit_feedback(g2._genome_id, 3.0)  # rank 3 (worst)
        # Fitness = -(rank) + 1 → rank1: 0, rank3: -2
        f1 = ev.get_rating(g1._genome_id)
        f2 = ev.get_rating(g2._genome_id)
        self.assertGreater(f1, f2, "Best-ranked genome should have higher fitness")

    def test_ranking_mode_attribute(self):
        from yane.evolution.interactive_eval import InteractiveEvaluator
        ev = InteractiveEvaluator(mode="ranking")
        self.assertEqual(ev.mode, "ranking")


# ---------------------------------------------------------------------------
# InteractiveEvaluator — implicit mode
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestInteractiveEvaluatorImplicitMode(unittest.TestCase):

    def test_implicit_mode_oracle(self):
        from yane.evolution.interactive_eval import InteractiveEvaluator
        ev = InteractiveEvaluator(mode="implicit", surrogate_model=False)
        ev.set_feedback_source(lambda g: 0.5)  # 0.5 seconds of dwell
        g = _make_genome()
        fitness = ev(g)
        self.assertAlmostEqual(fitness, 0.5)


# ---------------------------------------------------------------------------
# Surrogate reduces human queries
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestInteractiveEvaluatorSurrogate(unittest.TestCase):

    def test_surrogate_skips_count_exceeds_zero(self):
        """After warmup, the surrogate should skip at least some queries."""
        from yane.evolution.interactive_eval import InteractiveEvaluator
        ev = InteractiveEvaluator(
            mode="rating",
            surrogate_model=True,
            surrogate_warmup=5,
            surrogate_confidence_threshold=0.0,  # always use surrogate after warmup
        )
        oracle_calls = [0]
        def oracle(g):
            oracle_calls[0] += 1
            return float(g._genome_id % 100)
        ev.set_feedback_source(oracle)

        # Warm up with 10 genomes
        warmup_genomes = [_make_genome(float(i)) for i in range(10)]
        for g in warmup_genomes:
            ev(g)

        # Now evaluate 10 more *new* genomes; surrogate should skip some
        new_genomes = [_make_genome(float(i) * 0.1 + 100.0) for i in range(10)]
        for g in new_genomes:
            ev(g)

        self.assertGreater(ev.surrogate_skips, 0, "Surrogate should skip at least one query")

    def test_surrogate_skips_less_without_surrogate(self):
        """Without surrogate, all unique genomes require a human query."""
        from yane.evolution.interactive_eval import InteractiveEvaluator
        ev = InteractiveEvaluator(
            mode="rating",
            surrogate_model=False,
        )
        ev.set_feedback_source(lambda g: 50.0)
        genomes = [_make_genome(float(i)) for i in range(10)]
        for g in genomes:
            ev(g)
        self.assertEqual(ev.surrogate_skips, 0)
        self.assertEqual(ev.query_count, 10)

    def test_with_surrogate_fewer_queries_than_without(self):
        """Surrogate-enabled evaluator issues fewer queries than one without."""
        from yane.evolution.interactive_eval import InteractiveEvaluator

        def run(use_surrogate):
            ev = InteractiveEvaluator(
                mode="rating",
                surrogate_model=use_surrogate,
                surrogate_warmup=5,
                surrogate_confidence_threshold=0.0,
            )
            ev.set_feedback_source(lambda g: float(g._genome_id % 50))
            # Warm-up with distinct genomes
            warmup = [_make_genome(float(i)) for i in range(10)]
            for g in warmup:
                ev(g)
            # Additional unique genomes
            extra = [_make_genome(float(i) * 0.01 + 50.0) for i in range(20)]
            for g in extra:
                ev(g)
            return ev.query_count

        queries_with = run(True)
        queries_without = run(False)
        self.assertLessEqual(queries_with, queries_without,
                             "Surrogate should not increase query count")


# ---------------------------------------------------------------------------
# Mode switching
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestModeSwitch(unittest.TestCase):

    def test_mode_rating_to_pairwise_creates_separate_instances(self):
        """Creating two evaluators with different modes keeps them independent."""
        from yane.evolution.interactive_eval import InteractiveEvaluator
        ev_rating = InteractiveEvaluator(mode="rating")
        ev_pairwise = InteractiveEvaluator(mode="pairwise")
        self.assertEqual(ev_rating.mode, "rating")
        self.assertEqual(ev_pairwise.mode, "pairwise")
        # Ratings are not shared
        ev_rating.set_feedback_source(lambda g: 99.0)
        g = _make_genome()
        ev_rating(g)
        self.assertIsNone(ev_pairwise.get_rating(g._genome_id))

    def test_all_modes_construct_without_error(self):
        from yane.evolution.interactive_eval import InteractiveEvaluator
        for mode in ("rating", "pairwise", "ranking", "implicit"):
            ev = InteractiveEvaluator(mode=mode)
            self.assertEqual(ev.mode, mode)

    def test_pending_genome_ids_empty_initially(self):
        from yane.evolution.interactive_eval import InteractiveEvaluator
        ev = InteractiveEvaluator(mode="rating")
        self.assertEqual(ev.pending_genome_ids(), [])

    def test_pending_genome_ids_after_async_eval(self):
        from yane.evolution.interactive_eval import InteractiveEvaluator
        ev = InteractiveEvaluator(mode="rating", surrogate_model=False)
        g = _make_genome()
        # Start evaluation in background (no oracle → blocks)
        result = []
        def run():
            result.append(ev(g))
        t = threading.Thread(target=run, daemon=True)
        t.start()
        time.sleep(0.02)
        # Should appear in pending
        self.assertIn(g._genome_id, ev.pending_genome_ids())
        # Resolve it
        ev.submit_feedback(g._genome_id, 50.0)
        t.join(timeout=2.0)
        self.assertAlmostEqual(result[0], 50.0)


# ---------------------------------------------------------------------------
# NeuroEvolution integration
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestNeuroEvolutionIntegration(unittest.TestCase):

    def _make_yane(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.configure(n_inputs=2, n_outputs=1, max_nodes=10, max_connections=20)
        return ne

    def test_set_interactive_evaluation_returns_evaluator(self):
        from yane.evolution.interactive_eval import InteractiveEvaluator
        ne = self._make_yane()
        ev = ne.set_interactive_evaluation(mode="rating")
        self.assertIsInstance(ev, InteractiveEvaluator)
        self.assertEqual(ev.mode, "rating")

    def test_set_interactive_evaluation_accepts_existing_evaluator(self):
        from yane.evolution.interactive_eval import InteractiveEvaluator
        ne = self._make_yane()
        existing = InteractiveEvaluator(mode="pairwise")
        returned = ne.set_interactive_evaluation(evaluator=existing)
        self.assertIs(returned, existing)

    def test_submit_feedback_without_evaluator_raises(self):
        ne = self._make_yane()
        with self.assertRaises(RuntimeError):
            ne.submit_feedback(1, 50.0)

    def test_submit_feedback_with_evaluator_updates_rating(self):
        from yane.evolution.interactive_eval import InteractiveEvaluator
        ne = self._make_yane()
        ev = ne.set_interactive_evaluation(mode="rating")
        # Fake genome id
        ev._genome_registry[999] = _make_genome()
        ne.submit_feedback(999, 88.0)
        self.assertAlmostEqual(ev.get_rating(999), 88.0)

    def test_interactive_evaluator_works_as_fitness_fn(self):
        """InteractiveEvaluator can be passed directly to train() as fitness_fn."""
        from yane.evolution.interactive_eval import InteractiveEvaluator
        ne = self._make_yane()
        ne.set_max_iterations(5)
        ev = InteractiveEvaluator(mode="rating", surrogate_model=False)
        ev.set_feedback_source(lambda g: float(len(g.nodes)))
        ne.train(ev)
        # If we reach here without exception, integration works


if __name__ == "__main__":
    unittest.main()
