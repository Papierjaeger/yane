"""Tests für Self-Play / Adversarial Populations (evolution/self_play.py).

Akzeptanzkriterien:
  1. Fitness ist korrekt Nullsumme
  2. Elo-Ratings steigen in beiden Populationen bei gesundem Arms Race
  3. Tests: Nullsummen-Fitness; getrennte Spezies; Elo-Update; Pairing-Mechanismen
"""
from __future__ import annotations

import unittest

import pytest

from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_genome(weight: float = 0.5) -> Genome:
    g = Genome()
    inp = Node(NodeType.INPUT, 0); inp.activation = ActivationType.LINEAR; inp.input_index = 0
    out = Node(NodeType.OUTPUT, 1); out.activation = ActivationType.SIGMOID; out.bias = 0.0
    g.nodes.extend([inp, out]); g.input_nodes.append(inp); g.output_nodes.append(out)
    c = Connection(out, 10); c.weight = weight; inp.connections.append(c)
    g._invalidate_topology()
    return g


def _zero_sum_game(ga: Genome, gb: Genome) -> tuple[float, float]:
    """Simple game: genome with larger output wins; scores sum to 0."""
    ga.reset(); gb.reset()
    score_a = ga.forward([0.5])[0]
    score_b = gb.forward([0.5])[0]
    # Zero-sum: winner +1, loser -1, draw 0
    if score_a > score_b:
        return 1.0, -1.0
    elif score_b > score_a:
        return -1.0, 1.0
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Nullsummen-Fitness — acceptance criterion 1
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestZeroSumFitness(unittest.TestCase):

    def test_sum_of_scores_is_zero(self):
        """Game scores must sum to 0 (zero-sum invariant)."""
        ga = _make_genome(weight=0.1)
        gb = _make_genome(weight=0.9)
        score_a, score_b = _zero_sum_game(ga, gb)
        self.assertAlmostEqual(score_a + score_b, 0.0, places=10)

    def test_winner_positive_loser_negative(self):
        """Winner gets positive score; loser negative."""
        ga = _make_genome(weight=0.1)   # low weight → low score
        gb = _make_genome(weight=5.0)   # high weight → high score → gb wins
        score_a, score_b = _zero_sum_game(ga, gb)
        self.assertLess(score_a, 0.0)
        self.assertGreater(score_b, 0.0)

    def test_draw_both_zero(self):
        """Identical genomes: draw → both scores are 0."""
        g = _make_genome(weight=1.0)
        gc = g.copy()
        score_a, score_b = _zero_sum_game(g, gc)
        self.assertAlmostEqual(score_a, 0.0, places=10)
        self.assertAlmostEqual(score_b, 0.0, places=10)

    def test_apply_game_result_elo_update(self):
        """Elo must change after apply_game_result."""
        from yane.evolution.self_play import AdversarialSystem
        system = AdversarialSystem(n_populations=2)
        ga = _make_genome(weight=0.1); ga.fitness = 0.0
        gb = _make_genome(weight=0.9); gb.fitness = 0.0
        system.set_population(0, [ga])
        system.set_population(1, [gb])
        # gb wins (+1 for b, -1 for a)
        system.apply_game_result(ga, 0, gb, 1, -1.0, 1.0)
        # gb should have Elo > 1000; ga < 1000
        self.assertGreater(system.get_elo(gb), 1000.0)
        self.assertLess(system.get_elo(ga), 1000.0)

    def test_elo_sum_conserved(self):
        """After one match, total Elo change sums to 0 (shared EloRating)."""
        from yane.evolution.self_play import AdversarialSystem
        system = AdversarialSystem(n_populations=2, elo_k=32)
        ga = _make_genome(); gb = _make_genome()
        system.set_population(0, [ga]); system.set_population(1, [gb])
        elo_before_a = system.get_elo(ga)
        elo_before_b = system.get_elo(gb)
        system.apply_game_result(ga, 0, gb, 1, 1.0, -1.0)
        elo_after_a = system.get_elo(ga)
        elo_after_b = system.get_elo(gb)
        delta = (elo_after_a - elo_before_a) + (elo_after_b - elo_before_b)
        self.assertAlmostEqual(delta, 0.0, places=6,
                               msg="Total Elo change must sum to 0 (zero-sum invariant)")


# ---------------------------------------------------------------------------
# Elo-Update — acceptance criterion part 2
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestEloUpdate(unittest.TestCase):

    def test_consistent_winner_gains_elo(self):
        """Genome that always wins should have Elo > 1000."""
        from yane.evolution.self_play import AdversarialSystem
        system = AdversarialSystem(n_populations=2, pairing="round_robin")
        strong = _make_genome(weight=5.0)   # always wins
        weak_pool = [_make_genome(weight=0.1) for _ in range(5)]
        system.set_population(0, [strong])
        system.set_population(1, weak_pool)
        system.apply_zero_sum_batch(_zero_sum_game, 0, 1)
        self.assertGreater(system.get_elo(strong), 1000.0)

    def test_consistent_loser_loses_elo(self):
        """Genome that always loses should have Elo < 1000."""
        from yane.evolution.self_play import AdversarialSystem
        system = AdversarialSystem(n_populations=2, pairing="round_robin")
        weak = _make_genome(weight=0.01)
        strong_pool = [_make_genome(weight=5.0) for _ in range(5)]
        system.set_population(0, [weak])
        system.set_population(1, strong_pool)
        system.apply_zero_sum_batch(_zero_sum_game, 0, 1)
        self.assertLess(system.get_elo(weak), 1000.0)

    def test_fitness_set_to_elo(self):
        """genome.fitness must equal its current Elo after game."""
        from yane.evolution.self_play import AdversarialSystem
        system = AdversarialSystem(n_populations=2)
        ga = _make_genome(weight=0.1); gb = _make_genome(weight=0.9)
        system.set_population(0, [ga]); system.set_population(1, [gb])
        system.apply_zero_sum_batch(_zero_sum_game, 0, 1)
        self.assertAlmostEqual(ga.fitness, system.get_elo(ga))
        self.assertAlmostEqual(gb.fitness, system.get_elo(gb))


# ---------------------------------------------------------------------------
# Pairing-Mechanismen — acceptance criterion part 3
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestPairingMechanisms(unittest.TestCase):

    def _make_pops(self, n=3, weight=0.5):
        return [_make_genome(weight=weight + i * 0.1) for i in range(n)]

    def test_round_robin_all_pairs(self):
        from yane.evolution.self_play import _round_robin_pairs
        pop_a = self._make_pops(3)
        pop_b = self._make_pops(2)
        pairs = _round_robin_pairs(pop_a, pop_b)
        self.assertEqual(len(pairs), 6)  # 3 × 2

    def test_random_pairs_correct_count(self):
        import random
        from yane.evolution.self_play import _random_pairs
        pop_a = self._make_pops(5)
        pop_b = self._make_pops(5)
        rng = random.Random(0)
        pairs = _random_pairs(pop_a, pop_b, n_matches=7, rng=rng)
        self.assertEqual(len(pairs), 7)

    def test_best_vs_rest_pairs(self):
        from yane.evolution.self_play import _best_vs_rest_pairs, AdversarialSystem
        from yane.evolution.interactive_eval import EloRating
        pop_a = self._make_pops(3)
        pop_b = self._make_pops(4)
        elo_a = EloRating(); elo_b = EloRating()
        pairs = _best_vs_rest_pairs(pop_a, pop_b, elo_a, elo_b)
        # best_a vs all b (4) + all a (3) vs best_b = 7 pairs
        self.assertEqual(len(pairs), len(pop_b) + len(pop_a))

    def test_adversarial_system_round_robin(self):
        from yane.evolution.self_play import AdversarialSystem
        system = AdversarialSystem(n_populations=2, pairing="round_robin")
        pop_a = self._make_pops(3, weight=0.2)
        pop_b = self._make_pops(3, weight=0.8)
        system.set_population(0, pop_a)
        system.set_population(1, pop_b)
        system.apply_zero_sum_batch(_zero_sum_game, 0, 1)
        # After 9 games (3×3 round robin), all Elo should be updated
        for g in pop_b:
            elo = system.get_elo(g)
            self.assertGreater(elo, 1000.0, "pop_b (stronger) should have Elo > 1000")

    def test_adversarial_system_random(self):
        from yane.evolution.self_play import AdversarialSystem
        system = AdversarialSystem(n_populations=2, pairing="random", n_matches=5, seed=0)
        pop_a = self._make_pops(4)
        pop_b = self._make_pops(4)
        system.set_population(0, pop_a)
        system.set_population(1, pop_b)
        # Should not crash
        system.apply_zero_sum_batch(_zero_sum_game, 0, 1)

    def test_invalid_pairing_raises(self):
        from yane.evolution.self_play import AdversarialSystem
        with self.assertRaises(ValueError):
            AdversarialSystem(n_populations=2, pairing="invalid_strategy")

    def test_n_populations_less_than_2_raises(self):
        from yane.evolution.self_play import AdversarialSystem
        with self.assertRaises(ValueError):
            AdversarialSystem(n_populations=1)


# ---------------------------------------------------------------------------
# Arms Race Indicator
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestArmsRaceIndicator(unittest.TestCase):

    def test_arms_race_indicator_range(self):
        """Arms race indicator must be in [0, 1]."""
        from yane.evolution.self_play import AdversarialSystem
        system = AdversarialSystem(n_populations=2, pairing="random", n_matches=3, seed=1)
        pop_a = [_make_genome(weight=0.1 + i * 0.1) for i in range(4)]
        pop_b = [_make_genome(weight=0.5 + i * 0.1) for i in range(4)]
        system.set_population(0, pop_a); system.set_population(1, pop_b)
        for _ in range(5):
            system.apply_zero_sum_batch(_zero_sum_game, 0, 1)
            system.record_elo_snapshot()
        indicator = system.arms_race_indicator
        self.assertGreaterEqual(indicator, 0.0)
        self.assertLessEqual(indicator, 1.0)

    def test_arms_race_indicator_no_history(self):
        from yane.evolution.self_play import AdversarialSystem
        system = AdversarialSystem(n_populations=2)
        self.assertAlmostEqual(system.arms_race_indicator, 0.0)

    def test_adversarial_result_arms_race(self):
        """AdversarialResult.arms_race_indicator works correctly."""
        from yane.evolution.self_play import AdversarialResult
        result = AdversarialResult(
            populations=[[_make_genome()], [_make_genome()]],
            elo_histories=[[1000, 1010, 1020], [1000, 1015, 1025]],
            n_generations=3,
        )
        # Both populations rise in gen 1→2 and 2→3 → indicator = 1.0
        self.assertAlmostEqual(result.arms_race_indicator, 1.0)


# ---------------------------------------------------------------------------
# train_adversarial() standalone
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestTrainAdversarial(unittest.TestCase):

    def test_returns_adversarial_result(self):
        from yane.evolution.self_play import train_adversarial, AdversarialResult
        pop_a = [_make_genome(weight=0.1 * i) for i in range(1, 5)]
        pop_b = [_make_genome(weight=0.5 + 0.1 * i) for i in range(4)]
        result = train_adversarial(
            populations=[pop_a, pop_b],
            game_fn=_zero_sum_game,
            n_generations=3,
            pairing="random",
            n_matches=3,
            seed=0,
        )
        self.assertIsInstance(result, AdversarialResult)

    def test_best_genome_accessible(self):
        from yane.evolution.self_play import train_adversarial
        pop_a = [_make_genome(weight=0.1 * i) for i in range(1, 5)]
        pop_b = [_make_genome(weight=0.5 + 0.1 * i) for i in range(4)]
        result = train_adversarial(
            [pop_a, pop_b], _zero_sum_game, n_generations=2, seed=0
        )
        best = result.best_genome(0)
        self.assertIsInstance(best, Genome)


# ---------------------------------------------------------------------------
# NeuroEvolution integration
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestNeuroEvolutionIntegration(unittest.TestCase):

    def test_set_adversarial_populations_returns_system(self):
        import yane
        from yane.evolution.self_play import AdversarialSystem
        ne = yane.NeuroEvolution()
        system = ne.set_adversarial_populations(n_populations=2)
        self.assertIsInstance(system, AdversarialSystem)

    def test_train_adversarial_requires_setup(self):
        import yane
        ne = yane.NeuroEvolution()
        ne.configure(n_inputs=1, n_outputs=1, max_nodes=5)
        with self.assertRaises(RuntimeError):
            ne.train_adversarial(lambda a, b: (1.0, -1.0), n_generations=1)

    def test_train_adversarial_no_crash(self):
        import yane
        ne = yane.NeuroEvolution(seed=0)
        ne.configure(n_inputs=1, n_outputs=1, max_nodes=5, max_connections=10)
        ne.set_adversarial_populations(n_populations=2, pairing="random", n_matches=2)
        result = ne.train_adversarial(
            _zero_sum_game, n_generations=2, pop_size=6
        )
        self.assertGreater(result.n_generations, 0)

    def test_yane_exports(self):
        import yane
        self.assertTrue(hasattr(yane, "AdversarialSystem"))
        self.assertTrue(hasattr(yane, "AdversarialResult"))
        self.assertTrue(hasattr(yane, "train_adversarial"))


if __name__ == "__main__":
    unittest.main()
