"""Self-Play / Adversarial Populations — Kompetitive Co-Evolution für YANE.

Teilt die Population in N gegnerische Sub-Populationen.  Jedes Genome
bekommt ein Elo-Rating; Fitness ist Nullsumme (was A gewinnt, verliert B).

Kernkonzepte
------------
**Zero-Sum Fitness:**
Wenn Genome A gegen B antritt und ``score_a > score_b``:
  - A gewinnt, B verliert
  - Elo-Update nach Standard-Formel (K=32)
  - Fitness(A) = Elo(A), Fitness(B) = Elo(B)

**Pairing-Strategien:**
``"round_robin"``  — jedes Genome tritt gegen jedes andere Sub-Populationsgen einmal an (alle Paarungen erschöpfend).
``"random"``       — zufällig sampled, ``n_matches`` Spiele pro Generation.
``"best_vs_rest"`` — das Beste jeder Population tritt gegen alle anderen an.

**Arms Race Indicator:**
Steigt die mittlere Elo in BEIDEN Populationen über die letzten Generationen,
liegt ein gesunder Wettrüstungs-Effekt vor.

Integration::

    yane.set_adversarial_populations(n_populations=2, pairing="round_robin")
    result = yane.train_adversarial(game_fn, n_generations=100)
    print(result.arms_race_indicator)
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome

# Reuse EloRating from interactive_eval — single source of truth
from yane.evolution.interactive_eval import EloRating


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AdversarialResult:
    """Result of one ``train_adversarial()`` run."""

    populations: list[list["Genome"]]
    """Final N sub-populations (sorted by Elo, best first)."""

    elo_histories: list[list[float]]
    """Mean Elo per generation per sub-population."""

    n_generations: int
    """Generations trained."""

    @property
    def arms_race_indicator(self) -> float:
        """Fraction of generations where mean Elo rose in ALL populations.

        1.0 = perfect arms race; 0.0 = no co-improvement.
        """
        if not self.elo_histories or len(self.elo_histories[0]) < 2:
            return 0.0
        n_gen = len(self.elo_histories[0])
        n_racing = 0
        for g in range(1, n_gen):
            all_rose = all(
                self.elo_histories[p][g] > self.elo_histories[p][g - 1]
                for p in range(len(self.elo_histories))
            )
            if all_rose:
                n_racing += 1
        return n_racing / (n_gen - 1)

    def best_genome(self, pop_id: int = 0) -> "Genome":
        return self.populations[pop_id][0]


# ---------------------------------------------------------------------------
# Pairing helpers
# ---------------------------------------------------------------------------

def _round_robin_pairs(
    pop_a: list["Genome"],
    pop_b: list["Genome"],
) -> list[tuple["Genome", "Genome"]]:
    """All cross-population pairs (|pop_a| × |pop_b|)."""
    return [(a, b) for a in pop_a for b in pop_b]


def _random_pairs(
    pop_a: list["Genome"],
    pop_b: list["Genome"],
    n_matches: int,
    rng: random.Random,
) -> list[tuple["Genome", "Genome"]]:
    """N randomly sampled cross-population pairs."""
    pairs = []
    for _ in range(n_matches):
        a = rng.choice(pop_a)
        b = rng.choice(pop_b)
        pairs.append((a, b))
    return pairs


def _best_vs_rest_pairs(
    pop_a: list["Genome"],
    pop_b: list["Genome"],
    elo_a: EloRating,
    elo_b: EloRating | None = None,
) -> list[tuple["Genome", "Genome"]]:
    """Best genome from each population plays all genomes in the other."""
    if elo_b is None:
        elo_b = elo_a
    best_a = max(pop_a, key=lambda g: elo_a.get(g._genome_id))
    best_b = max(pop_b, key=lambda g: elo_b.get(g._genome_id))
    pairs: list[tuple[Genome, Genome]] = []
    for b in pop_b:
        pairs.append((best_a, b))
    for a in pop_a:
        pairs.append((a, best_b))
    return pairs


# ---------------------------------------------------------------------------
# Core adversarial system
# ---------------------------------------------------------------------------

class AdversarialSystem:
    """Manages N competing sub-populations with zero-sum Elo fitness.

    Parameters
    ----------
    n_populations :
        Number of competing sub-populations (≥ 2).
    pairing :
        Pairing strategy: ``"round_robin"``, ``"random"``, or
        ``"best_vs_rest"``.
    n_matches :
        Matches per genome pair per generation (used by ``"random"`` pairing).
    elo_k :
        Elo K-factor (default 32).
    seed :
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        n_populations: int = 2,
        pairing: str = "round_robin",
        n_matches: int = 10,
        elo_k: float = 32.0,
        seed: int | None = None,
    ) -> None:
        if n_populations < 2:
            raise ValueError("n_populations must be ≥ 2")
        if pairing not in ("round_robin", "random", "best_vs_rest"):
            raise ValueError(f"Unknown pairing: {pairing!r}")

        self.n_populations = n_populations
        self.pairing = pairing
        self.n_matches = n_matches
        self._rng = random.Random(seed)

        # Single shared EloRating covering all genomes across all populations.
        # Using one instance preserves the zero-sum Elo invariant globally.
        self._elo = EloRating(k_factor=elo_k)
        # Current populations (set by train_adversarial or manually)
        self._populations: list[list["Genome"]] = [[] for _ in range(n_populations)]
        # Per-generation mean Elo history (for arms_race_indicator)
        self._elo_history: list[list[float]] = [[] for _ in range(n_populations)]

    # ------------------------------------------------------------------
    # Population management
    # ------------------------------------------------------------------

    def set_population(self, pop_id: int, genomes: list["Genome"]) -> None:
        """Assign genomes to sub-population *pop_id*."""
        self._populations[pop_id] = list(genomes)

    def get_population(self, pop_id: int) -> list["Genome"]:
        return list(self._populations[pop_id])

    def get_elo(self, genome: "Genome", pop_id: int = 0) -> float:
        return self._elo.get(genome._genome_id)

    # ------------------------------------------------------------------
    # Zero-sum game application
    # ------------------------------------------------------------------

    def apply_game_result(
        self,
        genome_a: "Genome",
        pop_a: int,
        genome_b: "Genome",
        pop_b: int,
        score_a: float,
        score_b: float,
    ) -> None:
        """Apply one game result: update Elo ratings and genome fitness.

        Zero-sum: ``score_a + score_b`` must sum to a constant (or 0).
        The winner (higher score) gains Elo; the loser loses Elo.

        Parameters
        ----------
        genome_a, genome_b :
            The two competing genomes.
        pop_a, pop_b :
            Sub-population indices of each genome.
        score_a, score_b :
            Raw performance scores from the game function.
        """
        id_a = genome_a._genome_id
        id_b = genome_b._genome_id

        if score_a > score_b:
            self._elo.update(winner_id=id_a, loser_id=id_b)
        elif score_b > score_a:
            self._elo.update(winner_id=id_b, loser_id=id_a)
        # Draw: no Elo change

        # Apply Elo as fitness (shared rating)
        genome_a.fitness = self._elo.get(id_a)
        genome_b.fitness = self._elo.get(id_b)

    def apply_zero_sum_batch(
        self,
        game_fn: Callable[["Genome", "Genome"], tuple[float, float]],
        pop_a: int,
        pop_b: int,
    ) -> None:
        """Run all pairings between sub-populations *pop_a* and *pop_b*.

        Parameters
        ----------
        game_fn :
            ``(genome_a, genome_b) → (score_a, score_b)`` — scores must be
            zero-sum: ``score_a + score_b = constant``.
        pop_a, pop_b :
            Indices of the two competing sub-populations.
        """
        genomes_a = self._populations[pop_a]
        genomes_b = self._populations[pop_b]
        if not genomes_a or not genomes_b:
            return

        if self.pairing == "round_robin":
            pairs = _round_robin_pairs(genomes_a, genomes_b)
        elif self.pairing == "random":
            pairs = _random_pairs(genomes_a, genomes_b, self.n_matches, self._rng)
        else:  # best_vs_rest
            pairs = _best_vs_rest_pairs(genomes_a, genomes_b, self._elo, self._elo)

        for ga, gb in pairs:
            try:
                score_a, score_b = game_fn(ga, gb)
                self.apply_game_result(ga, pop_a, gb, pop_b,
                                       float(score_a), float(score_b))
            except Exception:
                pass

    def record_elo_snapshot(self) -> None:
        """Record mean Elo per sub-population for this generation."""
        for p in range(self.n_populations):
            pops = self._populations[p]
            if pops:
                mean_elo = sum(self._elo.get(g._genome_id) for g in pops) / len(pops)
            else:
                mean_elo = 1000.0
            self._elo_history[p].append(mean_elo)

    @property
    def elo_history(self) -> list[list[float]]:
        return [list(h) for h in self._elo_history]

    @property
    def arms_race_indicator(self) -> float:
        """Fraction of generations where mean Elo rose in ALL sub-populations."""
        if not self._elo_history or len(self._elo_history[0]) < 2:
            return 0.0
        n_gen = len(self._elo_history[0])
        n_racing = 0
        for g in range(1, n_gen):
            all_rose = all(
                self._elo_history[p][g] > self._elo_history[p][g - 1]
                for p in range(self.n_populations)
            )
            if all_rose:
                n_racing += 1
        return n_racing / (n_gen - 1)


# ---------------------------------------------------------------------------
# Standalone train function
# ---------------------------------------------------------------------------

def train_adversarial(
    populations: list[list["Genome"]],
    game_fn: Callable[["Genome", "Genome"], tuple[float, float]],
    mutation_fn: Callable[["Genome"], "Genome"] | None = None,
    n_generations: int = 100,
    n_survivors: int | None = None,
    pairing: str = "round_robin",
    n_matches: int = 10,
    elo_k: float = 32.0,
    seed: int | None = None,
) -> AdversarialResult:
    """Evolve N competing populations via self-play.

    Each generation:
    1. Play all cross-population games via *game_fn*.
    2. Update Elo ratings (zero-sum).
    3. Keep top *n_survivors* by Elo; fill remainder with mutated copies.

    Parameters
    ----------
    populations :
        List of N genome lists (one per sub-population).
    game_fn :
        ``(genome_a, genome_b) → (score_a, score_b)`` — zero-sum scores.
    mutation_fn :
        Produce an offspring from a parent genome.  When *None*, uses
        ``genome.copy()`` (no mutation — pure Elo).
    n_generations :
        Training iterations.
    n_survivors :
        Elites kept per generation.  Defaults to half the population.
    pairing, n_matches, elo_k, seed :
        Passed to :class:`AdversarialSystem`.

    Returns
    -------
    AdversarialResult
    """
    system = AdversarialSystem(
        n_populations=len(populations),
        pairing=pairing,
        n_matches=n_matches,
        elo_k=elo_k,
        seed=seed,
    )
    for p, pops in enumerate(populations):
        system.set_population(p, pops)

    rng = random.Random(seed)
    n_pop = len(populations[0]) if populations else 0
    survivors = n_survivors or max(1, n_pop // 2)

    for gen in range(n_generations):
        # All cross-population matchups
        for a in range(len(populations)):
            for b in range(a + 1, len(populations)):
                system.apply_zero_sum_batch(game_fn, a, b)

        # Record Elo snapshot
        system.record_elo_snapshot()

        # Evolve each sub-population: keep top-k, fill with offspring
        for p in range(len(populations)):
            pops = system.get_population(p)
            elo = system._elo
            sorted_pop = sorted(pops, key=lambda g: elo.get(g._genome_id), reverse=True)
            top = sorted_pop[:survivors]
            # Fill remainder with mutated copies of the top
            while len(top) < n_pop:
                parent = rng.choice(sorted_pop[:max(1, survivors)])
                if mutation_fn is not None:
                    child = mutation_fn(parent)
                else:
                    child = parent.copy()
                top.append(child)
            system.set_population(p, top)

    final_pops = [
        sorted(system.get_population(p),
               key=lambda g: system._elo.get(g._genome_id), reverse=True)
        for p in range(len(populations))
    ]
    return AdversarialResult(
        populations=final_pops,
        elo_histories=system.elo_history,
        n_generations=n_generations,
    )
