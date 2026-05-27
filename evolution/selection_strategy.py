"""Selection strategy plugin interface and built-in implementations.

All strategies share the same protocol:

    pick(candidates, score) -> Genome

The *candidates* list is the pre-filtered pool (species members or full
population); *score* is the composite score function already computed by the
population (shared_fitness × offspring_factor × novelty × efficiency × …).
Strategies own only the *algorithm* — the scoring formula stays in Population.

Default behaviour (no plugin set) is identical to the classic tournament
selection used in NEAT-Python: pick 3 random candidates, return the best.
"""
from __future__ import annotations

import random
from typing import Callable, Protocol, runtime_checkable

from yane.core.genome import Genome


@runtime_checkable
class SelectionStrategy(Protocol):
    """Protocol satisfied by any object that implements *pick*."""

    def pick(
        self,
        candidates: list[Genome],
        score: Callable[[Genome], float],
    ) -> Genome:
        """Return one genome from *candidates* using the given *score* key."""
        ...


# ── Built-in strategies ────────────────────────────────────────────────────────


class TournamentSelection:
    """Classic k-tournament: sample *k* at random, return the highest-scoring.

    This is the default and preserves the original NEAT behaviour.
    """

    def __init__(self, k: int = 3) -> None:
        self.k = k

    def pick(
        self,
        candidates: list[Genome],
        score: Callable[[Genome], float],
    ) -> Genome:
        k = min(self.k, len(candidates))
        sampled = random.sample(candidates, k)
        return max(sampled, key=score)

    def __repr__(self) -> str:
        return f"TournamentSelection(k={self.k})"


class ElitistSelection:
    """Restrict selection to the top *top_frac* of the pool, then sample uniformly.

    Ensures parents are always among the best performers; at *top_frac=1.0*
    it degenerates to uniform random selection.
    """

    def __init__(self, top_frac: float = 0.2) -> None:
        self.top_frac = top_frac

    def pick(
        self,
        candidates: list[Genome],
        score: Callable[[Genome], float],
    ) -> Genome:
        n = max(1, int(len(candidates) * self.top_frac))
        top = sorted(candidates, key=score, reverse=True)[:n]
        return random.choice(top)

    def __repr__(self) -> str:
        return f"ElitistSelection(top_frac={self.top_frac})"


class FitnessProportional:
    """Roulette-wheel (fitness-proportional) selection.

    Each genome is chosen with probability proportional to its score.
    Scores are clipped to zero before normalisation so negative values
    don't invert the distribution.
    """

    def pick(
        self,
        candidates: list[Genome],
        score: Callable[[Genome], float],
    ) -> Genome:
        scores = [max(0.0, score(g)) for g in candidates]
        total = sum(scores)
        if total <= 0.0:
            return random.choice(candidates)
        threshold = random.uniform(0.0, total)
        cumulative = 0.0
        for g, s in zip(candidates, scores):
            cumulative += s
            if cumulative >= threshold:
                return g
        return candidates[-1]

    def __repr__(self) -> str:
        return "FitnessProportional()"


class RankSelection:
    """Rank-based selection: probability proportional to rank, not raw score.

    Avoids the over-selection of runaway super-individuals while preserving
    selection pressure when fitness values span many orders of magnitude.
    """

    def pick(
        self,
        candidates: list[Genome],
        score: Callable[[Genome], float],
    ) -> Genome:
        ranked = sorted(candidates, key=score)  # worst first
        n = len(ranked)
        # weight[i] = i+1, so the best genome has weight n
        total = n * (n + 1) // 2
        threshold = random.uniform(0.0, total)
        cumulative = 0.0
        for rank, g in enumerate(ranked, start=1):
            cumulative += rank
            if cumulative >= threshold:
                return g
        return ranked[-1]

    def __repr__(self) -> str:
        return "RankSelection()"


class NoveltyOnlySelection:
    """Greedy novelty selection: always pick the highest-scoring candidate.

    Useful for open-ended exploration — the *score* function already
    incorporates the novelty weight set on the population, so this simply
    removes the stochasticity of tournament sampling.
    """

    def pick(
        self,
        candidates: list[Genome],
        score: Callable[[Genome], float],
    ) -> Genome:
        return max(candidates, key=score)

    def __repr__(self) -> str:
        return "NoveltyOnlySelection()"
