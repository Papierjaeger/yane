"""Interactive / Human-in-the-Loop evaluation for YANE.

Provides :class:`InteractiveEvaluator` — a callable fitness function that
collects human feedback (ratings, pairwise comparisons, rankings) and
optionally uses a lightweight linear surrogate to reduce the number of
required human queries.

Modes
-----
``"rating"``
    User assigns a score (0–100) to each genome.  The score is used directly
    as fitness.

``"pairwise"``
    User picks the preferred genome from each pair.  Elo ratings drive fitness.
    To compare a genome, a known-rated genome is chosen as opponent.

``"ranking"``
    User orders K genomes by preference.  Fitness = K − rank + 1 (best = K).

``"implicit"``
    Dwell-time (seconds spent inspecting) is the implicit fitness signal.

Usage (synchronous / programmatic)
-----------------------------------
::

    eval = InteractiveEvaluator(mode="rating")

    # Attach a feedback oracle (used by tests and programmatic callers).
    eval.set_feedback_source(lambda g: some_oracle(g))

    yane.set_interactive_evaluation(eval)
    yane.train(eval)

Usage (GUI / async)
-------------------
Call ``submit_feedback(genome_id, value)`` from the GUI thread.  The fitness
function blocks until feedback arrives (via ``threading.Event``).
"""
from __future__ import annotations

import math
import threading
from collections import deque
from typing import Callable, Literal

from yane.core.genome import Genome
from yane.evolution.landscape import genome_descriptor_vector


# ---------------------------------------------------------------------------
# Elo rating system
# ---------------------------------------------------------------------------

class EloRating:
    """Elo rating calculator for pairwise genome comparisons.

    Parameters
    ----------
    k_factor : float
        Maximum rating change per match (standard chess K = 32).
    default_rating : float
        Starting Elo for every new genome.
    """

    def __init__(self, k_factor: float = 32.0, default_rating: float = 1000.0) -> None:
        self.k_factor = k_factor
        self.default_rating = default_rating
        self._ratings: dict[int, float] = {}

    def get(self, genome_id: int) -> float:
        return self._ratings.get(genome_id, self.default_rating)

    def update(self, winner_id: int, loser_id: int) -> None:
        """Apply a pairwise result: *winner_id* beat *loser_id*."""
        r_w = self.get(winner_id)
        r_l = self.get(loser_id)
        expected_w = 1.0 / (1.0 + 10.0 ** ((r_l - r_w) / 400.0))
        expected_l = 1.0 - expected_w
        self._ratings[winner_id] = r_w + self.k_factor * (1.0 - expected_w)
        self._ratings[loser_id] = r_l + self.k_factor * (0.0 - expected_l)

    def all_rated(self) -> list[int]:
        return list(self._ratings.keys())


# ---------------------------------------------------------------------------
# Linear rating surrogate
# ---------------------------------------------------------------------------

class _RatingSurrogate:
    """Lightweight linear surrogate for predicting human ratings.

    Trained on ``(genome_descriptor_vector, rating)`` pairs.  Predictions are
    only trusted when the training buffer is large enough
    (``>= warmup_queries``) and the confidence is above ``min_confidence``.
    """

    def __init__(self, warmup_queries: int = 10) -> None:
        self.warmup_queries = warmup_queries
        self._features: deque[list[float]] = deque(maxlen=500)
        self._targets: deque[float] = deque(maxlen=500)
        self._weights: list[float] | None = None
        self._bias: float = 0.0
        self._queries_saved: int = 0

    def observe(self, genome: Genome, rating: float) -> None:
        features = genome_descriptor_vector(genome)
        self._features.append(features)
        self._targets.append(rating)
        if len(self._features) >= self.warmup_queries:
            self._fit()

    def predict(self, genome: Genome) -> float | None:
        """Return a prediction or *None* if the model is not ready."""
        if self._weights is None:
            return None
        features = genome_descriptor_vector(genome)
        return sum(w * f for w, f in zip(self._weights, features)) + self._bias

    def _fit(self) -> None:
        """Ordinary least-squares via gradient descent (no scipy dependency)."""
        xs = list(self._features)
        ys = list(self._targets)
        n = len(xs)
        dim = len(xs[0])
        if self._weights is None:
            self._weights = [0.0] * dim
        lr = 1e-3
        for _ in range(200):
            grad_w = [0.0] * dim
            grad_b = 0.0
            for x, y in zip(xs, ys):
                pred = sum(w * xi for w, xi in zip(self._weights, x)) + self._bias
                err = pred - y
                for j in range(dim):
                    grad_w[j] += err * x[j] / n
                grad_b += err / n
            self._weights = [w - lr * g for w, g in zip(self._weights, grad_w)]
            self._bias -= lr * grad_b


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

FeedbackMode = Literal["rating", "pairwise", "ranking", "implicit"]


class InteractiveEvaluator:
    """Human-in-the-loop fitness function.

    Acts as a callable ``(genome) -> float`` suitable for
    ``NeuroEvolution.train()``.  Human feedback is collected via
    :meth:`submit_feedback` or a synchronous oracle set via
    :meth:`set_feedback_source`.

    Parameters
    ----------
    mode : str
        Feedback collection mode (``"rating"``, ``"pairwise"``,
        ``"ranking"``, ``"implicit"``).
    surrogate_model : bool
        Enable linear surrogate to predict ratings and skip human queries
        for genomes the surrogate is confident about.
    surrogate_warmup : int
        Number of real human queries before the surrogate activates.
    surrogate_confidence_threshold : float
        Normalised confidence [0, 1]; only skip queries above this.
    """

    def __init__(
        self,
        mode: FeedbackMode = "rating",
        surrogate_model: bool = True,
        surrogate_warmup: int = 10,
        surrogate_confidence_threshold: float = 0.7,
    ) -> None:
        if mode not in ("rating", "pairwise", "ranking", "implicit"):
            raise ValueError(f"Unknown mode: {mode!r}")

        self.mode: FeedbackMode = mode
        self.surrogate_model = surrogate_model
        self._confidence_threshold = surrogate_confidence_threshold

        # Human ratings: genome_id -> fitness
        self._ratings: dict[int, float] = {}
        # Pending query events: genome_id -> Event (GUI/async path)
        self._pending: dict[int, threading.Event] = {}
        # Lock protecting _ratings and _pending
        self._lock = threading.Lock()

        # Elo (used in pairwise mode only)
        self.elo = EloRating()

        # Surrogate
        self._surrogate = _RatingSurrogate(warmup_queries=surrogate_warmup) if surrogate_model else None

        # Synchronous oracle (set for tests / programmatic use)
        self._feedback_source: Callable[[Genome], float] | None = None
        # For pairwise oracle: (genome_a, genome_b) -> 0 (a wins) or 1 (b wins)
        self._compare_source: Callable[[Genome, int], int] | None = None

        # Metrics
        self._query_count: int = 0   # human queries issued
        self._surrogate_skips: int = 0  # queries skipped via surrogate

        # Genome registry (for pairwise: need to look up genomes by id)
        self._genome_registry: dict[int, Genome] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def query_count(self) -> int:
        """Total human queries issued (real evaluations, not surrogate)."""
        return self._query_count

    @property
    def surrogate_skips(self) -> int:
        """Total queries skipped via the surrogate model."""
        return self._surrogate_skips

    def set_feedback_source(
        self,
        fn: Callable[[Genome], float],
        compare_fn: Callable[[Genome, int], int] | None = None,
    ) -> None:
        """Attach a synchronous oracle for programmatic / test use.

        Parameters
        ----------
        fn :
            For modes ``"rating"``, ``"ranking"``, ``"implicit"``:
            ``fn(genome) -> float``.
        compare_fn :
            For mode ``"pairwise"``:
            ``compare_fn(genome_a, opponent_genome) -> int``
            where 0 = genome_a wins, 1 = opponent wins.
            If *None*, *fn* is used on both genomes and the higher score wins.
        """
        self._feedback_source = fn
        self._compare_source = compare_fn

    def submit_feedback(self, genome_id: int, value: float) -> None:
        """Record feedback from the GUI or external caller.

        Parameters
        ----------
        genome_id :
            The ``_genome_id`` attribute of the genome being rated.
        value :
            For ``"rating"`` / ``"implicit"``: fitness score (0–100).
            For ``"pairwise"``: winner indicator — 0 if *genome_id* won,
            1 if the opponent won (the pending comparison partner).
            For ``"ranking"``: rank position (1 = best, K = worst).
        """
        with self._lock:
            if self.mode == "pairwise":
                self._apply_pairwise_result(genome_id, int(value))
            elif self.mode == "ranking":
                self._ratings[genome_id] = float(-value + 1)
            else:
                self._ratings[genome_id] = float(value)
            event = self._pending.pop(genome_id, None)

        if event is not None:
            event.set()

        # Teach the surrogate (use registered genome if available)
        if self._surrogate is not None:
            genome = self._genome_registry.get(genome_id)
            if genome is not None:
                with self._lock:
                    fitness = self._ratings.get(genome_id, 0.0)
                self._surrogate.observe(genome, fitness)

    def get_rating(self, genome_id: int) -> float | None:
        """Return the current rating for *genome_id*, or *None* if unknown."""
        with self._lock:
            return self._ratings.get(genome_id)

    def pending_genome_ids(self) -> list[int]:
        """Return genome IDs currently waiting for feedback."""
        with self._lock:
            return list(self._pending.keys())

    # ------------------------------------------------------------------
    # Callable fitness function interface
    # ------------------------------------------------------------------

    def __call__(self, genome: Genome) -> float:
        gid = genome._genome_id
        self._genome_registry[gid] = genome

        # 1. Already rated — return cached fitness
        with self._lock:
            cached = self._ratings.get(gid)
        if cached is not None:
            return cached

        # 2. Surrogate prediction (if confident enough)
        if self._surrogate is not None:
            pred = self._surrogate.predict(genome)
            if pred is not None:
                # Scale confidence by training size
                n = len(self._surrogate._features)
                warmup = self._surrogate.warmup_queries
                confidence = min(1.0, (n - warmup) / max(1, warmup) * 0.5 + 0.5) if n >= warmup else 0.0
                if confidence >= self._confidence_threshold:
                    self._surrogate_skips += 1
                    return pred

        # 3. Use synchronous oracle if available
        if self._feedback_source is not None:
            return self._query_oracle(genome)

        # 4. Block until GUI provides feedback
        return self._wait_for_feedback(genome)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query_oracle(self, genome: Genome) -> float:
        self._query_count += 1
        gid = genome._genome_id

        if self.mode == "pairwise":
            rated_ids = self.elo.all_rated()
            if not rated_ids:
                # First genome: assign default Elo, no comparison needed
                fitness = self.elo.default_rating
                self.elo._ratings[gid] = fitness  # register so next genome can find it
                with self._lock:
                    self._ratings[gid] = fitness
                if self._surrogate is not None:
                    self._surrogate.observe(genome, fitness)
                return fitness

            # Pick the most-recently-rated genome as opponent
            opponent_id = rated_ids[-1]
            opponent = self._genome_registry.get(opponent_id)

            if self._compare_source is not None:
                winner = self._compare_source(genome, opponent or genome)
            else:
                # Fall back: use feedback_source as scorer
                assert self._feedback_source is not None
                score_a = self._feedback_source(genome)
                if opponent is not None:
                    score_b = self._feedback_source(opponent)
                    winner = 0 if score_a >= score_b else 1
                else:
                    winner = 0

            if winner == 0:
                self.elo.update(gid, opponent_id)
            else:
                self.elo.update(opponent_id, gid)

            fitness = self.elo.get(gid)
            with self._lock:
                self._ratings[gid] = fitness
                # Also refresh opponent's cached rating so repeated comparisons
                # against the same genome see its current Elo.
                self._ratings[opponent_id] = self.elo.get(opponent_id)
            if self._surrogate is not None:
                self._surrogate.observe(genome, fitness)
            return fitness

        else:
            fitness = float(self._feedback_source(genome))
            with self._lock:
                self._ratings[gid] = fitness
            if self._surrogate is not None:
                self._surrogate.observe(genome, fitness)
            return fitness

    def _wait_for_feedback(self, genome: Genome) -> float:
        gid = genome._genome_id
        event = threading.Event()
        with self._lock:
            if gid in self._ratings:
                return self._ratings[gid]
            self._pending[gid] = event
        self._query_count += 1
        event.wait()  # block until submit_feedback() is called
        with self._lock:
            return self._ratings.get(gid, 0.0)

    def _apply_pairwise_result(self, genome_id: int, winner: int) -> None:
        """Update Elo given a pairwise result (called from submit_feedback)."""
        # Find the most recent pending pair partner
        pending_ids = [k for k in self._pending if k != genome_id]
        opponent_id = pending_ids[-1] if pending_ids else None

        if opponent_id is None:
            # No pending opponent — just record the raw value as Elo proxy
            self._ratings[genome_id] = self.elo.get(genome_id)
            return

        if winner == 0:
            self.elo.update(genome_id, opponent_id)
        else:
            self.elo.update(opponent_id, genome_id)

        self._ratings[genome_id] = self.elo.get(genome_id)
        self._ratings[opponent_id] = self.elo.get(opponent_id)
