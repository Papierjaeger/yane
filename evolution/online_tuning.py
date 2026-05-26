"""Online hyperparameter tuning via UCB1 bandit.

Tunes discrete hyperparameter values during a single training run by treating
each (param → value) combination as a bandit arm and using fitness delta as
reward.
"""
from __future__ import annotations

import math
import random
from typing import Any


class UCB1Bandit:
    """UCB1 multi-armed bandit for discrete hyperparameter candidates.

    Parameters
    ----------
    param_name : str
        Name of the hyperparameter being tuned (e.g. ``"mutation_rate"``).
    candidates : list[Any]
        Discrete values to try (each value is one arm).
    exploration_fraction : float
        Fraction of total iterations spent in pure exploration
        (default 0.2 = first 20 % of iterations).
    """

    def __init__(
        self,
        param_name: str,
        candidates: list[Any],
        exploration_fraction: float = 0.2,
    ) -> None:
        if not candidates:
            raise ValueError("Must provide at least one candidate value")
        if not 0.0 <= exploration_fraction <= 1.0:
            raise ValueError("exploration_fraction must be in [0, 1]")

        self.param_name = param_name
        self.candidates = list(candidates)
        self.exploration_fraction = exploration_fraction

        # Per-arm stats
        self.counts: list[int] = [0] * len(candidates)
        self.rewards: list[float] = [0.0] * len(candidates)

        # State
        self.total_iterations: int = 0
        self.total_attempts: int = 0
        self._rng = random.Random()
        self._exploration_mode: bool = True

    def select(self, iteration: int, max_iterations: int) -> Any:
        """Select a candidate value via UCB1 (exploration → exploitation).

        During the exploration phase (first ``exploration_fraction`` of
        iterations), values are chosen uniformly at random.  Afterwards
        the arm with the highest UCB1 score is chosen.

        Parameters
        ----------
        iteration : int
            Current training iteration.
        max_iterations : int
            Total expected iterations (used to compute exploration cutoff).

        Returns
        -------
        Any
            The chosen candidate value.
        """
        self._last_idx = self._select_arm(iteration, max_iterations)
        self.counts[self._last_idx] += 1
        self.total_attempts += 1
        return self.candidates[self._last_idx]

    def update(self, reward: float) -> None:
        """Record the fitness-delta reward for the last selected arm."""
        if hasattr(self, "_last_idx") and self._last_idx is not None:
            self.rewards[self._last_idx] += float(reward)

    def _select_arm(self, iteration: int, max_iterations: int) -> int:
        """Internal: return the chosen arm index."""
        explore_until = int(max_iterations * self.exploration_fraction)
        self._exploration_mode = iteration < explore_until

        if self._exploration_mode:
            return self._rng.randint(0, len(self.candidates) - 1)

        total_pulls = sum(self.counts)
        if total_pulls == 0:
            return self._rng.randint(0, len(self.candidates) - 1)

        log_t = math.log(total_pulls + 1)
        scores = [
            (self.rewards[i] / max(self.counts[i], 1)
             + math.sqrt(2.0 * log_t / max(self.counts[i], 1)))
            for i in range(len(self.candidates))
        ]
        return max(range(len(scores)), key=lambda i: scores[i])

    def get_diagnostics(self) -> dict:
        """Return current bandit state for diagnostics."""
        total = sum(self.counts) or 1
        return {
            f"bandit_{self.param_name}_counts": list(self.counts),
            f"bandit_{self.param_name}_rewards": [round(r, 4) for r in self.rewards],
            f"bandit_{self.param_name}_candidates": list(self.candidates),
            f"bandit_{self.param_name}_exploration": self._exploration_mode,
            f"bandit_{self.param_name}_best_idx": max(
                range(len(self.rewards)), key=lambda i: self.rewards[i] / max(self.counts[i], 1)
            ) if sum(self.counts) > 0 else 0,
        }

    def best_value(self) -> Any:
        """Return the candidate with the highest average reward so far."""
        if sum(self.counts) == 0:
            return self.candidates[0]
        best_idx = max(
            range(len(self.rewards)),
            key=lambda i: self.rewards[i] / max(self.counts[i], 1),
        )
        return self.candidates[best_idx]

    def arm_swap_count(self) -> int:
        """Count how many times the selected arm changed (exploration rate proxy)."""
        # Only meaningful if we track history; simplified version:
        return sum(1 for c in self.counts if c > 0)
