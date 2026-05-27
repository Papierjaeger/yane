"""Fitness surrogate model — cheap linear filter before expensive evaluation.

Uses genome descriptor vectors to predict fitness.  Genomes predicted to
be in the lower fraction are skipped (real eval saved).
"""
from __future__ import annotations

import math
from collections import deque

from yane.core.genome import Genome
from yane.evolution.landscape import genome_descriptor_vector


class FitnessSurrogate:
    """Linear fitness surrogate with adaptive filtering.

    Parameters
    ----------
    warmup_evals : int
        Number of real evaluations before the surrogate kicks in.
    surrogate_frac : float
        Fraction of genomes to filter out (bottom fraction by predicted fitness).
    buffer_generations : int
        How many generations of data to keep in the training buffer.
    """

    def __init__(
        self,
        warmup_evals: int = 200,
        surrogate_frac: float = 0.5,
        buffer_generations: int = 3,
    ) -> None:
        self.warmup_evals = warmup_evals
        self.surrogate_frac = max(0.0, min(0.95, surrogate_frac))
        self.buffer_generations = buffer_generations

        # Training data (ring buffer by generation)
        self._feature_buffer: deque[list[float]] = deque(maxlen=10000)
        self._fitness_buffer: deque[float] = deque(maxlen=10000)
        self._total_real_evals: int = 0
        self._filtered: int = 0
        self._passed: int = 0

        # Model parameters (linear regression: w · x + b)
        self._weights: list[float] | None = None  # 12-D weight vector
        self._bias: float = 0.0

    def predict(self, genome: Genome) -> float | None:
        """Predict fitness for a genome, or None if model is untrained."""
        if self._weights is None:
            return None
        features = genome_descriptor_vector(genome)
        return sum(w * f for w, f in zip(self._weights, features)) + self._bias

    def should_evaluate(self, genome: Genome) -> bool:
        """Decide whether this genome needs real evaluation.

        During warmup, all genomes are evaluated.  After warmup, genomes
        predicted to be in the bottom ``surrogate_frac`` are filtered out.
        """
        self._total_real_evals += 1

        if self._total_real_evals <= self.warmup_evals:
            return True

        pred = self.predict(genome)
        if pred is None:
            return True

        # Rank prediction against recent buffer
        buffer_fits = list(self._fitness_buffer)
        if not buffer_fits:
            return True

        # Count how many in buffer have lower predicted fitness
        below = sum(1 for f in buffer_fits if f < pred)
        rank = below / len(buffer_fits)

        if rank < self.surrogate_frac:
            # Predicted to be in bottom fraction → filter out
            self._filtered += 1
            return False

        self._passed += 1
        return True

    def train(self, genomes: list[Genome]) -> None:
        """Train/update the linear model on evaluated genomes.

        Uses ordinary least squares via the normal equation.
        """
        if len(genomes) < 12:  # need at least as many as features
            return

        # Build feature matrix X (n × 12) and target y (n)
        X = [genome_descriptor_vector(g) for g in genomes]
        y = [float(g.fitness) for g in genomes]

        # Add to buffer
        self._feature_buffer.extend(X)
        self._fitness_buffer.extend(y)

        # Use buffer for training
        buf_X = list(self._feature_buffer)
        buf_y = list(self._fitness_buffer)
        n = len(buf_X)
        if n < 12:
            return

        n_features = len(buf_X[0])
        # Build X^T * X and X^T * y
        XtX = [[0.0] * n_features for _ in range(n_features)]
        Xty = [0.0] * n_features
        for i in range(n):
            xi = buf_X[i]
            yi = buf_y[i]
            for j in range(n_features):
                Xty[j] += xi[j] * yi
                for k in range(n_features):
                    XtX[j][k] += xi[j] * xi[k]

        # Add small regularization (ridge, λ = 0.01)
        lam = 0.01
        for j in range(n_features):
            XtX[j][j] += lam * n

        # Solve XtX * w = Xty via Gaussian elimination
        w = self._solve_linear(XtX, Xty)
        if w is not None:
            self._weights = w
            self._bias = 0.0

    def _solve_linear(self, A: list[list[float]], b: list[float]) -> list[float] | None:
        """Solve Ax = b via Gaussian elimination with partial pivoting."""
        n = len(A)
        if n == 0:
            return None
        # Augmented matrix
        aug = [A[i][:] + [b[i]] for i in range(n)]
        for col in range(n):
            # Pivot
            pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
            if abs(aug[pivot][col]) < 1e-12:
                continue
            aug[col], aug[pivot] = aug[pivot], aug[col]
            piv_val = aug[col][col]
            for j in range(col, n + 1):
                aug[col][j] /= piv_val
            for row in range(n):
                if row != col:
                    factor = aug[row][col]
                    if abs(factor) > 1e-12:
                        for j in range(col, n + 1):
                            aug[row][j] -= factor * aug[col][j]
        return [aug[i][n] for i in range(n)]

    def get_spearman_rho(self) -> float:
        """Compute Spearman rank correlation between predictions and actuals."""
        if self._weights is None or len(self._fitness_buffer) < 10:
            return 0.0
        predictions = [
            sum(w * f for w, f in zip(self._weights, features))
            for features in self._feature_buffer
        ]
        actuals = list(self._fitness_buffer)
        n = len(predictions)
        if n < 10:
            return 0.0
        # Rank both
        pred_order = sorted(range(n), key=lambda i: predictions[i])
        actual_order = sorted(range(n), key=lambda i: actuals[i])
        pred_ranks = [0] * n
        actual_ranks = [0] * n
        for rank, idx in enumerate(pred_order):
            pred_ranks[idx] = rank
        for rank, idx in enumerate(actual_order):
            actual_ranks[idx] = rank
        d_sq = sum((pred_ranks[i] - actual_ranks[i]) ** 2 for i in range(n))
        return 1.0 - (6.0 * d_sq) / (n * (n * n - 1))

    def get_diagnostics(self) -> dict:
        return {
            "surrogate_trained": self._weights is not None,
            "surrogate_warmup": self._total_real_evals,
            "surrogate_warmup_remaining": max(0, self.warmup_evals - self._total_real_evals),
            "surrogate_filtered": self._filtered,
            "surrogate_passed": self._passed,
            "surrogate_spearman_rho": round(self.get_spearman_rho(), 4),
            "surrogate_frac": self.surrogate_frac,
        }
