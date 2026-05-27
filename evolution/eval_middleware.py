"""Composable evaluation middleware for genome fitness functions."""
from __future__ import annotations

import dataclasses
import random
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Protocol

from yane.core.genome import Genome


def _aggregate(values: list[float], aggregation: str) -> float:
    """Aggregate a list of floats using the named strategy."""
    if not values:
        return 0.0
    if aggregation == "min":
        return min(values)
    if aggregation == "max":
        return max(values)
    if aggregation == "median":
        sv = sorted(values)
        return sv[len(sv) // 2]
    return sum(values) / len(values)


@dataclasses.dataclass
class EvalContext:
    """Mutable context shared by evaluation middleware."""

    diagnostics: dict[str, Any] = dataclasses.field(default_factory=dict)


class EvalMiddleware(Protocol):
    def __call__(
        self,
        genome: Genome,
        eval_fn: Callable[[Genome], float],
        ctx: EvalContext,
    ) -> float:
        ...


def genome_fingerprint(genome: Genome) -> tuple:
    """Return a topology+weight fingerprint for cache invalidation."""
    nodes = tuple(
        (
            n.innovation,
            getattr(n.type, "value", str(n.type)),
            getattr(n.activation, "value", str(n.activation)),
            round(float(n.bias), 12),
            bool(n.persist_value),
            int(n.max_triggers),
            int(n.input_index),
            round(float(n.input_scale), 12),
            round(float(n.output_scale), 12),
            round(float(getattr(n, "leak_alpha", 1.0)), 12),
            round(float(getattr(n, "memory_gate", 0.0)), 12),
        )
        for n in genome.nodes
    )
    connections = []
    for src in genome.nodes:
        for conn in src.connections:
            connections.append((
                src.innovation,
                conn.target.innovation,
                conn.innovation,
                round(float(conn.weight), 12),
                bool(conn.enabled),
            ))
    return nodes, tuple(sorted(connections))


class CachingMiddleware:
    """LRU cache keyed by genome topology and weights."""

    def __init__(self, maxsize: int = 512) -> None:
        self.maxsize = max(1, int(maxsize))
        self._cache: OrderedDict[tuple, float] = OrderedDict()
        self.hits: int = 0
        self.misses: int = 0

    def __call__(self, genome: Genome, eval_fn, ctx: EvalContext) -> float:
        key = genome_fingerprint(genome)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            value = self._cache[key]
        else:
            self.misses += 1
            value = eval_fn(genome)
            self._cache[key] = value
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)
        total = self.hits + self.misses
        ctx.diagnostics["cache_hits"] = self.hits
        ctx.diagnostics["cache_misses"] = self.misses
        ctx.diagnostics["cache_hit_rate"] = self.hits / total if total else 0.0
        return value


class TimingMiddleware:
    """Measure wall-clock evaluation time."""

    def __init__(self) -> None:
        self.n: int = 0
        self.total_ms: float = 0.0
        self.last_ms: float = 0.0

    def __call__(self, genome: Genome, eval_fn, ctx: EvalContext) -> float:
        start = time.perf_counter()
        try:
            return eval_fn(genome)
        finally:
            self.last_ms = (time.perf_counter() - start) * 1000.0
            self.total_ms += self.last_ms
            self.n += 1
            ctx.diagnostics["eval_time_last_ms"] = self.last_ms
            ctx.diagnostics["eval_time_middleware_mean_ms"] = self.total_ms / self.n


class RetryMiddleware:
    """Retry flaky evaluators and aggregate successful attempts."""

    def __init__(self, n: int = 3, aggregation: str = "mean") -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        if aggregation not in {"mean", "min", "max"}:
            raise ValueError("aggregation must be 'mean', 'min', or 'max'")
        self.n = int(n)
        self.aggregation = aggregation
        self.retry_count: int = 0

    def __call__(self, genome: Genome, eval_fn, ctx: EvalContext) -> float:
        values: list[float] = []
        last_error: Exception | None = None
        for attempt in range(self.n):
            try:
                values.append(float(eval_fn(genome)))
            except Exception as exc:
                last_error = exc
                if attempt < self.n - 1:
                    self.retry_count += 1
        ctx.diagnostics["retry_count"] = self.retry_count
        if not values:
            assert last_error is not None
            raise last_error
        if self.aggregation == "min":
            return min(values)
        if self.aggregation == "max":
            return max(values)
        return sum(values) / len(values)


class ComponentMiddleware:
    """Evaluate named auxiliary fitness components and combine them."""

    def __init__(
        self,
        components: dict[str, Callable[[Genome], float]],
        weights: dict[str, float] | None = None,
        include_base: bool = True,
        base_name: str = "base",
    ) -> None:
        self.components = dict(components)
        self.weights = dict(weights or {})
        self.include_base = bool(include_base)
        self.base_name = base_name

    def __call__(self, genome: Genome, eval_fn, ctx: EvalContext) -> float:
        raw: dict[str, float] = {}
        total = 0.0
        if self.include_base:
            base = float(eval_fn(genome))
            raw[self.base_name] = base
            total += base * self.weights.get(self.base_name, 1.0)
        for name, fn in self.components.items():
            value = float(fn(genome))
            raw[name] = value
            total += value * self.weights.get(name, 1.0)
        ctx.diagnostics["component_values"] = raw
        ctx.diagnostics["component_weights"] = {
            name: self.weights.get(name, 1.0)
            for name in raw
        }
        ctx.diagnostics["component_total"] = total
        return total


class CaseBatchMiddleware:
    """Evaluate fixed train/validation case lists with separate diagnostics."""

    def __init__(
        self,
        train_cases: list[Any] | tuple[Any, ...],
        case_fn: Callable[[Genome, Any], float],
        aggregation: str = "mean",
        validation_cases: list[Any] | tuple[Any, ...] | None = None,
        validation_name: str = "validation",
    ) -> None:
        if aggregation not in {"mean", "min", "max"}:
            raise ValueError("aggregation must be 'mean', 'min', or 'max'")
        self.train_cases = tuple(train_cases)
        self.validation_cases = tuple(validation_cases or ())
        self.case_fn = case_fn
        self.aggregation = aggregation
        self.validation_name = validation_name

    def __call__(self, genome: Genome, eval_fn, ctx: EvalContext) -> float:
        # The wrapped eval_fn is intentionally ignored: this middleware turns a
        # single genome evaluation into a deterministic case batch.
        train_scores = [float(self.case_fn(genome, case)) for case in self.train_cases]
        fitness = self._aggregate(train_scores)
        ctx.diagnostics["case_scores"] = train_scores
        ctx.diagnostics["case_count"] = len(train_scores)
        ctx.diagnostics["case_success_rate"] = (
            sum(1 for score in train_scores if score > 0.0) / len(train_scores)
            if train_scores else 0.0
        )
        if self.validation_cases:
            validation_scores = [
                float(self.case_fn(genome, case))
                for case in self.validation_cases
            ]
            ctx.diagnostics[f"{self.validation_name}_scores"] = validation_scores
            ctx.diagnostics[f"{self.validation_name}_fitness"] = self._aggregate(validation_scores)
        return fitness

    def _aggregate(self, values: list[float]) -> float:
        return _aggregate(values, self.aggregation)


class NoiseMiddleware:
    """Perturb genome weights with Gaussian noise for robust evaluation.

    Creates *n_samples* copies of the genome, adds N(0, *sigma*) noise to
    each copy's connection weights, evaluates all copies, and returns the
    aggregated result. The original genome is never modified.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian noise applied to weights.
    n_samples : int
        Number of noisy evaluations to average (must be >= 1).
    aggregation : str
        How to combine the noisy evaluations: ``"mean"``, ``"min"``,
        ``"max"``, or ``"median"``.
    """

    def __init__(
        self,
        sigma: float = 0.05,
        n_samples: int = 3,
        aggregation: str = "mean",
    ) -> None:
        if sigma < 0.0:
            raise ValueError("sigma must be >= 0")
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if aggregation not in {"mean", "min", "max", "median"}:
            raise ValueError("aggregation must be 'mean', 'min', 'max', or 'median'")
        self.sigma = float(sigma)
        self.n_samples = int(n_samples)
        self.aggregation = aggregation
        self._rng = random.Random()

    def _perturb_weights(self, genome: Genome) -> Genome:
        """Return a deep copy of *genome* with N(0, sigma) added to weights."""
        copy = genome.copy()
        for node in copy.nodes:
            for conn in node.connections:
                if conn.enabled:
                    conn.weight += self._rng.gauss(0.0, self.sigma)
        return copy

    def __call__(
        self,
        genome: Genome,
        eval_fn: Callable[[Genome], float],
        ctx: EvalContext,
    ) -> float:
        if self.sigma == 0.0:
            return eval_fn(genome)
        values: list[float] = []
        for _ in range(self.n_samples):
            noisy = self._perturb_weights(genome)
            values.append(float(eval_fn(noisy)))
        ctx.diagnostics["noise_sigma"] = self.sigma
        ctx.diagnostics["noise_n_samples"] = self.n_samples
        ctx.diagnostics["noise_raw_values"] = values
        return _aggregate(values, self.aggregation)


def apply_middleware(
    genome: Genome,
    eval_fn: Callable[[Genome], float],
    middlewares: list[EvalMiddleware],
    ctx: EvalContext,
) -> float:
    """Apply middleware in LIFO order; last added middleware is outermost."""

    wrapped = eval_fn
    for middleware in middlewares:
        inner = wrapped

        def wrapped(g: Genome, mw=middleware, fn=inner) -> float:
            return mw(g, fn, ctx)

    return wrapped(genome)
