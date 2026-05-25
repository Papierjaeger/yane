"""Composable evaluation middleware for genome fitness functions."""
from __future__ import annotations

import dataclasses
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Protocol

from yane.core.genome import Genome


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
