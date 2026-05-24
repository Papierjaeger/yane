"""Async/distributed-friendly evaluation queue primitives."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

from yane.core.genome import Genome


@dataclass
class AsyncEvaluation:
    genome: Genome
    future: Future


class AsyncEvaluationQueue:
    """Submit genome evaluations concurrently and collect completed results."""

    def __init__(self, fitness_fn: Callable[[Genome], float], max_workers: int = 4, timeout_s: float | None = None) -> None:
        self.fitness_fn = fitness_fn
        self.timeout_s = timeout_s
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_workers))
        self._pending: list[AsyncEvaluation] = []

    def submit(self, genome: Genome) -> Future:
        future = self._executor.submit(self.fitness_fn, genome)
        self._pending.append(AsyncEvaluation(genome, future))
        return future

    def completed(self) -> list[tuple[Genome, float]]:
        done: list[tuple[Genome, float]] = []
        still_pending: list[AsyncEvaluation] = []
        for item in self._pending:
            if item.future.done():
                done.append((item.genome, float(item.future.result(timeout=0))))
            else:
                still_pending.append(item)
        self._pending = still_pending
        return done

    def drain(self) -> list[tuple[Genome, float]]:
        results: list[tuple[Genome, float]] = []
        future_to_genome = {item.future: item.genome for item in self._pending}
        for future in as_completed(future_to_genome, timeout=self.timeout_s):
            results.append((future_to_genome[future], float(future.result())))
        self._pending.clear()
        return results

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)


def evaluate_batch_async(
    genomes: list[Genome],
    fitness_fn: Callable[[Genome], float],
    max_workers: int = 4,
    timeout_s: float | None = None,
) -> list[tuple[Genome, float]]:
    queue = AsyncEvaluationQueue(fitness_fn, max_workers=max_workers, timeout_s=timeout_s)
    try:
        for genome in genomes:
            queue.submit(genome)
        return queue.drain()
    finally:
        queue.shutdown()
