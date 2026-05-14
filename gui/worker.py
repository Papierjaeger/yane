"""Background threads for training and the API server."""
from __future__ import annotations
import gc
import threading
import time
from typing import Callable

from PySide6.QtCore import QThread, Signal

from yane.core.genome import Genome
from yane.util.logger import get_logger

_log = get_logger()

_EMIT_INTERVAL_S     = 0.2    # emit UI update at most every 200 ms
_MEMORY_CHECK_EVERY  = 500    # check resource limits every N iterations
_GC_EVERY            = 5000   # force gc.collect() + malloc_trim every N iterations
_LOG_EVERY           = 1000   # write a memory snapshot to the log


def _trim_memory() -> None:
    from yane.neuro_evolution import _return_memory_to_os
    _return_memory_to_os()


class TrainingWorker(QThread):
    """Runs the manual training loop in a background thread."""

    iteration_done = Signal(int, float, object, dict)
    error_occurred = Signal(str)

    def __init__(self, yane, evaluate_fn: Callable[[Genome], float], parent=None) -> None:
        super().__init__(parent)
        self._yane = yane
        self._evaluate = evaluate_fn
        self._running = False
        self._paused = False
        self._iteration = 0

    @property
    def iteration(self) -> int:
        return self._iteration

    def _start_watchdog(self) -> None:
        """Watchdog thread: if iteration doesn't advance in 3 s, dumps all stack traces."""
        import threading
        import traceback
        import sys

        last_seen = [-1]

        def _watch():
            while self._running:
                time.sleep(3)
                if not self._running:
                    break
                cur = self._iteration
                if cur == last_seen[0]:
                    frames = sys._current_frames()
                    stacks = []
                    for tid, frame in frames.items():
                        name = next(
                            (t.name for t in threading.enumerate() if t.ident == tid),
                            str(tid),
                        )
                        stack = "".join(traceback.format_stack(frame))
                        stacks.append(f"--- Thread '{name}' (id={tid}) ---\n{stack}")
                    _log.warning(
                        "WATCHDOG: worker stuck at iter=%d for 3+ s\n%s",
                        cur, "\n".join(stacks),
                    )
                last_seen[0] = cur

        threading.Thread(target=_watch, daemon=True, name="yane-watchdog").start()

    def run(self) -> None:
        self._running = True
        self._iteration = 0
        last_emit = 0.0
        self._start_watchdog()
        _log.info("Training started")

        while self._running:
            while self._paused and self._running:
                time.sleep(0.05)

            try:
                _t_iter = time.perf_counter()

                genome = self._yane.next_genome()
                _t_eval = time.perf_counter()

                fitness = self._evaluate(genome)
                _t_submit = time.perf_counter()

                self._yane.submit_fitness(fitness)
                self._iteration += 1
                _t_done = time.perf_counter()

                # Log if any single step takes > 200 ms (likely the freeze culprit)
                _next_ms   = (_t_eval   - _t_iter)   * 1000
                _eval_ms   = (_t_submit - _t_eval)   * 1000
                _submit_ms = (_t_done   - _t_submit) * 1000
                _total_ms  = (_t_done   - _t_iter)   * 1000
                if _total_ms > 200:
                    _log.warning(
                        "SLOW iter=%d: total=%.0fms  "
                        "next_genome=%.0fms  evaluate=%.0fms  submit=%.0fms  "
                        "nodes=%d  conns=%d",
                        self._iteration, _total_ms,
                        _next_ms, _eval_ms, _submit_ms,
                        len(genome.nodes), genome.connection_count,
                    )

                if self._iteration % _MEMORY_CHECK_EVERY == 0:
                    self._yane._enforce_memory_limit()
                    guard = self._yane._resource_guard
                    while self._running and not guard.system_ok():
                        time.sleep(0.5)

                if self._iteration % _GC_EVERY == 0 and self._running:
                    _t_gc = time.perf_counter()
                    collected = gc.collect()
                    if self._running:
                        _trim_memory()
                    gc_ms = (time.perf_counter() - _t_gc) * 1000
                    if gc_ms > 100:
                        _log.warning(
                            "SLOW gc.collect() at iter=%d: %.1fms  collected=%d objects",
                            self._iteration, gc_ms, collected,
                        )

                if self._iteration % _LOG_EVERY == 0:
                    mem = self._yane.population_memory_info()
                    _log.info(
                        "iter=%d  genomes=%d  avg_nodes=%.1f  max_nodes=%d  "
                        "avg_conns=%.1f  max_conns=%d  fitness=%.4f",
                        self._iteration,
                        mem["total_genomes"],
                        mem["avg_nodes_per_genome"],
                        mem["largest_genome_nodes"],
                        mem["avg_connections_per_genome"],
                        mem["largest_genome_connections"],
                        fitness,
                    )

                now = time.perf_counter()
                if now - last_emit >= _EMIT_INTERVAL_S:
                    last_emit = now
                    try:
                        best = self._yane.get_best().copy()
                    except RuntimeError:
                        best = genome.copy()
                    mem = self._yane.population_memory_info()
                    self.iteration_done.emit(self._iteration, fitness, best, mem)

            except Exception as exc:
                _log.error("Training error at iter %d: %s", self._iteration, exc, exc_info=True)
                self.error_occurred.emit(str(exc))
                break

        _log.info("Training stopped at iter=%d", self._iteration)

    def pause(self) -> None:
        self._paused = True
        _log.info("Training paused at iter=%d", self._iteration)

    def resume(self) -> None:
        self._paused = False
        _log.info("Training resumed at iter=%d", self._iteration)

    def stop(self) -> None:
        self._paused = False
        self._running = False


class ServerThread(threading.Thread):
    """Runs the FastAPI/uvicorn server in a daemon thread."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self._server = None

    def run(self) -> None:
        import asyncio
        import uvicorn
        from yane.api.server import app

        _log.info("API server starting on %s:%d", self.host, self.port)
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="error")
        self._server = uvicorn.Server(config)
        asyncio.run(self._server.serve())
        _log.info("API server stopped")

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
