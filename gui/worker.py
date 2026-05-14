"""Background threads for training and the API server."""
from __future__ import annotations
import gc
import threading
import time
from typing import Callable

from PySide6.QtCore import QThread, Signal

from yane.core.genome import Genome
from yane.neuro_evolution import _return_memory_to_os

_EMIT_INTERVAL_S    = 0.2    # emit UI update at most every 200 ms
_MEMORY_CHECK_EVERY = 500    # check resource limits every N iterations
_GC_EVERY           = 5000   # force gc.collect() + malloc_trim every N iterations


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

    def run(self) -> None:
        self._running = True
        self._iteration = 0
        last_emit = 0.0

        while self._running:
            while self._paused and self._running:
                time.sleep(0.05)

            try:
                genome = self._yane.next_genome()
                fitness = self._evaluate(genome)
                self._yane.submit_fitness(fitness)
                self._iteration += 1

                if self._iteration % _MEMORY_CHECK_EVERY == 0:
                    self._yane._enforce_memory_limit()
                    guard = self._yane._resource_guard
                    while self._running and not guard.system_ok():
                        time.sleep(0.5)

                if self._iteration % _GC_EVERY == 0 and self._running:
                    gc.collect()
                    if self._running:
                        _return_memory_to_os()

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
                self.error_occurred.emit(str(exc))
                break

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

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

        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="error")
        self._server = uvicorn.Server(config)
        asyncio.run(self._server.serve())

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
