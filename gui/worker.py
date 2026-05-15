"""Background threads for training and the API server."""
from __future__ import annotations
import gc
import multiprocessing as _mp
import threading
import time
from typing import Callable

from PySide6.QtCore import QThread, Signal

from yane.core.genome import Genome
from yane.neuro_evolution import _return_memory_to_os

_EMIT_INTERVAL_S    = 0.2    # emit UI update at most every 200 ms
_MEMORY_CHECK_EVERY = 500    # check resource limits every N iterations
_GC_EVERY           = 5000   # force gc.collect() + malloc_trim every N iterations


def _close_env(eval_fn) -> None:
    env = getattr(eval_fn, "_env", None)
    if env is not None:
        try:
            env.close()
        except Exception:
            pass


def _eval_worker_loop(make_eval_fn, genome_q, result_q) -> None:
    """Runs in a forked child process — inherits make_eval_fn closure directly."""
    eval_fn = make_eval_fn()
    while True:
        genome = genome_q.get()
        if genome is None:       # shutdown sentinel
            break
        result_q.put(eval_fn(genome))


class EvalWorker:
    """One forked subprocess that owns a single gym env and evaluates genomes.

    Uses fork (Linux) so make_eval_fn closures are inherited without pickling.
    Only the genome (now picklable) travels through the queue.
    """

    def __init__(self, make_eval_fn: Callable) -> None:
        ctx = _mp.get_context("fork")
        self._genome_q: _mp.Queue = ctx.Queue()
        self._result_q: _mp.Queue = ctx.Queue()
        self._proc = ctx.Process(
            target=_eval_worker_loop,
            args=(make_eval_fn, self._genome_q, self._result_q),
            daemon=True,
        )
        self._proc.start()

    def submit(self, genome) -> None:
        self._genome_q.put(genome)

    def result(self) -> float:
        return self._result_q.get()

    def close(self) -> None:
        self._genome_q.put(None)
        self._proc.join(timeout=2)
        if self._proc.is_alive():
            self._proc.terminate()


class TrainingWorker(QThread):
    """Runs the manual training loop in a background thread.

    When yane._n_workers > 1, evaluates a batch of genomes in parallel using
    a ThreadPoolExecutor — each worker gets its own gym env instance via
    make_eval_fn so environments are never shared between threads.
    """

    iteration_done = Signal(int, float, object, dict)
    error_occurred = Signal(str)

    def __init__(
        self,
        yane,
        evaluate_fn: Callable[[Genome], float],
        make_eval_fn: Callable | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._yane = yane
        self._evaluate = evaluate_fn
        self._make_eval_fn = make_eval_fn
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

        n_workers = getattr(self._yane, '_n_workers', 1)
        if n_workers > 1 and self._make_eval_fn is not None:
            self._run_parallel(n_workers, last_emit)
        else:
            self._run_sequential(last_emit)

    def _run_sequential(self, last_emit: float) -> None:
        while self._running:
            while self._paused and self._running:
                time.sleep(0.05)

            try:
                genome = self._yane.next_genome()
                fitness = self._evaluate(genome)
                self._yane.submit_fitness(fitness)
                self._iteration += 1
                if self._yane.min_fitness is not None and fitness >= self._yane.min_fitness:
                    self._running = False

                self._maybe_maintain(genome, fitness)

                now = time.perf_counter()
                if now - last_emit >= _EMIT_INTERVAL_S:
                    last_emit = now
                    self._emit_update(genome, fitness)

            except Exception as exc:
                self.error_occurred.emit(str(exc))
                break

        self._emit_final()
        _close_env(self._evaluate)

    def _run_parallel(self, n_workers: int, last_emit: float) -> None:
        # One forked subprocess per worker — each inherits make_eval_fn via
        # fork so gym env closures don't need to be pickled. Only genomes
        # (picklable) travel through the queue.
        workers = [EvalWorker(self._make_eval_fn) for _ in range(n_workers)]

        try:
            while self._running:
                while self._paused and self._running:
                    time.sleep(0.05)

                try:
                    genomes = self._yane.next_genome_batch(n_workers)
                    for w, g in zip(workers, genomes):
                        w.submit(g)
                    fitnesses = [w.result() for w in workers]
                    results = list(zip(genomes, fitnesses))
                    self._yane.submit_fitness_batch(results)
                    self._iteration += len(results)

                    best_fitness = max(fitnesses)
                    best_genome  = genomes[fitnesses.index(best_fitness)]
                    if self._yane.min_fitness is not None and best_fitness >= self._yane.min_fitness:
                        self._running = False

                    if self._iteration % _MEMORY_CHECK_EVERY < n_workers:
                        self._yane._enforce_memory_limit()
                        guard = self._yane._resource_guard
                        while self._running and not guard.system_ok():
                            time.sleep(0.5)

                    if self._iteration % _GC_EVERY < n_workers and self._running:
                        gc.collect()
                        _return_memory_to_os()

                    now = time.perf_counter()
                    if now - last_emit >= _EMIT_INTERVAL_S:
                        last_emit = now
                        self._emit_update(best_genome, best_fitness)

                except Exception as exc:
                    self.error_occurred.emit(str(exc))
                    break
        finally:
            for w in workers:
                w.close()

        self._emit_final()

    def _maybe_maintain(self, genome: Genome, fitness: float) -> None:
        if self._iteration % _MEMORY_CHECK_EVERY == 0:
            self._yane._enforce_memory_limit()
            guard = self._yane._resource_guard
            while self._running and not guard.system_ok():
                time.sleep(0.5)
        if self._iteration % _GC_EVERY == 0 and self._running:
            gc.collect()
            _return_memory_to_os()

    def _emit_update(self, genome: Genome, fitness: float) -> None:
        try:
            best = self._yane.get_best().copy()
        except RuntimeError:
            best = genome.copy()
        mem = self._yane.population_memory_info()
        self.iteration_done.emit(self._iteration, fitness, best, mem)

    def _emit_final(self) -> None:
        try:
            best = self._yane.get_best().copy()
            mem = self._yane.population_memory_info()
            self.iteration_done.emit(self._iteration, best.fitness, best, mem)
        except RuntimeError:
            pass

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop(self) -> None:
        self._paused = False
        self._running = False


class EpisodeRunner(QThread):
    """Loops gym episodes with render until stop() is called."""

    frame_ready   = Signal(object)  # numpy array frame
    score_updated = Signal(float)   # cumulative reward after each step

    def __init__(self, genome, example, parent=None) -> None:
        super().__init__(parent)
        self._genome  = genome
        self._example = example
        self._running = True

    def run(self) -> None:
        last_frame = 0.0
        step_delay = 0.0

        def render_cb(frame):
            nonlocal last_frame
            if not self._running:
                return
            now = time.perf_counter()
            if now - last_frame < 1 / 30:
                return
            last_frame = now
            self.frame_ready.emit(frame)

        def step_cb(total):
            if self._running:
                self.score_updated.emit(total)
            return step_delay

        eval_fn = self._example.make_eval(render_cb, step_cb, demo=True)
        env = getattr(eval_fn, "_env", None)
        if env is not None:
            step_delay = env.metadata.get("render_fps", 30) ** -1

        try:
            while self._running:
                eval_fn(self._genome)
        except Exception:
            pass
        finally:
            _close_env(eval_fn)

    def stop(self) -> None:
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
