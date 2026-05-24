"""Background threads for training and the API server."""
from __future__ import annotations
import gc
import math
import multiprocessing as mp
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from PySide6.QtCore import QThread, Signal

from yane.core.genome import Genome
from yane.neuro_evolution import _aggregate_fitnesses, _return_memory_to_os

_EMIT_INTERVAL_S    = 0.5    # emit UI update at most every 500 ms
_MEMORY_CHECK_EVERY = 500    # check resource limits every N iterations
_GC_EVERY           = 5000   # force gc.collect() + malloc_trim every N iterations

# ---------------------------------------------------------------------------
# Worker-count formula
# ---------------------------------------------------------------------------

def _optimal_workers(eval_ms: float, batch_size: int, overhead_ms: float, cap: int) -> int:
    """Return the worker count that minimises wall time for a batch.

    Cost model:
        sequential:  batch_size × eval_ms
        w workers:   batch_size × eval_ms / w  +  overhead_ms

    MP beats sequential iff  seq_time > overhead_ms.  When it does, the
    minimum beneficial w is  ⌈seq_time / (seq_time − overhead_ms)⌉  and
    the optimal w (where per-worker work equals overhead) is
    seq_time / overhead_ms.  Returns 1 when sequential is always faster.
    """
    seq_time = batch_size * eval_ms
    if seq_time <= overhead_ms:
        return 1
    min_beneficial = math.ceil(seq_time / (seq_time - overhead_ms))
    min_beneficial = max(2, min_beneficial)
    optimal = max(min_beneficial, int(seq_time / overhead_ms))
    return min(cap, optimal)


# ---------------------------------------------------------------------------
# Multiprocessing helpers — must be at module level to be picklable
# ---------------------------------------------------------------------------

_mp_eval_fn        = None   # set once per subprocess by _mp_initializer
_mp_n_evaluations  = 1
_mp_aggregation    = "mean"
_mp_sigma_penalty  = 0.0

def _mp_initializer(make_eval_fn, render_cb,
                    n_evaluations=1, aggregation="mean", sigma_penalty=0.0):
    """Called once in each worker process to create the fitness evaluator.
    Uses fork semantics: make_eval_fn is inherited from the parent process,
    so closures (gym environments, datasets) work without pickling.
    """
    global _mp_eval_fn, _mp_n_evaluations, _mp_aggregation, _mp_sigma_penalty
    _mp_eval_fn       = make_eval_fn(render_cb)
    _mp_n_evaluations = n_evaluations
    _mp_aggregation   = aggregation
    _mp_sigma_penalty = sigma_penalty

def _mp_evaluate(genome: Genome) -> tuple[float, float]:
    """Evaluate a single genome — runs inside a worker process."""
    if _mp_n_evaluations <= 1:
        start = time.perf_counter()
        fitness = _mp_eval_fn(genome)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return fitness, elapsed_ms

    fitnesses: list[float] = []
    total_ms = 0.0
    for _ in range(_mp_n_evaluations):
        start = time.perf_counter()
        f = _mp_eval_fn(genome)
        total_ms += (time.perf_counter() - start) * 1000.0
        fitnesses.append(f)
    return _aggregate_fitnesses(fitnesses, _mp_aggregation, _mp_sigma_penalty), total_ms


def _timed_evaluate(eval_fn: Callable, genome: Genome) -> tuple[float, float]:
    start = time.perf_counter()
    fitness = eval_fn(genome)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return fitness, elapsed_ms


def _close_env(eval_fn) -> None:
    env = getattr(eval_fn, "_env", None)
    if env is not None:
        try:
            env.close()
        except Exception:
            pass


class TrainingWorker(QThread):
    """Runs the manual training loop in a background thread.

    The eval function is created inside run() on the worker thread so that
    gym.make() (which can take 0.5–2 s and initialises pygame/OpenGL) never
    blocks the main thread.

    When yane._n_workers > 1, evaluates a batch of genomes in parallel using
    a ThreadPoolExecutor — each worker thread gets its own gym env instance
    via make_eval_fn. ThreadPoolExecutor is used (not multiprocessing) because
    forking a running Qt process is unsafe due to inherited mutex state.
    Box2D-based environments release the GIL during physics simulation so
    threads provide real concurrency for LunarLander, BipedalWalker, CarRacing.
    """

    iteration_done   = Signal(int, float, object, dict)
    error_occurred   = Signal(str)
    info_message     = Signal(str)   # non-fatal status update (e.g. MP fallback)
    workers_resolved = Signal(int)   # actual worker count chosen (0 = sequential)

    def __init__(
        self,
        yane,
        make_eval_fn: Callable,   # called on the worker thread to create the eval fn
        render_cb: Callable | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._yane = yane
        self._make_eval_fn = make_eval_fn
        self._render_cb = render_cb
        self._evaluate: Callable | None = None
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

        # Curriculum requires sequential evaluation: stage advances must be
        # atomic and the population re-evaluation after reset_stagnation() would
        # race with MP batch submissions.
        if self._yane._curriculum is not None:
            self.workers_resolved.emit(1)
            self.info_message.emit("Curriculum aktiv — Training läuft sequenziell.")
            self._run_sequential(last_emit)
            return

        n_workers = getattr(self._yane, '_n_workers', 1)
        if n_workers == 0:
            # Auto: let _run_multiprocess measure eval speed and choose
            self._run_multiprocess(mp.cpu_count(), last_emit, auto=True)
        elif n_workers > 1:
            self.workers_resolved.emit(n_workers)
            self._run_multiprocess(n_workers, last_emit, auto=False)
        else:
            self.workers_resolved.emit(1)
            self._run_sequential(last_emit)

    def _run_sequential(self, last_emit: float, auto: bool = False) -> None:
        _OVERHEAD_MS      = 16.0
        _MP_RECHECK_EVERY = 50

        if self._yane._curriculum is not None:
            fitness_fn = None
        else:
            try:
                fitness_fn = self._make_eval_fn(self._render_cb)
            except Exception as exc:
                self.error_occurred.emit(str(exc))
                return

        switch_to_mp_n  = 0
        switch_to_mp_ms = 0.0
        last_cur_stage  = self._yane.curriculum_stage

        def on_iter(iteration: int, fitness: float, elapsed_ms: float) -> bool:
            nonlocal last_emit, switch_to_mp_n, switch_to_mp_ms, last_cur_stage
            self._iteration = iteration

            # Detect curriculum stage advancement and emit a status message.
            cur_stage = self._yane.curriculum_stage
            if cur_stage is not None and cur_stage != last_cur_stage:
                last_cur_stage = cur_stage
                cur   = self._yane._curriculum
                stage = cur.current_stage
                self.info_message.emit(
                    f"Curriculum: Stage {cur.stage_index + 1}/{cur.stage_count}"
                    f" — {stage.name or 'unnamed'}"
                )

            while self._paused and self._running:
                time.sleep(0.05)
            if not self._running:
                return False

            if auto and iteration % _MP_RECHECK_EVERY == 0:
                batch_size = self._yane._population.max_size
                needed = _optimal_workers(elapsed_ms, batch_size, _OVERHEAD_MS,
                                         mp.cpu_count())
                if needed >= 2:
                    self.workers_resolved.emit(needed)
                    self.info_message.emit(
                        f"Auto → {needed} Worker "
                        f"(eval {elapsed_ms:.2f}ms/Genome nach {iteration} It.)"
                    )
                    switch_to_mp_n  = needed
                    switch_to_mp_ms = elapsed_ms
                    return False

            time.sleep(0)  # yield GIL to Qt event loop

            now = time.perf_counter()
            if now - last_emit >= _EMIT_INTERVAL_S:
                last_emit = now
                self._emit_update(iteration, fitness)

            return True

        try:
            self._yane.train(fitness_fn, on_iteration=on_iter)
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            return

        if switch_to_mp_n > 0:
            self._run_multiprocess(mp.cpu_count(), last_emit,
                                   auto=True, bootstrap_eval_ms=switch_to_mp_ms)
            return

        self._emit_final()
        _close_env(fitness_fn)

    def _run_parallel(self, n_workers: int, last_emit: float) -> None:
        # Each thread owns its own eval_fn (and thus its own gym env).
        # Only the first gets the render callback; parallel rendering would race.
        try:
            eval_fns = [self._make_eval_fn(self._render_cb if i == 0 else None)
                        for i in range(n_workers)]
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            return

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            while self._running:
                while self._paused and self._running:
                    time.sleep(0.05)

                try:
                    genomes = self._yane.next_genome_batch(n_workers)
                    futures = [pool.submit(self._yane._run_evaluations, g, fn) for fn, g in zip(eval_fns, genomes)]
                    timed = [f.result() for f in futures]
                    fitnesses = [r.fitness for r in timed]
                    results = [
                        (genome, r.fitness, r.elapsed_ms)
                        for genome, r in zip(genomes, timed)
                    ]
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
                        self._emit_update(self._iteration, best_fitness)

                except Exception as exc:
                    self.error_occurred.emit(str(exc))
                    break

        for fn in eval_fns:
            _close_env(fn)

        self._emit_final()

    def _run_multiprocess(self, n_workers: int, last_emit: float,
                          auto: bool = False,
                          bootstrap_eval_ms: float | None = None) -> None:
        """Evaluate genomes in parallel subprocesses using fork.

        Genome objects are pickled for IPC; the fitness function is inherited
        from the parent process via fork so closures (datasets, gym envs) work
        without additional pickling.  Each subprocess calls make_eval_fn once
        in its initializer to set up its own environment.

        bootstrap_eval_ms: when set, skip the bootstrap eval (caller already
        measured representative eval time, e.g. after switching from sequential).
        """
        import multiprocessing as mp

        ctx = mp.get_context('fork')

        if bootstrap_eval_ms is not None:
            eval_ms = bootstrap_eval_ms
        else:
            # Bootstrap: evaluate the seed genome sequentially, measure speed,
            # then decide whether MP overhead is worth it (see cost model below).
            try:
                seed_fn = self._make_eval_fn(None)
                seed_g  = self._yane.next_genome()
                seed_result = self._yane._run_evaluations(seed_g, seed_fn)
                seed_fit, eval_ms = seed_result.fitness, seed_result.elapsed_ms
                self._yane.submit_fitness(seed_fit, eval_ms)
                self._iteration += 1
                _close_env(seed_fn)
            except Exception as exc:
                self.error_occurred.emit(str(exc))
                return

        # Decide how many workers to use.
        #
        # Cost model (for a batch of k genomes):
        #   sequential:   k × eval_ms
        #   w workers:    k × eval_ms / w  +  overhead
        #
        # MP beats sequential when:  overhead  <  k × eval_ms × (w−1)/w
        # → MP is NEVER beneficial when seq_time ≤ overhead (any overhead dominates).
        # → When seq_time > overhead, there exist w≥2 that win.
        #
        # Minimum w that beats sequential:
        #   w ≥ seq_time / (seq_time − overhead)
        #
        # "Optimal" w (eval-work-per-worker equals overhead, diminishing returns):
        #   w_opt = seq_time / overhead  (real-valued; round up to nearest int ≥ min_beneficial)
        _OVERHEAD_MS  = 16.0
        batch_size    = self._yane._population.max_size
        n_workers_max = n_workers   # preserve cpu_count cap before any override
        chosen        = _optimal_workers(eval_ms, batch_size, _OVERHEAD_MS, n_workers)
        seq_time      = batch_size * eval_ms

        if auto:
            # In auto mode never fall back to sequential based on the bootstrap
            # alone: the first genome is often empty (no connections) and evaluates
            # much faster than later, more complex genomes. Start with at least 2
            # workers and let the EMA rescaling converge to the right count — it
            # will switch to sequential after ~10 batches if truly warranted.
            if chosen <= 1:
                chosen = 2
            n_workers = chosen
            self.workers_resolved.emit(n_workers)
            self.info_message.emit(
                f"Auto → {n_workers} Worker "
                f"({eval_ms:.2f}ms/Genome, EMA justiert laufend)"
            )
        elif chosen <= 1:
            self.workers_resolved.emit(1)
            self.info_message.emit(
                f"MP-Overhead > Nutzen ({eval_ms:.2f}ms/Genome, "
                f"seq={seq_time:.0f}ms ≤ overhead={_OVERHEAD_MS:.0f}ms) "
                f"— Training läuft sequenziell."
            )
            self._run_sequential(0.0)
            return
        else:
            self.workers_resolved.emit(n_workers)

        def _make_pool(nw: int):
            return ctx.Pool(
                processes=nw,
                initializer=_mp_initializer,
                initargs=(
                    self._make_eval_fn, None,
                    self._yane._runner.n_evaluations,
                    self._yane._runner.aggregation,
                    self._yane._runner.sigma_penalty,
                ),
            )

        try:
            pool = _make_pool(n_workers)
        except Exception as exc:
            self.error_occurred.emit(f"Multiprocessing Pool konnte nicht erstellt werden: {exc}")
            return

        # Evaluate a full generation per round so workers stay busy.
        batch_size = max(n_workers * 4, self._yane._population.max_size)

        # Adaptive scaling state:
        # eval_ema — exponential moving average of eval time per genome (ms).
        # Estimated as: (batch_wall - overhead) * n_workers / batch_size.
        # Every _RESCALE_EVERY batches the optimal worker count is recomputed;
        # if it differs by >= 2, the pool is recreated.
        _ALPHA         = 0.25   # EMA smoothing (higher = reacts faster)
        _RESCALE_EVERY = 10     # check every N batches
        eval_ema       = eval_ms
        batch_count    = 0

        # Disable Python's automatic cyclic GC for the duration of the
        # multiprocessing loop.  Automatic GC triggered inside next_genome_batch()
        # (during allocation) can try to finalise numpy/C-ext objects while the
        # pool's IPC threads hold C-level references to them → segfault.
        # Explicit gc.collect() calls below run at quiescent points (after
        # pool.map() has returned) and are safe.
        gc.disable()
        try:
            while self._running:
                while self._paused and self._running:
                    time.sleep(0.05)

                try:
                    genomes   = self._yane.next_genome_batch(batch_size)
                    chunksize = max(1, len(genomes) // n_workers)

                    t_map = time.perf_counter()
                    timed = pool.map(_mp_evaluate, genomes, chunksize=chunksize)
                    map_ms    = (time.perf_counter() - t_map) * 1000.0

                    fitnesses = [fitness for fitness, _elapsed_ms in timed]
                    results = [
                        (genome, fitness, elapsed_ms)
                        for genome, (fitness, elapsed_ms) in zip(genomes, timed)
                    ]
                    self._yane.submit_fitness_batch(results)
                    self._iteration += len(results)
                    batch_count += 1

                    # Update eval estimate and optionally rescale pool
                    est = max(0.01, (map_ms - _OVERHEAD_MS) * n_workers / len(genomes))
                    eval_ema = (1 - _ALPHA) * eval_ema + _ALPHA * est

                    if auto and batch_count % _RESCALE_EVERY == 0:
                        new_opt = _optimal_workers(eval_ema, batch_size, _OVERHEAD_MS, n_workers_max)
                        if new_opt <= 1:
                            # EMA has converged: sequential is faster.
                            # Switch to sequential (with auto=True so it can switch
                            # back to MP if eval time rises later).
                            pool.terminate(); pool.join()
                            self.workers_resolved.emit(1)
                            self.info_message.emit(
                                f"Auto → sequenziell "
                                f"({eval_ema:.2f}ms/Genome, Overhead überwiegt)"
                            )
                            self._run_sequential(last_emit, auto=True)
                            return
                        elif abs(new_opt - n_workers) >= 2:
                            pool.terminate(); pool.join()
                            n_workers  = new_opt
                            batch_size = max(n_workers * 4,
                                            self._yane._population.max_size)
                            pool = _make_pool(n_workers)
                            self.workers_resolved.emit(n_workers)
                            self.info_message.emit(
                                f"Auto → {n_workers} Worker "
                                f"(eval {eval_ema:.1f}ms/Genome, angepasst)"
                            )

                    best_fitness = max(fitnesses)
                    best_genome  = genomes[fitnesses.index(best_fitness)]
                    if self._yane.min_fitness is not None and best_fitness >= self._yane.min_fitness:
                        self._running = False

                    if self._iteration % _MEMORY_CHECK_EVERY < batch_size:
                        self._yane._enforce_memory_limit()
                        guard = self._yane._resource_guard
                        while self._running and not guard.system_ok():
                            time.sleep(0.5)

                    if self._iteration % _GC_EVERY < batch_size and self._running:
                        gc.collect()
                        _return_memory_to_os()

                    now = time.perf_counter()
                    if now - last_emit >= _EMIT_INTERVAL_S:
                        last_emit = now
                        self._emit_update(self._iteration, best_fitness)

                    time.sleep(0)   # yield GIL to Qt event loop

                except Exception as exc:
                    self.error_occurred.emit(str(exc))
                    break
        finally:
            gc.enable()
            pool.terminate()
            pool.join()

        self._emit_final()

    def _emit_update(self, iteration: int, fitness: float) -> None:
        try:
            best = self._yane.get_best().copy()
        except RuntimeError:
            return
        mem = self._yane.population_memory_info()
        self.iteration_done.emit(iteration, fitness, best, mem)

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
