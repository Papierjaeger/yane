"""Background threads for training and the API server."""
from __future__ import annotations
import gc
import math
import multiprocessing as mp
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from PySide6.QtCore import QThread, Signal

from yane.evolution.remote_evaluation import RemoteEvaluationClient
from yane.gui._mp_eval import _mp_evaluate, _mp_initializer, _episode_spawn_worker, _close_env, _FRAME_INTERVAL
from yane.gui.remote_config import RemoteEvaluationConfig
from yane.neuro_evolution import _return_memory_to_os

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
        remote_config: RemoteEvaluationConfig | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._yane = yane
        self._make_eval_fn = make_eval_fn
        self._render_cb = render_cb
        self._remote_config = remote_config or RemoteEvaluationConfig()
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

        if self._remote_config.enabled:
            self._run_remote(last_emit)
            return

        # Curriculum requires sequential evaluation: stage advances must be
        # atomic and the population re-evaluation after reset_stagnation() would
        # race with MP batch submissions.
        if self._yane._curriculum is not None:
            self.workers_resolved.emit(1)
            self.info_message.emit("Curriculum aktiv — Training läuft sequenziell.")
            self._run_sequential(last_emit)
            return

        from yane.gui._mp_eval import SpawnRequired
        use_spawn = isinstance(self._make_eval_fn, SpawnRequired)

        n_workers = getattr(self._yane, '_n_workers', 1)
        if n_workers == 0:
            self._run_multiprocess(mp.cpu_count(), last_emit, auto=True,
                                   use_spawn=use_spawn)
        elif n_workers > 1:
            self.workers_resolved.emit(n_workers)
            self._run_multiprocess(n_workers, last_emit, auto=False,
                                   use_spawn=use_spawn)
        else:
            self.workers_resolved.emit(1)
            if use_spawn:
                self.info_message.emit(
                    "Atari: läuft in separatem Prozess (kein Qt-Konflikt)."
                )
                self._run_multiprocess(1, last_emit, auto=False, use_spawn=True)
            else:
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

    def _run_remote(self, last_emit: float) -> None:
        cfg = self._remote_config
        self.workers_resolved.emit(len(cfg.worker_urls))
        self.info_message.emit(
            f"Remote Evaluation aktiv — {len(cfg.worker_urls)} Worker-URL(s)."
        )
        client = RemoteEvaluationClient(
            worker_urls=list(cfg.worker_urls),
            token=cfg.token,
            timeout_s=cfg.timeout_s,
            max_retries=cfg.max_retries,
        )
        batch_size = cfg.effective_batch_size
        try:
            if not self._yane._population._evaluated:
                seed = self._yane.next_genome()
                started = time.perf_counter()
                seed_results = client.evaluate_batch([seed])
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if not seed_results:
                    raise RuntimeError("Remote-Bootstrap-Evaluierung fehlgeschlagen.")
                _genome, seed_fitness = seed_results[0]
                self._yane.submit_fitness(seed_fitness, elapsed_ms)
                self._iteration += 1
                if (
                    self._yane.min_fitness is not None
                    and seed_fitness >= self._yane.min_fitness
                ):
                    self._running = False

            while self._running:
                while self._paused and self._running:
                    time.sleep(0.05)
                try:
                    genomes = self._yane.next_genome_batch(batch_size)
                    started = time.perf_counter()
                    remote_results = client.evaluate_batch(genomes)
                    elapsed_per = ((time.perf_counter() - started) * 1000.0) / max(
                        1, len(remote_results)
                    )
                    if not remote_results:
                        raise RuntimeError("Keine Remote-Evaluierung erfolgreich.")
                    results = [
                        (genome, fitness, elapsed_per)
                        for genome, fitness in remote_results
                    ]
                    self._yane.submit_fitness_batch(results)
                    self._iteration += len(results)

                    _, best_fitness = max(remote_results, key=lambda item: item[1])
                    if (
                        self._yane.min_fitness is not None
                        and best_fitness >= self._yane.min_fitness
                    ):
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
                    time.sleep(0)
                except Exception as exc:
                    self.error_occurred.emit(str(exc))
                    break
        finally:
            client.close()

        self._emit_final()

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
                          bootstrap_eval_ms: float | None = None,
                          use_spawn: bool = False) -> None:
        """Evaluate genomes in parallel subprocesses.

        Uses fork by default (fast, no pickling of closures).  Pass
        use_spawn=True for evaluators that conflict with the GUI process's
        loaded shared libraries (e.g. Atari/ALE vs PySide6's Qt); spawn
        workers start with a clean process image and avoid the conflict.
        make_eval_fn must be picklable when use_spawn=True.

        bootstrap_eval_ms: when set, skip the bootstrap eval (caller already
        measured representative eval time, e.g. after switching from sequential).
        """
        import multiprocessing as mp

        ctx = mp.get_context('spawn' if use_spawn else 'fork')

        # Build the pool first.  For spawn-based evaluators the bootstrap must
        # run inside a pool worker (gym.make can't be called in this process).
        # For fork-based evaluators the sequential bootstrap runs first so we
        # can measure eval speed before creating the pool.

        if bootstrap_eval_ms is not None:
            eval_ms = bootstrap_eval_ms
        elif not use_spawn:
            # Sequential bootstrap: measure eval speed before creating the pool.
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
        else:
            eval_ms = 500.0  # generous default; EMA will refine quickly

        # Decide how many workers to use.
        #
        # Cost model (for a batch of k genomes):
        #   sequential:   k × eval_ms
        #   w workers:    k × eval_ms / w  +  overhead
        _OVERHEAD_MS  = 16.0
        batch_size    = self._yane._population.max_size
        n_workers_max = n_workers   # preserve cpu_count cap before any override
        chosen        = _optimal_workers(eval_ms, batch_size, _OVERHEAD_MS, n_workers)
        seq_time      = batch_size * eval_ms

        if auto:
            if chosen <= 1:
                chosen = 2
            n_workers = chosen
            self.workers_resolved.emit(n_workers)
            self.info_message.emit(
                f"Auto → {n_workers} Worker "
                f"({eval_ms:.2f}ms/Genome, EMA justiert laufend)"
            )
        elif chosen <= 1 and not use_spawn:
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

        # Fork: expose input transform via module-level slot so child inherits it.
        # Spawn: pass it explicitly via initargs (fork globals aren't inherited).
        input_transform = getattr(self._yane, '_input_transform', None)
        if not use_spawn:
            import yane.gui._mp_eval as _mp_eval_mod
            _mp_eval_mod._mp_input_transform = input_transform

        def _make_pool(nw: int):
            extra = (input_transform,) if use_spawn else ()
            return ctx.Pool(
                processes=nw,
                initializer=_mp_initializer,
                initargs=(
                    self._make_eval_fn, None,
                    self._yane._runner.n_evaluations,
                    self._yane._runner.aggregation,
                    self._yane._runner.sigma_penalty,
                ) + extra,
            )

        try:
            pool = _make_pool(n_workers)
        except Exception as exc:
            self.error_occurred.emit(f"Multiprocessing Pool konnte nicht erstellt werden: {exc}")
            return

        # For spawn-based evaluators the seed genome hasn't been evaluated yet
        # (we can't call make_eval_fn in this process).  Do it now via the pool.
        if use_spawn and bootstrap_eval_ms is None:
            try:
                seed_g = self._yane.next_genome()
                (seed_fit, seed_ms), = pool.map(_mp_evaluate, [seed_g])
                eval_ms = seed_ms
                self._yane.submit_fitness(seed_fit, eval_ms)
                self._iteration += 1
            except Exception as exc:
                pool.terminate(); pool.join()
                self.error_occurred.emit(str(exc))
                return

        # Spawn mode: 1 genome per pool.map() call so the outer loop ticks at
        # ~eval_ms and the regular _EMIT_INTERVAL_S gate gives smooth updates
        # without imap IPC overhead or mid-batch signal bursts.
        # Fork mode: fill workers to amortise IPC/fork overhead.
        batch_size = n_workers if use_spawn else max(n_workers * 4, self._yane._population.max_size)

        # Adaptive scaling: eval_ema tracks eval time per genome (ms).
        _ALPHA         = 0.25
        _RESCALE_EVERY = 10
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
                    map_ms = (time.perf_counter() - t_map) * 1000.0

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
                            if use_spawn:
                                # Cannot fall back to sequential for
                                # spawn-required evaluators — keep 1 worker.
                                n_workers = 1
                                pool = _make_pool(n_workers)
                                self.workers_resolved.emit(1)
                            else:
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
        try:
            meta = self._yane.get_meta_optimizer_diagnostics()
            if meta.get("enabled"):
                mem["meta_optimizer"] = meta
        except Exception:
            pass
        try:
            fg = self._yane.get_feature_gating_diagnostics()
            if fg.get("enabled"):
                mem["feature_gating"] = fg
        except Exception:
            pass
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


class AutoSetupWorker(QThread):
    """Profiles the problem and configures P0 meta-adaptive features in background.

    Runs profile_problem (~20 warmup evals) then sets up KB, MetaOptimizer,
    and FeatureGating so the normal TrainingWorker can start immediately after.
    """

    setup_done     = Signal(dict)   # profile summary + applied-param info
    error_occurred = Signal(str)
    status_message = Signal(str)

    def __init__(self, yane, make_eval_fn: "Callable", n_warmup: int = 20,
                 parent=None) -> None:
        super().__init__(parent)
        self._yane = yane
        self._make_eval_fn = make_eval_fn
        self._n_warmup = n_warmup

    def run(self) -> None:
        from yane.evolution.auto_train import apply_cold_start_defaults, pick_pop_size
        try:
            self.status_message.emit("Profiling problem…")
            eval_fn = self._make_eval_fn(None)
            profile = self._yane.profile_problem(eval_fn, n_warmup=self._n_warmup)

            pop_size = pick_pop_size(profile)
            self._yane.set_population_size(pop_size)
            self._yane.set_matrix_forward(True)
            _species_for_task = {
                "classification": 10, "regression": 10,
                "rl_discrete": 5, "rl_continuous": 5,
            }
            self._yane.set_target_species(
                _species_for_task.get(profile.task_type, 5)
            )

            if self._yane._knowledge_base is None:
                self._yane.set_knowledge_base()
            suggestions: list = []
            try:
                suggestions = self._yane._knowledge_base.suggest(profile, top_k=3)
            except Exception:
                pass

            applied_params: dict = {}
            kb_used = False
            kb_conf = 0.0
            if suggestions and suggestions[0].get("confidence", 0.0) >= 0.25:
                kb_used = True
                kb_conf = suggestions[0].get("confidence", 0.0)
                for pname, pval in (suggestions[0].get("params") or {}).items():
                    try:
                        self._yane.set_param(pname, pval)
                        applied_params[pname] = pval
                    except Exception:
                        pass
            else:
                applied_params = apply_cold_start_defaults(self._yane, profile)

            self.status_message.emit("Configuring MetaOptimizer & FeatureGating…")
            self._yane.set_meta_optimizer(
                enabled=True, tune_interval=5,
                plateau_patience=30, phase_min_gens=10, max_overhead_pct=10.0,
            )
            # Use the same minimums as auto_train() so GUI and API behave
            # consistently.  The exact values scale with run length in
            # auto_train(); here we use the floor values (≥50/≥20/≥60) since
            # we don't know the final iteration budget at setup time.
            self._yane.set_auto_features(
                enabled=True, max_concurrent=2,
                test_interval=50, test_duration=20, degradation_patience=60,
                include_heavyweight=profile.task_type in ("rl_discrete", "rl_continuous"),
            )

            self.setup_done.emit({
                "profile":        profile,
                "task_type":      profile.task_type,
                "difficulty":     round(profile.estimated_difficulty, 3),
                "noise":          round(profile.noise_level, 3),
                "temporal":       round(profile.temporal_dependency, 3),
                "pop_size":       pop_size,
                "kb_used":        kb_used,
                "kb_conf":        round(kb_conf, 3),
                "kb_entries":     len(self._yane._knowledge_base) if self._yane._knowledge_base else 0,
                "applied_params": applied_params,
                "cold_start":     not kb_used,
            })
        except Exception as exc:
            self.error_occurred.emit(str(exc))


class EpisodeRunner(QThread):
    """Loops gym episodes with render until stop() is called.

    For SpawnRequired examples (e.g. Atari/ALE) the evaluation runs in a
    dedicated spawn subprocess so that ale_py (system Qt6) and PySide6
    (bundled Qt6) never share the same process.
    """

    frame_ready   = Signal(object)  # numpy array frame
    score_updated = Signal(float)   # cumulative reward after each step

    def __init__(self, genome, example, parent=None) -> None:
        super().__init__(parent)
        self._genome  = genome
        self._example = example
        self._running = True

    def run(self) -> None:
        from yane.gui._mp_eval import SpawnRequired
        # Check the underlying eval-maker, not the ExampleConfig wrapper.
        maker = getattr(self._example, "make_eval", None)
        if maker is not None and isinstance(maker, SpawnRequired):
            self._run_spawned(maker)
        else:
            self._run_inline()

    def _run_inline(self) -> None:
        """Original inline evaluation path (non-Atari envs)."""
        last_frame = 0.0
        step_delay = 0.0

        def render_cb(frame):
            nonlocal last_frame
            if not self._running:
                return
            now = time.perf_counter()
            if now - last_frame < _FRAME_INTERVAL:
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

    def _run_spawned(self, maker) -> None:
        """Run evaluation in a spawn subprocess (avoids Qt6 conflict)."""
        ctx = mp.get_context("spawn")
        frame_queue: mp.Queue = ctx.Queue()
        score_queue: mp.Queue = ctx.Queue()
        cmd_queue: mp.Queue = ctx.Queue()

        worker = ctx.Process(
            target=_episode_spawn_worker,
            args=(maker, self._genome, frame_queue, score_queue, cmd_queue),
            daemon=True,
        )
        worker.start()

        last_frame = 0.0
        try:
            while self._running and worker.is_alive():
                # Pump frames from the subprocess
                try:
                    frame = frame_queue.get(timeout=0.05)
                    if frame is None:  # sentinel: episode ended
                        break
                    now = time.perf_counter()
                    if now - last_frame >= _FRAME_INTERVAL:
                        last_frame = now
                        self.frame_ready.emit(frame)
                except (queue.Empty, EOFError):
                    pass

                # Drain all queued scores; emit only the latest.
                latest_score = None
                while True:
                    try:
                        latest_score = score_queue.get_nowait()
                    except (queue.Empty, EOFError):
                        break
                if latest_score is not None:
                    self.score_updated.emit(latest_score)
        finally:
            self._running = False
            try:
                cmd_queue.put("stop")
            except Exception:
                pass
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2)
            cmd_queue.cancel_join_thread()
            for q in (frame_queue, score_queue, cmd_queue):
                q.close()

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
