"""Multiprocessing helpers — deliberately PySide6-free.

Spawn workers import this module to resolve _mp_initializer / _mp_evaluate.
Keeping PySide6 out means the Qt6 conflict between ale_py (system Qt6) and
PySide6 (PyPI-bundled Qt6) never occurs in those workers.
"""
from __future__ import annotations
import time
import traceback

from yane.core.genome import Genome
from yane.neuro_evolution import _aggregate_fitnesses

class SpawnRequired:
    """Mixin for eval-maker classes that must run in a spawn subprocess.

    Inherit from this instead of setting _requires_clean_process manually.
    TrainingWorker checks isinstance(make_eval_fn, SpawnRequired) to decide
    whether to use spawn (clean process, no Qt6 conflict) instead of fork.
    """
    __slots__ = ()


_mp_eval_fn         = None
_mp_n_evaluations   = 1
_mp_aggregation     = "mean"
_mp_sigma_penalty   = 0.0
_mp_input_transform = None   # set before forking (fork mode) or via initargs (spawn mode)


def _mp_initializer(make_eval_fn, render_cb,
                    n_evaluations=1, aggregation="mean", sigma_penalty=0.0,
                    input_transform=None):
    global _mp_eval_fn, _mp_n_evaluations, _mp_aggregation, _mp_sigma_penalty, _mp_input_transform
    _mp_eval_fn       = make_eval_fn(render_cb)
    _mp_n_evaluations = n_evaluations
    _mp_aggregation   = aggregation
    _mp_sigma_penalty = sigma_penalty
    if input_transform is not None:
        _mp_input_transform = input_transform


def _mp_evaluate(genome: Genome) -> tuple[float, float]:
    if _mp_input_transform is not None:
        _t = _mp_input_transform
        _orig = genome.forward
        genome.__dict__["forward"] = lambda data: _orig(_t(data))
    try:
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
    finally:
        if _mp_input_transform is not None:
            genome.__dict__.pop("forward", None)


_FRAME_INTERVAL = 1.0 / 30  # display cap shared with worker.py


# ---------------------------------------------------------------------------
# Episode-runner spawn worker (also deliberately PySide6-free)
# ---------------------------------------------------------------------------

def _close_env(eval_fn) -> None:
    """Safely close a gym environment attached to *eval_fn*."""
    env = getattr(eval_fn, "_env", None)
    if env is not None:
        try:
            env.close()
        except Exception:
            pass


def _episode_spawn_worker(maker, genome, frame_queue, score_queue, cmd_queue) -> None:
    """Run evaluation episodes in a clean spawn subprocess.

    Parameters
    ----------
    maker : SpawnRequired
        An eval-maker instance (e.g. _AtariEvalMaker).  Must be picklable.
    genome : Genome
        The genome to evaluate.
    frame_queue : multiprocessing.Queue
        Queue for sending rendered frames (numpy arrays) back to the GUI.
        A ``None`` value signals episode end.
    score_queue : multiprocessing.Queue
        Queue for sending cumulative scores back to the GUI.
    cmd_queue : multiprocessing.Queue
        Queue for receiving stop commands from the GUI thread.
    """
    import queue

    _running = True

    def _render_cb(frame):
        if not _running:
            return
        try:
            frame_queue.put(frame, timeout=0.01)
        except queue.Full:
            pass

    def _step_cb(total):
        nonlocal _running
        if not _running:
            return 0.0
        try:
            score_queue.put(total, timeout=0.01)
        except queue.Full:
            pass
        try:
            cmd = cmd_queue.get_nowait()
            if cmd == "stop":
                _running = False
        except queue.Empty:
            pass
        return 0.0

    eval_fn = maker(_render_cb, _step_cb, demo=True)

    try:
        while _running:
            eval_fn(genome)
    except Exception:
        traceback.print_exc()
    finally:
        _close_env(eval_fn)
        try:
            frame_queue.put(None, timeout=0.5)  # sentinel
        except Exception:
            pass
