"""Async training endpoints.

The evaluator lives in the external application.  Three patterns are supported:

1. **External-loop** (manual): ``GET /population/next`` → eval externally →
   ``POST /population/fitness``.  No endpoint here required; this is the
   existing manual-loop API.

2. **Background train** (server-side fitness fn): ``POST /train/start`` launches
   training in a background thread using a registered Python fitness function.
   Not REST-able for arbitrary callables — use pattern 1 for external evaluators.

3. **Auto-train**: ``POST /train/auto`` calls ``auto_train()`` in a background
   thread with configurable options.

State machine:
  idle → running → stopped|finished
  running → stopping (POST /train/stop requested)
"""
from __future__ import annotations

import threading
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from yane.api.state import state

router = APIRouter(tags=["training"])

# ---------------------------------------------------------------------------
# Shared training state (protected by a lock)
# ---------------------------------------------------------------------------

_lock = threading.Lock()

_train_state: dict[str, Any] = {
    "status": "idle",       # idle | running | stopping | finished | failed
    "stop_reason": None,
    "best_fitness": None,
    "iterations": 0,
    "elapsed_s": 0.0,
    "error": None,
    "auto_result": None,    # AutoTrainResult when auto_train finishes
    "started_at": None,
    "finished_at": None,
}
_train_thread: threading.Thread | None = None


def _set(key: str, value: Any) -> None:
    with _lock:
        _train_state[key] = value


def _update(updates: dict) -> None:
    with _lock:
        _train_state.update(updates)


def _wait_for_idle(timeout: float = 10.0) -> bool:
    """Block until the background training thread has fully exited.

    Joins the thread directly rather than polling status to avoid
    race conditions between the status update and actual thread completion.
    Returns True if idle within *timeout* seconds, False on timeout.
    """
    thread = _train_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=timeout)
        return not thread.is_alive()
    return True


# ---------------------------------------------------------------------------
# GET /train/status
# ---------------------------------------------------------------------------

@router.get("/status", summary="Current training status and live metrics")
def train_status() -> dict:
    """Return the current training state.

    - ``status``: ``"idle"`` | ``"running"`` | ``"stopping"`` | ``"finished"`` | ``"failed"``
    - ``best_fitness``: latest best fitness (``null`` if not started)
    - ``iterations``: evaluations completed
    - ``elapsed_s``: wall-clock seconds since training started
    - ``stop_reason``: why training stopped (``null`` while running)
    - ``error``: error message if status is ``"failed"``
    """
    with _lock:
        snap = dict(_train_state)
    # Live elapsed calculation
    if snap["status"] == "running" and snap["started_at"] is not None:
        snap["elapsed_s"] = time.monotonic() - snap["started_at"]
    return snap


# ---------------------------------------------------------------------------
# POST /train/stop
# ---------------------------------------------------------------------------

@router.post("/stop", summary="Request graceful stop of the running training")
def train_stop() -> dict:
    """Signal the background training thread to stop after the current iteration."""
    with _lock:
        if _train_state["status"] not in ("running",):
            raise HTTPException(400, f"Training is not running (status={_train_state['status']!r}).")
        _train_state["status"] = "stopping"
    state.stop_requested = True
    return {"ok": True, "message": "Stop requested — training will finish the current iteration."}


# ---------------------------------------------------------------------------
# POST /train/reset
# ---------------------------------------------------------------------------

@router.post("/reset", summary="Reset training state to idle (after finished/failed)")
def train_reset() -> dict:
    """Reset state to idle so a new training run can be started.

    Waits up to 5 s for any background thread to finish before resetting.
    """
    _wait_for_idle(timeout=5.0)
    with _lock:
        if _train_state["status"] == "running":
            raise HTTPException(400, "Training is still running. POST /train/stop first.")
        _train_state.update({
            "status": "idle",
            "stop_reason": None,
            "best_fitness": None,
            "iterations": 0,
            "elapsed_s": 0.0,
            "error": None,
            "auto_result": None,
            "started_at": None,
            "finished_at": None,
        })
    state.stop_requested = False
    return {"ok": True}


# ---------------------------------------------------------------------------
# POST /train/auto — background auto_train()
# ---------------------------------------------------------------------------

class AutoTrainRequest(BaseModel):
    n_inputs: int = Field(..., ge=1)
    n_outputs: int = Field(..., ge=1)
    fitness_fn_name: str = Field(
        ...,
        description="Name of a fitness function registered via POST /train/register_fn. "
                    "This function is used for both problem profiling and training.",
    )
    target_fitness: float | None = None
    max_time_seconds: float | None = Field(None, gt=0.0)
    max_iterations: int | None = Field(None, ge=1)
    problem_name: str | None = None
    n_warmup: int = Field(20, ge=5, le=500)


@router.post("/auto", summary="Run auto_train() (zero-config) in background")
def train_auto(req: AutoTrainRequest) -> dict:
    """Start ``auto_train()`` in a background thread using a registered fitness function.

    **Workflow:**

    1. Register your fitness function:
       ``POST /train/register_fn  {"name": "my_fn", "source": "def fitness_fn(genome): ..."}``

    2. Launch auto-train:
       ``POST /train/auto  {"n_inputs": 4, "n_outputs": 2, "fitness_fn_name": "my_fn"}``

    3. Poll for progress:
       ``GET /train/status``

    ``auto_train()`` automatically profiles the problem, selects hyperparameters
    via the Knowledge Base, enables promising research features via Feature Gating,
    and runs the MetaOptimizer — all without manual ``set_*()`` calls.

    Returns immediately with ``{"ok": true, "status": "running"}``.
    The full ``auto_config_report`` is available in ``GET /train/status`` once finished.
    """
    fn = _fitness_registry.get(req.fitness_fn_name)
    if fn is None:
        raise HTTPException(
            404,
            f"Fitness function {req.fitness_fn_name!r} not registered. "
            "Register it first with POST /train/register_fn.",
        )

    with _lock:
        if _train_state["status"] == "running":
            raise HTTPException(409, "Training already running.")

    def _run() -> None:
        _update({"status": "running", "started_at": time.monotonic(), "error": None})
        try:
            result = state.auto_train(
                fn,
                n_inputs=req.n_inputs,
                n_outputs=req.n_outputs,
                target_fitness=req.target_fitness,
                max_time_seconds=req.max_time_seconds,
                problem_name=req.problem_name,
                n_warmup=req.n_warmup,
            )
            _update({
                "status": "finished",
                "stop_reason": "auto_train_complete",
                "best_fitness": result.final_fitness,
                "iterations": result.total_generations,
                "auto_result": result.auto_config_report,
                "finished_at": time.monotonic(),
            })
        except Exception as exc:
            _update({"status": "failed", "error": str(exc), "finished_at": time.monotonic()})

    global _train_thread
    _train_thread = threading.Thread(target=_run, daemon=True)
    _train_thread.start()
    return {"ok": True, "status": "running", "fitness_fn": req.fitness_fn_name}


# ---------------------------------------------------------------------------
# POST /train/iterations — run N iterations of the manual evaluation loop
# ---------------------------------------------------------------------------

class TrainIterationsRequest(BaseModel):
    n_iterations: int = Field(100, ge=1, le=100000,
                              description="Number of evaluate→submit cycles to run "
                                          "using the currently registered fitness function.")
    fitness_fn_name: str | None = Field(
        None,
        description="Name of a pre-registered fitness function (see POST /train/register_fn). "
                    "Leave null to use the last registered function.",
    )


# Simple in-process registry for named fitness functions uploaded as source code.
# NOT production-safe (exec of arbitrary code) — disable via config for public APIs.
_fitness_registry: dict[str, Any] = {}


class RegisterFitnessRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    source: str = Field(
        ...,
        description="Python source defining a function named 'fitness_fn(genome) -> float'. "
                    "WARNING: executed server-side — only use in trusted environments.",
    )


@router.post("/register_fn",
             summary="Register a named server-side fitness function (trusted environments only)")
def register_fitness_fn(req: RegisterFitnessRequest) -> dict:
    """Compile and register a named fitness function from Python source.

    The source must define a top-level function ``fitness_fn(genome) -> float``.

    **Security warning**: this executes arbitrary Python on the server.
    Disable this endpoint (or restrict with auth) in public-facing deployments.
    """
    ns: dict = {}
    try:
        exec(compile(req.source, "<api_fitness_fn>", "exec"), ns)  # noqa: S102
    except SyntaxError as e:
        raise HTTPException(422, f"Syntax error in fitness function: {e}")
    if "fitness_fn" not in ns:
        raise HTTPException(422, "Source must define a function named 'fitness_fn'.")
    _fitness_registry[req.name] = ns["fitness_fn"]
    return {"ok": True, "name": req.name, "registered_functions": list(_fitness_registry)}


@router.post("/iterations",
             summary="Run N training iterations with a server-side fitness function")
def train_iterations(req: TrainIterationsRequest) -> dict:
    """Run *n_iterations* training steps synchronously using a registered fitness function.

    Returns after all iterations complete.  For long runs use ``POST /train/auto``.
    For external evaluators use the manual ``/population/next`` + ``/population/fitness`` loop.
    """
    if not state.is_configured:
        raise HTTPException(400, "NeuroEvolution not configured yet.")

    name = req.fitness_fn_name
    if name is not None:
        fn = _fitness_registry.get(name)
        if fn is None:
            raise HTTPException(404, f"Fitness function {name!r} not registered.")
    elif _fitness_registry:
        fn = next(iter(_fitness_registry.values()))
    else:
        raise HTTPException(400,
                            "No fitness function registered. POST /train/register_fn first.")

    with _lock:
        if _train_state["status"] == "running":
            raise HTTPException(409, "Training already running.")

    _update({"status": "running", "started_at": time.monotonic(), "error": None})
    try:
        state.set_max_iterations(req.n_iterations)
        state.train(fn)
        try:
            best = state.get_best()
            best_fit = best.fitness
        except Exception:
            best_fit = None
        _update({
            "status": "finished",
            "stop_reason": getattr(state, "stop_reason", "max_iterations"),
            "best_fitness": best_fit,
            "iterations": req.n_iterations,
            "elapsed_s": time.monotonic() - _train_state["started_at"],
            "finished_at": time.monotonic(),
        })
        return {"ok": True, "best_fitness": best_fit, "iterations": req.n_iterations}
    except Exception as exc:
        _update({"status": "failed", "error": str(exc), "finished_at": time.monotonic()})
        raise HTTPException(500, str(exc))
