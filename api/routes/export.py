"""Genome export and behaviour-cloning endpoints.

Endpoints:
  GET  /export/python        — best genome as Python source
  GET  /export/symbolic      — best genome as symbolic formula
  POST /export/c_array       — best genome as C99 .h + .cc files
  POST /export/onnx          — best genome as ONNX model (requires onnx)
  GET  /export/wasm          — best genome as standalone HTML/JS
  POST /export/lottery_ticket — find + apply Lottery Ticket (IMP pruning)
  POST /clone                — behaviour cloning from JSON demonstrations
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from yane.api.state import state

router = APIRouter(tags=["export"])

_NOT_CONFIGURED = "NeuroEvolution not configured yet."
_NO_BEST = "No evaluated genomes yet — run training first."


def _best():
    if not state.is_configured:
        raise HTTPException(400, _NOT_CONFIGURED)
    try:
        return state.get_best()
    except RuntimeError:
        raise HTTPException(404, _NO_BEST)


# ---------------------------------------------------------------------------
# Python source export
# ---------------------------------------------------------------------------

@router.get("/python", response_class=PlainTextResponse,
            summary="Export best genome as Python source code")
def export_python() -> str:
    """Return the best genome as a self-contained Python function."""
    genome = _best()
    from yane.evolution.genome_export import genome_to_python
    return genome_to_python(genome)


# ---------------------------------------------------------------------------
# Symbolic regression export
# ---------------------------------------------------------------------------

class SymbolicRequest(BaseModel):
    format: str = Field("python", pattern=r"^(python|text|latex|sympy)$")
    input_names: list[str] | None = None
    fold_constants: bool = True


@router.post("/symbolic", summary="Export best genome as a symbolic formula")
def export_symbolic(req: SymbolicRequest) -> dict:
    """Convert the best genome to a closed-form symbolic expression.

    Raises 422 for cyclic (recurrent) genomes.
    """
    genome = _best()
    try:
        formula = genome.to_symbolic(
            input_names=req.input_names,
            format=req.format,
            fold_constants=req.fold_constants,
        )
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"format": req.format, "formula": formula}


# ---------------------------------------------------------------------------
# C-Array export
# ---------------------------------------------------------------------------

class CArrayRequest(BaseModel):
    prefix: str = Field("yane_net", min_length=1, max_length=64,
                        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")


@router.post("/c_array", summary="Export best genome as C99 embedded-deployment files")
def export_c_array(req: CArrayRequest) -> dict:
    """Generate C99 `.h` and `.cc` files and return their contents inline.

    The generated code depends only on ``<math.h>`` and is suitable for
    microcontroller deployment.
    """
    genome = _best()
    with tempfile.TemporaryDirectory() as tmp:
        from yane.evolution.tflite_export import genome_to_c_array
        h_path, cc_path = genome_to_c_array(genome, path=tmp, prefix=req.prefix)
        header = h_path.read_text(encoding="utf-8")
        source = cc_path.read_text(encoding="utf-8")
    return {
        "prefix": req.prefix,
        "header_filename": f"{req.prefix}.h",
        "source_filename": f"{req.prefix}.cc",
        "header": header,
        "source": source,
    }


# ---------------------------------------------------------------------------
# WASM / JS export
# ---------------------------------------------------------------------------

@router.get("/wasm", response_class=PlainTextResponse,
            summary="Export best genome as standalone HTML/JS (browser-runnable)")
def export_wasm() -> str:
    """Return the best genome as a self-contained HTML file with embedded JS."""
    genome = _best()
    from yane.evolution.wasm_export import genome_to_html
    return genome_to_html(genome)


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

@router.get("/onnx", summary="Export best genome as ONNX model (base64-encoded)")
def export_onnx(opset_version: int = 17, unroll_steps: int = 1) -> dict:
    """Return the ONNX model as a base64-encoded string.

    Requires the ``onnx`` package (``pip install onnx``).
    """
    genome = _best()
    try:
        from yane.evolution.onnx_export import genome_to_onnx
        import base64
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            tmp_path = tmp.name
        genome_to_onnx(genome, path=tmp_path, opset_version=opset_version, unroll_steps=unroll_steps)
        data = Path(tmp_path).read_bytes()
        Path(tmp_path).unlink(missing_ok=True)
        return {
            "format": "onnx",
            "opset_version": opset_version,
            "size_bytes": len(data),
            "base64": base64.b64encode(data).decode("ascii"),
        }
    except ImportError as e:
        raise HTTPException(501, str(e))


# ---------------------------------------------------------------------------
# Lottery Ticket (IMP pruning)
# ---------------------------------------------------------------------------

class LotteryTicketRequest(BaseModel):
    target_sparsity: float = Field(0.5, ge=0.0, lt=1.0)
    max_fitness_drop: float = Field(0.05, ge=0.0)
    iterations: int = Field(5, ge=1, le=50)
    apply: bool = Field(False, description="Apply the ticket to the best genome in-place.")
    fitness_fn_name: str | None = Field(
        None, description="Name of a registered fitness function (from POST /train/register_fn)."
    )


@router.post("/lottery_ticket", summary="Find (and optionally apply) the Lottery Ticket")
def lottery_ticket(req: LotteryTicketRequest) -> dict:
    """Run Iterative Magnitude Pruning on the best genome.

    Requires a server-side fitness function registered via
    ``POST /train/register_fn``.  Returns the ticket's sparsity and fitness.
    """
    genome = _best()
    from yane.api.routes.training import _fitness_registry
    name = req.fitness_fn_name
    if name is not None:
        fn = _fitness_registry.get(name)
        if fn is None:
            raise HTTPException(404, f"Fitness function {name!r} not registered.")
    elif _fitness_registry:
        fn = next(iter(_fitness_registry.values()))
    else:
        raise HTTPException(400, "No fitness function registered. POST /train/register_fn first.")

    from yane.evolution.sparse_neat import find_lottery_ticket, apply_ticket
    ticket = find_lottery_ticket(
        genome,
        fn,
        target_sparsity=req.target_sparsity,
        max_fitness_drop=req.max_fitness_drop,
        iterations=req.iterations,
    )
    if req.apply:
        apply_ticket(genome, ticket)

    return {
        "sparsity": ticket.sparsity,
        "fitness": ticket.fitness,
        "original_fitness": ticket.original_fitness,
        "n_active_connections": len(ticket.mask),
        "applied": req.apply,
    }


# ---------------------------------------------------------------------------
# Behaviour Cloning
# ---------------------------------------------------------------------------

class Demonstration(BaseModel):
    inputs: list[float]
    targets: list[float]


class CloneRequest(BaseModel):
    demonstrations: list[Demonstration] = Field(
        ..., min_length=1,
        description="List of (inputs, targets) pairs for supervised pre-training.",
    )
    n_steps: int = Field(200, ge=1, le=50000)
    sigma: float = Field(0.05, gt=0.0, le=5.0)
    seed_population: bool = Field(
        False,
        description="Replace the population seed with the cloned genome.",
    )
    noise_sigma: float = Field(
        0.05, ge=0.0, le=2.0,
        description="Weight noise added to each seeded copy (diversity).",
    )


@router.post("/clone", summary="Behaviour cloning: pre-train best genome on expert demonstrations")
def behaviour_clone(req: CloneRequest) -> dict:
    """Supervised pre-training via Lamarckian hill-climbing against expert demonstrations.

    The best current genome is refined to minimise MSE on the provided
    (inputs, targets) pairs.  Returns initial/final MSE and optionally
    seeds the population with the cloned genome.
    """
    if not state.is_configured:
        raise HTTPException(400, _NOT_CONFIGURED)
    demos = [(d.inputs, d.targets) for d in req.demonstrations]
    result = state.behaviour_clone(
        demonstrations=demos,
        n_steps=req.n_steps,
        sigma=req.sigma,
        seed_population=req.seed_population,
        noise_sigma=req.noise_sigma,
    )
    return {
        "initial_mse": result.initial_mse,
        "final_mse": result.final_mse,
        "compression_ratio": result.compression_ratio,
        "n_steps_run": result.n_steps_run,
        "seeded_population": req.seed_population,
        "cloned_fitness": result.cloned_genome.fitness,
    }
