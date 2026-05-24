"""YANE API server — run with:  uvicorn yane.api.server:app --reload"""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from yane.neuro_evolution import NeuroEvolution
from yane.api.routes import network, population as pop_routes
from yane.util.logger import setup_logging as _setup_log, log_info, log_warning, log_error
from yane.util.presets import list_presets, load_preset


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Set up structured logging on startup."""
    _setup_log("api")
    log_info("API server starting")
    yield
    log_info("API server shutting down")


app = FastAPI(
    title="YANE",
    description="Yet Another Neuro Evolution — HTTP interface",
    version="0.1.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

state: NeuroEvolution = NeuroEvolution()

app.include_router(network.router, prefix="/network")
app.include_router(pop_routes.router, prefix="/population")


class ConfigureRequest(BaseModel):
    # Topology
    n_inputs: int = Field(..., ge=1, description="Number of input nodes")
    n_outputs: int = Field(..., ge=1, description="Number of output nodes")
    max_nodes: int | None = Field(None, ge=2, description="Hard cap on nodes per genome")
    max_connections: int | None = Field(None, ge=1, description="Hard cap on connections per genome")
    n_initial_hidden: int = Field(0, ge=0, description="Hidden nodes added to the initial genome")
    stateful: bool = Field(True, description="Output nodes persist their value across ticks")
    # Population
    population_size: int | None = Field(None, ge=2, description="Number of genomes in the population")
    target_species: int | None = Field(None, ge=0, description="Target number of species (0 = auto)")
    n_workers: int | None = Field(None, ge=0, description="Number of parallel workers (0 = auto)")
    # Elitism
    elite_count: int | None = Field(None, ge=0, description="Global elite genomes to protect from pruning")
    species_elite_count: int | None = Field(None, ge=0, description="Elite genomes per species to protect")
    # Lamarck
    lamarck_steps: int | None = Field(None, ge=0, description="Explicit Lamarck steps (0 = adaptive)")
    lamarck_sigma: float | None = Field(None, gt=0.0, description="Lamarck sigma multiplier")
    lamarck_adaptive_max_steps: int | None = Field(None, ge=0, description="Adaptive Lamarck max steps")
    lamarck_adaptive_top_k: float | None = Field(None, gt=0.0, le=1.0, description="Adaptive Lamarck top-k fraction")
    # Convergence & early stopping
    convergence_spread_eps: float | None = Field(None, gt=0.0, description="Fitness IQR threshold for convergence")
    convergence_min_stagnation: float | None = Field(None, gt=0.0, description="Min stagnation fraction for convergence")
    early_stop_factor: float | None = Field(None, ge=0.0, description="Early stopping tolerance factor")
    # Fitness
    fitness_shaping: bool | None = Field(None, description="Enable rank-based fitness shaping")
    interspecies_crossover_rate: float | None = Field(None, ge=0.0, le=1.0, description="Interspecies crossover probability")
    speciation_metric: str | None = Field(None, pattern=r"^(topology|topology_no_disabled)$")
    # Resource limits
    max_process_gb: float | None = Field(None, gt=0.0, description="Hard RAM cap per process in GB")
    # Reproducibility
    seed: int | None = Field(None, description="Random seed for reproducibility")


@app.post("/configure", tags=["setup"], summary="Initialise topology and evolutionary settings")
def configure(req: ConfigureRequest) -> dict:
    if req.seed is not None:
        state.set_seed(req.seed)
    if req.population_size is not None:
        state.set_population_size(req.population_size)
    if req.n_workers is not None:
        state.set_n_workers(req.n_workers)
    state.configure(
        req.n_inputs,
        req.n_outputs,
        max_nodes=req.max_nodes,
        max_connections=req.max_connections,
        n_initial_hidden=req.n_initial_hidden,
        stateful=req.stateful,
    )
    if req.target_species is not None:
        state.set_target_species(req.target_species)
    if req.elite_count is not None:
        sec = req.species_elite_count if req.species_elite_count is not None else 1
        state.set_elitism(req.elite_count, sec)
    if req.lamarck_steps is not None:
        sigma = req.lamarck_sigma if req.lamarck_sigma is not None else 1.0
        state.set_lamarck(n_steps=req.lamarck_steps, sigma=sigma)
    elif req.lamarck_adaptive_max_steps is not None or req.lamarck_adaptive_top_k is not None:
        state.set_lamarck_adaptive(
            max_steps=req.lamarck_adaptive_max_steps if req.lamarck_adaptive_max_steps is not None else 3,
            top_k=req.lamarck_adaptive_top_k if req.lamarck_adaptive_top_k is not None else 0.2,
            sigma=req.lamarck_sigma if req.lamarck_sigma is not None else 1.0,
        )
    if req.convergence_spread_eps is not None:
        stag = req.convergence_min_stagnation if req.convergence_min_stagnation is not None else 1.0
        state.set_convergence_stop(req.convergence_spread_eps, stag)
    if req.early_stop_factor is not None:
        state.set_early_stopping(req.early_stop_factor)
    if req.fitness_shaping is not None:
        state.set_fitness_shaping(req.fitness_shaping)
    if req.interspecies_crossover_rate is not None:
        state.set_interspecies_crossover(req.interspecies_crossover_rate)
    if req.speciation_metric is not None:
        state.set_speciation_metric(req.speciation_metric)
    if req.max_process_gb is not None:
        state.set_resource_limits(max_process_gb=req.max_process_gb)
    return state._config_dict()


@app.get("/health", tags=["setup"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/presets", tags=["setup"], summary="List available experiment presets")
def presets() -> list[dict]:
    return [p.to_json() | {"path": str(p.path)} for p in list_presets()]


@app.get("/presets/{name}", tags=["setup"], summary="Load an experiment preset")
def preset(name: str) -> dict:
    try:
        p = load_preset(name)
        return p.to_json() | {"path": str(p.path)}
    except FileNotFoundError:
        raise HTTPException(404, f"Preset not found: {name}")


# ── Checkpoint endpoints ────────────────────────────────────────────────────


@app.post("/checkpoint/save", tags=["checkpoint"], summary="Save population to a checkpoint file")
def save_checkpoint(path: str) -> dict:
    """Save the current population and InnovationTracker to *path* (``.pkl``)."""
    if not state.is_configured:
        raise HTTPException(400, "NeuroEvolution not configured yet.")
    try:
        state.save_checkpoint(path)
        log_info("Checkpoint saved  path=%s", path)
        return {"ok": True, "path": path}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/checkpoint/load", tags=["checkpoint"], summary="Load population from a checkpoint file")
def load_checkpoint(path: str) -> dict:
    """Replace the current population with one loaded from *path* (``.pkl``).

    The current configure() state is overwritten by the checkpoint contents.
    """
    try:
        state.load_checkpoint(path)
        log_info("Checkpoint loaded  path=%s", path)
        return {"ok": True, "path": path, **state._config_dict()}
    except FileNotFoundError:
        raise HTTPException(404, f"Checkpoint not found: {path}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Diagnostics endpoint ────────────────────────────────────────────────────


@app.get("/population/diagnostics", tags=["population"],
         summary="Full training diagnostics (same data as GUI Debug tab)")
def diagnostics() -> dict:
    """Return ``population_memory_info()`` plus the current best genome."""
    if not state.is_configured:
        raise HTTPException(400, "NeuroEvolution not configured yet.")
    mem = state.population_memory_info()
    try:
        best = state.get_best()
        mem["best_genome"] = best.memory_info()
        mem["best_fitness"] = best.fitness
    except RuntimeError:
        mem["best_genome"] = None
        mem["best_fitness"] = None
    return mem


# ── Best genome export ──────────────────────────────────────────────────────


@app.get("/population/best/export", tags=["population"],
         summary="Export the best genome as structured JSON")
def export_best() -> dict:
    """Return the best genome's full structure (nodes, connections, weights)."""
    if not state.is_configured:
        raise HTTPException(400, "NeuroEvolution not configured yet.")
    try:
        genome = state.get_best()
    except RuntimeError as e:
        raise HTTPException(404, str(e))
    info = genome.memory_info()
    nodes = []
    for node in genome.nodes:
        nodes.append({
            "innovation": node.innovation,
            "type": node.type.name,
            "activation": node.activation_name,
            "input_scale": node.input_scale,
            "output_scale": node.output_scale,
            "persistent": node.persistent,
            "value": node.value,
        })
    connections = []
    for conn in genome.connections:
        connections.append({
            "innovation": conn.innovation,
            "source": conn.source.innovation,
            "target": conn.target.innovation,
            "weight": conn.weight,
            "enabled": conn.enabled,
        })
    return {
        "fitness": genome.fitness,
        "n_inputs": info["input_nodes"],
        "n_outputs": info["output_nodes"],
        "n_nodes": info["nodes"],
        "n_connections": info["connections"],
        "nodes": nodes,
        "connections": connections,
    }
