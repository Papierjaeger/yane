"""YANE API server — run with:  uvicorn yane.api.server:app --reload"""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from yane.neuro_evolution import NeuroEvolution
from yane.api.routes import network, population as pop_routes
from yane.util.logger import setup_logging as _setup_log, log_info, log_warning, log_error


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
    max_nodes: Optional[int] = Field(None, ge=2, description="Hard cap on nodes per genome")
    max_connections: Optional[int] = Field(None, ge=1, description="Hard cap on connections per genome")
    n_initial_hidden: int = Field(0, ge=0, description="Hidden nodes added to the initial genome")
    stateful: bool = Field(True, description="Output nodes persist their value across ticks")
    # Population
    population_size: Optional[int] = Field(None, ge=2, description="Number of genomes in the population")
    target_species: Optional[int] = Field(None, ge=1, description="Target number of species")
    # Lamarck
    lamarck_steps: Optional[int] = Field(None, ge=0, description="Explicit Lamarck steps (0 = adaptive)")
    lamarck_sigma: Optional[float] = Field(None, gt=0.0, description="Lamarck sigma multiplier")
    lamarck_adaptive_max_steps: Optional[int] = Field(None, ge=0, description="Adaptive Lamarck max steps")
    lamarck_adaptive_top_k: Optional[float] = Field(None, gt=0.0, le=1.0, description="Adaptive Lamarck top-k fraction")
    # Reproducibility
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


@app.post("/configure", tags=["setup"], summary="Initialise topology and evolutionary settings")
def configure(req: ConfigureRequest) -> dict:
    if req.seed is not None:
        state.set_seed(req.seed)
    if req.population_size is not None:
        state.set_population_size(req.population_size)
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
    if req.lamarck_steps is not None:
        sigma = req.lamarck_sigma if req.lamarck_sigma is not None else 1.0
        state.set_lamarck(n_steps=req.lamarck_steps, sigma=sigma)
    elif req.lamarck_adaptive_max_steps is not None or req.lamarck_adaptive_top_k is not None:
        state.set_lamarck_adaptive(
            max_steps=req.lamarck_adaptive_max_steps if req.lamarck_adaptive_max_steps is not None else 3,
            top_k=req.lamarck_adaptive_top_k if req.lamarck_adaptive_top_k is not None else 0.2,
            sigma=req.lamarck_sigma if req.lamarck_sigma is not None else 1.0,
        )
    return state._config_dict()


@app.get("/health", tags=["setup"])
def health() -> dict:
    return {"status": "ok"}
