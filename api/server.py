"""YANE API server — run with:  uvicorn yane.api.server:app --reload"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


@app.post("/configure", tags=["setup"], summary="Initialise with input/output count")
def configure(n_inputs: int, n_outputs: int) -> dict:
    state.configure(n_inputs, n_outputs)
    return {"ok": True, "n_inputs": n_inputs, "n_outputs": n_outputs}


@app.get("/health", tags=["setup"])
def health() -> dict:
    return {"status": "ok"}
