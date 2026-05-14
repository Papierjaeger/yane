"""YANE API server — run with:  uvicorn yane.api.server:app --reload"""
from fastapi import FastAPI

from yane.neuro_evolution import NeuroEvolution
from yane.api.routes import network, population as pop_routes

app = FastAPI(
    title="YANE",
    description="Yet Another Neuro Evolution — HTTP interface",
    version="0.1.0",
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
