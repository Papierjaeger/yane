from pydantic import BaseModel


class InputRequest(BaseModel):
    data: list[float]


class OutputResponse(BaseModel):
    outputs: list[float]


class FitnessRequest(BaseModel):
    fitness: float


class PopulationStatus(BaseModel):
    size: int
    evaluated: int
    unevaluated: int
    best_fitness: float | None


class GenomeInfo(BaseModel):
    fitness: float
    n_nodes: int
    n_connections: int
    n_inputs: int
    n_outputs: int
