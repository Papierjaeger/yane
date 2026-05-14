from fastapi import APIRouter, HTTPException
from yane.api.deps import get_state
from yane.api.models import FitnessRequest, PopulationStatus, GenomeInfo

router = APIRouter(tags=["population"])


@router.post("/next", summary="Select the next genome for evaluation")
def next_genome() -> GenomeInfo:
    genome = get_state().next_genome()
    return _genome_info(genome)


@router.post("/fitness", summary="Submit fitness for the current genome")
def submit_fitness(req: FitnessRequest) -> dict:
    s = get_state()
    if s.current_genome is None:
        raise HTTPException(400, "No active genome. Call POST /population/next first.")
    s.submit_fitness(req.fitness)
    return {"ok": True}


@router.get("/status", summary="Population statistics", response_model=PopulationStatus)
def status() -> PopulationStatus:
    s = get_state()
    if not s.is_configured:
        raise HTTPException(400, "NeuroEvolution not configured yet. Call POST /configure.")
    pop = s._population
    try:
        best_fitness = pop.get_best().fitness
    except RuntimeError:
        best_fitness = None
    return PopulationStatus(
        size=pop.size,
        evaluated=pop.evaluated_count,
        unevaluated=pop.unevaluated_count,
        best_fitness=best_fitness,
    )


@router.get("/best", summary="Best genome found so far", response_model=GenomeInfo)
def best() -> GenomeInfo:
    s = get_state()
    if not s.is_configured:
        raise HTTPException(400, "NeuroEvolution not configured yet. Call POST /configure.")
    try:
        genome = s._population.get_best()
    except RuntimeError as e:
        raise HTTPException(404, str(e))
    return _genome_info(genome)


def _genome_info(genome) -> GenomeInfo:
    info = genome.memory_info()
    return GenomeInfo(
        fitness=genome.fitness,
        n_nodes=info["nodes"],
        n_connections=info["connections"],
        n_inputs=info["input_nodes"],
        n_outputs=info["output_nodes"],
    )
