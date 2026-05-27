from fastapi import APIRouter, HTTPException
from yane.api.deps import get_state
from yane.api.models import InputRequest, OutputResponse

router = APIRouter(tags=["network"])


def _active_genome():
    genome = get_state().current_genome
    if genome is None:
        raise HTTPException(400, "No active genome. Call POST /population/next first.")
    return genome


@router.post("/inputs", summary="Set input values for the current genome")
def set_inputs(req: InputRequest) -> dict:
    _active_genome().set_inputs(req.data)
    return {"ok": True}


@router.post("/tick", summary="Execute one tick on the current genome")
def tick() -> dict:
    _active_genome().tick()
    return {"ok": True}


@router.get("/outputs", summary="Get current output values", response_model=OutputResponse)
def get_outputs() -> OutputResponse:
    return OutputResponse(outputs=_active_genome().get_outputs())


@router.post("/forward", summary="Full forward pass: set inputs, propagate, return outputs",
             response_model=OutputResponse)
def forward(req: InputRequest) -> OutputResponse:
    return OutputResponse(outputs=_active_genome().forward(req.data))


@router.post("/reset", summary="Reset triggered nodes and non-persistent values")
def reset() -> dict:
    _active_genome().reset()
    return {"ok": True}
