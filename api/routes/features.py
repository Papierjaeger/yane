"""Feature-configuration endpoints — all Layer-3 research features.

Every endpoint maps to one or more ``NeuroEvolution.set_*()`` calls and
returns the updated configuration dict so callers can verify what changed.

Endpoints:
  POST /features/attention
  POST /features/stdp
  POST /features/neuromodulation
  POST /features/ltc
  POST /features/input_grouping
  POST /features/output_grouping
  POST /features/conv_neat
  POST /features/hybrid
  POST /features/probabilistic
  POST /features/island_model
  POST /features/hardware
  POST /features/budget
  POST /features/augmentation
  POST /features/phylogeny
  POST /features/safety
  POST /features/curiosity
  POST /features/darts
  POST /features/shared_weights
  POST /features/novelty
  POST /features/anytime
  POST /features/continual_learning
  GET  /features/phylogeny/tree
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from yane.api.state import state

router = APIRouter(tags=["features"])

_NOT_CONFIGURED = "NeuroEvolution not configured yet."


def _cfg() -> dict:
    """Return a light summary of changed feature state."""
    return {"ok": True}


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class AttentionRequest(BaseModel):
    enabled: bool = True
    head_dim: int = Field(4, ge=1, le=64)
    num_heads: int = Field(2, ge=1, le=16)


@router.post("/attention", summary="Configure evolvable attention heads")
def set_attention(req: AttentionRequest) -> dict:
    if not state.is_configured:
        raise HTTPException(400, _NOT_CONFIGURED)
    state.set_attention(enabled=req.enabled, head_dim=req.head_dim, num_heads=req.num_heads)
    return {"ok": True, "enabled": req.enabled, "head_dim": req.head_dim, "num_heads": req.num_heads}


# ---------------------------------------------------------------------------
# STDP
# ---------------------------------------------------------------------------

class STDPRequest(BaseModel):
    enabled: bool = True
    weight_min: float = Field(-1.0, le=0.0)
    weight_max: float = Field(1.0, ge=0.0)
    hebb_sigma: float = Field(0.1, gt=0.0)


@router.post("/stdp", summary="Configure Spike-Timing-Dependent Plasticity")
def set_stdp(req: STDPRequest) -> dict:
    if not state.is_configured:
        raise HTTPException(400, _NOT_CONFIGURED)
    state.set_stdp(
        enabled=req.enabled,
        weight_min=req.weight_min,
        weight_max=req.weight_max,
        hebb_sigma=req.hebb_sigma,
    )
    return {"ok": True, "enabled": req.enabled}


# ---------------------------------------------------------------------------
# Neuromodulation
# ---------------------------------------------------------------------------

class NeuromodulationRequest(BaseModel):
    enabled: bool = True


@router.post("/neuromodulation", summary="Enable dynamic neuromodulation")
def set_neuromodulation(req: NeuromodulationRequest) -> dict:
    if not state.is_configured:
        raise HTTPException(400, _NOT_CONFIGURED)
    state.set_neuromodulation(enabled=req.enabled)
    return {"ok": True, "enabled": req.enabled}


# ---------------------------------------------------------------------------
# LTC (Liquid Time-Constant nodes)
# ---------------------------------------------------------------------------

class LTCRequest(BaseModel):
    enabled: bool = True


@router.post("/ltc", summary="Enable Liquid Time-Constant ODE neurons")
def set_ltc(req: LTCRequest) -> dict:
    if not state.is_configured:
        raise HTTPException(400, _NOT_CONFIGURED)
    state.set_ltc(enabled=req.enabled)
    return {"ok": True, "enabled": req.enabled}


# ---------------------------------------------------------------------------
# Input grouping
# ---------------------------------------------------------------------------

class InputGroupingRequest(BaseModel):
    enabled: bool = True
    n_groups: int = Field(2, ge=1, le=64)
    n_raw: int | None = Field(None, ge=1)


@router.post("/input_grouping", summary="Enable evolvable input-aggregation groups")
def set_input_grouping(req: InputGroupingRequest) -> dict:
    if not state.is_configured:
        raise HTTPException(400, _NOT_CONFIGURED)
    state.set_input_grouping(
        enabled=req.enabled,
        n_groups=req.n_groups,
        n_raw=req.n_raw,
    )
    return {"ok": True, "enabled": req.enabled, "n_groups": req.n_groups}


# ---------------------------------------------------------------------------
# Output grouping
# ---------------------------------------------------------------------------

class OutputGroupingRequest(BaseModel):
    enabled: bool = True
    n_proto: int | None = Field(None, ge=1)
    n_outputs: int | None = Field(None, ge=1)


@router.post("/output_grouping", summary="Enable evolvable output-expansion groups")
def set_output_grouping(req: OutputGroupingRequest) -> dict:
    if not state.is_configured:
        raise HTTPException(400, _NOT_CONFIGURED)
    state.set_output_grouping(
        enabled=req.enabled,
        n_proto=req.n_proto,
        n_outputs=req.n_outputs,
    )
    return {"ok": True, "enabled": req.enabled}


# ---------------------------------------------------------------------------
# Convolutional NEAT
# ---------------------------------------------------------------------------

class ConvNEATRequest(BaseModel):
    enabled: bool = True
    n_image_channels: int = Field(1, ge=1, le=16)
    n_blocks: int = Field(2, ge=1, le=8)
    out_channels: int = Field(8, ge=1, le=128)
    kernel_size: int = Field(3, ge=1, le=7)


@router.post("/conv_neat", summary="Enable Convolutional NEAT image preprocessing")
def set_conv_neat(req: ConvNEATRequest) -> dict:
    if not state.is_configured:
        raise HTTPException(400, _NOT_CONFIGURED)
    state.set_conv_neat(
        enabled=req.enabled,
        n_image_channels=req.n_image_channels,
        n_blocks=req.n_blocks,
        out_channels=req.out_channels,
        kernel_size=req.kernel_size,
    )
    return {"ok": True, "enabled": req.enabled}


# ---------------------------------------------------------------------------
# Gradient-NEAT Hybrid (backprop interleaved)
# ---------------------------------------------------------------------------

class HybridRequest(BaseModel):
    enabled: bool = True
    bp_interval: int = Field(10, ge=1, le=1000)
    bp_epochs: int = Field(5, ge=1, le=100)
    bp_lr: float = Field(1e-3, gt=0.0, le=1.0)
    bp_batch_size: int = Field(32, ge=1, le=512)
    top_k: int = Field(5, ge=1, le=100)
    replay_buffer_size: int = Field(500, ge=10, le=50000)


@router.post("/hybrid", summary="Enable Gradient-NEAT hybrid backprop mode")
def set_hybrid(req: HybridRequest) -> dict:
    if not state.is_configured:
        raise HTTPException(400, _NOT_CONFIGURED)
    state.set_hybrid_mode(
        enabled=req.enabled,
        bp_interval=req.bp_interval,
        bp_epochs=req.bp_epochs,
        bp_lr=req.bp_lr,
        bp_batch_size=req.bp_batch_size,
        top_k=req.top_k,
        replay_buffer_size=req.replay_buffer_size,
    )
    return {"ok": True, "enabled": req.enabled}


# ---------------------------------------------------------------------------
# Probabilistic / Bayesian NEAT
# ---------------------------------------------------------------------------

class ProbabilisticRequest(BaseModel):
    enabled: bool = True
    noise_std: float = Field(0.05, gt=0.0, le=5.0)
    inference_mode: bool = False


@router.post("/probabilistic", summary="Enable probabilistic Gaussian output noise")
def set_probabilistic(req: ProbabilisticRequest) -> dict:
    if not state.is_configured:
        raise HTTPException(400, _NOT_CONFIGURED)
    state.set_probabilistic(
        enabled=req.enabled,
        noise_std=req.noise_std,
        inference_mode=req.inference_mode,
    )
    return {"ok": True, "enabled": req.enabled, "noise_std": req.noise_std}


# ---------------------------------------------------------------------------
# Island model
# ---------------------------------------------------------------------------

class IslandModelRequest(BaseModel):
    enabled: bool = True
    n_islands: int = Field(4, ge=2, le=32)
    migrate_interval: int = Field(20, ge=1, le=1000)
    migrate_count: int = Field(2, ge=1, le=50)


@router.post("/island_model", summary="Enable multi-population island model with migration")
def set_island_model(req: IslandModelRequest) -> dict:
    if not state.is_configured:
        raise HTTPException(400, _NOT_CONFIGURED)
    if req.enabled:
        state.set_island_model(
            n_islands=req.n_islands,
            migration_interval=req.migrate_interval,
            migration_count=req.migrate_count,
        )
    else:
        state._island_model = None
    return {"ok": True, "enabled": req.enabled, "n_islands": req.n_islands}


# ---------------------------------------------------------------------------
# Hardware constraints
# ---------------------------------------------------------------------------

_PLATFORMS = ["cortex-m4", "cortex-m7", "esp32", "raspberry-pi-zero",
              "raspberry-pi-4", "desktop", "mobile-arm"]


class HardwareRequest(BaseModel):
    target_platform: str | None = Field(None, description=f"One of: {_PLATFORMS}")
    max_flops: int | None = Field(None, ge=1)
    max_memory_bytes: int | None = Field(None, ge=1)
    max_latency_us: float | None = Field(None, gt=0.0)
    penalty_scale: float = Field(1.0, gt=0.0)
    enabled: bool = True


@router.post("/hardware", summary="Configure hardware deployment constraints")
def set_hardware(req: HardwareRequest) -> dict:
    if not state.is_configured:
        raise HTTPException(400, _NOT_CONFIGURED)
    if not req.enabled:
        state._hw_constraints = None
        return {"ok": True, "enabled": False}
    if req.target_platform is not None and req.target_platform not in _PLATFORMS:
        raise HTTPException(422, f"Unknown platform. Choose from: {_PLATFORMS}")
    state.set_hardware_constraints(
        target_platform=req.target_platform,
        max_flops=req.max_flops,
        max_memory_bytes=req.max_memory_bytes,
        max_latency_us=req.max_latency_us,
        penalty_scale=req.penalty_scale,
    )
    return {"ok": True, "enabled": True, "target_platform": req.target_platform}


# ---------------------------------------------------------------------------
# Resource budget
# ---------------------------------------------------------------------------

class BudgetRequest(BaseModel):
    total_time: str | None = Field(None, description="E.g. '30min', '2h', '90s'")
    max_memory: str | None = Field(None, description="E.g. '4GB', '80%', 'auto'")
    max_cpu_pct: float | None = Field(None, ge=1.0, le=100.0)
    target_platform: str | None = None
    preset: str | None = Field(None, description="'auto' for auto-detection")


@router.post("/budget", summary="Configure resource budget (time, memory, CPU)")
def set_budget(req: BudgetRequest) -> dict:
    if not state.is_configured:
        raise HTTPException(400, _NOT_CONFIGURED)
    try:
        state.set_budget(
            preset=req.preset,
            total_time=req.total_time,
            max_memory=req.max_memory,
            max_cpu_pct=req.max_cpu_pct,
            target_platform=req.target_platform,
        )
    except Exception as e:
        raise HTTPException(400, str(e))
    status = state.budget_status()
    return {"ok": True, **status}


# ---------------------------------------------------------------------------
# Evolutionary data augmentation
# ---------------------------------------------------------------------------

class AugmentationRequest(BaseModel):
    enabled: bool = True
    population_augmentations: int = Field(8, ge=2, le=100)
    pipeline_length: int = Field(3, ge=1, le=20)
    evolution_interval: int = Field(20, ge=1, le=500)
    mutation_sigma: float = Field(0.1, gt=0.0, le=2.0)


@router.post("/augmentation", summary="Enable evolutionary input data augmentation")
def set_augmentation(req: AugmentationRequest) -> dict:
    if not state.is_configured:
        raise HTTPException(400, _NOT_CONFIGURED)
    state.set_evolutionary_augmentation(
        enabled=req.enabled,
        population_augmentations=req.population_augmentations,
        pipeline_length=req.pipeline_length,
        evolution_interval=req.evolution_interval,
        mutation_sigma=req.mutation_sigma,
    )
    return {"ok": True, "enabled": req.enabled}


# ---------------------------------------------------------------------------
# Phylogeny tracking
# ---------------------------------------------------------------------------

class PhylogenyRequest(BaseModel):
    enabled: bool = True
    max_size: int | None = Field(None, ge=10, le=100000)


@router.post("/phylogeny", summary="Enable genome phylogeny (Stammbaum) tracking")
def set_phylogeny(req: PhylogenyRequest) -> dict:
    if not state.is_configured:
        raise HTTPException(400, _NOT_CONFIGURED)
    if req.enabled:
        state.enable_phylogeny(max_size=req.max_size)
    else:
        state.disable_phylogeny()
    return {"ok": True, "enabled": req.enabled}


@router.get("/phylogeny/tree", summary="Export the phylogeny tree as JSON")
def get_phylogeny_tree() -> dict:
    if not state.is_configured:
        raise HTTPException(400, _NOT_CONFIGURED)
    tree = state.get_phylogeny()
    if tree is None or not tree.is_enabled:
        raise HTTPException(404, "Phylogeny tracking not enabled. POST /features/phylogeny first.")
    return tree.to_dict()


# ---------------------------------------------------------------------------
# Safety constraints (simplified — predefined threshold-based mode)
# ---------------------------------------------------------------------------

class SafetyRequest(BaseModel):
    enabled: bool = True
    min_safe_frac: float = Field(0.0, ge=0.0, le=1.0)
    output_lower_bound: float | None = Field(
        None, description="Hard constraint: all outputs must be >= this value."
    )
    output_upper_bound: float | None = Field(
        None, description="Hard constraint: all outputs must be <= this value."
    )


@router.post("/safety", summary="Configure safety constraints (output bounds)")
def set_safety(req: SafetyRequest) -> dict:
    if not state.is_configured:
        raise HTTPException(400, _NOT_CONFIGURED)
    if not req.enabled:
        state.set_safety_constraints(None)
        return {"ok": True, "enabled": False}

    from yane.evolution.safety import SafetyConstraint
    constraints = []
    if req.output_lower_bound is not None:
        lb = req.output_lower_bound
        constraints.append(SafetyConstraint(
            name="output_lower_bound",
            check=lambda g, _lb=lb: all(v >= _lb for v in (g.forward([0.0] * g._n_inputs)
                                         if hasattr(g, "_n_inputs") else [])),
            mode="hard",
            penalty=-1e6,
        ))
    if req.output_upper_bound is not None:
        ub = req.output_upper_bound
        constraints.append(SafetyConstraint(
            name="output_upper_bound",
            check=lambda g, _ub=ub: all(v <= _ub for v in (g.forward([0.0] * g._n_inputs)
                                         if hasattr(g, "_n_inputs") else [])),
            mode="hard",
            penalty=-1e6,
        ))
    state.set_safety_constraints(constraints, min_safe_frac=req.min_safe_frac)
    return {"ok": True, "enabled": True, "n_constraints": len(constraints)}


# ---------------------------------------------------------------------------
# Curiosity (also in registry, here for completeness with weight param)
# ---------------------------------------------------------------------------

class CuriosityRequest(BaseModel):
    enabled: bool = True
    weight: float = Field(0.3, ge=0.0, le=5.0)
    network_size: int = Field(8, ge=4, le=128)
    lr: float = Field(0.01, gt=0.0, le=1.0)


@router.post("/curiosity", summary="Configure intrinsic curiosity module")
def set_curiosity(req: CuriosityRequest) -> dict:
    if not state.is_configured:
        raise HTTPException(400, _NOT_CONFIGURED)
    state.set_curiosity(
        enabled=req.enabled,
        weight=req.weight,
        network_size=req.network_size,
        lr=req.lr,
    )
    return {"ok": True, "enabled": req.enabled}


# ---------------------------------------------------------------------------
# DARTS-Lite
# ---------------------------------------------------------------------------

class DARTSRequest(BaseModel):
    enabled: bool = True
    prune_threshold: float = Field(0.1, ge=0.0, le=1.0)


@router.post("/darts", summary="Configure DARTS-Lite differentiable architecture search")
def set_darts(req: DARTSRequest) -> dict:
    if not state.is_configured:
        raise HTTPException(400, _NOT_CONFIGURED)
    state.set_darts_mode(enabled=req.enabled, prune_threshold=req.prune_threshold)
    return {"ok": True, "enabled": req.enabled}


# ---------------------------------------------------------------------------
# Shared weights
# ---------------------------------------------------------------------------

class SharedWeightsRequest(BaseModel):
    enabled: bool = True


@router.post("/shared_weights", summary="Enable weight sharing between connection groups")
def set_shared_weights(req: SharedWeightsRequest) -> dict:
    if not state.is_configured:
        raise HTTPException(400, _NOT_CONFIGURED)
    state.set_shared_weights(enabled=req.enabled)
    return {"ok": True, "enabled": req.enabled}


# ---------------------------------------------------------------------------
# Novelty search
# ---------------------------------------------------------------------------

class NoveltyRequest(BaseModel):
    enabled: bool = True
    archive_size: int = Field(200, ge=10, le=10000)


@router.post("/novelty", summary="Enable novelty-based selection")
def set_novelty(req: NoveltyRequest) -> dict:
    if not state.is_configured:
        raise HTTPException(400, _NOT_CONFIGURED)
    state.set_novelty_search(enabled=req.enabled)
    return {"ok": True, "enabled": req.enabled}


# ---------------------------------------------------------------------------
# Anytime evaluation
# ---------------------------------------------------------------------------

class AnytimeRequest(BaseModel):
    enabled: bool = True
    min_evals: int = Field(1, ge=1, le=100)
    max_evals: int = Field(5, ge=1, le=1000)
    promotion_frac: float = Field(0.3, ge=0.01, le=1.0)
    aggregation: str = Field("mean", pattern=r"^(mean|median|min|max)$")


@router.post("/anytime", summary="Configure anytime (progressive) evaluation budget")
def set_anytime(req: AnytimeRequest) -> dict:
    if not state.is_configured:
        raise HTTPException(400, _NOT_CONFIGURED)
    state.set_anytime_eval(
        enabled=req.enabled,
        min_evals=req.min_evals,
        max_evals=req.max_evals,
        promotion_frac=req.promotion_frac,
        aggregation=req.aggregation,
    )
    return {"ok": True, "enabled": req.enabled}


# ---------------------------------------------------------------------------
# Continual learning
# ---------------------------------------------------------------------------

class ContinualLearningRequest(BaseModel):
    mode: str = Field("ewc", pattern=r"^(ewc|progressive|replay|hybrid)$")
    lambda_ewc: float = Field(0.1, ge=0.0, le=100.0)
    replay_weight: float = Field(0.5, ge=0.0, le=10.0)
    replay_buffer_size: int = Field(500, ge=10, le=50000)
    n_progressive_nodes: int = Field(2, ge=0, le=50)


@router.post("/continual_learning", summary="Configure continual / lifelong learning mode")
def set_continual_learning(req: ContinualLearningRequest) -> dict:
    if not state.is_configured:
        raise HTTPException(400, _NOT_CONFIGURED)
    state.set_continual_learning(
        mode=req.mode,
        lambda_ewc=req.lambda_ewc,
        replay_weight=req.replay_weight,
        replay_buffer_size=req.replay_buffer_size,
        n_progressive_nodes=req.n_progressive_nodes,
    )
    return {"ok": True, "mode": req.mode}
