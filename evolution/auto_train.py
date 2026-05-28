"""Phase 6: Zero-Config AutoTrainResult and report helpers for auto_train()."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AutoTrainResult:
    """Result returned by :meth:`NeuroEvolution.auto_train`.

    Attributes
    ----------
    best_genome:          Best genome found during training.
    final_fitness:        Its raw fitness.
    total_generations:    Number of completed generations.
    wall_time:            Total elapsed seconds including profiling.
    active_features:      Research features still active at end of training.
    final_params:         Snapshot of all tracked parameter values at end.
    problem_profile:      ProblemProfile produced by the profiler, or None.
    auto_config_report:   Human-readable string documenting every auto decision.
    kb_suggestions_used:  True if a Knowledge-Base suggestion was applied.
    n_kb_suggestions:     Number of KB suggestions returned (0 = cold start).
    """
    best_genome: Any
    final_fitness: float
    total_generations: int
    wall_time: float
    active_features: list = field(default_factory=list)
    final_params: dict = field(default_factory=dict)
    problem_profile: Any = None
    auto_config_report: str = ""
    kb_suggestions_used: bool = False
    n_kb_suggestions: int = 0


# ---------------------------------------------------------------------------
# Cold-start defaults
# ---------------------------------------------------------------------------

_COLD_START_DEFAULTS: dict[str, dict] = {
    "classification": {
        "lamarck.mode": "hill_climbing",
        "lamarck.n_steps": 3,
    },
    "regression": {
        "lamarck.mode": "hill_climbing",
        "lamarck.n_steps": 3,
    },
    "rl_discrete": {
        "lamarck.mode": "hill_climbing",
        "lamarck.n_steps": 5,
    },
    "rl_continuous": {
        "lamarck.mode": "cma_es",
        "lamarck.n_steps": 5,
    },
}


def apply_cold_start_defaults(ne: Any, profile: Any) -> dict:
    """Apply conservative defaults for a cold-start run; return applied params."""
    task_type = getattr(profile, "task_type", "classification")
    defaults = _COLD_START_DEFAULTS.get(task_type, _COLD_START_DEFAULTS["classification"])
    applied: dict = {}
    for name, value in defaults.items():
        try:
            ne.set_param(name, value)
            applied[name] = value
        except Exception:
            pass
    return applied


# ---------------------------------------------------------------------------
# Pop-size heuristic
# ---------------------------------------------------------------------------

def pick_pop_size(profile: Any) -> int:
    """Return a sensible population size based on the problem profile."""
    difficulty = getattr(profile, "estimated_difficulty", 0.5)
    n_in = getattr(profile, "n_inputs", 2)
    n_out = getattr(profile, "n_outputs", 1)
    base = 50 + int(difficulty * 100)         # 50 (easy) … 150 (hard)
    dim_factor = max(1, (n_in + n_out) // 4)  # scale with network size
    return min(200, max(20, base * dim_factor))


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(
    *,
    profile: Any,
    problem_name: str | None,
    kb_entries: int,
    kb_conf: float,
    applied_params: dict,
    cold_start: bool,
    meta_diag: dict,
    feat_diag: dict,
    active_features: list,
    pop_size: int,
    max_iters: int,
    wall_time: float,
    total_generations: int,
    final_fitness: float,
) -> str:
    lines: list[str] = ["─── Auto-Configuration Report ───"]

    # Problem section
    task = getattr(profile, "task_type", "unknown") if profile else "unknown"
    n_in = getattr(profile, "n_inputs", "?") if profile else "?"
    n_out = getattr(profile, "n_outputs", "?") if profile else "?"
    diff = getattr(profile, "estimated_difficulty", 0.0) if profile else 0.0
    noise = getattr(profile, "noise_level", 0.0) if profile else 0.0
    temporal = getattr(profile, "temporal_dependency", 0.0) if profile else 0.0
    name_part = f" — {problem_name}" if problem_name else ""
    lines.append(f"Problem{name_part}: {task} ({n_in}→{n_out})")
    lines.append(
        f"Difficulty: {diff:.2f} | Noise: {noise:.2f} | Temporal: {temporal:.2f}"
    )
    lines.append("")

    # Knowledge Base section
    if cold_start:
        lines.append(f"Knowledge Base: cold start ({kb_entries} entries)")
        lines.append("Parameters: conservative defaults applied")
    else:
        lines.append(
            f"Knowledge Base: {kb_entries} entries, "
            f"confidence {kb_conf:.2f} — suggestion applied"
        )
        if applied_params:
            param_str = ", ".join(f"{k}={v}" for k, v in list(applied_params.items())[:5])
            lines.append(f"Applied params: {param_str}")
    lines.append("")

    # MetaOptimizer section
    if meta_diag.get("enabled"):
        phase = meta_diag.get("phase", "?")
        n_ticks = meta_diag.get("n_ticks", 0)
        lines.append(f"MetaOptimizer: enabled — phase={phase}, ticks={n_ticks}")
    else:
        lines.append("MetaOptimizer: disabled")

    # Feature Gating section
    if feat_diag.get("enabled"):
        n_active = feat_diag.get("n_active", 0)
        if active_features:
            lines.append(
                f"Feature Gating: {n_active} active — {', '.join(active_features)}"
            )
        else:
            lines.append("Feature Gating: enabled — no features retained")
    else:
        lines.append("Feature Gating: disabled")
    lines.append("")

    # Training section
    lines.append("Training:")
    lines.append(f"  pop_size={pop_size}, max_iterations={max_iters}")
    lines.append("")

    # Summary
    mins, secs = divmod(wall_time, 60)
    time_str = f"{int(mins)}m{secs:.1f}s" if mins >= 1 else f"{wall_time:.2f}s"
    lines.append(
        f"Wall time: {time_str} | Generations: {total_generations} "
        f"| Best fitness: {final_fitness:.4g}"
    )

    return "\n".join(lines)
