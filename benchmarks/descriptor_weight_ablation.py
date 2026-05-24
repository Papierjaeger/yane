"""Ablation benchmark for fixed vs adaptive descriptor/fitness weights.

Usage:
    python -m yane.benchmarks.descriptor_weight_ablation --max-iter 1000 --seeds 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from yane import FitnessComponent, NeuroEvolution
from yane.core.genome import Genome


_XOR_INPUTS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
_XOR_TARGETS = [0.0, 1.0, 1.0, 0.0]
_XOR_SOLVE_THRESHOLD = -0.02


@dataclass
class AblationConfig:
    name: str
    components: str
    mode: str = "fixed"


@dataclass
class AblationRun:
    config: str
    seed: int
    solved: bool
    iterations: int
    elapsed_s: float
    best_fitness: float
    final_weights: dict[str, float]
    weight_history_len: int


CONFIGS = [
    AblationConfig("task_only", "none"),
    AblationConfig("topology_fixed", "topology", "fixed"),
    AblationConfig("behavior_fixed", "behavior", "fixed"),
    AblationConfig("topology_behavior_fixed", "both", "fixed"),
    AblationConfig("topology_behavior_adaptive", "both", "adaptive"),
]


def _xor_fitness(genome: Genome) -> float:
    total_err = 0.0
    for inputs, target in zip(_XOR_INPUTS, _XOR_TARGETS):
        genome.reset()
        outputs = genome.forward(list(inputs))
        pred = outputs[0] if outputs else 0.0
        total_err += (pred - target) ** 2
    return -total_err / len(_XOR_INPUTS)


def _component_set(kind: str) -> list[FitnessComponent]:
    topology = [
        FitnessComponent(
            "fewer_hidden",
            lambda g: max(0, len(g.nodes) - len(g.input_nodes) - len(g.output_nodes)),
            weight=0.002,
            maximize=False,
        ),
        FitnessComponent("fewer_connections", lambda g: g.connection_count, weight=0.001, maximize=False),
    ]
    behavior = [
        FitnessComponent("output_span", _output_span, weight=0.02),
        FitnessComponent("output_centering", _output_centering, weight=0.01, maximize=False),
    ]
    if kind == "topology":
        return topology
    if kind == "behavior":
        return behavior
    if kind == "both":
        return topology + behavior
    return []


def _probe_outputs(genome: Genome) -> list[float]:
    values: list[float] = []
    for inputs in _XOR_INPUTS:
        genome.reset()
        out = genome.forward(list(inputs))
        values.append(float(out[0] if out else 0.0))
    return values


def _output_span(genome: Genome) -> float:
    values = _probe_outputs(genome)
    return max(values) - min(values)


def _output_centering(genome: Genome) -> float:
    values = _probe_outputs(genome)
    return abs((sum(values) / len(values)) - 0.5)


def run_one(config: AblationConfig, seed: int, max_iter: int) -> AblationRun:
    yane = NeuroEvolution(seed=seed)
    yane.configure(2, 1, n_initial_hidden=2, max_nodes=20, max_connections=60)
    yane.set_population_size(80)
    yane.set_min_fitness(_XOR_SOLVE_THRESHOLD)
    yane.set_max_iterations(max_iter)

    components = _component_set(config.components)
    if components:
        yane.set_fitness_components(
            components,
            mode=config.mode,
            min_weight=0.0,
            max_weight=1.0,
            adaptation_rate=0.35,
            collapse_floor=0.03,
        )

    t0 = time.perf_counter()
    iterations = yane.train(_xor_fitness)
    elapsed_s = time.perf_counter() - t0
    best = yane.get_best()
    scalarizer = yane.get_fitness_component_weights()
    weights = scalarizer.weights if scalarizer is not None else {}
    history_len = len(scalarizer.history) if scalarizer is not None else 0
    return AblationRun(
        config=config.name,
        seed=seed,
        solved=best.fitness >= _XOR_SOLVE_THRESHOLD,
        iterations=iterations,
        elapsed_s=elapsed_s,
        best_fitness=float(best.fitness),
        final_weights=weights,
        weight_history_len=history_len,
    )


def summarize(rows: list[AblationRun]) -> dict:
    by_config: dict[str, list[AblationRun]] = {}
    for row in rows:
        by_config.setdefault(row.config, []).append(row)
    return {
        name: {
            "n": len(items),
            "solved": sum(1 for item in items if item.solved),
            "mean_iterations": statistics.mean(item.iterations for item in items),
            "mean_best_fitness": statistics.mean(item.best_fitness for item in items),
            "mean_elapsed_s": statistics.mean(item.elapsed_s for item in items),
        }
        for name, items in by_config.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = [
        run_one(config, seed, args.max_iter)
        for config in CONFIGS
        for seed in range(args.seeds)
    ]
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "max_iter": args.max_iter,
        "seeds": args.seeds,
        "runs": [asdict(row) for row in rows],
        "summary": summarize(rows),
    }
    text = json.dumps(payload, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
