"""Ablation benchmark for fixed, hand-adaptive, and evolved policies.

Usage:
    python -m yane.benchmarks.meta_policy_ablation --max-iter 1000 --seeds 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from yane import NeuroEvolution
from yane.core.genome import Genome


_XOR_INPUTS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
_XOR_TARGETS = [0.0, 1.0, 1.0, 0.0]
_XOR_SOLVE_THRESHOLD = -0.02


@dataclass
class MetaPolicyConfig:
    name: str
    mode: str


@dataclass
class MetaPolicyRun:
    config: str
    seed: int
    solved: bool
    iterations: int
    elapsed_s: float
    best_fitness: float
    diagnostics: dict


CONFIGS = [
    MetaPolicyConfig("fixed_policy", "fixed"),
    MetaPolicyConfig("handadaptive_policy", "handadaptive"),
    MetaPolicyConfig("evolved_policy", "evolved"),
]


def _xor_fitness(genome: Genome) -> float:
    total_err = 0.0
    for inputs, target in zip(_XOR_INPUTS, _XOR_TARGETS):
        genome.reset()
        out = genome.forward(list(inputs))
        pred = out[0] if out else 0.0
        total_err += (pred - target) ** 2
    return -total_err / len(_XOR_INPUTS)


def run_one(config: MetaPolicyConfig, seed: int, max_iter: int) -> MetaPolicyRun:
    yane = NeuroEvolution(seed=seed)
    yane.configure(2, 1, n_initial_hidden=2, max_nodes=20, max_connections=60)
    yane.set_population_size(80)
    yane.set_min_fitness(_XOR_SOLVE_THRESHOLD)
    yane.set_max_iterations(max_iter)
    from yane.benchmarks import wire_db, BENCHMARK_DB_PATH
    wire_db(yane, f"meta_policy/{config.mode}", BENCHMARK_DB_PATH)

    if config.mode == "fixed":
        yane.set_interspecies_crossover(0.02)
        yane.set_lamarck_budget(50)
        yane.set_operator_scheduler(False)
    elif config.mode == "handadaptive":
        yane.set_adaptive_interspecies_crossover(min_rate=0.0, max_rate=0.2)
        yane.set_lamarck_budget(50)
        yane.set_operator_scheduler(True)
        yane.set_adaptive_control(True)
    elif config.mode == "evolved":
        yane.set_meta_adaptive_policies(enabled=True, seed=seed)

    t0 = time.perf_counter()
    iterations = yane.train(_xor_fitness)
    elapsed_s = time.perf_counter() - t0
    best = yane.get_best()
    info = yane.population_memory_info()
    diagnostics = {
        "operator_scheduler": info.get("operator_scheduler"),
        "adaptive_controller": info.get("adaptive_controller"),
        "meta_adaptive_policies": info.get("meta_adaptive_policies"),
        "interspecies_crossover_current": info.get("interspecies_crossover_current"),
        "lamarck_budget_per_gen": info.get("lamarck_budget_per_gen"),
    }
    return MetaPolicyRun(
        config=config.name,
        seed=seed,
        solved=best.fitness >= _XOR_SOLVE_THRESHOLD,
        iterations=iterations,
        elapsed_s=elapsed_s,
        best_fitness=float(best.fitness),
        diagnostics=diagnostics,
    )


def summarize(rows: list[MetaPolicyRun]) -> dict:
    by_config: dict[str, list[MetaPolicyRun]] = {}
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
