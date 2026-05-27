"""Compare direct NEAT encoding with CPPN-generated substrate starts.

Usage:
    python -m yane.benchmarks.cppn_indirect_ablation --max-iter 1000 --seeds 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from yane import CPPNGenome, NeuroEvolution, generate_genome_from_cppn, hyperneat_substrate
from yane.core.genome import Genome


_XOR_INPUTS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
_XOR_TARGETS = [0.0, 1.0, 1.0, 0.0]
_XOR_SOLVE_THRESHOLD = -0.02


@dataclass
class CppnRun:
    config: str
    seed: int
    solved: bool
    iterations: int
    elapsed_s: float
    best_fitness: float
    initial_connections: int


def _xor_fitness(genome: Genome) -> float:
    total_err = 0.0
    for inputs, target in zip(_XOR_INPUTS, _XOR_TARGETS):
        genome.reset()
        out = genome.forward(list(inputs))
        pred = out[0] if out else 0.0
        total_err += (pred - target) ** 2
    return -total_err / len(_XOR_INPUTS)


def _make_direct(seed: int, max_iter: int) -> NeuroEvolution:
    yane = NeuroEvolution(seed=seed)
    yane.configure(2, 1, n_initial_hidden=2, max_nodes=20, max_connections=60)
    yane.set_population_size(80)
    yane.set_min_fitness(_XOR_SOLVE_THRESHOLD)
    yane.set_max_iterations(max_iter)
    return yane


def _make_cppn(seed: int, max_iter: int) -> NeuroEvolution:
    yane = NeuroEvolution(seed=seed)
    yane._apply_seed()
    substrate = hyperneat_substrate(2, 1, hidden_layers=(2,))
    initial = generate_genome_from_cppn(CPPNGenome(), substrate, threshold=0.0, tracker=yane._tracker)
    initial.max_nodes = 20
    initial.max_connections = 60
    from yane.evolution.population import Population

    yane._population_size = 80
    yane._population = Population(max_size=80, initial_genome=initial, tracker=yane._tracker)
    yane._n_inputs = 2
    yane._n_outputs = 1
    yane._max_nodes = 20
    yane._max_connections = 60
    yane._n_initial_hidden = 2
    yane.set_min_fitness(_XOR_SOLVE_THRESHOLD)
    yane.set_max_iterations(max_iter)
    return yane


def run_one(config: str, seed: int, max_iter: int) -> CppnRun:
    yane = _make_direct(seed, max_iter) if config == "direct" else _make_cppn(seed, max_iter)
    from yane.benchmarks import wire_db, BENCHMARK_DB_PATH
    wire_db(yane, f"cppn/{config}", BENCHMARK_DB_PATH)
    initial_connections = (
        yane.population._unevaluated[0].connection_count
        if yane.population and yane.population._unevaluated else 0
    )
    t0 = time.perf_counter()
    iterations = yane.train(_xor_fitness)
    elapsed_s = time.perf_counter() - t0
    best = yane.get_best()
    return CppnRun(
        config=config,
        seed=seed,
        solved=best.fitness >= _XOR_SOLVE_THRESHOLD,
        iterations=iterations,
        elapsed_s=elapsed_s,
        best_fitness=float(best.fitness),
        initial_connections=initial_connections,
    )


def summarize(rows: list[CppnRun]) -> dict:
    by_config: dict[str, list[CppnRun]] = {}
    for row in rows:
        by_config.setdefault(row.config, []).append(row)
    return {
        name: {
            "n": len(items),
            "solved": sum(1 for item in items if item.solved),
            "mean_iterations": statistics.mean(item.iterations for item in items),
            "mean_best_fitness": statistics.mean(item.best_fitness for item in items),
            "mean_elapsed_s": statistics.mean(item.elapsed_s for item in items),
            "mean_initial_connections": statistics.mean(item.initial_connections for item in items),
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
        for config in ("direct", "cppn_substrate")
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
