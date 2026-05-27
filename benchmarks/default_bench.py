"""Benchmark GUI defaults across built-in examples.

The suite is intentionally short enough for tuning passes while still using the
same ExampleConfig values the GUI applies on example selection.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Case:
    example: str
    seeds: list[int]
    max_iterations: int
    population: int | None = None
    target_species: int | None = None
    max_nodes: int | None = None
    max_connections: int | None = None
    n_initial_hidden: int | None = None
    curriculum: bool | None = None
    complexity_nodes: float = 0.0
    complexity_connections: float = 0.0
    timeout_s: float = 15.0
    lamarck_steps: int | None = None
    fitness_shaping: bool | None = None


def _dataset_accuracy(example, genome) -> tuple[int | None, int | None]:
    samples = example.sequence_samples or example.test_cases
    if not samples:
        return None, None
    correct = 0
    total = 0
    for inputs, expected in samples:
        genome.reset()
        outputs = genome.forward(inputs)
        if example.output_scale:
            ok = all(
                round(o * s) == round(e * s)
                for o, e, s in zip(outputs, expected, example.output_scale)
            )
        else:
            ok = sum(abs(o - e) for o, e in zip(outputs, expected)) <= 0.05 * len(expected)
        correct += int(ok)
        total += 1
    return correct, total


def _run_case(case: Case) -> dict[str, Any]:
    from yane import NeuroEvolution
    from yane.gui.examples import load_examples

    ex = next(e for e in load_examples() if e.name == case.example)
    rows = []
    for seed in case.seeds:
        yane = NeuroEvolution(seed=seed)
        yane.configure(
            ex.n_inputs,
            ex.n_outputs,
            max_nodes=case.max_nodes if case.max_nodes is not None else ex.max_nodes,
            max_connections=(
                case.max_connections if case.max_connections is not None else ex.max_connections
            ),
            n_initial_hidden=(
                case.n_initial_hidden
                if case.n_initial_hidden is not None
                else ex.n_initial_hidden
            ),
            stateful=ex.stateful,
        )
        yane.set_population_size(case.population or ex.default_population)
        yane.set_target_species(case.target_species or ex.default_target_species)
        yane.set_max_iterations(case.max_iterations)
        yane.set_min_fitness(ex.target_fitness)
        if case.complexity_nodes or case.complexity_connections:
            yane.set_complexity_penalty(case.complexity_nodes, case.complexity_connections)
        lamarck_steps = (
            case.lamarck_steps
            if case.lamarck_steps is not None
            else ex.default_lamarck_steps
        )
        if lamarck_steps > 0:
            yane.set_lamarck(n_steps=lamarck_steps)
        fitness_shaping = (
            case.fitness_shaping
            if case.fitness_shaping is not None
            else ex.default_fitness_shaping
        )
        if fitness_shaping:
            yane.set_fitness_shaping(True)

        use_curriculum = (
            case.curriculum
            if case.curriculum is not None
            else ex.default_curriculum
        )
        if ex.make_curriculum is not None and use_curriculum:
            yane.set_curriculum(ex.make_curriculum(normalize=True, target_fitness=ex.target_fitness))
            eval_factory = ex.make_eval
        else:
            eval_factory = ex.make_eval

        from yane.benchmarks import wire_db, BENCHMARK_DB_PATH
        wire_db(yane, f"default/{case.example}", BENCHMARK_DB_PATH)

        stop = []
        t0 = time.perf_counter()
        def _on_iteration(_iteration: int, _fitness: float, _elapsed_ms: float) -> bool:
            return (time.perf_counter() - t0) < case.timeout_s

        iters = yane.train(
            eval_factory(),
            on_iteration=_on_iteration,
            on_stop=lambda reason: stop.append(reason),
        )
        elapsed = time.perf_counter() - t0
        best = yane.get_best()
        acc, acc_total = _dataset_accuracy(ex, best)
        info = yane.population_memory_info()
        rows.append({
            "seed": seed,
            "iterations": int(iters),
            "elapsed_s": float(elapsed),
            "stop": stop[-1] if stop else None,
            "best_fitness": float(best.fitness),
            "solved": bool(best.fitness >= ex.target_fitness),
            "accuracy": acc,
            "accuracy_total": acc_total,
            "nodes": int(len(best.nodes)),
            "connections": int(best.connection_count),
            "species": info.get("species_count"),
            "curriculum_stage": info.get("curriculum_stage_index"),
            "curriculum_advances": info.get("curriculum_n_advances"),
        })

    solved = sum(1 for r in rows if r["solved"])
    fitnesses = [r["best_fitness"] for r in rows]
    accuracies = [
        r["accuracy"] / r["accuracy_total"]
        for r in rows
        if r["accuracy"] is not None and r["accuracy_total"]
    ]
    return {
        "case": asdict(case),
        "target_fitness": ex.target_fitness,
        "runs": rows,
        "summary": {
            "solved": solved,
            "n": len(rows),
            "success_rate": solved / len(rows),
            "mean_fitness": float(statistics.mean(fitnesses)),
            "median_fitness": float(statistics.median(fitnesses)),
            "mean_accuracy": float(statistics.mean(accuracies)) if accuracies else None,
            "mean_elapsed_s": float(statistics.mean(r["elapsed_s"] for r in rows)),
        },
    }


def _quick_cases() -> list[Case]:
    seeds = [0, 1, 2]
    return [
        Case("XOR", seeds, 4000, timeout_s=10.0),
        Case("Regression 2→2", seeds, 4000, timeout_s=10.0),
        Case("Regression 3→3", seeds, 6000, timeout_s=12.0),
        Case("Multiplication", seeds, 6000, timeout_s=12.0),
        Case("Sequence: Pi-Ziffern", seeds, 6000, timeout_s=12.0),
        Case("CartPole", seeds, 3000, timeout_s=12.0),
        Case("MountainCar (Discrete)", seeds, 2500, timeout_s=12.0),
        Case("Frozen Lake", seeds, 3000, timeout_s=12.0),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Run the default quick tuning suite")
    ap.add_argument("--out", default=None, help="Output JSON path")
    args = ap.parse_args()

    cases = _quick_cases()
    results = []
    for case in cases:
        result = _run_case(case)
        results.append(result)
        s = result["summary"]
        acc = s["mean_accuracy"]
        acc_s = f" acc={acc:.2f}" if acc is not None else ""
        print(
            f"{case.example:<26} solved={s['solved']}/{s['n']} "
            f"fit={s['mean_fitness']:.3f}{acc_s} time={s['mean_elapsed_s']:.2f}s",
            flush=True,
        )
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }
    out = Path(args.out) if args.out else Path("benchmarks/results") / (
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_default_bench.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(out)


if __name__ == "__main__":
    main()
