"""Long-running benchmark harness for all GUI examples.

The runner mirrors the GUI defaults as closely as possible, executes examples
sequentially, and repeats successful cases to detect seed variance.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import random
import signal
import statistics
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from yane import NeuroEvolution
from yane.evolution.quality_diversity import descriptor_from_outputs
from yane.gui.examples import ExampleConfig, load_examples
from yane.gui.research_features import (
    ResearchFeatureConfig,
    apply_research_features,
    configure_cppn_substrate_population,
)

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is a project dependency, keep fallback cheap
    np = None

try:
    import psutil
except Exception:  # pragma: no cover - psutil is a project dependency
    psutil = None


@dataclass
class RunRow:
    example: str
    seed: int
    repeat_index: int
    solved: bool
    stop: str | None
    iterations: int
    elapsed_s: float
    best_fitness: float
    target_fitness: float
    nodes: int
    connections: int
    species: int | None
    stagnation: int | None
    fitness_iqr: float | None
    accuracy: int | None
    accuracy_total: int | None
    max_abs_error: float | None
    mean_abs_error: float | None
    target_output_verified: bool | None


def _safe_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _dataset_check(example: ExampleConfig, genome) -> dict[str, Any]:
    samples = example.sequence_samples or example.test_cases
    if not samples:
        return {
            "accuracy": None,
            "accuracy_total": None,
            "max_abs_error": None,
            "mean_abs_error": None,
            "target_output_verified": None,
            "mismatches": [],
        }

    correct = 0
    total = 0
    errors: list[float] = []
    mismatches: list[dict[str, Any]] = []

    for inputs, expected in samples:
        genome.reset()
        outputs = genome.forward(inputs)
        if example.output_scale:
            scaled_outputs = [
                float(o) * float(s) for o, s in zip(outputs, example.output_scale)
            ]
            scaled_expected = [
                float(e) * float(s) for e, s in zip(expected, example.output_scale)
            ]
            sample_errors = [
                abs(o - e) for o, e in zip(scaled_outputs, scaled_expected)
            ]
            ok = all(
                round(o) == round(e)
                for o, e in zip(scaled_outputs, scaled_expected)
            )
        else:
            sample_errors = [abs(float(o) - float(e)) for o, e in zip(outputs, expected)]
            ok = sum(sample_errors) <= 0.05 * len(expected)

        correct += int(ok)
        total += 1
        errors.extend(sample_errors)
        if not ok and len(mismatches) < 12:
            mismatches.append({
                "input": inputs,
                "expected": expected,
                "output": [float(o) for o in outputs],
                "error": float(max(sample_errors) if sample_errors else 0.0),
            })

    return {
        "accuracy": correct,
        "accuracy_total": total,
        "max_abs_error": float(max(errors)) if errors else 0.0,
        "mean_abs_error": float(statistics.mean(errors)) if errors else 0.0,
        "target_output_verified": correct == total,
        "mismatches": mismatches,
    }


def _make_research_cfg(example: ExampleConfig, config: dict[str, Any]) -> ResearchFeatureConfig:
    return ResearchFeatureConfig(
        n_inputs=example.n_inputs,
        n_outputs=example.n_outputs,
        max_nodes=example.max_nodes,
        max_connections=example.max_connections,
        population_size=example.default_population,
        target_species=example.default_target_species,
        allow_memory=example.stateful,
        output_sanitize=True,
        output_fallback=0.0,
        cppn_substrate=bool(config.get("cppn_substrate", False)),
        cppn_hidden=int(config.get("cppn_hidden", 2)),
        matrix_forward=bool(config.get("matrix_forward", False)),
        fitness_components=bool(config.get("fitness_components", False)),
        fitness_component_mode=str(config.get("fitness_component_mode", "Fix")),
        meta_adaptive=False,
        module_library=False,
        module_insert_rate=0.02,
    )


def _configure_quality_diversity(yane: NeuroEvolution, example: ExampleConfig, config: dict[str, Any]) -> None:
    if not config.get("quality_diversity", False):
        return
    if config.get("quality_diversity_descriptor") == "Behavior":
        rng = random.Random(42)
        probes = [
            [rng.uniform(-1.0, 1.0) for _ in range(example.n_inputs)]
            for _ in range(2)
        ]
        bins = tuple(8 for _ in range(max(1, example.n_outputs * len(probes))))
        ranges = tuple((-1.0, 1.0) for _ in bins)
        yane.set_quality_diversity(
            descriptor_from_outputs(probes),
            bins=bins,
            ranges=ranges,
            max_cells=500,
        )
        return

    yane.set_quality_diversity(
        descriptor_fn=lambda g: (
            float(max(0, len(g.nodes) - len(g.input_nodes) - len(g.output_nodes))),
            float(g.connection_count),
        ),
        bins=(12, 16),
        ranges=((0.0, float(max(1, example.max_nodes or 100))),
                (0.0, float(max(1, example.max_connections or 200)))),
        max_cells=500,
    )


def _configure_yane(example: ExampleConfig, seed: int) -> NeuroEvolution:
    config = dict(example.default_config)
    policies = dict(example.default_adaptive_policies)

    yane = NeuroEvolution(seed=seed)
    yane.set_output_sanitizing(True, fallback=0.0)
    yane.configure(
        example.n_inputs,
        example.n_outputs,
        max_nodes=example.max_nodes,
        max_connections=example.max_connections,
        n_initial_hidden=example.n_initial_hidden,
        stateful=example.stateful,
    )
    yane.set_population_size(example.default_population)
    yane.set_n_workers(1)
    yane.set_target_species(example.default_target_species)

    research_cfg = _make_research_cfg(example, config)
    if research_cfg.cppn_substrate:
        configure_cppn_substrate_population(yane, research_cfg)

    yane.set_fitness_shaping(bool(config.get("fitness_shaping", False)))
    yane.set_novelty_search(bool(config.get("novelty", True)))
    yane.set_speciation(bool(config.get("speciation", True)))
    yane.set_crossover(bool(config.get("crossover", True)))
    yane.set_diversity_injection(bool(config.get("diversity_injection", True)))

    if policies.get("interspecies_mode") == "Adaptiv":
        yane.set_adaptive_interspecies_crossover(
            min_rate=float(policies.get("interspecies_min_rate", 0.0)),
            max_rate=float(policies.get("interspecies_max_rate", 0.2)),
        )
    else:
        yane.set_interspecies_crossover(float(config.get("interspecies_rate", 0.0)))

    if config.get("early_stop_factor", 0.0) > 0.0:
        yane.set_early_stopping(float(config["early_stop_factor"]))
    if config.get("efficiency_max_ms", 0.0) > 0.0 and config.get("efficiency_penalty", 0.0) > 0.0:
        yane.set_efficiency_penalty(
            float(config["efficiency_max_ms"]),
            float(config["efficiency_penalty"]),
        )

    yane.set_elitism(int(config.get("elite_global", 1)), int(config.get("elite_species", 1)))

    optimizer_map = {
        "Hill-Climbing": "hill_climbing",
        "NES": "nes",
        "SA": "sa",
        "CMA-ES": "cma_es",
    }
    optimizer = optimizer_map.get(str(config.get("lamarck_optimizer", "Hill-Climbing")), "hill_climbing")
    schedule = str(config.get("lamarck_schedule", "Adaptiv"))
    if schedule == "Explizit":
        yane.set_lamarck(n_steps=int(config.get("lamarck_steps", 0)), mode=optimizer)
    elif schedule == "Aus":
        yane.set_lamarck_adaptive(max_steps=0)
    else:
        yane.set_lamarck_adaptive(mode=optimizer)
    budget = int(policies.get("lamarck_budget", 0) or 0)
    yane.set_lamarck_budget(budget if budget > 0 else None)

    yane.set_adaptive_control(bool(policies.get("adaptive_controller", False)))
    yane.set_operator_scheduler(bool(policies.get("operator_scheduler", False)))
    research_cfg = ResearchFeatureConfig(
        **{
            **asdict(research_cfg),
            "meta_adaptive": bool(policies.get("meta_adaptive", False)),
            "module_library": bool(policies.get("module_library", False)),
            "module_insert_rate": float(policies.get("module_insert_rate", 0.02)),
        }
    )
    apply_research_features(yane, research_cfg)
    _configure_quality_diversity(yane, example, config)

    if int(config.get("multi_eval", 1)) > 1:
        yane.set_multi_eval(
            n=int(config.get("multi_eval", 1)),
            aggregation=str(config.get("aggregation", "mean")),
            sigma_penalty=float(config.get("sigma_penalty", 0.0)),
        )
    yane.set_min_fitness(example.target_fitness)
    return yane


def _make_eval(example: ExampleConfig):
    if example.make_curriculum is not None and example.default_curriculum:
        return None
    if example.supports_normalization and not example.default_config.get("normalize", True):
        return example.make_eval(normalize=False)
    return example.make_eval()


def _run_once(
    example: ExampleConfig,
    seed: int,
    repeat_index: int,
    timeout_s: float,
    snapshot_interval_s: float,
) -> tuple[RunRow, dict[str, Any]]:
    yane = _configure_yane(example, seed)
    if example.make_curriculum is not None and example.default_curriculum:
        yane.set_curriculum(
            example.make_curriculum(normalize=True, target_fitness=example.target_fitness)
        )
        eval_fn = None
    else:
        eval_fn = _make_eval(example)

    snapshots: list[dict[str, Any]] = []
    stop: list[str] = []
    start = time.perf_counter()
    next_snapshot = snapshot_interval_s

    def on_iteration(iteration: int, fitness: float, elapsed_ms: float) -> bool:
        nonlocal next_snapshot
        elapsed = time.perf_counter() - start
        if elapsed >= next_snapshot:
            mem = yane.population_memory_info()
            snapshots.append({
                "elapsed_s": round(elapsed, 3),
                "iteration": iteration,
                "fitness": _safe_float(fitness),
                "best_fitness": _safe_float(mem.get("max_fitness")),
                "avg_fitness": _safe_float(mem.get("avg_fitness")),
                "species": mem.get("species_count"),
                "stagnation": mem.get("stagnation_count"),
                "fitness_iqr": _safe_float(mem.get("fitness_iqr")),
                "nodes": mem.get("largest_genome_nodes"),
                "connections": mem.get("largest_genome_connections"),
            })
            next_snapshot += snapshot_interval_s
        return elapsed < timeout_s

    iterations = yane.train(
        eval_fn,
        run_name=f"long/{example.name.replace(' ', '_')}/seed_{seed}_r{repeat_index}",
        on_iteration=on_iteration,
        on_stop=lambda reason: stop.append(reason),
    )
    elapsed = time.perf_counter() - start
    best = yane.get_best()
    mem = yane.population_memory_info()
    check = _dataset_check(example, best)
    solved = bool(float(best.raw_fitness) >= float(example.target_fitness))
    row = RunRow(
        example=example.name,
        seed=seed,
        repeat_index=repeat_index,
        solved=solved,
        stop=stop[-1] if stop else None,
        iterations=int(iterations),
        elapsed_s=float(elapsed),
        best_fitness=float(best.raw_fitness),
        target_fitness=float(example.target_fitness),
        nodes=int(len(best.nodes)),
        connections=int(best.connection_count),
        species=mem.get("species_count"),
        stagnation=mem.get("stagnation_count"),
        fitness_iqr=_safe_float(mem.get("fitness_iqr")),
        accuracy=check["accuracy"],
        accuracy_total=check["accuracy_total"],
        max_abs_error=check["max_abs_error"],
        mean_abs_error=check["mean_abs_error"],
        target_output_verified=check["target_output_verified"],
    )
    details = {
        "row": asdict(row),
        "diagnostics": mem,
        "snapshots": snapshots,
        "mismatches": check["mismatches"],
        "config": yane.get_config(),
    }
    return row, details


def _summarize(rows: list[RunRow]) -> dict[str, Any]:
    solved = [r for r in rows if r.solved]
    fitnesses = [r.best_fitness for r in rows]
    return {
        "runs": len(rows),
        "solved": len(solved),
        "success_rate": len(solved) / len(rows) if rows else 0.0,
        "best_fitness": max(fitnesses) if fitnesses else None,
        "mean_fitness": statistics.mean(fitnesses) if fitnesses else None,
        "median_fitness": statistics.median(fitnesses) if fitnesses else None,
        "mean_elapsed_s": statistics.mean(r.elapsed_s for r in rows) if rows else None,
        "mean_iterations": statistics.mean(r.iterations for r in rows) if rows else None,
        "all_outputs_verified": all(
            r.target_output_verified is not False for r in rows if r.solved
        ),
    }


def _format_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Langzeittest aller Beispiele",
        "",
        f"**Datum:** {payload['created_at']}",
        f"**Maximale Laufzeit pro Run:** {payload['max_minutes']} Min",
        f"**Wiederholungen bei geloestem Beispiel:** {payload['solved_repeats']}",
        "",
        "## Zusammenfassung",
        "",
        "| Beispiel | Runs | Geloest | Best | Mittel | Output-Check | Diagnose |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for result in payload["results"]:
        s = result["summary"]
        rows = [RunRow(**r["row"]) for r in result["runs"]]
        last = rows[-1] if rows else None
        output = "n/a"
        if last and last.target_output_verified is not None:
            output = "ok" if s["all_outputs_verified"] else "pruefen"
        diagnosis = result.get("diagnosis", "")
        lines.append(
            f"| {result['example']} | {s['runs']} | {s['solved']} | "
            f"{s['best_fitness']:.4f} | {s['mean_fitness']:.4f} | "
            f"{output} | {diagnosis} |"
        )

    lines += ["", "## Details", ""]
    for result in payload["results"]:
        lines += [f"### {result['example']}", ""]
        s = result["summary"]
        lines.append(
            f"- Erfolg: {s['solved']}/{s['runs']} Runs, beste Fitness {s['best_fitness']:.6g}, "
            f"mittlere Fitness {s['mean_fitness']:.6g}."
        )
        lines.append(f"- Diagnose: {result.get('diagnosis', 'keine auffaellige Diagnose')}.")
        for run in result["runs"]:
            r = RunRow(**run["row"])
            acc = ""
            if r.accuracy is not None:
                acc = f", Accuracy {r.accuracy}/{r.accuracy_total}, max err {r.max_abs_error:.4g}"
            lines.append(
                f"- Seed {r.seed} Run {r.repeat_index}: {'geloest' if r.solved else 'nicht geloest'}, "
                f"Fitness {r.best_fitness:.6g}, {r.iterations} Iter, {r.elapsed_s:.1f}s, "
                f"Stop {r.stop}{acc}."
            )
        if result.get("recommendation"):
            lines.append(f"- Empfehlung: {result['recommendation']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _json_default(value: Any) -> Any:
    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    if hasattr(value, "__fspath__"):
        return os.fspath(value)
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "__dict__"):
        return {
            str(k): v
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    return str(value)


def _write_outputs(payload: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )
    md_path.write_text(_format_md(payload), encoding="utf-8")


def _diagnose(example: ExampleConfig, rows: list[RunRow], runs: list[dict[str, Any]]) -> tuple[str, str]:
    best = max(rows, key=lambda r: r.best_fitness)
    if any(r.solved for r in rows):
        if any(r.target_output_verified is False for r in rows if r.solved):
            return (
                "Target erreicht, Outputs nicht voll verifiziert",
                "Target-Fitness verschaerfen oder Output-basierte Stop-Bedingung ergaenzen.",
            )
        if len(rows) > 1 and sum(r.solved for r in rows) < len(rows):
            return (
                "loest, aber seed-abhaengig",
                "Defaults robuster machen: mehr Population oder mehr strukturelle Diversitaet.",
            )
        return ("stabil geloest" if len(rows) > 1 else "geloest", "")

    gap = example.target_fitness - best.best_fitness
    stagnant = best.stagnation is not None and best.stagnation > max(500, best.iterations // 2)
    low_iqr = best.fitness_iqr is not None and abs(best.fitness_iqr) < 1e-6
    if stagnant and low_iqr:
        return (
            "starke Stagnation / Fitness-Kollaps",
            "Mehr explorative Struktur ist noetig: Inselmodell, Curiosity oder problem-spezifische State-/Reward-Zerlegung pruefen.",
        )
    if stagnant:
        return (
            f"stagnierend, Abstand zum Target {gap:.4g}",
            "Default-Population, QD-Druck oder Lamarck-Budget erhoehen; bei Sparse Reward Feature-Task einplanen.",
        )
    return (
        f"verbessert, aber Target noch {gap:.4g} entfernt",
        "Laenger laufen lassen oder Ziel/Reward-Shaping anhand stabiler Mehrfachlaeufe kalibrieren.",
    )


def _run_example(
    example: ExampleConfig,
    *,
    timeout_s: float,
    solved_repeats: int,
    snapshot_seconds: float,
) -> dict[str, Any]:
    print(f"\n== {example.name} ==", flush=True)
    rows: list[RunRow] = []
    runs: list[dict[str, Any]] = []
    row, details = _run_once(
        example,
        seed=0,
        repeat_index=1,
        timeout_s=timeout_s,
        snapshot_interval_s=snapshot_seconds,
    )
    rows.append(row)
    runs.append(details)
    print(
        f"seed=0 run=1 solved={row.solved} fitness={row.best_fitness:.6g} "
        f"time={row.elapsed_s:.1f}s",
        flush=True,
    )

    if row.solved:
        for repeat_index in range(2, solved_repeats + 1):
            seed = repeat_index - 1
            row, details = _run_once(
                example,
                seed=seed,
                repeat_index=repeat_index,
                timeout_s=timeout_s,
                snapshot_interval_s=snapshot_seconds,
            )
            rows.append(row)
            runs.append(details)
            print(
                f"seed={seed} run={repeat_index} solved={row.solved} "
                f"fitness={row.best_fitness:.6g} time={row.elapsed_s:.1f}s",
                flush=True,
            )
            if not row.solved:
                print(
                    "stopping repeats: a 30-minute repeat did not reach target",
                    flush=True,
                )
                break

    diagnosis, recommendation = _diagnose(example, rows, runs)
    return {
        "example": example.name,
        "summary": _summarize(rows),
        "diagnosis": diagnosis,
        "recommendation": recommendation,
        "runs": runs,
    }


def _worker_run_example(
    example_name: str,
    timeout_s: float,
    solved_repeats: int,
    snapshot_seconds: float,
    result_path: str,
    status_queue: "mp.Queue",
) -> None:
    try:
        example = next(e for e in load_examples() if e.name == example_name)
        status_queue.put({"event": "started", "example": example_name, "pid": os.getpid()})
        result = _run_example(
            example,
            timeout_s=timeout_s,
            solved_repeats=solved_repeats,
            snapshot_seconds=snapshot_seconds,
        )
        Path(result_path).write_text(
            json.dumps(result, indent=2, default=_json_default),
            encoding="utf-8",
        )
        status_queue.put({"event": "done", "example": example_name, "path": result_path})
    except BaseException as exc:
        status_queue.put({
            "event": "failed",
            "example": example_name,
            "error": repr(exc),
        })


def _parse_parallel(value: str) -> str | int:
    value = value.strip().lower()
    if value == "auto":
        return value
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--parallel must be an integer or 'auto'") from exc
    if n < 1:
        raise argparse.ArgumentTypeError("--parallel must be >= 1")
    return n


def _parallel_limit(parallel: str | int) -> int:
    if isinstance(parallel, int):
        return parallel
    cores = psutil.cpu_count(logical=False) if psutil is not None else None
    cores = cores or os.cpu_count() or 2
    return max(1, cores - 1)


def _system_has_headroom(max_cpu_percent: float, min_free_gb: float) -> tuple[bool, str]:
    if psutil is None:
        return True, "psutil unavailable"
    cpu = psutil.cpu_percent(interval=0.2)
    mem = psutil.virtual_memory()
    free_gb = mem.available / (1024 ** 3)
    ok = cpu <= max_cpu_percent and free_gb >= min_free_gb
    return ok, f"cpu={cpu:.1f}% free={free_gb:.1f}GB"


def _system_load(min_free_gb: float) -> tuple[float, float, str]:
    if psutil is None:
        return 0.0, float("inf"), "psutil unavailable"
    cpu = psutil.cpu_percent(interval=0.2)
    mem = psutil.virtual_memory()
    free_gb = mem.available / (1024 ** 3)
    return cpu, free_gb, f"cpu={cpu:.1f}% free={free_gb:.1f}GB"


def _pause_process(process: mp.Process) -> None:
    if process.pid is None:
        return
    if psutil is not None:
        psutil.Process(process.pid).suspend()
    else:
        os.kill(process.pid, signal.SIGSTOP)


def _resume_process(process: mp.Process) -> None:
    if process.pid is None:
        return
    if psutil is not None:
        psutil.Process(process.pid).resume()
    else:
        os.kill(process.pid, signal.SIGCONT)


def _run_parallel(
    examples: list[ExampleConfig],
    *,
    payload: dict[str, Any],
    json_path: Path,
    md_path: Path,
    timeout_s: float,
    solved_repeats: int,
    snapshot_seconds: float,
    parallel: str | int,
    max_cpu_percent: float,
    pause_cpu_percent: float,
    resume_cpu_percent: float,
    min_free_gb: float,
    poll_seconds: float,
) -> None:
    limit = _parallel_limit(parallel)
    run_dir = json_path.with_suffix("")
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
    status_queue: mp.Queue = ctx.Queue()
    pending = list(examples)
    active: dict[str, dict[str, Any]] = {}
    completed: dict[str, dict[str, Any]] = {}
    failed: dict[str, str] = {}

    print(
        f"parallel scheduler: limit={limit} max_cpu={max_cpu_percent}% "
        f"pause_cpu={pause_cpu_percent}% resume_cpu={resume_cpu_percent}% "
        f"min_free={min_free_gb}GB",
        flush=True,
    )

    def _start_next() -> bool:
        running_count = sum(1 for job in active.values() if not job["paused"])
        if not pending or len(active) >= limit:
            return False
        cpu, free_gb, reason = _system_load(min_free_gb)
        if (cpu > max_cpu_percent or free_gb < min_free_gb) and running_count > 0:
            print(f"waiting for headroom ({reason})", flush=True)
            return False
        example = pending.pop(0)
        safe_name = (
            example.name.replace("/", "_")
            .replace(" ", "_")
            .replace(":", "")
            .replace("→", "to")
        )
        result_path = run_dir / f"{safe_name}.json"
        process = ctx.Process(
            target=_worker_run_example,
            args=(
                example.name,
                timeout_s,
                solved_repeats,
                snapshot_seconds,
                str(result_path),
                status_queue,
            ),
            name=f"bench-{example.name}",
        )
        process.start()
        active[example.name] = {
            "process": process,
            "path": result_path,
            "paused": False,
            "started_at": time.monotonic(),
        }
        print(f"started {example.name} pid={process.pid} ({reason})", flush=True)
        return True

    def _throttle_active() -> None:
        if not active:
            return
        cpu, free_gb, reason = _system_load(min_free_gb)
        paused = [name for name, job in active.items() if job["paused"]]
        running = [name for name, job in active.items() if not job["paused"]]
        if (cpu > pause_cpu_percent or free_gb < min_free_gb) and len(running) > 1:
            name = max(running, key=lambda n: active[n]["started_at"])
            try:
                _pause_process(active[name]["process"])
                active[name]["paused"] = True
                print(f"paused {name} ({reason})", flush=True)
            except Exception as exc:
                print(f"pause failed {name}: {exc!r}", flush=True)
            return
        if paused and cpu < resume_cpu_percent and free_gb >= min_free_gb:
            name = min(paused, key=lambda n: active[n]["started_at"])
            try:
                _resume_process(active[name]["process"])
                active[name]["paused"] = False
                print(f"resumed {name} ({reason})", flush=True)
            except Exception as exc:
                print(f"resume failed {name}: {exc!r}", flush=True)

    while pending and len(active) < limit:
        if not _start_next():
            break

    while active or pending:
        _throttle_active()
        try:
            while True:
                msg = status_queue.get_nowait()
                event = msg.get("event")
                name = msg.get("example")
                if event == "done":
                    path = Path(str(msg["path"]))
                    completed[str(name)] = json.loads(path.read_text(encoding="utf-8"))
                    print(f"completed {name}", flush=True)
                elif event == "failed":
                    failed[str(name)] = str(msg.get("error"))
                    print(f"failed {name}: {failed[str(name)]}", flush=True)
                elif event == "started":
                    print(f"worker running {name} pid={msg.get('pid')}", flush=True)
        except queue.Empty:
            pass

        for name, job in list(active.items()):
            process = job["process"]
            result_path = job["path"]
            if process.is_alive():
                continue
            process.join()
            active.pop(name)
            if name not in completed and name not in failed:
                if result_path.exists():
                    completed[name] = json.loads(result_path.read_text(encoding="utf-8"))
                    print(f"completed {name}", flush=True)
                else:
                    failed[name] = f"exitcode={process.exitcode}"
                    print(f"failed {name}: {failed[name]}", flush=True)

        ordered_results = [
            completed[e.name]
            for e in examples
            if e.name in completed
        ]
        if failed:
            ordered_results += [
                {
                    "example": name,
                    "summary": {
                        "runs": 0,
                        "solved": 0,
                        "success_rate": 0.0,
                        "best_fitness": 0.0,
                        "mean_fitness": 0.0,
                        "median_fitness": 0.0,
                        "mean_elapsed_s": 0.0,
                        "mean_iterations": 0.0,
                        "all_outputs_verified": True,
                    },
                    "diagnosis": "worker failed",
                    "recommendation": failed[name],
                    "runs": [],
                }
                for name in failed
            ]
        payload["results"] = ordered_results
        _write_outputs(payload, json_path, md_path)

        while pending and len(active) < limit:
            if not _start_next():
                break

        if active or pending:
            time.sleep(max(1.0, poll_seconds))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-minutes", type=float, default=30.0)
    parser.add_argument("--solved-repeats", type=int, default=10)
    parser.add_argument("--snapshot-seconds", type=float, default=60.0)
    parser.add_argument("--examples", nargs="*", default=None)
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument(
        "--parallel",
        type=_parse_parallel,
        default=1,
        help="Number of examples to run at once, or 'auto' for physical cores minus one.",
    )
    parser.add_argument("--max-cpu-percent", type=float, default=75.0)
    parser.add_argument(
        "--pause-cpu-percent",
        type=float,
        default=92.0,
        help="Pause one running worker when total CPU rises above this value.",
    )
    parser.add_argument(
        "--resume-cpu-percent",
        type=float,
        default=70.0,
        help="Resume one paused worker when total CPU drops below this value.",
    )
    parser.add_argument("--min-free-gb", type=float, default=3.0)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()

    wanted = set(args.examples or [])
    examples = [e for e in load_examples() if not wanted or e.name in wanted]
    if wanted:
        missing = sorted(wanted - {e.name for e in examples})
        if missing:
            raise SystemExit(f"Unknown examples: {', '.join(missing)}")

    payload: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "max_minutes": args.max_minutes,
        "solved_repeats": args.solved_repeats,
        "results": [],
    }
    out_prefix = args.out_prefix or datetime.now().strftime("%Y-%m-%d_%H-%M-%S_long_examples")
    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{out_prefix}.json"
    md_path = out_dir / f"{out_prefix}.md"

    timeout_s = args.max_minutes * 60.0
    if args.parallel != 1:
        _run_parallel(
            examples,
            payload=payload,
            json_path=json_path,
            md_path=md_path,
            timeout_s=timeout_s,
            solved_repeats=args.solved_repeats,
            snapshot_seconds=args.snapshot_seconds,
            parallel=args.parallel,
            max_cpu_percent=args.max_cpu_percent,
            pause_cpu_percent=args.pause_cpu_percent,
            resume_cpu_percent=args.resume_cpu_percent,
            min_free_gb=args.min_free_gb,
            poll_seconds=args.poll_seconds,
        )
        print(f"\n{json_path}")
        print(md_path)
        return

    for example in examples:
        payload["results"].append(_run_example(
            example,
            timeout_s=timeout_s,
            solved_repeats=args.solved_repeats,
            snapshot_seconds=args.snapshot_seconds,
        ))
        _write_outputs(payload, json_path, md_path)

    _write_outputs(payload, json_path, md_path)
    print(f"\n{json_path}")
    print(md_path)


if __name__ == "__main__":
    main()
