"""Adaptive Feature Benchmark Suite for YANE.

Compares fixed vs adaptive configurations across four feature axes:
  - interspecies_crossover: fixed-off / fixed-on / adaptive
  - lamarck_budget: unlimited / capped-per-gen
  - operator_scheduler: off / on
  - adaptive_controller: off / on

Usage (quick smoke test):
    python -m yane.benchmarks.adaptive_suite --max-iter 2000 --seeds 1 --no-save

Full comparison (takes a while):
    python -m yane.benchmarks.adaptive_suite --max-iter 10000 --seeds 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from yane import NeuroEvolution
from yane.core.genome import Genome


# ---------------------------------------------------------------------------
# Configurations under test
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """One combination of adaptive features to benchmark."""
    name: str
    interspecies: str = "off"       # "off" | "fixed" | "adaptive"
    lamarck_budget: int | None = None  # None = unlimited; int = max evals/gen
    operator_scheduler: bool = False
    adaptive_controller: bool = False
    lamarck_steps: int = 3


ALL_CONFIGS = [
    Config("baseline",
           interspecies="off", lamarck_budget=None,
           operator_scheduler=False, adaptive_controller=False),
    Config("interspecies_fixed",
           interspecies="fixed", lamarck_budget=None,
           operator_scheduler=False, adaptive_controller=False),
    Config("interspecies_adaptive",
           interspecies="adaptive", lamarck_budget=None,
           operator_scheduler=False, adaptive_controller=False),
    Config("budget_lamarck",
           interspecies="off", lamarck_budget=50,
           operator_scheduler=False, adaptive_controller=False),
    Config("operator_scheduler",
           interspecies="off", lamarck_budget=None,
           operator_scheduler=True, adaptive_controller=False),
    Config("adaptive_ctrl",
           interspecies="adaptive", lamarck_budget=50,
           operator_scheduler=True, adaptive_controller=True),
    Config("full_adaptive",
           interspecies="adaptive", lamarck_budget=100,
           operator_scheduler=True, adaptive_controller=True,
           lamarck_steps=5),
]


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

@dataclass
class BenchRun:
    config_name: str
    seed: int
    solved: bool
    iterations: int
    elapsed_s: float
    best_fitness: float
    stop_reason: str
    diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# XOR problem — lightweight, no gymnasium required
# ---------------------------------------------------------------------------

_XOR_INPUTS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
_XOR_TARGETS = [0.0, 1.0, 1.0, 0.0]
_XOR_SOLVE_THRESHOLD = -0.02


def _xor_fitness(genome: Genome) -> float:
    """MSE-based XOR fitness (negative, higher = better; 0 = perfect)."""
    genome.reset()
    total_err = 0.0
    for (a, b), target in zip(_XOR_INPUTS, _XOR_TARGETS):
        out = genome.forward([a, b])
        pred = out[0] if out else 0.0
        total_err += (pred - target) ** 2
    return -total_err / len(_XOR_INPUTS)


# ---------------------------------------------------------------------------
# Problem registry
# ---------------------------------------------------------------------------

def _make_xor(seed: int, max_iter: int, target: float) -> tuple[NeuroEvolution, Callable]:
    yane = NeuroEvolution(seed=seed)
    yane.configure(n_inputs=2, n_outputs=1, n_initial_hidden=2,
                   max_nodes=20, max_connections=60)
    yane.set_population_size(80)
    yane.set_min_fitness(target)
    yane.set_max_iterations(max_iter)
    return yane, _xor_fitness


def _make_pole(seed: int, max_iter: int, target: float) -> tuple[NeuroEvolution, Callable]:
    """CartPole-v1 (requires gymnasium)."""
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=seed)
    n_inputs = len(obs)

    def evaluate(genome: Genome) -> float:
        total = 0.0
        obs, _ = env.reset(seed=seed)
        genome.reset()
        done = False
        while not done:
            out = genome.forward(list(obs))
            action = 0 if (out[0] if out else 0.0) < 0 else 1
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        return total

    yane = NeuroEvolution(seed=seed)
    yane.configure(n_inputs=n_inputs, n_outputs=1, max_nodes=30, max_connections=90)
    yane.set_population_size(100)
    yane.set_min_fitness(target)
    yane.set_max_iterations(max_iter)
    return yane, evaluate


PROBLEMS = {
    "xor": {
        "factory": _make_xor,
        "target": _XOR_SOLVE_THRESHOLD,
        "has_gym": False,
    },
    "cartpole": {
        "factory": _make_pole,
        "target": 195.0,
        "has_gym": True,
    },
}


# ---------------------------------------------------------------------------
# Apply a Config to a NeuroEvolution instance
# ---------------------------------------------------------------------------

def _apply_config(yane: NeuroEvolution, cfg: Config) -> None:
    yane.set_lamarck_adaptive(max_steps=cfg.lamarck_steps)

    if cfg.interspecies == "fixed":
        yane.set_interspecies_crossover(rate=0.05)
    elif cfg.interspecies == "adaptive":
        yane.set_adaptive_interspecies_crossover(min_rate=0.01, max_rate=0.15)

    if cfg.lamarck_budget is not None:
        yane.set_lamarck_budget(budget_per_gen=cfg.lamarck_budget)

    if cfg.operator_scheduler:
        yane.set_operator_scheduler(enabled=True)

    if cfg.adaptive_controller:
        yane.set_adaptive_control(enabled=True)


# ---------------------------------------------------------------------------
# Run one configuration × one seed
# ---------------------------------------------------------------------------

def run_one(
    problem: str,
    cfg: Config,
    seed: int,
    max_iter: int,
) -> BenchRun:
    spec = PROBLEMS[problem]
    target = spec["target"]
    yane, evaluate = spec["factory"](seed, max_iter, target)
    _apply_config(yane, cfg)

    stop_reason = ["manual"]

    def on_stop(reason: str) -> None:
        stop_reason[0] = reason

    start = time.perf_counter()
    iterations = yane.train(evaluate, run_name=f"{problem}_{cfg.name}_s{seed}",
                             on_stop=on_stop)
    elapsed = time.perf_counter() - start

    best_genome = yane.get_best()
    best_fitness = best_genome.fitness if best_genome is not None else float("-inf")
    info = yane.population_memory_info()

    diag: dict = {
        "species_count": info.get("species_count"),
        "max_fitness": info.get("max_fitness"),
        "lamarck_n_applied": info.get("lamarck_n_applied"),
        "lamarck_budget_exhausted_count": info.get("lamarck_budget_exhausted_count"),
        "n_crossover": info.get("n_crossover"),
        "n_interspecies_crossover": info.get("n_interspecies_crossover"),
        "interspecies_success_rate": info.get("interspecies_success_rate"),
    }
    if "adaptive_controller" in info:
        ac = info["adaptive_controller"]
        diag["ctrl_tick"] = ac.get("controller_tick")
        diag["ctrl_plateau"] = ac.get("signals", {}).get("plateau_ratio")
    if "operator_scheduler" in info:
        os_ = info["operator_scheduler"]
        diag["sched_tick"] = os_.get("tick")
        diag["sched_qd_pressure"] = os_.get("qd_pressure")
        diag["sched_pruning_pressure"] = os_.get("pruning_pressure")

    return BenchRun(
        config_name=cfg.name,
        seed=seed,
        solved=best_fitness >= target,
        iterations=iterations,
        elapsed_s=elapsed,
        best_fitness=best_fitness,
        stop_reason=stop_reason[0],
        diagnostics=diag,
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(runs: list[BenchRun], configs: list[Config]) -> None:
    cfg_names = [c.name for c in configs]
    print()
    header = f"{'config':<22} {'solved':>8} {'med_iter':>10} {'mean_best':>12} {'mean_s':>8}"
    print(header)
    print("-" * len(header))
    for name in cfg_names:
        rows = [r for r in runs if r.config_name == name]
        if not rows:
            continue
        n = len(rows)
        solved = sum(r.solved for r in rows)
        solved_rows = [r for r in rows if r.solved]
        med_iter = statistics.median(r.iterations for r in solved_rows) if solved_rows else None
        mean_best = statistics.mean(r.best_fitness for r in rows)
        mean_time = statistics.mean(r.elapsed_s for r in rows)
        med_s = f"{med_iter:.0f}" if med_iter is not None else "-"
        print(f"{name:<22} {solved:>3}/{n:<4} {med_s:>10} {mean_best:>12.4f} {mean_time:>8.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive Feature Benchmark Suite")
    parser.add_argument(
        "--problem", default="xor",
        choices=list(PROBLEMS),
        help="Problem to benchmark (default: xor)",
    )
    parser.add_argument("--max-iter", type=int, default=5_000,
                        help="Max training iterations per run (default: 5000)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds per config (default: 3)")
    parser.add_argument(
        "--configs", nargs="+",
        default=None,
        choices=[c.name for c in ALL_CONFIGS] + ["all"],
        help="Which configs to run (default: all)",
    )
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).parent / "results")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving results to disk")
    args = parser.parse_args()

    if args.problem in PROBLEMS and PROBLEMS[args.problem]["has_gym"]:
        try:
            import gymnasium  # noqa: F401
        except ImportError:
            print(f"gymnasium not available — problem '{args.problem}' requires it. "
                  "Falling back to xor.")
            args.problem = "xor"

    # Filter configs
    if args.configs is None or "all" in (args.configs or []):
        selected_configs = ALL_CONFIGS
    else:
        name_set = set(args.configs)
        selected_configs = [c for c in ALL_CONFIGS if c.name in name_set]

    runs: list[BenchRun] = []
    total = len(selected_configs) * args.seeds
    done = 0
    for cfg in selected_configs:
        for seed in range(args.seeds):
            done += 1
            print(
                f"[{done}/{total}] problem={args.problem} config={cfg.name} seed={seed} ... ",
                end="",
                flush=True,
            )
            run = run_one(args.problem, cfg, seed, args.max_iter)
            runs.append(run)
            mark = "solved" if run.solved else "miss"
            print(
                f"{mark} fitness={run.best_fitness:.4f} "
                f"iter={run.iterations} {run.elapsed_s:.1f}s"
            )

    print_summary(runs, selected_configs)

    if not args.no_save:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = args.out_dir / f"{ts}_adaptive_suite_{args.problem}.json"
        payload = {
            "problem": args.problem,
            "max_iter": args.max_iter,
            "seeds": args.seeds,
            "runs": [asdict(r) for r in runs],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
