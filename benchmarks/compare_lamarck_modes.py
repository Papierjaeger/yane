"""Compare Lamarckian weight optimizers on the same benchmark.

Examples:
    python -m yane.benchmarks.compare_lamarck_modes --env Acrobot-v1 --seeds 3
    python -m yane.benchmarks.compare_lamarck_modes --env LunarLander-v3 --modes hc nes sa cma_es
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ModeRun:
    mode: str
    seed: int
    solved: bool
    iterations: int
    elapsed_s: float
    best_fitness: float
    stop_reason: str


def _make_env_eval(env_id: str, seed: int):
    import gymnasium as gym
    from yane import NeuroEvolution

    targets = {
        "Acrobot-v1": -100.0,
        "LunarLander-v3": 200.0,
    }
    if env_id not in targets:
        raise ValueError(f"Unsupported env {env_id!r}; use one of {sorted(targets)}")

    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    n_inputs = len(obs)
    n_outputs = 1

    def evaluate(genome):
        total = 0.0
        obs, _ = env.reset(seed=seed)
        genome.reset()
        done = False
        while not done:
            outputs = genome.forward(list(obs))
            if env_id == "Acrobot-v1":
                action = int(round(outputs[0] * 2.0)) % 3
            else:
                action = int(round(outputs[0] * 3.0)) % 4
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        return total

    yane = NeuroEvolution(seed=seed)
    yane.configure(n_inputs=n_inputs, n_outputs=n_outputs, max_nodes=40, max_connections=120)
    yane.set_population_size(150)
    yane.set_min_fitness(targets[env_id])
    yane.set_max_iterations(20_000)
    return yane, evaluate, targets[env_id], env


def _configure_mode(yane, mode: str, steps: int) -> None:
    if mode == "none":
        yane.set_lamarck_adaptive(max_steps=0)
    elif mode == "hc":
        yane.set_lamarck_adaptive(max_steps=steps, mode="hill_climbing")
    elif mode == "nes":
        yane.set_lamarck_adaptive(max_steps=steps, mode="nes")
    elif mode == "sa":
        yane.set_lamarck_adaptive(max_steps=steps, mode="sa")
    elif mode == "cma_es":
        yane.set_lamarck_adaptive(max_steps=steps, mode="cma_es")
    else:
        raise ValueError(f"Unknown mode {mode!r}")


def run_one(env_id: str, mode: str, seed: int, steps: int) -> ModeRun:
    yane, evaluate, target, env = _make_env_eval(env_id, seed)
    _configure_mode(yane, mode, steps)
    stop_reason = ["manual"]

    def on_stop(reason: str) -> None:
        stop_reason[0] = reason

    start = time.perf_counter()
    try:
        iterations = yane.train(evaluate, run_name=f"{env_id}_{mode}", on_stop=on_stop)
        best = yane.get_best().fitness
    finally:
        env.close()
    elapsed = time.perf_counter() - start
    return ModeRun(
        mode=mode,
        seed=seed,
        solved=best >= target,
        iterations=iterations,
        elapsed_s=elapsed,
        best_fitness=best,
        stop_reason=stop_reason[0],
    )


def print_summary(runs: list[ModeRun]) -> None:
    print("\nmode       solved    median iter    mean best    mean time")
    print("----------------------------------------------------------")
    for mode in sorted({r.mode for r in runs}):
        rows = [r for r in runs if r.mode == mode]
        solved = sum(r.solved for r in rows)
        solved_rows = [r for r in rows if r.solved]
        med_iter = statistics.median(r.iterations for r in solved_rows) if solved_rows else None
        mean_best = statistics.mean(r.best_fitness for r in rows)
        mean_time = statistics.mean(r.elapsed_s for r in rows)
        med_s = f"{med_iter:.0f}" if med_iter is not None else "-"
        print(f"{mode:<10} {solved:>2}/{len(rows):<3} {med_s:>12} {mean_best:>12.3f} {mean_time:>10.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Lamarck optimizer modes")
    parser.add_argument("--env", default="Acrobot-v1", choices=("Acrobot-v1", "LunarLander-v3"))
    parser.add_argument("--modes", nargs="+", default=["hc", "nes"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    runs: list[ModeRun] = []
    for mode in args.modes:
        for seed in range(args.seeds):
            print(f"{args.env} mode={mode} seed={seed} ... ", end="", flush=True)
            run = run_one(args.env, mode, seed, args.steps)
            runs.append(run)
            mark = "ok" if run.solved else "miss"
            print(f"{mark} fitness={run.best_fitness:.3f} iter={run.iterations} {run.elapsed_s:.1f}s")

    print_summary(runs)
    if not args.no_save:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = args.out_dir / f"{ts}_lamarck_modes_{args.env}.json"
        path.write_text(json.dumps([asdict(r) for r in runs], indent=2), encoding="utf-8")
        print(f"Results saved to {path}")


if __name__ == "__main__":
    main()
