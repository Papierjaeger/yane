"""Standard YANE benchmark suite.

Runs a fixed set of benchmarks across multiple seeds and reports:
  - success rate
  - mean / median iterations to solve (among successful runs)
  - wall-clock time per run

Usage (fast CI suite — XOR + basic_multiplication only, < 3 min):
    python -m yane.benchmarks.run_suite --fast

Usage (full suite including gym environments):
    python -m yane.benchmarks.run_suite

Results are written to benchmarks/results/<timestamp>.json alongside a
human-readable table printed to stdout.
"""
from __future__ import annotations
import argparse
import json
import statistics
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable


@dataclass
class BenchmarkSpec:
    name: str
    make_yane: Callable       # () → configured NeuroEvolution ready to call train()
    make_eval: Callable       # (seed) → fitness_fn  (may receive seed for gym envs)
    n_seeds: int = 5
    timeout_s: float = 120.0
    ci: bool = False          # included in --fast / CI suite


@dataclass
class RunResult:
    seed: int
    solved: bool
    iterations: int
    elapsed_s: float
    best_fitness: float
    stop_reason: str


@dataclass
class BenchmarkResult:
    name: str
    runs: list[RunResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return sum(r.solved for r in self.runs) / len(self.runs) if self.runs else 0.0

    @property
    def solved_runs(self) -> list[RunResult]:
        return [r for r in self.runs if r.solved]

    @property
    def mean_iterations(self) -> float | None:
        s = self.solved_runs
        return statistics.mean(r.iterations for r in s) if s else None

    @property
    def median_iterations(self) -> float | None:
        s = self.solved_runs
        return statistics.median(r.iterations for r in s) if s else None

    @property
    def mean_elapsed_s(self) -> float | None:
        s = self.solved_runs
        return statistics.mean(r.elapsed_s for r in s) if s else None


@dataclass
class GateCheck:
    name: str
    passed: bool
    metric: str
    actual: float | None
    expected: float
    comparator: str


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------

def _make_xor(seed: int):
    from yane import NeuroEvolution
    from yane.examples.XOR import make_eval as _make_eval, TARGET_FITNESS
    yane = NeuroEvolution(seed=seed)
    yane.configure(n_inputs=2, n_outputs=1, max_nodes=20, max_connections=50)
    yane.set_min_fitness(TARGET_FITNESS)
    yane.set_max_iterations(5000)
    return yane, _make_eval(), TARGET_FITNESS


def _make_multiplication(seed: int):
    from yane import NeuroEvolution
    from yane.examples.basic_multiplication import make_eval as _make_eval, TARGET_FITNESS
    yane = NeuroEvolution(seed=seed)
    yane.configure(n_inputs=2, n_outputs=1, max_nodes=30, max_connections=80)
    yane.set_min_fitness(TARGET_FITNESS)
    yane.set_max_iterations(10000)
    return yane, _make_eval(), TARGET_FITNESS


def _make_regression_2_2(seed: int):
    from yane import NeuroEvolution
    from yane.examples.simple_2_2_continuous import make_eval as _make_eval, TARGET_FITNESS
    yane = NeuroEvolution(seed=seed)
    yane.configure(n_inputs=2, n_outputs=2, max_nodes=30, max_connections=80, n_initial_hidden=4)
    yane.set_population_size(150)
    yane.set_target_species(10)
    yane.set_min_fitness(TARGET_FITNESS)
    yane.set_max_iterations(20_000)
    return yane, _make_eval(), TARGET_FITNESS


def _make_cartpole(seed: int):
    import gymnasium as gym
    from yane import NeuroEvolution
    TARGET = 475.0

    env = gym.make("CartPole-v1")

    def evaluate(genome):
        total = 0.0
        obs, _ = env.reset(seed=seed)
        genome.reset()
        done = False
        while not done:
            action = 1 if genome.forward(list(obs))[0] > 0.5 else 0
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        return total

    yane = NeuroEvolution(seed=seed)
    yane.configure(n_inputs=4, n_outputs=1, max_nodes=20, max_connections=50)
    yane.set_population_size(150)
    yane.set_min_fitness(TARGET)
    yane.set_max_iterations(10000)
    return yane, evaluate, TARGET


def _make_acrobot(seed: int):
    import gymnasium as gym
    from yane import NeuroEvolution
    TARGET = -100.0

    env = gym.make("Acrobot-v1")

    def evaluate(genome):
        total = 0.0
        obs, _ = env.reset(seed=seed)
        genome.reset()
        done = False
        while not done:
            outputs = genome.forward(list(obs))
            action = int(round(outputs[0] * 2)) % 3
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        return total

    yane = NeuroEvolution(seed=seed)
    yane.configure(n_inputs=6, n_outputs=1, max_nodes=30, max_connections=80)
    yane.set_population_size(150)
    yane.set_min_fitness(TARGET)
    yane.set_max_iterations(20000)
    return yane, evaluate, TARGET


_SUITE: list[BenchmarkSpec] = [
    BenchmarkSpec(
        name="XOR",
        make_yane=_make_xor,
        make_eval=lambda seed: None,  # make_yane handles eval creation
        n_seeds=5,
        timeout_s=30.0,
        ci=True,
    ),
    BenchmarkSpec(
        name="basic_multiplication",
        make_yane=_make_multiplication,
        make_eval=lambda seed: None,
        n_seeds=5,
        timeout_s=60.0,
        ci=True,
    ),
    BenchmarkSpec(
        name="Regression 2->2",
        make_yane=_make_regression_2_2,
        make_eval=lambda seed: None,
        n_seeds=3,
        timeout_s=120.0,
        ci=True,
    ),
    BenchmarkSpec(
        name="CartPole-v1",
        make_yane=_make_cartpole,
        make_eval=lambda seed: None,
        n_seeds=5,
        timeout_s=120.0,
        ci=False,
    ),
    BenchmarkSpec(
        name="Acrobot-v1",
        make_yane=_make_acrobot,
        make_eval=lambda seed: None,
        n_seeds=5,
        timeout_s=300.0,
        ci=False,
    ),
]


DEFAULT_GATES = {
    "XOR": {"min_success_rate": 0.2, "min_best_fitness": -0.2, "max_mean_elapsed_s": 30.0},
    "basic_multiplication": {"min_success_rate": 0.2, "min_best_fitness": -15.0, "max_mean_elapsed_s": 60.0},
    "Regression 2->2": {"min_success_rate": 0.0, "min_best_fitness": -2.0, "max_mean_elapsed_s": 120.0},
    "CartPole-v1": {"min_success_rate": 0.0, "min_best_fitness": 50.0, "max_mean_elapsed_s": 120.0},
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_one(spec: BenchmarkSpec, seed: int) -> RunResult:
    yane, eval_fn, target = spec.make_yane(seed)
    stop_reason: list[str] = ["manual"]

    def on_stop(reason: str) -> None:
        stop_reason[0] = reason

    t0 = time.perf_counter()
    try:
        n_iter = yane.train(eval_fn, on_stop=on_stop)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return RunResult(
            seed=seed,
            solved=False,
            iterations=0,
            elapsed_s=elapsed,
            best_fitness=float('-inf'),
            stop_reason=f"error: {exc}",
        )
    elapsed = time.perf_counter() - t0
    best = yane.get_best()
    solved = best.fitness >= target

    return RunResult(
        seed=seed,
        solved=solved,
        iterations=n_iter,
        elapsed_s=elapsed,
        best_fitness=best.fitness,
        stop_reason=stop_reason[0],
    )


def run_suite(fast: bool = False, verbose: bool = True) -> list[BenchmarkResult]:
    specs = [s for s in _SUITE if not fast or s.ci]
    results: list[BenchmarkResult] = []

    for spec in specs:
        br = BenchmarkResult(name=spec.name)
        if verbose:
            print(f"\n{'─' * 60}")
            print(f"  {spec.name}  ({spec.n_seeds} seeds, timeout {spec.timeout_s:.0f}s)")
            print(f"{'─' * 60}")
        for seed in range(spec.n_seeds):
            if verbose:
                print(f"  seed {seed} … ", end="", flush=True)
            run = _run_one(spec, seed)
            br.runs.append(run)
            if verbose:
                status = "✓" if run.solved else "✗"
                print(f"{status}  fitness={run.best_fitness:.4f}  "
                      f"iter={run.iterations}  {run.elapsed_s:.1f}s  [{run.stop_reason}]")
        results.append(br)

    return results


def print_summary(results: list[BenchmarkResult]) -> None:
    print(f"\n{'═' * 80}")
    print(f"  {'Benchmark':<28}  {'Solved':>6}  {'Mean iter':>10}  {'Med iter':>9}  {'Mean time':>9}")
    print(f"{'─' * 80}")
    for br in results:
        solved_str = f"{sum(r.solved for r in br.runs)}/{len(br.runs)}"
        mean_it = f"{br.mean_iterations:.0f}" if br.mean_iterations is not None else "—"
        med_it  = f"{br.median_iterations:.0f}" if br.median_iterations is not None else "—"
        mean_t  = f"{br.mean_elapsed_s:.1f}s" if br.mean_elapsed_s is not None else "—"
        print(f"  {br.name:<28}  {solved_str:>6}  {mean_it:>10}  {med_it:>9}  {mean_t:>9}")
    print(f"{'═' * 80}\n")


def save_results(results: list[BenchmarkResult], out_dir: Path | None = None) -> Path:
    if out_dir is None:
        here = Path(__file__).parent
        out_dir = here / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = out_dir / f"{ts}.json"
    payload = {
        "timestamp": ts,
        "results": [
            {
                "name": br.name,
                "success_rate": br.success_rate,
                "mean_iterations": br.mean_iterations,
                "median_iterations": br.median_iterations,
                "runs": [asdict(r) for r in br.runs],
            }
            for br in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_gates(path: Path | None) -> dict:
    if path is None:
        return DEFAULT_GATES
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("gates", data)


def evaluate_gates(results: list[BenchmarkResult], gates: dict) -> list[GateCheck]:
    checks: list[GateCheck] = []
    by_name = {br.name: br for br in results}
    for name, cfg in gates.items():
        br = by_name.get(name)
        if br is None:
            checks.append(GateCheck(name, False, "present", None, 1.0, "exists"))
            continue
        if "min_success_rate" in cfg:
            actual = br.success_rate
            expected = float(cfg["min_success_rate"])
            checks.append(GateCheck(name, actual >= expected, "success_rate", actual, expected, ">="))
        best_fitnesses = [r.best_fitness for r in br.runs]
        if "min_best_fitness" in cfg:
            actual = max(best_fitnesses) if best_fitnesses else None
            expected = float(cfg["min_best_fitness"])
            checks.append(GateCheck(
                name, actual is not None and actual >= expected,
                "best_fitness", actual, expected, ">=",
            ))
        if "max_mean_elapsed_s" in cfg:
            vals = [r.elapsed_s for r in br.runs]
            actual = statistics.mean(vals) if vals else None
            expected = float(cfg["max_mean_elapsed_s"])
            checks.append(GateCheck(
                name, actual is not None and actual <= expected,
                "mean_elapsed_s", actual, expected, "<=",
            ))
    return checks


def format_gate_report(checks: list[GateCheck]) -> str:
    lines = [
        "# Benchmark Gate Report",
        "",
        "| Benchmark | Metric | Actual | Gate | Result |",
        "|---|---:|---:|---:|---|",
    ]
    for c in checks:
        actual = "n/a" if c.actual is None else f"{c.actual:.4f}"
        result = "PASS" if c.passed else "FAIL"
        lines.append(
            f"| {c.name} | {c.metric} | {actual} | {c.comparator} {c.expected:.4f} | {result} |"
        )
    lines.append("")
    return "\n".join(lines)


def save_gate_report(checks: list[GateCheck], out_dir: Path | None = None) -> Path:
    if out_dir is None:
        out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = out_dir / f"{ts}_gate_report.md"
    path.write_text(format_gate_report(checks), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="YANE benchmark suite")
    parser.add_argument(
        "--fast", action="store_true",
        help="Run CI-fast suite only (XOR + basic_multiplication, < 3 min)"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip saving results to benchmarks/results/"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Directory for result JSON files (default: benchmarks/results/)"
    )
    parser.add_argument(
        "--gate", type=Path, default=None, nargs="?", const=Path(""),
        help="Evaluate benchmark gates. Omit value to use built-in gates, or pass a JSON gate file."
    )
    args = parser.parse_args()

    results = run_suite(fast=args.fast, verbose=True)
    print_summary(results)

    if not args.no_save:
        path = save_results(results, out_dir=args.out_dir)
        print(f"Results saved to {path}")

    if args.gate is not None:
        gate_path = None if str(args.gate) == "" else args.gate
        checks = evaluate_gates(results, load_gates(gate_path))
        report = format_gate_report(checks)
        print(report)
        if not args.no_save:
            path = save_gate_report(checks, out_dir=args.out_dir)
            print(f"Gate report saved to {path}")
        if not all(c.passed for c in checks):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
