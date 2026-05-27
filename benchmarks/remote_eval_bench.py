"""Benchmark: remote HTTP evaluation vs local evaluation strategies.

Compares four approaches on a batch of N genomes with a simulated evaluation
that sleeps for a fixed duration (models a slow environment like a physics sim):

  1. Sequential (baseline)       – single-threaded loop
  2. Local thread pool           – AsyncEvaluationQueue with T threads
  3. Local process pool          – multiprocessing.Pool
  4. Remote HTTP (localhost)     – RemoteEvaluationClient → RemoteWorkerServer
                                   in a background thread (same machine)

Results are printed as a table.  Remote eval on the same machine naturally has
higher overhead than a real multi-machine setup; the benchmark is designed to
show the per-request overhead (serialisation + HTTP round-trip).

Usage:
    python -m yane.benchmarks.remote_eval_bench              # defaults
    python -m yane.benchmarks.remote_eval_bench --n 20 --sleep 0.05 --workers 4
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys
import threading
import time
import urllib.error
from pathlib import Path
from typing import Callable

from yane import NeuroEvolution
from yane.core.genome import Genome
from yane.evolution.async_evaluation import AsyncEvaluationQueue
from yane.evolution.remote_evaluation import RemoteEvaluationClient, RemoteWorkerServer

RESULTS_DIR = Path(__file__).parent / "results"


# ── Fitness function (sleep simulates a slow environment) ─────────────────────

def _make_fitness(sleep_s: float) -> Callable[[Genome], float]:
    def _fitness(genome: Genome) -> float:
        time.sleep(sleep_s)
        inputs = [1.0] * len(genome.input_nodes)
        out = genome.forward(inputs)
        return float(sum(out))
    return _fitness


def _pool_fitness(args):
    """Top-level function required for multiprocessing.Pool (can't pickle lambdas)."""
    genome, sleep_s = args
    time.sleep(sleep_s)
    inputs = [1.0] * len(genome.input_nodes)
    out = genome.forward(inputs)
    return float(sum(out))


# ── Strategies ────────────────────────────────────────────────────────────────

def run_sequential(genomes: list[Genome], fitness_fn: Callable) -> float:
    t0 = time.perf_counter()
    for g in genomes:
        fitness_fn(g)
    return time.perf_counter() - t0


def run_thread_pool(genomes: list[Genome], fitness_fn: Callable, n_workers: int) -> float:
    t0 = time.perf_counter()
    q = AsyncEvaluationQueue(fitness_fn, max_workers=n_workers)
    for g in genomes:
        q.submit(g)
    q.drain()
    q.shutdown()
    return time.perf_counter() - t0


def run_process_pool(genomes: list[Genome], sleep_s: float, n_workers: int) -> float:
    t0 = time.perf_counter()
    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.map(_pool_fitness, [(g, sleep_s) for g in genomes])
    return time.perf_counter() - t0


def run_remote_http(
    genomes: list[Genome], fitness_fn: Callable, n_workers: int, port: int
) -> float:
    """Start a local worker server in a background thread, run batch, stop."""
    server = RemoteWorkerServer(fitness_fn=fitness_fn, token="bench", port=port)
    app = server._build_app()

    import uvicorn

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    srv = uvicorn.Server(config)

    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()

    # Wait until the server is ready
    deadline = time.time() + 10.0
    while not srv.started and time.time() < deadline:
        time.sleep(0.05)
    if not srv.started:
        raise RuntimeError("Worker server did not start in time")

    try:
        client = RemoteEvaluationClient(
            worker_urls=[f"http://127.0.0.1:{port}"],
            token="bench",
            timeout_s=60.0,
            max_retries=1,
        )
        t0 = time.perf_counter()
        client.evaluate_batch(genomes)
        elapsed = time.perf_counter() - t0
    finally:
        srv.should_exit = True
        thread.join(timeout=5.0)

    return elapsed


# ── Main ──────────────────────────────────────────────────────────────────────

def _make_genomes(n: int, n_inputs: int = 4, n_outputs: int = 2) -> list[Genome]:
    yane = NeuroEvolution()
    yane.set_population_size(n)
    yane.configure(n_inputs, n_outputs)
    genomes = []
    for _ in range(n):
        genomes.append(yane.next_genome())
        yane.submit_fitness(0.0)
    return genomes


def _row(label: str, elapsed: float, n: int) -> str:
    per_genome = elapsed / n * 1000
    return f"  {label:<30s}  {elapsed:6.3f}s  ({per_genome:6.1f} ms/genome)"


def main(n: int = 10, sleep_s: float = 0.05, n_workers: int = 4, port: int = 8711,
         skip_remote: bool = False) -> None:
    print(f"\nRemote Eval Benchmark  n={n}  sleep={sleep_s:.3f}s  workers={n_workers}")
    print("─" * 62)

    genomes = _make_genomes(n)
    fitness_fn = _make_fitness(sleep_s)

    seq = run_sequential(genomes, fitness_fn)
    print(_row("Sequential (baseline)", seq, n))

    tp = run_thread_pool(genomes, fitness_fn, n_workers)
    print(_row(f"Thread pool (×{n_workers})", tp, n))
    print(f"    speedup vs sequential: {seq / tp:.1f}×")

    pp = run_process_pool(genomes, sleep_s, n_workers)
    print(_row(f"Process pool (×{n_workers})", pp, n))
    print(f"    speedup vs sequential: {seq / pp:.1f}×")

    if not skip_remote:
        try:
            import uvicorn
        except ImportError:
            print("  Remote HTTP: SKIP (uvicorn not installed)")
            skip_remote = True

    if not skip_remote:
        try:
            rh = run_remote_http(genomes, fitness_fn, n_workers, port)
            print(_row(f"Remote HTTP local (×{n_workers})", rh, n))
            print(f"    speedup vs sequential: {seq / rh:.1f}×")
            overhead = (rh - tp) / n * 1000
            print(f"    HTTP overhead vs thread pool: +{overhead:.1f} ms/genome")
        except Exception as exc:
            print(f"  Remote HTTP: ERROR ({exc})")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10,
                        help="Number of genomes to evaluate (default 10)")
    parser.add_argument("--sleep", type=float, default=0.05,
                        help="Simulated evaluation time per genome in seconds (default 0.05)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default 4)")
    parser.add_argument("--port", type=int, default=8711,
                        help="Local port for the worker server (default 8711)")
    parser.add_argument("--skip-remote", action="store_true",
                        help="Skip the remote HTTP benchmark")
    args = parser.parse_args()
    main(n=args.n, sleep_s=args.sleep, n_workers=args.workers,
         port=args.port, skip_remote=args.skip_remote)
