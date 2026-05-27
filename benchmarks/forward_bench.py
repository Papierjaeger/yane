"""Forward-pass microbenchmarks for YANE genomes.

Measures the time per ``Genome.forward()`` call across:
  - network sizes: n_nodes ∈ {10, 50, 200, 1000}
  - topologies: acyclic (no recurrent connections) vs cyclic (with recurrence)

Usage:
    python -m yane.benchmarks.forward_bench            # default sizes
    python -m yane.benchmarks.forward_bench --sizes 10 100 500

Output: table to stdout + optional JSON to benchmarks/results/.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import time
from pathlib import Path

from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType


# ---------------------------------------------------------------------------
# Genome builders
# ---------------------------------------------------------------------------

def _make_acyclic_genome(n_hidden: int, n_inputs: int = 4, n_outputs: int = 2) -> Genome:
    """Build a genome with a strict input→hidden→output feed-forward topology.

    No recurrent connections so the forward pass is O(1) per node.
    """
    g = Genome()
    inputs = [Node(NodeType.INPUT) for _ in range(n_inputs)]
    hiddens = [Node(NodeType.HIDDEN) for _ in range(n_hidden)]
    outputs = [Node(NodeType.OUTPUT) for _ in range(n_outputs)]
    g.nodes = inputs + hiddens + outputs
    g.input_nodes = inputs
    g.output_nodes = outputs

    rng = random.Random(42)
    # Connect every input to ~30% of hidden nodes.
    for inp in inputs:
        inp.activation = ActivationType.LINEAR
        inp.bias = 0.0
        for h in hiddens:
            if rng.random() < 0.3:
                c = Connection(h)
                c.weight = rng.gauss(0.0, 1.0)
                inp.connections.append(c)

    # Connect every hidden node to every output.
    for h in hiddens:
        h.activation = ActivationType.TANH
        h.bias = rng.gauss(0.0, 0.1)
        for out in outputs:
            c = Connection(out)
            c.weight = rng.gauss(0.0, 0.5)
            h.connections.append(c)

    for out in outputs:
        out.activation = ActivationType.TANH
        out.bias = 0.0

    g._invalidate_topology()
    return g


def _make_cyclic_genome(n_hidden: int, n_inputs: int = 4, n_outputs: int = 2) -> Genome:
    """Build a genome with recurrent connections between hidden nodes.

    About 10% of hidden→hidden pairs get a backward connection so the
    forward pass runs the full iterative solver.
    """
    g = _make_acyclic_genome(n_hidden, n_inputs, n_outputs)
    hiddens = [n for n in g.nodes if n.type == NodeType.HIDDEN]
    rng = random.Random(99)
    for h1 in hiddens:
        for h2 in hiddens:
            if h1 is not h2 and rng.random() < 0.1:
                # Back-edge: h2 → h1 (may form a cycle)
                c = Connection(h1)
                c.weight = rng.gauss(0.0, 0.3)
                h2.connections.append(c)
    g._invalidate_topology()
    return g


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _bench_one(genome: Genome, n_inputs: int, n_warmup: int = 200, n_timed: int = 2000) -> float:
    """Return median microseconds per forward() call."""
    inputs = [0.5] * n_inputs
    # Warmup to avoid JIT/cache cold-start effects.
    for _ in range(n_warmup):
        genome.reset()
        genome.forward(inputs)

    times: list[float] = []
    for _ in range(n_timed):
        genome.reset()
        t0 = time.perf_counter()
        genome.forward(inputs)
        times.append(time.perf_counter() - t0)

    times.sort()
    n = len(times)
    median_s = times[n // 2]
    return median_s * 1e6  # → microseconds


def run_benchmarks(sizes: list[int], n_inputs: int = 4, n_outputs: int = 2) -> list[dict]:
    results = []
    for n in sizes:
        for label, builder in (("acyclic", _make_acyclic_genome), ("cyclic", _make_cyclic_genome)):
            genome = builder(n, n_inputs, n_outputs)
            # Larger networks need fewer timed samples to stay fast.
            n_timed = max(200, 4000 // max(1, n // 50))
            n_warmup = max(50, n_timed // 10)
            us = _bench_one(genome, n_inputs, n_warmup=n_warmup, n_timed=n_timed)
            n_nodes = len(genome.nodes)
            n_conns = sum(len(nd.connections) for nd in genome.nodes)
            results.append({
                "topology":    label,
                "n_hidden":    n,
                "n_nodes":     n_nodes,
                "n_conns":     n_conns,
                "median_us":   round(us, 3),
            })
            print(f"  {label:8s}  hidden={n:5d}  nodes={n_nodes:6d}  conns={n_conns:7d}"
                  f"  {us:8.2f} µs / forward")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="YANE forward-pass microbenchmarks")
    parser.add_argument(
        "--sizes", nargs="+", type=int,
        default=[10, 50, 200, 1000],
        help="Number of hidden nodes to benchmark (default: 10 50 200 1000)",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Write results to benchmarks/results/forward_bench_<timestamp>.json",
    )
    args = parser.parse_args()

    print(f"YANE forward-pass microbenchmark  (sizes={args.sizes})")
    print("-" * 70)
    results = run_benchmarks(args.sizes)
    print("-" * 70)

    if args.save:
        out_dir = Path(__file__).parent / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"forward_bench_{ts}.json"
        with open(out_path, "w") as f:
            json.dump({"sizes": args.sizes, "results": results}, f, indent=2)
        print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
