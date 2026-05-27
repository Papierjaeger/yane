"""Profile genome serialization, matrix export, and multiprocessing IPC."""
from __future__ import annotations

import argparse
import json
import pickle
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class SerializationProfile:
    hidden_nodes: int
    connections: int
    pickle_bytes: int
    pickle_ms: float
    unpickle_ms: float
    matrix_export_ms: float | None
    matrix_bytes: int | None


def _make_genome(hidden_nodes: int, connections: int):
    from yane import NeuroEvolution
    from yane.evolution import smart_mutation

    yane = NeuroEvolution(seed=hidden_nodes + connections)
    yane.configure(4, 2, n_initial_hidden=hidden_nodes, max_nodes=max(10, hidden_nodes + 8),
                   max_connections=max(20, connections + 10))
    g = yane.next_genome()
    while g.connection_count < connections:
        if not smart_mutation.add_connection(g, yane._tracker):
            break
    return g


def profile_one(hidden_nodes: int, connections: int, repeats: int = 20) -> SerializationProfile:
    from yane.evolution.matrix_export import export_matrix_genome

    g = _make_genome(hidden_nodes, connections)
    pickle_times = []
    unpickle_times = []
    payload = b""
    for _ in range(repeats):
        start = time.perf_counter()
        payload = pickle.dumps(g, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_times.append((time.perf_counter() - start) * 1000.0)
        start = time.perf_counter()
        pickle.loads(payload)
        unpickle_times.append((time.perf_counter() - start) * 1000.0)

    matrix_ms = None
    matrix_bytes = None
    try:
        start = time.perf_counter()
        exported = export_matrix_genome(g)
        matrix_ms = (time.perf_counter() - start) * 1000.0
        matrix_bytes = exported.weights.nbytes + exported.bias.nbytes
    except ValueError:
        pass

    return SerializationProfile(
        hidden_nodes=hidden_nodes,
        connections=g.connection_count,
        pickle_bytes=len(payload),
        pickle_ms=statistics.median(pickle_times),
        unpickle_ms=statistics.median(unpickle_times),
        matrix_export_ms=matrix_ms,
        matrix_bytes=matrix_bytes,
    )


def run_profile(sizes: list[int], repeats: int = 20) -> list[SerializationProfile]:
    return [profile_one(size, max(1, size * 4), repeats=repeats) for size in sizes]


def save_profile(results: list[SerializationProfile], out_dir: Path | None = None) -> Path:
    if out_dir is None:
        out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = out_dir / f"{ts}_serialization_profile.json"
    path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile YANE serialization costs")
    parser.add_argument("--sizes", nargs="+", type=int, default=[0, 10, 50, 200])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    results = run_profile(args.sizes, repeats=args.repeats)
    for r in results:
        print(
            f"hidden={r.hidden_nodes:4d} conns={r.connections:5d} "
            f"pickle={r.pickle_bytes:8d}B {r.pickle_ms:7.3f}ms "
            f"unpickle={r.unpickle_ms:7.3f}ms matrix={r.matrix_export_ms}"
        )
    print(f"Saved to {save_profile(results, args.out_dir)}")


if __name__ == "__main__":
    main()
