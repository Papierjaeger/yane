"""Simple CPPN/HyperNEAT-style indirect connection generation."""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from yane.core.connection import Connection
from yane.core.node import Node

if TYPE_CHECKING:
    from yane.core.genome import Genome
    from yane.evolution.innovation import InnovationTracker


CoordinateMap = dict[int, tuple[float, float]]
WeightFn = Callable[[float, float, float, float, float], float]


def layered_coordinates(genome: "Genome") -> CoordinateMap:
    """Assign deterministic 2D coordinates to input/hidden/output nodes."""
    coords: CoordinateMap = {}

    def place(nodes: list[Node], x: float) -> None:
        n = max(1, len(nodes))
        for i, node in enumerate(nodes):
            y = 0.0 if n == 1 else -1.0 + 2.0 * i / (n - 1)
            coords[node.innovation] = (x, y)

    hidden = [n for n in genome.nodes if n not in genome.input_nodes and n not in genome.output_nodes]
    place(genome.input_nodes, -1.0)
    place(hidden, 0.0)
    place(genome.output_nodes, 1.0)
    return coords


def radial_cppn_weight(x1: float, y1: float, x2: float, y2: float, distance: float) -> float:
    """A deterministic CPPN-like pattern useful for smoke tests and demos."""
    return math.sin(3.0 * x1 + 5.0 * y2) * math.cos(2.0 * distance)


def generate_connections_from_coordinates(
    genome: "Genome",
    coordinates: CoordinateMap | None = None,
    weight_fn: WeightFn = radial_cppn_weight,
    threshold: float = 0.2,
    tracker: "InnovationTracker | None" = None,
    feed_forward_only: bool = True,
) -> int:
    """Generate connections from a coordinate-based weight function."""
    if coordinates is None:
        coordinates = layered_coordinates(genome)
    existing = {(src, conn.target) for src in genome.nodes for conn in src.connections}
    added = 0
    for src in genome.nodes:
        if src.innovation not in coordinates:
            continue
        x1, y1 = coordinates[src.innovation]
        for target in genome.nodes:
            if src is target or target in genome.input_nodes:
                continue
            if feed_forward_only and coordinates.get(src.innovation, (0.0, 0.0))[0] >= coordinates.get(target.innovation, (0.0, 0.0))[0]:
                continue
            if (src, target) in existing:
                continue
            if genome.max_connections is not None and genome.connection_count + added >= genome.max_connections:
                break
            if target.innovation not in coordinates:
                continue
            x2, y2 = coordinates[target.innovation]
            distance = math.hypot(x2 - x1, y2 - y1)
            weight = float(weight_fn(x1, y1, x2, y2, distance))
            if abs(weight) < threshold:
                continue
            innov = tracker.get_connection(src.innovation, target.innovation) if tracker else -1
            conn = Connection(target, innovation=innov)
            conn.weight = weight
            src.connections.append(conn)
            existing.add((src, target))
            added += 1
    if added:
        genome._invalidate_topology()
    return added
