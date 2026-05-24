"""Simple CPPN/HyperNEAT-style indirect connection generation."""
from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from yane.core.connection import Connection
from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.util.activation import ActivationType

if TYPE_CHECKING:
    from yane.evolution.innovation import InnovationTracker


CoordinateMap = dict[int, tuple[float, float]]
WeightFn = Callable[[float, float, float, float, float], float]


@dataclass
class Substrate:
    """HyperNEAT-style substrate coordinates and directed connection pairs."""

    coordinates: list[tuple[float, float]]
    input_indices: list[int]
    output_indices: list[int]
    hidden_indices: list[int]
    pairs: list[tuple[int, int]]


class CPPNGenome:
    """Small evolvable CPPN represented as a normal YANE Genome."""

    def __init__(self, genome: Genome | None = None) -> None:
        self.genome = genome or self._make_default_genome()

    @staticmethod
    def _make_default_genome() -> Genome:
        genome = Genome()
        for idx in range(5):
            node = Node(NodeType.INPUT, innovation=idx)
            node.input_index = idx
            node.activation = ActivationType.LINEAR
            genome.nodes.append(node)
            genome.input_nodes.append(node)
        out = Node(NodeType.OUTPUT, innovation=5)
        out.activation = ActivationType.TANH
        genome.nodes.append(out)
        genome.output_nodes.append(out)
        for inp in genome.input_nodes:
            conn = Connection(out, innovation=100 + inp.input_index)
            conn.weight = random.uniform(-1.0, 1.0)
            inp.connections.append(conn)
        genome._invalidate_topology()
        return genome

    def weight(self, x1: float, y1: float, x2: float, y2: float, distance: float) -> float:
        self.genome.reset()
        out = self.genome.forward([x1, y1, x2, y2, distance])
        return float(out[0] if out else 0.0)

    def mutate(self, tracker: "InnovationTracker | None" = None) -> None:
        self.genome.mutate(tracker)

    def copy(self) -> "CPPNGenome":
        return CPPNGenome(self.genome.copy())


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


def hyperneat_substrate(
    n_inputs: int,
    n_outputs: int,
    hidden_layers: tuple[int, ...] = (),
) -> Substrate:
    """Create layered substrate coordinates and feed-forward connection pairs."""
    layers: list[list[int]] = []
    coords: list[tuple[float, float]] = []
    total_layers = 2 + len(hidden_layers)
    layer_sizes = [n_inputs, *hidden_layers, n_outputs]
    idx = 0
    for layer_i, size in enumerate(layer_sizes):
        x = -1.0 if total_layers == 1 else -1.0 + 2.0 * layer_i / (total_layers - 1)
        layer: list[int] = []
        for j in range(size):
            y = 0.0 if size <= 1 else -1.0 + 2.0 * j / (size - 1)
            coords.append((x, y))
            layer.append(idx)
            idx += 1
        layers.append(layer)

    pairs: list[tuple[int, int]] = []
    for src_layer, tgt_layer in zip(layers, layers[1:]):
        for src in src_layer:
            for tgt in tgt_layer:
                pairs.append((src, tgt))
    input_indices = layers[0]
    output_indices = layers[-1]
    hidden_indices = [i for layer in layers[1:-1] for i in layer]
    return Substrate(coords, input_indices, output_indices, hidden_indices, pairs)


def generate_weight_pattern(
    cppn: CPPNGenome | Genome | WeightFn,
    substrate: Substrate,
    threshold: float = 0.2,
) -> list[tuple[tuple[int, int], float]]:
    """Generate a sparse substrate weight pattern from an evolvable CPPN."""
    if isinstance(cppn, CPPNGenome):
        weight_fn = cppn.weight
    elif isinstance(cppn, Genome):
        wrapped = CPPNGenome(cppn)
        weight_fn = wrapped.weight
    else:
        weight_fn = cppn

    pattern: list[tuple[tuple[int, int], float]] = []
    for src, tgt in substrate.pairs:
        x1, y1 = substrate.coordinates[src]
        x2, y2 = substrate.coordinates[tgt]
        distance = math.hypot(x2 - x1, y2 - y1)
        weight = float(weight_fn(x1, y1, x2, y2, distance))
        if abs(weight) >= threshold:
            pattern.append(((src, tgt), weight))
    return pattern


def build_genome_from_substrate(
    substrate: Substrate,
    pattern: list[tuple[tuple[int, int], float]],
    tracker: "InnovationTracker | None" = None,
) -> Genome:
    """Build a YANE Genome from substrate nodes and a CPPN-generated pattern."""
    genome = Genome()
    node_map: dict[int, Node] = {}
    for idx, _coord in enumerate(substrate.coordinates):
        if idx in substrate.input_indices:
            node = Node(NodeType.INPUT, innovation=tracker.next() if tracker else idx)
            node.input_index = len(genome.input_nodes)
            node.activation = ActivationType.LINEAR
            genome.input_nodes.append(node)
        elif idx in substrate.output_indices:
            node = Node(NodeType.OUTPUT, innovation=tracker.next() if tracker else idx)
            node.activation = ActivationType.SIGMOID
            genome.output_nodes.append(node)
        else:
            node = Node(NodeType.HIDDEN, innovation=tracker.next() if tracker else idx)
            node.activation = ActivationType.TANH
        genome.nodes.append(node)
        node_map[idx] = node

    for (src_i, tgt_i), weight in pattern:
        src = node_map[src_i]
        tgt = node_map[tgt_i]
        innov = tracker.get_connection(src.innovation, tgt.innovation) if tracker else -1
        conn = Connection(tgt, innovation=innov)
        conn.weight = weight
        src.connections.append(conn)
    genome._invalidate_topology()
    return genome


def generate_genome_from_cppn(
    cppn: CPPNGenome | Genome | WeightFn,
    substrate: Substrate,
    threshold: float = 0.2,
    tracker: "InnovationTracker | None" = None,
) -> Genome:
    """Generate a network genome by decoding an evolvable CPPN on a substrate."""
    pattern = generate_weight_pattern(cppn, substrate, threshold=threshold)
    return build_genome_from_substrate(substrate, pattern, tracker=tracker)


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
