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
    evolve: bool = False,
    cppn: "CPPNGenome | Genome | WeightFn | None" = None,
    variance_threshold: float = 0.03,
    max_depth: int = 3,
    initial_resolution: int = 3,
) -> "Substrate":
    """Create layered substrate coordinates and feed-forward connection pairs.

    When ``evolve=True`` (ES-HyperNEAT mode) a CPPN-driven Quadtree search
    adds hidden nodes at positions where the CPPN shows high variance.
    Requires *cppn* to be provided.

    Parameters
    ----------
    n_inputs, n_outputs :
        Fixed topology for input and output layers.
    hidden_layers :
        Extra manual hidden layers placed between inputs and outputs.
    evolve :
        When *True*, run the ES-HyperNEAT algorithm to discover hidden-node
        positions automatically.  Requires *cppn*.
    cppn :
        CPPN used for variance-based node placement (only needed when
        ``evolve=True``).
    variance_threshold, max_depth, initial_resolution :
        ES-HyperNEAT Quadtree parameters (see :func:`es_hyperneat_substrate`).
    """
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
    base = Substrate(coords, input_indices, output_indices, hidden_indices, pairs)

    if evolve:
        if cppn is None:
            raise ValueError("hyperneat_substrate(evolve=True) requires a cppn argument.")
        return _es_hyperneat_evolve_substrate(
            cppn, base,
            variance_threshold=variance_threshold,
            max_depth=max_depth,
            initial_resolution=initial_resolution,
        )
    return base


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
    evolve_substrate: bool = False,
    es_variance_threshold: float = 0.03,
    es_max_depth: int = 3,
    es_initial_resolution: int = 3,
) -> Genome:
    """Generate a network genome by decoding an evolvable CPPN on a substrate.

    When ``evolve_substrate=True`` the function first runs the ES-HyperNEAT
    Quadtree algorithm to discover new hidden-node positions, augments
    *substrate* with those positions, then decodes the CPPN as usual.

    Parameters
    ----------
    cppn :
        CPPN callable, :class:`CPPNGenome`, or plain :class:`Genome`.
    substrate :
        Substrate defining input/output positions (and optionally initial
        hidden positions).  When ``evolve_substrate=True`` additional hidden
        nodes may be inserted.
    threshold :
        Minimum |weight| to include a connection.
    tracker :
        Optional :class:`InnovationTracker` for consistent innovation numbers.
    evolve_substrate :
        When *True*, run ES-HyperNEAT Quadtree placement to discover hidden
        node positions automatically.
    es_variance_threshold :
        Minimum CPPN-output variance in a region to trigger a node placement
        or further subdivision.
    es_max_depth :
        Maximum Quadtree recursion depth.
    es_initial_resolution :
        Number of initial grid cells per axis (before Quadtree refinement).
    """
    if evolve_substrate:
        substrate = _es_hyperneat_evolve_substrate(
            cppn, substrate,
            variance_threshold=es_variance_threshold,
            max_depth=es_max_depth,
            initial_resolution=es_initial_resolution,
        )
    pattern = generate_weight_pattern(cppn, substrate, threshold=threshold)
    return build_genome_from_substrate(substrate, pattern, tracker=tracker)


# ---------------------------------------------------------------------------
# ES-HyperNEAT — variance-based Quadtree substrate evolution
# ---------------------------------------------------------------------------

def _cppn_variance(
    weight_fn: WeightFn,
    x: float,
    y: float,
    eps: float = 0.05,
) -> float:
    """Estimate CPPN output variance at position *(x, y)* using nearby probes.

    Queries the CPPN at four nearby positions relative to *(x, y)* as both
    source and target.  High variance indicates a feature boundary —
    the ES-HyperNEAT signal to place a substrate node here.
    """
    offsets = [(-eps, 0.0), (eps, 0.0), (0.0, -eps), (0.0, eps)]
    values: list[float] = []
    for dx, dy in offsets:
        dist = math.hypot(dx, dy)
        w = weight_fn(x, y, x + dx, y + dy, dist)
        values.append(float(w))
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _quadtree_place_nodes(
    weight_fn: WeightFn,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    depth: int,
    max_depth: int,
    variance_threshold: float,
    positions: list[tuple[float, float]],
) -> None:
    """Recursive Quadtree subdivision that places nodes at high-variance regions.

    At each level:
    1. Compute CPPN variance at the region centre.
    2. If variance is above *variance_threshold* AND depth < *max_depth*:
       subdivide into four quadrants and recurse.
    3. If variance is above *variance_threshold* AND depth == *max_depth*:
       place a node at the centre (leaf with high variance).
    4. If variance is below threshold: stop (region is featureless).
    """
    if depth > max_depth:
        return
    cx = (x_lo + x_hi) / 2.0
    cy = (y_lo + y_hi) / 2.0
    var = _cppn_variance(weight_fn, cx, cy)
    if var < variance_threshold:
        return  # featureless region — prune
    if depth == max_depth:
        # Leaf with sufficient variance → place a node
        positions.append((cx, cy))
        return
    # Subdivide into 4 quadrants
    for qx_lo, qx_hi in [(x_lo, cx), (cx, x_hi)]:
        for qy_lo, qy_hi in [(y_lo, cy), (cy, y_hi)]:
            _quadtree_place_nodes(
                weight_fn, qx_lo, qx_hi, qy_lo, qy_hi,
                depth + 1, max_depth, variance_threshold, positions,
            )


def es_hyperneat_substrate(
    cppn: CPPNGenome | Genome | WeightFn,
    n_inputs: int,
    n_outputs: int,
    hidden_layers: tuple[int, ...] = (),
    variance_threshold: float = 0.03,
    max_depth: int = 3,
    initial_resolution: int = 3,
    x_range: tuple[float, float] = (-1.0, 1.0),
    y_range: tuple[float, float] = (-1.0, 1.0),
) -> Substrate:
    """Create a substrate whose hidden-node positions are determined by the CPPN.

    This is the ES-HyperNEAT (Evolvable Substrate HyperNEAT) algorithm: instead
    of manually specifying hidden node positions, the CPPN's own variance
    landscape drives a Quadtree search that discovers where nodes are needed.

    Parameters
    ----------
    cppn :
        A CPPN (``CPPNGenome``, plain ``Genome``, or weight callable) whose
        output variance guides node placement.
    n_inputs, n_outputs :
        Fixed topology for input and output layers (same as regular HyperNEAT).
    hidden_layers :
        Additional manually-placed hidden layer sizes.  Pass ``()`` to let
        the Quadtree determine all hidden positions.
    variance_threshold :
        Minimum CPPN-output variance in a region to subdivide or place a node.
        Lower values → more hidden nodes.  Typical range: 0.01–0.1.
    max_depth :
        Maximum Quadtree recursion depth (controls the maximum number of
        automatically placed nodes: up to ``4^max_depth``).
    initial_resolution :
        Number of equal-width cells per axis to scan before Quadtree refinement.
    x_range, y_range :
        Bounding box of the hidden-node search space in normalised coordinates.

    Returns
    -------
    Substrate
        A :class:`Substrate` that includes the regular input/output positions
        plus any hidden nodes discovered by the Quadtree search.
    """
    # Build base substrate (inputs + outputs + optional manual hidden layers)
    base = hyperneat_substrate(n_inputs, n_outputs, hidden_layers=hidden_layers)

    # Resolve CPPN to a weight function
    if isinstance(cppn, CPPNGenome):
        weight_fn: WeightFn = cppn.weight
    elif isinstance(cppn, Genome):
        weight_fn = CPPNGenome(cppn).weight
    else:
        weight_fn = cppn

    # Run Quadtree subdivision to discover hidden node positions
    discovered: list[tuple[float, float]] = []
    x_lo, x_hi = x_range
    y_lo, y_hi = y_range
    step_x = (x_hi - x_lo) / initial_resolution
    step_y = (y_hi - y_lo) / initial_resolution
    for i in range(initial_resolution):
        for j in range(initial_resolution):
            _quadtree_place_nodes(
                weight_fn,
                x_lo + i * step_x, x_lo + (i + 1) * step_x,
                y_lo + j * step_y, y_lo + (j + 1) * step_y,
                depth=0,
                max_depth=max_depth,
                variance_threshold=variance_threshold,
                positions=discovered,
            )

    # Deduplicate discovered positions (round to 4 decimals)
    seen: set[tuple[float, float]] = set()
    unique_hidden: list[tuple[float, float]] = []
    for pos in discovered:
        key = (round(pos[0], 4), round(pos[1], 4))
        if key not in seen:
            seen.add(key)
            unique_hidden.append(pos)

    if not unique_hidden:
        # Fallback: CPPN is too uniform — use one centre node
        cx = (x_lo + x_hi) / 2.0
        cy = (y_lo + y_hi) / 2.0
        unique_hidden = [(cx, cy)]

    # Merge with base substrate
    n_base = len(base.coordinates)
    new_coords = list(base.coordinates) + unique_hidden
    new_hidden_indices = list(base.hidden_indices) + list(
        range(n_base, n_base + len(unique_hidden))
    )

    # All new hidden nodes connect to all output nodes (full output connectivity)
    extra_pairs: list[tuple[int, int]] = []
    for h_idx in range(n_base, n_base + len(unique_hidden)):
        for o_idx in base.output_indices:
            extra_pairs.append((h_idx, o_idx))
    # Input nodes connect to all hidden nodes too
    for i_idx in base.input_indices:
        for h_idx in range(n_base, n_base + len(unique_hidden)):
            extra_pairs.append((i_idx, h_idx))

    return Substrate(
        coordinates=new_coords,
        input_indices=base.input_indices,
        output_indices=base.output_indices,
        hidden_indices=new_hidden_indices,
        pairs=base.pairs + extra_pairs,
    )


def _es_hyperneat_evolve_substrate(
    cppn: CPPNGenome | Genome | WeightFn,
    substrate: Substrate,
    variance_threshold: float = 0.03,
    max_depth: int = 3,
    initial_resolution: int = 3,
) -> Substrate:
    """Augment *substrate* with hidden nodes discovered via ES-HyperNEAT Quadtree."""
    if isinstance(cppn, CPPNGenome):
        weight_fn: WeightFn = cppn.weight
    elif isinstance(cppn, Genome):
        weight_fn = CPPNGenome(cppn).weight
    else:
        weight_fn = cppn

    discovered: list[tuple[float, float]] = []
    for i in range(initial_resolution):
        for j in range(initial_resolution):
            x_step = 2.0 / initial_resolution
            y_step = 2.0 / initial_resolution
            _quadtree_place_nodes(
                weight_fn,
                -1.0 + i * x_step, -1.0 + (i + 1) * x_step,
                -1.0 + j * y_step, -1.0 + (j + 1) * y_step,
                depth=0,
                max_depth=max_depth,
                variance_threshold=variance_threshold,
                positions=discovered,
            )

    if not discovered:
        return substrate

    # Deduplicate
    seen: set[tuple[float, float]] = set()
    existing = set(map(tuple, substrate.coordinates))
    new_hidden: list[tuple[float, float]] = []
    for pos in discovered:
        key = (round(pos[0], 4), round(pos[1], 4))
        if key not in seen and tuple(key) not in existing:
            seen.add(key)
            new_hidden.append(pos)

    if not new_hidden:
        return substrate

    n_base = len(substrate.coordinates)
    new_coords = list(substrate.coordinates) + new_hidden
    new_hidden_indices = list(substrate.hidden_indices) + list(
        range(n_base, n_base + len(new_hidden))
    )

    extra_pairs: list[tuple[int, int]] = []
    for h_idx in range(n_base, n_base + len(new_hidden)):
        for o_idx in substrate.output_indices:
            extra_pairs.append((h_idx, o_idx))
    for i_idx in substrate.input_indices:
        for h_idx in range(n_base, n_base + len(new_hidden)):
            extra_pairs.append((i_idx, h_idx))

    return Substrate(
        coordinates=new_coords,
        input_indices=substrate.input_indices,
        output_indices=substrate.output_indices,
        hidden_indices=new_hidden_indices,
        pairs=substrate.pairs + extra_pairs,
    )


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
