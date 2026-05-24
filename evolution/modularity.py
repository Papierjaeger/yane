"""Module-level topology helpers."""
from __future__ import annotations

import random
from collections import deque
from typing import TYPE_CHECKING

from yane.core.connection import Connection
from yane.core.node import Node, NodeType

if TYPE_CHECKING:
    from yane.core.genome import Genome
    from yane.evolution.innovation import InnovationTracker


def hidden_modules(genome: "Genome") -> list[list[Node]]:
    """Return weakly connected components among hidden nodes."""
    hidden = [n for n in genome.nodes if n.type is NodeType.HIDDEN]
    hidden_set = set(hidden)
    neighbors: dict[Node, set[Node]] = {n: set() for n in hidden}
    for src in genome.nodes:
        for conn in src.connections:
            tgt = conn.target
            if src in hidden_set and tgt in hidden_set:
                neighbors[src].add(tgt)
                neighbors[tgt].add(src)

    modules: list[list[Node]] = []
    seen: set[Node] = set()
    for start in hidden:
        if start in seen:
            continue
        comp: list[Node] = []
        queue = deque([start])
        seen.add(start)
        while queue:
            node = queue.popleft()
            comp.append(node)
            for nxt in neighbors[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        modules.append(comp)
    return modules


def duplicate_module(
    genome: "Genome",
    tracker: "InnovationTracker | None" = None,
    module: list[Node] | None = None,
) -> bool:
    """Duplicate a hidden-node module in-place.

    Copies hidden nodes, their internal connections, incoming connections from
    outside the module, and outgoing connections to outside targets. Returns
    False when no hidden module can be duplicated or size caps would be exceeded.
    """
    modules = hidden_modules(genome)
    if module is None:
        if not modules:
            return False
        module = random.choice(modules)
    if not module:
        return False
    if genome.max_nodes is not None and len(genome.nodes) + len(module) > genome.max_nodes:
        return False

    module_set = set(module)
    new_nodes: dict[Node, Node] = {}
    for old in module:
        innov = tracker.next() if tracker is not None else old.innovation
        new = old.copy()
        new.innovation = innov
        new.connections = []
        new.gate_node = None
        new_nodes[old] = new

    planned_connections: list[tuple[Node, Node, Connection]] = []
    for src in genome.nodes:
        for conn in src.connections:
            src_inside = src in module_set
            tgt_inside = conn.target in module_set
            if src_inside and tgt_inside:
                planned_connections.append((new_nodes[src], new_nodes[conn.target], conn))
            elif (not src_inside) and tgt_inside:
                planned_connections.append((src, new_nodes[conn.target], conn))
            elif src_inside and not tgt_inside:
                planned_connections.append((new_nodes[src], conn.target, conn))

    if genome.max_connections is not None:
        if genome.connection_count + len(planned_connections) > genome.max_connections:
            return False

    genome.nodes.extend(new_nodes.values())
    for src, target, old_conn in planned_connections:
        innov = (
            tracker.get_connection(src.innovation, target.innovation)
            if tracker is not None else old_conn.innovation
        )
        new_conn = Connection(target, innovation=innov)
        new_conn.weight = old_conn.weight
        new_conn.enabled = old_conn.enabled
        new_conn.spike_rate = old_conn.spike_rate
        new_conn.mutation = old_conn.mutation.copy()
        src.connections.append(new_conn)

    genome._invalidate_topology()
    return True
