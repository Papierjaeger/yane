"""Module-level topology helpers."""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from yane.core.connection import Connection
from yane.core.node import Node, NodeType

if TYPE_CHECKING:
    from yane.core.genome import Genome
    from yane.evolution.innovation import InnovationTracker


@dataclass
class ModuleBlueprint:
    """Serializable hidden-subgraph blueprint for reuse across genomes."""

    name: str
    nodes: list[Node]
    internal_edges: list[tuple[int, int, Connection]]
    input_edges: list[tuple[int, int, Connection]]
    output_edges: list[tuple[int, int, Connection]]
    source_fitness: float = 0.0
    uses: int = 0

    @property
    def size(self) -> int:
        return len(self.nodes)


@dataclass
class ModuleLibrary:
    """Small elite library of reusable hidden modules."""

    max_modules: int = 50
    min_fitness: float | None = None
    modules: list[ModuleBlueprint] = field(default_factory=list)
    n_added: int = 0
    n_inserted: int = 0
    n_reused: int = 0

    def add_from_genome(
        self,
        genome: "Genome",
        module: list[Node] | None = None,
        name: str | None = None,
    ) -> bool:
        if self.min_fitness is not None and genome.fitness < self.min_fitness:
            return False
        modules = [module] if module is not None else hidden_modules(genome)
        candidates = [m for m in modules if m]
        if not candidates:
            return False
        chosen = max(candidates, key=len)
        blueprint = extract_module_blueprint(
            genome,
            chosen,
            name=name or f"module_{self.n_added}",
        )
        if blueprint is None:
            return False
        self.modules.append(blueprint)
        self.modules.sort(key=lambda item: (item.source_fitness, item.size), reverse=True)
        if len(self.modules) > self.max_modules:
            self.modules = self.modules[: self.max_modules]
        self.n_added += 1
        return True

    def sample(self, rng: random.Random | None = None) -> ModuleBlueprint | None:
        if not self.modules:
            return None
        chooser = rng or random
        return chooser.choice(self.modules)

    def insert_into(
        self,
        genome: "Genome",
        tracker: "InnovationTracker | None" = None,
        rng: random.Random | None = None,
    ) -> bool:
        blueprint = self.sample(rng)
        if blueprint is None:
            return False
        if insert_module_blueprint(genome, blueprint, tracker):
            blueprint.uses += 1
            self.n_inserted += 1
            self.n_reused += 1
            return True
        return False

    def diagnostics(self) -> dict:
        total_uses = sum(m.uses for m in self.modules)
        return {
            "module_count": len(self.modules),
            "n_added": self.n_added,
            "n_inserted": self.n_inserted,
            "n_reused": self.n_reused,
            "total_uses": total_uses,
            "reuse_rate": total_uses / max(1, self.n_added),
            "modules": [
                {
                    "name": m.name,
                    "size": m.size,
                    "source_fitness": m.source_fitness,
                    "uses": m.uses,
                    "internal_edges": len(m.internal_edges),
                    "input_edges": len(m.input_edges),
                    "output_edges": len(m.output_edges),
                }
                for m in self.modules
            ],
        }


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


def _copy_connection_template(conn: Connection, target: Node) -> Connection:
    new_conn = Connection(target, innovation=conn.innovation)
    new_conn.weight = conn.weight
    new_conn.enabled = conn.enabled
    new_conn.spike_rate = conn.spike_rate
    new_conn.mutation = conn.mutation.copy()
    return new_conn


def extract_module_blueprint(
    genome: "Genome",
    module: list[Node],
    name: str = "module",
) -> ModuleBlueprint | None:
    """Extract a reusable blueprint from a hidden-node module."""
    if not module:
        return None
    module_set = set(module)
    index = {node: i for i, node in enumerate(module)}
    node_templates = []
    for node in module:
        clone = node.copy()
        clone.connections = []
        clone.gate_node = None
        node_templates.append(clone)

    internal_edges: list[tuple[int, int, Connection]] = []
    input_edges: list[tuple[int, int, Connection]] = []
    output_edges: list[tuple[int, int, Connection]] = []
    input_index = {node: i for i, node in enumerate(genome.input_nodes)}
    output_index = {node: i for i, node in enumerate(genome.output_nodes)}

    for src in genome.nodes:
        for conn in src.connections:
            target = conn.target
            if src in module_set and target in module_set:
                internal_edges.append((index[src], index[target], conn))
            elif src in input_index and target in module_set:
                input_edges.append((input_index[src], index[target], conn))
            elif src in module_set and target in output_index:
                output_edges.append((index[src], output_index[target], conn))

    return ModuleBlueprint(
        name=name,
        nodes=node_templates,
        internal_edges=internal_edges,
        input_edges=input_edges,
        output_edges=output_edges,
        source_fitness=float(getattr(genome, "fitness", 0.0)),
    )


def insert_module_blueprint(
    genome: "Genome",
    blueprint: ModuleBlueprint,
    tracker: "InnovationTracker | None" = None,
) -> bool:
    """Insert a module blueprint into a compatible genome."""
    if not blueprint.nodes:
        return False
    if genome.max_nodes is not None and len(genome.nodes) + len(blueprint.nodes) > genome.max_nodes:
        return False
    n_edges = len(blueprint.internal_edges) + len(blueprint.input_edges) + len(blueprint.output_edges)
    if genome.max_connections is not None and genome.connection_count + n_edges > genome.max_connections:
        return False
    if any(i >= len(genome.input_nodes) for i, _t, _c in blueprint.input_edges):
        return False
    if any(o >= len(genome.output_nodes) for _s, o, _c in blueprint.output_edges):
        return False

    new_nodes: list[Node] = []
    for old in blueprint.nodes:
        new = old.copy()
        new.connections = []
        new.gate_node = None
        new.innovation = tracker.next() if tracker is not None else old.innovation
        new_nodes.append(new)

    planned: list[tuple[Node, Node, Connection]] = []
    for src_i, tgt_i, conn in blueprint.internal_edges:
        planned.append((new_nodes[src_i], new_nodes[tgt_i], conn))
    for input_i, tgt_i, conn in blueprint.input_edges:
        planned.append((genome.input_nodes[input_i], new_nodes[tgt_i], conn))
    for src_i, output_i, conn in blueprint.output_edges:
        planned.append((new_nodes[src_i], genome.output_nodes[output_i], conn))

    genome.nodes.extend(new_nodes)
    for src, target, old_conn in planned:
        new_conn = _copy_connection_template(old_conn, target)
        if tracker is not None:
            new_conn.innovation = tracker.get_connection(src.innovation, target.innovation)
        src.connections.append(new_conn)
    genome._invalidate_topology()
    return True


def module_crossover(
    recipient: "Genome",
    donor: "Genome",
    tracker: "InnovationTracker | None" = None,
    rng: random.Random | None = None,
) -> bool:
    """Copy a compatible donor hidden module into recipient."""
    modules = hidden_modules(donor)
    modules = [m for m in modules if m]
    if not modules:
        return False
    chooser = rng or random
    module = chooser.choice(modules)
    blueprint = extract_module_blueprint(donor, module, name="crossover")
    if blueprint is None:
        return False
    return insert_module_blueprint(recipient, blueprint, tracker)


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
        new_conn = _copy_connection_template(old_conn, target)
        if tracker is not None:
            new_conn.innovation = tracker.get_connection(src.innovation, target.innovation)
        src.connections.append(new_conn)

    genome._invalidate_topology()
    return True
