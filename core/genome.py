from __future__ import annotations
import random

from yane.core.connection import Connection
from yane.core.node import Node, NodeType
from yane.evolution.mutation import Mutation

# Attributes shared between copy() and crossover() — scalars and Mutation objects.
_SCALAR_GENES = (
    'bypass_connection_prob', 'crossover_prob', 'offspring_factor',
    'species_threshold', 'sigma_global',
)
_MUTATION_GENES = (
    'mutation_bypass', 'mutation_add_node', 'mutation_remove_node',
    'mutation_add_connection', 'mutation_remove_connection',
    'mutation_crossover', 'mutation_offspring', 'mutation_species_threshold',
    'mutation_sigma',
)

# Strategy-gene mutation specs: (attr, mutation_attr, min_val, max_val | None)
_STRATEGY_MUTATION_SPECS = (
    ('bypass_connection_prob', 'mutation_bypass',             0.0,  1.0),
    ('crossover_prob',         'mutation_crossover',          0.0,  1.0),
    ('offspring_factor',       'mutation_offspring',           0.01, None),
    ('species_threshold',      'mutation_species_threshold',   0.01, 1.0),
    ('sigma_global',           'mutation_sigma',               0.01, None),
)


def _pick(a, b):
    """Return a or b with equal probability (used in crossover for gene selection)."""
    return a if random.random() < 0.5 else b


class Genome:
    def __init__(self) -> None:
        self.fitness: float = 0.0
        self.nodes: list[Node] = []
        self.input_nodes: list[Node] = []
        self.output_nodes: list[Node] = []
        self._triggered: set[Node] = set()
        self._connection_count: int = 0   # cached; updated by _invalidate_topology()
        self._exec_order: list | None = None  # topological order; None = not yet computed
        self._has_cycles: bool = False        # True = skip topo sort, use BFS
        self._reset_nodes: list | None = None # nodes that need explicit reset before forward
        self._compiled_forward = None         # cached closure; avoids attribute lookup in hot loop

        # Optional size caps — set by NeuroEvolution.configure()
        self.max_nodes: int | None = None
        self.max_connections: int | None = None

        # Structural mutation rates
        self.mutation_add_node = Mutation()
        self.mutation_remove_node = Mutation()
        self.mutation_add_connection = Mutation()
        self.mutation_remove_connection = Mutation()
        self.bypass_connection_prob: float = 0.5
        self.mutation_bypass = Mutation()

        # ── Self-adaptive strategy genes ─────────────────────────────────────
        # These are inherited and mutated like any other gene, so the
        # population discovers good values without any external tuning.

        # Probability this genome reproduces via crossover (vs. pure mutation)
        self.crossover_prob: float = 0.3
        self.mutation_crossover = Mutation()

        # Relative reproduction drive; higher → more likely to be selected
        # as a parent. Balanced by selection pressure over time.
        self.offspring_factor: float = 1.0
        self.mutation_offspring = Mutation()

        # Compatibility-distance threshold for same-species membership
        self.species_threshold: float = 0.3
        self.mutation_species_threshold = Mutation()

        # Global step-size scale for weight/bias mutations (CMA-ES style)
        self.sigma_global: float = 1.0
        self.mutation_sigma = Mutation()

        # Set by Population.submit(); initialised here so the attribute always exists.
        self.shared_fitness: float = 0.0

    # -------------------------------------------------------------------------
    # Tick mode
    # -------------------------------------------------------------------------

    def set_inputs(self, data: list[float]) -> None:
        n = len(data)
        for node in self.input_nodes:
            if node.input_index < n:
                node.value = data[node.input_index]
                self._triggered.add(node)

    def tick(self) -> None:
        if not self._triggered:
            return
        next_triggered: set[Node] = set()
        for node in self._triggered:
            node.fire(next_triggered)
        self._triggered = next_triggered

    def get_outputs(self) -> list[float]:
        return [node.value for node in self.output_nodes]

    def reset(self) -> None:
        self._triggered.clear()
        for node in self.nodes:
            if not node.persist_value:
                node.value = 0.0

    # -------------------------------------------------------------------------
    # Forward mode (full pass with cycle protection)
    # -------------------------------------------------------------------------

    def _compile_forward(self):
        """Build a closure that captures node lists so forward() makes zero attribute lookups."""
        reset_nodes = [
            n for n in self.nodes
            if n.persist_value and n not in self.input_nodes
        ]
        self._reset_nodes = reset_nodes
        input_nodes = self.input_nodes
        exec_order = self._exec_order
        output_nodes = self.output_nodes
        def _forward(data: list[float]) -> list[float]:
            for node in reset_nodes:
                node.value = 0.0
            n = len(data)
            for node in input_nodes:
                node.value = data[node.input_index] if node.input_index < n else 0.0
                node.fire_simple()
            for node in exec_order:
                node.fire_simple()
            return [node.value for node in output_nodes]

        return _forward

    def _build_exec_order(self) -> list[Node] | None:
        """Topological sort (Kahn's algorithm) for acyclic networks.

        Input nodes are treated as already-processed sources: their outgoing
        edges are counted but their in_degree contribution is immediately
        removed so their targets can enter the queue first.

        Returns non-input nodes in execution order, or None if cycles exist.
        """
        inputs = set(self.input_nodes)
        in_degree: dict[Node, int] = {n: 0 for n in self.nodes if n not in inputs}
        for node in self.nodes:
            for conn in node.connections:
                if conn.target in in_degree:
                    in_degree[conn.target] += 1

        # Input nodes are "pre-processed" — decrement their targets immediately.
        queue: list[Node] = []
        for inp in self.input_nodes:
            for conn in inp.connections:
                if conn.target in in_degree:
                    in_degree[conn.target] -= 1
                    if in_degree[conn.target] == 0:
                        queue.append(conn.target)
        # Nodes with no incoming connections at all (isolated hidden nodes).
        for node, deg in in_degree.items():
            if deg == 0 and node not in queue:
                queue.append(node)

        order: list[Node] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for conn in node.connections:
                t = conn.target
                if t in in_degree:
                    in_degree[t] -= 1
                    if in_degree[t] == 0:
                        queue.append(t)

        if len(order) != len(in_degree):
            return None  # cycle detected → fall back to BFS
        return order

    def forward(self, data: list[float]) -> list[float]:
        # Use cached topological order for acyclic networks (no BFS overhead).
        if self._exec_order is None and not self._has_cycles:
            result = self._build_exec_order()
            if result is not None:
                self._exec_order = result
            else:
                self._has_cycles = True

        if self._exec_order is not None:
            if self._compiled_forward is None:
                self._compiled_forward = self._compile_forward()
            return self._compiled_forward(data)

        # Slow path: BFS with cycle protection (recurrent networks).
        self._triggered.clear()
        for node in self.nodes:
            node.value = 0.0
        self.set_inputs(data)

        trigger_counts: dict[Node, int] = {}
        pending: list[Node] = [n for n in self.nodes if n in self._triggered]
        next_pending: set[Node] = set()

        while pending:
            next_pending.clear()
            for node in pending:
                cnt = trigger_counts.get(node, 0)
                if cnt >= node.max_triggers:
                    continue
                trigger_counts[node] = cnt + 1
                node.fire(next_pending)
            pending = [n for n in self.nodes if n in next_pending]

        return self.get_outputs()

    # -------------------------------------------------------------------------
    # Mutation
    # -------------------------------------------------------------------------

    def mutate(self) -> None:
        from yane.evolution import smart_mutation

        if self.mutation_add_node.mutate_bool(False):
            smart_mutation.add_node(self)
        if self.mutation_remove_node.mutate_bool(False):
            smart_mutation.remove_node(self)
        if self.mutation_add_connection.mutate_bool(False):
            smart_mutation.add_connection(self)
        if self.mutation_remove_connection.mutate_bool(False):
            smart_mutation.remove_connection(self)

        sigma = self.sigma_global
        for node in self.nodes:
            node.mutate(sigma)

        for attr, mut_attr, lo, hi in _STRATEGY_MUTATION_SPECS:
            val = getattr(self, mut_attr).mutate_value(getattr(self, attr))
            if lo is not None:
                val = max(lo, val)
            if hi is not None:
                val = min(hi, val)
            setattr(self, attr, val)

        for mut_attr in _MUTATION_GENES:
            getattr(self, mut_attr).mutate_rates()

    # -------------------------------------------------------------------------
    # Crossover
    # -------------------------------------------------------------------------

    def crossover(self, other: Genome) -> Genome:
        """Combine this genome (assumed fitter) with other.

        Fixed nodes (input/output) come from self. Hidden nodes are inherited
        from self always and from other with 50% probability each. Connections
        from self are always kept; connections from other are added where both
        endpoints exist in the child, or override existing weights at 50%.
        Strategy genes are drawn randomly from either parent.
        """
        child = Genome()
        child.max_nodes = self.max_nodes
        child.max_connections = self.max_connections

        node_map: dict[Node, Node] = {}  # parent node → child node

        # Input nodes from self (anchor structure)
        for node in self.input_nodes:
            n = node.copy()
            node_map[node] = n
            child.nodes.append(n)
            child.input_nodes.append(n)

        # Output nodes from self
        for node in self.output_nodes:
            n = node.copy()
            node_map[node] = n
            child.nodes.append(n)
            child.output_nodes.append(n)

        # Map other's fixed nodes to child's (by position)
        for s, o in zip(self.input_nodes, other.input_nodes):
            node_map[o] = node_map[s]
        for s, o in zip(self.output_nodes, other.output_nodes):
            node_map[o] = node_map[s]

        # Hidden nodes from self (always inherited)
        for node in self.nodes:
            if node.type == NodeType.HIDDEN:
                n = node.copy()
                node_map[node] = n
                child.nodes.append(n)

        # Hidden nodes from other (50% each, respecting max_nodes cap)
        for node in other.nodes:
            if node.type == NodeType.HIDDEN and node not in node_map:
                if child.max_nodes is not None and len(child.nodes) >= child.max_nodes:
                    break
                if random.random() < 0.5:
                    n = node.copy()
                    node_map[node] = n
                    child.nodes.append(n)

        # Build connections; track count to avoid O(N) scan per connection.
        existing: set[tuple[int, int]] = set()
        conn_count = 0

        def _add_conns(source_nodes: list[Node], override_weight: bool) -> None:
            nonlocal conn_count
            for old_src in source_nodes:
                if old_src not in node_map:
                    continue
                child_src = node_map[old_src]
                for conn in old_src.connections:
                    if conn.target not in node_map:
                        continue
                    child_tgt = node_map[conn.target]
                    key = (id(child_src), id(child_tgt))
                    if key not in existing:
                        if child.max_connections is not None and conn_count >= child.max_connections:
                            continue
                        new_conn = Connection(child_tgt)
                        new_conn.weight = conn.weight
                        new_conn.mutation = conn.mutation.copy()
                        child_src.connections.append(new_conn)
                        existing.add(key)
                        conn_count += 1
                    elif override_weight and random.random() < 0.5:
                        for c in child_src.connections:
                            if c.target is child_tgt:
                                c.weight = conn.weight
                                break

        _add_conns(self.nodes, False)
        _add_conns(other.nodes, True)

        # Strategy genes: random from either parent
        for attr in _SCALAR_GENES:
            setattr(child, attr, _pick(getattr(self, attr), getattr(other, attr)))
        for attr in _MUTATION_GENES:
            setattr(child, attr, _pick(getattr(self, attr), getattr(other, attr)).copy())

        child._invalidate_topology()
        return child

    # -------------------------------------------------------------------------
    # Copy
    # -------------------------------------------------------------------------

    def copy(self) -> Genome:
        genome = Genome()

        node_map: dict[Node, Node] = {}
        for node in self.nodes:
            new_node = node.copy()
            node_map[node] = new_node
            genome.nodes.append(new_node)
            if node.type == NodeType.INPUT:
                genome.input_nodes.append(new_node)
            elif node.type == NodeType.OUTPUT:
                genome.output_nodes.append(new_node)

        for old_node, new_node in node_map.items():
            for conn in old_node.connections:
                if conn.target in node_map:
                    new_node.connections.append(conn.copy(node_map))

        genome.fitness = self.fitness
        genome.max_nodes = self.max_nodes
        genome.max_connections = self.max_connections
        for attr in _SCALAR_GENES:
            setattr(genome, attr, getattr(self, attr))
        for attr in _MUTATION_GENES:
            setattr(genome, attr, getattr(self, attr).copy())

        genome._connection_count = self._connection_count
        # Topology caches reference old Node objects — must recompute for the copy.
        genome._exec_order = None
        genome._reset_nodes = None
        genome._compiled_forward = None
        genome._has_cycles = self._has_cycles
        return genome

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def _clear(self) -> None:
        """Break all internal reference cycles so Python can immediately GC this genome.

        Node A → Connection → Node B → Connection → Node A creates a cycle that
        Python's reference counter cannot resolve alone; it waits for the cyclic GC.
        In fast training loops genomes are discarded faster than the GC runs, causing
        RAM to grow. Explicitly clearing connections breaks every cycle here.
        """
        for node in self.nodes:
            node.connections.clear()
        self.nodes.clear()
        self.input_nodes.clear()
        self.output_nodes.clear()
        self._triggered.clear()

    def all_connections(self) -> list[tuple[Node, Connection]]:
        """All (source, connection) pairs in the network."""
        return [(node, conn) for node in self.nodes for conn in node.connections]

    def _invalidate_topology(self) -> None:
        self._connection_count = sum(len(n.connections) for n in self.nodes)
        self._exec_order = None
        self._has_cycles = False
        self._reset_nodes = None
        self._compiled_forward = None

    @property
    def connection_count(self) -> int:
        return self._connection_count

    def memory_info(self) -> dict:
        """Returns a breakdown of node/connection counts for memory profiling."""
        return {
            "nodes": len(self.nodes),
            "connections": self.connection_count,
            "input_nodes": len(self.input_nodes),
            "output_nodes": len(self.output_nodes),
            "max_nodes": self.max_nodes,
            "max_connections": self.max_connections,
        }
