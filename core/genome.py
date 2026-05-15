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
        for node in self.input_nodes:
            if 0 <= node.input_index < len(data):
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

    def forward(self, data: list[float]) -> list[float]:
        # Hard reset ignores persist_value so each call is independent (unlike tick/reset).
        self._triggered.clear()
        for node in self.nodes:
            node.value = 0.0
        self.set_inputs(data)

        # Filter self.nodes to maintain list order (deterministic, no lambda/dict overhead).
        # set() membership check is O(1); iterating self.nodes is O(n) with n typically < 30.
        trigger_counts: dict[Node, int] = {}
        pending: list[Node] = [n for n in self.nodes if n in self._triggered]

        while pending:
            next_pending: set[Node] = set()
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

    @property
    def connection_count(self) -> int:
        # Called O(N²) during speciation — computed fresh each time since
        # mutations can add/remove connections at any point.
        # Fast enough for typical network sizes (< 200 nodes).
        return sum(len(n.connections) for n in self.nodes)

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
