from __future__ import annotations
from collections import defaultdict

from yane.core.connection import Connection
from yane.core.node import Node, NodeType
from yane.evolution.mutation import Mutation


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

        self.mutation_add_node = Mutation()
        self.mutation_remove_node = Mutation()
        self.mutation_add_connection = Mutation()
        self.mutation_remove_connection = Mutation()
        self.bypass_connection_prob: float = 0.5
        self.mutation_bypass = Mutation()

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
        self.reset()
        self.set_inputs(data)

        pending = set(self._triggered)
        trigger_counts: defaultdict[Node, int] = defaultdict(int)

        while pending:
            next_pending: set[Node] = set()
            for node in pending:
                if trigger_counts[node] >= node.max_triggers:
                    continue
                trigger_counts[node] += 1
                node.fire(next_pending)
            pending = next_pending

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

        for node in self.nodes:
            node.mutate()

        self.bypass_connection_prob = self.mutation_bypass.mutate_value(self.bypass_connection_prob)
        self.bypass_connection_prob = max(0.0, min(1.0, self.bypass_connection_prob))

        self.mutation_add_node.mutate_rates()
        self.mutation_remove_node.mutate_rates()
        self.mutation_add_connection.mutate_rates()
        self.mutation_remove_connection.mutate_rates()
        self.mutation_bypass.mutate_rates()

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

        genome.max_nodes = self.max_nodes
        genome.max_connections = self.max_connections
        genome.mutation_add_node = self.mutation_add_node.copy()
        genome.mutation_remove_node = self.mutation_remove_node.copy()
        genome.mutation_add_connection = self.mutation_add_connection.copy()
        genome.mutation_remove_connection = self.mutation_remove_connection.copy()
        genome.bypass_connection_prob = self.bypass_connection_prob
        genome.mutation_bypass = self.mutation_bypass.copy()

        return genome

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def all_connections(self) -> list[tuple[Node, Connection]]:
        """All (source, connection) pairs in the network."""
        return [(node, conn) for node in self.nodes for conn in node.connections]

    @property
    def connection_count(self) -> int:
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
