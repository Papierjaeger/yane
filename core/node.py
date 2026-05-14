from __future__ import annotations
from enum import Enum

from yane.evolution.mutation import Mutation
from yane.util.activation import ActivationType, ActivationFunction
from yane.core.connection import Connection


class NodeType(Enum):
    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"


class Node:
    def __init__(self, node_type: NodeType = NodeType.HIDDEN) -> None:
        self.type = node_type
        self.value: float = 0.0
        self.bias: float = 0.0
        self.activation = ActivationType.SIGMOID
        self.persist_value: bool = False
        self.max_triggers: int = 3
        self.input_index: int = 0

        self.connections: list[Connection] = []

        self.mutation_bias = Mutation()
        self.mutation_activation = Mutation()
        self.mutation_persist = Mutation()
        self.mutation_max_triggers = Mutation()
        self.mutation_input_index = Mutation()

    def fire(self, next_triggered: set[Node]) -> None:
        activated = ActivationFunction.activate(self.activation, self.value + self.bias)
        for conn in self.connections:
            conn.target.value += conn.weight * activated
            next_triggered.add(conn.target)
        if not self.persist_value:
            self.value = 0.0

    def mutate(self) -> None:
        self.bias = self.mutation_bias.mutate_value(self.bias)
        self.activation = self.mutation_activation.mutate_enum(self.activation, ActivationType)
        self.persist_value = self.mutation_persist.mutate_bool(self.persist_value)
        self.max_triggers = self.mutation_max_triggers.mutate_int(self.max_triggers, 1, 10)

        if self.type == NodeType.INPUT:
            self.input_index = self.mutation_input_index.mutate_int(self.input_index, 0, 255)

        for conn in self.connections:
            conn.mutate()

        self.mutation_bias.mutate_rates()
        self.mutation_activation.mutate_rates()
        self.mutation_persist.mutate_rates()
        self.mutation_max_triggers.mutate_rates()
        self.mutation_input_index.mutate_rates()

    def copy(self) -> Node:
        n = Node(self.type)
        n.bias = self.bias
        n.activation = self.activation
        n.persist_value = self.persist_value
        n.max_triggers = self.max_triggers
        n.input_index = self.input_index
        n.mutation_bias = self.mutation_bias.copy()
        n.mutation_activation = self.mutation_activation.copy()
        n.mutation_persist = self.mutation_persist.copy()
        n.mutation_max_triggers = self.mutation_max_triggers.copy()
        n.mutation_input_index = self.mutation_input_index.copy()
        return n
