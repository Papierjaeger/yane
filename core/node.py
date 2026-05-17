from __future__ import annotations
from enum import Enum

from yane.evolution.mutation import Mutation
from yane.util.activation import ActivationType, ActivationFunction, ACTIVATION_FNS
from yane.core.connection import Connection


class NodeType(Enum):
    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"


class Node:
    __slots__ = (
        'type', 'value', 'bias', '_activation', '_activate_fn',
        'persist_value', 'max_triggers', 'input_index', 'input_scale', 'connections',
        'mutation_bias', 'mutation_activation', 'mutation_persist',
        'mutation_max_triggers', 'mutation_input_index', 'mutation_input_scale',
        'innovation',
    )

    # Slot-level defaults for __setstate__ (handles genomes serialised before input_scale existed).
    _SLOT_DEFAULTS: dict = {'innovation': -1, 'input_scale': 1.0}

    def __init__(self, node_type: NodeType = NodeType.HIDDEN, innovation: int = -1) -> None:
        self.type = node_type
        self.value: float = 0.0
        self.bias: float = 0.0
        self.innovation: int = innovation   # global unique ID; -1 = untracked legacy
        self._activation = ActivationType.SIGMOID
        self._activate_fn = ACTIVATION_FNS[ActivationType.SIGMOID]
        self.persist_value: bool = False
        self.max_triggers: int = 3
        self.input_index: int = 0
        self.input_scale: float = 1.0   # per-input scaling gene — evolves to normalise raw inputs

        self.connections: list[Connection] = []

        self.mutation_bias = Mutation()
        self.mutation_activation = Mutation()
        self.mutation_persist = Mutation()
        self.mutation_max_triggers = Mutation()
        self.mutation_input_index = Mutation()
        self.mutation_input_scale = Mutation()

    def __getstate__(self):
        return {slot: getattr(self, slot) for slot in self.__slots__}

    def __setstate__(self, state):
        for slot in self.__slots__:
            default = self._SLOT_DEFAULTS.get(slot)
            val = state.get(slot, default)
            # Reconstruct missing Mutation objects (e.g. genomes serialised before this gene existed)
            if val is None and slot.startswith('mutation_'):
                val = Mutation()
            object.__setattr__(self, slot, val)

    @property
    def activation(self) -> ActivationType:
        return self._activation

    @activation.setter
    def activation(self, value: ActivationType) -> None:
        self._activation = value
        self._activate_fn = ACTIVATION_FNS[value]

    def fire(self, next_triggered: set[Node]) -> None:
        activated = self._activate_fn(self.value + self.bias)
        for conn in self.connections:
            conn.target.value += conn.weight * activated
            next_triggered.add(conn.target)
        self.value = activated if self.persist_value else 0.0

    def fire_simple(self) -> None:
        """Fast path for acyclic (topologically sorted) networks — no set tracking."""
        activated = self._activate_fn(self.value + self.bias)
        for conn in self.connections:
            conn.target.value += conn.weight * activated
        self.value = activated if self.persist_value else 0.0

    def mutate(self, sigma: float = 1.0) -> None:
        self.bias = self.mutation_bias.mutate_value(self.bias, sigma)
        self.activation = self.mutation_activation.mutate_enum(self.activation, ActivationType)
        self.persist_value = self.mutation_persist.mutate_bool(self.persist_value)
        self.max_triggers = self.mutation_max_triggers.mutate_int(self.max_triggers, 1, 10)

        if self.type == NodeType.INPUT:
            self.input_index = self.mutation_input_index.mutate_int(self.input_index, 0, 255)
            # input_scale: multiplicative gene, kept strictly positive.
            # Allows the network to self-normalise raw input magnitudes without
            # requiring the user to pre-scale their data.
            new_scale = self.mutation_input_scale.mutate_value(self.input_scale, sigma)
            self.input_scale = max(1e-6, new_scale)
            self.mutation_input_scale.mutate_rates()

        for conn in self.connections:
            conn.mutate(sigma)

        self.mutation_bias.mutate_rates()
        self.mutation_activation.mutate_rates()
        self.mutation_persist.mutate_rates()
        self.mutation_max_triggers.mutate_rates()
        self.mutation_input_index.mutate_rates()

    def copy(self) -> Node:
        n = Node(self.type, innovation=self.innovation)
        n.bias = self.bias
        n.activation = self.activation
        n.persist_value = self.persist_value
        n.max_triggers = self.max_triggers
        n.input_index = self.input_index
        n.input_scale = self.input_scale
        n.mutation_bias = self.mutation_bias.copy()
        n.mutation_activation = self.mutation_activation.copy()
        n.mutation_persist = self.mutation_persist.copy()
        n.mutation_max_triggers = self.mutation_max_triggers.copy()
        n.mutation_input_index = self.mutation_input_index.copy()
        n.mutation_input_scale = self.mutation_input_scale.copy()
        return n
