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
        # _persist_value is the backing store; the public `persist_value` property
        # keeps _retain_value in sync automatically on every write.
        '_persist_value', '_retain_value',
        'max_triggers', 'input_index', 'input_scale', 'connections',
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
        # Use slot directly here to avoid property overhead during init.
        object.__setattr__(self, '_persist_value', False)
        # Pre-computed: True iff this node must keep its activated value after firing.
        # Updated by the persist_value property setter so it's always consistent.
        object.__setattr__(self, '_retain_value', node_type is NodeType.OUTPUT)
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

    # -- persist_value property -----------------------------------------------
    # Exposes _persist_value and keeps _retain_value in sync on every write.

    @property
    def persist_value(self) -> bool:
        return self._persist_value

    @persist_value.setter
    def persist_value(self, value: bool) -> None:
        self._persist_value = value
        self._retain_value = (self.type is NodeType.OUTPUT) or value

    # -- Pickle support -------------------------------------------------------

    def __getstate__(self):
        state = {}
        for slot in self.__slots__:
            state[slot] = getattr(self, slot)
        # Expose as 'persist_value' (not '_persist_value') so old unpickling code
        # and external tooling always sees the public name.
        state['persist_value'] = state.pop('_persist_value')
        return state

    def __setstate__(self, state):
        # Normalise: old pickles use 'persist_value'; new pickles also use it
        # (see __getstate__).  Either way, write to the backing slot.
        pv = state.pop('persist_value', None)
        if pv is None:
            pv = state.pop('_persist_value', False)

        for slot in self.__slots__:
            if slot in ('_persist_value', '_retain_value'):
                continue
            default = self._SLOT_DEFAULTS.get(slot)
            val = state.get(slot, default)
            if val is None and slot.startswith('mutation_'):
                val = Mutation()
            object.__setattr__(self, slot, val)

        object.__setattr__(self, '_persist_value', pv)
        object.__setattr__(self, '_retain_value', (self.type is NodeType.OUTPUT) or pv)

    # -- Activation property --------------------------------------------------

    @property
    def activation(self) -> ActivationType:
        return self._activation

    @activation.setter
    def activation(self, value: ActivationType) -> None:
        self._activation = value
        self._activate_fn = ACTIVATION_FNS[value]

    # -- Forward pass ---------------------------------------------------------

    def fire(self, next_triggered: set[Node]) -> None:
        v = self.value + self.bias
        # Guard: some activations (sin, cos) raise ValueError/OverflowError for
        # inf/nan inputs.  try/except costs ~15ns on the happy path (no exception)
        # vs ~70ns for math.isfinite().  Treat non-finite as 0 (dead node).
        try:
            activated = self._activate_fn(v)
        except (ValueError, OverflowError):
            activated = 0.0
        for conn in self.connections:
            conn.target.value += conn.weight * activated
            next_triggered.add(conn.target)
        # _retain_value is pre-computed from `type is OUTPUT or persist_value`,
        # avoiding the compound check on every fire() call.
        if self._retain_value:
            self.value = activated
        else:
            self.value = 0.0

    def fire_simple(self) -> None:
        """Fast path for acyclic (topologically sorted) networks — no set tracking."""
        v = self.value + self.bias
        try:
            activated = self._activate_fn(v)
        except (ValueError, OverflowError):
            activated = 0.0
        for conn in self.connections:
            conn.target.value += conn.weight * activated
        if self._retain_value:
            self.value = activated
        else:
            self.value = 0.0

    # -- Mutation / copy ------------------------------------------------------

    def mutate(self, sigma: float = 1.0) -> None:
        self.bias = self.mutation_bias.mutate_value(self.bias, sigma)
        self.activation = self.mutation_activation.mutate_enum(self.activation, ActivationType)
        # Use property setter so _retain_value stays in sync.
        self.persist_value = self.mutation_persist.mutate_bool(self._persist_value)
        self.max_triggers = self.mutation_max_triggers.mutate_int(self.max_triggers, 1, 10)

        if self.type == NodeType.INPUT:
            self.input_index = self.mutation_input_index.mutate_int(self.input_index, 0, 255)
            new_scale = self.mutation_input_scale.mutate_value(self.input_scale, sigma)
            self.input_scale = 1e-6 if new_scale < 1e-6 else new_scale
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
        # Use property setter so _retain_value is correctly initialised.
        n.persist_value = self._persist_value
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
