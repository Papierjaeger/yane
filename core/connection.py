from __future__ import annotations
import random
from typing import TYPE_CHECKING

from yane.evolution.mutation import Mutation

if TYPE_CHECKING:
    from yane.core.node import Node


class Connection:
    __slots__ = ('target', '_weight', 'mutation', 'innovation', '__weakref__',
                 '_weight_arr', '_weight_idx')

    def __init__(self, target: Node, innovation: int = -1) -> None:
        self.target = target
        self._weight: float = random.uniform(-1.0, 1.0)
        self.mutation = Mutation()
        self.innovation: int = innovation   # global unique ID; -1 = untracked legacy
        # Numpy weight-array sync — set by _compile_forward(), None otherwise.
        # When set, every weight assignment also writes to the pre-built array so
        # the compiled forward closure always sees the current weight without
        # rebuilding the array on each call.
        self._weight_arr = None   # np.ndarray | None
        self._weight_idx: int = 0

    # ------------------------------------------------------------------
    # weight property — keeps _weight_arr in sync on every assignment
    # ------------------------------------------------------------------

    @property
    def weight(self) -> float:
        return self._weight

    @weight.setter
    def weight(self, value: float) -> None:
        self._weight = value
        if self._weight_arr is not None:
            self._weight_arr[self._weight_idx] = value

    # ------------------------------------------------------------------
    # Pickle support
    # ------------------------------------------------------------------

    def __getstate__(self):
        # Expose as 'weight' (not '_weight') for backward compat.
        # _weight_arr / _weight_idx are not pickled — rebuilt by _compile_forward().
        return {
            'target': self.target, 'weight': self._weight,
            'mutation': self.mutation, 'innovation': self.innovation,
        }

    def __setstate__(self, state):
        self.target     = state['target']
        self._weight    = state.get('weight', state.get('_weight', 0.0))
        self.mutation   = state['mutation']
        self.innovation = state.get('innovation', -1)
        self._weight_arr = None
        self._weight_idx = 0

    # ------------------------------------------------------------------
    # Mutation / copy
    # ------------------------------------------------------------------

    def mutate(self, sigma: float = 1.0) -> None:
        # Goes through the property setter — updates _weight_arr if linked.
        self.weight = self.mutation.mutate_value(self._weight, sigma)
        self.mutation.mutate_rates()

    def copy(self, node_map: dict[Node, Node]) -> Connection:
        conn = Connection(node_map[self.target], innovation=self.innovation)
        conn.weight = self._weight   # property setter; _weight_arr is None on new conn
        conn.mutation = self.mutation.copy()
        return conn
