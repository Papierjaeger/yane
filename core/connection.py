from __future__ import annotations
import random
from typing import TYPE_CHECKING

from yane.evolution.mutation import Mutation

if TYPE_CHECKING:
    from yane.core.node import Node


class Connection:
    __slots__ = ('target', '_weight', 'mutation', 'innovation', '__weakref__',
                 '_weight_arr', '_weight_idx', 'enabled', 'spike_rate',
                 'weight_group',
                 # STDP / Hebbian plasticity — evolvable per-connection coefficients.
                 # Δw = hebb_a*pre + hebb_b*post + hebb_c*pre*post + hebb_d
                 # All default to 0.0 (no plasticity = zero cost).
                 'hebb_a', 'hebb_b', 'hebb_c', 'hebb_d',
                 # Base weight saved at episode start; restored by genome.reset().
                 # None = STDP not active for this connection.
                 '_base_weight')

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
        # Enabled flag — disabled connections remain in the genome but are
        # excluded from the forward pass.  Allows reversible pruning.
        self.enabled: bool = True
        # Spike probability: occasionally re-initialise weight completely to
        # escape weight-space local optima.  Self-adapts via mutation.rate_mutation_rate.
        self.spike_rate: float = 0.05
        # Shared weight group identifier (None = not grouped).
        self.weight_group: str | None = None
        # Hebbian plasticity coefficients (STDP).
        self.hebb_a: float = 0.0
        self.hebb_b: float = 0.0
        self.hebb_c: float = 0.0
        self.hebb_d: float = 0.0
        self._base_weight: float | None = None

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
        # _base_weight is episode-local state; not persisted across checkpoints.
        return {
            'target': self.target, 'weight': self._weight,
            'mutation': self.mutation, 'innovation': self.innovation,
            'enabled': self.enabled, 'spike_rate': self.spike_rate,
            'weight_group': self.weight_group,
            'hebb_a': self.hebb_a, 'hebb_b': self.hebb_b,
            'hebb_c': self.hebb_c, 'hebb_d': self.hebb_d,
        }

    def __setstate__(self, state):
        self.target      = state['target']
        self._weight     = state.get('weight', state.get('_weight', 0.0))
        self.mutation    = state['mutation']
        self.innovation  = state.get('innovation', -1)
        self._weight_arr = None
        self._weight_idx = 0
        self.enabled     = state.get('enabled', True)
        self.spike_rate  = state.get('spike_rate', 0.05)
        self.weight_group = state.get('weight_group', None)
        self.hebb_a      = state.get('hebb_a', 0.0)
        self.hebb_b      = state.get('hebb_b', 0.0)
        self.hebb_c      = state.get('hebb_c', 0.0)
        self.hebb_d      = state.get('hebb_d', 0.0)
        self._base_weight = None  # episode-local; never restored from pickle

    # ------------------------------------------------------------------
    # Mutation / copy
    # ------------------------------------------------------------------

    def mutate(self, sigma: float = 1.0) -> None:
        # Spike: occasionally re-initialise weight completely to escape local optima.
        # Self-adapts: spike_rate is scaled by the same rate_mutation_rate used for
        # all other per-connection rates, keeping the system fully self-adaptive.
        if random.random() < self.spike_rate:
            self.weight = random.gauss(0.0, sigma)
        else:
            self.weight = self.mutation.mutate_value(self._weight, sigma)
        if random.random() < self.mutation.rate_mutation_rate:
            scale = random.uniform(0.9, 1.1)
            self.spike_rate = max(0.001, min(0.3, self.spike_rate * scale))
        self.mutation.mutate_rates()

    def copy(self, node_map: dict[Node, Node]) -> Connection:
        conn = Connection(node_map[self.target], innovation=self.innovation)
        conn.weight = self._weight   # property setter; _weight_arr is None on new conn
        conn.mutation = self.mutation.copy()
        conn.enabled = self.enabled
        conn.spike_rate = self.spike_rate
        conn.weight_group = self.weight_group
        conn.hebb_a = self.hebb_a
        conn.hebb_b = self.hebb_b
        conn.hebb_c = self.hebb_c
        conn.hebb_d = self.hebb_d
        # _base_weight is episode-local; offspring starts fresh
        return conn
