from __future__ import annotations
import random
from typing import TYPE_CHECKING

from yane.evolution.mutation import Mutation

if TYPE_CHECKING:
    from yane.core.node import Node


class Connection:
    __slots__ = ('target', 'weight', 'mutation', '__weakref__')

    def __init__(self, target: Node) -> None:
        self.target = target
        self.weight: float = random.uniform(-1.0, 1.0)
        self.mutation = Mutation()

    def __getstate__(self):
        return {'target': self.target, 'weight': self.weight, 'mutation': self.mutation}

    def __setstate__(self, state):
        self.target = state['target']
        self.weight = state['weight']
        self.mutation = state['mutation']

    def mutate(self, sigma: float = 1.0) -> None:
        self.weight = self.mutation.mutate_value(self.weight, sigma)
        self.mutation.mutate_rates()

    def copy(self, node_map: dict[Node, Node]) -> Connection:
        conn = Connection(node_map[self.target])
        conn.weight = self.weight
        conn.mutation = self.mutation.copy()
        return conn
