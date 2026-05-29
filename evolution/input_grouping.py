"""Evolvable Input Aggregation Layer for YANE.

Reduces the effective input dimensionality for high-dimensional tasks by
grouping raw input channels into aggregated "virtual" inputs.  The grouping
structure is encoded on the :class:`~yane.core.genome.Genome` object (via its
``grouper`` attribute) and is preserved through crossover, mutation, and
checkpoints.

Overview
--------
A :class:`InputGrouper` maps *N* raw input values to *K* grouped values
(K ≤ N).  Each :class:`InputGroup` aggregates a subset of the raw inputs
using one of four reduction strategies: ``mean``, ``max``, ``sum``, or
``weighted_sum``.

Integration with ``NeuroEvolution``
------------------------------------
1. Call ``yane.set_input_grouping(n_groups=K, n_raw=N)`` **before**
   ``yane.configure()``.
2. The network topology is built with *K* input nodes (not *N*).
3. During training, ``InputGrouper.transform(raw)`` maps raw inputs to the
   *K* grouped values before they enter the network.

Mutation operators
------------------
``InputGrouper.split_group(group_idx)``
    Split one group into two, increasing *K* by 1.  Use the companion
    ``apply_split_to_genome(genome, group_idx)`` to also add the
    corresponding input node to the genome.

``InputGrouper.merge_groups(idx_a, idx_b)``
    Merge two groups into one, decreasing *K* by 1.  The companion
    ``apply_merge_to_genome(genome, idx_a, idx_b)`` removes the
    now-unused input node from the genome.

``InputGrouper.add_input_to_group(raw_idx, group_idx)``
    Move a raw input into a group.

``InputGrouper.remove_input_from_group(raw_idx, group_idx)``
    Remove a raw input from a group (must keep ≥ 1 member).

``InputGrouper.change_aggregation(group_idx, new_agg)``
    Switch the reduction strategy for a group.

``InputGrouper.create_group(members, agg)``
    Add a new empty group.  Use ``apply_create_group_to_genome`` to keep
    the genome's input_nodes count in sync.

Crossover
---------
``InputGrouper.crossover(other)``
    Uniform group-level crossover.  When parent groupers have different
    sizes, the longer parent's extra groups are inherited by the offspring.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome


# ---------------------------------------------------------------------------
# Aggregation type
# ---------------------------------------------------------------------------

class AggType(str, Enum):
    """Aggregation strategy for an :class:`InputGroup`."""
    MEAN = "mean"
    MAX = "max"
    SUM = "sum"
    WEIGHTED_SUM = "weighted_sum"


# ---------------------------------------------------------------------------
# InputGroup
# ---------------------------------------------------------------------------

@dataclass
class InputGroup:
    """One aggregation group: a subset of raw input channels."""

    members: list[int]
    """Indices into the raw input vector that belong to this group."""

    aggregation: AggType = AggType.MEAN
    """How to combine the member values into a single output."""

    weights: list[float] = field(default_factory=list)
    """Per-member weights used only when ``aggregation == WEIGHTED_SUM``.
    When empty or length-mismatch, falls back to unweighted mean."""

    enabled: bool = True
    """Disabled groups are skipped in :meth:`InputGrouper.transform`."""

    def copy(self) -> InputGroup:
        return InputGroup(
            members=list(self.members),
            aggregation=self.aggregation,
            weights=list(self.weights),
            enabled=self.enabled,
        )


# ---------------------------------------------------------------------------
# InputGrouper
# ---------------------------------------------------------------------------

class InputGrouper:
    """Evolvable input aggregation layer.

    Parameters
    ----------
    n_raw :
        Number of raw input channels (fixed; determines valid member indices).
    initial_groups :
        Starting group layout.  When *None* each raw input gets its own
        single-member group (identity mapping: K = N).
    """

    def __init__(
        self,
        n_raw: int,
        initial_groups: list[InputGroup] | None = None,
    ) -> None:
        self.n_raw = n_raw
        if initial_groups is not None:
            self.groups: list[InputGroup] = [g.copy() for g in initial_groups]
        else:
            # Identity: each raw input is its own group
            self.groups = [InputGroup(members=[i]) for i in range(n_raw)]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_outputs(self) -> int:
        """Number of values produced by :meth:`transform` (= enabled groups)."""
        return sum(1 for g in self.groups if g.enabled)

    # ------------------------------------------------------------------
    # Core transform
    # ------------------------------------------------------------------

    def transform(self, raw_inputs: list[float]) -> list[float]:
        """Map *N* raw inputs to *K* aggregated values.

        Disabled groups are skipped.  Groups with no valid members (indices
        out of range) produce 0.0.

        Parameters
        ----------
        raw_inputs :
            Raw input vector of length ``n_raw`` (or shorter — out-of-range
            members are silently skipped).

        Returns
        -------
        list[float]
            Aggregated values, one per enabled group, in group order.
        """
        result: list[float] = []
        for group in self.groups:
            if not group.enabled:
                continue
            members = [i for i in group.members if 0 <= i < len(raw_inputs)]
            if not members:
                result.append(0.0)
                continue
            values = [raw_inputs[i] for i in members]
            agg = group.aggregation
            if agg == AggType.MAX:
                result.append(max(values))
            elif agg == AggType.SUM:
                result.append(sum(values))
            elif agg == AggType.WEIGHTED_SUM and len(group.weights) == len(values):
                result.append(sum(w * v for w, v in zip(group.weights, values)))
            else:  # MEAN (default and fallback)
                result.append(sum(values) / len(values))
        return result

    # ------------------------------------------------------------------
    # Mutation operators
    # ------------------------------------------------------------------

    def split_group(self, group_idx: int) -> int:
        """Split a group in two, adding a new group at the end.

        The original group retains the first half of its members; the new
        group receives the remaining half.  If the group has only one member,
        both halves contain that single member (the group is duplicated).

        Parameters
        ----------
        group_idx :
            Index of the group to split.

        Returns
        -------
        int
            Index of the newly created group.
        """
        group = self.groups[group_idx]
        mid = max(1, len(group.members) // 2)
        first_half = group.members[:mid]
        second_half = group.members[mid:] if len(group.members) > 1 else list(group.members)
        group.members = first_half
        new_group = InputGroup(
            members=second_half,
            aggregation=group.aggregation,
            weights=list(group.weights[mid:]) if group.weights else [],
        )
        self.groups.append(new_group)
        return len(self.groups) - 1

    def merge_groups(self, idx_a: int, idx_b: int) -> None:
        """Merge group *idx_b* into group *idx_a* and disable *idx_b*.

        Members are deduplicated (each raw index appears at most once in the
        merged group).
        """
        if idx_a == idx_b or idx_a >= len(self.groups) or idx_b >= len(self.groups):
            return
        a = self.groups[idx_a]
        b = self.groups[idx_b]
        combined = list(dict.fromkeys(a.members + b.members))  # preserve order, deduplicate
        a.members = combined
        b.enabled = False

    def create_group(
        self,
        members: list[int],
        agg: AggType = AggType.MEAN,
    ) -> int:
        """Add a new group at the end.

        Parameters
        ----------
        members :
            Raw input indices that belong to the new group.
        agg :
            Aggregation strategy.

        Returns
        -------
        int
            Index of the new group.
        """
        self.groups.append(InputGroup(members=list(members), aggregation=agg))
        return len(self.groups) - 1

    def add_input_to_group(self, raw_idx: int, group_idx: int) -> None:
        """Add raw input *raw_idx* to group *group_idx*.

        No-op if *raw_idx* is already a member.
        """
        if 0 <= group_idx < len(self.groups):
            if raw_idx not in self.groups[group_idx].members:
                self.groups[group_idx].members.append(raw_idx)

    def remove_input_from_group(self, raw_idx: int, group_idx: int) -> None:
        """Remove raw input *raw_idx* from group *group_idx*.

        No-op if the group would become empty (must keep ≥ 1 member).
        """
        if 0 <= group_idx < len(self.groups):
            g = self.groups[group_idx]
            if raw_idx in g.members and len(g.members) > 1:
                g.members.remove(raw_idx)

    def change_aggregation(self, group_idx: int, new_agg: AggType) -> None:
        """Switch the aggregation strategy for group *group_idx*."""
        if 0 <= group_idx < len(self.groups):
            self.groups[group_idx].aggregation = new_agg

    # ------------------------------------------------------------------
    # Crossover and copy
    # ------------------------------------------------------------------

    def crossover(self, other: InputGrouper) -> InputGrouper:
        """Uniform group-level crossover.

        For positions where both parents have a group, each is chosen with
        50 % probability.  Extra groups from the larger parent (``self``) are
        appended unchanged.

        Parameters
        ----------
        other :
            The other parent's :class:`InputGrouper`.  Need not have the same
            number of groups — a crossover with different layouts is safe.

        Returns
        -------
        InputGrouper
            New child grouper with ``n_raw = self.n_raw``.
        """
        child = InputGrouper.__new__(InputGrouper)
        child.n_raw = self.n_raw
        child.groups = []
        a_groups = [g for g in self.groups if g.enabled]
        b_groups = [g for g in other.groups if g.enabled]
        shared = min(len(a_groups), len(b_groups))
        for i in range(shared):
            child.groups.append(
                a_groups[i].copy() if random.random() < 0.5 else b_groups[i].copy()
            )
        # Inherit extra groups from the longer parent (self)
        for g in a_groups[shared:]:
            child.groups.append(g.copy())
        return child

    def copy(self) -> InputGrouper:
        """Return a deep copy of this grouper."""
        c = InputGrouper.__new__(InputGrouper)
        c.n_raw = self.n_raw
        c.groups = [g.copy() for g in self.groups]
        return c


# ---------------------------------------------------------------------------
# Genome-level mutation helpers (modify both grouper and genome topology)
# ---------------------------------------------------------------------------

def apply_split_to_genome(genome: "Genome", group_idx: int) -> int:
    """Split a group and add a corresponding input node to *genome*.

    The grouper on *genome* must not be *None*.  After the call,
    ``genome.grouper.n_outputs`` is increased by 1 and a new
    ``NodeType.INPUT`` node has been appended to both
    ``genome.nodes`` and ``genome.input_nodes``.

    Parameters
    ----------
    genome :
        The genome whose ``grouper`` is modified.
    group_idx :
        Index of the group to split.

    Returns
    -------
    int
        Index of the newly created group.
    """
    from yane.core.node import Node, NodeType
    from yane.util.activation import ActivationType

    grouper = genome.grouper
    new_group_idx = grouper.split_group(group_idx)

    # Add a new input node to the genome topology
    next_innov = max((n.innovation for n in genome.nodes if n.innovation >= 0), default=-1) + 1
    new_node = Node(NodeType.INPUT, next_innov)
    new_node.activation = ActivationType.LINEAR
    genome.input_nodes.append(new_node)
    genome.nodes.append(new_node)
    genome._invalidate_topology()
    return new_group_idx


def apply_merge_to_genome(genome: "Genome", idx_a: int, idx_b: int) -> None:
    """Merge two groups and remove the now-unused input node from *genome*.

    The grouper on *genome* must not be *None*.  If ``idx_a == idx_b`` or
    either index is invalid, the call is a safe no-op.

    After the call, ``genome.grouper.n_outputs`` is decreased by 1 and the
    last ``NodeType.INPUT`` node has been removed from ``genome.input_nodes``
    (and ``genome.nodes``).  Any connections from that node are also removed.

    Parameters
    ----------
    genome :
        The genome whose ``grouper`` is modified.
    idx_a, idx_b :
        Group indices to merge.
    """
    from yane.core.node import NodeType

    grouper = genome.grouper
    old_n = grouper.n_outputs
    grouper.merge_groups(idx_a, idx_b)
    if grouper.n_outputs >= old_n:
        return  # no actual merge happened

    # Remove the last input node (simplest safe choice)
    if not genome.input_nodes:
        return
    node_to_remove = genome.input_nodes[-1]
    genome.input_nodes.remove(node_to_remove)
    if node_to_remove in genome.nodes:
        genome.nodes.remove(node_to_remove)
    # Purge connections targeting the removed node
    for src in genome.nodes:
        src.connections = [c for c in src.connections if c.target is not node_to_remove]
    genome._invalidate_topology()
