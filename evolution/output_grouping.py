"""Evolvable Output Synergy Layer for YANE.

Maps *K* internal proto-output nodes to *N* external output channels.
Symmetric counterpart to :mod:`~yane.evolution.input_grouping`.

The genome internally has *K* output nodes ("proto-outputs").  From the
outside, callers always receive *N* values — the :class:`OutputGrouper`
performs the expansion step.

Overview
--------
Each :class:`OutputGroup` represents one proto-output node and specifies
which external output indices it drives, and how (``copy``, ``scale``,
``affine``).

The initial layout is the identity: K = N, each proto-output drives exactly
one external output of the same index.  Mutations change the mapping.

Integration with ``NeuroEvolution``
------------------------------------
1. Call ``yane.set_output_grouping(n_proto=K, n_outputs=N)`` **before**
   ``yane.configure()``.
2. The network topology is built with *K* output nodes (not *N*).
3. During training, ``OutputGrouper.expand(proto)`` maps the network's *K*
   outputs to the *N* values returned from ``genome.forward()``.

Mutation operators
------------------
``OutputGrouper.split_group(group_idx)``
    Split a group's targets in two, creating a new proto-output node.
    Use ``apply_split_to_genome(genome, group_idx)`` to also add the
    corresponding output node to the genome.

``OutputGrouper.merge_groups(idx_a, idx_b)``
    Merge two groups; use ``apply_merge_to_genome`` to remove the
    now-unused output node.

``OutputGrouper.add_output_to_group(ext_idx, group_idx)``
    Route an additional external output to an existing group.

``OutputGrouper.remove_output_from_group(ext_idx, group_idx)``
    Remove an external output from a group (must keep ≥ 1 target).

``OutputGrouper.change_expansion(group_idx, new_exp)``
    Switch the expansion strategy for a group.

``OutputGrouper.create_group(targets, exp)``
    Add a new group driving the specified external outputs.

Crossover
---------
``OutputGrouper.crossover(other)``
    Uniform group-level crossover; different-sized parents are handled
    safely.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome


# ---------------------------------------------------------------------------
# Expansion type
# ---------------------------------------------------------------------------

class ExpType(str, Enum):
    """Expansion strategy for an :class:`OutputGroup`."""
    COPY = "copy"
    SCALE = "scale"
    AFFINE = "affine"


# ---------------------------------------------------------------------------
# OutputGroup
# ---------------------------------------------------------------------------

@dataclass
class OutputGroup:
    """One expansion group: a single proto-output that drives ≥ 1 external outputs."""

    targets: list[int]
    """External output indices driven by this group."""

    expansion: ExpType = ExpType.COPY
    """How the proto-output value is mapped to each target."""

    weights: list[float] = field(default_factory=list)
    """Per-target weights.
    For ``SCALE``: one float per target — ``ext[j] = proto * weights[k]``.
    For ``AFFINE``: two floats per target — ``ext[j] = proto * weights[2k] + weights[2k+1]``.
    Ignored for ``COPY``."""

    enabled: bool = True

    def copy(self) -> OutputGroup:
        return OutputGroup(
            targets=list(self.targets),
            expansion=self.expansion,
            weights=list(self.weights),
            enabled=self.enabled,
        )


# ---------------------------------------------------------------------------
# OutputGrouper
# ---------------------------------------------------------------------------

class OutputGrouper:
    """Evolvable output expansion layer.

    Parameters
    ----------
    n_outputs :
        Number of external output channels (fixed; what callers receive).
    initial_groups :
        Starting group layout.  When *None* each proto-output drives exactly
        one external output of the same index (identity, K = N).
    """

    def __init__(
        self,
        n_outputs: int,
        initial_groups: list[OutputGroup] | None = None,
    ) -> None:
        self.n_outputs = n_outputs
        if initial_groups is not None:
            self.groups: list[OutputGroup] = [g.copy() for g in initial_groups]
        else:
            # Identity: proto[i] → external[i], one-to-one
            self.groups = [OutputGroup(targets=[i]) for i in range(n_outputs)]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_proto(self) -> int:
        """Number of proto-output values consumed by :meth:`expand` (= enabled groups)."""
        return sum(1 for g in self.groups if g.enabled)

    # ------------------------------------------------------------------
    # Core expand
    # ------------------------------------------------------------------

    def expand(self, proto_outputs: list[float]) -> list[float]:
        """Map *K* proto-output values to *N* external output values.

        Proto-output *i* is the *i*-th value in the sequence of enabled groups
        (in group list order).  Disabled groups are skipped.  External outputs
        not targeted by any enabled group default to 0.0.

        Parameters
        ----------
        proto_outputs :
            Values produced by the network's output nodes (length = ``n_proto``).

        Returns
        -------
        list[float]
            Expanded outputs, always of length ``n_outputs``.
        """
        result = [0.0] * self.n_outputs
        proto_idx = 0
        for group in self.groups:
            if not group.enabled:
                continue
            val = proto_outputs[proto_idx] if proto_idx < len(proto_outputs) else 0.0
            for k, target in enumerate(group.targets):
                if not (0 <= target < self.n_outputs):
                    continue
                exp = group.expansion
                if exp == ExpType.SCALE and k < len(group.weights):
                    result[target] = val * group.weights[k]
                elif exp == ExpType.AFFINE and 2 * k + 1 < len(group.weights):
                    result[target] = val * group.weights[2 * k] + group.weights[2 * k + 1]
                else:  # COPY (default and fallback)
                    result[target] = val
            proto_idx += 1
        return result

    # ------------------------------------------------------------------
    # Mutation operators
    # ------------------------------------------------------------------

    def split_group(self, group_idx: int) -> int:
        """Split a group's targets in two, adding a new group.

        The original group retains the first half of its targets; the new
        group receives the remaining half.  If the group has only one target,
        both groups share that single target (duplication).

        Returns the index of the newly created group.
        """
        group = self.groups[group_idx]
        mid = max(1, len(group.targets) // 2)
        first_half = group.targets[:mid]
        second_half = group.targets[mid:] if len(group.targets) > 1 else list(group.targets)
        group.targets = first_half
        new_group = OutputGroup(
            targets=second_half,
            expansion=group.expansion,
            weights=list(group.weights[mid:]) if group.weights else [],
        )
        self.groups.append(new_group)
        return len(self.groups) - 1

    def merge_groups(self, idx_a: int, idx_b: int) -> None:
        """Merge group *idx_b* targets into group *idx_a* and disable *idx_b*.

        Target indices are deduplicated.
        """
        if idx_a == idx_b or idx_a >= len(self.groups) or idx_b >= len(self.groups):
            return
        a = self.groups[idx_a]
        b = self.groups[idx_b]
        combined = list(dict.fromkeys(a.targets + b.targets))
        a.targets = combined
        b.enabled = False

    def create_group(
        self,
        targets: list[int],
        exp: ExpType = ExpType.COPY,
    ) -> int:
        """Add a new group at the end.  Returns its index."""
        self.groups.append(OutputGroup(targets=list(targets), expansion=exp))
        return len(self.groups) - 1

    def add_output_to_group(self, ext_idx: int, group_idx: int) -> None:
        """Route external output *ext_idx* to group *group_idx*.

        No-op if already targeted.
        """
        if 0 <= group_idx < len(self.groups):
            if ext_idx not in self.groups[group_idx].targets:
                self.groups[group_idx].targets.append(ext_idx)

    def remove_output_from_group(self, ext_idx: int, group_idx: int) -> None:
        """Remove external output *ext_idx* from group *group_idx*.

        No-op if the group would become empty (must keep ≥ 1 target).
        """
        if 0 <= group_idx < len(self.groups):
            g = self.groups[group_idx]
            if ext_idx in g.targets and len(g.targets) > 1:
                g.targets.remove(ext_idx)

    def change_expansion(self, group_idx: int, new_exp: ExpType) -> None:
        """Switch the expansion strategy for group *group_idx*."""
        if 0 <= group_idx < len(self.groups):
            self.groups[group_idx].expansion = new_exp

    # ------------------------------------------------------------------
    # Crossover and copy
    # ------------------------------------------------------------------

    def crossover(self, other: OutputGrouper) -> OutputGrouper:
        """Uniform group-level crossover.

        Matching positions are chosen 50/50 from each parent.  Extra groups
        from the larger parent (``self``) are appended unchanged.  Different
        group counts are handled safely.
        """
        child = OutputGrouper.__new__(OutputGrouper)
        child.n_outputs = self.n_outputs
        child.groups = []
        a_groups = [g for g in self.groups if g.enabled]
        b_groups = [g for g in other.groups if g.enabled]
        shared = min(len(a_groups), len(b_groups))
        for i in range(shared):
            child.groups.append(
                a_groups[i].copy() if random.random() < 0.5 else b_groups[i].copy()
            )
        for g in a_groups[shared:]:
            child.groups.append(g.copy())
        return child

    def copy(self) -> OutputGrouper:
        """Return a deep copy of this grouper."""
        c = OutputGrouper.__new__(OutputGrouper)
        c.n_outputs = self.n_outputs
        c.groups = [g.copy() for g in self.groups]
        return c

    # ------------------------------------------------------------------
    # Python export helper
    # ------------------------------------------------------------------

    def to_python_expand_block(self, out_var_name: str, proto_indices: list[int]) -> str:
        """Generate the Python expand block for :func:`genome_to_python`.

        Parameters
        ----------
        out_var_name :
            Name of the node-value dict (typically ``"v"``).
        proto_indices :
            Node indices of the genome's output nodes in group order.

        Returns
        -------
        str
            Multi-line Python source (indented 4 spaces) that assigns
            ``_ext`` and ends with ``return _ext``.
        """
        lines: list[str] = [f"    _ext = [0.0] * {self.n_outputs}"]
        enabled = [(i, g) for i, g in enumerate(self.groups) if g.enabled]
        for seq_i, (_, group) in enumerate(enabled):
            if seq_i >= len(proto_indices):
                break
            ni = proto_indices[seq_i]
            val_expr = f"{out_var_name}.get({ni}, 0.0)"
            for k, target in enumerate(group.targets):
                if not (0 <= target < self.n_outputs):
                    continue
                exp = group.expansion
                if exp == ExpType.SCALE and k < len(group.weights):
                    expr = f"{val_expr} * {group.weights[k]!r}"
                elif exp == ExpType.AFFINE and 2 * k + 1 < len(group.weights):
                    w = group.weights[2 * k]
                    b = group.weights[2 * k + 1]
                    expr = f"{val_expr} * {w!r} + {b!r}"
                else:
                    expr = val_expr
                lines.append(f"    _ext[{target}] = {expr}")
        lines.append("    return _ext")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Genome-level mutation helpers
# ---------------------------------------------------------------------------

def apply_split_to_genome(genome: "Genome", group_idx: int) -> int:
    """Split a group and add an output node to *genome*.

    After the call, ``genome.out_grouper.n_proto`` is increased by 1 and a
    new ``NodeType.OUTPUT`` node is appended to ``genome.output_nodes`` and
    ``genome.nodes``.

    Returns the index of the newly created group.
    """
    from yane.core.node import Node, NodeType
    from yane.util.activation import ActivationType

    grouper = genome.out_grouper
    new_group_idx = grouper.split_group(group_idx)

    next_innov = max((n.innovation for n in genome.nodes if n.innovation >= 0), default=-1) + 1
    new_node = Node(NodeType.OUTPUT, next_innov)
    new_node.activation = ActivationType.SIGMOID
    genome.output_nodes.append(new_node)
    genome.nodes.append(new_node)
    genome._invalidate_topology()
    return new_group_idx


def apply_merge_to_genome(genome: "Genome", idx_a: int, idx_b: int) -> None:
    """Merge two groups and remove an output node from *genome*.

    The last output node is removed; any connections targeting it are pruned.
    """
    grouper = genome.out_grouper
    old_n = grouper.n_proto
    grouper.merge_groups(idx_a, idx_b)
    if grouper.n_proto >= old_n:
        return

    if not genome.output_nodes:
        return
    node_to_remove = genome.output_nodes[-1]
    genome.output_nodes.remove(node_to_remove)
    if node_to_remove in genome.nodes:
        genome.nodes.remove(node_to_remove)
    for src in genome.nodes:
        src.connections = [c for c in src.connections if c.target is not node_to_remove]
    genome._invalidate_topology()
