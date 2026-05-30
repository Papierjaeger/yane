"""Genome-Phylogenie: Stammbaum der Innovationen.

Tracks the evolutionary history of genomes across generations, recording
parent-child relationships, innovation origins, and fitness deltas.  Provides
tree analysis (ancestors, descendants, MRCA, depth) and export functions
(JSON dict, Graphviz DOT).

The tree uses the **primary parent** model (same as InnovationTracker) — a
genome with two parents (crossover) is linked to the fitter (primary) parent
only, giving a proper tree rather than a DAG.

Usage::

    yane.enable_phylogeny()
    result = yane.train(evaluator)
    tree = yane.get_phylogeny()
    print(tree.to_dot())
    import json; print(json.dumps(tree.to_dict(), indent=2))

Directly::

    from yane.evolution.phylogeny import PhylogenyTree

    tree = PhylogenyTree()
    tree.record(genome_id=1, parent_id=None, fitness=0.5, generation=0, innovations=[])
    tree.record(genome_id=2, parent_id=1,    fitness=0.8, generation=1, innovations=[10])
    print(tree.ancestry(2))   # [1]
    print(tree.depth(2))      # 1
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class PhylogenyNode:
    """One node in the phylogenetic tree — a snapshot of a genome at recording time.

    Attributes
    ----------
    genome_id:
        Unique genome ID (``genome._genome_id``).
    parent_id:
        Primary parent ID, or ``None`` for root genomes.
    fitness:
        Fitness at the time of recording.
    generation:
        Training generation when the genome was recorded.
    innovations:
        Innovation numbers **first introduced** by this genome (new connections
        or node splits not present in the parent).
    fitness_delta:
        ``fitness - parent_fitness`` (0.0 for root genomes).
    """
    genome_id: int
    parent_id: int | None
    fitness: float
    generation: int
    innovations: list[int] = field(default_factory=list)
    fitness_delta: float = 0.0


class PhylogenyTree:
    """Phylogenetic tree of evolved genomes.

    Disabled by default (zero runtime cost); call :meth:`enable` to activate
    recording.  Once enabled, call :meth:`record` after each genome evaluation.

    Parameters
    ----------
    max_size:
        Maximum number of nodes to keep.  Oldest root-less nodes are dropped
        when the limit is exceeded (``None`` = unlimited).
    """

    def __init__(self, max_size: int | None = None) -> None:
        self._enabled: bool = False
        self._nodes: dict[int, PhylogenyNode] = {}
        self._children: dict[int, list[int]] = {}  # parent_id → [child_ids]
        self._max_size = max_size

    # ------------------------------------------------------------------
    # Enable / disable
    # ------------------------------------------------------------------

    def enable(self) -> None:
        """Activate phylogeny recording."""
        self._enabled = True

    def disable(self) -> None:
        """Deactivate recording (existing data is retained)."""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        genome_id: int,
        parent_id: int | None,
        fitness: float,
        generation: int,
        innovations: "list[int] | None" = None,
    ) -> None:
        """Register a genome in the tree.

        If the genome has already been recorded (same ``genome_id``), the
        existing record is updated with the new fitness (in case it improves
        over time) but the parent and innovations are not changed.

        Parameters
        ----------
        genome_id:
            Unique genome ID.
        parent_id:
            Primary parent, or ``None`` for the initial genome.
        fitness:
            Current fitness.
        generation:
            Current training generation.
        innovations:
            New innovation numbers introduced by this genome.
        """
        if not self._enabled:
            return
        if innovations is None:
            innovations = []
        if genome_id in self._nodes:
            # Update fitness only
            self._nodes[genome_id].fitness = fitness
            nd = self._nodes[genome_id]
            if nd.parent_id is not None and nd.parent_id in self._nodes:
                nd.fitness_delta = fitness - self._nodes[nd.parent_id].fitness
            return

        if parent_id is not None and parent_id in self._nodes:
            fitness_delta = fitness - self._nodes[parent_id].fitness
        else:
            fitness_delta = 0.0  # root node — no parent to compare against

        node = PhylogenyNode(
            genome_id=genome_id,
            parent_id=parent_id,
            fitness=fitness,
            generation=generation,
            innovations=list(innovations),
            fitness_delta=fitness_delta,
        )
        self._nodes[genome_id] = node
        if parent_id is not None:
            self._children.setdefault(parent_id, []).append(genome_id)

        if self._max_size is not None and len(self._nodes) > self._max_size:
            self._trim()

    def _trim(self) -> None:
        """Remove the oldest node (smallest genome_id) without affecting descendants.

        Children of the removed node become new roots (their parent_id remains
        set but no longer resolves to a recorded node).
        """
        if not self._nodes:
            return
        oldest = min(self._nodes.keys())
        self._nodes.pop(oldest, None)
        # Remove from parent's children list (if parent is still recorded)
        nd_parent = None  # oldest root has no parent
        for nd in self._nodes.values():
            pass  # already removed; children keep their parent_id (now dangling)
        # Remove from children index of its own parent (not needed — parent was also oldest or gone)
        # Update the children index: remove oldest from any parent's child list
        for parent_children in self._children.values():
            if oldest in parent_children:
                parent_children.remove(oldest)
                break
        self._children.pop(oldest, None)

    def _remove_subtree(self, genome_id: int) -> None:
        """Remove a genome and all its descendants."""
        children = list(self._children.get(genome_id, []))
        for child_id in children:
            self._remove_subtree(child_id)
        self._nodes.pop(genome_id, None)
        self._children.pop(genome_id, None)

    # ------------------------------------------------------------------
    # Tree analysis
    # ------------------------------------------------------------------

    def ancestry(self, genome_id: int, max_depth: int = 1000) -> list[int]:
        """Return the ancestor chain (oldest first) for *genome_id*.

        Parameters
        ----------
        genome_id:
            Starting genome.
        max_depth:
            Maximum hops to follow (prevents infinite loops in corrupted data).

        Returns
        -------
        list[int]
            Parent IDs from the oldest ancestor to the direct parent.
            Empty list if *genome_id* is unknown or has no parent.
        """
        chain: list[int] = []
        current = genome_id
        seen: set[int] = set()
        for _ in range(max_depth):
            nd = self._nodes.get(current)
            if nd is None or nd.parent_id is None:
                break
            if nd.parent_id in seen:
                break  # cycle guard
            chain.append(nd.parent_id)
            seen.add(nd.parent_id)
            current = nd.parent_id
        chain.reverse()
        return chain

    def descendants(self, genome_id: int) -> list[int]:
        """Return all descendant genome IDs (BFS order)."""
        result: list[int] = []
        queue = list(self._children.get(genome_id, []))
        while queue:
            gid = queue.pop(0)
            result.append(gid)
            queue.extend(self._children.get(gid, []))
        return result

    def depth(self, genome_id: int) -> int:
        """Return the depth of *genome_id* in the tree (root = 0)."""
        return len(self.ancestry(genome_id))

    def root_ids(self) -> list[int]:
        """Return genome IDs with no recorded parent (tree roots)."""
        return [gid for gid, nd in self._nodes.items() if nd.parent_id is None]

    def mrca(self, genome_id_a: int, genome_id_b: int) -> int | None:
        """Return the Most Recent Common Ancestor of two genomes.

        Returns ``None`` if no common ancestor exists in the recorded data.
        """
        ancestors_a = set(self.ancestry(genome_id_a))
        ancestors_a.add(genome_id_a)
        # Walk b's ancestry until we hit an ancestor of a
        current = genome_id_b
        for _ in range(10000):
            if current in ancestors_a:
                return current
            nd = self._nodes.get(current)
            if nd is None or nd.parent_id is None:
                break
            current = nd.parent_id
        return None

    def innovation_attribution(self, genome_id: int) -> dict[int, float]:
        """Return fitness delta attributed per innovation for *genome_id*.

        Maps each innovation number introduced by *genome_id* to the
        genome's fitness delta (child_fitness - parent_fitness).  Innovations
        share the delta equally.

        Returns ``{}`` if *genome_id* is unknown or has no innovations.
        """
        nd = self._nodes.get(genome_id)
        if nd is None or not nd.innovations:
            return {}
        per_innovation = nd.fitness_delta / len(nd.innovations)
        return {innov: per_innovation for innov in nd.innovations}

    # ------------------------------------------------------------------
    # Properties / diagnostics
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of recorded genomes."""
        return len(self._nodes)

    def get_node(self, genome_id: int) -> PhylogenyNode | None:
        """Return the PhylogenyNode for *genome_id*, or ``None``."""
        return self._nodes.get(genome_id)

    def best_fitness_in_lineage(self, genome_id: int) -> float:
        """Return the highest fitness seen along the ancestry chain."""
        best = self._nodes[genome_id].fitness if genome_id in self._nodes else 0.0
        for anc_id in self.ancestry(genome_id):
            nd = self._nodes.get(anc_id)
            if nd is not None:
                best = max(best, nd.fitness)
        return best

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dictionary of the full tree.

        Structure::

            {
                "nodes": {
                    "<genome_id>": {
                        "parent_id": <int|null>,
                        "fitness": <float>,
                        "fitness_delta": <float>,
                        "generation": <int>,
                        "innovations": [<int>, ...],
                        "depth": <int>
                    },
                    ...
                },
                "roots": [<genome_id>, ...],
                "size": <int>
            }
        """
        nodes: dict[str, dict] = {}
        for gid, nd in self._nodes.items():
            nodes[str(gid)] = {
                "parent_id": nd.parent_id,
                "fitness": nd.fitness,
                "fitness_delta": nd.fitness_delta,
                "generation": nd.generation,
                "innovations": nd.innovations,
                "depth": self.depth(gid),
            }
        return {
            "nodes": nodes,
            "roots": self.root_ids(),
            "size": self.size,
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Return the tree as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_dot(
        self,
        label_fn: "callable | None" = None,
        max_nodes: int = 200,
    ) -> str:
        """Return a Graphviz DOT representation of the tree.

        Parameters
        ----------
        label_fn:
            Optional ``(PhylogenyNode) -> str`` for custom node labels.
            Default: ``"ID\\nf=<fitness>\\ngen=<gen>"``.
        max_nodes:
            Maximum number of nodes to include (oldest roots dropped first
            when limit exceeded).

        Returns
        -------
        str
            Valid DOT source suitable for ``dot -Tpng tree.dot -o tree.png``.
        """
        nodes = list(self._nodes.values())
        if len(nodes) > max_nodes:
            # Keep the most recent (highest genome_id)
            nodes = sorted(nodes, key=lambda n: n.genome_id)[-max_nodes:]
        visible_ids = {n.genome_id for n in nodes}

        def _label(nd: PhylogenyNode) -> str:
            if label_fn is not None:
                return label_fn(nd)
            return f"{nd.genome_id}\\nf={nd.fitness:.3f}\\ng={nd.generation}"

        lines = ["digraph phylogeny {", '    rankdir="TB";',
                 '    node [shape=box fontname="Helvetica" fontsize=9];']
        for nd in nodes:
            lbl = _label(nd).replace('"', '\\"')
            color = "lightblue" if nd.parent_id is None else "white"
            lines.append(f'    n{nd.genome_id} [label="{lbl}" style=filled fillcolor="{color}"];')
        for nd in nodes:
            if nd.parent_id is not None and nd.parent_id in visible_ids:
                delta_str = f"{nd.fitness_delta:+.3f}"
                lines.append(
                    f'    n{nd.parent_id} -> n{nd.genome_id} [label="{delta_str}"];'
                )
        lines.append("}")
        return "\n".join(lines)
