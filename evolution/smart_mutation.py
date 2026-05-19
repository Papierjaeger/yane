"""Smart structural mutations that preserve or smoothly transition network behavior."""
from __future__ import annotations
import math
import random

from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType


def add_node(genome, tracker=None) -> None:
    """Insert a new hidden node on a random existing connection.

    A → B (weight w)  becomes  A → N → B
    A→N keeps weight w; N→B gets weight 1.0 with LINEAR activation.
    Net output stays identical at the moment of insertion.
    Skips silently if genome.max_nodes is set and already reached,
    or if the genome has no enabled connections to split.
    """
    if genome.max_nodes is not None and len(genome.nodes) >= genome.max_nodes:
        return

    connections = [(n, c) for n, c in genome.all_connections() if c.enabled]
    if not connections:
        return  # nothing to split; adding an isolated node has no effect

    source, conn = random.choice(connections)
    old_target = conn.target

    if tracker is not None:
        node_innov, conn_in_innov, conn_out_innov = tracker.get_split(conn.innovation)
    else:
        node_innov = conn_in_innov = conn_out_innov = -1

    new_node = Node(NodeType.HIDDEN, innovation=node_innov)
    new_node.activation = ActivationType.LINEAR
    new_node.bias = 0.0

    conn.target = new_node
    conn.innovation = conn_in_innov if tracker is not None else conn.innovation

    bypass = Connection(old_target, innovation=conn_out_innov)
    bypass.weight = 1.0
    new_node.connections.append(bypass)

    genome.nodes.append(new_node)
    genome._invalidate_topology()


def remove_node(genome, tracker=None) -> None:
    """Remove a random hidden node.

    For each A→N→B pair, a bypass connection A→B is created with probability
    genome.bypass_connection_prob. Bypass weight = w_AN * w_NB.
    """
    hidden = [n for n in genome.nodes if n.type == NodeType.HIDDEN]
    if not hidden:
        return

    node_to_remove = random.choice(hidden)

    incoming = [(node, conn) for node, conn in genome.all_connections()
                if conn.target is node_to_remove]

    # Snapshot connections before iterating — if source_node IS node_to_remove
    # (self-loop), appending to source_node.connections would grow the list we're
    # iterating, causing an infinite loop that consumes all memory.
    outgoing_snapshot = list(node_to_remove.connections)

    for source_node, in_conn in incoming:
        for out_conn in outgoing_snapshot:
            if random.random() < genome.bypass_connection_prob:
                innov = (tracker.get_connection(source_node.innovation, out_conn.target.innovation)
                         if tracker is not None else -1)
                bypass = Connection(out_conn.target, innovation=innov)
                bypass.weight = in_conn.weight * out_conn.weight
                source_node.connections.append(bypass)

    for node in genome.nodes:
        node.connections = [c for c in node.connections if c.target is not node_to_remove]

    genome.nodes.remove(node_to_remove)
    genome._invalidate_topology()


def add_connection(genome, tracker=None) -> None:
    """Add a random connection between any two nodes (cycles allowed).

    Skips silently if genome.max_connections is set and already reached.
    """
    if len(genome.nodes) < 2:
        return

    if genome.max_connections is not None and genome.connection_count >= genome.max_connections:
        return

    source = random.choice(genome.nodes)
    target = random.choice(genome.nodes)

    for conn in source.connections:
        if conn.target is target:
            return

    innov = tracker.get_connection(source.innovation, target.innovation) if tracker is not None else -1

    # Symmetric Gaussian weight init (σ=0.3).
    #
    # The previous bimodal ±He init (std ≈ 1.0) forced connections to have
    # large |weight|, which immediately disrupts any well-tuned constant-output
    # network.  The genome lost fitness when a connection was added → evolution
    # pruned all connections → population converged to the constant-output basin.
    #
    # σ=0.3 is a measured compromise: small enough that the new connection rarely
    # makes the genome much worse (preserving existing fitness), yet large enough
    # to occasionally provide a useful directional signal for selection.
    # Multi-seed benchmark: avg fitness -5.7 vs -8.6 for bimodal, -8.9 for σ=0.1.
    weight = random.gauss(0.0, 0.3)

    conn = Connection(target, innovation=innov)
    conn.weight = weight
    source.connections.append(conn)
    genome._invalidate_topology()


def remove_connection(genome, tracker=None) -> None:
    """Remove a connection, preferring low-|weight| connections.

    Soft-min sampling: probability ∝ 1 / (|weight| + ε) so structurally
    less important connections (small weights) are removed more often.
    Only enabled connections are candidates — disabled ones are left intact.
    """
    candidates = [(node, i, conn)
                  for node in genome.nodes
                  for i, conn in enumerate(node.connections)
                  if conn.enabled]
    if not candidates:
        return
    probs = [1.0 / (abs(c[2].weight) + 1e-6) for c in candidates]
    node, idx, _ = random.choices(candidates, weights=probs, k=1)[0]
    node.connections.pop(idx)
    genome._invalidate_topology()


def rewire_connection(genome, tracker=None) -> None:
    """Atomically replace a low-weight connection with a new random one.

    Removes the weakest enabled connection (soft-min sampling from the bottom
    quartile) and immediately adds a new random connection.  Topology stays
    roughly the same size while the search space is explored.
    """
    candidates = [(node, i, conn)
                  for node in genome.nodes
                  for i, conn in enumerate(node.connections)
                  if conn.enabled]
    if not candidates:
        return
    candidates.sort(key=lambda x: abs(x[2].weight))
    n_pool = max(1, len(candidates) // 4)
    node, idx, _ = random.choice(candidates[:n_pool])
    node.connections.pop(idx)
    genome._invalidate_topology()
    add_connection(genome, tracker)


def disable_connection(genome) -> None:
    """Disable a connection, preferring low-|weight| connections.

    The connection stays in the genome (reversible) but is excluded from
    the forward pass.  Soft-min sampling biases toward small weights.
    """
    candidates = [(node, conn)
                  for node in genome.nodes
                  for conn in node.connections
                  if conn.enabled]
    if not candidates:
        return
    probs = [1.0 / (abs(c[1].weight) + 1e-6) for c in candidates]
    _, conn = random.choices(candidates, weights=probs, k=1)[0]
    conn.enabled = False
    genome._invalidate_topology()


def enable_connection(genome) -> None:
    """Re-enable a randomly chosen disabled connection."""
    candidates = [(node, conn)
                  for node in genome.nodes
                  for conn in node.connections
                  if not conn.enabled]
    if not candidates:
        return
    _, conn = random.choice(candidates)
    conn.enabled = True
    genome._invalidate_topology()
