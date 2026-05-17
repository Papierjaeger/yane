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
    Skips silently if genome.max_nodes is set and already reached.
    """
    if genome.max_nodes is not None and len(genome.nodes) >= genome.max_nodes:
        return

    connections = genome.all_connections()

    if connections:
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
    else:
        new_node = Node(NodeType.HIDDEN,
                        innovation=tracker.next() if tracker is not None else -1)

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
    fan_in = sum(1 for n in genome.nodes for c in n.connections if c.target is target)
    fan_in = max(fan_in, 1)
    std = math.sqrt(2.0 / (fan_in + 1))  # Xavier/He init
    conn = Connection(target, innovation=innov)
    conn.weight = random.gauss(0.0, std)
    source.connections.append(conn)
    genome._invalidate_topology()


def remove_connection(genome, tracker=None) -> None:
    """Remove a random connection from the network."""
    connections = [(node, i) for node in genome.nodes for i in range(len(node.connections))]

    if connections:
        node, idx = random.choice(connections)
        node.connections.pop(idx)
        genome._invalidate_topology()
