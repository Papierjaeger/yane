"""Austauschbare Genome-Codecs für Checkpoint-Serialisierung.

Protokoll::

    class MyCodec:
        name = "mycodec"
        def encode(self, genome: Genome) -> bytes: ...
        def decode(self, data: bytes) -> Genome: ...

Eingebaute Codecs: PickleCodec, JsonCodec.
"""
from __future__ import annotations

import io
import json
from typing import Protocol, runtime_checkable

from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class GenomeCodec(Protocol):
    """Protocol for pluggable genome serialization."""

    name: str

    def encode(self, genome: Genome) -> bytes:
        """Serialize *genome* to bytes."""
        ...

    def decode(self, data: bytes) -> Genome:
        """Deserialize bytes to a ``Genome`` instance."""
        ...


# ---------------------------------------------------------------------------
# PickleCodec — default, supports all genome types
# ---------------------------------------------------------------------------

class PickleCodec:
    """Standard pickle-based codec (default, supports all genome features)."""
    name = "pickle"

    def encode(self, genome: Genome) -> bytes:
        import pickle
        return pickle.dumps(genome)

    def decode(self, data: bytes) -> Genome:
        import pickle
        return pickle.loads(data)


# ---------------------------------------------------------------------------
# JsonCodec — human-readable, supports basic genomes only
# ---------------------------------------------------------------------------

class JsonCodec:
    """JSON-based codec (human-readable, basic genomes only).

    Limitations:
    - Only supports ActivationType built-in activations
    - Strategy genes (mutation rates) are stored as plain dicts
    - Memory/gate-node features may lose gate references
    """
    name = "json"

    def encode(self, genome: Genome) -> bytes:
        nodes = []
        for n in genome.nodes:
            act = n.activation
            act_str = act.value if isinstance(act, ActivationType) else str(act)
            nodes.append({
                "type": n.type.value,
                "innovation": n.innovation,
                "bias": round(n.bias, 12),
                "activation": act_str,
                "persist_value": n.persist_value,
                "max_triggers": n.max_triggers,
                "input_index": n.input_index,
                "input_scale": round(n.input_scale, 12),
                "output_scale": round(n.output_scale, 12),
                "leak_alpha": round(n.leak_alpha, 12),
                "memory_gate": round(n.memory_gate, 12),
                "connections": [
                    {
                        "target_innovation": c.target.innovation,
                        "weight": round(c.weight, 12),
                        "innovation": c.innovation,
                        "enabled": c.enabled,
                        "mutation": {
                            "shift_rate": float(c.mutation.shift_rate),
                            "custom_rate": float(c.mutation.custom_rate),
                        },
                    }
                    for c in n.connections
                ],
            })
        data = {
            "codec": self.name,
            "max_nodes": genome.max_nodes,
            "max_connections": genome.max_connections,
            "nodes": nodes,
            "strategy": {
                attr: _mutation_to_dict(getattr(genome, attr, None))
                for attr in (
                    "sigma_global", "mutation_add_node", "mutation_remove_node",
                    "mutation_add_connection", "mutation_remove_connection",
                    "mutation_rewire", "mutation_disable_connection",
                    "mutation_enable_connection",
                )
            },
        }
        return json.dumps(data, indent=2).encode("utf-8")

    def decode(self, data: bytes) -> Genome:
        raw = json.loads(data.decode("utf-8"))
        genome = Genome()
        genome.max_nodes = raw.get("max_nodes")
        genome.max_connections = raw.get("max_connections")

        # Rebuild nodes
        node_map: dict[int, Node] = {}
        for nd in raw["nodes"]:
            ntype = NodeType(nd["type"])
            node = Node(ntype, nd["innovation"])
            node.bias = nd["bias"]
            try:
                node.activation = ActivationType(nd["activation"])
            except ValueError:
                node.activation = nd["activation"]
            node.max_triggers = nd["max_triggers"]
            node.input_index = nd["input_index"]
            node.input_scale = nd["input_scale"]
            node.output_scale = nd["output_scale"]
            node.leak_alpha = nd["leak_alpha"]
            node.memory_gate = nd["memory_gate"]
            node.persist_value = nd["persist_value"]
            node_map[nd["innovation"]] = node
            genome.nodes.append(node)
            if ntype == NodeType.INPUT:
                genome.input_nodes.append(node)
            elif ntype == NodeType.OUTPUT:
                genome.output_nodes.append(node)

        # Rebuild connections
        for nd in raw["nodes"]:
            src = node_map[nd["innovation"]]
            for cd in nd.get("connections", []):
                tgt = node_map.get(cd["target_innovation"])
                if tgt is None:
                    continue
                conn = Connection(tgt, innovation=cd["innovation"])
                conn.weight = cd["weight"]
                conn.enabled = cd["enabled"]
                mut_dict = cd.get("mutation", {})
                conn.mutation.shift_rate = mut_dict.get("shift_rate", conn.mutation.shift_rate)
                conn.mutation.custom_rate = mut_dict.get("custom_rate", conn.mutation.custom_rate)
                src.connections.append(conn)

        # Restore strategy genes (Mutation objects)
        from yane.evolution.mutation import Mutation
        strat = raw.get("strategy", {})
        for key, val in strat.items():
            if val is not None:
                if isinstance(val, dict):
                    m = Mutation()
                    m.shift_rate = val.get("shift_rate", m.shift_rate)
                    m.custom_rate = val.get("custom_rate", m.custom_rate)
                    m.int_rate = val.get("int_rate", m.int_rate)
                    m.rate_mutation_rate = val.get("rate_mutation_rate", m.rate_mutation_rate)
                    setattr(genome, key, m)
                elif isinstance(val, (int, float)):
                    setattr(genome, key, val)

        genome._invalidate_topology()
        return genome


# ---------------------------------------------------------------------------
# Registry & helpers
# ---------------------------------------------------------------------------

_CODECS: dict[str, GenomeCodec] = {}


def register_codec(codec: GenomeCodec) -> None:
    """Register a custom codec by its ``.name``."""
    _CODECS[codec.name] = codec


def get_codec(name: str) -> GenomeCodec:
    """Return a registered codec by name (falls back to built-in)."""
    if name in _CODECS:
        return _CODECS[name]
    for cls in (PickleCodec, JsonCodec):
        inst = cls()
        if inst.name == name:
            return inst
    raise KeyError(f"No codec registered: {name!r}")


def detect_codec(data: bytes) -> str:
    """Peek at serialized data to detect which codec was used.

    Returns ``"pickle"``, ``"json"``, or raises ``ValueError`` if
    the format is unrecognised.
    """
    stripped = data.lstrip()
    if stripped.startswith(b"{"):
        try:
            header = json.loads(stripped[:4096].decode("utf-8"))
            return header.get("codec", "json")
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    # Pickle protocol markers: b'\x80' (protocol 2+), b'(' (protocol 0), b'\x00' (protocol 1)
    if stripped[:1] in (b"\x80", b"(", b"\x00", b"\x94"):
        return "pickle"
    raise ValueError("Unrecognised codec format")


def _mutation_to_dict(val):
    """Serialize a Mutation or scalar to a JSON-safe value."""
    from yane.evolution.mutation import Mutation
    if isinstance(val, Mutation):
        return {
            "shift_rate": val.shift_rate,
            "custom_rate": val.custom_rate,
            "int_rate": val.int_rate,
            "rate_mutation_rate": val.rate_mutation_rate,
        }
    return val


# Register built-ins
register_codec(PickleCodec())
register_codec(JsonCodec())
