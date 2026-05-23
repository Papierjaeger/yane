"""Checkpoint persistence for YANE training state.

Provides atomic write + validated load of the v1 checkpoint format.
NeuroEvolution builds / restores the payload dict; this module owns the
serialization format (version, required keys, type checks, atomic write).
"""
from __future__ import annotations
from pathlib import Path

VERSION = 1
_REQUIRED_KEYS = frozenset({"config", "population", "tracker"})


def write(path: str | Path, payload: dict) -> None:
    """Atomically pickle payload to path (via .tmp sibling, then replace)."""
    import pickle
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    tmp.replace(path)


def read(path: str | Path) -> dict:
    """Load and validate a checkpoint file; return the payload dict.

    Raises:
        FileNotFoundError: path does not exist.
        ValueError: unrecognised format or missing required keys.
        TypeError: payload fields have unexpected types (corrupted pickle).
    """
    import pickle
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = pickle.loads(path.read_bytes())
    if not isinstance(payload, dict) or payload.get("version") != VERSION:
        raise ValueError(f"Unsupported checkpoint format in {path}")

    missing = _REQUIRED_KEYS - payload.keys()
    if missing:
        raise ValueError(
            f"Checkpoint is missing required keys: {missing}. "
            f"The file may be corrupted or from an older version."
        )

    from yane.evolution.population import Population
    from yane.evolution.innovation import InnovationTracker
    if not isinstance(payload["population"], Population):
        raise TypeError("Checkpoint 'population' is not a Population object")
    if not isinstance(payload["tracker"], (InnovationTracker, type(None))):
        raise TypeError("Checkpoint 'tracker' is not an InnovationTracker")

    return payload
