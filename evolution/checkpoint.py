"""Checkpoint persistence for YANE training state.

Provides atomic write + validated load of the checkpoint format.
NeuroEvolution builds / restores the payload dict; this module owns the
serialization format (version, required keys, type checks, atomic write).

## Why Pickle?

The checkpoint uses Python Pickle for the main ``.pkl`` file and JSON only for
the human-readable metadata sidecar (``.pkl.json``).  Pickle is chosen for the
payload because:

- ``Genome`` and ``Population`` contain complex object graphs (nodes, connections,
  species, mutation parameters) with circular references.  Hand-rolling a
  lossless JSON representation would require re-implementing all of that and
  keeping it in sync with every future structural change.
- Python pickle handles ``__getstate__``/``__setstate__`` for objects whose
  compiled closures (``_compiled_forward``) cannot be pickled, stripping them
  automatically.  A JSON serializer would need the same escape hatch.
- The alternative (numpy-binary + JSON for weights) would work for simple
  feed-forward networks but breaks down for recurrent, memory-node, or
  QD-enabled genomes where the object graph carries state that isn't weights.

Security boundary: only load ``.pkl`` files from sources you trust.  Never
unpickle bytes received over a network connection without prior signature
verification — an attacker-controlled pickle can execute arbitrary code.
The JSON sidecar is safe to read from untrusted paths because it is parsed as
data, not executed.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import base64
import hashlib
import json

VERSION = 2
_REQUIRED_KEYS = frozenset({"config", "population", "tracker"})


def _config_hash(cfg: dict) -> str:
    """Deterministic SHA-256 hash of a config dict."""
    canonical = json.dumps(cfg, indent=2, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def check_compatibility(stored_hash: str, current_cfg: dict) -> str:
    """Compare a stored config hash with the current config.

    Returns ``"EXACT"`` when hashes match, ``"COMPATIBLE"`` otherwise.
    Breaking-change detection (comparing individual keys) is not yet
    implemented; callers should treat ``"COMPATIBLE"`` as *unknown*.
    """
    if stored_hash == _config_hash(current_cfg):
        return "EXACT"
    return "COMPATIBLE"


def _metadata_for(payload: dict) -> dict:
    cfg = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
    pop = payload.get("population")
    return {
        "version": payload.get("version", VERSION),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_hash": _config_hash(cfg),
        "config": cfg,
        "population_size": getattr(pop, "max_size", None),
        "evaluated": len(getattr(pop, "_evaluated", ())) if pop is not None else None,
        "unevaluated": len(getattr(pop, "_unevaluated", ())) if pop is not None else None,
        "requires_reattach": [
            "quality_diversity_descriptor"
        ] if getattr(pop, "_qd_enabled", False) and getattr(pop, "_qd_descriptor_fn", None) is None else [],
    }


def _migrate_v1(payload: dict) -> dict:
    payload = dict(payload)
    payload["version"] = VERSION
    payload.setdefault("normalizer", None)
    payload.setdefault("lamarck_n_applied", 0)
    payload.setdefault("lamarck_n_steps_total", 0)
    payload.setdefault("lamarck_time_ms", 0.0)
    payload.setdefault("lamarck_n_blocked_top_k", 0)
    payload.setdefault("n_invalid_fitness", 0)
    payload.setdefault("n_clipped_fitness", 0)
    payload.setdefault("n_early_stopped", 0)
    payload.setdefault("early_stopping_n", None)
    return payload


_MIGRATIONS = {1: _migrate_v1}


def migrate(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Unsupported checkpoint format")
    version = payload.get("version")
    if version == VERSION:
        return payload
    while version in _MIGRATIONS:
        payload = _MIGRATIONS[version](payload)
        version = payload.get("version")
    if version != VERSION:
        raise ValueError(f"Unsupported checkpoint version: {version}")
    return payload


def write(path: str | Path, payload: dict, codec: str = "pickle") -> None:
    """Atomically write payload to path (via .tmp sibling, then replace)."""
    import pickle
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["version"] = VERSION
    tmp = path.with_suffix(path.suffix + ".tmp")
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    if codec == "pickle":
        tmp.write_bytes(raw)
    elif codec == "json":
        wrapper = {
            "codec": "json",
            "payload_encoding": "pickle+base64",
            "version": VERSION,
            "payload": base64.b64encode(raw).decode("ascii"),
        }
        tmp.write_text(json.dumps(wrapper, indent=2), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported checkpoint codec: {codec!r}")
    tmp.replace(path)
    meta_path = path.with_suffix(path.suffix + ".json")
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    meta_tmp.write_text(json.dumps(_metadata_for(payload), indent=2), encoding="utf-8")
    meta_tmp.replace(meta_path)


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

    data = path.read_bytes()
    if data.lstrip().startswith(b"{"):
        wrapper = json.loads(data.decode("utf-8"))
        if wrapper.get("payload_encoding") != "pickle+base64":
            raise ValueError("Unsupported JSON checkpoint payload encoding")
        data = base64.b64decode(wrapper["payload"])

    payload = migrate(pickle.loads(data))

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
