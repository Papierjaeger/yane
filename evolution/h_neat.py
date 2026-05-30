"""Hierarchical NEAT (H-NEAT) — Mehrstufige Policy-Architektur.

Zweistufiges System:
  Manager-Genom  →  wählt Sub-Policy aus dem Worker-Pool
  Worker-Genome  →  führen die gewählte Policy aus

**Architektur:**
```
inputs
  ↓
Manager.forward(inputs)  →  [w0, w1, ..., wK-1]   (K Gewichte)
  ↓ softmax
Selektion (hard: argmax ODER soft: gewichtete Summe)
  ↓
output = worker_i.forward(inputs)   (hard)
       = Σ softmax(w_i) * worker_i.forward(inputs)   (soft)
```

**Selektion:**
- ``"hard"``: ``argmax(softmax(manager_outputs))`` wählt einen Worker.
  Für unterschiedliche Eingaben wählt der Manager unterschiedliche Worker
  → verhaltensbasierte Spezialisierung.
- ``"soft"``: gewichtete Summe aller Worker-Ausgaben.
  Ermöglicht glatte Interpolation zwischen Policies.

**Evolvierbarkeit:**
- Manager und Workers werden separat oder gemeinsam via NEAT evolviert.
- Mutations-Operatoren: `add_sub_policy`, `split_sub_policy`, `merge_sub_policies`.
- Vollständige Hierarchie ist pickle-/checkpoint-fähig.

Verwendung::

    h = HierarchicalGenome(manager, workers, selection_mode="hard")
    outputs = h.forward(inputs)
    h.add_sub_policy(new_worker)
    h.save("hier.pkl")
    h2 = HierarchicalGenome.load("hier.pkl")
"""
from __future__ import annotations

import math
import pickle
import random
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome


# ---------------------------------------------------------------------------
# Softmax helper
# ---------------------------------------------------------------------------

def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    total = sum(exps)
    return [e / total for e in exps] if total > 0 else [1.0 / len(values)] * len(values)


# ---------------------------------------------------------------------------
# HierarchicalGenome
# ---------------------------------------------------------------------------

class HierarchicalGenome:
    """Two-level hierarchical policy: one manager + N worker genomes.

    The manager receives the raw inputs and outputs N selection weights.
    Workers are the actual policies that generate action outputs.

    Parameters
    ----------
    manager :
        A NEAT genome whose output count equals the initial number of workers.
        Its ``n_outputs`` should match ``len(workers)``.
    workers :
        List of worker genomes (sub-policies).  All must share the same
        ``n_inputs`` and ``n_outputs`` as each other.
    selection_mode :
        ``"hard"`` — pick the worker with the highest softmax weight
        (argmax selection, state-dependent specialization).
        ``"soft"`` — weighted sum of all workers (smooth interpolation).
    """

    def __init__(
        self,
        manager: "Genome",
        workers: list["Genome"],
        selection_mode: str = "hard",
    ) -> None:
        if selection_mode not in ("hard", "soft"):
            raise ValueError(f"Unknown selection_mode: {selection_mode!r}")
        self.manager = manager
        self.workers = list(workers)
        self.selection_mode = selection_mode
        # Track which worker was last selected (for diagnostics)
        self.last_selected_idx: int = 0

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, inputs: list[float]) -> list[float]:
        """Execute the hierarchical policy.

        1. Manager selects (or weights) workers.
        2. Selected worker(s) process the inputs and return outputs.

        Parameters
        ----------
        inputs :
            Raw environment inputs (same count as worker ``n_inputs``).

        Returns
        -------
        list[float]
            Worker output vector (same length as worker ``n_outputs``).
        """
        if not self.workers:
            return []

        # Manager: select which worker(s) to use
        self.manager.reset()
        manager_out = self.manager.forward(inputs)

        # Pad or truncate to match number of workers
        n = len(self.workers)
        weights = list(manager_out[:n]) + [0.0] * max(0, n - len(manager_out))
        sw = _softmax(weights)

        if self.selection_mode == "hard":
            self.last_selected_idx = sw.index(max(sw))
            worker = self.workers[self.last_selected_idx]
            worker.reset()
            return worker.forward(inputs)
        else:
            # Soft: weighted sum of all worker outputs
            results: list[list[float]] = []
            for w in self.workers:
                w.reset()
                results.append(w.forward(inputs))
            if not results:
                return []
            n_out = len(results[0])
            out = [0.0] * n_out
            for i, (wgt, res) in enumerate(zip(sw, results)):
                for j in range(min(len(res), n_out)):
                    out[j] += wgt * res[j]
            return out

    def reset(self) -> None:
        """Reset manager and all worker states."""
        self.manager.reset()
        for w in self.workers:
            w.reset()

    # ------------------------------------------------------------------
    # Pool mutation operators
    # ------------------------------------------------------------------

    def add_sub_policy(self, worker: "Genome") -> None:
        """Add a new worker to the pool.

        Note: The manager's output count should match the new pool size.
        Callers are responsible for re-configuring the manager if needed.
        """
        self.workers.append(worker)

    def split_sub_policy(self, idx: int, rng: random.Random | None = None) -> None:
        """Split worker at *idx* into two slightly different workers.

        The original is kept; a mutated copy is inserted after it.
        """
        if not (0 <= idx < len(self.workers)):
            raise IndexError(f"Worker index {idx} out of range (pool size={len(self.workers)})")
        original = self.workers[idx]
        child = original.copy()
        # Slightly perturb the copy
        _rng = rng or random
        for src in child.nodes:
            for conn in src.connections:
                conn.weight += _rng.gauss(0.0, 0.1)
            src.bias += _rng.gauss(0.0, 0.05)
        child._invalidate_topology()
        self.workers.insert(idx + 1, child)

    def merge_sub_policies(self, idx_a: int, idx_b: int) -> None:
        """Merge two workers by blending their weights, remove *idx_b*.

        The merged policy replaces *idx_a*; *idx_b* is removed.
        Weight of merged = average of both parents' weights.
        """
        if idx_a == idx_b:
            return
        if not (0 <= idx_a < len(self.workers) and 0 <= idx_b < len(self.workers)):
            raise IndexError("Worker indices out of range")
        a = self.workers[idx_a]
        b = self.workers[idx_b]
        # Blend weights of matching connections
        conns_a = {conn.innovation: conn for src in a.nodes for conn in src.connections}
        conns_b = {conn.innovation: conn for src in b.nodes for conn in src.connections}
        for innov, conn_a in conns_a.items():
            if innov in conns_b:
                conn_a.weight = (conn_a.weight + conns_b[innov].weight) / 2.0
        a._invalidate_topology()
        # Remove b
        self.workers.pop(idx_b)

    # ------------------------------------------------------------------
    # Serialization (Checkpoint)
    # ------------------------------------------------------------------

    def save(self, path: "str | Path") -> None:
        """Save the complete hierarchy to a pickle file."""
        data = {
            "manager": self.manager,
            "workers": self.workers,
            "selection_mode": self.selection_mode,
        }
        with open(str(path), "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: "str | Path") -> "HierarchicalGenome":
        """Load a hierarchy from a checkpoint file."""
        with open(str(path), "rb") as f:
            data = pickle.load(f)
        return cls(
            manager=data["manager"],
            workers=data["workers"],
            selection_mode=data["selection_mode"],
        )

    def copy(self) -> "HierarchicalGenome":
        """Return a deep copy of this hierarchy."""
        return HierarchicalGenome(
            manager=self.manager.copy(),
            workers=[w.copy() for w in self.workers],
            selection_mode=self.selection_mode,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def n_workers(self) -> int:
        return len(self.workers)

    @property
    def worker_n_inputs(self) -> int:
        return len(self.workers[0].input_nodes) if self.workers else 0

    @property
    def worker_n_outputs(self) -> int:
        return len(self.workers[0].output_nodes) if self.workers else 0

    def selection_distribution(self, inputs_list: list[list[float]]) -> list[float]:
        """Return frequency each worker is selected over *inputs_list* (hard mode only)."""
        counts = [0] * len(self.workers)
        for inputs in inputs_list:
            self.manager.reset()
            manager_out = self.manager.forward(inputs)
            n = len(self.workers)
            weights = list(manager_out[:n]) + [0.0] * max(0, n - len(manager_out))
            sw = _softmax(weights)
            counts[sw.index(max(sw))] += 1
        total = max(1, sum(counts))
        return [c / total for c in counts]
