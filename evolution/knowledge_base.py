"""Cross-Run Knowledge Base — Phase 4 of P0 Meta-Adaptive Orchestration.

Stores problem profiles and the parameters / fitness achieved in past runs.
When asked for suggestions, it finds similar past runs via weighted k-NN and
returns a ranked list of parameter configurations to try.

Usage::

    # After configure() + train():
    kb = KnowledgeBase("my_kb.json")
    kb.learn(profile, yane.get_param_space(), best_fitness)

    # Before the next run:
    suggestions = kb.suggest(profile, top_k=5)
    best = suggestions[0]
    print(best["params"], best["confidence"])

Cold start: when the KB is empty (or no nearby entry is found), returns
conservative defaults based on the detected task type.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from yane.evolution.problem_profiler import ProblemProfile


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_TYPES = ["classification", "regression", "rl_discrete", "rl_continuous"]

# Feature vector index
# [0-3]  task-type one-hot
# [4]    log-normalised n_inputs
# [5]    log-normalised n_outputs
# [6]    estimated_difficulty
# [7]    noise_level (capped 0-1)
# [8]    temporal_dependency (capped 0-1)
# [9]    log-normalised state_dim_effective
_N_FEATURES = 10

# Higher weight → more influence on the k-NN distance
DEFAULT_FEATURE_WEIGHTS: list[float] = [
    1.5,  # classification one-hot
    1.5,  # regression one-hot
    1.5,  # rl_discrete one-hot
    1.5,  # rl_continuous one-hot
    0.5,  # n_inputs
    0.5,  # n_outputs
    0.8,  # estimated_difficulty
    1.0,  # noise_level
    1.0,  # temporal_dependency
    0.3,  # state_dim_effective
]

# Conservative parameter defaults by task type (used when KB is cold)
COLD_START_DEFAULTS: dict[str, dict] = {
    "classification": {
        "lamarck.mode": "hill_climbing",
        "lamarck.n_steps": 3,
        "pop.size": 50,
        "anytime.enabled": False,
        "recovery.enabled": False,
    },
    "regression": {
        "lamarck.mode": "hill_climbing",
        "lamarck.n_steps": 5,
        "pop.size": 100,
        "anytime.enabled": False,
        "recovery.enabled": False,
    },
    "rl_discrete": {
        "lamarck.mode": "hill_climbing",
        "lamarck.n_steps": 5,
        "pop.size": 100,
        "anytime.enabled": True,
        "anytime.promotion_frac": 0.3,
        "recovery.enabled": True,
        "recovery.cooldown": 20,
    },
    "rl_continuous": {
        "lamarck.mode": "cma_es",
        "lamarck.n_steps": 5,
        "pop.size": 150,
        "anytime.enabled": True,
        "anytime.promotion_frac": 0.3,
        "recovery.enabled": True,
        "recovery.cooldown": 20,
    },
}


# ---------------------------------------------------------------------------
# KBEntry
# ---------------------------------------------------------------------------

@dataclass
class KBEntry:
    """A single knowledge-base record.

    ``profile_vec`` is the feature vector used for k-NN distance computation.
    ``profile_dict`` is a serialisable copy of the original ``ProblemProfile``
    fields (for inspection and re-vectorisation after schema changes).
    """

    profile_vec: list[float]
    profile_dict: dict
    final_params: dict           # param_name → final value
    final_fitness: float
    timestamp: str
    run_id: str | None = None
    yane_version: str = "unknown"


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """Cross-run parameter knowledge base backed by a JSON file.

    Grows with every ``learn()`` call.  ``suggest()`` returns ranked
    parameter suggestions for the next run based on weighted k-NN.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        feature_weights: list[float] | None = None,
    ) -> None:
        self._path: Path | None = Path(path) if path else None
        self._entries: list[KBEntry] = []
        self._feature_weights: list[float] = list(
            feature_weights or DEFAULT_FEATURE_WEIGHTS
        )
        if self._path and self._path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(
        self,
        profile: "ProblemProfile",
        final_params: dict,
        final_fitness: float,
        *,
        run_id: str | None = None,
        yane_version: str = "unknown",
        auto_save: bool = True,
    ) -> None:
        """Record a completed run and persist (if a path is configured).

        Parameters
        ----------
        profile:       Problem profile returned by ``profile_problem()``.
        final_params:  Dict of param_name → final value (e.g. from
                       ``get_param_space()``, or any custom dict).
        final_fitness: Best fitness achieved during the run.
        run_id:        Optional run UUID for traceability.
        auto_save:     Write to disk immediately (default True).
        """
        entry = KBEntry(
            profile_vec=profile_to_vector(profile),
            profile_dict=_profile_to_dict(profile),
            final_params=dict(final_params),
            final_fitness=float(final_fitness),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            run_id=run_id,
            yane_version=str(yane_version),
        )
        self._entries.append(entry)
        if auto_save and self._path is not None:
            self.save()

    # ------------------------------------------------------------------
    # Suggestion
    # ------------------------------------------------------------------

    def suggest(
        self,
        profile: "ProblemProfile",
        top_k: int = 5,
    ) -> list[dict]:
        """Return ranked parameter suggestions for the given profile.

        Each suggestion is a dict with:
        - ``"params"``: param-name → value mapping
        - ``"expected_fitness"``: float or None
        - ``"confidence"``: float in [0, 1]
        - ``"distance"``: float (lower = more similar; ∞ for cold-start)
        - ``"source"``: ``"kb"`` or ``"cold_start"``
        - ``"run_id"``: str or None

        If the KB is empty, returns ``[cold_start_suggestion(task_type)]``.
        If the best match has very low confidence, the cold-start suggestion
        is prepended.
        """
        if not self._entries:
            return [cold_start_suggestion(profile.task_type)]

        query_vec = profile_to_vector(profile)
        ranked = self._rank(query_vec)
        k = min(top_k, len(ranked))
        top = ranked[:k]
        top_distances = [d for d, _ in top]

        # Filter params of the best entry by cross-neighbour agreement so that
        # params with high variance across similar runs (i.e. context-specific
        # details like recovery.cooldown) are not blindly applied.
        _filtered = _param_agreement(top)

        result: list[dict] = []
        for i, (dist, entry) in enumerate(top):
            conf = _confidence(dist, top_distances)
            result.append({
                # Best suggestion gets filtered params; lower-ranked entries
                # keep their raw params (used for diagnostics / fallback only).
                "params": _filtered if i == 0 else dict(entry.final_params),
                "expected_fitness": entry.final_fitness,
                "confidence": round(conf, 4),
                "distance": round(dist, 6),
                "source": "kb",
                "run_id": entry.run_id,
            })

        # Prepend cold start if the KB's best match has low confidence
        if result and result[0]["confidence"] < 0.25:
            result = [cold_start_suggestion(profile.task_type)] + result

        return result

    def best_suggestion(self, profile: "ProblemProfile", top_k: int = 5) -> dict:
        """Return the single highest-confidence suggestion."""
        suggestions = self.suggest(profile, top_k=top_k)
        return suggestions[0] if suggestions else cold_start_suggestion(profile.task_type)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> None:
        """Write KB to a JSON file.

        Parameters
        ----------
        path: Target file.  Uses ``self._path`` if omitted.
        """
        target = Path(path) if path else self._path
        if target is None:
            raise ValueError("No path configured; pass path= to save()")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "feature_weights": self._feature_weights,
            "entries": [_entry_to_dict(e) for e in self._entries],
        }
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def export_json(self, path: str | Path) -> None:
        """Export the KB to a portable JSON file (alias for ``save(path)``)."""
        self.save(path)

    def import_json(
        self,
        path: str | Path,
        *,
        merge: bool = True,
    ) -> None:
        """Load entries from a JSON file.

        Parameters
        ----------
        merge: If True (default), append to existing entries.
               If False, replace all existing entries.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        loaded = [_entry_from_dict(e) for e in raw.get("entries", [])]
        if merge:
            self._entries.extend(loaded)
        else:
            self._entries = loaded
            if "feature_weights" in raw:
                self._feature_weights = list(raw["feature_weights"])

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        """Remove all entries from memory (does not delete the file)."""
        self._entries.clear()

    def get_diagnostics(self) -> dict:
        return {
            "n_entries": len(self._entries),
            "feature_weights": list(self._feature_weights),
            "path": str(self._path) if self._path else None,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rank(self, query_vec: list[float]) -> list[tuple[float, KBEntry]]:
        """Return all entries sorted by distance to query_vec (ascending)."""
        ranked = [
            (_weighted_distance(query_vec, e.profile_vec, self._feature_weights), e)
            for e in self._entries
        ]
        ranked.sort(key=lambda x: x[0])
        return ranked

    def _load(self) -> None:
        try:
            self.import_json(self._path, merge=False)  # type: ignore[arg-type]
        except Exception:
            pass  # start fresh if file is empty or corrupt


# ---------------------------------------------------------------------------
# Public helpers (exposed so tests can call them directly)
# ---------------------------------------------------------------------------

def profile_to_vector(profile: "ProblemProfile") -> list[float]:
    """Convert a ``ProblemProfile`` to a ``_N_FEATURES``-element float vector.

    The vector is suitable for weighted Euclidean distance computation.
    All components are normalised to approximately [0, 1].
    """
    _LOG_MAX = math.log1p(1000)
    one_hot = [1.0 if profile.task_type == t else 0.0 for t in TASK_TYPES]
    numeric = [
        math.log1p(max(0, profile.n_inputs)) / _LOG_MAX,
        math.log1p(max(0, profile.n_outputs)) / _LOG_MAX,
        max(0.0, min(1.0, float(profile.estimated_difficulty))),
        max(0.0, min(1.0, float(profile.noise_level))),
        max(0.0, min(1.0, float(profile.temporal_dependency))),
        math.log1p(max(0, profile.state_dim_effective)) / _LOG_MAX,
    ]
    return one_hot + numeric


def cold_start_suggestion(task_type: str) -> dict:
    """Return a cold-start suggestion dict for the given task type."""
    defaults = dict(
        COLD_START_DEFAULTS.get(task_type, COLD_START_DEFAULTS["rl_continuous"])
    )
    return {
        "params": defaults,
        "expected_fitness": None,
        "confidence": 0.0,
        "distance": float("inf"),
        "source": "cold_start",
        "run_id": None,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _weighted_distance(
    v1: list[float],
    v2: list[float],
    weights: list[float],
) -> float:
    """Weighted Euclidean distance between two feature vectors."""
    if len(v1) != len(v2) or len(v1) != len(weights):
        # Pad or truncate gracefully
        n = min(len(v1), len(v2), len(weights))
        v1, v2, weights = v1[:n], v2[:n], weights[:n]
    return math.sqrt(
        sum(w * (a - b) ** 2 for w, a, b in zip(weights, v1, v2))
    )


def _confidence(
    distance: float,
    top_k_distances: list[float],
) -> float:
    """Confidence in [0, 1].

    Increases when:
    - The query is close to this entry (low distance relative to the spread).
    - Many of the top-K are within 1.5× of this entry's distance (neighbourhood
      agreement — genuinely similar runs).
    """
    max_d = max(top_k_distances) if top_k_distances else 1.0
    if max_d < 1e-10:
        return 1.0

    dist_score = max(0.0, 1.0 - distance / (max_d + 1e-10))

    # Neighbourhood-agreement: how many top-K entries agree with this one?
    # "Agree" = within 1.5× of THIS entry's distance (or absolute threshold)
    threshold = max(distance * 1.5, 0.05)
    agreeing = sum(1 for d in top_k_distances if d <= threshold)
    agreement = agreeing / max(1, len(top_k_distances))

    return min(1.0, 0.7 * dist_score + 0.3 * agreement)



def _param_agreement(
    top_entries: "list[tuple[float, KBEntry]]",
    min_agreement: float = 0.5,
) -> dict:
    """Return filtered params from the best entry, keeping only those where
    the top-k neighbours substantially agree on the value.

    For **categorical / boolean** params: agreement = fraction of neighbours
    that have exactly the same value as the best entry.

    For **numeric** params: agreement = 1 − CV (coefficient of variation,
    std/|mean|), capped to [0, 1].  High CV = neighbours disagree on the
    magnitude → exclude the param.

    Returns the full best-entry dict unchanged when fewer than 3 neighbours
    are available (variance estimate is unreliable with tiny samples).
    """
    if not top_entries:
        return {}
    best_params = dict(top_entries[0][1].final_params)
    if len(top_entries) < 3:
        return best_params  # too few neighbours for meaningful variance

    all_params = [e.final_params for _, e in top_entries]
    filtered: dict = {}

    for name, best_val in best_params.items():
        neighbor_vals = [p[name] for p in all_params if name in p]
        if len(neighbor_vals) < 2:
            filtered[name] = best_val  # unique to best entry — include it
            continue

        if isinstance(best_val, (str, bool)):
            agreement = sum(1 for v in neighbor_vals if v == best_val) / len(neighbor_vals)
        else:
            try:
                floats = [float(v) for v in neighbor_vals]
                mean = sum(floats) / len(floats)
                std = (sum((v - mean) ** 2 for v in floats) / len(floats)) ** 0.5
                cv = std / (abs(mean) + 1e-10)
                agreement = max(0.0, 1.0 - cv)
            except (TypeError, ValueError):
                agreement = 0.0  # non-numeric, non-categorical → skip

        if agreement >= min_agreement:
            filtered[name] = best_val

    return filtered


def _profile_to_dict(profile: "ProblemProfile") -> dict:
    return {
        "task_type": profile.task_type,
        "task_type_confidence": getattr(profile, "task_type_confidence", 1.0),
        "n_inputs": profile.n_inputs,
        "n_outputs": profile.n_outputs,
        "estimated_difficulty": profile.estimated_difficulty,
        "noise_level": profile.noise_level,
        "reward_sparsity": profile.reward_sparsity,
        "temporal_dependency": profile.temporal_dependency,
        "state_dim_effective": profile.state_dim_effective,
        "fitness_mean": getattr(profile, "fitness_mean", 0.0),
        "fitness_std": getattr(profile, "fitness_std", 0.0),
    }


def _entry_to_dict(e: KBEntry) -> dict:
    return {
        "profile_vec": e.profile_vec,
        "profile_dict": e.profile_dict,
        "final_params": e.final_params,
        "final_fitness": e.final_fitness,
        "timestamp": e.timestamp,
        "run_id": e.run_id,
        "yane_version": e.yane_version,
    }


def _entry_from_dict(d: dict) -> KBEntry:
    return KBEntry(
        profile_vec=d.get("profile_vec", []),
        profile_dict=d.get("profile_dict", {}),
        final_params=d.get("final_params", {}),
        final_fitness=float(d.get("final_fitness", 0.0)),
        timestamp=d.get("timestamp", ""),
        run_id=d.get("run_id"),
        yane_version=d.get("yane_version", "unknown"),
    )
