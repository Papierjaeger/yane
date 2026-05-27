"""Run history loader — reads completed training runs from the logs/ directory.

Each run is stored as a :class:`RunRecord`.  ``RunRecord.load()`` reads the
``fitness_history.csv`` and ``config.json`` files that every ``train()``
invocation writes automatically.  Runs can be grouped by configuration
hash for statistical comparison.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOG_ROOT = _PROJECT_ROOT / "logs"

# Config keys that are stable identifiers for "same experiment".
_HASH_KEYS = (
    "n_inputs", "n_outputs", "pop_size", "max_nodes",
    "max_connections", "n_initial_hidden", "stateful",
)


@dataclass
class RunRecord:
    """A single completed training run loaded from the logs/ directory."""

    name: str           # e.g. "XOR"
    run_dir: Path
    timestamp: str      # e.g. "2026-05-25_14-10-00"
    config: dict        # parsed config.json
    config_hash: str    # md5 over stable config keys for grouping
    iterations: list[int] = field(default_factory=list)
    best_fitness: list[float] = field(default_factory=list)
    mean_fitness: list[float] = field(default_factory=list)
    median_fitness: list[float] = field(default_factory=list)
    iqr_fitness: list[float] = field(default_factory=list)
    species_count: list[int] = field(default_factory=list)

    @property
    def final_best(self) -> float | None:
        return self.best_fitness[-1] if self.best_fitness else None

    @property
    def total_iterations(self) -> int:
        return self.iterations[-1] if self.iterations else 0

    @property
    def label(self) -> str:
        """Short display label: category + timestamp."""
        return f"{self.name} / {self.timestamp}"

    def to_chart_dict(self) -> dict:
        """Dict accepted by :class:`~yane.gui.canvas.MultiRunChart`."""
        return {
            "name": self.label,
            "iterations": list(self.iterations),
            "best_fitness": list(self.best_fitness),
        }

    def to_csv_rows(self) -> list[dict]:
        """All rows as list-of-dicts (for CSV export)."""
        rows = []
        for i, it in enumerate(self.iterations):
            rows.append({
                "iteration": it,
                "best_fitness": self.best_fitness[i] if i < len(self.best_fitness) else "",
                "mean_fitness": self.mean_fitness[i] if i < len(self.mean_fitness) else "",
                "median_fitness": self.median_fitness[i] if i < len(self.median_fitness) else "",
                "iqr_fitness": self.iqr_fitness[i] if i < len(self.iqr_fitness) else "",
                "species_count": self.species_count[i] if i < len(self.species_count) else "",
            })
        return rows

    @classmethod
    def load(cls, run_dir: Path) -> RunRecord | None:
        """Load a run from its timestamped directory.  Returns ``None`` on failure."""
        csv_path = run_dir / "fitness_history.csv"
        if not csv_path.exists():
            return None

        config: dict = {}
        config_path = run_dir / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        config_hash = _hash_config(config)
        name = run_dir.parent.name
        timestamp = run_dir.name

        record = cls(
            name=name,
            run_dir=run_dir,
            timestamp=timestamp,
            config=config,
            config_hash=config_hash,
        )
        try:
            with csv_path.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        record.iterations.append(int(row["iteration"]))
                        record.best_fitness.append(float(row.get("best_fitness", 0) or 0))
                        record.mean_fitness.append(float(row.get("mean_fitness", 0) or 0))
                        record.median_fitness.append(float(row.get("median_fitness", 0) or 0))
                        record.iqr_fitness.append(float(row.get("iqr_fitness", 0) or 0))
                        record.species_count.append(int(float(row.get("species_count", 0) or 0)))
                    except (ValueError, KeyError):
                        continue
        except Exception:
            return None

        if not record.iterations:
            return None
        return record

    @classmethod
    def list_runs(
        cls,
        category: str | None = None,
        log_root: Path | None = None,
    ) -> list[RunRecord]:
        """Return all available runs, newest first.

        Args:
            category: Restrict to a specific category (e.g. ``"XOR"``).
                ``None`` lists all categories.
            log_root: Override the default log root directory.
        """
        root = log_root or _DEFAULT_LOG_ROOT
        if not root.exists():
            return []

        runs: list[RunRecord] = []
        if category:
            cat_dir = root / category
            dirs = [cat_dir] if cat_dir.is_dir() else []
        else:
            dirs = [d for d in root.iterdir() if d.is_dir()]

        for cat_dir in sorted(dirs):
            for run_dir in sorted(cat_dir.iterdir(), reverse=True):
                if not run_dir.is_dir():
                    continue
                rec = cls.load(run_dir)
                if rec is not None:
                    runs.append(rec)
        return runs

    @classmethod
    def list_categories(cls, log_root: Path | None = None) -> list[str]:
        """Return sorted list of available category names."""
        root = log_root or _DEFAULT_LOG_ROOT
        if not root.exists():
            return []
        return sorted(d.name for d in root.iterdir() if d.is_dir())

    @classmethod
    def group_by_config(cls, runs: list[RunRecord]) -> dict[str, list[RunRecord]]:
        """Group runs by their config hash (i.e., same experiment configuration)."""
        groups: dict[str, list[RunRecord]] = {}
        for run in runs:
            groups.setdefault(run.config_hash, []).append(run)
        return groups


def _hash_config(config: dict) -> str:
    """Stable md5 hash over the experiment-identifying config keys."""
    stable = {k: config.get(k) for k in _HASH_KEYS}
    raw = json.dumps(stable, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()[:8]
