"""Experiment Tracking — Run / Experiment / RunDatabase.

A lightweight SQLite-backed database for recording training runs.

Usage::

    ne = NeuroEvolution()
    ne.set_run_database("runs.db")
    ne.experiment("my-xor")
    ne.configure(2, 1)
    ne.set_max_iterations(1000)
    ne.train(fitness_fn)          # run is auto-saved

    db = ne.get_run_database()
    for run in db.list_runs():
        print(run.run_id, run.final_best)

    # Reproduce a past run (config + seed; fitness_fn must be re-supplied):
    ne2 = db.reproduce_run(run.run_id)
    ne2.configure(2, 1)
    ne2.train(fitness_fn)
"""
from __future__ import annotations

import csv
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane import NeuroEvolution


@dataclass
class Run:
    """A single completed (or in-progress) training run."""

    run_id: str
    experiment_id: str | None
    name: str
    seed: int | None
    config: dict
    fitness_history: list[dict]
    diagnostics: dict
    start_time: str
    end_time: str | None
    stop_reason: str | None
    artifacts: dict = field(default_factory=dict)

    @property
    def final_best(self) -> float | None:
        if not self.fitness_history:
            return None
        return self.fitness_history[-1].get("best_fitness")

    @property
    def total_iterations(self) -> int:
        if not self.fitness_history:
            return 0
        return self.fitness_history[-1].get("iteration", 0)


@dataclass
class Experiment:
    """A named group of runs."""

    experiment_id: str
    name: str
    tags: list[str]
    created_at: str
    runs: list[Run] = field(default_factory=list)


class RunDatabase:
    """SQLite-backed run tracking database.

    Create once via ``NeuroEvolution.set_run_database(path)`` and all
    subsequent ``train()`` calls will automatically record runs.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self._path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._init_schema()

    # ── Schema ─────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name          TEXT NOT NULL,
                tags_json     TEXT NOT NULL DEFAULT '[]',
                created_at    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                run_id               TEXT PRIMARY KEY,
                experiment_id        TEXT,
                name                 TEXT NOT NULL,
                seed                 INTEGER,
                config_json          TEXT NOT NULL DEFAULT '{}',
                fitness_history_json TEXT NOT NULL DEFAULT '[]',
                diagnostics_json     TEXT NOT NULL DEFAULT '{}',
                artifacts_json       TEXT NOT NULL DEFAULT '{}',
                start_time           TEXT NOT NULL,
                end_time             TEXT,
                stop_reason          TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            );
            CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs (experiment_id);
            CREATE INDEX IF NOT EXISTS idx_runs_name       ON runs (name);
        """)
        # Migrate existing databases — add columns that post-date the initial schema.
        existing = {
            row[1]
            for row in self._db.execute("PRAGMA table_info(runs)").fetchall()
        }
        if "artifacts_json" not in existing:
            self._db.execute(
                "ALTER TABLE runs ADD COLUMN artifacts_json TEXT NOT NULL DEFAULT '{}'"
            )
        if "problem_profile_json" not in existing:
            self._db.execute(
                "ALTER TABLE runs ADD COLUMN problem_profile_json TEXT"
            )
        self._db.commit()

    # ── Experiment management ───────────────────────────────────────────────

    def create_experiment(
        self,
        name: str,
        tags: list[str] | None = None,
    ) -> Experiment:
        exp_id = str(uuid.uuid4())
        now = _now()
        self._db.execute(
            "INSERT INTO experiments (experiment_id, name, tags_json, created_at)"
            " VALUES (?, ?, ?, ?)",
            (exp_id, name, json.dumps(tags or []), now),
        )
        self._db.commit()
        return Experiment(
            experiment_id=exp_id,
            name=name,
            tags=tags or [],
            created_at=now,
        )

    def get_or_create_experiment(
        self,
        name: str,
        tags: list[str] | None = None,
    ) -> Experiment:
        row = self._db.execute(
            "SELECT * FROM experiments WHERE name = ? ORDER BY created_at DESC LIMIT 1",
            (name,),
        ).fetchone()
        if row:
            return Experiment(
                experiment_id=row["experiment_id"],
                name=row["name"],
                tags=json.loads(row["tags_json"]),
                created_at=row["created_at"],
            )
        return self.create_experiment(name, tags)

    def list_experiments(self) -> list[Experiment]:
        rows = self._db.execute(
            "SELECT * FROM experiments ORDER BY created_at DESC"
        ).fetchall()
        return [
            Experiment(
                experiment_id=r["experiment_id"],
                name=r["name"],
                tags=json.loads(r["tags_json"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # ── Run lifecycle ──────────────────────────────────────────────────────

    def start_run(
        self,
        run_id: str,
        name: str,
        seed: int | None,
        config: dict,
        experiment_id: str | None = None,
    ) -> None:
        """Insert a pending run row at the start of training."""
        self._db.execute(
            "INSERT INTO runs"
            " (run_id, experiment_id, name, seed, config_json, start_time)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, experiment_id, name, seed, json.dumps(config), _now()),
        )
        self._db.commit()

    def finish_run(
        self,
        run_id: str,
        fitness_history: list[dict],
        diagnostics: dict,
        stop_reason: str | None,
        artifacts: dict | None = None,
        problem_profile: dict | None = None,
    ) -> None:
        """Update the run row when training finishes."""
        self._db.execute(
            "UPDATE runs"
            " SET fitness_history_json = ?, diagnostics_json = ?,"
            "     artifacts_json = ?, end_time = ?, stop_reason = ?,"
            "     problem_profile_json = ?"
            " WHERE run_id = ?",
            (
                json.dumps(fitness_history),
                json.dumps(diagnostics),
                json.dumps(artifacts or {}),
                _now(),
                stop_reason,
                json.dumps(problem_profile) if problem_profile else None,
                run_id,
            ),
        )
        self._db.commit()

    # ── Queries ────────────────────────────────────────────────────────────

    def load_run(self, run_id: str) -> Run | None:
        row = self._db.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return self._row_to_run(row) if row else None

    def list_runs(
        self,
        experiment_id: str | None = None,
        name: str | None = None,
    ) -> list[Run]:
        if experiment_id is not None:
            rows = self._db.execute(
                "SELECT * FROM runs WHERE experiment_id = ? ORDER BY start_time DESC",
                (experiment_id,),
            ).fetchall()
        elif name is not None:
            rows = self._db.execute(
                "SELECT * FROM runs WHERE name = ? ORDER BY start_time DESC",
                (name,),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM runs ORDER BY start_time DESC"
            ).fetchall()
        return [self._row_to_run(r) for r in rows]

    def compare_runs(self, run_ids: list[str]) -> dict:
        """Return summary stats for a set of run IDs."""
        runs = [r for r in (self.load_run(rid) for rid in run_ids) if r is not None]
        results = []
        for run in runs:
            bests = [
                r["best_fitness"]
                for r in run.fitness_history
                if r.get("best_fitness") is not None
            ]
            results.append({
                "run_id": run.run_id,
                "name": run.name,
                "seed": run.seed,
                "final_best": run.final_best,
                "total_iterations": run.total_iterations,
                "stop_reason": run.stop_reason,
                "mean_best": sum(bests) / len(bests) if bests else None,
            })
        return {"runs": results, "count": len(results)}

    # ── Reproduce ─────────────────────────────────────────────────────────

    def reproduce_run(self, run_id: str) -> "NeuroEvolution":
        """Return a NeuroEvolution instance configured from a saved run.

        The fitness function is not serialised — the caller must pass it to
        ``train()`` as usual.  Call ``ne.configure(n_inputs, n_outputs)``
        after this to rebuild the population.
        """
        run = self.load_run(run_id)
        if run is None:
            raise KeyError(f"Run {run_id!r} not found in database")
        from yane import NeuroEvolution
        ne = NeuroEvolution(seed=run.seed)
        ne._load_config_dict(run.config)
        return ne

    # ── Housekeeping ──────────────────────────────────────────────────────

    def close(self) -> None:
        self._db.close()

    # ── Internal ─────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> Run:
        return Run(
            run_id=row["run_id"],
            experiment_id=row["experiment_id"],
            name=row["name"],
            seed=row["seed"],
            config=json.loads(row["config_json"]),
            fitness_history=json.loads(row["fitness_history_json"]),
            diagnostics=json.loads(row["diagnostics_json"]),
            artifacts=json.loads(row["artifacts_json"] or "{}"),
            start_time=row["start_time"],
            end_time=row["end_time"],
            stop_reason=row["stop_reason"],
        )


def load_fitness_history_from_csv(csv_path: Path) -> list[dict]:
    """Read the training CSV written by the logger and return a list of dicts."""
    if not csv_path.exists():
        return []
    rows: list[dict] = []
    try:
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry: dict = {}
                for key, val in row.items():
                    if val == "" or val is None:
                        continue
                    try:
                        entry[key] = int(val) if "." not in val else float(val)
                    except (ValueError, TypeError):
                        entry[key] = val
                if entry:
                    rows.append(entry)
    except Exception:
        pass
    return rows


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
