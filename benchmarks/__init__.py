"""YANE benchmark suite package."""
from __future__ import annotations

from pathlib import Path

BENCHMARK_DB_PATH: Path = Path(__file__).parent / "benchmark_runs.db"


def wire_db(ne, experiment: str, db_path: Path | None = None) -> None:
    """Attach the shared benchmark RunDatabase to *ne* and set the experiment name."""
    from yane.util.run_database import RunDatabase  # noqa: F401 (import check)
    path = db_path or BENCHMARK_DB_PATH
    ne.set_run_database(str(path))
    ne.experiment(experiment)
