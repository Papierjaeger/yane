"""Hyperparameter search utilities for YANE.

Supports grid search and random search over parameter grids.
Each configuration is run as a separate training session and
results are stored in the RunDatabase.
"""
from __future__ import annotations

import itertools
import random
import time
from typing import Any, Callable

from yane import NeuroEvolution
from yane.util.run_database import RunDatabase


def _product_dict(param_grid: dict[str, list]) -> list[dict]:
    """Expand a dict of lists into a list of dicts (Cartesian product)."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    result = []
    for combo in itertools.product(*values):
        result.append(dict(zip(keys, combo)))
    return result


def hyperparameter_search(
    param_grid: dict[str, list],
    n_seeds: int = 3,
    budget_iterations: int = 500,
    search_type: str = "grid",
    pop_size: int = 100,
    n_inputs: int = 2,
    n_outputs: int = 1,
    db_path: str | None = None,
    progress_callback: Callable | None = None,
) -> list[dict]:
    """Run hyperparameter search and return ranked results.

    Parameters
    ----------
    param_grid : dict[str, list]
        Mapping of parameter name → list of candidate values.
        Supported params: ``"pop_size"``, ``"target_species"``,
        ``"lamarck_steps"``, ``"mutation_rate"``, etc.
    n_seeds : int
        Number of random seeds per configuration.
    budget_iterations : int
        Max iterations per run.
    search_type : str
        ``"grid"`` (exhaustive) or ``"random"`` (sample).
    pop_size, n_inputs, n_outputs :
        Population and network topology settings.
    db_path : str or None
        Path to RunDatabase file.  If None, uses a temporary DB.
    progress_callback : callable or None
        Called with ``(current, total, config, seed)`` after each run.

    Returns
    -------
    list[dict]
        Results sorted by median fitness (descending).  Each dict has
        ``config``, ``seeds``, ``median_fitness``, ``mean_fitness``,
        ``best_fitness``, ``median_iterations``.
    """
    import tempfile
    if db_path is None:
        db_path = tempfile.mktemp(suffix=".db")

    db = RunDatabase(db_path)

    all_configs = _product_dict(param_grid)
    if search_type == "random":
        rng = random.Random(42)
        rng.shuffle(all_configs)

    total = len(all_configs) * n_seeds
    completed = 0
    results: list[dict] = []

    for config in all_configs:
        seed_fitnesses: list[float] = []
        seed_iterations: list[int] = []

        for seed in range(n_seeds):
            ne = NeuroEvolution(seed=seed)
            ne.set_population_size(config.get("pop_size", pop_size))
            ne.configure(
                n_inputs=n_inputs,
                n_outputs=n_outputs,
                max_nodes=config.get("max_nodes", None),
                max_connections=config.get("max_connections", None),
            )
            ne.set_run_database(db_path)
            ne.set_max_iterations(budget_iterations)

            # Apply config params
            if "target_species" in config:
                ne.set_target_species(config["target_species"])
            if "lamarck_steps" in config:
                ne._lamarck_steps = config["lamarck_steps"]
            if "mutation_rate" in config:
                from yane.core.genome import Genome as _G
                _G._STRUCT_FLOOR = config["mutation_rate"]

            def _eval(g):
                return sum(abs(c.weight) for src in g.nodes for c in src.connections)

            try:
                iters = ne.train(_eval)
                best = ne.get_best()
                seed_fitnesses.append(best.fitness)
                seed_iterations.append(iters)
            except Exception:
                seed_fitnesses.append(-float("inf"))
                seed_iterations.append(budget_iterations)

            completed += 1
            if progress_callback:
                progress_callback(completed, total, config, seed)

        sorted_f = sorted(seed_fitnesses, reverse=True)
        n = len(sorted_f)
        median_f = sorted_f[n // 2] if n > 0 else -float("inf")

        results.append({
            "config": config,
            "seeds": seed_fitnesses,
            "median_fitness": round(median_f, 6),
            "mean_fitness": round(sum(seed_fitnesses) / max(n, 1), 6),
            "best_fitness": round(max(seed_fitnesses), 6),
            "median_iterations": int(sorted(seed_iterations)[n // 2]) if n > 0 else 0,
        })

    results.sort(key=lambda r: r["median_fitness"], reverse=True)
    return results
