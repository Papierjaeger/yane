"""Multi-population island model for improved exploration.

Maintains N independent Population instances; periodically migrates the
best genomes between randomly paired islands.
"""
from __future__ import annotations

import random
from typing import Any

from yane.core.genome import Genome
from yane.evolution.population import Population


class IslandModel:
    """Manages multiple populations ("islands") with periodic migration.

    Parameters
    ----------
    n_islands : int
        Number of independent populations.
    island_kwargs : list[dict]
        List of keyword-argument dicts, one per island, passed to
        ``Population.__init__``.  If shorter than *n_islands*, the last
        entry is repeated.
    migration_interval : int
        How many spawn cycles between migrations.
    migration_count : int
        Number of genomes to migrate each time (from each island).
    rng_seed : int or None
        Seed for the island-level RNG (independent of population RNGs).
    """

    def __init__(
        self,
        n_islands: int,
        island_kwargs: list[dict] | None = None,
        migration_interval: int = 5,
        migration_count: int = 3,
        rng_seed: int | None = None,
    ) -> None:
        if n_islands < 1:
            raise ValueError("Need at least 1 island")
        self.n_islands = n_islands
        self.migration_interval = max(1, migration_interval)
        self.migration_count = max(1, migration_count)
        self._rng = random.Random(rng_seed)

        self.islands: list[Population] = []
        self._spawn_count: int = 0
        self._migration_events: list[dict] = []

        kwargs_list = island_kwargs or [{}]
        for i in range(n_islands):
            kw = kwargs_list[min(i, len(kwargs_list) - 1)]
            from yane.core.genome import Genome as _G
            from yane.evolution.innovation import InnovationTracker
            pop = Population(
                max_size=kw.get("max_size", 100),
                initial_genome=kw.get("initial_genome", _G()),
                tracker=kw.get("tracker", InnovationTracker()),
                target_species=kw.get("target_species", 5),
            )
            # Copy over relevant attributes
            for attr in (
                "_adaptive_pop_enabled", "_adaptive_pop_min", "_adaptive_pop_max",
                "_adaptive_pop_rate", "_crossover_enabled", "_speciation_enabled",
                "_novelty_enabled", "_diversity_injection_enabled",
                "_compat_threshold", "_weight_blend_alpha",
            ):
                if attr in kw:
                    setattr(pop, attr, kw[attr])
            self.islands.append(pop)

    def tick(self) -> list[dict]:
        """Check if migration should happen and migrate if so.

        Returns the list of migration events from this tick.
        """
        self._spawn_count += 1
        events: list[dict] = []
        if self._spawn_count % self.migration_interval == 0 and self.n_islands >= 2:
            events = self._migrate()
            self._migration_events.extend(events)
        return events

    def _migrate(self) -> list[dict]:
        """Migrate the best genomes between randomly paired islands."""
        events: list[dict] = []
        indices = list(range(self.n_islands))
        self._rng.shuffle(indices)

        for i in range(0, len(indices) - 1, 2):
            src_idx = indices[i]
            dst_idx = indices[i + 1]
            src = self.islands[src_idx]
            dst = self.islands[dst_idx]

            # Get top genomes from source
            migrants = src.get_top(self.migration_count)
            if not migrants:
                continue

            # Replace worst genomes in destination
            dst_evald = sorted(
                dst._evaluated, key=lambda g: g.fitness, reverse=True
            )
            replaced = 0
            for migrant in migrants:
                if dst_evald and migrant.fitness > dst_evald[-1].fitness:
                    # Remove worst
                    worst = dst_evald.pop()
                    dst._evaluated.remove(worst)
                    # Add migrant (fresh copy)
                    migrant_copy = migrant.copy()
                    dst._evaluated.append(migrant_copy)
                    replaced += 1
                else:
                    # Put in unevaluated for next round
                    dst._unevaluated.append(migrant.copy())

            if replaced > 0:
                events.append({
                    "generation": self._spawn_count,
                    "src": src_idx,
                    "dst": dst_idx,
                    "n_migrated": replaced,
                    "avg_fitness_src": sum(
                        g.fitness for g in src._evaluated[-self.migration_count:]
                    ) / max(self.migration_count, 1),
                    "avg_fitness_dst": sum(
                        g.fitness for g in dst._evaluated[:self.migration_count]
                    ) / max(self.migration_count, 1) if dst._evaluated else 0.0,
                })

        return events

    def get_best_across_all(self) -> Genome | None:
        """Return the single best genome across all islands."""
        best = None
        best_f = -float("inf")
        for pop in self.islands:
            try:
                g = pop.get_best()
                if g and g.fitness > best_f:
                    best = g
                    best_f = g.fitness
            except RuntimeError:
                pass
        return best

    def get_diagnostics(self) -> dict:
        """Return diagnostics for all islands."""
        return {
            "n_islands": self.n_islands,
            "migration_interval": self.migration_interval,
            "migration_count": self.migration_count,
            "total_migrations": len(self._migration_events),
            "last_migration_events": self._migration_events[-5:],
            "island_sizes": [len(p._evaluated) + len(p._unevaluated) for p in self.islands],
            "island_best_fitness": [
                p.get_best().fitness if p._evaluated else None
                for p in self.islands
            ],
            "island_species": [p.species_count for p in self.islands],
            "island_stagnation": [p.stagnation_count for p in self.islands],
        }
