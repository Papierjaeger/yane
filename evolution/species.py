"""Species: a group of structurally similar genomes competing internally."""
from __future__ import annotations
from operator import attrgetter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome

_fitness_key = attrgetter('fitness')

_species_counter: int = 0


def _next_species_id() -> int:
    global _species_counter
    _species_counter += 1
    return _species_counter


class Species:
    """A group of structurally similar genomes that compete internally."""

    def __init__(
        self,
        representative: Genome,
        spawn_count: int = 0,
        parent_id: int | None = None,
    ) -> None:
        self.species_id: int = _next_species_id()
        self.representative = representative
        self.members: list[Genome] = []
        self.best_fitness_seen: float = float('-inf')
        self.stagnation_count: int = 0   # spawn-cycles without fitness improvement
        self._cached_best: Genome | None = None  # maintained by _assign_species()
        # Per-species Lamarck diagnostics (cumulative).
        self.lamarck_n_applied: int = 0
        self.lamarck_n_steps_total: int = 0
        # Per-species mutation success tracking.
        self._mut_success: dict[str, int] = {}
        self._mut_total: dict[str, int] = {}
        # Mutation biases: multiplier per mutation type, defaults to 1.0.
        self.mutation_biases: dict[str, float] = {
            "add_node": 1.0, "remove_node": 1.0,
            "add_connection": 1.0, "remove_connection": 1.0,
            "rewire": 1.0, "disable": 1.0, "enable": 1.0,
        }
        # Lineage tracking
        self.created_at_spawn: int = spawn_count
        self.parent_species_id: int | None = parent_id
        self.merge_count: int = 0

    def add(self, genome: Genome) -> None:
        self.members.append(genome)

    def best(self) -> Genome:
        if self._cached_best is not None:
            return self._cached_best
        return max(self.members, key=_fitness_key)

    def avg_shared_fitness(self) -> float:
        if not self.members:
            return 0.0
        return sum(g.shared_fitness for g in self.members) / len(self.members)

    def update_stagnation(self, best_genome: Genome | None = None) -> None:
        """Update stagnation counter based on the current best member's fitness.

        best_genome: pre-computed best member (avoids a second max() scan when
        the caller already computed sp.best() for another purpose).
        """
        if not self.members:
            return
        current_best = (best_genome if best_genome is not None else self.best()).fitness
        if current_best > self.best_fitness_seen:
            self.best_fitness_seen = current_best
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
