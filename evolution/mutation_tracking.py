"""Mutation and interspecies success accounting for Population."""
from __future__ import annotations

from yane.core.genome import Genome
from yane.evolution.species import Species


def record_interspecies_success(population, genome: Genome, fitness: float) -> None:
    """Update rolling success counters for interspecies offspring."""
    parent_fitness = getattr(genome, "_interspecies_parent_fitness", None)
    if parent_fitness is None:
        return

    population._interspecies_n_offspring += 1
    if fitness >= parent_fitness:
        population._interspecies_n_improved += 1
    population._interspecies_offspring_fitness.append(fitness)
    if len(population._interspecies_offspring_fitness) > population._interspecies_offspring_fitness_max:
        population._interspecies_offspring_fitness.pop(0)
    if population._interspecies_n_offspring >= 100:
        population._interspecies_n_offspring //= 2
        population._interspecies_n_improved //= 2


def record_mutation_success(
    mutation_total: dict[str, int],
    mutation_success: dict[str, int],
    species: list[Species],
    genome: Genome,
    fitness: float,
) -> None:
    """Record which mutation operators produced an offspring improvement."""
    parent_fitness = getattr(genome, "_parent_fitness", None)
    improved = parent_fitness is not None and fitness >= parent_fitness
    for mutation_type in getattr(genome, "_mutation_types_fired", []):
        mutation_total[mutation_type] = mutation_total.get(mutation_type, 0) + 1
        if improved:
            mutation_success[mutation_type] = mutation_success.get(mutation_type, 0) + 1
        _record_species_mutation_success(species, genome, mutation_type, improved)


def _record_species_mutation_success(
    species: list[Species],
    genome: Genome,
    mutation_type: str,
    improved: bool,
) -> None:
    for sp in species:
        if genome not in sp.members:
            continue
        sp._mut_total[mutation_type] = sp._mut_total.get(mutation_type, 0) + 1
        if improved:
            sp._mut_success[mutation_type] = sp._mut_success.get(mutation_type, 0) + 1
        break
