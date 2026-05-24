"""Diagnostics builder for the YANE population.

build_population_info() gathers node/connection/fitness/species/Lamarck
stats into a single dict — the same structure returned by
NeuroEvolution.population_memory_info().
"""
from __future__ import annotations
import math
import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane.evolution.population import Population
    from yane.evolution.lamarck_refiner import LamarckRefiner
    from yane.evolution.fitness_sanitizer import FitnessSanitizer


def _fitness_iqr(evaluated: list) -> float:
    """Interquartile range of fitness across evaluated genomes (0.0 if < 4)."""
    if len(evaluated) < 4:
        return 0.0
    fits = sorted(g.fitness for g in evaluated)
    n = len(fits)
    q1 = fits[max(0, int(n * 0.25))]
    q3 = fits[min(n - 1, int(n * 0.75))]
    return q3 - q1


def build_population_info(
    population: Population,
    lamarck: LamarckRefiner,
    sanitizer: FitnessSanitizer,
    n_evaluations: int,
    eval_aggregation: str,
    n_early_stopped: int,
) -> dict:
    """Build the full diagnostics dict for a population snapshot."""
    all_genomes = population._evaluated + list(population._unevaluated)
    if not all_genomes:
        return {"total_genomes": 0}

    infos = [g.memory_info() for g in all_genomes]
    total_nodes = sum(i["nodes"] for i in infos)
    total_connections = sum(i["connections"] for i in infos)
    total_active_connections = sum(i.get("active_connections", 0) for i in infos)
    total_inactive_connections = sum(i.get("inactive_connections", 0) for i in infos)
    total_inactive_hidden = sum(i.get("inactive_hidden_nodes", 0) for i in infos)
    max_nodes = max(i["nodes"] for i in infos)
    max_connections = max(i["connections"] for i in infos)

    info: dict = {
        "total_genomes": len(all_genomes),
        "pop_evaluated":  len(population._evaluated),
        "pop_max":        population.max_size,
        "total_nodes": total_nodes,
        "total_connections": total_connections,
        "avg_nodes_per_genome": total_nodes / len(all_genomes),
        "avg_connections_per_genome": total_connections / len(all_genomes),
        "avg_active_connections_per_genome": total_active_connections / len(all_genomes),
        "avg_inactive_connections_per_genome": total_inactive_connections / len(all_genomes),
        "avg_inactive_hidden_nodes_per_genome": total_inactive_hidden / len(all_genomes),
        "largest_genome_nodes": max_nodes,
        "largest_genome_connections": max_connections,
        "species_count":           population.species_count,
        "target_species":          population._target_species,
        "compat_threshold":        population._compat_threshold,
        "stagnation_count":        population.stagnation_count,
        "stagnation_threshold":    population.stagnation_threshold,
        "since_last_injection":    population._since_last_injection,
        "spawn_count":             population._spawn_count,
        "dbg_last_adj_n":          population._dbg_last_adj_n,
        "dbg_adj_count":           population._dbg_adj_count,
        "novelty_weight":          population.novelty_weight,
        "efficiency_weight":       population.efficiency_weight,
        "min_fitness": min((g.fitness for g in population._evaluated), default=0.0),
        "max_fitness": max((g.fitness for g in population._evaluated), default=0.0),
        "avg_fitness": (sum(g.fitness for g in population._evaluated)
                        / max(1, len(population._evaluated))),
        "top5_avg_fitness": (
            sum(f for f in sorted(
                (g.fitness for g in population._evaluated), reverse=True
            )[:5]) / min(5, len(population._evaluated))
            if population._evaluated else 0.0
        ),
        "median_fitness": (statistics.median(g.fitness for g in population._evaluated)
                           if population._evaluated else 0.0),
        "fitness_iqr": _fitness_iqr(population._evaluated),
        "n_evaluations":    n_evaluations,
        "eval_aggregation": eval_aggregation,
        # Elitism
        "elite_count":         population.elite_count,
        "species_elite_count": population.species_elite_count,
        # Fitness sanitizing
        "sanitize_enabled":    sanitizer.enabled,
        "n_invalid_fitness":   sanitizer.n_invalid,
        "n_clipped_fitness":   sanitizer.n_clipped,
        # Offspring counters
        "n_crossover":              population._n_crossover,
        "n_mutation_only":          population._n_mutation_only,
        "n_diversity_injection":    population._n_diversity_injection,
        "n_interspecies_crossover": population._n_interspecies_crossover,
        "species_spawn_scores":     list(population._last_species_spawn_scores),
        "n_early_stopped":          n_early_stopped,
        # Mutation success tracking
        "mutation_success": dict(population._mutation_success),
        "mutation_total":   dict(population._mutation_total),
        # Lamarck diagnostics
        "lamarck_mode":             lamarck.mode,
        "lamarck_n_applied":        lamarck.n_applied,
        "lamarck_n_steps_total":    lamarck.n_steps_total,
        "lamarck_time_ms":          lamarck.time_ms,
        "lamarck_n_blocked_top_k":  lamarck.n_blocked_top_k,
        "lamarck_per_species": [
            {
                "species_idx": i,
                "members": len(sp.members),
                "stagnation_count": sp.stagnation_count,
                "lamarck_n_applied": sp.lamarck_n_applied,
                "lamarck_n_steps_total": sp.lamarck_n_steps_total,
            }
            for i, sp in enumerate(population._species)
        ],
        # Per-species mutation biases
        "species_mutation_biases": [
            {
                "species_idx": i,
                "members": len(sp.members),
                "biases": dict(sp.mutation_biases),
            }
            for i, sp in enumerate(population._species)
        ],
        # Species lineage
        "species_lineage": [
            {
                "species_idx": i,
                "members": len(sp.members),
                "age": population._spawn_count - sp.created_at_spawn,
                "parent_species_id": sp.parent_species_id,
                "merge_count": sp.merge_count,
                "stagnation_count": sp.stagnation_count,
            }
            for i, sp in enumerate(population._species)
        ],
        # Best genome topology history
        "best_topology_history": population._best_topology_history,
    }

    # Population-wide average mutation rates
    evaluated = population._evaluated
    if evaluated:
        info["pop_avg_sigma_global"] = sum(
            g.sigma_global for g in evaluated) / len(evaluated)
        info["pop_avg_lamarck_sigma"] = sum(
            getattr(g, 'lamarck_sigma', g.sigma_global) for g in evaluated
        ) / len(evaluated)
        info["pop_avg_add_node"] = sum(
            g.mutation_add_node.bool_rate for g in evaluated) / len(evaluated)
        info["pop_avg_rem_node"] = sum(
            g.mutation_remove_node.bool_rate for g in evaluated) / len(evaluated)
        info["pop_avg_add_conn"] = sum(
            g.mutation_add_connection.bool_rate for g in evaluated) / len(evaluated)
        info["pop_avg_rem_conn"] = sum(
            g.mutation_remove_connection.bool_rate for g in evaluated) / len(evaluated)
        all_nodes = [n for g in evaluated for n in g.nodes]
        if all_nodes:
            info["pop_avg_bias_rate"]  = sum(
                n.mutation_bias.shift_rate for n in all_nodes) / len(all_nodes)
            info["pop_avg_activ_rate"] = sum(
                n.mutation_activation.custom_rate for n in all_nodes) / len(all_nodes)
        all_conns = [c for g in evaluated for n in g.nodes for c in n.connections]
        if all_conns:
            info["pop_avg_weight_rate"] = sum(
                c.mutation.shift_rate for c in all_conns) / len(all_conns)

    # Eval-time statistics
    eval_times = [
        g.eval_time_ms
        for g in population._evaluated
        if g.eval_time_ms is not None and math.isfinite(g.eval_time_ms)
    ]
    if eval_times:
        sorted_times = sorted(eval_times)
        n = len(sorted_times)
        p95_idx = min(int(math.ceil(0.95 * n)) - 1, n - 1)
        info["eval_time_mean_ms"]   = statistics.mean(sorted_times)
        info["eval_time_median_ms"] = statistics.median(sorted_times)
        info["eval_time_p95_ms"]    = sorted_times[max(0, p95_idx)]
        info["eval_time_max_ms"]    = sorted_times[-1]

    # Fitness landscape diagnostics
    stag_th = max(1, population.stagnation_threshold)
    info["plateau_ratio"] = min(1.0, population.stagnation_count / stag_th)

    n_sub = population._total_submitted
    info["jump_rate"] = population._n_new_best / n_sub if n_sub > 0 else 0.0

    fitnesses = [g.fitness for g in population._evaluated if not math.isnan(g.fitness)]
    if len(fitnesses) >= 2:
        f_min, f_max = min(fitnesses), max(fitnesses)
        if f_max > f_min:
            bins = 10
            bin_width = (f_max - f_min) / bins
            hist = [0] * bins
            for f in fitnesses:
                idx = min(bins - 1, int((f - f_min) / bin_width))
                hist[idx] += 1
            info["fitness_histogram"] = {
                "min": f_min, "max": f_max, "bins": bins,
                "counts": hist,
                "bin_edges": [f_min + i * bin_width for i in range(bins + 1)],
            }
        else:
            info["fitness_histogram"] = None
    else:
        info["fitness_histogram"] = None

    return info
