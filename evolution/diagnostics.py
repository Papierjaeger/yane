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
    from yane.evolution.adaptive_controller import AdaptiveController
    from yane.evolution.operator_scheduler import OperatorScheduler
    from yane.evolution.descriptors import AdaptiveFitnessComponentWeights
    from yane.evolution.meta_adaptive import MetaAdaptivePolicyEvolver


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
    adaptive_ctrl: "AdaptiveController | None" = None,
    operator_scheduler: "OperatorScheduler | None" = None,
    fitness_component_weights: "AdaptiveFitnessComponentWeights | None" = None,
    meta_adaptive: "MetaAdaptivePolicyEvolver | None" = None,
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
        "species_tuning_enabled":  getattr(population, "_species_tuning_enabled", True),
        "target_species_min":      getattr(population, "_target_species_min", None),
        "target_species_max":      getattr(population, "_target_species_max", None),
        "compat_threshold":        population._compat_threshold,
        "compat_tune_interval":    getattr(population, "_compat_tune_interval", 1),
        "compat_last_adjustment":  getattr(population, "_dbg_last_adj_step", 0.0),
        "compat_adjustment_trend": list(getattr(population, "_dbg_adj_trend", [])),
        "stagnation_count":        population.stagnation_count,
        "stagnation_threshold":    population.stagnation_threshold,
        "since_last_injection":    population._since_last_injection,
        "spawn_count":             population._spawn_count,
        "dbg_last_adj_n":          population._dbg_last_adj_n,
        "dbg_adj_count":           population._dbg_adj_count,
        "novelty_weight":          population.novelty_weight,
        "efficiency_weight":       population.efficiency_weight,
        "min_fitness": min((g.raw_fitness for g in population._evaluated), default=0.0),
        "max_fitness": max((g.raw_fitness for g in population._evaluated), default=0.0),
        "avg_fitness": (sum(g.raw_fitness for g in population._evaluated)
                        / max(1, len(population._evaluated))),
        "top5_avg_fitness": (
            sum(f for f in sorted(
                (g.raw_fitness for g in population._evaluated), reverse=True
            )[:5]) / min(5, len(population._evaluated))
            if population._evaluated else 0.0
        ),
        "median_fitness": (statistics.median(g.raw_fitness for g in population._evaluated)
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
        # Weight / bias / output clipping counters
        "n_weight_clipped":    population._n_weight_clipped,
        "n_bias_clipped":      population._n_bias_clipped,
        # Adaptive population size
        "adaptive_pop_enabled":       population._adaptive_pop_enabled,
        "n_pop_size_adjustments":     population._n_pop_size_adjustments,
        "n_output_sanitized":  sum(
            getattr(g, 'n_output_sanitized', 0) for g in all_genomes
        ),
        # Ablation flags
        "ablation_novelty_enabled":            population._novelty_enabled,
        "ablation_speciation_enabled":         population._speciation_enabled,
        "ablation_crossover_enabled":          population._crossover_enabled,
        "ablation_diversity_injection_enabled": population._diversity_injection_enabled,
        # Offspring counters
        "n_crossover":              population._n_crossover,
        "n_mutation_only":          population._n_mutation_only,
        "n_diversity_injection":    population._n_diversity_injection,
        "n_interspecies_crossover": population._n_interspecies_crossover,
        "interspecies_crossover_mode": population._interspecies_crossover_mode,
        "interspecies_crossover_rate": population._interspecies_crossover_rate,
        "interspecies_crossover_current": population._interspecies_crossover_current,
        "interspecies_crossover_min": population._interspecies_crossover_min,
        "interspecies_crossover_max": population._interspecies_crossover_max,
        "interspecies_crossover_last_reason": population._interspecies_crossover_last_reason,
        "species_spawn_scores":     list(population._last_species_spawn_scores),
        "n_early_stopped":          n_early_stopped,
        # Multi-objective diagnostics
        "multi_objective_enabled":  getattr(population, "_multi_objective_enabled", False),
        "multi_objective_maximize": getattr(population, "_multi_objective_maximize", None),
        # Quality Diversity diagnostics
        "quality_diversity_enabled": getattr(population, "_qd_enabled", False),
        "quality_diversity_cells": (
            len(population._qd_archive.cells) if getattr(population, "_qd_archive", None) else 0
        ),
        "quality_diversity_coverage": (
            population._qd_archive.coverage if getattr(population, "_qd_archive", None) else 0.0
        ),
        "quality_diversity_cells_data": (
            [
                {"cell": list(cell), "fitness": elite.fitness, "descriptor": list(elite.descriptor)}
                for cell, elite in sorted(population._qd_archive.cells.items())
            ]
            if getattr(population, "_qd_archive", None) else []
        ),
        "n_quality_diversity_updates": getattr(population, "_n_qd_updates", 0),
        "n_quality_diversity_injections": getattr(population, "_n_qd_injections", 0),
        # Mutation success tracking
        "mutation_success": dict(population._mutation_success),
        "mutation_total":   dict(population._mutation_total),
        # Lamarck diagnostics
        "lamarck_mode":             lamarck.mode,
        "lamarck_n_applied":        lamarck.n_applied,
        "lamarck_n_steps_total":    lamarck.n_steps_total,
        "lamarck_time_ms":          lamarck.time_ms,
        "lamarck_n_blocked_top_k":  lamarck.n_blocked_top_k,
        "lamarck_n_improved":       getattr(lamarck, "n_improved", 0),
        "lamarck_budget_per_gen":   getattr(lamarck, "budget_per_gen", None),
        "lamarck_budget_used":      getattr(lamarck, "_budget_used_this_gen", 0),
        "lamarck_budget_exhausted_count": getattr(lamarck, "_budget_exhausted_count", 0),
        "lamarck_species_stats":    dict(getattr(lamarck, "_species_stats", {})),
        # Interspecies crossover diagnostics (new)
        "interspecies_n_offspring":  getattr(population, "_interspecies_n_offspring", 0),
        "interspecies_n_improved":   getattr(population, "_interspecies_n_improved", 0),
        "interspecies_n_rejected":   getattr(population, "_interspecies_n_rejected", 0),
        "interspecies_success_rate": (
            getattr(population, "_interspecies_n_improved", 0) /
            max(1, getattr(population, "_interspecies_n_offspring", 1))
            if getattr(population, "_interspecies_n_offspring", 0) > 0 else 0.0
        ),
        "interspecies_recent_fitness": list(
            getattr(population, "_interspecies_offspring_fitness", [])[-10:]
        ),
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
    if getattr(population, "_module_library", None) is not None:
        info["module_library"] = population._module_library.diagnostics()
        info["module_insert_rate"] = getattr(population, "_module_insert_rate", 0.0)

    # Population-wide average mutation rates
    evaluated = population._evaluated
    if evaluated:
        pareto_points = [
            {
                "objectives": list(g.objectives),
                "fitness": g.fitness,
                "nodes": len(g.nodes),
                "connections": g.connection_count,
            }
            for g in evaluated
            if getattr(g, "objectives", None) is not None and len(g.objectives) >= 2
        ]
        if pareto_points:
            info["pareto_points"] = pareto_points
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
            weights = [float(c.weight) for c in all_conns]
            sorted_w = sorted(weights)
            n_w = len(sorted_w)
            mean_w = statistics.mean(sorted_w)
            std_w = statistics.pstdev(sorted_w) if n_w > 1 else 0.0
            p05 = sorted_w[max(0, min(n_w - 1, int(n_w * 0.05)))]
            p95 = sorted_w[max(0, min(n_w - 1, int(n_w * 0.95)))]
            lo, hi, bins = -5.0, 5.0, 20
            width = (hi - lo) / bins
            hist = [0] * bins
            under = over = 0
            for w in weights:
                if w < lo:
                    under += 1
                    idx = 0
                elif w > hi:
                    over += 1
                    idx = bins - 1
                else:
                    idx = min(bins - 1, max(0, int((w - lo) / width)))
                hist[idx] += 1
            info["weight_health"] = {
                "count": n_w,
                "mean": mean_w,
                "std": std_w,
                "p05": p05,
                "p95": p95,
                "dead_fraction": sum(1 for w in weights if abs(w) < 0.01) / n_w,
                "saturated_fraction": sum(1 for w in weights if abs(w) > 4.9) / n_w,
                "histogram": {
                    "min": lo,
                    "max": hi,
                    "bins": bins,
                    "counts": hist,
                    "underflow": under,
                    "overflow": over,
                    "bin_edges": [lo + i * width for i in range(bins + 1)],
                },
                "warning": (
                    "weight_collapse" if std_w < 0.05
                    else "weight_explosion" if std_w > 3.0
                    else None
                ),
            }

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

    # Adaptive Controller diagnostics
    if adaptive_ctrl is not None:
        info["adaptive_controller"] = adaptive_ctrl.get_diagnostics()

    # Operator Scheduler diagnostics
    if operator_scheduler is not None:
        info["operator_scheduler"] = operator_scheduler.get_diagnostics()
    elif population._operator_scheduler is not None:
        info["operator_scheduler"] = population._operator_scheduler.get_diagnostics()

    if fitness_component_weights is not None:
        info["fitness_component_weights"] = fitness_component_weights.get_diagnostics()

    if meta_adaptive is not None:
        info["meta_adaptive_policies"] = meta_adaptive.get_diagnostics()

    return info
