"""Adaptive Operator Scheduler for YANE.

Manages adaptive weighting for mutation operators, QD pressure,
novelty pressure, and pruning.  Tracks operator success rates
per species and globally, then adjusts weights accordingly.

All weights are multipliers on top of the genome's own rates:
  effective_rate = genome_rate * operator_weight
Weights are clamped to [MIN_WEIGHT, MAX_WEIGHT].
"""
from __future__ import annotations
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane.evolution.population import Population


# ---------------------------------------------------------------------------
# Operator categories
# ---------------------------------------------------------------------------

# Structural operators change network topology
STRUCTURAL_OPS = ("add_node", "remove_node", "add_connection", "remove_connection", "rewire")
# Weight operators affect only weights/biases
WEIGHT_OPS = ("disable", "enable")
# All tracked operator types
ALL_OPS = STRUCTURAL_OPS + WEIGHT_OPS

MIN_WEIGHT = 0.1
MAX_WEIGHT = 5.0
DEFAULT_WEIGHT = 1.0


# ---------------------------------------------------------------------------
# OperatorScheduler
# ---------------------------------------------------------------------------

class OperatorScheduler:
    """Adaptively schedules mutation, QD, novelty, and pruning operators.

    Call tick() each generation to update weights based on current signals.
    Apply weights via apply_to_genome() before mutating a child.
    """

    def __init__(self) -> None:
        # Global operator weights: {op_name: weight}
        self.global_weights: dict[str, float] = {op: DEFAULT_WEIGHT for op in ALL_OPS}
        # Per-species weights: {species_id: {op_name: weight}}
        self._species_weights: dict[int, dict[str, float]] = {}

        # QD pressure: 0.0 = fitness only, 1.0 = full novelty/diversity pressure
        self.qd_pressure: float = 0.0
        self.qd_pressure_min: float = 0.0
        self.qd_pressure_max: float = 1.0
        self.qd_pressure_mode: str = "off"  # "off" | "fixed" | "adaptive"

        # Pruning pressure: multiplier on remove_node/remove_connection
        self.pruning_pressure: float = 1.0
        self.pruning_pressure_min: float = 0.5
        self.pruning_pressure_max: float = 5.0
        self.pruning_pressure_mode: str = "adaptive"

        # Update momentum (0.9 = slow change, 0 = instant)
        self.momentum: float = 0.9

        # Global operator success tracking
        self._global_success: dict[str, int] = {op: 0 for op in ALL_OPS}
        self._global_total: dict[str, int] = {op: 0 for op in ALL_OPS}
        # Per-species success tracking
        self._species_success: dict[int, dict[str, int]] = {}
        self._species_total: dict[int, dict[str, int]] = {}

        # Diagnostics
        self._tick_count: int = 0
        self._last_signals: dict = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_qd_pressure(
        self,
        mode: str = "adaptive",
        min_pressure: float = 0.0,
        max_pressure: float = 1.0,
    ) -> None:
        self.qd_pressure_mode = mode
        self.qd_pressure_min = min_pressure
        self.qd_pressure_max = max_pressure

    def configure_pruning(
        self,
        mode: str = "adaptive",
        min_pressure: float = 0.5,
        max_pressure: float = 5.0,
    ) -> None:
        self.pruning_pressure_mode = mode
        self.pruning_pressure_min = min_pressure
        self.pruning_pressure_max = max_pressure

    # ------------------------------------------------------------------
    # Success tracking
    # ------------------------------------------------------------------

    def record_success(
        self,
        op: str,
        improved: bool,
        species_id: int | None = None,
    ) -> None:
        """Record a mutation outcome for operator weighting."""
        if op not in ALL_OPS:
            return
        self._global_total[op] = self._global_total.get(op, 0) + 1
        if improved:
            self._global_success[op] = self._global_success.get(op, 0) + 1
        if species_id is not None:
            if species_id not in self._species_total:
                self._species_total[species_id] = {o: 0 for o in ALL_OPS}
                self._species_success[species_id] = {o: 0 for o in ALL_OPS}
            self._species_total[species_id][op] = self._species_total[species_id].get(op, 0) + 1
            if improved:
                self._species_success[species_id][op] = self._species_success[species_id].get(op, 0) + 1

    def sync_from_population(self, population: Population) -> None:
        """Pull operator success data from population's mutation tracking."""
        for op in ALL_OPS:
            total = population._mutation_total.get(op, 0)
            success = population._mutation_success.get(op, 0)
            self._global_total[op] = total
            self._global_success[op] = success
        for sp in population._species:
            sid = id(sp)
            if sid not in self._species_total:
                self._species_total[sid] = {o: 0 for o in ALL_OPS}
                self._species_success[sid] = {o: 0 for o in ALL_OPS}
            for op in ALL_OPS:
                self._species_total[sid][op] = sp._mut_total.get(op, 0)
                self._species_success[sid][op] = sp._mut_success.get(op, 0)

    # ------------------------------------------------------------------
    # Tick: update all weights
    # ------------------------------------------------------------------

    def tick(self, population: Population) -> None:
        """Update operator weights based on population state. Call once per generation."""
        self._tick_count += 1
        self.sync_from_population(population)

        stagnation_threshold = max(1, population.stagnation_threshold)
        plateau = min(1.0, population.stagnation_count / stagnation_threshold)
        novelty_score = (
            sum(population._novelty_cache.values()) / len(population._novelty_cache)
            if population._novelty_cache else 0.0
        )

        # Best genome complexity
        best_nodes = 0
        best_conns = 0
        if population._evaluated:
            best = max(population._evaluated, key=lambda g: g.fitness)
            best_nodes = len(best.nodes)
            best_conns = best.connection_count

        self._last_signals = {
            "plateau": plateau,
            "novelty": novelty_score,
            "best_nodes": best_nodes,
            "best_connections": best_conns,
            "species_count": population.species_count,
        }

        # Update global operator weights from success rates
        self._update_global_weights(plateau, novelty_score, best_nodes, best_conns)

        # Update per-species weights
        for sp in population._species:
            sid = id(sp)
            self._update_species_weights(sid, sp.stagnation_count, stagnation_threshold)

        # Update QD pressure
        self._update_qd_pressure(plateau, novelty_score)

        # Update pruning pressure
        self._update_pruning_pressure(plateau, best_nodes, best_conns)

    def _update_global_weights(
        self,
        plateau: float,
        novelty: float,
        best_nodes: int,
        best_conns: int,
    ) -> None:
        """Adjust global operator weights from success rates and signals."""
        min_data = 10  # minimum samples before adjusting

        for op in ALL_OPS:
            total = self._global_total.get(op, 0)
            success = self._global_success.get(op, 0)
            old_weight = self.global_weights[op]

            if total >= min_data:
                success_rate = success / total
                # target: boost operators with > 50% success, dampen < 10% success
                target_mult = 0.5 + 1.5 * max(0.0, min(1.0, success_rate))
                target_weight = DEFAULT_WEIGHT * target_mult
            else:
                target_weight = old_weight  # not enough data

            # During plateau: boost exploration ops
            if plateau > 0.5:
                if op in ("add_node", "add_connection", "rewire"):
                    target_weight = min(MAX_WEIGHT, target_weight * (1.0 + 0.5 * plateau))
                elif op in ("disable",):
                    target_weight = min(MAX_WEIGHT, target_weight * (1.0 + 0.3 * plateau))

            # If network is growing large: boost remove ops (pruning)
            complexity = (best_nodes + best_conns) / 100.0
            if complexity > 0.5 and op in ("remove_node", "remove_connection"):
                target_weight = min(MAX_WEIGHT, target_weight * (1.0 + complexity * 0.5))

            # Momentum update
            new_weight = old_weight * self.momentum + target_weight * (1 - self.momentum)
            self.global_weights[op] = max(MIN_WEIGHT, min(MAX_WEIGHT, new_weight))

    def _update_species_weights(
        self,
        species_id: int,
        stagnation_count: int,
        stagnation_threshold: int,
    ) -> None:
        """Adjust per-species weights from species-level success rates."""
        if species_id not in self._species_weights:
            self._species_weights[species_id] = {op: DEFAULT_WEIGHT for op in ALL_OPS}

        sp_total = self._species_total.get(species_id, {})
        sp_success = self._species_success.get(species_id, {})
        stag_frac = min(1.0, stagnation_count / max(1, stagnation_threshold))

        for op in ALL_OPS:
            total = sp_total.get(op, 0)
            success = sp_success.get(op, 0)
            old_weight = self._species_weights[species_id][op]

            if total >= 5:
                success_rate = success / total
                target_mult = 0.5 + 1.5 * max(0.0, min(1.0, success_rate))
                target_weight = DEFAULT_WEIGHT * target_mult
            else:
                target_weight = old_weight

            # Stagnant species: boost exploration
            if stag_frac > 0.5 and op in ("add_node", "add_connection", "rewire"):
                target_weight = min(MAX_WEIGHT, target_weight * (1.0 + stag_frac * 0.3))

            new_weight = old_weight * self.momentum + target_weight * (1 - self.momentum)
            self._species_weights[species_id][op] = max(MIN_WEIGHT, min(MAX_WEIGHT, new_weight))

    def _update_qd_pressure(self, plateau: float, novelty: float) -> None:
        if self.qd_pressure_mode == "off":
            self.qd_pressure = 0.0
        elif self.qd_pressure_mode == "fixed":
            pass  # keep current value
        else:  # adaptive
            # More QD pressure at plateau + low novelty
            low_novelty_pressure = max(0.0, 0.3 - novelty) / 0.3
            pressure = plateau * 0.6 + low_novelty_pressure * 0.4
            target = self.qd_pressure_min + (self.qd_pressure_max - self.qd_pressure_min) * pressure
            self.qd_pressure = (
                self.qd_pressure * self.momentum + target * (1 - self.momentum)
            )
            self.qd_pressure = max(self.qd_pressure_min,
                                   min(self.qd_pressure_max, self.qd_pressure))

    def _update_pruning_pressure(
        self,
        plateau: float,
        best_nodes: int,
        best_conns: int,
    ) -> None:
        if self.pruning_pressure_mode == "off":
            self.pruning_pressure = 1.0
        elif self.pruning_pressure_mode == "fixed":
            pass
        else:  # adaptive
            # More pruning when complexity grows + stagnation
            complexity = min(1.0, (best_nodes + best_conns) / 100.0)
            pressure = plateau * 0.4 + complexity * 0.6
            target = (
                self.pruning_pressure_min
                + (self.pruning_pressure_max - self.pruning_pressure_min) * pressure
            )
            self.pruning_pressure = (
                self.pruning_pressure * self.momentum + target * (1 - self.momentum)
            )
            self.pruning_pressure = max(
                self.pruning_pressure_min,
                min(self.pruning_pressure_max, self.pruning_pressure),
            )
            # Apply pruning weight to remove ops
            if "remove_node" in self.global_weights:
                self.global_weights["remove_node"] = max(
                    MIN_WEIGHT,
                    min(MAX_WEIGHT, self.global_weights["remove_node"] * (1 + (self.pruning_pressure - 1) * 0.1))
                )
            if "remove_connection" in self.global_weights:
                self.global_weights["remove_connection"] = max(
                    MIN_WEIGHT,
                    min(MAX_WEIGHT, self.global_weights["remove_connection"] * (1 + (self.pruning_pressure - 1) * 0.1))
                )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def get_weights_for_species(self, species_id: int | None) -> dict[str, float]:
        """Return combined (global * species) weights for a genome's species."""
        if species_id is None or species_id not in self._species_weights:
            return dict(self.global_weights)
        sp_w = self._species_weights[species_id]
        return {
            op: max(MIN_WEIGHT, min(MAX_WEIGHT, self.global_weights[op] * sp_w.get(op, DEFAULT_WEIGHT)))
            for op in ALL_OPS
        }

    def apply_to_genome(self, genome, species_id: int | None = None) -> None:
        """Scale genome mutation rates by scheduler weights (in-place).

        Only scales the bool_rate of each MutationGene — does not change sigma.
        The genome keeps its base rate; the scheduler multiplies it.
        """
        from yane.evolution.mutation import Mutation
        weights = self.get_weights_for_species(species_id)
        for op, gene_attr in (
            ("add_node", "mutation_add_node"),
            ("remove_node", "mutation_remove_node"),
            ("add_connection", "mutation_add_connection"),
            ("remove_connection", "mutation_remove_connection"),
            ("rewire", "mutation_rewire"),
            ("disable", "mutation_disable_connection"),
            ("enable", "mutation_enable_connection"),
        ):
            w = weights.get(op, DEFAULT_WEIGHT)
            gene = getattr(genome, gene_attr, None)
            if gene is not None:
                gene.bool_rate = max(Mutation.MIN_RATE, min(0.999, gene.bool_rate * w))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> dict:
        success_rates = {}
        for op in ALL_OPS:
            total = self._global_total.get(op, 0)
            success = self._global_success.get(op, 0)
            success_rates[op] = success / total if total > 0 else 0.0

        return {
            "tick": self._tick_count,
            "global_weights": dict(self.global_weights),
            "qd_pressure": self.qd_pressure,
            "qd_pressure_mode": self.qd_pressure_mode,
            "pruning_pressure": self.pruning_pressure,
            "pruning_pressure_mode": self.pruning_pressure_mode,
            "operator_success_rates": success_rates,
            "last_signals": dict(self._last_signals),
            "n_species_tracked": len(self._species_weights),
        }

    def __getstate__(self) -> dict:
        return self.__dict__.copy()

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.__dict__.setdefault("qd_pressure_mode", "off")
        self.__dict__.setdefault("pruning_pressure_mode", "adaptive")
        self.__dict__.setdefault("_tick_count", 0)
        self.__dict__.setdefault("_last_signals", {})
