"""Adaptive Control Layer for YANE.

Central component that reads population/lamarck signals and adjusts
adaptive features (interspecies-crossover, lamarck, operator-scheduler,
population-size) through a unified policy interface.

Policy values: "off" | "fixed" | "adaptive" | "auto"
"""
from __future__ import annotations
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane.evolution.population import Population
    from yane.evolution.lamarck_refiner import LamarckRefiner


# ---------------------------------------------------------------------------
# Signal bundle — read-only snapshot computed each tick
# ---------------------------------------------------------------------------

class AdaptiveSignals:
    """Unified population signals for policy decisions."""

    __slots__ = (
        "plateau_ratio",        # [0,1] stagnation_count / threshold
        "fitness_trend",        # recent delta in best fitness (normalized)
        "diversity_score",      # mean novelty in [0,1]
        "species_stagnation",   # fraction of species that are stagnant
        "eval_cost",            # mean eval_time_ms (None if unknown)
        "best_complexity",      # (n_nodes, n_connections) of best genome
        "n_species",            # current species count
        "target_species",       # configured target
    )

    def __init__(self) -> None:
        self.plateau_ratio: float = 0.0
        self.fitness_trend: float = 0.0
        self.diversity_score: float = 0.0
        self.species_stagnation: float = 0.0
        self.eval_cost: float | None = None
        self.best_complexity: tuple[int, int] = (0, 0)
        self.n_species: int = 0
        self.target_species: int = 5


# ---------------------------------------------------------------------------
# Policy decision record — stored for diagnostics
# ---------------------------------------------------------------------------

class PolicyDecision:
    """Records a single adaptive policy adjustment."""

    __slots__ = ("feature", "reason", "old_value", "new_value", "trigger_signal")

    def __init__(
        self,
        feature: str,
        reason: str,
        old_value: float,
        new_value: float,
        trigger_signal: float,
    ) -> None:
        self.feature = feature
        self.reason = reason
        self.old_value = old_value
        self.new_value = new_value
        self.trigger_signal = trigger_signal


# ---------------------------------------------------------------------------
# Feature policy state
# ---------------------------------------------------------------------------

class FeaturePolicy:
    """Holds the configuration and current state for one adaptive feature."""

    def __init__(
        self,
        name: str,
        mode: str = "off",
        fixed_value: float = 0.0,
        min_value: float = 0.0,
        max_value: float = 1.0,
        budget: float | None = None,
    ) -> None:
        self.name = name
        self.mode = mode          # "off" | "fixed" | "adaptive" | "auto"
        self.fixed_value = fixed_value
        self.min_value = min_value
        self.max_value = max_value
        self.budget = budget      # optional budget (e.g. max evaluations per step)

        self.current_value: float = fixed_value
        self.last_reason: str = "init"
        self.last_trigger: float = 0.0

    @property
    def effective_value(self) -> float:
        if self.mode == "off":
            return 0.0
        if self.mode == "fixed":
            return self.fixed_value
        return self.current_value

    def clamp(self, value: float) -> float:
        return max(self.min_value, min(self.max_value, value))


# ---------------------------------------------------------------------------
# AdaptiveController
# ---------------------------------------------------------------------------

class AdaptiveController:
    """Central adaptive control layer.

    Reads unified population signals once per tick and dispatches
    policy updates to each registered adaptive feature.  All decisions
    are recorded in ``decisions`` for diagnostics.
    """

    # Fitness history length for trend computation
    _HISTORY_LEN = 10

    def __init__(self) -> None:
        self.policies: dict[str, FeaturePolicy] = {
            "interspecies_crossover": FeaturePolicy(
                "interspecies_crossover", mode="off",
                min_value=0.0, max_value=0.2,
            ),
            "lamarck": FeaturePolicy(
                "lamarck", mode="off",
                min_value=0, max_value=10,
            ),
            "operator_weights": FeaturePolicy(
                "operator_weights", mode="adaptive",
                min_value=0.3, max_value=3.0,
            ),
            "population_size": FeaturePolicy(
                "population_size", mode="off",
                min_value=10, max_value=1000,
            ),
            "qd_pressure": FeaturePolicy(
                "qd_pressure", mode="off",
                min_value=0.0, max_value=1.0,
            ),
            "pruning": FeaturePolicy(
                "pruning", mode="adaptive",
                min_value=0.0, max_value=1.0,
            ),
        }

        self.signals = AdaptiveSignals()
        self.decisions: list[PolicyDecision] = []
        self._max_decisions: int = 200

        # Short fitness history for trend computation
        self._fitness_history: list[float] = []
        self._tick_count: int = 0

        # Lamarck evaluation budget tracking (resets each generation)
        self._lamarck_evals_this_gen: int = 0
        self._lamarck_gen_budget: int | None = None  # None = unlimited

        # Cross-species success tracking: (offspring_count, improvement_count)
        self._interspecies_offspring_count: int = 0
        self._interspecies_improvement_count: int = 0
        self._interspecies_fitness_before: float | None = None

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def configure_interspecies_crossover(
        self,
        mode: str,
        fixed_rate: float = 0.0,
        min_rate: float = 0.0,
        max_rate: float = 0.2,
    ) -> None:
        p = self.policies["interspecies_crossover"]
        p.mode = mode
        p.fixed_value = fixed_rate
        p.min_value = min_rate
        p.max_value = max_rate
        p.current_value = fixed_rate

    def configure_lamarck(
        self,
        mode: str,
        fixed_steps: float = 0,
        min_steps: float = 0,
        max_steps: float = 10,
        budget_per_gen: int | None = None,
    ) -> None:
        p = self.policies["lamarck"]
        p.mode = mode
        p.fixed_value = fixed_steps
        p.min_value = min_steps
        p.max_value = max_steps
        p.budget = budget_per_gen
        self._lamarck_gen_budget = budget_per_gen

    def configure_population_size(
        self,
        mode: str,
        min_size: int = 10,
        max_size: int = 1000,
    ) -> None:
        p = self.policies["population_size"]
        p.mode = mode
        p.min_value = min_size
        p.max_value = max_size

    def configure_qd_pressure(
        self,
        mode: str,
        min_pressure: float = 0.0,
        max_pressure: float = 1.0,
    ) -> None:
        p = self.policies["qd_pressure"]
        p.mode = mode
        p.min_value = min_pressure
        p.max_value = max_pressure

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _compute_signals(self, population: Population) -> None:
        """Read population state into self.signals."""
        s = self.signals
        threshold = max(1, population.stagnation_threshold)

        s.plateau_ratio = min(1.0, population.stagnation_count / threshold)

        evaluated = population._evaluated
        if evaluated:
            best_fitness = max(g.fitness for g in evaluated)
            self._fitness_history.append(best_fitness)
            if len(self._fitness_history) > self._HISTORY_LEN:
                self._fitness_history.pop(0)
            if len(self._fitness_history) >= 2:
                delta = self._fitness_history[-1] - self._fitness_history[0]
                scale = abs(self._fitness_history[0]) if self._fitness_history[0] != 0 else 1.0
                s.fitness_trend = delta / scale if math.isfinite(delta / scale) else 0.0
            else:
                s.fitness_trend = 0.0

            best = max(evaluated, key=lambda g: g.fitness)
            s.best_complexity = (len(best.nodes), best.connection_count)
        else:
            s.fitness_trend = 0.0
            s.best_complexity = (0, 0)

        # Diversity: use mean novelty score from novelty cache
        if population._novelty_cache:
            vals = list(population._novelty_cache.values())
            s.diversity_score = sum(vals) / len(vals) if vals else 0.0
        else:
            s.diversity_score = 0.0

        # Species stagnation fraction
        if population._species:
            stagnant = sum(
                1 for sp in population._species
                if sp.stagnation_count >= threshold
            )
            s.species_stagnation = stagnant / len(population._species)
        else:
            s.species_stagnation = 0.0

        # Eval cost
        if population._eval_time_min is not None and population._eval_time_max is not None:
            eval_times = [
                g.eval_time_ms for g in evaluated
                if g.eval_time_ms is not None and math.isfinite(g.eval_time_ms)
            ]
            s.eval_cost = sum(eval_times) / len(eval_times) if eval_times else None
        else:
            s.eval_cost = None

        s.n_species = population.species_count
        s.target_species = population._target_species

    # ------------------------------------------------------------------
    # Per-feature policy ticks
    # ------------------------------------------------------------------

    def _tick_interspecies_crossover(self, population: Population) -> None:
        p = self.policies["interspecies_crossover"]
        if p.mode not in ("adaptive", "auto"):
            return

        s = self.signals
        # Pressure: plateau + species stagnation + low diversity
        diversity_pressure = max(0.0, 1.0 - s.diversity_score * 2)  # low novelty → more pressure
        pressure = max(s.plateau_ratio, s.species_stagnation, diversity_pressure * 0.5)

        old_val = p.current_value
        new_val = p.clamp(p.min_value + (p.max_value - p.min_value) * pressure)

        # Protection: if we recently had a fitness drop after crossover, back off
        if (
            self._interspecies_offspring_count >= 5
            and self._interspecies_improvement_count / max(1, self._interspecies_offspring_count) < 0.2
        ):
            new_val = p.clamp(new_val * 0.7)
            reason = "adaptive:crossover_low_success"
        elif pressure <= 0.05:
            reason = "adaptive:baseline"
        elif s.species_stagnation >= s.plateau_ratio and s.species_stagnation >= diversity_pressure:
            reason = "adaptive:species_stagnation"
        elif s.plateau_ratio >= s.species_stagnation and s.plateau_ratio >= diversity_pressure * 0.5:
            reason = "adaptive:global_plateau"
        else:
            reason = "adaptive:low_diversity"

        p.current_value = new_val
        p.last_reason = reason
        p.last_trigger = pressure

        if abs(new_val - old_val) > 1e-6:
            self._record(p.name, reason, old_val, new_val, pressure)

        # Apply to population
        population._interspecies_crossover_current = new_val
        population._interspecies_crossover_last_reason = reason

    def _tick_lamarck(self) -> None:
        p = self.policies["lamarck"]
        if p.mode not in ("adaptive", "auto"):
            return
        # Budget reset per generation is handled externally by reset_lamarck_budget()
        # The adaptive steps are computed in LamarckRefiner.adaptive_steps() already.
        # Here we expose the budget state in diagnostics.
        if self._lamarck_gen_budget is not None:
            p.last_reason = (
                f"budget:{self._lamarck_evals_this_gen}/{self._lamarck_gen_budget}"
            )
        else:
            p.last_reason = "adaptive:stagnation"

    def _tick_qd_pressure(self) -> None:
        p = self.policies["qd_pressure"]
        if p.mode not in ("adaptive", "auto"):
            return
        s = self.signals
        # More QD pressure at plateau, less when fitness is improving
        pressure = s.plateau_ratio * 0.7 + s.species_stagnation * 0.3
        new_val = p.clamp(p.min_value + (p.max_value - p.min_value) * pressure)
        old_val = p.current_value
        p.current_value = new_val
        p.last_trigger = pressure
        if s.plateau_ratio > 0.5:
            p.last_reason = "adaptive:plateau"
        else:
            p.last_reason = "adaptive:baseline"
        if abs(new_val - old_val) > 1e-4:
            self._record(p.name, p.last_reason, old_val, new_val, pressure)

    def _tick_pruning(self) -> None:
        p = self.policies["pruning"]
        if p.mode not in ("adaptive", "auto"):
            return
        s = self.signals
        n_nodes, n_conns = s.best_complexity
        # More pruning pressure when complexity grows and fitness stagnates
        complexity_score = min(1.0, (n_nodes + n_conns) / 100.0)
        pressure = s.plateau_ratio * 0.5 + complexity_score * 0.5
        new_val = p.clamp(p.min_value + (p.max_value - p.min_value) * pressure)
        old_val = p.current_value
        p.current_value = new_val
        p.last_trigger = pressure
        p.last_reason = "adaptive:complexity_plateau" if pressure > 0.4 else "adaptive:baseline"
        if abs(new_val - old_val) > 1e-4:
            self._record(p.name, p.last_reason, old_val, new_val, pressure)

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    def tick(self, population: Population, lamarck: "LamarckRefiner | None" = None) -> None:
        """Called once per spawn cycle to update all adaptive features."""
        self._tick_count += 1
        self._compute_signals(population)
        self._tick_interspecies_crossover(population)
        self._tick_lamarck()
        self._tick_qd_pressure()
        self._tick_pruning()

    # ------------------------------------------------------------------
    # Lamarck budget tracking
    # ------------------------------------------------------------------

    def reset_lamarck_budget(self) -> None:
        """Reset the per-generation Lamarck evaluation budget counter."""
        self._lamarck_evals_this_gen = 0

    def consume_lamarck_budget(self, n_evals: int) -> bool:
        """Record n_evals Lamarck evaluations. Returns True if still within budget."""
        self._lamarck_evals_this_gen += n_evals
        if self._lamarck_gen_budget is None:
            return True
        return self._lamarck_evals_this_gen <= self._lamarck_gen_budget

    def lamarck_budget_remaining(self) -> int | None:
        """Return remaining Lamarck evaluation budget this generation, or None if unlimited."""
        if self._lamarck_gen_budget is None:
            return None
        return max(0, self._lamarck_gen_budget - self._lamarck_evals_this_gen)

    # ------------------------------------------------------------------
    # Cross-species success tracking
    # ------------------------------------------------------------------

    def record_interspecies_offspring(self, improved: bool) -> None:
        """Track cross-species crossover outcome for protection logic."""
        self._interspecies_offspring_count += 1
        if improved:
            self._interspecies_improvement_count += 1
        # Rolling window: reset counts periodically
        if self._interspecies_offspring_count >= 50:
            self._interspecies_offspring_count //= 2
            self._interspecies_improvement_count //= 2

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _record(
        self,
        feature: str,
        reason: str,
        old_value: float,
        new_value: float,
        trigger_signal: float,
    ) -> None:
        self.decisions.append(PolicyDecision(feature, reason, old_value, new_value, trigger_signal))
        if len(self.decisions) > self._max_decisions:
            self.decisions = self.decisions[-self._max_decisions:]

    def get_diagnostics(self) -> dict:
        """Return a snapshot of all policy states and recent decisions."""
        s = self.signals
        return {
            "controller_tick": self._tick_count,
            "signals": {
                "plateau_ratio": s.plateau_ratio,
                "fitness_trend": s.fitness_trend,
                "diversity_score": s.diversity_score,
                "species_stagnation": s.species_stagnation,
                "eval_cost": s.eval_cost,
                "best_complexity_nodes": s.best_complexity[0],
                "best_complexity_connections": s.best_complexity[1],
                "n_species": s.n_species,
                "target_species": s.target_species,
            },
            "policies": {
                name: {
                    "mode": p.mode,
                    "current_value": p.current_value,
                    "fixed_value": p.fixed_value,
                    "min_value": p.min_value,
                    "max_value": p.max_value,
                    "budget": p.budget,
                    "last_reason": p.last_reason,
                    "last_trigger": p.last_trigger,
                }
                for name, p in self.policies.items()
            },
            "lamarck_budget": {
                "used": self._lamarck_evals_this_gen,
                "limit": self._lamarck_gen_budget,
                "remaining": self.lamarck_budget_remaining(),
            },
            "interspecies_success": {
                "offspring_count": self._interspecies_offspring_count,
                "improvement_count": self._interspecies_improvement_count,
                "success_rate": (
                    self._interspecies_improvement_count / self._interspecies_offspring_count
                    if self._interspecies_offspring_count > 0 else 0.0
                ),
            },
            "recent_decisions": [
                {
                    "feature": d.feature,
                    "reason": d.reason,
                    "old_value": d.old_value,
                    "new_value": d.new_value,
                    "trigger_signal": d.trigger_signal,
                }
                for d in self.decisions[-10:]
            ],
        }

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.__dict__.setdefault("_lamarck_gen_budget", None)
        self.__dict__.setdefault("_lamarck_evals_this_gen", 0)
        self.__dict__.setdefault("_interspecies_offspring_count", 0)
        self.__dict__.setdefault("_interspecies_improvement_count", 0)
        self.__dict__.setdefault("_tick_count", 0)
        self.__dict__.setdefault("_fitness_history", [])
