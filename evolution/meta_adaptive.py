"""Meta-adaptive policy genes for YANE adaptive controls."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from yane.evolution.operator_scheduler import MAX_WEIGHT, MIN_WEIGHT

if TYPE_CHECKING:
    from yane.evolution.lamarck_refiner import LamarckRefiner
    from yane.evolution.operator_scheduler import OperatorScheduler
    from yane.evolution.population import Population


@dataclass
class PolicyGeneBounds:
    operator_exploration: tuple[float, float] = (0.3, 3.0)
    lamarck_budget: tuple[int, int] = (0, 200)
    interspecies_rate: tuple[float, float] = (0.0, 0.3)


@dataclass
class PolicyGenes:
    """Compact evolvable policy genome for adaptive feature parameters."""

    operator_exploration: float = 1.0
    lamarck_budget: int = 50
    interspecies_rate: float = 0.05

    def clamped(self, bounds: PolicyGeneBounds) -> "PolicyGenes":
        op_lo, op_hi = bounds.operator_exploration
        budget_lo, budget_hi = bounds.lamarck_budget
        inter_lo, inter_hi = bounds.interspecies_rate
        return PolicyGenes(
            operator_exploration=max(op_lo, min(op_hi, float(self.operator_exploration))),
            lamarck_budget=int(max(budget_lo, min(budget_hi, int(self.lamarck_budget)))),
            interspecies_rate=max(inter_lo, min(inter_hi, float(self.interspecies_rate))),
        )

    def mixed_with(self, other: "PolicyGenes", strength: float, bounds: PolicyGeneBounds) -> "PolicyGenes":
        s = max(0.0, min(1.0, float(strength)))
        return PolicyGenes(
            operator_exploration=self.operator_exploration * (1.0 - s) + other.operator_exploration * s,
            lamarck_budget=round(self.lamarck_budget * (1.0 - s) + other.lamarck_budget * s),
            interspecies_rate=self.interspecies_rate * (1.0 - s) + other.interspecies_rate * s,
        ).clamped(bounds)

    def mutated(self, rng: random.Random, strength: float, bounds: PolicyGeneBounds) -> "PolicyGenes":
        s = max(0.0, float(strength))
        budget_span = max(1, bounds.lamarck_budget[1] - bounds.lamarck_budget[0])
        return PolicyGenes(
            operator_exploration=self.operator_exploration * rng.lognormvariate(0.0, 0.25 * s),
            lamarck_budget=round(self.lamarck_budget + rng.gauss(0.0, budget_span * 0.08 * s)),
            interspecies_rate=self.interspecies_rate + rng.gauss(0.0, 0.03 * s),
        ).clamped(bounds)


@dataclass
class PolicyGeneRecord:
    source: str
    score: float
    previous_score: float | None
    genes: PolicyGenes
    reason: str


class MetaAdaptivePolicyEvolver:
    """Evolves adaptive-policy parameters at global and per-species scope."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        seed: int | None = None,
        bounds: PolicyGeneBounds | None = None,
        mutation_strength: float = 1.0,
        history_max: int = 200,
    ) -> None:
        self.enabled = enabled
        self.bounds = bounds or PolicyGeneBounds()
        self.global_genes = PolicyGenes().clamped(self.bounds)
        self.species_genes: dict[int, PolicyGenes] = {}
        self._scores: dict[str, float] = {}
        self._previous_scores: dict[str, float] = {}
        self._rng = random.Random(seed)
        self.mutation_strength = max(0.0, float(mutation_strength))
        self.history_max = max(1, int(history_max))
        self.history: list[PolicyGeneRecord] = []
        self.last_reason = "init"
        self.tick_count = 0

    def tick(
        self,
        population: "Population",
        scheduler: "OperatorScheduler | None" = None,
        lamarck: "LamarckRefiner | None" = None,
    ) -> None:
        self.tick_count += 1
        if not self.enabled:
            self.last_reason = "disabled"
            return
        self._sync_species(population)
        scores = self._measure_scores(population)
        best_genes = self._best_genes(scores)
        self._evolve_scope("global", scores.get("global", 0.0), best_genes)
        for sp in population._species:
            key = f"species:{id(sp)}"
            self._evolve_scope(key, scores.get(key, 0.0), best_genes)
        self._scores = scores
        self._apply(population, scheduler, lamarck)

    def _sync_species(self, population: "Population") -> None:
        current = {id(sp) for sp in population._species}
        for sid in current:
            self.species_genes.setdefault(sid, self.global_genes.mutated(self._rng, 0.25, self.bounds))
        for sid in list(self.species_genes):
            if sid not in current:
                self.species_genes.pop(sid, None)

    def _measure_scores(self, population: "Population") -> dict[str, float]:
        scores: dict[str, float] = {}
        scores["global"] = max((g.fitness for g in population._evaluated), default=0.0)
        for sp in population._species:
            key = f"species:{id(sp)}"
            scores[key] = max((g.fitness for g in sp.members), default=0.0)
        return scores

    def _best_genes(self, scores: dict[str, float]) -> PolicyGenes:
        candidates: list[tuple[float, PolicyGenes]] = [(scores.get("global", 0.0), self.global_genes)]
        for sid, genes in self.species_genes.items():
            candidates.append((scores.get(f"species:{sid}", 0.0), genes))
        return max(candidates, key=lambda item: item[0])[1]

    def _evolve_scope(self, key: str, score: float, best_genes: PolicyGenes) -> None:
        genes = self.global_genes if key == "global" else self.species_genes[int(key.split(":", 1)[1])]
        prev = self._scores.get(key)
        plateau = prev is not None and score <= prev
        if plateau:
            genes = genes.mixed_with(best_genes, 0.35, self.bounds).mutated(
                self._rng, self.mutation_strength, self.bounds
            )
            reason = "evolve:plateau"
        else:
            genes = genes.mutated(self._rng, 0.25 * self.mutation_strength, self.bounds)
            reason = "evolve:improved"
        if key == "global":
            self.global_genes = genes
            source = "global"
        else:
            sid = int(key.split(":", 1)[1])
            self.species_genes[sid] = genes
            source = key
        self.last_reason = reason
        self._record(source, score, prev, genes, reason)

    def _apply(
        self,
        population: "Population",
        scheduler: "OperatorScheduler | None",
        lamarck: "LamarckRefiner | None",
    ) -> None:
        genes = self.global_genes.clamped(self.bounds)
        population.configure_interspecies_crossover(
            genes.interspecies_rate,
            mode="fixed",
            min_rate=0.0,
            max_rate=max(genes.interspecies_rate, self.bounds.interspecies_rate[1]),
        )
        if lamarck is not None:
            lamarck.set_budget(genes.lamarck_budget if genes.lamarck_budget > 0 else None)
        if scheduler is not None:
            for op in ("add_node", "add_connection", "rewire"):
                old = scheduler.global_weights.get(op, 1.0)
                scheduler.global_weights[op] = max(
                    MIN_WEIGHT,
                    min(MAX_WEIGHT, old * 0.5 + genes.operator_exploration * 0.5),
                )
        for sp in population._species:
            sp_genes = self.species_genes.get(id(sp), genes).clamped(self.bounds)
            for op in ("add_node", "add_connection", "rewire"):
                sp.mutation_biases[op] = max(MIN_WEIGHT, min(MAX_WEIGHT, sp_genes.operator_exploration))

    def _record(
        self,
        source: str,
        score: float,
        previous_score: float | None,
        genes: PolicyGenes,
        reason: str,
    ) -> None:
        self.history.append(
            PolicyGeneRecord(
                source=source,
                score=float(score),
                previous_score=(float(previous_score) if previous_score is not None else None),
                genes=genes,
                reason=reason,
            )
        )
        if len(self.history) > self.history_max:
            self.history = self.history[-self.history_max:]

    def get_diagnostics(self) -> dict:
        return {
            "enabled": self.enabled,
            "tick": self.tick_count,
            "last_reason": self.last_reason,
            "bounds": {
                "operator_exploration": self.bounds.operator_exploration,
                "lamarck_budget": self.bounds.lamarck_budget,
                "interspecies_rate": self.bounds.interspecies_rate,
            },
            "global_genes": self.global_genes.__dict__.copy(),
            "species_genes": {
                str(sid): genes.__dict__.copy()
                for sid, genes in self.species_genes.items()
            },
            "scores": dict(self._scores),
            "history": [
                {
                    "source": row.source,
                    "score": row.score,
                    "previous_score": row.previous_score,
                    "genes": row.genes.__dict__.copy(),
                    "reason": row.reason,
                }
                for row in self.history[-20:]
            ],
        }

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.__dict__.setdefault("enabled", True)
        self.__dict__.setdefault("bounds", PolicyGeneBounds())
        self.__dict__.setdefault("global_genes", PolicyGenes().clamped(self.bounds))
        self.__dict__.setdefault("species_genes", {})
        self.__dict__.setdefault("_scores", {})
        self.__dict__.setdefault("_previous_scores", {})
        self.__dict__.setdefault("_rng", random.Random())
        self.__dict__.setdefault("mutation_strength", 1.0)
        self.__dict__.setdefault("history_max", 200)
        self.__dict__.setdefault("history", [])
        self.__dict__.setdefault("last_reason", "init")
        self.__dict__.setdefault("tick_count", 0)
