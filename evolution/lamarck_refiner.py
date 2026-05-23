from __future__ import annotations
import random
import time
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from yane.core.genome import Genome
    from yane.evolution.population import Population


class LamarckRefiner:
    """Owns Lamarckian refinement state and algorithms."""

    def __init__(self) -> None:
        self.steps: int = 0         # explicit mode: >0 = always-on before eval
        self.sigma: float = 1.0     # step-size multiplier on genome.sigma_global
        # Adaptive mode — fires when steps == 0 and max_steps > 0 (default).
        self.max_steps: int = 3     # ceiling: steps at full stagnation
        self.top_k: float = 0.2     # only refine top fraction of evaluated pool
        # Cumulative counters
        self.n_applied: int = 0
        self.n_steps_total: int = 0
        self.time_ms: float = 0.0
        self.n_blocked_top_k: int = 0

    @property
    def mode(self) -> str:
        if self.steps > 0:
            return "explicit"
        if self.max_steps > 0:
            return "adaptive"
        return "off"

    def set_explicit(self, n_steps: int, sigma: float) -> None:
        self.steps = max(0, n_steps)
        self.sigma = sigma
        if n_steps > 0:
            self.max_steps = 0  # explicit overrides adaptive

    def set_adaptive(self, max_steps: int, top_k: float, sigma: float) -> None:
        self.max_steps = max(0, max_steps)
        self.top_k = max(0.0, min(1.0, top_k))
        self.sigma = sigma
        self.steps = 0  # adaptive requires explicit to be off

    def refine(
        self,
        genome: Genome,
        fitness_fn: Callable[[Genome], float],
        baseline_fitness: float | None = None,
        n_steps: int | None = None,
    ) -> float:
        """Hill-climb weights and biases; return best fitness achieved.

        Only weights and biases are touched — topology is unchanged, so the
        compiled forward pass stays valid without rebuilding.
        Each step: perturb → evaluate → keep if better, else revert.

        Args:
            baseline_fitness: known fitness before hill-climbing.
            n_steps: override for the number of attempts; defaults to self.steps.
        """
        steps = self.steps if n_steps is None else n_steps
        if steps <= 0:
            return baseline_fitness if baseline_fitness is not None else fitness_fn(genome)

        conns = [conn for node in genome.nodes for conn in node.connections if conn.enabled]
        nodes = genome.nodes
        if not conns and not nodes:
            return baseline_fitness if baseline_fitness is not None else fitness_fn(genome)

        sigma = genome.sigma_global * self.sigma
        if not (0.0 < sigma < 1e6):
            return baseline_fitness if baseline_fitness is not None else fitness_fn(genome)

        t0 = time.perf_counter()
        best_fitness = fitness_fn(genome) if baseline_fitness is None else baseline_fitness

        for _ in range(steps):
            saved_weights = [c.weight for c in conns]
            saved_biases  = [n.bias   for n in nodes]
            for c in conns:
                c.weight += random.gauss(0.0, sigma)
            for n in nodes:
                n.bias += random.gauss(0.0, sigma)
            new_fitness = fitness_fn(genome)
            if new_fitness > best_fitness:
                best_fitness = new_fitness
            else:
                for c, w in zip(conns, saved_weights):
                    c.weight = w
                for n, b in zip(nodes, saved_biases):
                    n.bias = b

        self.time_ms += (time.perf_counter() - t0) * 1000.0
        return best_fitness

    def adaptive_steps(
        self,
        genome: Genome,
        fitness: float,
        population: Population,
    ) -> int:
        """Compute Lamarck steps based on species/population stagnation + top-K gate.

        Uses genome's own species stagnation when available, falling back to
        the global population stagnation.

        Returns 0 if adaptive mode is off or conditions aren't met.
        """
        if self.max_steps <= 0 or population is None:
            return 0

        sp = population.get_species_for_genome(genome)
        if sp is not None and sp.stagnation_count > 0:
            stag_count = sp.stagnation_count
            stag_threshold = population.stagnation_threshold
        else:
            stag_count = population.stagnation_count
            stag_threshold = max(1, population.stagnation_threshold)

        stag_frac = min(1.0, stag_count / max(1, stag_threshold))
        n_steps = round(stag_frac * self.max_steps)
        if n_steps <= 0:
            return 0

        evaluated = population._evaluated
        if evaluated and self.top_k < 1.0:
            k = max(1, int(len(evaluated) * self.top_k))
            threshold = sorted(
                [g.fitness for g in evaluated], reverse=True
            )[min(k - 1, len(evaluated) - 1)]
            if fitness < threshold:
                self.n_blocked_top_k += 1
                if stag_frac >= 1.0 and self.n_blocked_top_k in (100, 1000, 10_000):
                    from yane.util.logger import log_warning
                    log_warning(
                        "Lamarck blocked %d× by top-k gate despite full stagnation "
                        "(fitness=%.4f < pool-threshold=%.4f, top_k=%.2f). "
                        "Consider n_evaluations>1 or lamarck_top_k=1.0 for noisy environments.",
                        self.n_blocked_top_k, fitness, threshold, self.top_k,
                    )
                return 0

        return n_steps
