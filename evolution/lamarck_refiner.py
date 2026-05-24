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
        # NES mode — if True, refine_nes() replaces refine() everywhere.
        self.nes_mode: bool = False
        self.nes_lr: float = 0.01   # learning rate for gradient step
        # SA (Simulated Annealing) mode — mutually exclusive with nes_mode.
        self.sa_mode: bool = False
        self.sa_cooling: float = 0.95   # geometric cooling factor per step
        self.sa_t0: float | None = None  # initial temperature; None = use sigma
        # CMA-ES mode — small full-covariance local optimizer.
        self.cma_mode: bool = False
        self.cma_population: int = 6
        # Cumulative counters
        self.n_applied: int = 0
        self.n_steps_total: int = 0
        self.time_ms: float = 0.0
        self.n_blocked_top_k: int = 0

    @property
    def mode(self) -> str:
        if self.nes_mode:
            prefix = "nes_"
        elif self.sa_mode:
            prefix = "sa_"
        elif self.cma_mode:
            prefix = "cma_es_"
        else:
            prefix = ""
        if self.steps > 0:
            return f"{prefix}explicit"
        if self.max_steps > 0:
            return f"{prefix}adaptive"
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

    def set_nes(
        self,
        k: int = 5,
        sigma: float = 1.0,
        learning_rate: float = 0.01,
        adaptive: bool = False,
    ) -> None:
        """Switch to NES (Natural Evolution Strategies) mode.

        Args:
            k:             Perturbation pairs per refinement call.  Costs 2k+1
                           fitness evaluations per genome.
            sigma:         Noise scale multiplier on genome.lamarck_sigma.
            learning_rate: Step size for the gradient update (default 0.01).
            adaptive:      If True, use adaptive scheduling (like set_lamarck_adaptive)
                           rather than applying to every genome.
        """
        self.nes_mode = True
        self.sa_mode = False   # mutually exclusive
        self.cma_mode = False
        self.nes_lr = float(learning_rate)
        self.sigma = float(sigma)
        if adaptive:
            self.steps = 0
            if self.max_steps <= 0:
                self.max_steps = k   # sensible default
        else:
            self.steps = max(1, k)
            self.max_steps = 0

    def set_sa(
        self,
        k: int = 5,
        sigma: float = 1.0,
        cooling_rate: float = 0.95,
        t0: float | None = None,
        adaptive: bool = False,
    ) -> None:
        """Switch to Simulated Annealing mode.

        Args:
            k:            Steps per refinement call.
            sigma:        Noise scale multiplier on genome.lamarck_sigma.
            cooling_rate: Geometric cooling factor per step (0 < cooling < 1).
                          Default 0.95 gives a gradual temperature decay.
            t0:           Initial temperature.  None → use the effective sigma,
                          which keeps perturbation and temperature on the same scale.
            adaptive:     If True, use adaptive scheduling instead of always-on.
        """
        self.sa_mode = True
        self.nes_mode = False   # mutually exclusive
        self.cma_mode = False
        self.sigma = float(sigma)
        self.sa_cooling = max(0.0, min(1.0, float(cooling_rate)))
        self.sa_t0 = float(t0) if t0 is not None else None
        if adaptive:
            self.steps = 0
            if self.max_steps <= 0:
                self.max_steps = k
        else:
            self.steps = max(1, k)
            self.max_steps = 0

    def set_cma_es(
        self,
        k: int = 3,
        sigma: float = 1.0,
        population: int = 6,
        adaptive: bool = False,
    ) -> None:
        """Switch to a compact full-covariance CMA-ES refinement mode."""
        self.cma_mode = True
        self.nes_mode = False
        self.sa_mode = False
        self.sigma = float(sigma)
        self.cma_population = max(2, int(population))
        if adaptive:
            self.steps = 0
            if self.max_steps <= 0:
                self.max_steps = k
        else:
            self.steps = max(1, k)
            self.max_steps = 0

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

        # Use the genome's own Lamarck sigma (evolvable, independent of sigma_global).
        # self.sigma is a global multiplier for runtime override via set_lamarck(sigma=).
        # Fall back to sigma_global for genomes pickled before lamarck_sigma existed.
        genome_sigma = getattr(genome, 'lamarck_sigma', genome.sigma_global)
        sigma = genome_sigma * self.sigma
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

    def refine_cma_es(
        self,
        genome: Genome,
        fitness_fn: Callable[[Genome], float],
        baseline_fitness: float | None = None,
        n_steps: int | None = None,
    ) -> float:
        """Small full-covariance CMA-ES update over weights and biases."""
        import numpy as np
        steps = self.steps if n_steps is None else n_steps
        if steps <= 0:
            return baseline_fitness if baseline_fitness is not None else fitness_fn(genome)

        conns = [c for node in genome.nodes for c in node.connections if c.enabled]
        nodes = genome.nodes
        params = conns + nodes
        if not params:
            return baseline_fitness if baseline_fitness is not None else fitness_fn(genome)

        genome_sigma = getattr(genome, 'lamarck_sigma', genome.sigma_global)
        sigma = genome_sigma * self.sigma
        if not (0.0 < sigma < 1e6):
            return baseline_fitness if baseline_fitness is not None else fitness_fn(genome)

        t0 = time.perf_counter()
        orig_w = [c.weight for c in conns]
        orig_b = [n.bias for n in nodes]
        mean = np.array(orig_w + orig_b, dtype=np.float64)
        dim = len(mean)
        cov = np.eye(dim, dtype=np.float64) * (sigma ** 2)
        lam = max(self.cma_population, 2 * dim)
        mu = max(1, lam // 2)
        best_fitness = fitness_fn(genome) if baseline_fitness is None else baseline_fitness
        best_vec = mean.copy()

        def write(vec) -> None:
            for c, value in zip(conns, vec[:len(conns)]):
                c.weight = float(value)
            for n, value in zip(nodes, vec[len(conns):]):
                n.bias = float(value)

        for _ in range(steps):
            candidates = np.random.multivariate_normal(mean, cov, size=lam)
            scored = []
            for vec in candidates:
                write(vec)
                scored.append((fitness_fn(genome), vec))
            scored.sort(key=lambda item: item[0], reverse=True)
            if scored[0][0] > best_fitness:
                best_fitness = scored[0][0]
                best_vec = scored[0][1].copy()
            elites = np.array([vec for _fit, vec in scored[:mu]], dtype=np.float64)
            mean = elites.mean(axis=0)
            centered = elites - mean
            if len(centered) > 1:
                cov = (centered.T @ centered) / (len(centered) - 1)
                cov += np.eye(dim) * 1e-9

        if best_fitness > (baseline_fitness if baseline_fitness is not None else float("-inf")):
            write(best_vec)
        else:
            for c, w in zip(conns, orig_w):
                c.weight = w
            for n, b in zip(nodes, orig_b):
                n.bias = b

        self.time_ms += (time.perf_counter() - t0) * 1000.0
        return best_fitness

    def refine_nes(
        self,
        genome: Genome,
        fitness_fn: Callable[[Genome], float],
        baseline_fitness: float | None = None,
        n_steps: int | None = None,
    ) -> float:
        """NES gradient update via antithetic perturbations; return best fitness.

        Uses k antithetic pairs (2k fitness evaluations) to estimate a gradient,
        applies one directed step, then accepts if better — otherwise reverts.
        Total cost: 2k+1 evaluations vs. k+1 for hill-climbing at equal steps.

        Args:
            baseline_fitness: known fitness before refinement.
            n_steps: number of perturbation pairs (k); defaults to self.steps.
        """
        k = self.steps if n_steps is None else n_steps
        if k <= 0:
            return baseline_fitness if baseline_fitness is not None else fitness_fn(genome)

        conns = [c for node in genome.nodes for c in node.connections if c.enabled]
        nodes = genome.nodes
        if not conns and not nodes:
            return baseline_fitness if baseline_fitness is not None else fitness_fn(genome)

        genome_sigma = getattr(genome, 'lamarck_sigma', genome.sigma_global)
        sigma = genome_sigma * self.sigma
        if not (0.0 < sigma < 1e6):
            return baseline_fitness if baseline_fitness is not None else fitness_fn(genome)

        t0 = time.perf_counter()
        baseline = fitness_fn(genome) if baseline_fitness is None else baseline_fitness

        orig_w = [c.weight for c in conns]
        orig_b = [n.bias   for n in nodes]
        d, d_b = len(orig_w), len(orig_b)

        grad_w = [0.0] * d
        grad_b = [0.0] * d_b

        for _ in range(k):
            noise_w = [random.gauss(0.0, 1.0) for _ in range(d)]
            noise_b = [random.gauss(0.0, 1.0) for _ in range(d_b)]

            # Positive perturbation
            for i, c in enumerate(conns):
                c.weight = orig_w[i] + sigma * noise_w[i]
            for i, n in enumerate(nodes):
                n.bias = orig_b[i] + sigma * noise_b[i]
            f_plus = fitness_fn(genome)

            # Negative perturbation (antithetic for variance reduction)
            for i, c in enumerate(conns):
                c.weight = orig_w[i] - sigma * noise_w[i]
            for i, n in enumerate(nodes):
                n.bias = orig_b[i] - sigma * noise_b[i]
            f_minus = fitness_fn(genome)

            # Antithetic gradient estimate: (f+ - f-) / 2 * noise
            diff = (f_plus - f_minus) * 0.5
            for i in range(d):
                grad_w[i] += diff * noise_w[i]
            for i in range(d_b):
                grad_b[i] += diff * noise_b[i]

        # Normalised gradient step: theta += lr * g / (k * sigma)
        scale = self.nes_lr / (k * sigma)
        for i, c in enumerate(conns):
            c.weight = orig_w[i] + scale * grad_w[i]
        for i, n in enumerate(nodes):
            n.bias = orig_b[i] + scale * grad_b[i]

        new_fitness = fitness_fn(genome)
        if new_fitness > baseline:
            best_fitness = new_fitness
        else:
            for i, c in enumerate(conns):
                c.weight = orig_w[i]
            for i, n in enumerate(nodes):
                n.bias = orig_b[i]
            best_fitness = baseline

        self.time_ms += (time.perf_counter() - t0) * 1000.0
        return best_fitness

    def refine_sa(
        self,
        genome: Genome,
        fitness_fn: Callable[[Genome], float],
        baseline_fitness: float | None = None,
        n_steps: int | None = None,
    ) -> float:
        """Simulated Annealing over weights and biases; return best fitness seen.

        Uses a geometric cooling schedule: T_k = T0 * cooling^k.  Worse moves
        are accepted with probability exp(Δf / T_k).  The best fitness seen
        across the chain is returned even if the walk ends at a worse point.

        Args:
            baseline_fitness: known fitness before SA.
            n_steps: number of SA steps; defaults to self.steps.
        """
        import math
        steps = self.steps if n_steps is None else n_steps
        if steps <= 0:
            return baseline_fitness if baseline_fitness is not None else fitness_fn(genome)

        conns = [c for node in genome.nodes for c in node.connections if c.enabled]
        nodes = genome.nodes
        if not conns and not nodes:
            return baseline_fitness if baseline_fitness is not None else fitness_fn(genome)

        genome_sigma = getattr(genome, 'lamarck_sigma', genome.sigma_global)
        sigma = genome_sigma * self.sigma
        if not (0.0 < sigma < 1e6):
            return baseline_fitness if baseline_fitness is not None else fitness_fn(genome)

        t0 = time.perf_counter()
        current_fitness = fitness_fn(genome) if baseline_fitness is None else baseline_fitness
        best_fitness = current_fitness
        T = self.sa_t0 if self.sa_t0 is not None else sigma
        cooling = self.sa_cooling

        for k in range(steps):
            saved_w = [c.weight for c in conns]
            saved_b = [n.bias   for n in nodes]
            for c in conns:
                c.weight += random.gauss(0.0, sigma)
            for n in nodes:
                n.bias += random.gauss(0.0, sigma)
            new_fitness = fitness_fn(genome)
            T_k = T * (cooling ** k)
            delta = new_fitness - current_fitness
            if delta > 0.0 or (T_k > 0.0 and random.random() < math.exp(delta / T_k)):
                current_fitness = new_fitness
                if new_fitness > best_fitness:
                    best_fitness = new_fitness
            else:
                for c, w in zip(conns, saved_w):
                    c.weight = w
                for n, b in zip(nodes, saved_b):
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
