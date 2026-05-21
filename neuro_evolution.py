from __future__ import annotations
import gc
import math
import random
import statistics
import time
from typing import Callable


def _return_memory_to_os() -> None:
    """Ask glibc to return freed heap pages to the OS (Linux only)."""
    try:
        import ctypes
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def sanitize_fitness(
    value: float,
    fallback: float = 0.0,
    clip_low: float | None = None,
    clip_high: float | None = None,
) -> tuple[float, bool, bool]:
    """Sanitize a raw fitness value.

    Returns (sanitized_value, was_invalid, was_clipped).

    was_invalid: True when value was nan or inf (replaced by fallback).
    was_clipped: True when value was finite but outside [clip_low, clip_high].
    """
    if not math.isfinite(value):
        return fallback, True, False
    clipped = False
    if clip_low is not None and value < clip_low:
        value = clip_low
        clipped = True
    if clip_high is not None and value > clip_high:
        value = clip_high
        clipped = True
    return value, False, clipped


def _aggregate_fitnesses(fitnesses: list[float], aggregation: str, sigma_penalty: float) -> float:
    """Combine multiple fitness values into one.

    aggregation: "mean" | "median" | "min"
    sigma_penalty: subtract sigma_penalty * std from the result (0 = no penalty).
    """
    if len(fitnesses) == 1:
        return fitnesses[0]
    if aggregation == "median":
        result = statistics.median(fitnesses)
    elif aggregation == "min":
        result = min(fitnesses)
    else:
        result = statistics.mean(fitnesses)
    if sigma_penalty > 0.0:
        result -= sigma_penalty * statistics.pstdev(fitnesses)
    return result


from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.evolution.population import Population
from yane.evolution.efficiency_penalty import EfficiencyPenalty
from yane.util.resource_guard import ResourceGuard


class NeuroEvolution:
    def __init__(self) -> None:
        self._population: Population | None = None
        self.min_fitness: float | None = None
        self.max_iterations: int | None = None
        self._current_genome: Genome | None = None
        self._efficiency_penalty: EfficiencyPenalty | None = None
        self._resource_guard = ResourceGuard()
        self._resource_check_interval: int = 50  # check psutil every N iters (was 1 = ~5% overhead)
        self._population_size: int = 100
        self._n_workers: int = 1
        self._lamarck_steps: int = 0      # explicit mode: >0 = always-on before eval
        self._lamarck_sigma: float = 1.0  # step-size multiplier on genome.sigma_global
        # Adaptive Lamarck — fires automatically based on stagnation pressure.
        # Active when _lamarck_steps == 0 and _lamarck_max_steps > 0 (default).
        self._lamarck_max_steps: int = 3   # ceiling: steps at full stagnation
        self._lamarck_top_k: float = 0.2   # only refine top 20 % of evaluated pool
        self._lamarck_n_applied: int = 0   # cumulative refinements performed
        self._lamarck_n_steps_total: int = 0  # cumulative hill-climbing steps
        self._n_evaluations: int = 1
        self._eval_aggregation: str = "mean"
        self._eval_sigma_penalty: float = 0.0
        # Elitism (applied to population on configure())
        self._elite_count: int = 1
        self._species_elite_count: int = 1
        # Fitness sanitizing (disabled by default)
        self._sanitize: bool = False
        self._sanitize_fallback: float = 0.0
        self._sanitize_clip_low: float | None = None
        self._sanitize_clip_high: float | None = None
        self._n_invalid_fitness: int = 0   # nan / inf seen
        self._n_clipped_fitness: int = 0   # values clamped by clip bounds
        from yane.evolution.innovation import InnovationTracker
        self._tracker = InnovationTracker()

    @property
    def current_genome(self) -> Genome | None:
        return self._current_genome

    @property
    def is_configured(self) -> bool:
        return self._population is not None

    @property
    def population(self) -> Population | None:
        return self._population

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def configure(
        self,
        n_inputs: int,
        n_outputs: int,
        max_nodes: int | None = None,
        max_connections: int | None = None,
        n_initial_hidden: int = 0,
        stateful: bool = True,
    ) -> None:
        """Set up the input/output topology and initialise the population.

        The initial genome is fully connected (every input → every output).
        max_nodes / max_connections cap network growth per genome.
        n_initial_hidden pre-adds hidden nodes so the network has structural
        capacity from the start — useful for non-linearly-separable tasks.
        """
        from yane.core.connection import Connection
        from yane.util.activation import ActivationType
        import random

        tracker = self._tracker
        initial = Genome()
        initial.max_nodes = max_nodes
        initial.max_connections = max_connections
        initial.allow_memory = stateful

        for i in range(n_inputs):
            node = Node(NodeType.INPUT, innovation=tracker.next())
            node.input_index = i
            # LINEAR pass-through so raw input values reach the network unchanged.
            node.activation = ActivationType.LINEAR
            initial.nodes.append(node)
            initial.input_nodes.append(node)
        for _ in range(n_outputs):
            node = Node(NodeType.OUTPUT, innovation=tracker.next())
            node.persist_value = stateful
            initial.nodes.append(node)
            initial.output_nodes.append(node)

        if n_initial_hidden > 0:
            hidden_nodes = []
            for _ in range(n_initial_hidden):
                h = Node(NodeType.HIDDEN, innovation=tracker.next())
                initial.nodes.append(h)
                hidden_nodes.append(h)
            for inp in initial.input_nodes:
                for h in hidden_nodes:
                    innov = tracker.get_connection(inp.innovation, h.innovation)
                    conn = Connection(h, innovation=innov)
                    conn.weight = random.uniform(-1.0, 1.0)
                    inp.connections.append(conn)
            for h in hidden_nodes:
                for out in initial.output_nodes:
                    innov = tracker.get_connection(h.innovation, out.innovation)
                    conn = Connection(out, innovation=innov)
                    conn.weight = random.uniform(-1.0, 1.0)
                    h.connections.append(conn)
        # else: empty start — bootstrap adds connections via add_connection(tracker)

        initial._invalidate_topology()
        self._population = Population(
            max_size=self._population_size,
            initial_genome=initial,
            tracker=tracker,
        )
        self._population.elite_count = self._elite_count
        self._population.species_elite_count = self._species_elite_count

    def set_min_fitness(self, value: float) -> None:
        self.min_fitness = value

    def set_population_size(self, n: int) -> None:
        self._population_size = n
        if self._population is not None:
            self._population.max_size = n

    def set_max_iterations(self, n: int) -> None:
        self.max_iterations = n

    def set_resource_limits(
        self,
        min_free_gb: float = 2.0,
        max_used_percent: float = 85.0,
        max_process_gb: float | None = None,
    ) -> None:
        """Configure memory limits for training.

        min_free_gb / max_used_percent: pause training when system memory is low.
        max_process_gb: hard cap on this process's RAM. When exceeded, the
                        population is halved (keeping the best genomes) and the
                        GC is forced until usage drops below the limit.
        """
        self._resource_guard = ResourceGuard(
            min_free_gb=min_free_gb,
            max_used_percent=max_used_percent,
            max_process_gb=max_process_gb,
        )

    def set_efficiency_penalty(self, max_ms: float, penalty_per_ms: float) -> None:
        self._efficiency_penalty = EfficiencyPenalty(max_ms, penalty_per_ms)

    def set_multi_eval(
        self,
        n: int,
        aggregation: str = "mean",
        sigma_penalty: float = 0.0,
    ) -> None:
        """Evaluate each genome n times per generation and aggregate results.

        Useful for stochastic environments where a single episode is too noisy.

        aggregation:
            "mean"   — arithmetic mean (default)
            "median" — statistical median; robust against outlier episodes
            "min"    — worst-case fitness; most conservative

        sigma_penalty:
            Subtract sigma_penalty * std from the aggregated fitness.
            Penalises high-variance genomes regardless of aggregation mode.
            Has no effect when n=1 (std undefined).

        Cost: n fitness-function calls per genome instead of 1.
        Note: the manual loop (next_genome / submit_fitness) is not affected;
              aggregate manually if needed.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        valid = ("mean", "median", "min")
        if aggregation not in valid:
            raise ValueError(f"aggregation must be one of {valid}, got {aggregation!r}")
        self._n_evaluations = n
        self._eval_aggregation = aggregation
        self._eval_sigma_penalty = max(0.0, sigma_penalty)

    def set_fitness_sanitizing(
        self,
        *,
        fallback: float = 0.0,
        clip_low: float | None = None,
        clip_high: float | None = None,
    ) -> None:
        """Enable central fitness sanitizing.

        Applied to every fitness value before it reaches the population —
        in train(), submit_fitness(), and submit_fitness_batch().

        fallback:  value used when fitness is nan or inf (default 0.0).
        clip_low:  if set, fitness is clamped to at least this value.
        clip_high: if set, fitness is clamped to at most this value.

        Diagnostic counters (_n_invalid_fitness, _n_clipped_fitness) are
        accessible via population_memory_info() and reset when configure()
        is called.

        Call without arguments to enable sanitizing with its defaults:
            yane.set_fitness_sanitizing()
        """
        self._sanitize = True
        self._sanitize_fallback = fallback
        self._sanitize_clip_low = clip_low
        self._sanitize_clip_high = clip_high

    def set_lamarck(self, n_steps: int = 5, sigma: float = 1.0) -> None:
        """Explicit Lamarckian refinement: hill-climb every genome before evaluation.

        When n_steps > 0 this overrides the adaptive mode — every genome is
        refined for exactly n_steps hill-climbing attempts before its fitness
        is measured.  n_steps = 0 re-enables adaptive mode.

        Args:
            n_steps: hill-climbing attempts per genome (default 5).
                     0 = use adaptive mode (default behaviour).
            sigma:   multiplier on genome.sigma_global (default 1.0).
        """
        self._lamarck_steps = max(0, n_steps)
        self._lamarck_sigma = sigma
        if n_steps > 0:
            self._lamarck_max_steps = 0  # explicit overrides adaptive

    def set_lamarck_adaptive(
        self,
        max_steps: int = 3,
        top_k: float = 0.2,
        sigma: float = 1.0,
    ) -> None:
        """Configure the built-in adaptive Lamarck refinement.

        Adaptive Lamarck fires automatically during stagnation without any
        manual activation.  The number of hill-climbing steps scales linearly
        from 0 (no stagnation) to max_steps (full stagnation), and only
        genomes whose fitness falls in the top top_k fraction of the evaluated
        pool are refined — keeping the cost proportional to usefulness.

        Args:
            max_steps: maximum hill-climbing steps at full stagnation (default 3).
                       0 disables adaptive mode entirely.
            top_k:     fraction of the pool eligible for refinement (default 0.2).
                       1.0 = refine all genomes.
            sigma:     multiplier on genome.sigma_global (default 1.0).
        """
        self._lamarck_max_steps = max(0, max_steps)
        self._lamarck_top_k = max(0.0, min(1.0, top_k))
        self._lamarck_sigma = sigma
        self._lamarck_steps = 0  # adaptive mode requires explicit to be off

    # -------------------------------------------------------------------------
    # Training (automatic loop)
    # -------------------------------------------------------------------------

    def train(self, fitness_fn: Callable[[Genome], float]) -> int:
        """Run the evolutionary loop.

        Runs until min_fitness is reached or max_iterations is hit.
        If neither is set, runs indefinitely.
        Automatically pauses when system memory is low and resumes when it recovers.
        Returns the number of iterations performed.
        """
        self._ensure_configured()

        lamarck = self._lamarck_steps > 0

        iterations = 0
        while True:
            genome = self._population.select_for_evaluation()

            if lamarck:
                self._lamarck_refine(genome, fitness_fn)  # refines weights in-place

            fitness, elapsed_ms = self._run_evaluations(genome, fitness_fn)
            fitness = self._apply_sanitize(fitness)

            if self._efficiency_penalty is not None:
                fitness = self._efficiency_penalty.apply(fitness, elapsed_ms)

            self._population.submit(genome, fitness, elapsed_ms)
            iterations += 1

            if self.min_fitness is not None and fitness >= self.min_fitness:
                break
            if self.max_iterations is not None and iterations >= self.max_iterations:
                break

            if iterations % self._resource_check_interval == 0:
                self._enforce_memory_limit()
                while not self._resource_guard.system_ok():
                    time.sleep(0.5)

        return iterations

    def get_best(self) -> Genome:
        self._ensure_configured()
        return self._population.get_best()

    def get_ensemble(self, k: int = 3) -> list[Genome]:
        """Return the top-k genomes by fitness for ensemble inference."""
        self._ensure_configured()
        return self._population.get_top(k)

    def forward_ensemble(self, inputs: list[float], k: int = 3) -> list[float]:
        """Run inputs through the top-k genomes and return averaged outputs.

        Provides more robust predictions than a single genome by averaging
        the outputs of the k best-performing networks found so far.
        """
        top_k = self.get_ensemble(k)
        if not top_k:
            raise RuntimeError("No evaluated genomes yet.")
        all_outputs = [g.forward(inputs) for g in top_k]
        n_out = len(all_outputs[0])
        return [
            sum(out[i] for out in all_outputs) / len(all_outputs)
            for i in range(n_out)
        ]

    def population_memory_info(self) -> dict:
        """Returns node/connection counts across all genomes — useful for diagnosing memory growth."""
        self._ensure_configured()
        all_genomes = self._population._evaluated + list(self._population._unevaluated)
        if not all_genomes:
            return {"total_genomes": 0}
        infos = [g.memory_info() for g in all_genomes]
        total_nodes = sum(i["nodes"] for i in infos)
        total_connections = sum(i["connections"] for i in infos)
        max_nodes = max(i["nodes"] for i in infos)
        max_connections = max(i["connections"] for i in infos)
        info = {
            "total_genomes": len(all_genomes),
            "pop_evaluated":  len(self._population._evaluated),
            "pop_max":        self._population.max_size,
            "total_nodes": total_nodes,
            "total_connections": total_connections,
            "avg_nodes_per_genome": total_nodes / len(all_genomes),
            "avg_connections_per_genome": total_connections / len(all_genomes),
            "largest_genome_nodes": max_nodes,
            "largest_genome_connections": max_connections,
            "species_count":           self._population.species_count,
            "compat_threshold":        self._population._compat_threshold,
            "stagnation_count":        self._population.stagnation_count,
            "stagnation_threshold":    self._population.stagnation_threshold,
            "since_last_injection":    self._population._since_last_injection,
            "novelty_weight":          self._population.novelty_weight,
            "efficiency_weight":       self._population.efficiency_weight,
            "min_fitness": min((g.fitness for g in self._population._evaluated), default=0.0),
            "max_fitness": max((g.fitness for g in self._population._evaluated), default=0.0),
            "avg_fitness": (sum(g.fitness for g in self._population._evaluated)
                            / max(1, len(self._population._evaluated))),
            "n_evaluations":    self._n_evaluations,
            "eval_aggregation": self._eval_aggregation,
            # Elitism configuration
            "elite_count":         self._population.elite_count,
            "species_elite_count": self._population.species_elite_count,
            # Fitness sanitizing diagnostics
            "sanitize_enabled":    self._sanitize,
            "n_invalid_fitness":   self._n_invalid_fitness,
            "n_clipped_fitness":   self._n_clipped_fitness,
            # Offspring counters
            "n_crossover":           self._population._n_crossover,
            "n_mutation_only":       self._population._n_mutation_only,
            "n_diversity_injection": self._population._n_diversity_injection,
            # Lamarck diagnostics
            "lamarck_mode":          "explicit" if self._lamarck_steps > 0 else
                                     ("adaptive" if self._lamarck_max_steps > 0 else "off"),
            "lamarck_n_applied":     self._lamarck_n_applied,
            "lamarck_n_steps_total": self._lamarck_n_steps_total,
            # Best genome topology history: list of (total_submitted, n_nodes, n_conn, fitness)
            "best_topology_history": self._population._best_topology_history,
        }

        # Eval-time statistics (computed on demand from current evaluated genomes)
        eval_times = [
            g.eval_time_ms
            for g in self._population._evaluated
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

        return info

    def set_target_species(self, n: int) -> None:
        """Set the target number of species the population tries to maintain.

        The adaptive compatibility threshold rises/falls automatically to keep
        the actual species count close to this target.  Higher values protect
        more structural niches and help escape local optima (especially for
        XOR-like tasks where intermediate structures are temporarily worse).

        Default: 5.  For small discrete-mapping tasks (binary increment,
        XOR variants): 10–20 works significantly better.
        """
        n = max(1, n)
        if self._population is not None:
            self._population._target_species = n

    def set_elitism(
        self,
        elite_count: int = 1,
        species_elite_count: int = 1,
    ) -> None:
        """Configure explicit elitism for the population.

        elite_count:
            Number of top-fitness genomes (globally) that are never removed
            by pruning. Default 1 preserves the global best at all times.
            Set to 0 to disable global elite protection.

        species_elite_count:
            Number of top-fitness genomes per species that are never removed
            by pruning. Default 1 preserves each species' best genome,
            regardless of species size — this protects structural innovations
            even in single-genome species. Set to 0 to disable.

        Elite genomes are preserved unchanged in the evaluated pool across
        all generations. They CAN be selected as parents — mutated copies are
        what enter the offspring pool, never the elite object itself.
        """
        elite_count = max(0, elite_count)
        species_elite_count = max(0, species_elite_count)
        if self._population is not None:
            self._population.elite_count = elite_count
            self._population.species_elite_count = species_elite_count
        # Store so configure() picks them up if called after set_elitism()
        self._elite_count = elite_count
        self._species_elite_count = species_elite_count

    def set_n_workers(self, n: int) -> None:
        """Number of parallel workers for evaluation.

        0 = Auto (GUI measures eval speed and picks the optimal count).
        1 = sequential (default).
        n > 1 = fixed worker count.
        """
        self._n_workers = max(0, n)

    # -------------------------------------------------------------------------
    # Manual loop (for complex multi-step evaluation)
    # -------------------------------------------------------------------------

    def next_genome(self) -> Genome:
        """Select the next genome for evaluation. Use submit_fitness() when done."""
        self._ensure_configured()
        self._current_genome = self._population.select_for_evaluation()
        return self._current_genome

    def submit_fitness(self, fitness: float, elapsed_ms: float | None = None) -> None:
        if self._current_genome is None:
            raise RuntimeError("Call next_genome() before submit_fitness().")
        self._population.submit(self._current_genome, self._apply_sanitize(fitness), elapsed_ms)
        self._current_genome = None

    def next_genome_batch(self, n: int) -> list[Genome]:
        """Select n distinct genomes for parallel evaluation.

        At least one genome must have been evaluated before calling this method;
        spawn requires an evaluated population to generate offspring from.
        """
        self._ensure_configured()
        if not self._population._evaluated:
            raise RuntimeError(
                "next_genome_batch() requires at least one evaluated genome. "
                "Call next_genome() + submit_fitness() once before using the batch API."
            )
        while len(self._population._unevaluated) < n:
            self._population._spawn_offspring()
        return list(self._population._unevaluated[:n])

    def submit_fitness_batch(self, results: list[tuple]) -> None:
        """Submit fitness values for a batch of genomes."""
        for item in results:
            if len(item) == 3:
                genome, fitness, elapsed_ms = item
            else:
                genome, fitness = item
                elapsed_ms = None
            self._population.submit(genome, self._apply_sanitize(fitness), elapsed_ms)

    # -------------------------------------------------------------------------
    # Tick mode (operates on current_genome)
    # -------------------------------------------------------------------------

    def set_inputs(self, data: list[float]) -> None:
        genome = self._require_current_genome()
        expected = len(genome.input_nodes)
        if len(data) != expected:
            raise ValueError(f"Expected {expected} inputs, got {len(data)}.")
        genome.set_inputs(data)

    def tick(self) -> None:
        self._require_current_genome().tick()

    def get_outputs(self) -> list[float]:
        return self._require_current_genome().get_outputs()

    def reset(self) -> None:
        self._require_current_genome().reset()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _apply_sanitize(self, fitness: float) -> float:
        """Apply configured sanitizing; count and log anomalies. No-op when disabled."""
        if not self._sanitize:
            return fitness
        clean, invalid, clipped = sanitize_fitness(
            fitness,
            fallback=self._sanitize_fallback,
            clip_low=self._sanitize_clip_low,
            clip_high=self._sanitize_clip_high,
        )
        if invalid:
            self._n_invalid_fitness += 1
            from yane.util.logger import get_logger
            get_logger().warning(
                "sanitize_fitness: invalid value %r replaced with fallback %r "
                "(total invalid: %d)",
                fitness, self._sanitize_fallback, self._n_invalid_fitness,
            )
        elif clipped:
            self._n_clipped_fitness += 1
        return clean

    def _run_evaluations(
        self, genome: Genome, fitness_fn: Callable[[Genome], float]
    ) -> tuple[float, float]:
        """Evaluate genome (possibly multiple times) → (fitness, total_elapsed_ms).

        After the normal evaluation, adaptive Lamarck refinement fires when
        stagnation pressure is high and the genome ranks in the top fraction of
        the pool — improving weights without touching topology.  Explicit Lamarck
        (_lamarck_steps > 0) is handled in train() instead and skips this path.
        """
        start = time.perf_counter()
        if self._n_evaluations <= 1:
            fitness = fitness_fn(genome)
        else:
            fitnesses: list[float] = []
            for _ in range(self._n_evaluations):
                fitnesses.append(fitness_fn(genome))
            fitness = _aggregate_fitnesses(
                fitnesses, self._eval_aggregation, self._eval_sigma_penalty
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Adaptive Lamarck: fires after the baseline is known, only when
        # _lamarck_steps == 0 (explicit mode is off) and stagnation is high.
        if self._lamarck_steps == 0:
            n_steps = self._adaptive_lamarck_steps(fitness)
            if n_steps > 0:
                fitness = self._lamarck_refine(
                    genome, fitness_fn,
                    baseline_fitness=fitness,
                    n_steps=n_steps,
                )
                self._lamarck_n_applied += 1
                self._lamarck_n_steps_total += n_steps

        return fitness, elapsed_ms

    def _lamarck_refine(
        self,
        genome: Genome,
        fitness_fn: Callable[[Genome], float],
        baseline_fitness: float | None = None,
        n_steps: int | None = None,
    ) -> float:
        """Hill-climb weights and biases, return best fitness achieved.

        Only weights and biases are touched — topology (connections, nodes) is
        unchanged, so the compiled forward pass stays valid without rebuilding.
        Each step: perturb → evaluate → keep if better, else revert.

        Args:
            baseline_fitness: known fitness before hill-climbing.  If None a
                              fresh evaluation is performed to establish the baseline.
            n_steps:          override for the number of hill-climbing attempts.
                              Defaults to self._lamarck_steps.
        """
        steps = self._lamarck_steps if n_steps is None else n_steps
        if steps <= 0:
            return baseline_fitness if baseline_fitness is not None else fitness_fn(genome)

        conns = [conn for node in genome.nodes for conn in node.connections if conn.enabled]
        nodes = genome.nodes
        if not conns and not nodes:
            return baseline_fitness if baseline_fitness is not None else fitness_fn(genome)

        sigma = genome.sigma_global * self._lamarck_sigma
        if not (0.0 < sigma < 1e6):
            return baseline_fitness if baseline_fitness is not None else fitness_fn(genome)

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

        return best_fitness

    def _adaptive_lamarck_steps(self, fitness: float) -> int:
        """Compute how many Lamarck steps to apply based on stagnation + top-K.

        Returns 0 if adaptive mode is off or conditions aren't met.
        """
        if self._lamarck_max_steps <= 0 or self._population is None:
            return 0
        pop = self._population
        stag_frac = min(1.0, pop.stagnation_count / max(1, pop.stagnation_threshold))
        n_steps = round(stag_frac * self._lamarck_max_steps)
        if n_steps <= 0:
            return 0

        # Top-K gate: only refine genomes that rank in the top fraction of the pool.
        evaluated = pop._evaluated
        if evaluated and self._lamarck_top_k < 1.0:
            k = max(1, int(len(evaluated) * self._lamarck_top_k))
            # List comprehension is faster than a generator here: sorted() exhausts
            # generators lazily (per-yield overhead); a flat float list is cheaper.
            threshold = sorted(
                [g.fitness for g in evaluated], reverse=True
            )[min(k - 1, len(evaluated) - 1)]
            if fitness < threshold:
                return 0

        return n_steps

    def _enforce_memory_limit(self) -> None:
        if not self._resource_guard.process_over_limit():
            return
        from yane.util.logger import get_logger
        log = get_logger()
        before = len(self._population._evaluated)
        while self._resource_guard.process_over_limit() and len(self._population._evaluated) > 1:
            target = max(1, len(self._population._evaluated) // 2)
            self._population.shrink_to(target)
            gc.collect()
            _return_memory_to_os()
        after = len(self._population._evaluated)
        log.warning(
            "Memory limit enforced: population shrunk from %d to %d evaluated genomes",
            before, after,
        )

    def trim_memory(self) -> None:
        """Force Python to return freed heap pages to the OS. Call periodically."""
        gc.collect()
        _return_memory_to_os()

    def _ensure_configured(self) -> None:
        if not self.is_configured:
            raise RuntimeError("Call configure(n_inputs, n_outputs) first.")

    def _require_current_genome(self) -> Genome:
        if self._current_genome is None:
            raise RuntimeError("Call next_genome() first.")
        return self._current_genome
