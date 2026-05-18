from __future__ import annotations
import gc
import random
import time
from typing import Callable


def _return_memory_to_os() -> None:
    """Ask glibc to return freed heap pages to the OS (Linux only)."""
    try:
        import ctypes
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except Exception:
        pass

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
        self._lamarck_steps: int = 0      # 0 = disabled; >0 = hill-climbing steps per genome
        self._lamarck_sigma: float = 1.0  # multiplier on genome.sigma_global (1.0 = unscaled)
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

    def set_lamarck(self, n_steps: int = 5, sigma: float = 1.0) -> None:
        """Enable Lamarckian weight refinement after each NEAT mutation.

        Before a genome is evaluated, its weights and biases are hill-climbed
        for `n_steps` attempts.  Each attempt perturbs all weights and biases
        with Gaussian noise and keeps the perturbation only if it improves fitness.

        The perturbation std-dev is  genome.sigma_global * sigma.  Because
        sigma_global is a self-adaptive strategy gene that evolves with each
        genome, the search step size automatically tunes itself — genomes that
        prefer large mutations search broadly, those that have converged search
        finely.  The sigma parameter here is just a global scale factor on top
        of that (default 1.0 = use sigma_global as-is).

        Cost: n_steps extra fitness-function calls per genome per generation.
        Benefit: weights converge much faster for the topology found by NEAT,
        especially on regression and continuous-output tasks.

        Args:
            n_steps: hill-climbing attempts per genome (default 5).
                     0 disables Lamarck entirely.
            sigma:   multiplier on genome.sigma_global (default 1.0).
                     < 1.0 for finer search, > 1.0 for wider jumps.
        """
        self._lamarck_steps = max(0, n_steps)
        self._lamarck_sigma = sigma

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
                self._lamarck_refine(genome, fitness_fn)

            start = time.perf_counter()
            fitness = fitness_fn(genome)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            if self._efficiency_penalty is not None:
                fitness = self._efficiency_penalty.apply(fitness, elapsed_ms)

            self._population.submit(genome, fitness)
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
        return {
            "total_genomes": len(all_genomes),
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
            "min_fitness": min((g.fitness for g in self._population._evaluated), default=0.0),
            "max_fitness": max((g.fitness for g in self._population._evaluated), default=0.0),
            "avg_fitness": (sum(g.fitness for g in self._population._evaluated)
                            / max(1, len(self._population._evaluated))),
        }

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

    def set_n_workers(self, n: int) -> None:
        """Number of parallel workers for evaluation (default 1 = sequential)."""
        self._n_workers = max(1, n)

    # -------------------------------------------------------------------------
    # Manual loop (for complex multi-step evaluation)
    # -------------------------------------------------------------------------

    def next_genome(self) -> Genome:
        """Select the next genome for evaluation. Use submit_fitness() when done."""
        self._ensure_configured()
        self._current_genome = self._population.select_for_evaluation()
        return self._current_genome

    def submit_fitness(self, fitness: float) -> None:
        if self._current_genome is None:
            raise RuntimeError("Call next_genome() before submit_fitness().")
        self._population.submit(self._current_genome, fitness)
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

    def submit_fitness_batch(self, results: list[tuple[Genome, float]]) -> None:
        """Submit fitness values for a batch of genomes."""
        for genome, fitness in results:
            self._population.submit(genome, fitness)

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

    def _lamarck_refine(self, genome: Genome, fitness_fn: Callable[[Genome], float]) -> None:
        """Hill-climb weights and biases for `_lamarck_steps` attempts.

        Only weights and biases are touched — topology (connections, nodes) is
        unchanged, so the compiled forward pass stays valid without rebuilding
        the execution order.  Each step:
          1. Perturb all weights + biases with Gaussian noise (σ = _lamarck_sigma).
          2. Evaluate fitness.
          3. Keep the perturbation if fitness improved; otherwise revert.

        The improved weights are stored back on the genome object and inherited
        by the population on submit() — this is the Lamarckian part.
        """
        conns = [conn for node in genome.nodes for conn in node.connections]
        nodes = genome.nodes
        if not conns and not nodes:
            return

        # Use the genome's own sigma_global as step size — it evolves along with
        # the genome, so well-adapted genomes automatically search at the right scale.
        # _lamarck_sigma acts as a multiplier (default 1.0 → pure sigma_global).
        sigma = genome.sigma_global * self._lamarck_sigma
        if not (0.0 < sigma < 1e6):  # defensive: skip if sigma is inf/nan/zero
            return
        best_fitness = fitness_fn(genome)

        for _ in range(self._lamarck_steps):
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
