from __future__ import annotations
import time
from typing import Callable

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
        self._resource_check_interval: int = 10
        self._population_size: int = 100

    @property
    def current_genome(self) -> Genome | None:
        return self._current_genome

    @property
    def is_configured(self) -> bool:
        return self._population is not None

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def configure(
        self,
        n_inputs: int,
        n_outputs: int,
        max_nodes: int | None = None,
        max_connections: int | None = None,
    ) -> None:
        """Set up the input/output topology and initialise the population.

        max_nodes / max_connections cap network growth per genome.
        Without caps, networks can grow without bound and consume all memory.
        """
        initial = Genome()
        initial.max_nodes = max_nodes
        initial.max_connections = max_connections

        for i in range(n_inputs):
            node = Node(NodeType.INPUT)
            node.input_index = i
            initial.nodes.append(node)
            initial.input_nodes.append(node)
        for _ in range(n_outputs):
            node = Node(NodeType.OUTPUT)
            initial.nodes.append(node)
            initial.output_nodes.append(node)

        self._population = Population(max_size=self._population_size, initial_genome=initial)

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
    ) -> None:
        """Configure when training pauses to protect system memory.

        Training resumes automatically once memory is available again.
        """
        self._resource_guard = ResourceGuard(
            min_free_gb=min_free_gb,
            max_used_percent=max_used_percent,
        )

    def set_efficiency_penalty(self, max_ms: float, penalty_per_ms: float) -> None:
        self._efficiency_penalty = EfficiencyPenalty(max_ms, penalty_per_ms)

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

        iterations = 0
        while True:
            genome = self._population.select_for_evaluation()

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
                self._resource_guard.wait_if_needed()

        return iterations

    def get_best(self) -> Genome:
        self._ensure_configured()
        return self._population.get_best()

    def population_memory_info(self) -> dict:
        """Returns node/connection counts across all genomes — useful for diagnosing memory growth."""
        self._ensure_configured()
        all_genomes = self._population._evaluated + self._population._unevaluated
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
        }

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

    def _ensure_configured(self) -> None:
        if not self.is_configured:
            raise RuntimeError("Call configure(n_inputs, n_outputs) first.")

    def _require_current_genome(self) -> Genome:
        if self._current_genome is None:
            raise RuntimeError("Call next_genome() first.")
        return self._current_genome
