from __future__ import annotations
import math
import random

from yane.core.genome import Genome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compatibility(g1: Genome, g2: Genome) -> float:
    """Structural distance in [0, 1]: 0 = identical topology, 1 = fully different."""
    max_nodes = max(len(g1.nodes), len(g2.nodes), 1)
    max_conns = max(g1.connection_count, g2.connection_count, 1)
    node_diff = abs(len(g1.nodes) - len(g2.nodes)) / max_nodes
    conn_diff = abs(g1.connection_count - g2.connection_count) / max_conns
    return (node_diff + conn_diff) / 2.0


def _euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ---------------------------------------------------------------------------
# Species
# ---------------------------------------------------------------------------

class Species:
    """A group of structurally similar genomes that compete internally."""

    def __init__(self, representative: Genome) -> None:
        self.representative = representative
        self.members: list[Genome] = []

    def add(self, genome: Genome) -> None:
        self.members.append(genome)

    def best(self) -> Genome:
        return max(self.members, key=lambda g: g.fitness)

    def avg_shared_fitness(self) -> float:
        if not self.members:
            return 0.0
        return sum(g.shared_fitness for g in self.members) / len(self.members)


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

class Population:
    def __init__(self, max_size: int = 100, initial_genome: Genome | None = None) -> None:
        self.max_size = max_size
        seed = initial_genome or Genome()
        self._unevaluated: list[Genome] = [seed]
        # Separate copy as template for diversity injection — never evaluated,
        # never cleared by _prune().
        self._template: Genome = seed.copy()
        self._evaluated: list[Genome] = []
        self._species: list[Species] = []

        # Stagnation tracking — threshold scales automatically with population size.
        self._best_fitness_seen: float = float('-inf')
        self._stagnation_count: int = 0   # resets ONLY on fitness improvement
        self._since_last_injection: int = 0  # separate counter for injection pacing

        # Novelty search — probe inputs are fixed (seed 42) so behavior descriptors
        # are comparable across the lifetime of the population. No user config needed.
        n_in = len(self._template.input_nodes)
        _rng = random.Random(42)
        self._probe_inputs: list[list[float]] = [
            [_rng.uniform(-1.0, 1.0) for _ in range(n_in)]
            for _ in range(10)
        ] if n_in > 0 else []
        # id(genome) → behavior descriptor (concatenated outputs on probe_inputs)
        self._behaviors: dict[int, list[float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_for_evaluation(self) -> Genome:
        if not self._unevaluated:
            self._spawn_offspring()
        return self._unevaluated[0]

    def submit(self, genome: Genome, fitness: float) -> None:
        if genome not in self._unevaluated:
            return
        genome.fitness = fitness
        genome.shared_fitness = fitness
        self._unevaluated.remove(genome)
        self._evaluated.append(genome)

        # Compute behavior descriptor immediately after evaluation — the genome's
        # weights are still intact and forward() is deterministic.
        if self._probe_inputs:
            desc: list[float] = []
            for inp in self._probe_inputs:
                desc.extend(genome.forward(inp))
            self._behaviors[id(genome)] = desc

        if fitness > self._best_fitness_seen:
            self._best_fitness_seen = fitness
            self._stagnation_count = 0
            self._since_last_injection = 0
        else:
            self._stagnation_count += 1
            self._since_last_injection += 1

        self._prune()

    def get_best(self) -> Genome:
        if not self._evaluated:
            raise RuntimeError("No evaluated genomes yet.")
        return max(self._evaluated, key=lambda g: g.fitness)

    def get_top(self, k: int) -> list[Genome]:
        """Return up to k best evaluated genomes, sorted by fitness descending."""
        return sorted(self._evaluated, key=lambda g: g.fitness, reverse=True)[:k]

    @property
    def size(self) -> int:
        return len(self._evaluated) + len(self._unevaluated)

    @property
    def evaluated_count(self) -> int:
        return len(self._evaluated)

    @property
    def unevaluated_count(self) -> int:
        return len(self._unevaluated)

    @property
    def species_count(self) -> int:
        return len(self._species)

    @property
    def stagnation_count(self) -> int:
        return self._stagnation_count

    @property
    def stagnation_threshold(self) -> int:
        """Evaluations without improvement before diversity injection triggers."""
        return 2 * self.max_size

    @property
    def novelty_weight(self) -> float:
        """Selection bonus weight for novel genomes.

        Rises from 0.1 (no stagnation) to 0.5 (full stagnation) so novelty
        pressure increases exactly when fitness search is stuck. Requires no
        manual tuning — it adapts to the current training dynamics.
        """
        stagnation_frac = self._stagnation_count / max(1, self.stagnation_threshold)
        return 0.1 + 0.4 * min(1.0, stagnation_frac)

    # ------------------------------------------------------------------
    # Novelty
    # ------------------------------------------------------------------

    def _compute_novelty(self) -> dict[int, float]:
        """Mean behavioral distance of each genome to all others (population-relative).

        Returns a dict {id(genome): normalized_novelty} in [0, 1].
        Genomes without a behavior descriptor (e.g. from before probe_inputs
        were configured) receive novelty 0.
        """
        pairs = [
            (g, self._behaviors[id(g)])
            for g in self._evaluated
            if id(g) in self._behaviors
        ]
        if len(pairs) <= 1:
            return {id(g): 0.0 for g in self._evaluated}

        raw: dict[int, float] = {}
        for i, (g, bv) in enumerate(pairs):
            total = sum(
                _euclidean(bv, bv2)
                for j, (_, bv2) in enumerate(pairs)
                if i != j
            )
            raw[id(g)] = total / (len(pairs) - 1)

        max_nov = max(raw.values()) or 1.0
        result = {gid: v / max_nov for gid, v in raw.items()}
        # Genomes with no descriptor get 0
        for g in self._evaluated:
            result.setdefault(id(g), 0.0)
        return result

    # ------------------------------------------------------------------
    # Speciation & fitness sharing
    # ------------------------------------------------------------------

    def _assign_species(self) -> None:
        for sp in self._species:
            sp.members = []

        for genome in self._evaluated:
            placed = False
            for sp in self._species:
                threshold = (genome.species_threshold + sp.representative.species_threshold) / 2
                if _compatibility(genome, sp.representative) < threshold:
                    sp.add(genome)
                    placed = True
                    break
            if not placed:
                new_sp = Species(genome)
                new_sp.add(genome)
                self._species.append(new_sp)

        self._species = [sp for sp in self._species if sp.members]
        for sp in self._species:
            sp.representative = sp.best()

    def _compute_shared_fitness(self) -> None:
        for sp in self._species:
            n = len(sp.members)
            for genome in sp.members:
                genome.shared_fitness = genome.fitness / n

    # ------------------------------------------------------------------
    # Offspring spawning
    # ------------------------------------------------------------------

    def _inject_fresh_genome(self) -> None:
        """Escape stagnation by injecting a heavily-mutated genome.

        Alternates between mutating the global best (directed exploration near
        a known-good solution) and mutating the topology template (broad
        exploration of the search space).
        """
        if self._evaluated and random.random() < 0.5:
            base = self.get_best().copy()
        else:
            base = self._template.copy()
        for _ in range(random.randint(2, 5)):
            base.mutate()
        base.fitness = 0.0
        base.shared_fitness = 0.0
        self._unevaluated.append(base)
        self._since_last_injection = 0  # pace injections; stagnation_count untouched

    def _spawn_offspring(self) -> None:
        if not self._evaluated:
            self._unevaluated.append(self._template.copy())
            return

        if self._since_last_injection >= self.stagnation_threshold:
            self._inject_fresh_genome()
            return

        self._assign_species()
        self._compute_shared_fitness()

        novelty = self._compute_novelty()
        nw = self.novelty_weight

        # Elite genomes: best of each species + global best reproduce as exact copies.
        elites: set[int] = set()
        global_best = self.get_best()
        elites.add(id(global_best))
        for sp in self._species:
            elites.add(id(sp.best()))

        # Fitness-proportional selection weighted by offspring_factor and novelty bonus.
        min_fit = min(g.shared_fitness for g in self._evaluated)
        weights = [
            max(0.0, g.shared_fitness - min_fit + 1e-6)
            * g.offspring_factor
            * (1.0 + nw * novelty.get(id(g), 0.0))
            for g in self._evaluated
        ]
        if sum(weights) == 0:
            weights = [1.0] * len(self._evaluated)

        parent = random.choices(self._evaluated, weights=weights, k=1)[0]

        if id(parent) in elites:
            child = parent.copy()
            child.fitness = 0.0
            self._unevaluated.append(child)
            return

        if random.random() < parent.crossover_prob and len(self._evaluated) >= 2:
            sp_members = next(
                (sp.members for sp in self._species if parent in sp.members),
                self._evaluated,
            )
            candidates = [g for g in sp_members if g is not parent]
            if not candidates:
                candidates = [g for g in self._evaluated if g is not parent]
            other = random.choice(candidates)
            fitter, weaker = (parent, other) if parent.fitness >= other.fitness else (other, parent)
            child = fitter.crossover(weaker)
        else:
            child = parent.copy()

        child.mutate()
        child.fitness = 0.0
        self._unevaluated.append(child)

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def _prune(self) -> None:
        while self._evaluated and len(self._evaluated) + len(self._unevaluated) > self.max_size:
            if len(self._evaluated) <= 1:
                break

            protected: set[int] = {id(self.get_best())}
            for sp in self._species:
                if sp.members:
                    protected.add(id(sp.best()))

            candidates = [g for g in self._evaluated if id(g) not in protected]
            if not candidates:
                break

            worst = min(candidates, key=lambda g: g.fitness)
            self._behaviors.pop(id(worst), None)
            self._evaluated.remove(worst)
            # Remove from its species before clearing to release the reference
            # so the genome can be GC'd without clearing all species.
            for sp in self._species:
                if worst in sp.members:
                    sp.members.remove(worst)
                    break
            worst._clear()

        # Drop now-empty species; keep non-empty ones so species_count stays visible.
        self._species = [sp for sp in self._species if sp.members]

    def shrink_to(self, target_size: int) -> None:
        """Keep only the best `target_size` evaluated genomes (memory pressure)."""
        target_size = max(1, target_size)
        if len(self._evaluated) <= target_size:
            return
        self._evaluated.sort(key=lambda g: g.fitness, reverse=True)
        discarded = self._evaluated[target_size:]
        self._evaluated = self._evaluated[:target_size]
        for g in discarded:
            self._behaviors.pop(id(g), None)
            g._clear()
        self._species = []
