from __future__ import annotations
import random

import numpy as np

from yane.core.genome import Genome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compatibility(g1: Genome, g2: Genome) -> float:
    """Structural distance in [0, 1]: 0 = identical topology, 1 = fully different."""
    n1, n2 = len(g1.nodes), len(g2.nodes)
    c1, c2 = g1.connection_count, g2.connection_count
    node_diff = abs(n1 - n2) / max(n1, n2, 1)
    conn_diff = abs(c1 - c2) / max(c1, c2, 1)
    return (node_diff + conn_diff) / 2.0


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
            # Store as float32 array — avoids list→ndarray conversion in _compute_novelty.
            rows = [genome.forward(inp) for inp in self._probe_inputs]
            self._behaviors[id(genome)] = np.array(rows, dtype=np.float32).ravel()

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
        """Mean behavioral distance to all others, vectorised with NumPy.

        Returns {id(genome): normalized_novelty} in [0, 1].
        Pure-Python O(N²) loops were the dominant runtime cost (≈90%);
        NumPy broadcasting reduces this to a single matrix operation.
        """
        pairs = [
            (id(g), self._behaviors[id(g)])
            for g in self._evaluated
            if id(g) in self._behaviors
        ]
        if len(pairs) <= 1:
            return {id(g): 0.0 for g in self._evaluated}

        gids = [gid for gid, _ in pairs]
        mat = np.stack([bv for _, bv in pairs])          # (N, D) float32

        # Pairwise Euclidean distances in one vectorised step
        diff = mat[:, None, :] - mat[None, :, :]         # (N, N, D)
        dists = np.sqrt((diff * diff).sum(axis=2))        # (N, N)

        N = len(pairs)
        np.fill_diagonal(dists, 0.0)
        mean_dists = dists.sum(axis=1) / (N - 1)         # (N,)

        max_d = float(mean_dists.max())
        normalized = (mean_dists / max_d if max_d > 0 else mean_dists).tolist()

        result = dict(zip(gids, normalized))
        for g in self._evaluated:
            result.setdefault(id(g), 0.0)
        return result

    # ------------------------------------------------------------------
    # Speciation & fitness sharing
    # ------------------------------------------------------------------

    def _assign_species(self) -> None:
        for sp in self._species:
            sp.members = []

        # Precompute (n_nodes, n_connections) once per genome to avoid O(N) property
        # lookup inside the O(N²) compatibility loop.
        def _stats(g):
            return len(g.nodes), g.connection_count

        rep_stats = {id(sp.representative): _stats(sp.representative) for sp in self._species}

        for genome in self._evaluated:
            gn, gc = _stats(genome)
            placed = False
            for sp in self._species:
                rn, rc = rep_stats[id(sp.representative)]
                threshold = (genome.species_threshold + sp.representative.species_threshold) / 2
                node_diff = abs(gn - rn) / max(gn, rn, 1)
                conn_diff = abs(gc - rc) / max(gc, rc, 1)
                if (node_diff + conn_diff) / 2.0 < threshold:
                    sp.add(genome)
                    placed = True
                    break
            if not placed:
                new_sp = Species(genome)
                new_sp.add(genome)
                self._species.append(new_sp)
                rep_stats[id(genome)] = (gn, gc)

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
        """Escape stagnation by injecting a diverse genome.

        Uses three strategies with equal probability:
        - Heavily mutated best genome (exploit known-good structure + weights)
        - Template copy with re-randomised weights (same topology, fresh weights)
        - Heavily mutated template (structural + weight exploration)
        Fresh weights are critical: mutating from the same initial weights can
        keep the population in the same attractor basin indefinitely.
        """
        strategy = random.randint(0, 2)
        if strategy == 0 and self._evaluated:
            base = self.get_best().copy()
            for _ in range(random.randint(2, 5)):
                base.mutate()
        else:
            base = self._template.copy()
            if strategy == 1:
                # Re-randomise all weights to escape the initial-weight attractor.
                for node in base.nodes:
                    for conn in node.connections:
                        conn.weight = random.uniform(-1.0, 1.0)
            else:
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
