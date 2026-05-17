from __future__ import annotations
import random

import numpy as np

from yane.core.genome import Genome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compatibility(g1: Genome, g2: Genome) -> float:
    """NEAT-style compatibility distance δ = c1·E/N + c2·D/N + c3·W̄

    When innovation numbers are available, uses the original NEAT formula
    with excess (E), disjoint (D) and average weight difference (W̄) of
    matching genes.  Falls back to the old topology-count heuristic for
    genomes without innovation tracking.

    Coefficients tuned to keep δ in roughly [0, 2]:
      c1 = 1.0 (excess)   c2 = 1.0 (disjoint)   c3 = 0.4 (weight diff)

    Hot path: both dicts are cached per genome (invalidated on topology change),
    so repeated comparisons against stable species representatives cost nothing
    beyond a dict lookup.
    """
    g1_innov, max_innov_g1 = g1._get_innov_cache()
    g2_innov, max_innov_g2 = g2._get_innov_cache()

    if not g1_innov and not g2_innov:
        # Legacy fallback: simple topology distance
        n1, n2 = len(g1.nodes), len(g2.nodes)
        c1, c2 = g1.connection_count, g2.connection_count
        node_diff = abs(n1 - n2) / max(n1, n2, 1)
        conn_diff = abs(c1 - c2) / max(c1, c2, 1)
        return (node_diff + conn_diff) / 2.0

    smaller_max = min(max_innov_g1, max_innov_g2)
    N = max(len(g1_innov), len(g2_innov), 1)

    excess = disjoint = matching = 0
    weight_diff_sum = 0.0

    # Iterate g1's genes; classify each as matching, disjoint, or excess.
    for innov, conn1 in g1_innov.items():
        conn2 = g2_innov.get(innov)
        if conn2 is not None:
            matching += 1
            weight_diff_sum += abs(conn1.weight - conn2.weight)
        elif innov > smaller_max:
            excess += 1
        else:
            disjoint += 1

    # Count g2-exclusive genes (g1 already handled matching ones above).
    for innov in g2_innov:
        if innov not in g1_innov:
            if innov > smaller_max:
                excess += 1
            else:
                disjoint += 1

    W_bar = weight_diff_sum / matching if matching > 0 else 0.0
    return (1.0 * excess / N) + (1.0 * disjoint / N) + (0.4 * W_bar)


# ---------------------------------------------------------------------------
# Species
# ---------------------------------------------------------------------------

class Species:
    """A group of structurally similar genomes that compete internally."""

    def __init__(self, representative: Genome) -> None:
        self.representative = representative
        self.members: list[Genome] = []
        self.best_fitness_seen: float = float('-inf')
        self.stagnation_count: int = 0   # spawn-cycles without fitness improvement

    def add(self, genome: Genome) -> None:
        self.members.append(genome)

    def best(self) -> Genome:
        return max(self.members, key=lambda g: g.fitness)

    def avg_shared_fitness(self) -> float:
        if not self.members:
            return 0.0
        return sum(g.shared_fitness for g in self.members) / len(self.members)

    def update_stagnation(self) -> None:
        """Update stagnation counter based on the current best member's fitness."""
        if not self.members:
            return
        current_best = self.best().fitness
        if current_best > self.best_fitness_seen:
            self.best_fitness_seen = current_best
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

class Population:
    def __init__(
        self,
        max_size: int = 100,
        initial_genome: Genome | None = None,
        tracker=None,
        target_species: int = 5,
    ) -> None:
        self.max_size = max_size
        seed = initial_genome or Genome()
        self._unevaluated: list[Genome] = [seed]
        # Separate copy as template for diversity injection — never evaluated,
        # never cleared by _prune().
        self._template: Genome = seed.copy()
        self._evaluated: list[Genome] = []
        self._species: list[Species] = []
        self._tracker = tracker  # InnovationTracker shared with NeuroEvolution

        # Stagnation tracking — threshold scales automatically with population size.
        self._best_fitness_seen: float = float('-inf')
        self._last_improvement_connections: int = 0   # connections when fitness last improved
        self._stagnation_count: int = 0   # resets ONLY on fitness improvement
        self._since_last_injection: int = 0  # separate counter for injection pacing

        # Structural stagnation: tracks how many evals the *topology* of the best
        # genome hasn't changed, even if fitness keeps improving (e.g. via Lamarck).
        # Lamarck causes fitness to creep up via weight tuning while the structure
        # never changes → _stagnation_count never triggers injection → the population
        # converges to weight-variants of one topology with no structural diversity.
        self._best_topology: tuple[int, int] = (0, 0)  # (n_nodes, n_connections)
        self._topology_stagnation_count: int = 0

        # Adaptive global species threshold (replaces per-genome thresholds).
        # Step size 0.005 is small relative to the actual δ distribution
        # ([0, ~0.15] for bootstrap genomes) to avoid oscillation.
        self._target_species: int = target_species
        self._compat_threshold: float = 0.2

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
        # Novelty archive: persists the most novel descriptors seen across all
        # generations so novelty doesn't collapse as the population converges.
        self._novelty_archive: list[np.ndarray] = []
        self._novelty_archive_max: int = 200
        # Novelty cache: recomputed at most once per max_size evaluations.
        # A dirty flag alone doesn't help since submit() and spawn() alternate —
        # instead, track how many evals since last recompute.
        self._novelty_cache: dict[int, float] = {}
        self._novelty_evals_since_recompute: int = 0

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
            # float64: float32 overflows when networks produce large outputs (e.g. SINE).
            rows = [genome.forward(inp) for inp in self._probe_inputs]
            self._behaviors[id(genome)] = np.array(rows, dtype=np.float64).ravel()
        self._novelty_evals_since_recompute += 1

        topo = (len(genome.nodes), genome.connection_count)
        if fitness > self._best_fitness_seen:
            self._best_fitness_seen = fitness
            self._stagnation_count = 0
            self._since_last_injection = 0
            self._last_improvement_connections = genome.connection_count
            # If the new best has a different topology, reset structural stagnation.
            if topo != self._best_topology:
                self._best_topology = topo
                self._topology_stagnation_count = 0
            else:
                # Fitness improved but topology unchanged (typical Lamarck effect):
                # still count as structural stagnation.
                self._topology_stagnation_count += 1
        else:
            self._stagnation_count += 1
            self._since_last_injection += 1
            self._topology_stagnation_count += 1

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
        return self.max_size

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
        """Mean behavioral distance to population + archive, vectorised with NumPy.

        Returns {id(genome): normalized_novelty} in [0, 1].
        The archive preserves the most novel descriptors seen across all
        generations so novelty doesn't collapse as the population converges.
        """
        pairs = [
            (id(g), self._behaviors[id(g)])
            for g in self._evaluated
            if id(g) in self._behaviors
        ]
        if len(pairs) <= 1:
            return {id(g): 0.0 for g in self._evaluated}

        gids = [gid for gid, _ in pairs]
        pop_mat = np.stack([bv for _, bv in pairs])      # (N, D) float32

        # Include archive in the reference set so novelty is measured against
        # all interesting behaviours seen, not just the current population.
        if self._novelty_archive:
            archive_mat = np.stack(self._novelty_archive)  # (A, D)
            ref_mat = np.concatenate([pop_mat, archive_mat], axis=0)
        else:
            ref_mat = pop_mat

        # Mean distance from each population member to all reference points.
        diff = pop_mat[:, None, :] - ref_mat[None, :, :]  # (N, R, D)
        dists = np.sqrt((diff * diff).sum(axis=2))         # (N, R)
        # Exclude self-distance (always 0 for archive members it won't be there,
        # but for population members the self-row should be masked).
        N = len(pairs)
        np.fill_diagonal(dists[:, :N], np.inf)
        finite = np.where(np.isinf(dists), 0.0, dists)
        counts = np.where(np.isinf(dists), 0, 1).sum(axis=1)
        mean_dists = finite.sum(axis=1) / np.maximum(counts, 1)

        max_d = float(mean_dists.max())
        normalized = (mean_dists / max_d if max_d > 0 else mean_dists).tolist()

        result = dict(zip(gids, normalized))
        for g in self._evaluated:
            result.setdefault(id(g), 0.0)

        # Add genomes that are novel enough to the archive.
        threshold = 0.3
        for (gid, bv), nov in zip(pairs, normalized):
            if nov >= threshold:
                self._novelty_archive.append(bv)
        # Keep archive bounded: discard oldest entries when full.
        if len(self._novelty_archive) > self._novelty_archive_max:
            self._novelty_archive = self._novelty_archive[-self._novelty_archive_max:]

        return result

    # ------------------------------------------------------------------
    # Speciation & fitness sharing
    # ------------------------------------------------------------------

    def _assign_species(self) -> None:
        for sp in self._species:
            sp.members = []

        threshold = self._compat_threshold
        # Map id(species) → species for O(1) fast-path lookup.
        species_by_id: dict[int, Species] = {id(sp): sp for sp in self._species}

        for genome in self._evaluated:
            placed = False

            # Fast path: try last-known species first — avoids full search for
            # stable genomes that haven't drifted away from their cluster.
            last_sp_id = getattr(genome, '_last_species_id', None)
            if last_sp_id is not None:
                sp = species_by_id.get(last_sp_id)
                if sp is not None and _compatibility(genome, sp.representative) < threshold:
                    sp.add(genome)
                    placed = True

            if not placed:
                for sp in self._species:
                    if _compatibility(genome, sp.representative) < threshold:
                        sp.add(genome)
                        genome._last_species_id = id(sp)
                        placed = True
                        break
            if not placed:
                new_sp = Species(genome)
                new_sp.add(genome)
                self._species.append(new_sp)
                species_by_id[id(new_sp)] = new_sp
                genome._last_species_id = id(new_sp)

        self._species = [sp for sp in self._species if sp.members]
        for sp in self._species:
            sp.representative = sp.best()
            sp.update_stagnation()   # track per-species fitness progress

        # Adapt threshold so species count converges to target_species.
        # Step 0.005 keeps adaptation smooth relative to δ range [0, ~0.15].
        n = len(self._species)
        if n > self._target_species:
            self._compat_threshold = min(1.5, self._compat_threshold + 0.005)
        elif n < self._target_species:
            self._compat_threshold = max(0.01, self._compat_threshold - 0.005)

    def _compute_shared_fitness(self) -> None:
        # No parsimony penalty: even a tiny coefficient causes evolution to
        # converge to empty/minimal genomes when the fitness signal is weak
        # (e.g. CartPole baseline fitness ~11 for always-push-left; 0.0001 ×
        # 4 connections = 0.0004 disadvantage makes connected genomes lose
        # tournaments against the empty baseline). Compact networks emerge
        # naturally through remove_connection/remove_node mutations when they
        # don't improve fitness.
        for sp in self._species:
            n = len(sp.members)
            for genome in sp.members:
                genome.shared_fitness = genome.fitness / n

    # ------------------------------------------------------------------
    # Offspring spawning
    # ------------------------------------------------------------------

    def _inject_structural_diversity(self) -> None:
        """Force structural exploration when topology stagnates despite fitness gains.

        Called when the best genome's topology hasn't changed for a long time,
        typically because Lamarck weight optimisation makes all structural
        mutations look harmful in the short term.  We inject a genome that has
        a different topology from the current best — either expanding the best
        with new connections / nodes, or starting fresh from the template.
        """
        from yane.evolution import smart_mutation
        strategy = random.randint(0, 2)

        if strategy == 0 and self._evaluated:
            # Expand the best genome: force-add 2–5 connections.
            base = self.get_best().copy()
            for _ in range(random.randint(2, 5)):
                smart_mutation.add_connection(base, self._tracker)

        elif strategy == 1 and self._evaluated:
            # Expand the best genome: split a connection (add_node) if possible,
            # then add a few extra connections for structural variety.
            base = self.get_best().copy()
            smart_mutation.add_node(base, self._tracker)
            for _ in range(random.randint(1, 3)):
                smart_mutation.add_connection(base, self._tracker)

        else:
            # Fresh template with random connections — maximally different topology.
            base = self._template.copy()
            n_in = len(base.input_nodes)
            n_out = len(base.output_nodes)
            for _ in range(random.randint(n_out, min(n_in * n_out, 20))):
                smart_mutation.add_connection(base, self._tracker)
            for node in base.nodes:
                for conn in node.connections:
                    conn.weight = random.gauss(0, 1.0)

        base.fitness = 0.0
        base.shared_fitness = 0.0
        self._unevaluated.append(base)
        # Do NOT reset _since_last_injection — this injection is structural, not
        # fitness-stagnation-based; the two counters are independent.

    def _inject_fresh_genome(self) -> None:
        """Escape stagnation by injecting a diverse genome.

        Four strategies (equal probability):
        0 - Best genome + forced structural expansion: add 1-3 new connections
            to the best topology so evolution can explore larger neighbourhoods.
        1 - Best genome + weight-only mutations (exploit, don't explore structure)
        2 - Template copy with random connections and re-randomised weights
        3 - Heavily mutated template (structural + weight exploration)
        """
        from yane.evolution import smart_mutation
        strategy = random.randint(0, 3)

        if strategy == 0 and self._evaluated:
            # Force-add connections to the best topology to escape structural plateaus.
            base = self.get_best().copy()
            n_add = random.randint(1, 3)
            for _ in range(n_add):
                smart_mutation.add_connection(base, self._tracker)
            for _ in range(random.randint(1, 3)):
                base.mutate(self._tracker)

        elif strategy == 1 and self._evaluated:
            # Weight-only exploration of the current best topology.
            base = self.get_best().copy()
            sigma = base.sigma_global
            for node in base.nodes:
                for conn in node.connections:
                    conn.weight += random.gauss(0, sigma)

        elif strategy == 2:
            base = self._template.copy()
            n_in = len(base.input_nodes)
            n_out = len(base.output_nodes)
            for _ in range(random.randint(n_out, min(n_in * n_out, 50))):
                smart_mutation.add_connection(base, self._tracker)
            for node in base.nodes:
                for conn in node.connections:
                    conn.weight = random.gauss(0, 1.0)

        else:
            base = self._template.copy()
            for _ in range(random.randint(3, 8)):
                base.mutate(self._tracker)

        base.fitness = 0.0
        base.shared_fitness = 0.0
        self._unevaluated.append(base)
        self._since_last_injection = 0  # pace injections; stagnation_count untouched

    def _bootstrap_initial_population(self) -> None:
        """Fill unevaluated with a diverse random initial generation.

        Each genome gets at least n_inputs random connections so every input
        has a chance to influence an output. This is the minimum needed to
        produce behavioural diversity — fewer connections often perform no
        better (or worse) than doing nothing, which kills the selection gradient.
        """
        from yane.evolution import smart_mutation
        n_in = len(self._template.input_nodes)
        n_out = len(self._template.output_nodes)
        min_conn = n_out
        max_conn = min(n_in * n_out, max(n_in, 50))
        for _ in range(self.max_size):
            g = self._template.copy()
            n_conn = random.randint(min_conn, max_conn)
            for _ in range(n_conn):
                smart_mutation.add_connection(g, self._tracker)
            self._unevaluated.append(g)

    def _spawn_offspring(self) -> None:
        if not self._evaluated:
            if not self._unevaluated:
                self._bootstrap_initial_population()
            return

        if self._since_last_injection >= self.stagnation_threshold:
            self._inject_fresh_genome()
            return

        # Structural stagnation: if the topology of the best genome hasn't changed
        # for 3× the population size, force structural exploration even when fitness
        # keeps creeping up (e.g. via Lamarck weight tuning). Without this check,
        # Lamarck causes infinite fitness drift while the structure never evolves.
        if self._topology_stagnation_count >= 3 * self.max_size:
            self._topology_stagnation_count = 0
            self._inject_structural_diversity()
            return

        self._assign_species()
        self._compute_shared_fitness()

        if self._novelty_evals_since_recompute >= max(1, self.max_size // 2):
            self._novelty_cache = self._compute_novelty()
            self._novelty_evals_since_recompute = 0
        novelty = self._novelty_cache
        nw = self.novelty_weight

        # ── Species health: penalise species that lag behind improving others ─
        # Health is only applied when there is meaningful fitness spread across
        # species (some are improving, others are not).  When all species are
        # stuck at the same fitness level (spread < 0.5), no penalty applies —
        # that would unfairly eliminate structurally diverse intermediate species
        # that simply need more time (e.g. XOR / binary-increment problems).
        #
        # When spread is large:
        #   health = 0.4 × relative_position + 0.6 × stagnation_score
        # where relative_position = 0 for worst species, 1 for best
        # and stagnation_score = 1 - stagnation_count / (3 × average_stagnation)
        # Best species always health=1.0 (never penalised).
        _genome_health: dict[int, float] = {}
        if len(self._species) > 1:
            best_sp   = max(self._species, key=lambda s: s.best_fitness_seen)
            all_best  = [sp.best_fitness_seen for sp in self._species]
            sp_max    = max(all_best)
            sp_min    = min(all_best)
            spread    = sp_max - sp_min

            if spread >= 0.5:
                # Meaningful progress difference → apply health weighting
                avg_stag = sum(sp.stagnation_count for sp in self._species) / len(self._species)
                for sp in self._species:
                    if sp is best_sp:
                        h = 1.0
                    else:
                        rel_pos  = (sp.best_fitness_seen - sp_min) / spread   # [0, 1]
                        stag_sc  = max(0.0, 1.0 - sp.stagnation_count / max(avg_stag * 3, 1))
                        h = 0.4 * rel_pos + 0.6 * stag_sc
                    for g in sp.members:
                        _genome_health[id(g)] = h
            # else: spread < 0.5 → all species at similar fitness → no health penalty

        # Tournament selection (k=3): pick k random candidates, keep the best.
        # Shift fitness to be non-negative before multiplying by offspring_factor
        # and novelty — multiplying a negative shared_fitness by offspring_factor
        # inverts the ranking (larger factor → more negative → worse), which would
        # cause selection to prefer genomes with smaller offspring_factor.
        k = min(3, len(self._evaluated))
        candidates = random.sample(self._evaluated, k)
        min_fit = min(g.shared_fitness for g in self._evaluated)
        parent = max(
            candidates,
            key=lambda g: (
                max(0.0, g.shared_fitness - min_fit + 1e-6)
                * g.offspring_factor
                * (1.0 + nw * novelty.get(id(g), 0.0))
                * _genome_health.get(id(g), 1.0)
            ),
        )

        # Note: elite protection is handled by _prune() which never removes the
        # global best or species bests. Here, even elite parents produce mutated
        # children — otherwise a small population fills with unmutated clones and
        # exploration stalls (especially critical with the empty-connection start).
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

        child.mutate(self._tracker)
        child.fitness = 0.0
        self._unevaluated.append(child)

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def _prune(self) -> None:
        if not (self._evaluated and len(self._evaluated) + len(self._unevaluated) > self.max_size):
            return

        # Compute the protected set once — we only ever remove non-protected genomes,
        # so the global best and species champions never change during this prune pass.
        # Only protect species champions of multi-member species.
        # Single-genome species don't represent "protected innovation" —
        # they're just isolated genomes that happened to form their own
        # cluster. Protecting all single-genome champions would block
        # pruning entirely when every genome is in its own species.
        protected: set[int] = {id(self.get_best())}
        for sp in self._species:
            if len(sp.members) > 1:
                protected.add(id(sp.best()))

        while self._evaluated and len(self._evaluated) + len(self._unevaluated) > self.max_size:
            if len(self._evaluated) <= 1:
                break

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
                    # Keep representative up-to-date so it doesn't hold a
                    # stale reference to the cleared genome.
                    if sp.representative is worst:
                        sp.representative = sp.best() if sp.members else None
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
