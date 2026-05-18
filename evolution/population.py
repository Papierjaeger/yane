from __future__ import annotations
import math
import random
from operator import attrgetter

import numpy as np

# Genome forward() can produce inf/overflow when output activation evolves to
# LINEAR and weights grow large — this is expected and handled by nan_to_num
# in submit(). Silence the cosmetic warnings.
np.seterr(over='ignore', invalid='ignore')

from yane.core.genome import Genome

# Module-level key functions — C-level attrgetter is ~2× faster than a Python
# lambda for attribute access in max()/min()/sorted() calls.
_fitness_key        = attrgetter('fitness')
_shared_fitness_key = attrgetter('shared_fitness')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compatibility(g1: Genome, g2: Genome, threshold: float | None = None) -> float:
    """NEAT-style compatibility distance δ = c1·E/N + c2·D/N + c3·W̄

    When innovation numbers are available, uses the original NEAT formula
    with excess (E), disjoint (D) and average weight difference (W̄) of
    matching genes.  Falls back to the old topology-count heuristic for
    genomes without innovation tracking.

    Coefficients tuned to keep δ in roughly [0, 2]:
      c1 = 1.0 (excess)   c2 = 1.0 (disjoint)   c3 = 0.4 (weight diff)

    Hot path: innovation dicts are cached per genome (invalidated on topology
    change).  Cache access is inlined here to avoid function-call overhead on
    the ~500K calls per training run.
    """
    # Inline _get_innov_cache() — avoids function-call overhead on this hot path.
    # The cache stores (innov_dict, max_innov, n_innov, key_frozenset) so the
    # frozenset is built once and reused for C-level set intersection below.
    _cache1 = g1._innov_cache
    if _cache1 is None:
        _d = {conn.innovation: conn
              for src in g1.nodes for conn in src.connections
              if conn.innovation >= 0}
        _cache1 = (_d, max(_d, default=-1), len(_d), frozenset(_d))
        g1._innov_cache = _cache1
    g1_innov, max_innov_g1, n1_innov, g1_keys = _cache1

    _cache2 = g2._innov_cache
    if _cache2 is None:
        _d = {conn.innovation: conn
              for src in g2.nodes for conn in src.connections
              if conn.innovation >= 0}
        _cache2 = (_d, max(_d, default=-1), len(_d), frozenset(_d))
        g2._innov_cache = _cache2
    g2_innov, max_innov_g2, n2_innov, g2_keys = _cache2

    if not g1_innov and not g2_innov:
        # Legacy fallback: simple topology distance
        n1, n2 = len(g1.nodes), len(g2.nodes)
        c1, c2 = g1.connection_count, g2.connection_count
        node_diff = abs(n1 - n2) / ((n1 if n1 >= n2 else n2) or 1)
        conn_diff = abs(c1 - c2) / ((c1 if c1 >= c2 else c2) or 1)
        return (node_diff + conn_diff) / 2.0

    # Structural identity fast-path: if both genomes have the same innovation
    # numbers, excess=0 and disjoint=0.  Only the weight-difference term remains.
    # Skips the frozenset intersection + two classification loops (~40% of calls
    # in stable populations where weight-only mutations dominate).
    if g1_keys == g2_keys:
        if n1_innov == 0:
            return 0.0
        weight_diff_sum = 0.0
        for k in g1_keys:
            d = g1_innov[k].weight - g2_innov[k].weight
            weight_diff_sum += d if d >= 0.0 else -d
        return 0.4 * weight_diff_sum / n1_innov

    # Inline min/max — avoids Python function-call overhead on this hot path.
    smaller_max = max_innov_g1 if max_innov_g1 <= max_innov_g2 else max_innov_g2
    N = n1_innov if n1_innov >= n2_innov else n2_innov
    if N == 0:
        N = 1

    # Use the pre-computed frozensets for C-level set intersection.
    matching_set = g1_keys & g2_keys   # C-level frozenset intersection

    matching = 0
    weight_diff_sum = 0.0
    for k in matching_set:
        matching += 1
        d = g1_innov[k].weight - g2_innov[k].weight
        weight_diff_sum += d if d >= 0.0 else -d

    excess = disjoint = 0
    for innov in g1_keys:
        if innov not in matching_set:
            if innov > smaller_max:
                excess += 1
            else:
                disjoint += 1
    for innov in g2_keys:
        if innov not in matching_set:
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
        self._cached_best: Genome | None = None  # maintained by _assign_species()

    def add(self, genome: Genome) -> None:
        self.members.append(genome)

    def best(self) -> Genome:
        # Return the incremental cache when available (set by _assign_species).
        # Falls back to a full scan for callers outside that code path (e.g. tests).
        if self._cached_best is not None:
            return self._cached_best
        return max(self.members, key=_fitness_key)

    def avg_shared_fitness(self) -> float:
        if not self.members:
            return 0.0
        return sum(g.shared_fitness for g in self.members) / len(self.members)

    def update_stagnation(self, best_genome: 'Genome | None' = None) -> None:
        """Update stagnation counter based on the current best member's fitness.

        best_genome: pre-computed best member (avoids a second max() scan when
        the caller already computed sp.best() for another purpose).
        """
        if not self.members:
            return
        current_best = (best_genome if best_genome is not None else self.best()).fitness
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

        # Lazy species-assignment: only re-assign structurally-changed genomes each
        # spawn cycle.  Weight-only mutations restore the genome to its last species
        # without running _compatibility().  Every _force_full_assign_interval spawns
        # a full rebuild is forced to prevent drift accumulation.  0 = never force.
        self._spawn_count: int = 0
        self._force_full_assign_interval: int = 50

        # Cached minimum shared_fitness — updated by _compute_shared_fitness() so
        # _spawn_offspring() can read it in O(1) instead of scanning all genomes.
        self._min_shared_fitness: float = 0.0

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

        # Incremental eval-time range tracking — updated in O(1) on each submit().
        # _compute_efficiency_scores() uses these instead of a separate min/max pass.
        # Set to None when no valid eval_time has been seen yet.
        self._eval_time_min: float | None = None
        self._eval_time_max: float | None = None
        # Dirty flag: True when the range expanded (new fastest/slowest genome) so
        # _compute_efficiency_scores() actually needs to rescale all stored scores.
        self._efficiency_dirty: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_for_evaluation(self) -> Genome:
        if not self._unevaluated:
            self._spawn_offspring()
        return self._unevaluated[0]

    def submit(self, genome: Genome, fitness: float, elapsed_ms: float | None = None) -> None:
        if genome not in self._unevaluated:
            return
        genome.fitness = fitness
        genome.shared_fitness = fitness
        genome.eval_time_ms = elapsed_ms
        genome.selection_score = 0.0
        self._unevaluated.remove(genome)
        self._evaluated.append(genome)
        # Update eval-time range and compute this genome's efficiency score in O(1).
        # Full rescale of all genomes happens in _spawn_offspring() → _compute_efficiency_scores().
        t = elapsed_ms
        if t is not None and math.isfinite(t) and t >= 0.0:
            if self._eval_time_min is None or t < self._eval_time_min:
                self._eval_time_min = t
            if self._eval_time_max is None or t > self._eval_time_max:
                self._eval_time_max = t
        old_min, old_max = self._eval_time_min, self._eval_time_max
        lo, hi = old_min, old_max
        span = (hi - lo) if (lo is not None and hi is not None) else 0.0
        if span <= 1e-9 or t is None or not math.isfinite(t):
            genome.efficiency_score = 1.0
        else:
            score = (hi - t) / span
            genome.efficiency_score = score if 0.0 <= score <= 1.0 else (1.0 if score > 1.0 else 0.0)
        # Mark all efficiency scores as needing rescale if the range changed.
        if self._eval_time_min != old_min or self._eval_time_max != old_max:
            self._efficiency_dirty = True

        # Compute behavior descriptor immediately after evaluation — the genome's
        # weights are still intact and forward() is deterministic.
        # Reset between probes so persistent hidden nodes can't carry state
        # from one probe to the next (would otherwise accumulate to ±Inf).
        if self._probe_inputs:
            rows = []
            for inp in self._probe_inputs:
                genome.reset()
                rows.append(genome.forward(inp))
            # float64 + sanitise: any NaN/Inf from a degenerate network must not
            # propagate into the novelty matrix (would break compatibility distance).
            arr = np.array(rows, dtype=np.float64).ravel()
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
            self._behaviors[id(genome)] = arr
        self._novelty_evals_since_recompute += 1

        _cc = genome.connection_count   # cache property once; avoids descriptor overhead in the two reads below
        topo = (len(genome.nodes), _cc)
        if fitness > self._best_fitness_seen:
            self._best_fitness_seen = fitness
            self._stagnation_count = 0
            self._since_last_injection = 0
            self._last_improvement_connections = _cc
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
        return max(self._evaluated, key=_fitness_key)

    def get_top(self, k: int) -> list[Genome]:
        """Return up to k best evaluated genomes, sorted by fitness descending."""
        return sorted(self._evaluated, key=_fitness_key, reverse=True)[:k]

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

    @property
    def efficiency_weight(self) -> float:
        """Selection pressure for efficient genomes.

        Efficiency matters most while task fitness is improving. During
        stagnation it fades out so larger or slower structural experiments can
        survive long enough to become useful.
        """
        stagnation_frac = self._stagnation_count / max(1, self.stagnation_threshold)
        return 0.5 * max(0.0, 1.0 - min(1.0, stagnation_frac))

    def _compute_efficiency_scores(self) -> None:
        """Normalize evaluation speed into [0, 1], where 1 is fastest.

        Uses the cached _eval_time_min/_eval_time_max tracked incrementally in
        submit(). Skipped when the range hasn't changed since the last rescale
        (common in steady-state evolution where eval times cluster tightly).
        Called once per _spawn_offspring() (not per submit).
        """
        if not self._efficiency_dirty:
            return
        self._efficiency_dirty = False
        lo, hi = self._eval_time_min, self._eval_time_max
        if lo is None or hi is None or hi - lo <= 1e-9:
            for g in self._evaluated:
                g.efficiency_score = 1.0
            return
        span = hi - lo
        for g in self._evaluated:
            t = g.eval_time_ms
            if t is None or not math.isfinite(t):
                g.efficiency_score = 1.0
            else:
                score = (hi - t) / span
                g.efficiency_score = score if 0.0 <= score <= 1.0 else (1.0 if score > 1.0 else 0.0)

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

        # Pairwise Euclidean distances using the identity:
        #   ||a - b||² = ||a||² + ||b||² - 2·(a·b)
        # This avoids materialising the (N, R, D) tensor produced by broadcasting
        # subtraction — matrix multiplication is cache-efficient and BLAS-optimised.
        N = len(pairs)
        pop_sq  = (pop_mat  * pop_mat ).sum(axis=1, keepdims=True)  # (N, 1)
        ref_sq  = (ref_mat  * ref_mat ).sum(axis=1)                  # (R,)
        cross   = pop_mat @ ref_mat.T                                 # (N, R) via BLAS
        dist_sq = pop_sq + ref_sq - 2.0 * cross
        np.maximum(dist_sq, 0.0, out=dist_sq)   # clip floating-point negatives
        dists   = np.sqrt(dist_sq)               # (N, R)
        # Mask self-distances (population vs. itself) so they don't pull the mean down.
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
            sp._cached_best = None   # reset incremental cache for this cycle

        threshold = self._compat_threshold
        # Map id(species) → species for O(1) fast-path lookup.
        species_by_id: dict[int, Species] = {id(sp): sp for sp in self._species}

        for genome in self._evaluated:
            placed = False

            # Lazy fast-path: if the genome's topology hasn't changed since its
            # last assignment (_species_stale is False), restore it to its last
            # species without running _compatibility().  Weight-only mutations
            # never set _species_stale, so this skips most compatibility calls.
            if not genome._species_stale:
                last_sp_id = genome._last_species_id
                if last_sp_id is not None:
                    sp = species_by_id.get(last_sp_id)
                    if sp is not None:
                        sp.members.append(genome)
                        if sp._cached_best is None or genome.fitness > sp._cached_best.fitness:
                            sp._cached_best = genome
                        placed = True

            if not placed:
                # Full assignment for stale or unplaced genomes.
                genome._species_stale = False

                # Fast path: try last-known species first — avoids full search.
                last_sp_id = genome._last_species_id
                if last_sp_id is not None:
                    sp = species_by_id.get(last_sp_id)
                    if sp is not None and _compatibility(genome, sp.representative) < threshold:
                        sp.members.append(genome)
                        if sp._cached_best is None or genome.fitness > sp._cached_best.fitness:
                            sp._cached_best = genome
                        placed = True

                if not placed:
                    for sp in self._species:
                        if _compatibility(genome, sp.representative) < threshold:
                            sp.members.append(genome)
                            if sp._cached_best is None or genome.fitness > sp._cached_best.fitness:
                                sp._cached_best = genome
                            genome._last_species_id = id(sp)
                            placed = True
                            break
                if not placed:
                    new_sp = Species(genome)
                    new_sp.members.append(genome)
                    new_sp._cached_best = genome
                    self._species.append(new_sp)
                    species_by_id[id(new_sp)] = new_sp
                    genome._last_species_id = id(new_sp)

        self._species = [sp for sp in self._species if sp.members]
        for sp in self._species:
            best = sp.best()    # O(1) — returns _cached_best built above
            sp.representative = best
            sp.update_stagnation(best)

        # Adapt threshold so species count converges to target_species.
        # Step 0.02: 4× larger than the original 0.005 so the species count
        # converges to target_species faster (reducing compatibility calls).
        n = len(self._species)
        if n > self._target_species:
            self._compat_threshold = min(1.5, self._compat_threshold + 0.02)
        elif n < self._target_species:
            self._compat_threshold = max(0.01, self._compat_threshold - 0.02)

    def _compute_shared_fitness(self) -> None:
        # No parsimony penalty: even a tiny coefficient causes evolution to
        # converge to empty/minimal genomes when the fitness signal is weak
        # (e.g. CartPole baseline fitness ~11 for always-push-left; 0.0001 ×
        # 4 connections = 0.0004 disadvantage makes connected genomes lose
        # tournaments against the empty baseline). Compact networks emerge
        # naturally through remove_connection/remove_node mutations when they
        # don't improve fitness.
        min_sf = float('inf')
        for sp in self._species:
            # Multiply by reciprocal — avoids one division per genome.
            inv_n = 1.0 / len(sp.members)
            for genome in sp.members:
                sf = genome.fitness * inv_n
                genome.shared_fitness = sf
                if sf < min_sf:
                    min_sf = sf
        # Cache so _spawn_offspring() can skip the O(N) min() scan.
        self._min_shared_fitness = min_sf if min_sf != float('inf') else 0.0

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

        self._spawn_count += 1
        # Periodic forced full rebuild: every _force_full_assign_interval spawns
        # mark all genomes stale so _assign_species() re-checks every genome.
        # This prevents weight-drift from permanently misassigning genomes.
        # 0 = never force a full rebuild (pure lazy assignment).
        if (self._force_full_assign_interval > 0
                and self._spawn_count % self._force_full_assign_interval == 0):
            for g in self._evaluated:
                g._species_stale = True

        self._assign_species()
        self._compute_shared_fitness()
        self._compute_efficiency_scores()

        if self._novelty_evals_since_recompute >= max(1, self.max_size // 2):
            self._novelty_cache = self._compute_novelty()
            self._novelty_evals_since_recompute = 0
        novelty = self._novelty_cache
        nw = self.novelty_weight
        ew = self.efficiency_weight

        # Species-health placeholder — disabled after benchmarking showed it
        # consistently hurts performance for discrete/XOR-type tasks.
        # When all species stagnate at the same fitness level (spread ≈ 0),
        # any differential penalty eliminates useful structural variants before
        # they can tune their weights.  The stagnation_count field is kept on
        # Species for diagnostics and future use.
        _genome_health: dict[int, float] = {}

        # Tournament selection (k=3): pick k random candidates, keep the best.
        # Shift fitness to be non-negative before multiplying by offspring_factor
        # and novelty — multiplying a negative shared_fitness by offspring_factor
        # inverts the ranking (larger factor → more negative → worse), which would
        # cause selection to prefer genomes with smaller offspring_factor.
        k = min(3, len(self._evaluated))
        candidates = random.sample(self._evaluated, k)
        min_fit = self._min_shared_fitness   # cached by _compute_shared_fitness(), O(1)
        def _selection_score(g: Genome) -> float:
            efficiency_factor = 1.0 - ew * (1.0 - g.efficiency_score)
            score = (
                max(0.0, g.shared_fitness - min_fit + 1e-6)
                * g.offspring_factor
                * (1.0 + nw * novelty.get(id(g), 0.0))
                * efficiency_factor
                * _genome_health.get(id(g), 1.0)
            )
            g.selection_score = score
            return score

        parent = max(candidates, key=_selection_score)

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
        # Store genome object references directly — avoids calling id() on every
        # genome in the candidates comprehension (was 600K+ id() calls per run).
        protected: set = {self.get_best()}
        for sp in self._species:
            if len(sp.members) > 1:
                protected.add(sp.best())

        while self._evaluated and len(self._evaluated) + len(self._unevaluated) > self.max_size:
            if len(self._evaluated) <= 1:
                break

            candidates = [g for g in self._evaluated if g not in protected]
            if not candidates:
                break

            worst = min(candidates, key=_fitness_key)
            self._behaviors.pop(id(worst), None)
            self._evaluated.remove(worst)
            # Remove from its species before clearing to release the reference
            # so the genome can be GC'd without clearing all species.
            for sp in self._species:
                if worst in sp.members:
                    sp.members.remove(worst)
                    # Invalidate incremental best cache if the removed genome was cached.
                    if sp._cached_best is worst:
                        sp._cached_best = None  # triggers full scan on next best() call
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
        self._evaluated.sort(key=_fitness_key, reverse=True)
        discarded = self._evaluated[target_size:]
        self._evaluated = self._evaluated[:target_size]
        for g in discarded:
            self._behaviors.pop(id(g), None)
            g._clear()
        self._species = []
