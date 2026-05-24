from __future__ import annotations
import math
import random
from heapq import nlargest
from operator import attrgetter

import numpy as np

# Genome forward() can produce inf/overflow when output activation evolves to
# LINEAR and weights grow large — this is expected and handled by nan_to_num
# in submit(). Silence the cosmetic warnings.
np.seterr(over='ignore', invalid='ignore')

from yane.core.genome import Genome
from yane.evolution.mutation import Mutation
from yane.evolution.species import Species  # noqa: F401 (re-exported for tests)
from yane.evolution.compatibility import (  # noqa: F401 (re-exported for tests/genome.py)
    compatibility as _compatibility,
    _COMPAT_NUMPY_THRESHOLD,
)

# Module-level key functions — C-level attrgetter is ~2× faster than a Python
# lambda for attribute access in max()/min()/sorted() calls.
_fitness_key        = attrgetter('fitness')
_shared_fitness_key = attrgetter('shared_fitness')


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
        # Elitism: how many top genomes to protect from pruning globally and per species.
        # elite_count=1 preserves the global best; species_elite_count=1 preserves the
        # best of each species (regardless of species size). Elites are never deleted but
        # CAN still be selected as parents — only mutated copies leave the evaluated pool.
        self.elite_count: int = 1
        self.species_elite_count: int = 1
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
        # Speciation metric: "topology" (default) or "topology_no_disabled"
        self._speciation_metric: str = "topology"
        # Small species protection: don't remove species younger than this (in spawn cycles).
        self._min_species_age: int = max_size       # default: 1 full generation
        self._min_species_size: int = 2             # species with fewer members are protected
        # Diagnostics: track last threshold adjustment for debugging.
        self._dbg_last_adj_n: int = -1       # n at last _assign_species() threshold check
        self._dbg_adj_count: int = 0         # how many times threshold was actually changed

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

        # Offspring counters — cumulative over the population's lifetime.
        self._n_crossover: int = 0
        self._n_mutation_only: int = 0
        self._n_diversity_injection: int = 0
        self._n_interspecies_crossover: int = 0
        self._species_spawn_credit: dict[int, float] = {}
        self._species_offspring_total: dict[int, int] = {}
        self._last_species_spawn_scores: list[dict[str, float | int]] = []

        # Mutation success tracking — per-type counters for adaptive weighting.
        # _mutation_success[type] = count where mutation type improved fitness
        # _mutation_total[type]  = count where mutation type was applied
        self._mutation_success: dict[str, int] = {}
        self._mutation_total: dict[str, int] = {}

        # Fitness shaping: when enabled, shared_fitness values are replaced by
        # rank-normalised scores in [1/N, 1] before tournament selection.
        self._fitness_shaping: bool = False

        # Interspecies crossover: probability that the second parent comes from
        # a *different* species.  0.0 = intraspecies only (default).
        self._interspecies_crossover_rate: float = 0.0

        # Best-genome topology history — one entry per fitness improvement.
        # Each entry: (total_submitted, n_nodes, n_connections, fitness)
        self._topology_history_max: int = 200
        self._best_topology_history: list[tuple[int, int, int, float]] = []
        self._total_submitted: int = 0
        self._n_new_best: int = 0  # how many submits beat the previous best fitness

        # Weight/bias clipping — None = disabled (default).
        # When set to (w_max, b_max), all weights and biases are clamped to
        # [-w_max, w_max] and [-b_max, b_max] after each mutation.
        self._weight_clip: tuple[float, float] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_for_evaluation(self) -> Genome:
        if not self._unevaluated:
            self._spawn_offspring()
        return self._unevaluated[0]

    def submit(self, genome: Genome, fitness: float, elapsed_ms: float | None = None) -> None:
        if genome not in self._unevaluated:
            raise ValueError("Genome was not selected for evaluation or already submitted.")
        self._total_submitted += 1
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

        # Assign to species immediately while fitness and topology are fresh.
        # _prune() (called below) handles removal from sp.members if the genome
        # turns out to be the worst and gets evicted right away.
        if genome._species_stale:
            self._assign_one_genome(genome)

        _cc = genome.connection_count   # cache property once; avoids descriptor overhead in the two reads below
        topo = (len(genome.nodes), _cc)
        if fitness > self._best_fitness_seen:
            self._n_new_best += 1
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
            # Record topology at each fitness improvement.
            self._best_topology_history.append(
                (self._total_submitted, len(genome.nodes), _cc, fitness)
            )
            if len(self._best_topology_history) > self._topology_history_max:
                self._best_topology_history = self._best_topology_history[-self._topology_history_max:]
        else:
            self._stagnation_count += 1
            self._since_last_injection += 1
            self._topology_stagnation_count += 1

        # ── Mutation success tracking ─────────────────────────────────────
        # Track which mutation types led to fitness improvements.
        # "Improvement" = child fitness >= parent fitness.
        parent_fit = getattr(genome, '_parent_fitness', None)
        improved = parent_fit is not None and fitness >= parent_fit
        for mt in getattr(genome, '_mutation_types_fired', []):
            self._mutation_total[mt] = self._mutation_total.get(mt, 0) + 1
            if improved:
                self._mutation_success[mt] = self._mutation_success.get(mt, 0) + 1
            # Per-species tracking: find the genome's species and update.
            for sp in self._species:
                if genome in sp.members:
                    sp._mut_total[mt] = sp._mut_total.get(mt, 0) + 1
                    if improved:
                        sp._mut_success[mt] = sp._mut_success.get(mt, 0) + 1
                    break

        self._prune()

    def reset_stagnation(self) -> None:
        """Reset stagnation counters after a curriculum stage switch.

        Moves all evaluated genomes to the front of the unevaluated queue so
        known-good genomes are re-assessed under the new stage before fresh
        offspring can dominate diagnostics.  Species assignments
        are kept intact.  The stagnation and best-fitness-seen counters are
        cleared so the new stage starts from a clean baseline.
        """
        self._unevaluated = self._evaluated + self._unevaluated
        self._evaluated.clear()
        self._stagnation_count = 0
        self._topology_stagnation_count = 0
        self._since_last_injection = 0
        self._best_fitness_seen = float('-inf')
        self._n_new_best = 0
        self._novelty_cache.clear()
        self._novelty_evals_since_recompute = 0

    def get_best(self) -> Genome:
        if not self._evaluated:
            raise RuntimeError("No evaluated genomes yet.")
        return max(self._evaluated, key=_fitness_key)

    def get_top(self, k: int) -> list[Genome]:
        """Return up to k best evaluated genomes, sorted by fitness descending."""
        return sorted(self._evaluated, key=_fitness_key, reverse=True)[:k]

    def worst_fitness(self) -> float | None:
        """Return the fitness of the worst evaluated genome, or None if pool is empty."""
        if not self._evaluated:
            return None
        return min(g.fitness for g in self._evaluated)

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

    def get_species_for_genome(self, genome: Genome) -> Species | None:
        """Return the Species that *genome* currently belongs to, or None.

        Uses the cached ``_last_species_id`` for O(species) id-match. Returns
        None if the genome was never assigned or its species was removed.
        """
        last_id = genome._last_species_id
        if last_id is not None:
            for sp in self._species:
                if id(sp) == last_id:
                    return sp
        return None

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

    def _place_or_cap(self, genome: Genome) -> 'Species | None':
        """Create a new species for genome, or force-assign to nearest if at the hard cap.

        Hard cap = max(target_species, max_size // 2).  With species_elite_count=1
        each species consumes one protected slot in _prune().  Allowing species_count
        to approach max_size means every genome is a species-elite → no pruning
        candidates → population grows without bound.

        Returns the newly created Species, or None if genome was force-assigned.
        """
        cap = max(self._effective_target_species(), self.max_size // 2)
        _only_enabled = self._speciation_metric == "topology_no_disabled"
        if self._species and len(self._species) >= cap:
            # Force-assign to nearest existing species by compatibility distance.
            nearest = min(
                self._species,
                key=lambda sp: (
                    _compatibility(genome, sp.representative, _only_enabled)
                    if sp.representative is not None else float('inf')
                ),
            )
            nearest.members.append(genome)
            if nearest._cached_best is None or genome.fitness > nearest._cached_best.fitness:
                nearest._cached_best = genome
            genome._last_species_id = id(nearest)
            return None
        else:
            new_sp = Species(genome, spawn_count=self._spawn_count, parent_id=None)
            new_sp.members.append(genome)
            new_sp._cached_best = genome
            self._species.append(new_sp)
            genome._last_species_id = id(new_sp)
            return new_sp

    def _effective_target_species(self) -> int:
        """Return the effective target species count.

        When ``_target_species <= 0``, auto-compute from population size:
        ``max(2, int(sqrt(max_size)))``. Otherwise return the configured value.
        """
        if self._target_species <= 0:
            return max(2, int(self.max_size ** 0.5))
        return self._target_species

    def _assign_one_genome(self, genome: Genome) -> None:
        """Assign a single genome to its species immediately after evaluation.

        Called from submit() so sp.members stays up-to-date without waiting
        for the next _assign_species() call.  O(species) per call.

        The genome's _species_stale flag is cleared here.  _prune() correctly
        removes genomes that are evicted right after submission.
        """
        threshold = self._compat_threshold
        _only_enabled = self._speciation_metric == "topology_no_disabled"
        placed = False

        # Try last-known species first (O(1) id-match with small species list).
        last_id = genome._last_species_id
        if last_id is not None:
            for sp in self._species:
                if id(sp) == last_id:
                    if (sp.representative is not None
                            and _compatibility(genome, sp.representative, _only_enabled) < threshold):
                        sp.members.append(genome)
                        if sp._cached_best is None or genome.fitness > sp._cached_best.fitness:
                            sp._cached_best = genome
                        placed = True
                    break   # found the remembered species — stop searching even if not placed

        if not placed:
            for sp in self._species:
                if (sp.representative is not None
                        and _compatibility(genome, sp.representative, _only_enabled) < threshold):
                    sp.members.append(genome)
                    if sp._cached_best is None or genome.fitness > sp._cached_best.fitness:
                        sp._cached_best = genome
                    genome._last_species_id = id(sp)
                    placed = True
                    break

        if not placed:
            self._place_or_cap(genome)

        genome._species_stale = False

    def _assign_species(self) -> None:
        """Assign evaluated genomes to species.

        Steady-state path (most calls): O(species) — sp.members is already
        up-to-date because submit() calls _assign_one_genome() for every new
        genome.  Only stale genomes (topology changed after last assignment, which
        does not happen to genomes in _evaluated) need re-processing.

        Periodic full-rebuild path (every _force_full_assign_interval spawns):
        clears and rebuilds sp.members for all genomes to prevent any drift that
        could accumulate from the incremental path over many generations.
        """
        threshold = self._compat_threshold
        _only_enabled = self._speciation_metric == "topology_no_disabled"

        if (self._force_full_assign_interval > 0
                and self._spawn_count % self._force_full_assign_interval == 0):
            # Full rebuild: clear all member lists and re-assign every genome.
            for sp in self._species:
                sp.members = []
                sp._cached_best = None

            species_by_id: dict[int, Species] = {id(sp): sp for sp in self._species}

            for genome in self._evaluated:
                genome._species_stale = False
                placed = False

                last_sp_id = genome._last_species_id
                if last_sp_id is not None:
                    sp = species_by_id.get(last_sp_id)
                    if sp is not None and _compatibility(genome, sp.representative, _only_enabled) < threshold:
                        sp.members.append(genome)
                        if sp._cached_best is None or genome.fitness > sp._cached_best.fitness:
                            sp._cached_best = genome
                        placed = True

                if not placed:
                    for sp in self._species:
                        if _compatibility(genome, sp.representative, _only_enabled) < threshold:
                            sp.members.append(genome)
                            if sp._cached_best is None or genome.fitness > sp._cached_best.fitness:
                                sp._cached_best = genome
                            genome._last_species_id = id(sp)
                            placed = True
                            break

                if not placed:
                    new_sp = self._place_or_cap(genome)
                    if new_sp is not None:
                        species_by_id[id(new_sp)] = new_sp

        else:
            # Incremental path: only re-process genuinely stale genomes.
            # In steady state this list is empty — evaluated genomes never change
            # topology after evaluation, so _species_stale stays False.
            stale = [g for g in self._evaluated if g._species_stale]
            for genome in stale:
                # Remove from current species before re-assigning.
                old_id = genome._last_species_id
                if old_id is not None:
                    for sp in self._species:
                        if id(sp) == old_id:
                            if genome in sp.members:
                                sp.members.remove(genome)
                                if sp._cached_best is genome:
                                    sp._cached_best = None
                            break

                genome._species_stale = False
                placed = False
                for sp in self._species:
                    if (sp.members and sp.representative is not None
                            and _compatibility(genome, sp.representative, _only_enabled) < threshold):
                        sp.members.append(genome)
                        if sp._cached_best is None or genome.fitness > sp._cached_best.fitness:
                            sp._cached_best = genome
                        genome._last_species_id = id(sp)
                        placed = True
                        break

                if not placed:
                    self._place_or_cap(genome)

        self._species = [sp for sp in self._species if sp.members]
        for sp in self._species:
            best = sp.best()    # O(1) via _cached_best
            sp.representative = best
            sp.update_stagnation(best)

        # Active species merging: if two representatives are closer than the
        # threshold they belong to the same species.  Stop once at target count.
        n_active = len(self._species)
        changed = True
        while changed and n_active > self._effective_target_species():
            changed = False
            for i in range(len(self._species)):
                if not self._species[i].members:
                    continue
                for j in range(i + 1, len(self._species)):
                    if not self._species[j].members:
                        continue
                    sp_a, sp_b = self._species[i], self._species[j]
                    if _compatibility(sp_a.representative, sp_b.representative, _only_enabled) < threshold:
                        bigger, smaller = (
                            (sp_a, sp_b) if len(sp_a.members) >= len(sp_b.members)
                            else (sp_b, sp_a)
                        )
                        for g in smaller.members:
                            g._last_species_id = id(bigger)
                        bigger.members.extend(smaller.members)
                        if (smaller._cached_best is not None and (
                                bigger._cached_best is None or
                                smaller._cached_best.fitness > bigger._cached_best.fitness)):
                            bigger._cached_best = smaller._cached_best
                        bigger.representative = bigger._cached_best or bigger.representative
                        bigger.merge_count += 1
                        smaller.members = []
                        n_active -= 1
                        changed = True
                        break
                if changed:
                    break
        self._species = [sp for sp in self._species if sp.members]

        # Adapt threshold so species count converges to target_species.
        # Dead-band ±1 prevents oscillation after a merge hits the target.
        #
        # Step is calibrated for per-evaluation frequency: _assign_species() runs
        # once per genome evaluation in steady state, so ~max_size times per
        # generation.  0.3/max_size ≈ 0.3 per generation, matching the original
        # NEAT recommendation.  The old hardcoded 0.02 caused ~2.0 swing per
        # injection cycle (100 evals × 0.02), sweeping the entire threshold range
        # and causing wild oscillation between sp=1 and sp=many.
        n = len(self._species)
        self._dbg_last_adj_n = n
        _step = max(0.001, 0.3 / max(1, self.max_size))
        if n > self._effective_target_species():
            self._compat_threshold = min(1.5, self._compat_threshold + _step)
            self._dbg_adj_count += 1
        elif n < self._effective_target_species():
            self._compat_threshold = max(0.01, self._compat_threshold - _step)
            self._dbg_adj_count += 1

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

    def _apply_fitness_shaping(self) -> None:
        """Replace shared_fitness values with rank-normalised scores [1/N, 1].

        Rank 1 = worst, rank N = best.  The scores are in (0, 1] so the
        existing tournament selection formula ``max(0, sf - min_sf + 1e-6)``
        produces a linear spread from ~0 to ~(N-1)/N regardless of raw fitness
        scale — selection pressure stays constant across tasks with very
        different fitness ranges.
        """
        evaluated = self._evaluated
        n = len(evaluated)
        if n < 2:
            return
        sorted_genomes = sorted(evaluated, key=_shared_fitness_key)
        for rank, genome in enumerate(sorted_genomes, 1):
            genome.shared_fitness = rank / n
        self._min_shared_fitness = 1.0 / n

    # ------------------------------------------------------------------
    # Offspring spawning
    # ------------------------------------------------------------------

    def _apply_mutation_success_weights(self) -> None:
        """Adjust mutation bool_rates based on historical success.

        Called once per spawn cycle. Mutation types with high success rates
        get gentle rate boosts; low-success types are dampened. The update
        is deliberately slow (momentum 0.9) to avoid oscillation.
        """
        if not self._evaluated:
            return
        base = self._evaluated[0]  # use first genome's rates as reference
        for mt, gene_attr in (
            ("add_node", "mutation_add_node"),
            ("remove_node", "mutation_remove_node"),
            ("add_connection", "mutation_add_connection"),
            ("remove_connection", "mutation_remove_connection"),
            ("rewire", "mutation_rewire"),
            ("disable", "mutation_disable_connection"),
            ("enable", "mutation_enable_connection"),
        ):
            total = self._mutation_total.get(mt, 0)
            if total < 10:  # not enough data yet
                continue
            success_rate = self._mutation_success.get(mt, 0) / total
            # Target rate: scale around current rate based on success.
            # Success > 50% → boost (up to 2×), success < 10% → dampen (down to 0.5×).
            target_mult = 0.5 + 1.5 * max(0.0, min(1.0, success_rate))
            for genome in self._evaluated:
                gene = getattr(genome, gene_attr)
                # Gentle momentum update per genome
                gene.bool_rate = gene.bool_rate * 0.9 + gene.bool_rate * target_mult * 0.1
                # Clamp to Mutation's valid range
                gene.bool_rate = max(Mutation.MIN_RATE, min(0.999, gene.bool_rate))

    def _apply_species_mutation_biases(self) -> None:
        """Update per-species mutation biases based on species-level success.

        Each species develops its own mutation tendency profile. Successful
        mutation types get boosted; unsuccessful ones are dampened.
        """
        for sp in self._species:
            if not sp.members:
                continue
            for mt in sp.mutation_biases:
                total = sp._mut_total.get(mt, 0)
                if total < 5:  # not enough data
                    continue
                success_rate = sp._mut_success.get(mt, 0) / total
                target = 0.5 + 1.5 * max(0.0, min(1.0, success_rate))
                # Gentle momentum toward target
                sp.mutation_biases[mt] = sp.mutation_biases[mt] * 0.9 + target * 0.1
                # Clamp to reasonable range
                sp.mutation_biases[mt] = max(0.3, min(3.0, sp.mutation_biases[mt]))

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
        self._n_diversity_injection += 1
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
        self._n_diversity_injection += 1
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

    def _species_spawn_scores(self) -> list[tuple[Species, float]]:
        """Return positive spawn scores for active species.

        Score per species = avg_shared_fitness (shifted >= 0)
                           * stagnation_factor (0.1-1.0)
                           * youth_bonus       (1.0-2.0 for new species)

        A floor derived from the active species count acts as a minimum budget:
        weak species shrink, but they do not disappear from reproduction solely
        because one current champion species is much fitter.
        """
        active = [sp for sp in self._species if sp.members]
        if not active:
            return []
        avg_fits = [sp.avg_shared_fitness() for sp in active]
        min_avg = min(avg_fits)
        max_avg = max(avg_fits)
        span = max_avg - min_avg
        raw_scores: list[tuple[Species, float]] = []
        result: list[tuple[Species, float]] = []
        for sp, avg_fit in zip(active, avg_fits):
            quality = 1.0 if span <= 1e-12 else max(0.0, (avg_fit - min_avg) / span)
            stag_factor = max(0.1, 1.0 - sp.stagnation_count / max(1, self.stagnation_threshold))
            age = self._spawn_count - sp.created_at_spawn
            youth_bonus = 1.0 + max(0.0, 1.0 - age / max(1, self._min_species_age))
            raw_scores.append((sp, quality * stag_factor * youth_bonus))
        max_raw = max(score for _sp, score in raw_scores)
        floor = max(1e-6, 0.05 * max_raw / len(active))
        for sp, score in raw_scores:
            result.append((sp, max(floor, score)))
        return result

    def _select_parent_species(self) -> Species | None:
        """Pick which species contributes the next offspring.

        Uses weighted fair queuing rather than independent roulette picks.  Each
        call adds the species' normalized spawn share to a rolling credit bucket;
        the species with the largest credit spawns and spends one credit.  This
        gives good species proportionally more offspring while guaranteeing that
        low-score and young protected species receive their minimum budget over
        time.
        """
        scored = self._species_spawn_scores()
        if not scored:
            return None
        if len(scored) == 1:
            sp = scored[0][0]
            self._species_offspring_total[id(sp)] = self._species_offspring_total.get(id(sp), 0) + 1
            return sp

        active_ids = {id(sp) for sp, _score in scored}
        self._species_spawn_credit = {
            sid: credit
            for sid, credit in self._species_spawn_credit.items()
            if sid in active_ids
        }
        total = sum(score for _sp, score in scored)
        if total <= 0.0:
            share = 1.0 / len(scored)
            for sp, _score in scored:
                self._species_spawn_credit[id(sp)] = self._species_spawn_credit.get(id(sp), 0.0) + share
        else:
            for sp, score in scored:
                self._species_spawn_credit[id(sp)] = (
                    self._species_spawn_credit.get(id(sp), 0.0) + score / total
                )

        chosen, _score = max(
            scored,
            key=lambda item: (
                self._species_spawn_credit.get(id(item[0]), 0.0),
                item[1],
                item[0].best().fitness,
            ),
        )
        chosen_id = id(chosen)
        self._species_spawn_credit[chosen_id] = self._species_spawn_credit.get(chosen_id, 0.0) - 1.0
        self._species_offspring_total[chosen_id] = self._species_offspring_total.get(chosen_id, 0) + 1
        self._last_species_spawn_scores = [
            {
                "species_id": id(sp),
                "members": len(sp.members),
                "score": score,
                "share": score / total if total > 0.0 else 1.0 / len(scored),
                "credit": self._species_spawn_credit.get(id(sp), 0.0),
                "offspring_total": self._species_offspring_total.get(id(sp), 0),
            }
            for sp, score in scored
        ]
        return chosen

    def _spawn_offspring(self) -> None:
        if not self._evaluated:
            if not self._unevaluated:
                self._bootstrap_initial_population()
            return

        # Always run species management — skipping it during stagnation injection
        # causes unbounded species growth (no merging/threshold adjustment) which
        # in turn makes every genome a species-elite and breaks _prune().
        self._spawn_count += 1
        self._assign_species()

        # Safety net: if all species have gone extinct (e.g., all pruned),
        # rescue the population by creating a fresh species from the best genome.
        if not self._species and self._evaluated:
            best = max(self._evaluated, key=_fitness_key)
            rescue_sp = Species(best, spawn_count=self._spawn_count)
            rescue_sp.members = list(self._evaluated)
            rescue_sp._cached_best = best
            for g in self._evaluated:
                g._last_species_id = id(rescue_sp)
            self._species = [rescue_sp]

        # Periodically adjust mutation rates based on success history.
        if self._spawn_count % max(1, self.max_size // 2) == 0:
            self._apply_mutation_success_weights()
            self._apply_species_mutation_biases()

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
        self._compute_shared_fitness()
        if self._fitness_shaping:
            self._apply_fitness_shaping()
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

        # Species-budget selection: choose which species spawns next (proportional
        # to avg shared fitness × stagnation_factor × youth_bonus), then run
        # tournament within that species only.  Falls back to global pool when
        # no species is assigned yet or the selected species has no members.
        _parent_sp = self._select_parent_species()
        _pool = (_parent_sp.members
                 if _parent_sp is not None and _parent_sp.members
                 else self._evaluated)
        k = min(3, len(_pool))
        candidates = random.sample(_pool, k)
        parent = max(candidates, key=_selection_score)

        # Note: elite protection is handled by _prune() which never removes the
        # global best or species bests. Here, even elite parents produce mutated
        # children — otherwise a small population fills with unmutated clones and
        # exploration stalls (especially critical with the empty-connection start).
        if random.random() < parent.crossover_prob and len(self._evaluated) >= 2:
            # Interspecies crossover: with a small probability pick the second
            # parent from a *different* species to combine structural innovations.
            use_interspecies = (
                self._interspecies_crossover_rate > 0.0
                and len(self._species) >= 2
                and random.random() < self._interspecies_crossover_rate
            )
            if use_interspecies:
                # Find parent's species, then pick from any other species.
                parent_sp = next(
                    (sp for sp in self._species if parent in sp.members),
                    None,
                )
                other_species_members = [
                    g for sp in self._species
                    if sp is not parent_sp
                    for g in sp.members
                    if g is not parent
                ]
                candidates = other_species_members or [g for g in self._evaluated if g is not parent]
                other = random.choice(candidates)
                self._n_interspecies_crossover += 1
            else:
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
            self._n_crossover += 1
        else:
            child = parent.copy()
            self._n_mutation_only += 1

        # Apply per-species mutation biases before mutating.
        # Find parent's species and adjust mutation rates accordingly.
        parent_sp = next((sp for sp in self._species if parent in sp.members), None)
        if parent_sp is not None:
            for mt, gene_attr in (
                ("add_node", "mutation_add_node"),
                ("remove_node", "mutation_remove_node"),
                ("add_connection", "mutation_add_connection"),
                ("remove_connection", "mutation_remove_connection"),
                ("rewire", "mutation_rewire"),
                ("disable", "mutation_disable_connection"),
                ("enable", "mutation_enable_connection"),
            ):
                bias = parent_sp.mutation_biases.get(mt, 1.0)
                gene = getattr(child, gene_attr)
                gene.bool_rate = max(Mutation.MIN_RATE, min(0.999, gene.bool_rate * bias))

        child.mutate(self._tracker)
        if self._weight_clip is not None:
            w_max, b_max = self._weight_clip
            for node in child.nodes:
                node.bias = max(-b_max, min(b_max, node.bias))
                for conn in node.connections:
                    conn.weight = max(-w_max, min(w_max, conn.weight))
        child._parent_fitness = parent.fitness
        child.fitness = 0.0
        self._unevaluated.append(child)

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def _prune(self) -> None:
        if not (self._evaluated and len(self._evaluated) + len(self._unevaluated) > self.max_size):
            return

        # Build the protected set from configured elite counts.
        #
        # Global elites: top elite_count genomes by fitness — never deleted.
        # Species elites: best species_elite_count genomes per species — never deleted.
        #   Applies to all species regardless of size; single-genome species are
        #   protected too. Degenerate case (every genome in its own species) is
        #   prevented in practice by _target_species / adaptive compat threshold.
        #
        # Elites are preserved in _evaluated across all pruning passes. They can
        # still be selected as parents and produce mutated children — only copies
        # ever leave the evaluated pool; the elite object itself is never modified.
        protected: set = set()
        if self.elite_count > 0:
            for g in nlargest(self.elite_count, self._evaluated, key=_fitness_key):
                protected.add(g)
        if self.species_elite_count > 0:
            for sp in self._species:
                if not sp.members:
                    continue
                # Protect all members of very young or very small species —
                # gives them time to establish before facing pruning pressure.
                species_age = self._spawn_count - sp.created_at_spawn
                if species_age < self._min_species_age or len(sp.members) < self._min_species_size:
                    protected.update(sp.members)
                    continue
                # Single-member species (typically fresh injection genomes) are only
                # protected when the global elite count doesn't already cover them.
                # This prevents bad-fitness injection genomes from surviving indefinitely
                # simply because they happen to be in their own species.
                if len(sp.members) < 2 and sp.best() not in protected:
                    continue
                if self.species_elite_count == 1:
                    # sp.best() uses _cached_best in O(1) when available (set by
                    # _assign_species), avoiding a full O(n) scan per species.
                    protected.add(sp.best())
                else:
                    for g in nlargest(self.species_elite_count, sp.members, key=_fitness_key):
                        protected.add(g)

        while self._evaluated and len(self._evaluated) + len(self._unevaluated) > self.max_size:
            if len(self._evaluated) <= 1:
                break

            candidates = [g for g in self._evaluated if g not in protected]
            if not candidates:
                # Emergency: species elites fill all slots (species explosion).
                # Fall back to protecting only the global elite so pruning can proceed.
                if self.elite_count > 0:
                    fallback = set(nlargest(self.elite_count, self._evaluated, key=_fitness_key))
                else:
                    fallback = set()
                candidates = [g for g in self._evaluated if g not in fallback]
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
