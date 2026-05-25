from __future__ import annotations
import gc
import json
import random
import time
from pathlib import Path
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
from yane.evolution.fitness_sanitizer import FitnessSanitizer, sanitize_fitness  # noqa: F401
from yane.evolution.lamarck_refiner import LamarckRefiner
from yane.evolution.diagnostics import build_population_info, _fitness_iqr as _compute_fitness_iqr
from yane.evolution import checkpoint as _ckpt
from yane.evolution.evaluation import (  # noqa: F401  (EvaluationResult re-exported for GUI)
    EvaluationResult,
    EvaluationRunner,
    aggregate_fitnesses,
)
from yane.evolution.fitness_finalization import (
    FitnessFinalizationConfig,
    finalize_fitness_value,
)
from yane.evolution.quality_diversity import MAPElitesArchive
from yane.util.resource_guard import ResourceGuard
from yane.evolution.curriculum import Curriculum, CurriculumStage  # noqa: F401
from yane.evolution.adaptive_controller import AdaptiveController
from yane.evolution.operator_scheduler import OperatorScheduler
from yane.evolution.descriptors import AdaptiveFitnessComponentWeights, FitnessComponent
from yane.evolution.meta_adaptive import MetaAdaptivePolicyEvolver, PolicyGeneBounds, PolicyGenes
from yane.evolution.modularity import ModuleLibrary

# Re-export for gui/worker.py backwards compatibility
_aggregate_fitnesses = aggregate_fitnesses


def _derive_run_name(fitness_fn: Callable) -> str:
    """Derive a log category name from a fitness function.

    Uses ``fitness_fn.__name__`` unless it is a lambda or other unhelpful
    name — then falls back to ``"training"``.
    """
    raw = getattr(fitness_fn, '__name__', None)
    if raw and raw != '<lambda>':
        return raw
    return "training"




class NeuroEvolution:
    def __init__(self, seed: int | None = None) -> None:
        self._population: Population | None = None
        self.min_fitness: float | None = None
        self.max_iterations: int | None = None
        self._seed: int | None = seed
        self._current_genome: Genome | None = None
        self._efficiency_penalty: EfficiencyPenalty | None = None
        self._complexity_penalty_nodes: float = 0.0
        self._complexity_penalty_connections: float = 0.0
        self._resource_guard = ResourceGuard()
        self._resource_check_interval: int = 50  # check psutil every N iters (was 1 = ~5% overhead)
        self._population_size: int = 100
        self._adaptive_pop_enabled: bool = False
        self._adaptive_pop_min: int = 100
        self._adaptive_pop_max: int = 100
        self._adaptive_pop_rate: float = 0.05
        self._interspecies_crossover_mode: str = "fixed"
        self._interspecies_crossover_rate: float = 0.0
        self._interspecies_crossover_min: float = 0.0
        self._interspecies_crossover_max: float = 0.2
        self._n_workers: int = 1
        # Lamarckian refinement
        self._lamarck = LamarckRefiner()
        self._lamarck_eligible_species_indices: list[int] | None = None

        # Adaptive Control Layer
        self._adaptive_ctrl: AdaptiveController = AdaptiveController()
        self._adaptive_ctrl_enabled: bool = False

        # Operator Scheduler
        self._operator_scheduler: OperatorScheduler = OperatorScheduler()
        self._operator_scheduler_enabled: bool = False
        self._fitness_component_weights: AdaptiveFitnessComponentWeights | None = None
        self._meta_adaptive: MetaAdaptivePolicyEvolver | None = None
        self._meta_adaptive_enabled: bool = False
        self._module_library: ModuleLibrary | None = None
        self._module_insert_rate: float = 0.0
        # Multi-eval + early stopping
        self._runner = EvaluationRunner()
        # Elitism (applied to population on configure())
        self._elite_count: int = 1
        self._species_elite_count: int = 1
        # Fitness sanitizing (disabled by default)
        self._sanitizer = FitnessSanitizer()
        # Output sanitizing — replaces NaN/Inf in forward() results (disabled by default)
        self._output_sanitize: bool = False
        self._output_fallback: float = 0.0
        # Cached configure() parameters for logging / introspection.
        self._n_inputs: int = 0
        self._n_outputs: int = 0
        self._max_nodes: int | None = None
        self._max_connections: int | None = None
        self._n_initial_hidden: int = 0
        self._stateful: bool = True
        # Structured logging
        self._log_run_dir: Path | None = None
        self._log_run_name: str | None = None
        # Convergence detection
        self._convergence_spread_eps: float | None = None   # IQR threshold for convergence
        self._convergence_min_stagnation: float = 1.0       # stagnation fraction required
        self._max_evaluations: int | None = None
        self._n_evaluations_done: int = 0
        # Curriculum learning
        self._curriculum: "Curriculum | None" = None
        self._on_stage_advance: Callable | None = None
        from yane.evolution.innovation import InnovationTracker
        self._tracker = InnovationTracker()
        self._normalizer = None
        self._multi_objective_enabled: bool = False
        self._multi_objective_weights: tuple[float, ...] | None = None
        self._multi_objective_maximize: tuple[bool, ...] | None = None
        self._qd_enabled: bool = False
        self._qd_archive: MAPElitesArchive | None = None
        self._qd_descriptor_fn = None
        self._qd_descriptor_needs_reattach: bool = False
        # Matrix-accelerated forward pass
        self._matrix_forward_enabled: bool = False
        self._matrix_cache = None   # MatrixForwardCache, lazy-imported
        self._matrix_hits: int = 0
        self._matrix_misses: int = 0
        # Structured logging options
        self._log_format: str = "csv"           # "csv", "jsonlines", or "both"
        self._tensorboard_logdir: Path | None = None
        self._tensorboard_writer = None         # SummaryWriter, created lazily
        self._log_callbacks: list[Callable] = []

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

        self._apply_seed()

        # Cache for logging / introspection.
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._max_nodes = max_nodes
        self._max_connections = max_connections
        self._n_initial_hidden = n_initial_hidden
        self._stateful = stateful

        tracker = self._tracker
        initial = Genome()
        initial.max_nodes = max_nodes
        initial.max_connections = max_connections
        initial.allow_memory = stateful
        initial._output_sanitize = self._output_sanitize
        initial._output_fallback = self._output_fallback

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
        self._population._adaptive_pop_enabled = self._adaptive_pop_enabled
        self._population._adaptive_pop_min = self._adaptive_pop_min
        self._population._adaptive_pop_max = self._adaptive_pop_max
        self._population._adaptive_pop_rate = self._adaptive_pop_rate
        self._population.configure_interspecies_crossover(
            self._interspecies_crossover_rate,
            mode=self._interspecies_crossover_mode,
            min_rate=self._interspecies_crossover_min,
            max_rate=self._interspecies_crossover_max,
        )
        self._population._multi_objective_enabled = self._multi_objective_enabled
        self._population._multi_objective_maximize = self._multi_objective_maximize
        self._population._qd_enabled = self._qd_enabled
        self._population._qd_archive = self._qd_archive
        self._population._qd_descriptor_fn = self._qd_descriptor_fn
        # Wire operator scheduler
        self._population._operator_scheduler = (
            self._operator_scheduler if self._operator_scheduler_enabled else None
        )
        self._population._module_library = self._module_library
        self._population._module_insert_rate = self._module_insert_rate
        self._qd_enabled = getattr(self._population, "_qd_enabled", self._qd_enabled)
        self._qd_archive = getattr(self._population, "_qd_archive", self._qd_archive)
        self._qd_descriptor_fn = getattr(self._population, "_qd_descriptor_fn", self._qd_descriptor_fn)
        self._qd_descriptor_needs_reattach = bool(self._qd_enabled and self._qd_descriptor_fn is None)
        if self._qd_descriptor_needs_reattach:
            import warnings
            warnings.warn(
                "Quality Diversity archive was restored, but its descriptor callback "
                "must be reattached with set_quality_diversity().",
                RuntimeWarning,
                stacklevel=2,
            )
        self._population._qd_enabled = self._qd_enabled
        self._population._qd_archive = self._qd_archive
        self._population._qd_descriptor_fn = self._qd_descriptor_fn

    def set_seed(self, seed: int | None) -> None:
        """Set or clear the random seed.

        Seeds Python's ``random`` module and NumPy's global RNG so that network
        initialisation and evolutionary operators are reproducible.

        Note: gym environment seeds are not controlled by YANE — call
        ``env.reset(seed=...)`` inside your fitness function for full
        reproducibility with stochastic environments.

        The seed is applied at the start of every ``configure()`` call.
        """
        self._seed = seed

    def _apply_seed(self) -> None:
        """Apply the stored seed to all framework-controlled RNGs."""
        if self._seed is None:
            return
        random.seed(self._seed)
        try:
            import numpy as np
            np.random.seed(self._seed)
        except ImportError:
            pass

    def set_min_fitness(self, value: float) -> None:
        self.min_fitness = value

    def set_population_size(self, n: int) -> None:
        self._population_size = n
        if self._population is not None:
            self._population.max_size = n

    def set_adaptive_population(
        self,
        min_size: int,
        max_size: int,
        growth_rate: float = 0.05,
        enabled: bool = True,
    ) -> None:
        """Enable adaptive population sizing.

        Automatically grows or shrinks the population between [min_size, max_size]
        based on species diversity and stagnation pressure.  The current size is
        preserved — it must fall within [min_size, max_size].

        Args:
            min_size:    Minimum allowed population size.
            max_size:    Maximum allowed population size.
            growth_rate: Fractional change per generation (default 0.05 = 5 %).
            enabled:     Pass False to disable without losing the configured limits.
        """
        if min_size < 1:
            raise ValueError(f"min_size must be >= 1, got {min_size}")
        if max_size < min_size:
            raise ValueError(f"max_size must be >= min_size, got {max_size} < {min_size}")
        self._adaptive_pop_enabled = enabled
        self._adaptive_pop_min = min_size
        self._adaptive_pop_max = max_size
        self._adaptive_pop_rate = max(0.0, growth_rate)
        if self._population is not None:
            self._population._adaptive_pop_enabled = enabled
            self._population._adaptive_pop_min = min_size
            self._population._adaptive_pop_max = max_size
            self._population._adaptive_pop_rate = self._adaptive_pop_rate

    def set_max_iterations(self, n: int) -> None:
        self.max_iterations = n

    def set_max_evaluations(self, n: int) -> None:
        """Stop training after n total fitness evaluations.

        Unlike max_iterations which counts spawned genomes, this counts actual
        fitness-function calls — so with n_evaluations=3 per genome each
        iteration uses 3 evaluations toward this limit.

        Args:
            n: Maximum number of fitness-function calls before training stops.
        """
        self._max_evaluations = max(1, n)

    def set_convergence_stop(
        self,
        fitness_spread_eps: float,
        min_stagnation: float = 1.0,
    ) -> None:
        """Stop training when the population has converged.

        Training stops when **both** conditions hold simultaneously:
        - ``stagnation_count >= stagnation_threshold * min_stagnation``
        - IQR of fitness in the evaluated population < ``fitness_spread_eps``

        The IQR check detects when all genomes cluster at similar fitness
        (evolutionary plateau); the stagnation guard prevents premature stops
        during warm-up phases when the population is small.

        Args:
            fitness_spread_eps: IQR threshold below which the population is
                considered converged.  A small positive value (e.g. 0.01) works
                well for normalised fitness; tune to your fitness scale.
            min_stagnation: Fraction of ``stagnation_threshold`` that must be
                reached before convergence is checked.  Default 1.0 = full
                stagnation required. Lower values allow earlier stops.
        """
        self._convergence_spread_eps = fitness_spread_eps
        self._convergence_min_stagnation = max(0.0, min_stagnation)

    def set_early_stopping(self, factor: float = 1.0) -> None:
        """Enable early stopping for per-genome generator fitness functions.

        When a generator-based fitness function is used (``yield``-based), each
        yielded value is treated as one episode result.  After N is calibrated
        from the first complete (non-stopped) run, subsequent evaluations use
        extrapolation and a 20 % warmup before checking:

            estimated = cumulative_so_far * (N / episodes_so_far)
            stop if  estimated  <  best_fitness - abs(best_fitness) * factor

        The sign-independent threshold means the rule works correctly for both
        positive and negative fitness scales.  N is calibrated automatically —
        no configuration needed.  Until N is known (first run) the generator
        always runs to completion.

        Args:
            factor: Tolerance fraction relative to the current best fitness.
                    0.0 = stop only when extrapolated fitness is below the
                    current best.  1.0 (default) = stop when extrapolated
                    fitness is more than 100 % below the best (very lenient).
                    Smaller values make stopping more aggressive.
        """
        self._runner.early_stopping_factor = factor

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
        self._runner.configure_multi_eval(n, aggregation, sigma_penalty)

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
        self._sanitizer.configure(fallback=fallback, clip_low=clip_low, clip_high=clip_high)

    def set_lamarck(
        self,
        n_steps: int = 5,
        sigma: float = 1.0,
        mode: str = "hill_climbing",
        learning_rate: float = 0.01,
        cooling_rate: float = 0.95,
        cma_population: int = 6,
    ) -> None:
        """Explicit weight refinement before each genome evaluation.

        ``mode='hill_climbing'`` (default): n_steps independent hill-climbing
        attempts, each requiring 1 fitness evaluation.

        ``mode='nes'``: Natural Evolution Strategies.  n_steps antithetic
        perturbation pairs are used to estimate a gradient, then one directed
        weight update is applied.  Costs ``2*n_steps + 1`` evaluations but
        makes a single directed step rather than n_steps random attempts.

        ``mode='sa'``: Simulated Annealing.  n_steps perturbations with a
        geometric cooling schedule; worse moves are accepted probabilistically.
        Returns the best fitness seen across the chain.

        ``mode='cma_es'``: compact full-covariance CMA-ES over weights/biases.

        In all modes, n_steps = 0 re-enables adaptive scheduling.

        Args:
            n_steps:       steps / perturbation pairs (default 5; 0 = adaptive).
            sigma:         multiplier on ``genome.lamarck_sigma`` (default 1.0).
            mode:          ``'hill_climbing'``, ``'nes'``, ``'sa'``, or ``'cma_es'``.
            learning_rate: NES gradient step size (only used when mode='nes').
            cooling_rate:  SA geometric cooling factor (only used when mode='sa').
            cma_population: CMA-ES samples per step.
        """
        if mode == "nes":
            self._lamarck.set_nes(
                k=n_steps, sigma=sigma, learning_rate=learning_rate, adaptive=False
            )
        elif mode == "sa":
            self._lamarck.set_sa(
                k=n_steps, sigma=sigma, cooling_rate=cooling_rate, adaptive=False
            )
        elif mode == "cma_es":
            self._lamarck.set_cma_es(
                k=n_steps, sigma=sigma, population=cma_population, adaptive=False
            )
        else:
            self._lamarck.nes_mode = False
            self._lamarck.sa_mode = False
            self._lamarck.cma_mode = False
            self._lamarck.set_explicit(n_steps, sigma)

    def set_lamarck_adaptive(
        self,
        max_steps: int = 3,
        top_k: float = 0.2,
        sigma: float = 1.0,
        mode: str = "hill_climbing",
        learning_rate: float = 0.01,
        cooling_rate: float = 0.95,
        cma_population: int = 6,
    ) -> None:
        """Configure adaptive Lamarckian refinement (fires during stagnation).

        Steps scale linearly from 0 (no stagnation) to max_steps (full
        stagnation); only genomes in the top top_k fraction are refined.

        Args:
            max_steps:     maximum steps at full stagnation (default 3).
                           0 disables adaptive mode entirely.
            top_k:         fraction eligible for refinement (default 0.2).
            sigma:         multiplier on ``genome.lamarck_sigma`` (default 1.0).
            mode:          ``'hill_climbing'`` (default), ``'nes'``, ``'sa'``, or ``'cma_es'``.
            learning_rate: NES gradient step size (only used when mode='nes').
            cooling_rate:  SA geometric cooling factor (only used when mode='sa').
        """
        if mode == "nes":
            self._lamarck.set_nes(
                k=max_steps, sigma=sigma, learning_rate=learning_rate, adaptive=True
            )
            self._lamarck.top_k = max(0.0, min(1.0, top_k))
        elif mode == "sa":
            self._lamarck.set_sa(
                k=max_steps, sigma=sigma, cooling_rate=cooling_rate, adaptive=True
            )
            self._lamarck.top_k = max(0.0, min(1.0, top_k))
        elif mode == "cma_es":
            self._lamarck.set_cma_es(
                k=max_steps, sigma=sigma, population=cma_population, adaptive=True
            )
            self._lamarck.top_k = max(0.0, min(1.0, top_k))
        else:
            self._lamarck.nes_mode = False
            self._lamarck.sa_mode = False
            self._lamarck.cma_mode = False
            self._lamarck.set_adaptive(max_steps, top_k, sigma)

    def set_lamarck_budget(self, budget_per_gen: int | None) -> None:
        """Limit Lamarck refinement to at most budget_per_gen evaluations per generation.

        When the budget is exhausted within a generation, further genomes in that
        generation are not refined.  The counter resets at the start of each generation.
        None (default) = unlimited.

        Args:
            budget_per_gen: Maximum extra fitness evaluations for Lamarck per generation.
                            None = no limit.
        """
        self._lamarck.set_budget(budget_per_gen)

    def set_lamarck_per_species(
        self,
        enabled_species: list[int] | None = None,
    ) -> None:
        """Restrict Lamarck refinement to specific species by their array index.

        This allows per-species decisions: species 0 might be refined while
        species 1 and 2 evolve by mutation only.

        Args:
            enabled_species: List of species *indices* (position in population._species)
                             that should receive Lamarck refinement.  None (default) = all
                             species are eligible.

        Note: species indices change as species are born and die.  This is most
        useful when combined with stable speciation (high min_species_age).
        The setting is applied at evaluation time and re-evaluated each generation.
        """
        if enabled_species is None:
            self._lamarck_eligible_species_indices: list[int] | None = None
        else:
            self._lamarck_eligible_species_indices = list(enabled_species)

    def set_adaptive_control(self, enabled: bool = True) -> None:
        """Enable or disable the central Adaptive Control Layer.

        When enabled, the AdaptiveController automatically ticks each generation
        and adjusts interspecies crossover, QD pressure, pruning, and Lamarck
        budget based on unified population signals (plateau, diversity, etc.).

        Individual feature policies can be configured via
        ``get_adaptive_controller().configure_*()``.
        """
        self._adaptive_ctrl_enabled = enabled
        if enabled and self._population is not None:
            # Sync population's interspecies policy into the controller
            pop = self._population
            self._adaptive_ctrl.configure_interspecies_crossover(
                mode=pop._interspecies_crossover_mode,
                fixed_rate=pop._interspecies_crossover_rate,
                min_rate=pop._interspecies_crossover_min,
                max_rate=pop._interspecies_crossover_max,
            )

    def get_adaptive_controller(self) -> AdaptiveController:
        """Return the AdaptiveController for advanced configuration."""
        return self._adaptive_ctrl

    def set_operator_scheduler(self, enabled: bool = True) -> None:
        """Enable or disable the adaptive Operator Scheduler.

        When enabled, mutation operator weights are automatically adjusted
        each generation based on operator success rates and population signals.
        Use ``get_operator_scheduler()`` for detailed configuration.
        """
        self._operator_scheduler_enabled = enabled
        if self._population is not None:
            self._population._operator_scheduler = (
                self._operator_scheduler if enabled else None
            )

    def get_operator_scheduler(self) -> OperatorScheduler:
        """Return the OperatorScheduler for advanced configuration."""
        return self._operator_scheduler

    def set_fitness_components(
        self,
        components: list[FitnessComponent],
        *,
        mode: str = "fixed",
        min_weight: float = 0.0,
        max_weight: float = 5.0,
        adaptation_rate: float = 0.25,
        collapse_floor: float = 0.05,
        history_max: int = 200,
    ) -> None:
        """Attach weighted auxiliary fitness components.

        Components are evaluated for each genome and added to task fitness as a
        scalar shaping term. With ``mode="adaptive"``, weights are updated once
        per generation during stagnation and every update is kept in diagnostics.
        """
        self._fitness_component_weights = AdaptiveFitnessComponentWeights(
            components,
            mode=mode,
            min_weight=min_weight,
            max_weight=max_weight,
            adaptation_rate=adaptation_rate,
            collapse_floor=collapse_floor,
            history_max=history_max,
        )

    def get_fitness_component_weights(self) -> AdaptiveFitnessComponentWeights | None:
        """Return the fitness-component scalarizer, if configured."""
        return self._fitness_component_weights

    def set_meta_adaptive_policies(
        self,
        enabled: bool = True,
        *,
        seed: int | None = None,
        bounds: PolicyGeneBounds | None = None,
        initial_genes: PolicyGenes | None = None,
        mutation_strength: float = 1.0,
    ) -> None:
        """Enable evolvable policy genes for adaptive-control parameters.

        The meta-policy layer evolves compact genes for operator exploration,
        Lamarck budget, and interspecies-crossover rate. It compares global and
        per-species scores each generation, clamps all values to safe bounds,
        and applies them through existing YANE controls.
        """
        self._meta_adaptive_enabled = enabled
        if self._meta_adaptive is None:
            self._meta_adaptive = MetaAdaptivePolicyEvolver(
                enabled=enabled,
                seed=seed if seed is not None else self._seed,
                bounds=bounds,
                mutation_strength=mutation_strength,
            )
        else:
            self._meta_adaptive.enabled = enabled
            if bounds is not None:
                self._meta_adaptive.bounds = bounds
            self._meta_adaptive.mutation_strength = max(0.0, float(mutation_strength))
        if initial_genes is not None and self._meta_adaptive is not None:
            self._meta_adaptive.global_genes = initial_genes.clamped(self._meta_adaptive.bounds)
        if enabled:
            self.set_operator_scheduler(True)

    def get_meta_adaptive_policies(self) -> MetaAdaptivePolicyEvolver | None:
        """Return the meta-adaptive policy evolver, if configured."""
        return self._meta_adaptive

    def set_module_library(
        self,
        enabled: bool = True,
        *,
        max_modules: int = 50,
        min_fitness: float | None = None,
        insert_rate: float = 0.02,
    ) -> None:
        """Enable reusable hidden-module storage and insertion mutations."""
        if enabled:
            if self._module_library is None:
                self._module_library = ModuleLibrary(
                    max_modules=max_modules,
                    min_fitness=min_fitness,
                )
            else:
                self._module_library.max_modules = max_modules
                self._module_library.min_fitness = min_fitness
            self._module_insert_rate = max(0.0, min(1.0, float(insert_rate)))
        else:
            self._module_library = None
            self._module_insert_rate = 0.0
        if self._population is not None:
            self._population._module_library = self._module_library
            self._population._module_insert_rate = self._module_insert_rate

    def get_module_library(self) -> ModuleLibrary | None:
        """Return the active module library, if configured."""
        return self._module_library

    def set_matrix_forward(self, enabled: bool = True) -> None:
        """Enable matrix-accelerated forward() for compatible genomes during training.

        When enabled, YANE transparently replaces ``genome.forward()`` with a
        NumPy matrix path for feed-forward (acyclic, non-stateful) genomes.
        Genomes with cycles, memory nodes, or unsupported activations fall back
        to the standard forward path automatically.

        Diagnostics are reported via ``population_memory_info()`` under the
        keys ``matrix_forward_hits`` and ``matrix_forward_misses``.

        Args:
            enabled: Pass ``False`` to disable (default: ``True``).
        """
        self._matrix_forward_enabled = enabled
        if enabled and self._matrix_cache is None:
            from yane.evolution.matrix_export import MatrixForwardCache
            self._matrix_cache = MatrixForwardCache()

    def set_log_format(self, format: str) -> None:
        """Set the output format for training-run logs.

        Args:
            format: One of ``"csv"`` (default), ``"jsonlines"``, or ``"both"``.
                ``"jsonlines"`` writes one JSON object per heartbeat (every 100
                iterations) to ``fitness_history.jsonl`` — the full diagnostics
                dict including all species, Lamarck, and adaptive stats.
                ``"both"`` writes both the CSV and the JSONL file.
        """
        if format not in ("csv", "jsonlines", "both"):
            raise ValueError(
                f"format must be 'csv', 'jsonlines', or 'both'; got {format!r}"
            )
        self._log_format = format

    def set_tensorboard_logdir(self, path: str | Path | None) -> None:
        """Write scalar summaries to a TensorBoard log directory.

        Requires ``torch`` or the standalone ``tensorboard`` package::

            pip install torch          # or: pip install tensorboard

        Each heartbeat (every 100 iterations) writes all numeric diagnostics
        keys under the tag ``yane/<key>``.

        Args:
            path: Directory for SummaryWriter.  ``None`` disables TensorBoard.
        Raises:
            ImportError: If neither ``torch.utils.tensorboard`` nor
                ``tensorboard`` is installed when *path* is not ``None``.
        """
        if path is None:
            if self._tensorboard_writer is not None:
                try:
                    self._tensorboard_writer.close()
                except Exception:
                    pass
            self._tensorboard_logdir = None
            self._tensorboard_writer = None
            return
        self._tensorboard_logdir = Path(path)
        # Validate that a SummaryWriter can be created now (fail fast).
        try:
            from torch.utils.tensorboard import SummaryWriter as _SW
        except ImportError:
            try:
                from tensorboard.summary.writer.event_file_writer import EventFileWriter as _SW  # noqa: F401
                from torch.utils.tensorboard import SummaryWriter as _SW  # type: ignore
            except ImportError:
                raise ImportError(
                    "TensorBoard logging requires 'torch' or 'tensorboard'. "
                    "Install it with:  pip install torch  (or pip install tensorboard)"
                ) from None

    def set_log_callbacks(
        self,
        on_generation: "Callable[[dict], None] | None" = None,
    ) -> None:
        """Register a callback called at every heartbeat with the full diagnostics dict.

        The callback receives the same dict as ``population_memory_info()``
        plus an ``"iteration"`` key.  It runs synchronously in the training
        thread — keep it fast or offload to a queue.

        Args:
            on_generation: Callable ``fn(info: dict) -> None``.
                Pass ``None`` to remove any existing callback.
        """
        self._log_callbacks = [on_generation] if on_generation is not None else []

    def set_curriculum(
        self,
        stages: list,
        on_stage_advance: Callable | None = None,
    ) -> None:
        """Configure curriculum learning: a sequence of progressively harder tasks.

        When curriculum is active, ``train()`` does **not** require a
        ``fitness_fn`` argument — pass ``None`` or omit it.

        Args:
            stages: List of ``CurriculumStage`` objects, each with a
                ``fitness_fn`` (callable), an optional ``target_fitness``
                threshold, and an optional ``name``.  The last stage uses
                ``min_fitness`` (or ``max_iterations``) as its stop condition.
            on_stage_advance: Optional callback invoked when a stage advances.
                Signature: ``callback(stage_index: int, stage: CurriculumStage)``.

        Example::

            from yane.evolution.curriculum import CurriculumStage
            ne.set_curriculum([
                CurriculumStage(easy_fn,   target_fitness=0.90, name="easy"),
                CurriculumStage(medium_fn, target_fitness=0.95, name="medium"),
                CurriculumStage(hard_fn,   name="hard"),  # last stage
            ])
            ne.set_target_fitness(0.98)  # stop condition for the final stage
            ne.train()

        Note: generator-based fitness functions (early-stopping protocol) are
        not supported inside curriculum stages.
        """
        stages_list = list(stages)
        if not stages_list:
            raise ValueError("set_curriculum() requires at least one stage")
        for s in stages_list:
            if not isinstance(s, CurriculumStage):
                raise TypeError(
                    f"stages must be CurriculumStage instances, got {type(s)}"
                )
        self._curriculum = Curriculum(stages_list)
        self._on_stage_advance = on_stage_advance

    @property
    def curriculum_stage(self) -> int | None:
        """Current curriculum stage index, or None if no curriculum is set."""
        return self._curriculum.stage_index if self._curriculum else None

    def _config_dict(self) -> dict:
        """Return all NeuroEvolution settings as a JSON-serializable dict."""
        pop = self._population
        return {
            # Reproducibility
            "seed": self._seed,
            # Network
            "n_inputs": self._n_inputs,
            "n_outputs": self._n_outputs,
            "max_nodes": self._max_nodes,
            "max_connections": self._max_connections,
            "n_initial_hidden": self._n_initial_hidden,
            "stateful": self._stateful,
            # Population
            "population_size": self._population_size,
            "adaptive_pop_enabled": self._adaptive_pop_enabled,
            "adaptive_pop_min": self._adaptive_pop_min,
            "adaptive_pop_max": self._adaptive_pop_max,
            "adaptive_pop_rate": self._adaptive_pop_rate,
            "n_workers": self._n_workers,
            "target_species": pop._target_species if pop else None,
            "compat_threshold": pop._compat_threshold if pop else None,
            "interspecies_crossover_mode": (
                pop._interspecies_crossover_mode if pop else self._interspecies_crossover_mode
            ),
            "interspecies_crossover_rate": (
                pop._interspecies_crossover_rate if pop else self._interspecies_crossover_rate
            ),
            "interspecies_crossover_min": (
                pop._interspecies_crossover_min if pop else self._interspecies_crossover_min
            ),
            "interspecies_crossover_max": (
                pop._interspecies_crossover_max if pop else self._interspecies_crossover_max
            ),
            "interspecies_crossover_current": (
                pop._interspecies_crossover_current if pop else self._interspecies_crossover_rate
            ),
            "elite_count": self._elite_count,
            "species_elite_count": self._species_elite_count,
            # Stopping criteria
            "min_fitness": self.min_fitness,
            "max_iterations": self.max_iterations,
            "max_evaluations": self._max_evaluations,
            "convergence_spread_eps": self._convergence_spread_eps,
            "convergence_min_stagnation": self._convergence_min_stagnation,
            # Evaluation
            "n_evaluations": self._runner.n_evaluations,
            "eval_aggregation": self._runner.aggregation,
            "eval_sigma_penalty": self._runner.sigma_penalty,
            # Lamarck
            "lamarck_mode": self._lamarck.mode,
            "lamarck_steps": self._lamarck.steps,
            "lamarck_max_steps": self._lamarck.max_steps,
            "lamarck_top_k": self._lamarck.top_k,
            "lamarck_sigma": self._lamarck.sigma,
            "lamarck_nes_mode": self._lamarck.nes_mode,
            "lamarck_nes_lr": self._lamarck.nes_lr,
            "lamarck_sa_mode": self._lamarck.sa_mode,
            "lamarck_sa_cooling": self._lamarck.sa_cooling,
            "lamarck_sa_t0": self._lamarck.sa_t0,
            "lamarck_cma_mode": self._lamarck.cma_mode,
            "lamarck_cma_population": self._lamarck.cma_population,
            # Efficiency penalty
            "efficiency_penalty": (
                {"max_ms": self._efficiency_penalty.max_ms,
                 "penalty_per_ms": self._efficiency_penalty.penalty_per_ms}
                if self._efficiency_penalty is not None else None
            ),
            "complexity_penalty_nodes": self._complexity_penalty_nodes,
            "complexity_penalty_connections": self._complexity_penalty_connections,
            # Resource limits
            "resource_check_interval": self._resource_check_interval,
            "memory_limit_gb": self._resource_guard.max_process_gb,
            # Fitness sanitizing
            "sanitize_enabled": self._sanitizer.enabled,
            "sanitize_fallback": self._sanitizer.fallback,
            "sanitize_clip_low": self._sanitizer.clip_low,
            "sanitize_clip_high": self._sanitizer.clip_high,
            # Output sanitizing
            "output_sanitize": self._output_sanitize,
            "output_sanitize_fallback": self._output_fallback,
            # Multi-objective
            "multi_objective_enabled": self._multi_objective_enabled,
            "multi_objective_weights": self._multi_objective_weights,
            "multi_objective_maximize": self._multi_objective_maximize,
            # Quality Diversity
            "quality_diversity_enabled": self._qd_enabled,
            "quality_diversity_bins": self._qd_archive.bins if self._qd_archive else None,
            "quality_diversity_ranges": self._qd_archive.ranges if self._qd_archive else None,
            "quality_diversity_descriptor_needs_reattach": self._qd_descriptor_needs_reattach,
            "fitness_component_weights": (
                {
                    "mode": self._fitness_component_weights.mode,
                    "weights": self._fitness_component_weights.weights,
                    "last_reason": self._fitness_component_weights.last_reason,
                }
                if self._fitness_component_weights is not None else None
            ),
            "meta_adaptive_enabled": self._meta_adaptive_enabled,
            "meta_adaptive_policies": (
                self._meta_adaptive.get_diagnostics()
                if self._meta_adaptive is not None else None
            ),
            "module_library_enabled": self._module_library is not None,
            "module_insert_rate": self._module_insert_rate,
        }

    # -------------------------------------------------------------------------
    # Training (automatic loop)
    # -------------------------------------------------------------------------

    def train(
        self,
        fitness_fn: Callable[[Genome], float] | None = None,
        run_name: str | None = None,
        on_stop: Callable[[str], None] | None = None,
        on_iteration: Callable[[int, float, float], bool] | None = None,
    ) -> int:
        """Run the evolutionary loop.

        Runs until a stop condition is reached or indefinitely if none is set.
        Stop conditions (checked in priority order):
        - ``min_fitness`` — stops when a genome reaches the target fitness.
        - ``max_evaluations`` — stops after N total fitness-function calls.
        - ``max_iterations`` — stops after N genome evaluations.
        - Convergence — stops when IQR < ``fitness_spread_eps`` at full stagnation.
        - ``"curriculum_complete"`` — all curriculum stages finished (curriculum
          mode only).
        - ``"external"`` — ``on_iteration`` returned ``False``.

        Automatically pauses when system memory is low and resumes when it
        recovers.

        Structured logging is automatically set up: a timestamped directory
        under ``logs/<run_name>/`` receives ``run.log``, ``config.json``,
        ``fitness_history.csv`` and ``best_genome.json``.

        Args:
            fitness_fn: Function that evaluates a genome and returns fitness.
                Omit (or pass ``None``) when curriculum is set via
                ``set_curriculum()``.
            run_name: Category name for logs (default: derived from *fitness_fn*).
            on_stop: Optional callback called with the stop reason string when
                training ends.  Possible values: ``"target_reached"``,
                ``"max_evaluations"``, ``"max_iterations"``, ``"converged"``,
                ``"curriculum_complete"``, ``"external"``.
            on_iteration: Optional callback invoked after each genome is
                evaluated.  Signature:
                ``callback(iteration: int, fitness: float, elapsed_ms: float) -> bool``.
                Return ``False`` to stop training (stop reason ``"external"``).
        Returns:
            Number of iterations performed.
        """
        self._ensure_configured()

        # Curriculum takes priority; fitness_fn may be None when curriculum is set.
        if self._curriculum is not None:
            fitness_fn = self._curriculum
        elif fitness_fn is None:
            raise ValueError(
                "fitness_fn is required when no curriculum is set. "
                "Call set_curriculum() first or pass a fitness function."
            )

        # --- Structured logging setup ----------------------------------------
        name = run_name or (
            self._curriculum.current_stage.name or "curriculum"
            if self._curriculum else _derive_run_name(fitness_fn)
        )
        self._log_run_name = name
        from yane.util.logger import setup_logging as _setup, write_json as _wj, write_csv as _wc, write_jsonl as _wjl, log_info as _li
        self._log_run_dir = _setup(name)
        _li("Training started  run_name=%s  pop_size=%d  max_iter=%s  min_fitness=%s",
            name, self._population_size, self.max_iterations, self.min_fitness)
        _wj(self._log_run_dir / "config.json", self._config_dict())

        # --- Logging state ---------------------------------------------------
        _log_interval = max(1, self._population_size // 10)  # log ~10× per generation
        _csv_header = "iteration,best_fitness,mean_fitness,median_fitness,iqr_fitness,species_count,stagnation_count,nodes,connections"
        _csv_path = self._log_run_dir / "fitness_history.csv"
        _jsonl_path = self._log_run_dir / "fitness_history.jsonl"
        _use_csv   = self._log_format in ("csv", "both")
        _use_jsonl = self._log_format in ("jsonlines", "both")

        # TensorBoard writer: create once per run if configured.
        if self._tensorboard_logdir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter as _SW
                self._tensorboard_writer = _SW(log_dir=str(self._tensorboard_logdir))
            except Exception as _tb_err:
                _li("TensorBoard init failed: %s", _tb_err)
                self._tensorboard_writer = None
        else:
            self._tensorboard_writer = None

        self._n_evaluations_done = 0
        stop_reason: str | None = None
        iterations = 0
        _gen_size = max(1, self._population_size)  # generation boundary for periodic ticks
        while True:
            genome = self._population.select_for_evaluation()

            result = self._run_evaluations(genome, fitness_fn)
            fitness = self._finalize_fitness(result.fitness, result.elapsed_ms, genome)
            self._population.submit(genome, fitness, result.elapsed_ms)
            iterations += 1
            self._n_evaluations_done += self._runner.n_evaluations

            # Tick adaptive components once per generation
            if iterations % _gen_size == 0:
                if self._fitness_component_weights is not None:
                    self._fitness_component_weights.tick(self._population)
                if self._meta_adaptive_enabled and self._meta_adaptive is not None:
                    self._meta_adaptive.tick(
                        self._population,
                        self._operator_scheduler if self._operator_scheduler_enabled else None,
                        self._lamarck,
                    )
                if self._adaptive_ctrl_enabled:
                    self._adaptive_ctrl.tick(self._population, self._lamarck)
                self._lamarck.reset_generation_budget()

            # --- Periodic CSV logging + heartbeat ---------------------------
            _heartbeat_now = (iterations % 100 == 0)
            _csv_now = (iterations % _log_interval == 0)

            if _heartbeat_now:
                # Full diagnostics once; shared by heartbeat log and CSV.
                mem = self.population_memory_info()
                if _use_csv and _csv_now:
                    _wc(_csv_path, _csv_header,
                        f"{iterations},"
                        f"{mem.get('max_fitness', 0)},"
                        f"{mem.get('avg_fitness', 0)},"
                        f"{mem.get('median_fitness', 0)},"
                        f"{mem.get('fitness_iqr', 0)},"
                        f"{mem.get('species_count', 0)},"
                        f"{mem.get('stagnation_count', 0)},"
                        f"{mem.get('largest_genome_nodes', 0)},"
                        f"{mem.get('largest_genome_connections', 0)}")
                if _use_jsonl:
                    _wjl(_jsonl_path, {**mem, "iteration": iterations})
                if self._tensorboard_writer is not None:
                    _tb = self._tensorboard_writer
                    for _k, _v in mem.items():
                        if isinstance(_v, (int, float)) and _v == _v:  # skip NaN
                            try:
                                _tb.add_scalar(f"yane/{_k}", float(_v), iterations)
                            except Exception:
                                pass
                for _cb in self._log_callbacks:
                    try:
                        _cb({**mem, "iteration": iterations})
                    except Exception:
                        pass
                _li("iter=%d  best=%.4f  avg=%.2f  species=%d  stagn=%d  nodes=%d  conns=%d",
                    iterations,
                    mem.get("max_fitness", 0.0),
                    mem.get("avg_fitness", 0.0),
                    mem.get("species_count", 0),
                    mem.get("stagnation_count", 0),
                    mem.get("largest_genome_nodes", 0),
                    mem.get("largest_genome_connections", 0))
                self._write_crash_snapshot(iterations, mem, _li)
            elif _csv_now and _use_csv:
                # Cheap path: compute only the fields the CSV needs, no BFS.
                _pop = self._population
                _evald = _pop._evaluated
                if _evald:
                    _fits = [g.raw_fitness for g in _evald]
                    _n = len(_fits)
                    _max_fit = max(_fits)
                    _avg_fit = sum(_fits) / _n
                    _sorted = sorted(_fits)
                    _med_fit = (
                        _sorted[_n // 2]
                        if _n % 2
                        else (_sorted[_n // 2 - 1] + _sorted[_n // 2]) * 0.5
                    )
                    _iqr_fit = _compute_fitness_iqr(_evald)
                else:
                    _max_fit = _avg_fit = _med_fit = _iqr_fit = 0.0
                _all_g = _evald + list(_pop._unevaluated)
                _max_nodes = max((len(g.nodes) for g in _all_g), default=0)
                _max_conns = max((g.connection_count for g in _all_g), default=0)
                _wc(_csv_path, _csv_header,
                    f"{iterations},"
                    f"{_max_fit},{_avg_fit},{_med_fit},{_iqr_fit},"
                    f"{_pop.species_count},{_pop.stagnation_count},"
                    f"{_max_nodes},{_max_conns}")

            # --- Curriculum stage advancement --------------------------------
            _stage_advanced = False
            if self._curriculum is not None:
                self._curriculum.record_fitness(fitness)
                if self._curriculum.maybe_advance():
                    _stage_advanced = True
                    stage = self._curriculum.current_stage
                    _li("Curriculum: advanced to stage %d/%d '%s'",
                        self._curriculum.stage_index + 1,
                        self._curriculum.stage_count,
                        stage.name or "")
                    self._population.reset_stagnation()
                    if self._on_stage_advance is not None:
                        try:
                            self._on_stage_advance(
                                self._curriculum.stage_index, stage
                            )
                        except Exception:
                            pass
                elif self._curriculum.is_complete():
                    stop_reason = "curriculum_complete"
                    _li("Curriculum: all stages complete  iterations=%d", iterations)
                    break

            # --- on_iteration callback ---------------------------------------
            if on_iteration is not None:
                if on_iteration(iterations, fitness, result.elapsed_ms) is False:
                    stop_reason = "external"
                    break

            # Skip stop-condition check when a stage just advanced: the fitness
            # that triggered the advance belongs to the old task and must not be
            # used to satisfy min_fitness or other global stop criteria.
            if not _stage_advanced:
                stop_reason = self._check_stop_reason(genome.raw_fitness, iterations, _li)
                # When on an intermediate curriculum stage, min_fitness reflects
                # the final task's target — it must not fire on easier-stage
                # scores that happen to exceed it.  Each intermediate stage is
                # governed solely by its own CurriculumStage.target_fitness
                # (via maybe_advance()).
                if (stop_reason == "target_reached"
                        and self._curriculum is not None
                        and not self._curriculum.is_last_stage):
                    stop_reason = None
                if stop_reason is not None:
                    break

            if iterations % self._resource_check_interval == 0:
                self._enforce_memory_limit()
                while not self._resource_guard.system_ok():
                    time.sleep(0.5)

        # --- on_stop callback ------------------------------------------------
        if on_stop is not None:
            try:
                on_stop(stop_reason or "manual")
            except Exception:
                pass

        # --- End-of-run artefacts -------------------------------------------
        self._write_run_summary(name, stop_reason, iterations, _wj, _li)
        if self._tensorboard_writer is not None:
            try:
                self._tensorboard_writer.close()
            except Exception:
                pass
            self._tensorboard_writer = None

        return iterations

    def get_best(self) -> Genome:
        self._ensure_configured()
        return self._population.get_best()

    def get_ensemble(self, k: int = 3) -> list[Genome]:
        """Return the top-k genomes by fitness for ensemble inference."""
        self._ensure_configured()
        return self._population.get_top(k)

    def forward_batch(self, batch) -> list[list[float]]:
        """Vectorized forward pass on the best genome for a batch of inputs.

        Delegates to ``Genome.forward_batch()``: ~10–100× faster than
        sequential forward() for acyclic networks.  Falls back to sequential
        for cyclic or stateful (memory) networks.

        Args:
            batch: N input vectors (list-of-lists or 2-D ndarray).
        Returns:
            list of N output vectors.
        """
        self._ensure_configured()
        return self._population.get_best().forward_batch(batch)

    def forward_ensemble(
        self,
        inputs: list[float],
        k: int = 3,
        mode: str = "mean",
    ) -> list[float]:
        """Run inputs through the top-k genomes and aggregate outputs.

        Parameters
        ----------
        inputs : list[float]
            Network inputs shared by all ensemble members.
        k : int
            Number of top genomes to include (by fitness).
        mode : str
            Aggregation strategy:
            - ``"mean"``     — arithmetic mean of all outputs (default, original behaviour).
            - ``"vote"``     — hard argmax vote per genome; returns one-hot of plurality class.
              Requires discrete outputs (each genome's argmax is its vote).
            - ``"weighted"`` — fitness-weighted mean; genomes with higher fitness
              contribute proportionally more.
        """
        top_k = self.get_ensemble(k)
        if not top_k:
            raise RuntimeError("No evaluated genomes yet.")
        all_outputs = [g.forward(inputs) for g in top_k]
        n_out = len(all_outputs[0])

        if mode == "mean":
            return [
                sum(out[i] for out in all_outputs) / len(all_outputs)
                for i in range(n_out)
            ]

        if mode == "vote":
            votes = [0] * n_out
            for out in all_outputs:
                winner = max(range(n_out), key=lambda i: out[i])
                votes[winner] += 1
            total = len(all_outputs)
            return [v / total for v in votes]

        if mode == "weighted":
            fitnesses = [max(g.fitness, 0.0) for g in top_k]
            total_fit = sum(fitnesses)
            if total_fit == 0.0:
                weights = [1.0 / len(top_k)] * len(top_k)
            else:
                weights = [f / total_fit for f in fitnesses]
            return [
                sum(w * out[i] for w, out in zip(weights, all_outputs))
                for i in range(n_out)
            ]

        raise ValueError(f"Unknown ensemble mode {mode!r}. Use 'mean', 'vote', or 'weighted'.")

    def population_memory_info(self) -> dict:
        """Returns node/connection/fitness/species diagnostics for the population."""
        self._ensure_configured()
        info = build_population_info(
            self._population,
            self._lamarck,
            self._sanitizer,
            self._runner.n_evaluations,
            self._runner.aggregation,
            self._runner.n_early_stopped,
            adaptive_ctrl=self._adaptive_ctrl if self._adaptive_ctrl_enabled else None,
            operator_scheduler=self._operator_scheduler if self._operator_scheduler_enabled else None,
            fitness_component_weights=self._fitness_component_weights,
            meta_adaptive=self._meta_adaptive if self._meta_adaptive_enabled else None,
        )
        if self._curriculum is not None:
            info.update(self._curriculum.info())
        if self._matrix_forward_enabled:
            info["matrix_forward_hits"] = self._matrix_hits
            info["matrix_forward_misses"] = self._matrix_misses
        return info

    def set_target_species(self, n: int) -> None:
        """Set the target number of species the population tries to maintain.

        The adaptive compatibility threshold rises/falls automatically to keep
        the actual species count close to this target.  Higher values protect
        more structural niches and help escape local optima (especially for
        XOR-like tasks where intermediate structures are temporarily worse).

        Pass 0 to auto-compute from population size: ``sqrt(pop_size)``.
        Default: 5.  For small discrete-mapping tasks (binary increment,
        XOR variants): 10–20 works significantly better.
        """
        if self._population is not None:
            self._population._target_species = max(1, n) if n > 0 else 0

    def set_speciation_metric(self, metric: str) -> None:
        """Choose the compatibility metric used for species assignment.

        ``"topology"`` (default):
            Standard NEAT compatibility — considers all connections including
            disabled ones.  Fast, cached.

        ``"topology_no_disabled"``:
            Ignores disabled connections.  Genomes that differ only in their
            disable/enable patterns become more compatible, encouraging species
            to specialise in different structural subsets.  Slightly slower
            (cache is skipped for accuracy).
        """
        if metric not in ("topology", "topology_no_disabled"):
            raise ValueError(f"Unknown speciation metric: {metric}")
        if self._population is not None:
            self._population._speciation_metric = metric

    # ── Getters ────────────────────────────────────────────────────────────

    def get_config(self) -> dict:
        """Return the full configuration dict (same format as config.json)."""
        return self._config_dict()

    def get_target_species(self) -> int:
        return (self._population._target_species if self._population is not None
                else self._config_dict().get("target_species", 5))

    def get_elitism(self) -> tuple[int, int]:
        if self._population is not None:
            return self._population.elite_count, self._population.species_elite_count
        return self._elite_count, self._species_elite_count

    def get_lamarck_mode(self) -> str:
        return self._lamarck.mode

    def get_speciation_metric(self) -> str:
        return (self._population._speciation_metric if self._population is not None
                else "topology")

    # ── Advanced configuration setters ────────────────────────────────────

    def set_fitness_shaping(self, enabled: bool) -> None:
        """Enable or disable rank-based fitness shaping before selection.

        When enabled, ``shared_fitness`` values are replaced by rank-normalised
        scores in ``[1/N, 1]`` before tournament selection runs.  This makes
        selection pressure scale-invariant — a genome that is 100× better than
        another doesn't dominate 100× more, it just ranks higher.  Useful for
        tasks with very sparse rewards or extreme fitness outliers.

        Default: disabled.
        """
        if self._population is not None:
            self._population._fitness_shaping = enabled

    def set_novelty_search(self, enabled: bool) -> None:
        """Enable or disable novelty search as a selection pressure.

        When disabled, the novelty bonus is zeroed out so selection is driven
        purely by fitness (and efficiency, if configured).  Useful for ablation
        studies comparing fitness-only evolution against novelty-augmented runs.

        Default: enabled.
        """
        if self._population is not None:
            self._population._novelty_enabled = enabled

    def set_speciation(self, enabled: bool) -> None:
        """Enable or disable speciation.

        When disabled, the population is treated as a single species — no
        compatibility threshold, no species budget, no fitness sharing across
        niches.  Useful for ablation studies comparing NEAT-style speciation
        against a flat tournament selection baseline.

        Default: enabled.
        """
        if self._population is not None:
            self._population._speciation_enabled = enabled

    def set_crossover(self, enabled: bool) -> None:
        """Enable or disable sexual reproduction (crossover).

        When disabled, every offspring is produced by copying and mutating a
        single parent — equivalent to a mutation-only EA.  Useful for ablation
        studies quantifying the benefit of NEAT-style crossover.

        Default: enabled.
        """
        if self._population is not None:
            self._population._crossover_enabled = enabled

    def set_diversity_injection(self, enabled: bool) -> None:
        """Enable or disable stagnation-triggered diversity injection.

        When disabled, neither fresh-genome injection nor structural-diversity
        injection fires, even under prolonged stagnation.  Useful for ablation
        studies isolating the effect of the escape mechanism.

        Default: enabled.
        """
        if self._population is not None:
            self._population._diversity_injection_enabled = enabled

    def set_interspecies_crossover(self, rate: float) -> None:
        """Enable interspecies crossover at the given probability.

        When a crossover event fires, this is the probability that the second
        parent is chosen from a *different* species.  At 0.0 (default) all
        crossover is intraspecies (classical NEAT).  A small value such as 0.05
        occasionally combines structural innovations from separate niches.

        Edge case: when only one species exists, falls back to intraspecies
        selection regardless of rate.

        Args:
            rate: Probability in [0.0, 1.0].
        """
        rate = max(0.0, min(1.0, rate))
        self._interspecies_crossover_mode = "fixed"
        self._interspecies_crossover_rate = rate
        self._interspecies_crossover_min = 0.0
        self._interspecies_crossover_max = max(rate, 0.2)
        if self._population is not None:
            self._population.configure_interspecies_crossover(rate, mode="fixed")

    def set_adaptive_interspecies_crossover(
        self,
        min_rate: float = 0.0,
        max_rate: float = 0.2,
        enabled: bool = True,
    ) -> None:
        """Adapt cross-species crossover pressure from stagnation signals.

        The live probability ramps from ``min_rate`` to ``max_rate`` as global
        or per-species stagnation increases.  With fewer than two species the
        live rate is forced to zero.
        """
        lo = max(0.0, min(1.0, min_rate))
        hi = max(lo, min(1.0, max_rate))
        self._interspecies_crossover_mode = "adaptive" if enabled else "fixed"
        self._interspecies_crossover_rate = lo
        self._interspecies_crossover_min = lo
        self._interspecies_crossover_max = hi
        if self._population is not None:
            self._population.configure_interspecies_crossover(
                lo,
                mode=self._interspecies_crossover_mode,
                min_rate=lo,
                max_rate=hi,
            )

    def set_spike_rate(self, rate: float) -> None:
        """Set the initial spike mutation rate for new connections.

        Spike mutation replaces a connection's weight with a fresh random
        sample from N(0, sigma_global) instead of perturbing it — a form
        of "hard reset" that helps escape local optima.  The rate self-adapts
        from this initial value.

        Default: 0.05.  Range: [0.001, 0.3].
        """
        rate = max(0.001, min(0.3, rate))
        # Apply to all existing connections' spike_rate attribute.
        if self._population is not None:
            for genome in self._population._evaluated:
                for node in genome.nodes:
                    for conn in node.connections:
                        conn.spike_rate = rate

    def set_weight_clipping(
        self,
        w_max: float | None = None,
        b_max: float | None = None,
    ) -> None:
        """Clamp all weights and biases after each mutation.

        After a child genome is mutated, every connection weight is clamped to
        ``[-w_max, w_max]`` and every node bias to ``[-b_max, b_max]``.

        This prevents weight explosion in recurrent or deep networks and keeps
        the search space bounded, at the cost of losing very large-weight
        solutions.

        Args:
            w_max: Maximum absolute weight value.  ``None`` (default) disables
                   weight clipping.
            b_max: Maximum absolute bias value.  ``None`` inherits ``w_max``
                   when ``w_max`` is provided; otherwise disables bias clipping.

        Pass both as ``None`` (or call with no arguments) to disable clipping.
        """
        if w_max is None:
            clip = None
        else:
            effective_b = b_max if b_max is not None else w_max
            clip = (float(w_max), float(effective_b))
        if self._population is not None:
            self._population._weight_clip = clip

    def set_output_sanitizing(
        self,
        enabled: bool = True,
        fallback: float = 0.0,
    ) -> None:
        """Replace NaN/Inf values in genome forward-pass outputs with *fallback*.

        Useful when activation functions can overflow (e.g. unbounded outputs,
        very large weights). Does NOT sanitize internal node values — only the
        final output vector is checked, keeping the forward hot-path fast.

        Diagnostic counter ``n_output_sanitized`` in ``population_memory_info()``
        tracks cumulative replacement events across all genomes.

        Args:
            enabled: True (default) to enable sanitizing; False to disable.
            fallback: Replacement value for NaN/Inf outputs (default 0.0).
        """
        self._output_sanitize = enabled
        self._output_fallback = fallback
        if self._population is not None:
            for genome in (self._population._evaluated
                           + list(self._population._unevaluated)):
                genome._output_sanitize = enabled
                genome._output_fallback = fallback

    def set_complexity_penalty(
        self,
        node_penalty: float = 0.0,
        connection_penalty: float = 0.0,
    ) -> None:
        """Enable an optional soft penalty for larger genomes.

        Defaults are zero.  When enabled, finalized fitness is reduced by
        ``node_penalty * hidden_nodes + connection_penalty * connections``.
        """
        self._complexity_penalty_nodes = max(0.0, float(node_penalty))
        self._complexity_penalty_connections = max(0.0, float(connection_penalty))

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

    def set_normalizer(self, normalizer) -> None:
        """Attach a normalizer to this experiment.

        The normalizer is stored alongside the population in every
        ``save_checkpoint()`` call and restored by ``load_checkpoint()``.
        Any object that implements the normalizer interface from
        ``yane.util.normalization`` is accepted (``ScaleNormalizer``,
        ``MinMaxNormalizer``, ``ZScoreNormalizer``, ``ClipNormalizer``,
        ``RunningStatsNormalizer``, or a custom class with the same API).

        Pass ``None`` to detach the current normalizer.
        """
        self._normalizer = normalizer

    def get_normalizer(self):
        """Return the currently attached normalizer, or ``None`` if none is set."""
        return self._normalizer

    def set_multi_objective(
        self,
        enabled: bool = True,
        weights: list[float] | tuple[float, ...] | None = None,
        maximize: list[bool] | tuple[bool, ...] | None = None,
    ) -> None:
        """Enable vector fitness with Pareto-shaped selection.

        Fitness functions may return ``(objective_1, objective_2, ...)`` instead
        of a scalar. The vector is stored on ``genome.objectives``. A weighted
        scalar is still written to ``genome.fitness`` for existing stop criteria,
        logging, and APIs, while parent selection can use Pareto rank and
        crowding distance inside ``Population``.
        """
        self._multi_objective_enabled = enabled
        self._multi_objective_weights = tuple(float(w) for w in weights) if weights is not None else None
        self._multi_objective_maximize = tuple(bool(m) for m in maximize) if maximize is not None else None
        if self._population is not None:
            self._population._multi_objective_enabled = enabled
            self._population._multi_objective_maximize = self._multi_objective_maximize

    def set_quality_diversity(
        self,
        descriptor_fn,
        bins: list[int] | tuple[int, ...],
        ranges: list[tuple[float, float]] | tuple[tuple[float, float], ...],
        enabled: bool = True,
        max_cells: int | None = None,
    ) -> None:
        """Enable MAP-Elites archive updates during training.

        ``descriptor_fn(genome)`` must return a fixed-length behavior descriptor.
        The archive stores the best genome per descriptor cell and uses archived
        elites as one source for diversity injection during stagnation.
        """
        self._qd_enabled = enabled
        self._qd_descriptor_fn = descriptor_fn
        self._qd_archive = MAPElitesArchive(bins=bins, ranges=ranges, max_cells=max_cells)
        self._qd_descriptor_needs_reattach = False
        if self._population is not None:
            self._population._qd_enabled = enabled
            self._population._qd_archive = self._qd_archive
            self._population._qd_descriptor_fn = descriptor_fn

    def get_quality_diversity_archive(self) -> MAPElitesArchive | None:
        """Return the MAP-Elites archive, or None when QD is disabled."""
        return self._qd_archive

    # -------------------------------------------------------------------------
    # Checkpoints
    # -------------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        """Save population state to a checkpoint file.

        Saves the full Population (all evaluated genomes, species, stagnation
        counters) together with the InnovationTracker so training can be resumed
        exactly where it left off.  Also stores the current NeuroEvolution
        configuration dict for reference.

        The file is written atomically (to a ``.tmp`` sibling first) so a crash
        during the write never leaves a corrupt checkpoint.

        Args:
            path: Destination file path (e.g. ``"checkpoints/run1.pkl"``).
        """
        self._ensure_configured()
        _ckpt.write(path, {
            "version": _ckpt.VERSION,
            "config": self._config_dict(),
            "population": self._population,
            "tracker": self._tracker,
            "normalizer": self._normalizer,
            "lamarck_n_applied":     self._lamarck.n_applied,
            "lamarck_n_steps_total": self._lamarck.n_steps_total,
            "lamarck_time_ms":       self._lamarck.time_ms,
            "lamarck_n_blocked_top_k": self._lamarck.n_blocked_top_k,
            "n_invalid_fitness":     self._sanitizer.n_invalid,
            "n_clipped_fitness":     self._sanitizer.n_clipped,
            "n_early_stopped":       self._runner.n_early_stopped,
            "early_stopping_n":      self._runner.early_stopping_n,
            "adaptive_ctrl_enabled":      self._adaptive_ctrl_enabled,
            "adaptive_ctrl":              self._adaptive_ctrl,
            "operator_scheduler_enabled": self._operator_scheduler_enabled,
            "operator_scheduler":         self._operator_scheduler,
            "meta_adaptive_enabled":      self._meta_adaptive_enabled,
            "meta_adaptive":              self._meta_adaptive,
            "module_library":             self._module_library,
            "module_insert_rate":         self._module_insert_rate,
        })

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore population state from a checkpoint file.

        Replaces the current population and innovation tracker with the saved
        state.  The NeuroEvolution configuration (n_inputs, n_outputs, etc.) is
        restored from the checkpoint; any configure() call made before
        load_checkpoint() is overwritten.

        After loading you can call train() or next_genome() immediately.

        Args:
            path: Path to a checkpoint file written by save_checkpoint().

        Raises:
            FileNotFoundError: if *path* does not exist.
            ValueError: if the checkpoint format is unrecognised.
        """
        payload = _ckpt.read(path)
        cfg = payload["config"]
        self._population  = payload["population"]
        self._tracker     = payload["tracker"]
        self._lamarck.n_applied       = payload.get("lamarck_n_applied", 0)
        self._lamarck.n_steps_total   = payload.get("lamarck_n_steps_total", 0)
        self._lamarck.time_ms         = payload.get("lamarck_time_ms", 0.0)
        self._lamarck.n_blocked_top_k = payload.get("lamarck_n_blocked_top_k", 0)
        self._sanitizer.n_invalid     = payload.get("n_invalid_fitness", 0)
        self._sanitizer.n_clipped     = payload.get("n_clipped_fitness", 0)
        self._runner.n_early_stopped  = payload.get("n_early_stopped", 0)
        self._runner.early_stopping_n = payload.get("early_stopping_n", None)
        self._normalizer              = payload.get("normalizer", None)
        if "adaptive_ctrl" in payload:
            self._adaptive_ctrl = payload["adaptive_ctrl"]
        self._adaptive_ctrl_enabled = payload.get("adaptive_ctrl_enabled", False)
        if "operator_scheduler" in payload:
            self._operator_scheduler = payload["operator_scheduler"]
        self._operator_scheduler_enabled = payload.get("operator_scheduler_enabled", False)
        self._meta_adaptive = payload.get("meta_adaptive", None)
        self._meta_adaptive_enabled = payload.get("meta_adaptive_enabled", False)
        self._module_library = payload.get("module_library", None)
        self._module_insert_rate = payload.get("module_insert_rate", 0.0)
        if self._operator_scheduler_enabled and self._population is not None:
            self._population._operator_scheduler = self._operator_scheduler
        if self._population is not None:
            self._population._module_library = self._module_library
            self._population._module_insert_rate = self._module_insert_rate
        # Restore cached config so _config_dict() + logs are accurate.
        self._n_inputs           = cfg.get("n_inputs", self._n_inputs)
        self._n_outputs          = cfg.get("n_outputs", self._n_outputs)
        self._max_nodes          = cfg.get("max_nodes", self._max_nodes)
        self._max_connections    = cfg.get("max_connections", self._max_connections)
        self._n_initial_hidden   = cfg.get("n_initial_hidden", self._n_initial_hidden)
        self._stateful           = cfg.get("stateful", self._stateful)
        self._population_size    = cfg.get("population_size", self._population_size)
        self._seed               = cfg.get("seed", self._seed)
        self._interspecies_crossover_mode = cfg.get(
            "interspecies_crossover_mode",
            getattr(self._population, "_interspecies_crossover_mode", self._interspecies_crossover_mode),
        )
        self._interspecies_crossover_rate = cfg.get(
            "interspecies_crossover_rate",
            getattr(self._population, "_interspecies_crossover_rate", self._interspecies_crossover_rate),
        )
        self._interspecies_crossover_min = cfg.get(
            "interspecies_crossover_min",
            getattr(self._population, "_interspecies_crossover_min", self._interspecies_crossover_min),
        )
        self._interspecies_crossover_max = cfg.get(
            "interspecies_crossover_max",
            getattr(self._population, "_interspecies_crossover_max", self._interspecies_crossover_max),
        )
        self._population.configure_interspecies_crossover(
            self._interspecies_crossover_rate,
            mode=self._interspecies_crossover_mode,
            min_rate=self._interspecies_crossover_min,
            max_rate=self._interspecies_crossover_max,
        )
        self._multi_objective_enabled = cfg.get("multi_objective_enabled", self._multi_objective_enabled)
        weights = cfg.get("multi_objective_weights", self._multi_objective_weights)
        maximize = cfg.get("multi_objective_maximize", self._multi_objective_maximize)
        self._multi_objective_weights = tuple(weights) if weights is not None else None
        self._multi_objective_maximize = tuple(maximize) if maximize is not None else None
        self._population._multi_objective_enabled = self._multi_objective_enabled
        self._population._multi_objective_maximize = self._multi_objective_maximize

    def warm_start_from_checkpoint(
        self,
        path: str | Path,
        fitness_fn: Callable[[Genome], float] | None = None,
        min_fitness: float | None = None,
        reset_strategy: bool = False,
    ) -> int:
        """Use genomes from a checkpoint as the current population.

        This is transfer learning rather than exact resume: call ``configure()``
        for the new task first, then warm-start from a related checkpoint.

        I/O size mismatches are handled automatically:
        - Fewer checkpoint inputs than new task: extra input nodes are appended
          (no connections initially — evolution adds them).
        - More checkpoint inputs than new task: excess input nodes and all their
          outgoing connections are dropped.
        - Same rules apply to output nodes.

        If ``fitness_fn`` is provided, genomes are evaluated on the new task
        immediately and only genomes meeting ``min_fitness`` are kept; otherwise
        they are queued for re-evaluation.

        Returns the number of imported genomes.
        """
        self._ensure_configured()
        payload = _ckpt.read(path)
        source_pop = payload["population"]
        genomes = [g.copy() for g in (source_pop._evaluated + list(source_pop._unevaluated))]
        if not genomes:
            raise ValueError("Checkpoint population is empty")

        kept: list[Genome] = []
        for genome in genomes:
            self._adapt_genome_topology(genome, self._n_inputs, self._n_outputs,
                                        self._tracker, self._stateful)
            genome.max_nodes = self._max_nodes
            genome.max_connections = self._max_connections
            genome.allow_memory = self._stateful
            genome._output_sanitize = self._output_sanitize
            genome._output_fallback = self._output_fallback
            genome._last_species_id = None
            genome._species_stale = True
            if reset_strategy:
                self._reset_genome_strategy(genome)
            if fitness_fn is not None:
                fitness = self._finalize_fitness(
                    self._run_evaluations(genome, fitness_fn).fitness,
                    None,
                    genome,
                )
                if min_fitness is not None and fitness < min_fitness:
                    continue
                genome.fitness = fitness
                genome.shared_fitness = fitness
            kept.append(genome)

        if not kept:
            raise ValueError("No checkpoint genomes passed the warm-start filter")

        self._tracker = payload.get("tracker", self._tracker)
        self._population = Population(
            max_size=self._population_size,
            initial_genome=kept[0].copy(),
            tracker=self._tracker,
            target_species=self.get_target_species(),
        )
        self._population.elite_count = self._elite_count
        self._population.species_elite_count = self._species_elite_count
        self._population._adaptive_pop_enabled = self._adaptive_pop_enabled
        self._population._adaptive_pop_min = self._adaptive_pop_min
        self._population._adaptive_pop_max = self._adaptive_pop_max
        self._population._adaptive_pop_rate = self._adaptive_pop_rate
        self._population._multi_objective_enabled = self._multi_objective_enabled
        self._population._multi_objective_maximize = self._multi_objective_maximize
        if fitness_fn is None:
            self._population._unevaluated = kept[:self._population_size]
            self._population._evaluated = []
        else:
            self._population._evaluated = kept[:self._population_size]
            self._population._unevaluated = []
            for genome in self._population._evaluated:
                self._population._assign_one_genome(genome)
        return len(kept[:self._population_size])

    @staticmethod
    def _reset_genome_strategy(genome: Genome) -> None:
        from yane.core.genome import _MUTATION_GENES, _SCALAR_GENES
        default = Genome()
        for attr in _SCALAR_GENES:
            setattr(genome, attr, getattr(default, attr))
        for attr in _MUTATION_GENES:
            setattr(genome, attr, getattr(default, attr).copy())

    def _adapt_genome_topology(
        self,
        genome: Genome,
        target_inputs: int,
        target_outputs: int,
        tracker,
        stateful: bool,
    ) -> None:
        """Adapt a checkpoint genome's I/O to match the current task's shape.

        Extra input/output nodes are appended (no connections — evolution
        adds them).  Surplus nodes beyond the target count are stripped along
        with all their outgoing (inputs) or incoming (outputs) connections.
        """
        from yane.core.node import NodeType as _NT
        from yane.core.node import Node as _Node
        from yane.util.activation import ActivationType as _AT

        cur_in  = len(genome.input_nodes)
        cur_out = len(genome.output_nodes)

        # ── Inputs ──────────────────────────────────────────────────────────
        if cur_in < target_inputs:
            for i in range(cur_in, target_inputs):
                node = _Node(_NT.INPUT, innovation=tracker.next())
                node.input_index = i
                node.activation = _AT.LINEAR
                genome.nodes.append(node)
                genome.input_nodes.append(node)
        elif cur_in > target_inputs:
            removed = set(genome.input_nodes[target_inputs:])
            genome.input_nodes = genome.input_nodes[:target_inputs]
            genome.nodes = [n for n in genome.nodes if n not in removed]
            # Input nodes own their outgoing connections — removing the node
            # removes all connections originating from it automatically.

        # Fix input_index on all remaining input nodes (may have shifted).
        for idx, node in enumerate(genome.input_nodes):
            node.input_index = idx

        # ── Outputs ──────────────────────────────────────────────────────────
        if cur_out < target_outputs:
            for _ in range(cur_out, target_outputs):
                node = _Node(_NT.OUTPUT, innovation=tracker.next())
                node.persist_value = stateful
                genome.nodes.append(node)
                genome.output_nodes.append(node)
        elif cur_out > target_outputs:
            removed = set(genome.output_nodes[target_outputs:])
            genome.output_nodes = genome.output_nodes[:target_outputs]
            genome.nodes = [n for n in genome.nodes if n not in removed]
            # Remove connections that point to removed output nodes.
            for node in genome.nodes:
                node.connections = [c for c in node.connections
                                    if c.target not in removed]

        if cur_in != target_inputs or cur_out != target_outputs:
            # Clear gate_node references that point to removed nodes.
            live = set(genome.nodes)
            for node in genome.nodes:
                if node.gate_node is not None and node.gate_node not in live:
                    node.gate_node = None
            genome._invalidate_topology()

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
        self._population.submit(
            self._current_genome,
            self._finalize_fitness(fitness, elapsed_ms, self._current_genome),
            elapsed_ms,
        )
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
            self._population.submit(
                genome,
                self._finalize_fitness(fitness, elapsed_ms, genome),
                elapsed_ms,
            )

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
        return self._sanitizer.apply(fitness)

    def _finalize_fitness(
        self,
        fitness,
        elapsed_ms: float | None,
        genome: Genome | None = None,
    ) -> float:
        """Sanitize + efficiency penalty. Applied by every submission path."""
        return finalize_fitness_value(
            fitness,
            elapsed_ms,
            genome,
            FitnessFinalizationConfig(
                sanitizer=self._sanitizer,
                multi_objective_enabled=self._multi_objective_enabled,
                multi_objective_weights=self._multi_objective_weights,
                fitness_component_weights=self._fitness_component_weights,
                efficiency_penalty=self._efficiency_penalty,
                complexity_penalty_nodes=self._complexity_penalty_nodes,
                complexity_penalty_connections=self._complexity_penalty_connections,
            ),
        )

    def _run_evaluations(
        self, genome: Genome, fitness_fn: Callable[[Genome], float]
    ) -> EvaluationResult:
        if self._matrix_forward_enabled:
            return self._run_with_matrix_forward(genome, fitness_fn)
        return self._runner.run(genome, fitness_fn, self._population, self._lamarck)

    def _run_with_matrix_forward(
        self, genome: Genome, fitness_fn: Callable[[Genome], float]
    ) -> EvaluationResult:
        from yane.evolution.matrix_export import is_matrix_compatible, forward_matrix
        patched = False
        if is_matrix_compatible(genome):
            try:
                exported = self._matrix_cache.get(genome)
                _exp = exported

                def _matrix_fwd(data):
                    if type(data) is not list:
                        data = [float(x) for x in data]
                    return forward_matrix(_exp, data)

                genome.__dict__["forward"] = _matrix_fwd
                patched = True
                self._matrix_hits += 1
            except Exception:
                self._matrix_misses += 1
        else:
            self._matrix_misses += 1
        try:
            return self._runner.run(genome, fitness_fn, self._population, self._lamarck)
        finally:
            if patched:
                genome.__dict__.pop("forward", None)

    def _lamarck_refine(
        self,
        genome: Genome,
        fitness_fn: Callable[[Genome], float],
        baseline_fitness: float | None = None,
        n_steps: int | None = None,
    ) -> float:
        return self._lamarck.refine(
            genome, fitness_fn,
            baseline_fitness=baseline_fitness,
            n_steps=n_steps,
        )

    def _adaptive_lamarck_steps(self, genome: Genome, fitness: float) -> int:
        return self._lamarck.adaptive_steps(genome, fitness, self._population)

    def _check_stop_reason(
        self,
        fitness: float,
        iterations: int,
        _li: Callable,
    ) -> str | None:
        if self.min_fitness is not None and fitness >= self.min_fitness:
            _li("Training stopped: target fitness reached  fitness=%.6f  iterations=%d",
                fitness, iterations)
            return "target_reached"
        if self._max_evaluations is not None and self._n_evaluations_done >= self._max_evaluations:
            _li("Training stopped: max evaluations reached  evals=%d  iterations=%d",
                self._n_evaluations_done, iterations)
            return "max_evaluations"
        if self.max_iterations is not None and iterations >= self.max_iterations:
            _li("Training stopped: max iterations reached  iterations=%d", iterations)
            return "max_iterations"
        if (self._convergence_spread_eps is not None
                and iterations % max(1, self._population_size // 5) == 0):
            pop = self._population
            stag_frac = pop.stagnation_count / max(1, pop.stagnation_threshold)
            if stag_frac >= self._convergence_min_stagnation:
                iqr = _compute_fitness_iqr(pop._evaluated)
                if iqr < self._convergence_spread_eps:
                    _li("Training stopped: converged  iqr=%.6f  stagnation=%d  iterations=%d",
                        iqr, pop.stagnation_count, iterations)
                    return "converged"
        return None

    def _write_crash_snapshot(
        self,
        iterations: int,
        mem: dict,
        _li: Callable,
    ) -> None:
        if self._log_run_dir is None:
            return
        try:
            snap = {
                "iteration": iterations,
                "best_fitness": mem.get("max_fitness"),
                "avg_fitness": mem.get("avg_fitness"),
                "fitness_iqr": mem.get("fitness_iqr"),
                "species_count": mem.get("species_count"),
                "stagnation_count": mem.get("stagnation_count"),
                "nodes": mem.get("largest_genome_nodes"),
                "connections": mem.get("largest_genome_connections"),
                "lamarck_n_applied": mem.get("lamarck_n_applied"),
            }
            snap_path = self._log_run_dir / "_crash_state.json"
            tmp = snap_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(snap), encoding="utf-8")
            tmp.replace(snap_path)
        except Exception as e:
            _li("Failed to write crash state snapshot: %s", e)

    def _write_run_summary(
        self,
        name: str,
        stop_reason: str | None,
        iterations: int,
        _wj: Callable,
        _li: Callable,
    ) -> None:
        if self._log_run_dir is None:
            return
        try:
            best = self.get_best()
        except RuntimeError:
            _li("Run summary skipped: population was reset during a curriculum stage advance")
            return
        import pickle
        pkl_path = self._log_run_dir / "best_genome.pkl"
        pkl_path.write_bytes(pickle.dumps(best))
        mem = self.population_memory_info()
        _wj(self._log_run_dir / "summary.json", {
            "run_name": name,
            "stop_reason": stop_reason or "manual",
            "iterations": iterations,
            "best_fitness": best.fitness,
            "best_nodes": len(best.nodes),
            "best_connections": best.connection_count,
            "final_species_count": mem.get("species_count", 0),
            "final_stagnation": mem.get("stagnation_count", 0),
            "n_evaluations_done":       self._n_evaluations_done,
            "lamarck_n_applied":        mem.get("lamarck_n_applied", 0),
            "lamarck_n_steps_total":    mem.get("lamarck_n_steps_total", 0),
            "lamarck_n_blocked_top_k":  mem.get("lamarck_n_blocked_top_k", 0),
            "n_invalid_fitness":        mem.get("n_invalid_fitness", 0),
            "n_clipped_fitness":        mem.get("n_clipped_fitness", 0),
        })
        _li("Training finished  best_fitness=%.6f  nodes=%d  connections=%d  iterations=%d  "
            "stop_reason=%s  evals=%d  lamarck_applied=%d  lamarck_blocked_top_k=%d",
            best.fitness, len(best.nodes), best.connection_count, iterations,
            stop_reason or "manual", self._n_evaluations_done,
            mem.get("lamarck_n_applied", 0), mem.get("lamarck_n_blocked_top_k", 0))

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
