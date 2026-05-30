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
from yane.evolution.eval_middleware import EvalContext, EvalMiddleware, apply_middleware
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
from yane.evolution.events import EventBus
from yane.evolution.fitness_transform import (  # noqa: F401  (re-exported for users)
    RankTransform, SigmaScaling, LinearNormalize, ClipTransform, ChainTransform,
)
from yane.evolution.param_registry import ParamRegistry, build_default_registry
from yane.evolution.auto_train import AutoTrainResult  # noqa: F401  (re-exported for users)

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
        self._hw_constraints = None  # HardwareConstraints | None
        self._aug_pool = None        # AugmentationPool | None
        self._interactive_evaluator = None  # InteractiveEvaluator | None
        self._budget_enforcer = None  # BudgetEnforcer | None
        self._input_grouping_enabled: bool = False
        self._input_grouping_n_raw: int | None = None
        self._input_grouping_n_groups: int | None = None
        self._input_grouping_initial: list | None = None  # list[InputGroup] | None
        self._output_grouping_enabled: bool = False
        self._output_grouping_n_outputs: int | None = None
        self._output_grouping_n_proto: int | None = None
        self._output_grouping_initial: list | None = None  # list[OutputGroup] | None
        self._conv_neat_enabled: bool = False
        self._conv_neat_stack = None  # ConvStack | None (template for new genomes)
        self._attention_enabled: bool = False
        self._attention_block = None  # AttentionBlock | None (template)
        self._ltc_enabled: bool = False
        self._neuromodulation_enabled: bool = False
        self._adversarial_system = None  # AdversarialSystem | None
        self._cooperative_system = None  # CooperativeSystem | None
        self._minimal_criterion = None  # MinimalCriterion | None
        self._open_ended_mode: str | None = None
        self._continual_learner = None  # ContinualLearner | None
        self._task_evaluators: list[tuple[str, object]] = []  # (name, evaluator)
        self._stdp_enabled: bool = False
        self._stdp_weight_min: float = -5.0
        self._stdp_weight_max: float = 5.0
        self._stdp_hebb_sigma: float = 0.05
        self._hybrid_mode_enabled: bool = False
        self._hybrid_bp_interval: int = 10
        self._hybrid_bp_epochs: int = 50
        self._hybrid_bp_lr: float = 0.01
        self._hybrid_bp_batch_size: int = 32
        self._hybrid_top_k: int = 3
        self._hybrid_train_data: list | None = None  # list[(inputs, targets)] | None
        self._hybrid_replay_buffer = None  # ReplayBuffer | None
        self._complexity_penalty_nodes: float = 0.0
        self._complexity_penalty_connections: float = 0.0
        self._resource_guard = ResourceGuard()
        self._resource_check_interval: int = 50  # check psutil every N iters (was 1 = ~5% overhead)
        self._population_size: int = 100
        self._target_species: int | None = 5
        self._target_species_min: int | None = None
        self._target_species_max: int | None = None
        self._compat_tune_interval: int = 1
        self._compat_threshold_min: float = 0.01
        self._compat_threshold_max: float = 1.5
        self._adaptive_pop_enabled: bool = False
        self._adaptive_pop_min: int = 100
        self._adaptive_pop_max: int = 100
        self._adaptive_pop_rate: float = 0.05
        self._adaptive_pop_schedule: str = "performance_based"
        self._island_model: Any = None
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
        self._eval_middlewares: list[EvalMiddleware] = []
        self._eval_middleware_diagnostics: dict = {}
        # Automatic checkpoint rolling
        self._checkpoint_policy_enabled: bool = False
        self._checkpoint_interval: int = 100
        self._checkpoint_keep_best: bool = True
        self._checkpoint_max_keep: int = 5
        self._checkpoint_path_template: str = "{run_name}_{iteration}.pkl"
        self._checkpoint_paths: list[str] = []
        self._best_checkpoint_path: str | None = None
        self._last_checkpoint_iteration: int | None = None
        self._last_checkpoint_best_fitness: float = -float("inf")
        self._last_checkpoint_compatibility: dict | None = None
        # Elitism (applied to population on configure())
        self._elite_count: int = 1
        self._species_elite_count: int = 1
        # Fitness sanitizing (disabled by default)
        self._sanitizer = FitnessSanitizer()
        # Output sanitizing — replaces NaN/Inf in forward() results (disabled by default)
        self._output_sanitize: bool = False
        self._output_fallback: float = 0.0
        # Probabilistic / Bayesian NEAT (disabled by default)
        self._prob_enabled: bool = False
        self._prob_noise_std: float = 0.05
        self._prob_inference_mode: bool = False
        # Safety-Constrained Evolution (Safe NEAT)
        self._safety_system = None  # SafetySystem | None
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
        self._phylogeny: "PhylogenyTree | None" = None
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
        self._tracking_backends: list = []      # TrackingBackend instances
        # Event system
        self._event_bus: EventBus = EventBus()
        # Anomaly detection (None = disabled)
        self._anomaly_detectors = None
        # Fitness transform applied at each generation boundary (None = disabled)
        self._fitness_transform = None
        self._auto_fitness_shaping_enabled: bool = False
        self._auto_fitness_shaping_report = None
        # Online hyperparameter tuning (UCB1 bandit)
        self._online_tuning_enabled: bool = False
        self._online_tuning_bandits: dict[str, Any] = {}
        self._online_tuning_last_best: float | None = None
        self._online_tuning_original_struct_floor: float | None = None
        # Validation (None = disabled)
        self._validation_fn: Callable | None = None
        self._last_validation_fitness: float | None = None
        # Track best fitness for "new_best" events
        self._last_best_fitness: float = -float("inf")
        # Weight inheritance (fitness-weighted crossover blending)
        self._weight_inheritance_enabled: bool = False
        self._weight_blend_alpha: float = 0.7
        # Weight-health N-generation streak tracking
        self._weight_warning_streak: int = 0
        self._weight_warning_consecutive_threshold: int = 5
        # Adaptive Policy System registry
        self._policy_registry: Any = None  # lazy-imported PolicyRegistry
        self._policy_tick_enabled: bool = False
        # Adaptive recovery / guarded early stop
        self._adaptive_recovery_enabled: bool = False
        self._recovery_strategies: list[str] = ["diversity_boost", "partial_restart", "lamarck_burst"]
        self._recovery_cooldown: int = 20
        self._recovery_escalate: bool = True
        self._recovery_diversity_iqr_threshold: float = 1e-4
        self._recovery_injection_frac: float = 0.1
        self._recovery_early_stopping_patience: int = 500
        self._recovery_warmup: int = 100
        self._recovery_min_delta: float = 1e-4
        self._recovery_events: list[dict] = []
        self._last_recovery_generation: int = -10**9
        self._recovery_strategy_index: int = 0
        self._recovery_successes: int = 0
        self._recovery_checked: int = 0
        self._pending_recoveries: list[dict] = []
        self._recovery_best_fitness: float = -float("inf")
        self._recovery_last_improvement_generation: int = 0
        self._recovery_last_lamarck_budget: int | None = None
        self.stopped_early: bool = False
        self.stop_reason: str = ""
        # Experiment tracking (None = disabled, no overhead)
        self._run_database = None          # RunDatabase | None
        self._active_run_id: str | None = None
        self._active_experiment_id: str | None = None
        # Selection strategy (None = population uses its own default TournamentSelection)
        self._selection_strategy = None
        self._selection_strategies_by_species: dict = {}
        # Compatibility distance (None = population uses its own default TopologyDistance)
        self._compatibility_distance = None
        # Input transform applied to genome.forward() inputs (None = disabled)
        self._input_transform = None
        self._n_raw_inputs: int | None = None
        # Run report autosave (None = disabled)
        self._report_autosave_template: str | None = None
        self._report_autosave_format: str = "html"
        self._transfer_freeze_records: dict[int, dict[str, float]] = {}
        self._transfer_frozen_layers: list[str] = []
        self._transfer_unfreeze_enabled: bool = False
        self._transfer_unfreeze_start_generation: int = 0
        self._transfer_unfreeze_generations: int = 0
        self._transfer_unfreeze_progress: float = 0.0
        # Post-training pruning
        self._post_pruning_enabled: bool = False
        self._post_pruning_threshold: float = 0.01
        self._post_pruning_max_drop_frac: float = 0.02
        # Curiosity (intrinsic reward)
        self._curiosity_enabled: bool = False
        self._curiosity_weight: float = 0.3
        self._curiosity_network_size: int = 8
        self._curiosity_lr: float = 0.01
        self._curiosity_module = None   # IntrinsicCuriosityModule, created lazily
        # DARTS-Lite (differentiable architecture search)
        self._darts_enabled: bool = False
        self._darts_prune_threshold: float = 0.1
        # Shared Weights
        self._shared_weights_enabled: bool = False
        # Unified Parameter Registry (lazy-init on first access)
        self._param_registry: ParamRegistry | None = None
        # Knowledge Base (None = disabled)
        self._knowledge_base: Any = None   # KnowledgeBase | None
        self._last_problem_profile: Any = None   # ProblemProfile | None
        # MetaOptimizer (None = disabled)
        self._meta_optimizer_obj: Any = None   # MetaOptimizer | None
        self._meta_optimizer_enabled: bool = False
        # Feature Gating (None = disabled)
        self._feature_gate: Any = None   # FeatureGate | None
        self._feature_gate_enabled: bool = False

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

        # Conv NEAT: attach a conv_stack to the initial genome if configured.
        if self._conv_neat_enabled and self._conv_neat_stack is not None:
            initial.conv_stack = self._conv_neat_stack.copy()

        # Attention: attach an attention_block to the initial genome if configured.
        if self._attention_enabled and self._attention_block is not None:
            initial.attention_block = self._attention_block.copy()

        # Output Grouping: attach an out_grouper to the initial genome if configured.
        # The out_grouper maps n_outputs proto-outputs → _n_external_outputs expanded outputs.
        if self._output_grouping_enabled:
            from yane.evolution.output_grouping import OutputGrouper, OutputGroup
            n_ext = self._output_grouping_n_outputs or n_outputs
            initial.out_grouper = OutputGrouper(
                n_outputs=n_ext,
                initial_groups=self._output_grouping_initial,
            )
            # Trim/rebuild groups to match n_outputs (the configured network output count)
            if len([g for g in initial.out_grouper.groups if g.enabled]) != n_outputs:
                per = n_ext / max(1, n_outputs)
                import math as _math
                groups = []
                for k in range(n_outputs):
                    start = int(round(k * per))
                    end = int(round((k + 1) * per))
                    targets = list(range(start, min(end, n_ext))) or [k % n_ext]
                    groups.append(OutputGroup(targets=targets))
                initial.out_grouper = OutputGrouper(n_outputs=n_ext, initial_groups=groups)

        # Input Grouping: attach a grouper to the initial genome if configured.
        # The grouper maps _n_raw inputs → n_inputs grouped inputs.
        if self._input_grouping_enabled:
            from yane.evolution.input_grouping import InputGrouper
            n_raw = self._input_grouping_n_raw or n_inputs
            initial.grouper = InputGrouper(
                n_raw=n_raw,
                initial_groups=self._input_grouping_initial,
            )
            # Trim groups to match n_inputs (the configured network input count)
            if len([g for g in initial.grouper.groups if g.enabled]) != n_inputs:
                # Rebuild with n_inputs groups distributing n_raw evenly
                import math
                groups = []
                per = n_raw / max(1, n_inputs)
                for k in range(n_inputs):
                    start = int(round(k * per))
                    end = int(round((k + 1) * per))
                    members = list(range(start, min(end, n_raw))) or [k % n_raw]
                    from yane.evolution.input_grouping import InputGroup, AggType
                    groups.append(InputGroup(members=members))
                initial.grouper = InputGrouper(n_raw=n_raw, initial_groups=groups)

        self._population = Population(
            max_size=self._population_size,
            initial_genome=initial,
            tracker=tracker,
            target_species=self._target_species,
        )
        self._apply_speciation_tuning_config(self._population)
        self._population.elite_count = self._elite_count
        self._population.species_elite_count = self._species_elite_count
        self._population._adaptive_pop_enabled = self._adaptive_pop_enabled
        self._population._adaptive_pop_min = self._adaptive_pop_min
        self._population._adaptive_pop_max = self._adaptive_pop_max
        self._population._adaptive_pop_rate = self._adaptive_pop_rate
        self._population._adaptive_pop_schedule = self._adaptive_pop_schedule
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
        # Apply selection strategy overrides if set before configure().
        if self._selection_strategy is not None:
            self._population.selection_strategy = self._selection_strategy
        if self._selection_strategies_by_species:
            self._population.selection_strategies_by_species = dict(
                self._selection_strategies_by_species
            )
        # Apply compatibility distance override if set before configure().
        if self._compatibility_distance is not None:
            self._population.compatibility_distance = self._compatibility_distance
        # Apply weight inheritance.
        self._population._weight_blend_alpha = (
            self._weight_blend_alpha if self._weight_inheritance_enabled else -1.0
        )
        # Curiosity: create forward model now that dimensions are known.
        if self._curiosity_enabled and self._curiosity_module is None:
            from yane.evolution.experimental import IntrinsicCuriosityModule
            self._curiosity_module = IntrinsicCuriosityModule(
                self._n_inputs, self._n_outputs,
                self._curiosity_network_size, self._curiosity_lr,
            )
        # Validate input transform dimensions.
        if self._input_transform is not None and self._n_raw_inputs is not None:
            try:
                _dummy_out = self._input_transform([0.0] * self._n_raw_inputs)
                if len(_dummy_out) != self._n_inputs:
                    raise ValueError(
                        f"set_input_transform: transform output length {len(_dummy_out)}"
                        f" != n_inputs {self._n_inputs} (n_raw_inputs={self._n_raw_inputs})"
                    )
            except (TypeError, AttributeError):
                pass

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
        if n < 1:
            raise ValueError(f"population size must be >= 1, got {n}")
        self._population_size = n
        if self._population is not None:
            self._population.max_size = n

    def _apply_speciation_tuning_config(self, population: Population) -> None:
        population._target_species = self._target_species if self._target_species is not None else 0
        population._species_tuning_enabled = self._target_species is not None
        population._target_species_min = self._target_species_min
        population._target_species_max = self._target_species_max
        population._compat_tune_interval = max(1, self._compat_tune_interval)
        population._compat_threshold_min = self._compat_threshold_min
        population._compat_threshold_max = self._compat_threshold_max

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
            self._population._adaptive_pop_schedule = self._adaptive_pop_schedule

    def set_adaptive_pop_size(
        self,
        min_pop: int,
        max_pop: int,
        schedule: str = "performance_based",
        growth_rate: float = 0.05,
        enabled: bool = True,
    ) -> None:
        """Enable adaptive population sizing with an explicit schedule.

        This is the spec-compliant API replacing ``set_adaptive_population()``.
        Both methods update the same underlying configuration; the difference is
        this one exposes the *schedule* parameter directly.

        Args:
            min_pop:     Minimum allowed population size.
            max_pop:     Maximum allowed population size.
            schedule:    ``"linear_decay"`` — monotonically shrinks from
                         *max_pop* towards *min_pop* over the training run
                         (derived from ``max_iterations`` when set, otherwise
                         a fixed step of *growth_rate* per generation).

                         ``"performance_based"`` — grows on stagnation / low
                         diversity, shrinks when convergence is healthy.
            growth_rate: Fractional step size used by ``performance_based``
                         and as the per-generation step for ``linear_decay``
                         when ``max_iterations`` is unknown (default 0.05).
            enabled:     Pass False to disable without losing config.
        """
        _valid = {"linear_decay", "performance_based"}
        if schedule not in _valid:
            raise ValueError(
                f"schedule must be one of {_valid!r}, got {schedule!r}"
            )
        self._adaptive_pop_schedule = schedule
        self.set_adaptive_population(
            min_size=min_pop,
            max_size=max_pop,
            growth_rate=growth_rate,
            enabled=enabled,
        )
        if self._population is not None:
            self._population._adaptive_pop_schedule = schedule
            # For linear_decay, derive total generations from max_iterations if available
            if schedule == "linear_decay" and self.max_iterations is not None:
                pop_size = self._population.max_size
                self._population._adaptive_pop_total_gens = max(
                    1, self.max_iterations // max(1, pop_size)
                )

    def set_weight_inheritance(
        self,
        enabled: bool = True,
        blend_alpha: float = 0.7,
    ) -> None:
        """Enable fitness-weighted blending for crossover.

        When enabled, matching genes between parents are blended instead of
        randomly picked from either parent:

            weight = blend_alpha * w_fitter + (1 - blend_alpha) * w_weaker

        This preserves Lamarck-optimised weight information and can accelerate
        convergence on tasks where fine-tuned weights matter.

        Args:
            enabled:    Pass False to disable (reverts to 50/50 random pick).
            blend_alpha: Weight given to the fitter parent (default 0.7).
                         Must be in [0.0, 1.0].
        """
        if not 0.0 <= blend_alpha <= 1.0:
            raise ValueError(f"blend_alpha must be in [0, 1]; got {blend_alpha}")
        self._weight_inheritance_enabled = enabled
        self._weight_blend_alpha = blend_alpha
        if self._population is not None:
            self._population._weight_blend_alpha = blend_alpha if enabled else -1.0

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

    def set_adaptive_recovery(
        self,
        enabled: bool = True,
        strategies: list[str] | None = None,
        cooldown: int = 20,
        escalate: bool = True,
        diversity_iqr_threshold: float = 1e-4,
        injection_frac: float = 0.1,
        early_stopping_patience: int = 500,
        warmup: int = 100,
        min_delta: float = 1e-4,
    ) -> None:
        """Enable stagnation/diversity recovery with guarded early stopping.

        Recovery is intentionally conservative: it only acts after warmup and
        cooldown windows, and early stopping requires both long stagnation and a
        diversity/recovery-failure signal.
        """
        valid = {"diversity_boost", "partial_restart", "lamarck_burst"}
        chosen = list(strategies or ["diversity_boost", "partial_restart", "lamarck_burst"])
        unknown = [s for s in chosen if s not in valid]
        if unknown:
            raise ValueError(f"Unknown recovery strategies: {unknown}")
        self._adaptive_recovery_enabled = bool(enabled)
        self._recovery_strategies = chosen
        self._recovery_cooldown = max(1, int(cooldown))
        self._recovery_escalate = bool(escalate)
        self._recovery_diversity_iqr_threshold = max(0.0, float(diversity_iqr_threshold))
        self._recovery_injection_frac = max(0.0, min(1.0, float(injection_frac)))
        self._recovery_early_stopping_patience = max(1, int(early_stopping_patience))
        self._recovery_warmup = max(0, int(warmup))
        self._recovery_min_delta = max(0.0, float(min_delta))

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

    def set_budget(
        self,
        preset: str | None = None,
        total_time: "str | float | None" = None,
        max_memory: "str | int | None" = None,
        max_cpu_pct: float | None = None,
        target_platform: str | None = None,
    ) -> None:
        """Set a unified resource budget for training.

        Pass ``"auto"`` (or ``preset="auto"``) to auto-calibrate from the
        current hardware.  Otherwise specify individual limits using
        human-readable strings::

            yane.set_budget("auto")
            yane.set_budget(total_time="30min")
            yane.set_budget(total_time="1h", max_memory="auto")
            yane.set_budget(total_time="2h",
                            max_memory="4GB",
                            target_platform="cortex-m4")

        Parameters
        ----------
        preset :
            ``"auto"`` auto-detects available RAM and uses 80 % as the
            memory limit.  Any other string raises ``ValueError``.
        total_time :
            Wall-clock time limit.  Strings like ``"30min"``, ``"1h"``,
            ``"45s"`` are accepted, as well as bare seconds as float/int.
        max_memory :
            Per-process RSS cap.  ``"auto"`` → 80 % of available RAM;
            ``"4GB"`` → 4 000 000 000 bytes; ``"80%"`` → 80 % of available
            RAM.
        max_cpu_pct :
            CPU-usage ceiling (stored for diagnostics, not enforced).
        target_platform :
            Deployment target forwarded to any active
            :class:`~yane.evolution.hardware_aware.HardwareConstraints`.
        """
        from yane.evolution.resource_budget import (
            BudgetConfig, BudgetEnforcer, ResourceDiscovery,
            parse_time, parse_memory,
        )
        if isinstance(preset, str) and preset.strip().lower() == "auto":
            config = BudgetConfig(
                total_time_seconds=None,
                max_memory_bytes=ResourceDiscovery.auto_memory_budget(0.80),
                max_cpu_pct=75.0,
                target_platform=target_platform,
                auto=True,
            )
        elif preset is not None:
            raise ValueError(f"Unknown budget preset: {preset!r}. Use 'auto' or keyword args.")
        else:
            config = BudgetConfig(
                total_time_seconds=parse_time(total_time),
                max_memory_bytes=parse_memory(max_memory),
                max_cpu_pct=max_cpu_pct,
                target_platform=target_platform,
                auto=False,
            )
        self._budget_enforcer = BudgetEnforcer(config, ne_ref=self)
        # Propagate memory budget to ResourceGuard for consistent enforcement
        if config.max_memory_bytes is not None:
            max_gb = config.max_memory_bytes / 1_073_741_824
            self._resource_guard = ResourceGuard(max_process_gb=max_gb)

    def budget_status(self) -> dict:
        """Return the current resource-budget status dict.

        Returns an empty dict when no budget has been configured via
        :meth:`set_budget`.

        Example output::

            {
                "elapsed_seconds": 47.3,
                "time_budget_seconds": 1800.0,
                "time_remaining_seconds": 1752.7,
                "time_fraction_used": 0.026,
                "memory_budget_bytes": 8000000000,
                "memory_current_bytes": 312000000,
                "degradation_level": 0,
                "degradation_actions": [],
                "stop_requested": False,
                "emergency_checkpoint": None,
                "target_platform": None,
            }
        """
        if self._budget_enforcer is None:
            return {}
        return self._budget_enforcer.status()

    def set_checkpoint_policy(
        self,
        interval: int = 100,
        keep_best: bool = True,
        max_keep: int = 5,
        path_template: str = "{run_name}_{iteration}.pkl",
        enabled: bool = True,
    ) -> None:
        """Automatically save rolling checkpoints during training.

        ``path_template`` may use ``{run_name}``, ``{iteration}``,
        ``{generation}``, ``{best_fitness}`` and ``{kind}``. Relative paths are
        resolved inside the current run log directory when available.
        """
        self._checkpoint_policy_enabled = bool(enabled)
        self._checkpoint_interval = max(1, int(interval))
        self._checkpoint_keep_best = bool(keep_best)
        self._checkpoint_max_keep = max(1, int(max_keep))
        self._checkpoint_path_template = path_template
        self._checkpoint_paths = []
        self._best_checkpoint_path = None
        self._last_checkpoint_iteration = None
        self._last_checkpoint_best_fitness = -float("inf")

    def get_best_checkpoint_path(self) -> str | None:
        """Return the path of the best auto-saved checkpoint, if any."""
        return self._best_checkpoint_path

    def set_efficiency_penalty(self, max_ms: float, penalty_per_ms: float) -> None:
        self._efficiency_penalty = EfficiencyPenalty(max_ms, penalty_per_ms)

    def set_hardware_constraints(
        self,
        max_flops: int | None = None,
        max_memory_bytes: int | None = None,
        max_latency_us: float | None = None,
        target_platform: str = "desktop",
        penalty_scale: float = 1.0,
        bytes_per_node: int = 8,
        bytes_per_connection: int = 8,
    ) -> None:
        """Evolve genomes under deployment-hardware constraints.

        Each generation, the hardware cost of every genome is estimated from
        its topology (FLOPs, memory bytes, latency).  Genomes that exceed the
        configured budget receive a penalty subtracted from their fitness,
        proportional to the violation.  NEAT will naturally prefer smaller,
        faster networks without needing a separate objective.

        Parameters
        ----------
        max_flops:          Maximum floating-point operations per forward pass.
        max_memory_bytes:   Maximum on-device footprint in bytes.
        max_latency_us:     Maximum inference latency in microseconds.
        target_platform:    Platform profile for latency estimation.
                            Available: ``"cortex-m4"``, ``"cortex-m7"``,
                            ``"esp32"``, ``"raspberry-pi-zero"``,
                            ``"raspberry-pi-4"``, ``"desktop"``,
                            ``"mobile-arm"``.
        penalty_scale:      Multiply the violation fraction by this value.
                            Default 1.0: a 10% FLOP overshoot → −0.1 fitness.
        bytes_per_node:     Bytes in the minimal C-struct per node (default 8).
        bytes_per_connection: Bytes per connection in the deployment struct (default 8).

        Call ``hardware_profile(genome)`` to inspect metrics for any genome.
        Call ``hw_pareto_front()`` to get the non-dominated (fitness, cost) set.
        """
        from yane.evolution.hardware_aware import HardwareConstraints
        self._hw_constraints = HardwareConstraints(
            max_flops=max_flops,
            max_memory_bytes=max_memory_bytes,
            max_latency_us=max_latency_us,
            target_platform=target_platform,
            penalty_scale=penalty_scale,
            bytes_per_node=bytes_per_node,
            bytes_per_connection=bytes_per_connection,
        )

    def hardware_profile(self, genome, target_platform: str | None = None):
        """Return hardware cost metrics for *genome*.

        Returns a :class:`~yane.evolution.hardware_aware.HardwareMetrics`
        dataclass with ``flops``, ``memory_bytes``, and ``latency_us``.

        *target_platform* overrides the platform set via
        :meth:`set_hardware_constraints` (or defaults to ``"desktop"``).
        """
        from yane.evolution.hardware_aware import (
            compute_hardware_metrics, HardwareConstraints,
        )
        constraints = self._hw_constraints
        if target_platform is not None or constraints is None:
            constraints = HardwareConstraints(
                target_platform=target_platform or "desktop",
            )
        return compute_hardware_metrics(genome, constraints)

    def hw_pareto_front(self):
        """Return the non-dominated (fitness × hardware-cost) Pareto front.

        Evaluates all currently-evaluated genomes and returns the subset where
        no other genome is simultaneously higher-fitness AND cheaper on all
        hardware metrics.

        Each element in the returned list is a
        ``(genome, HardwareMetrics)`` tuple, sorted by descending fitness.
        Raises ``RuntimeError`` if no population exists or if hardware
        constraints have not been configured.
        """
        if self._population is None:
            raise RuntimeError("No active population — run configure() first.")
        if self._hw_constraints is None:
            raise RuntimeError("Call set_hardware_constraints() first.")
        from yane.evolution.hardware_aware import hw_pareto_front as _front
        evaluated = self._population._evaluated
        if not evaluated:
            return []
        return _front(list(evaluated), self._hw_constraints)

    def set_evolutionary_augmentation(
        self,
        enabled: bool = True,
        augmentation_space: list[str] | None = None,
        population_augmentations: int = 8,
        pipeline_length: int = 3,
        evolution_interval: int = 20,
        mutation_sigma: float = 0.1,
    ) -> None:
        """Co-evolve input augmentation pipelines alongside the genome population.

        When active, every call to ``genome.forward(inputs)`` during training
        transparently applies the current best augmentation pipeline to
        ``inputs`` before forwarding them through the network.  A small pool
        of candidate pipelines evolves via UCB1 selection and genetic
        operators (crossover + mutation), guided by the NEAT population's
        per-generation fitness improvement.

        This acts as a learned regulariser for small datasets: augmentation
        introduces variety in the training distribution without requiring
        labelled examples.

        Parameters
        ----------
        enabled:                Toggle augmentation on or off.
        augmentation_space:     Subset of transformations to use.
                                Defaults to all: ``["gaussian_noise",
                                "dropout_noise", "scaling", "translation",
                                "cutout"]``.
        population_augmentations: Number of candidate pipelines in the pool.
        pipeline_length:        Number of transformations per pipeline.
        evolution_interval:     Evolve the pool every N generations.
        mutation_sigma:         Std-dev for Gaussian mutation of probability
                                and magnitude parameters.
        """
        if not enabled:
            self._aug_pool = None
            return
        from yane.evolution.augmentation import AugmentationPool, AUGMENTATION_TYPES
        space = augmentation_space or AUGMENTATION_TYPES
        self._aug_pool = AugmentationPool(
            augmentation_space=space,
            population_size=population_augmentations,
            pipeline_length=pipeline_length,
            evolution_interval=evolution_interval,
            mutation_sigma=mutation_sigma,
            seed=self._seed,
        )

    def get_augmentation_diagnostics(self) -> dict:
        """Return diagnostics for the active augmentation pool.

        Returns ``{"enabled": False}`` when augmentation is not configured.
        """
        if self._aug_pool is None:
            return {"enabled": False}
        d = self._aug_pool.get_diagnostics()
        d["enabled"] = True
        return d

    def set_interactive_evaluation(
        self,
        evaluator: "InteractiveEvaluator | None" = None,
        mode: str = "rating",
        surrogate_model: bool = True,
        surrogate_update_interval: int = 5,
    ) -> "InteractiveEvaluator":
        """Enable Human-in-the-Loop evaluation.

        Attaches an :class:`~yane.evolution.interactive_eval.InteractiveEvaluator`
        that collects human feedback (ratings, pairwise comparisons, rankings)
        as the fitness signal.  Use :meth:`submit_feedback` to deliver feedback
        from the GUI or programmatically, or call
        :meth:`~yane.evolution.interactive_eval.InteractiveEvaluator.set_feedback_source`
        on the returned evaluator to attach a synchronous oracle.

        Parameters
        ----------
        evaluator :
            An already-configured :class:`~yane.evolution.interactive_eval.InteractiveEvaluator`
            instance.  When *None* (default), a new one is created from the
            remaining keyword arguments.
        mode :
            Feedback collection mode (``"rating"``, ``"pairwise"``,
            ``"ranking"``, ``"implicit"``).  Ignored when *evaluator* is given.
        surrogate_model :
            Enable a lightweight linear surrogate to predict ratings and
            reduce the number of required human queries.
        surrogate_update_interval :
            Unused parameter kept for API symmetry; the surrogate updates
            automatically after every real human query.

        Returns
        -------
        InteractiveEvaluator
            The evaluator attached to this instance (create a reference to
            call ``submit_feedback`` on it directly).

        Example
        -------
        ::

            from yane.evolution.interactive_eval import InteractiveEvaluator
            eval = InteractiveEvaluator(mode="rating")
            eval.set_feedback_source(lambda g: oracle(g))
            yane.set_interactive_evaluation(eval)
            yane.train(eval)
        """
        from yane.evolution.interactive_eval import InteractiveEvaluator as _IE
        if evaluator is None:
            evaluator = _IE(
                mode=mode,
                surrogate_model=surrogate_model,
            )
        self._interactive_evaluator = evaluator
        return evaluator

    def submit_feedback(self, genome_id: int, value: float) -> None:
        """Deliver human feedback for a genome awaiting evaluation.

        Parameters
        ----------
        genome_id :
            The ``_genome_id`` attribute of the genome being rated.
        value :
            For ``"rating"`` / ``"implicit"``: fitness score (0–100).
            For ``"pairwise"``: 0 if this genome won, 1 if opponent won.
            For ``"ranking"``: rank position (1 = best).

        Raises
        ------
        RuntimeError
            When no :class:`~yane.evolution.interactive_eval.InteractiveEvaluator`
            has been configured via :meth:`set_interactive_evaluation`.
        """
        if self._interactive_evaluator is None:
            raise RuntimeError(
                "No InteractiveEvaluator configured. "
                "Call set_interactive_evaluation() first."
            )
        self._interactive_evaluator.submit_feedback(genome_id, value)

    def set_input_grouping(
        self,
        enabled: bool = True,
        n_groups: int | None = None,
        n_raw: int | None = None,
        initial_groups: "list | None" = None,
    ) -> None:
        """Enable evolvable input aggregation (Input-Gruppierung).

        Groups raw input channels into *K* aggregated inputs before they reach
        the network.  Useful for high-dimensional inputs where direct NEAT
        connectivity would create an explosion of connections.

        Call **before** :meth:`configure` so the network topology is built with
        the correct (reduced) number of input nodes.

        Parameters
        ----------
        enabled :
            Toggle input grouping on or off.
        n_groups :
            Number of output groups (*K*).  Defaults to ``n_raw`` (identity
            mapping with individual groups per raw input).
        n_raw :
            Total number of raw input channels (*N*).  If *None*, inferred
            from ``n_inputs`` at :meth:`configure` time.
        initial_groups :
            List of :class:`~yane.evolution.input_grouping.InputGroup` objects
            for a custom initial layout.  When *None*, each raw input gets its
            own single-member group.

        Example
        -------
        ::

            from yane.evolution.input_grouping import InputGroup, AggType
            yane.set_input_grouping(n_groups=4, n_raw=16)
            yane.configure(n_inputs=4, n_outputs=2)   # network sees 4 grouped inputs
            yane.train(lambda g: eval_fn(g, raw_data))
        """
        if not enabled:
            self._input_grouping_enabled = False
            return
        from yane.evolution.input_grouping import InputGroup as _IG  # noqa: F401
        self._input_grouping_enabled = True
        self._input_grouping_n_raw = n_raw
        self._input_grouping_n_groups = n_groups
        self._input_grouping_initial = list(initial_groups) if initial_groups else None

    def get_input_grouping_diagnostics(self) -> dict:
        """Return diagnostics for the active input grouping configuration.

        Returns ``{"enabled": False}`` when input grouping is not active.
        """
        if not self._input_grouping_enabled:
            return {"enabled": False}
        pop = getattr(self, "_population", None)
        sample = pop.get_best() if (pop and pop._evaluated) else None
        grouper = getattr(sample, "grouper", None) if sample else None
        return {
            "enabled": True,
            "n_raw": self._input_grouping_n_raw,
            "n_groups": grouper.n_outputs if grouper else self._input_grouping_n_groups,
            "groups": [
                {"members": g.members, "agg": g.aggregation.value, "enabled": g.enabled}
                for g in grouper.groups
            ] if grouper else [],
        }

    def set_output_grouping(
        self,
        enabled: bool = True,
        n_proto: int | None = None,
        n_outputs: int | None = None,
        initial_groups: "list | None" = None,
    ) -> None:
        """Enable evolvable output expansion (Output-Gruppierung).

        The genome internally has *K* proto-output nodes.  Callers always
        receive *N* output values — the :class:`~yane.evolution.output_grouping.OutputGrouper`
        expands the network's *K* proto-outputs to *N* external outputs.

        Call **before** :meth:`configure` so the topology is built with the
        correct (reduced) number of output nodes.

        Parameters
        ----------
        enabled :
            Toggle output grouping on or off.
        n_proto :
            Number of internal output nodes (*K*).  When *None*, equals
            ``n_outputs`` (identity mapping).
        n_outputs :
            Number of external output channels (*N*).  When *None*, inferred
            from ``n_outputs`` at :meth:`configure` time.
        initial_groups :
            Custom initial group layout.

        Example
        -------
        ::

            yane.set_output_grouping(n_proto=2, n_outputs=4)
            yane.configure(n_inputs=3, n_outputs=2)  # 2 proto-outputs → 4 external
            yane.train(lambda g: eval_fn(g))          # forward() returns 4 values
        """
        if not enabled:
            self._output_grouping_enabled = False
            return
        self._output_grouping_enabled = True
        self._output_grouping_n_proto = n_proto
        self._output_grouping_n_outputs = n_outputs
        self._output_grouping_initial = list(initial_groups) if initial_groups else None

    def get_output_grouping_diagnostics(self) -> dict:
        """Return diagnostics for the active output grouping configuration.

        Returns ``{"enabled": False}`` when output grouping is not active.
        """
        if not self._output_grouping_enabled:
            return {"enabled": False}
        pop = getattr(self, "_population", None)
        sample = pop.get_best() if (pop and pop._evaluated) else None
        grouper = getattr(sample, "out_grouper", None) if sample else None
        return {
            "enabled": True,
            "n_outputs": self._output_grouping_n_outputs,
            "n_proto": grouper.n_proto if grouper else self._output_grouping_n_proto,
            "groups": [
                {"targets": g.targets, "exp": g.expansion.value, "enabled": g.enabled}
                for g in grouper.groups
            ] if grouper else [],
        }

    def set_stdp(
        self,
        enabled: bool = True,
        weight_min: float = -5.0,
        weight_max: float = 5.0,
        hebb_sigma: float = 0.05,
    ) -> None:
        """Enable Synaptische Plastizität (STDP / Hebbian intra-lifetime learning).

        When active, each ``genome.forward()`` call during evaluation applies a
        Hebb-rule weight update using the current pre- and post-synaptic
        activations.  At the end of every episode (``genome.reset()`` or at the
        end of each genome evaluation) the original evolved weights are restored
        so plasticity is **episoden-lokal** — it never accumulates across
        generations.

        **Hebb rule applied after each forward call:**
        ``Δw = A·pre + B·post + C·pre·post + D``

        A, B, C, D are per-connection genes evolved alongside connection weights.
        All default to 0.0 (no plasticity = zero cost).  Use
        :func:`~yane.evolution.stdp.set_hebb_coeffs` to initialise them for
        testing, or let evolution discover useful values from small random seeds.

        Parameters
        ----------
        enabled :
            Toggle STDP on or off.
        weight_min, weight_max :
            Clamp range for working weights — prevents runaway plasticity.
        hebb_sigma :
            Gaussian noise applied to Hebb coefficients each NEAT mutation.

        Example
        -------
        ::

            from yane.evolution.stdp import set_hebb_coeffs
            yane.set_stdp(enabled=True, weight_min=-3.0, weight_max=3.0)
            # Seed hebb coefficients for demonstration:
            for g in yane.population._unevaluated:
                set_hebb_coeffs(g, c=0.01, sigma=0.005)
            yane.train(fitness_fn)
        """
        self._stdp_enabled = bool(enabled)
        self._stdp_weight_min = float(weight_min)
        self._stdp_weight_max = float(weight_max)
        self._stdp_hebb_sigma = float(hebb_sigma)

    def set_neuromodulation(
        self,
        enabled: bool = True,
    ) -> None:
        """Enable Neuromodulation — kontextabhängige Gewichtung via Modulator-Knoten.

        When active, nodes marked with ``node.is_modulator = True`` act as
        modulatory interneurons: their activation value is applied as a
        multiplicative gain to all connections feeding into their target nodes.

        The modulation is **one-step-delayed**: the MODULATOR's activation from
        call T sets the gain for call T+1.  This avoids two-pass forward
        computation while still enabling context-sensitive gating.

        MODULATOR nodes are evolved by NEAT like normal hidden nodes.  Mark
        a node as a MODULATOR via ``genome.nodes[i].is_modulator = True``,
        or let :func:`~yane.evolution.neuromodulation.mutate_modulator_flags`
        discover them during evolution.

        Parameters
        ----------
        enabled :
            Toggle neuromodulation on or off.

        Example
        -------
        ::

            from yane.evolution.neuromodulation import make_node_modulator
            yane.set_neuromodulation(enabled=True)
            # After configure(), mark hidden node as modulator:
            for g in yane.population._unevaluated:
                if len(g.nodes) > 3:
                    make_node_modulator(g, 2)
            yane.train(fitness_fn)
        """
        self._neuromodulation_enabled = bool(enabled)

    def set_hybrid_mode(
        self,
        enabled: bool = True,
        bp_interval: int = 10,
        bp_epochs: int = 50,
        bp_lr: float = 0.01,
        bp_batch_size: int = 32,
        top_k: int = 3,
        train_data: "list | None" = None,
        replay_buffer_size: int = 10_000,
    ) -> None:
        """Enable Gradient-NEAT Hybrid mode (backprop interleaved with evolution).

        Every ``bp_interval`` generations the top-K genomes receive a short
        backprop fine-tuning phase before being returned to evolution.  This
        combines NEAT's global structural search with gradient descent's
        efficient local weight tuning.

        Requires PyTorch (``pip install torch``).  The backprop phase raises
        ``ImportError`` when PyTorch is absent; calling :meth:`set_hybrid_mode`
        itself does not require PyTorch.

        Parameters
        ----------
        enabled :
            Toggle hybrid mode on or off.
        bp_interval :
            Run a backprop phase every this many generations.
        bp_epochs :
            Gradient-descent steps per genome per backprop phase.
        bp_lr :
            Adam optimiser learning rate.
        bp_batch_size :
            Number of samples drawn from the replay buffer per gradient step.
        top_k :
            Number of best-fitness genomes to fine-tune each backprop phase.
        train_data :
            Optional list of ``(inputs, targets)`` pairs to use as supervised
            training data.  When *None*, the replay buffer is used (inputs
            seen during NEAT evaluation, labelled by the current best genome).
        replay_buffer_size :
            Maximum entries in the auto-accumulated replay buffer.

        Example
        -------
        ::

            yane.set_hybrid_mode(bp_interval=10, bp_epochs=30, bp_lr=0.005,
                                 train_data=[([0,0],[0]), ([0,1],[1]),
                                             ([1,0],[1]), ([1,1],[0])])
            yane.train(xor_fitness_fn)
        """
        if not enabled:
            self._hybrid_mode_enabled = False
            self._hybrid_replay_buffer = None
            return
        from yane.evolution.hybrid_neat import ReplayBuffer
        self._hybrid_mode_enabled = True
        self._hybrid_bp_interval = max(1, bp_interval)
        self._hybrid_bp_epochs = max(1, bp_epochs)
        self._hybrid_bp_lr = bp_lr
        self._hybrid_bp_batch_size = max(1, bp_batch_size)
        self._hybrid_top_k = max(1, top_k)
        self._hybrid_train_data = list(train_data) if train_data is not None else None
        self._hybrid_replay_buffer = ReplayBuffer(max_size=replay_buffer_size)

    def set_conv_neat(
        self,
        enabled: bool = True,
        conv_stack: "ConvStack | None" = None,
        n_image_channels: int = 1,
        n_blocks: int = 1,
        kernel_size: int = 3,
        out_channels: int = 8,
        activation: str = "relu",
    ) -> "ConvStack | None":
        """Enable a convolutional NEAT front-end for image inputs.

        Each genome gains a :class:`~yane.evolution.conv_neat.ConvStack` that
        preprocesses image inputs before the NEAT network.  Use
        :meth:`genome.forward_image` in your evaluator instead of
        :meth:`genome.forward`.

        Call **before** :meth:`configure`.  The network's ``n_inputs`` must
        equal :meth:`conv_n_inputs`.

        Parameters
        ----------
        enabled :
            Toggle on or off.
        conv_stack :
            Pre-built :class:`~yane.evolution.conv_neat.ConvStack`.  When
            *None*, a stack is built from the remaining parameters.
        n_image_channels :
            Number of channels in the input images.
        n_blocks :
            Number of conv blocks in the auto-built stack.
        kernel_size :
            Kernel size for every auto-built block.
        out_channels :
            Output channels for every auto-built block.
        activation :
            Activation function for every auto-built block.

        Returns
        -------
        ConvStack | None
            The stack attached to new genomes, or *None* when disabled.

        Example
        -------
        ::

            stack = yane.set_conv_neat(n_image_channels=1, n_blocks=2,
                                       kernel_size=3, out_channels=8)
            yane.configure(n_inputs=yane.conv_n_inputs(), n_outputs=10)
            yane.train(lambda g: eval_fn(g, images, labels))
        """
        if not enabled:
            self._conv_neat_enabled = False
            self._conv_neat_stack = None
            return None
        from yane.evolution.conv_neat import make_conv_stack, ConvStack as _CS
        if conv_stack is None:
            conv_stack = make_conv_stack(
                n_image_channels=n_image_channels,
                n_blocks=n_blocks,
                kernel_size=kernel_size,
                out_channels=out_channels,
                activation=activation,
            )
        self._conv_neat_enabled = True
        self._conv_neat_stack = conv_stack
        return conv_stack

    def conv_n_inputs(self) -> int:
        """Return the flat feature dimension produced by the active conv stack.

        Pass this value as ``n_inputs`` to :meth:`configure` when using
        :meth:`set_conv_neat`.  Raises ``RuntimeError`` when no stack is set.
        """
        if self._conv_neat_stack is None:
            raise RuntimeError(
                "No ConvStack configured.  Call set_conv_neat() first."
            )
        return self._conv_neat_stack.n_outputs

    def set_attention(
        self,
        enabled: bool = True,
        head_dim: int = 4,
        num_heads: int = 2,
        n_inputs: int | None = None,
    ) -> "AttentionBlock | None":
        """Enable Evolvable Attention Heads preprocessing.

        Attaches an :class:`~yane.evolution.attention.AttentionBlock` to each
        genome.  The block runs a multi-head self-attention computation on the
        raw inputs and produces a ``num_heads * head_dim``-dimensional feature
        vector, which is then fed into the NEAT network.

        Call **before** :meth:`configure`.  Pass ``n_inputs=yane.attention_n_inputs()``
        as the network's input count.

        Parameters
        ----------
        enabled :
            Toggle on or off.
        head_dim :
            Dimensionality of each attention head (K/Q/V projection size).
        num_heads :
            Number of parallel attention heads.
        n_inputs :
            Number of raw input features.  Defaults to the configured ``n_inputs``
            from the last :meth:`configure` call; must be set when called before
            ``configure()``.

        Returns
        -------
        AttentionBlock | None
        """
        if not enabled:
            self._attention_enabled = False
            self._attention_block = None
            return None
        from yane.evolution.attention import AttentionBlock as _AB
        n_in = n_inputs or getattr(self, "_n_inputs", None)
        if n_in is None:
            raise ValueError(
                "set_attention() requires n_inputs.  Pass it explicitly or call "
                "configure() first."
            )
        self._attention_enabled = True
        self._attention_block = _AB(n_inputs=n_in, head_dim=head_dim, num_heads=num_heads)
        return self._attention_block

    def attention_n_inputs(self) -> int:
        """Return the output dimension of the active attention block.

        Pass as ``n_inputs`` to :meth:`configure`.  Raises ``RuntimeError``
        when no block is set.
        """
        if self._attention_block is None:
            raise RuntimeError(
                "No AttentionBlock configured.  Call set_attention() first."
            )
        return self._attention_block.n_outputs

    def set_ltc(self, enabled: bool = True) -> None:
        """Enable Liquid Time-Constant (LTC) node dynamics.

        When active, nodes with ``node.tau < inf`` use the LTC ODE update rule
        after each ``genome.forward()`` call instead of the standard activation:

        ``x_{t+1} = x_t + dt * (-x_t/τ + activation(sum(inputs) + bias))``

        Mark individual nodes as LTC via
        :func:`~yane.evolution.ltc.make_node_ltc`.

        Parameters
        ----------
        enabled :
            Toggle LTC dynamics on or off.
        """
        self._ltc_enabled = bool(enabled)

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

    def set_anytime_eval(
        self,
        enabled: bool = True,
        min_evals: int = 1,
        max_evals: int = 5,
        promotion_frac: float = 0.3,
        aggregation: str = "mean",
    ) -> None:
        """Enable adaptive evaluation budgeting.

        Weak genomes receive ``min_evals`` fitness calls. Genomes whose
        provisional score is competitive with the current population are
        promoted to ``max_evals`` calls and aggregated.
        """
        self._runner.configure_anytime_eval(
            enabled=enabled,
            min_evals=min_evals,
            max_evals=max_evals,
            promotion_frac=promotion_frac,
            aggregation=aggregation,
        )

    def add_eval_middleware(self, middleware: EvalMiddleware) -> None:
        """Add evaluation middleware.

        Middleware order is LIFO: the most recently added middleware wraps the
        existing chain and sees the evaluation first.
        """
        self._eval_middlewares.append(middleware)

    def clear_eval_middleware(self) -> None:
        """Remove all evaluation middleware and reset middleware diagnostics."""
        self._eval_middlewares.clear()
        self._eval_middleware_diagnostics = {}

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

    def set_lamarck_momentum(
        self,
        enabled: bool = True,
        momentum_prob: float = 0.3,
        decay: float = 0.9,
    ) -> None:
        """Enable gradient-informed mutation direction via Lamarck-Momentum.

        After each Lamarckian refinement step the net weight-delta vector is
        stored on the genome.  During subsequent mutation, each parameter is
        nudged in the stored direction with probability ``momentum_prob``.
        The stored deltas decay exponentially so the signal fades after a few
        generations if no further refinement occurs.

        Args:
            enabled:       Pass False to disable without losing config.
            momentum_prob: Per-parameter probability of applying the momentum
                           nudge (default 0.3).
            decay:         Exponential decay factor applied to stored deltas
                           each generation (default 0.9; 1.0 = no decay).
        """
        if not (0.0 <= momentum_prob <= 1.0):
            raise ValueError(f"momentum_prob must be in [0, 1], got {momentum_prob}")
        if not (0.0 < decay <= 1.0):
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        self._lamarck.momentum_enabled = enabled
        self._lamarck.momentum_prob = momentum_prob
        self._lamarck.momentum_decay = decay

    def set_post_training_pruning(
        self,
        enabled: bool = True,
        threshold: float = 0.01,
        max_drop_frac: float = 0.02,
    ) -> None:
        """Enable automatic pruning of the best genome after ``train()`` completes.

        After training finishes, the best genome is pruned by removing connections
        whose absolute weight is below *threshold*.  The pruned genome is then
        re-evaluated once; if its fitness drops by more than *max_drop_frac*
        relative to the pre-prune fitness the pruning is rolled back and the
        original genome is restored.

        Results are stored on the genome via ``genome.prune_stats()``, including
        ``fitness_delta``, ``compression_rate``, and ``rolled_back``.

        Args:
            enabled:       Pass False to disable without losing config.
            threshold:     Minimum absolute weight to keep (default 0.01).
            max_drop_frac: Maximum tolerated relative fitness drop before
                           rollback.  E.g. 0.02 = up to 2% drop allowed.
        """
        if threshold < 0.0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        if not (0.0 <= max_drop_frac <= 1.0):
            raise ValueError(f"max_drop_frac must be in [0, 1], got {max_drop_frac}")
        self._post_pruning_enabled = enabled
        self._post_pruning_threshold = threshold
        self._post_pruning_max_drop_frac = max_drop_frac

    # ── Research features (Curiosity / DARTS / Shared Weights) ─────────────

    def set_curiosity(
        self,
        enabled: bool = True,
        weight: float = 0.3,
        network_size: int = 8,
        lr: float = 0.01,
    ) -> None:
        """Enable intrinsic curiosity reward.

        Adds a prediction-error bonus to fitness: genomes producing surprising
        (hard-to-predict) outputs receive a positive bonus, encouraging exploration.

        A 2-layer forward model (n_inputs → hidden → n_outputs) predicts each
        genome's output from its inputs; the prediction error becomes the bonus.
        The model is updated online after every genome evaluation.

        Args:
            enabled:      Pass False to disable without losing config.
            weight:       Bonus scale factor added to base fitness.
            network_size: Hidden layer size of the forward model.
            lr:           SGD learning rate for forward model updates.
        """
        self._curiosity_enabled = enabled
        self._curiosity_weight = weight
        self._curiosity_network_size = network_size
        self._curiosity_lr = lr
        if enabled and self._n_inputs > 0 and self._n_outputs > 0:
            from yane.evolution.experimental import IntrinsicCuriosityModule
            self._curiosity_module = IntrinsicCuriosityModule(
                self._n_inputs, self._n_outputs, network_size, lr
            )
        elif not enabled:
            self._curiosity_module = None

    def set_darts_mode(
        self,
        enabled: bool = True,
        prune_threshold: float = 0.1,
    ) -> None:
        """Enable DARTS-Lite differentiable architecture search.

        Each connection gets a gate value in [0, 1] updated every generation
        from |weight| via sigmoid.  At the end of training, connections whose
        gate falls below *prune_threshold* are removed from the best genome.

        Args:
            enabled:         Pass False to disable.
            prune_threshold: Gate cutoff for post-training pruning (default 0.1).
        """
        if not (0.0 <= prune_threshold <= 1.0):
            raise ValueError(f"prune_threshold must be in [0, 1], got {prune_threshold}")
        self._darts_enabled = enabled
        self._darts_prune_threshold = prune_threshold

    def set_shared_weights(self, enabled: bool = True) -> None:
        """Enable shared-weight groups.

        When enabled, connections can be assigned to a named weight group via
        ``genome.set_weight_group(conn, group_id)``.  All connections in a group
        share the same weight value; mutations and Lamarck refinement update the
        group value and sync all members automatically.

        Args:
            enabled: Pass False to disable.
        """
        self._shared_weights_enabled = enabled

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

    # ── Island Model ────────────────────────────────────────────────────────

    def set_island_model(
        self,
        n_islands: int = 4,
        migration_interval: int = 5,
        migration_count: int = 3,
    ) -> Any:
        """Enable multi-population island model.

        Each island is an independent ``Population`` instance.  Periodically,
        the best genomes migrate between islands.

        Args:
            n_islands: Number of independent populations.
            migration_interval: How many spawn cycles between migrations.
            migration_count: How many genomes to migrate each time.

        Returns:
            The ``IslandModel`` instance for advanced configuration.
        """
        from yane.evolution.islands import IslandModel

        self._ensure_configured()
        kw = {
            "max_size": self._population_size,
            "initial_genome": self._population._template.copy(),
            "tracker": self._tracker,
            "target_species": self.get_target_species() or 5,
            "_crossover_enabled": self._population._crossover_enabled,
            "_speciation_enabled": self._population._speciation_enabled,
            "_compat_threshold": self._population._compat_threshold,
            "_weight_blend_alpha": self._population._weight_blend_alpha,
        }
        self._island_model = IslandModel(
            n_islands=n_islands,
            island_kwargs=[dict(kw) for _ in range(n_islands)],
            migration_interval=migration_interval,
            migration_count=migration_count,
        )
        self._population = self._island_model.islands[0]
        return self._island_model

    def get_island_diagnostics(self) -> dict:
        """Return island model diagnostics."""
        if not hasattr(self, "_island_model") or self._island_model is None:
            return {}
        return self._island_model.get_diagnostics()

    # ── Policy System ───────────────────────────────────────────────────────

    def _ensure_policy_registry(self):
        if self._policy_registry is None:
            from yane.evolution.policy import PolicyRegistry
            self._policy_registry = PolicyRegistry()

    def register_policy(self, policy, enabled: bool = True) -> None:
        """Register an adaptive policy.

        The policy must conform to the ``AdaptivePolicy`` protocol:
        ``observe(ctx)``, ``decide(ctx) -> Action | None``, ``apply(ctx, action)``.
        """
        self._ensure_policy_registry()
        self._policy_registry.register(policy, enabled=enabled)
        self._policy_tick_enabled = True

    def set_policy_order(self, names: list[str]) -> None:
        """Set the policy evaluation order.

        Only registered policy names are used; unknown names are ignored.
        """
        self._ensure_policy_registry()
        self._policy_registry.set_order(names)

    def get_policy_diagnostics(self) -> dict:
        """Return diagnostics from the policy registry."""
        if self._policy_registry is None:
            return {}
        return self._policy_registry.get_diagnostics()

    def set_online_tuning(
        self,
        enabled: bool = True,
        params: list[str] | None = None,
    ) -> None:
        """Enable UCB1 bandit-based hyperparameter tuning during training.

        The bandit tunes discrete values for the specified parameters by
        treating each (param → value) as an arm and using fitness delta as
        reward.  Exploration happens during the first 20 % of iterations;
        thereafter the best-performing arm is exploited.

        Supported parameters and their candidate values:

        ``"mutation_rate"``
            Structure mutation probability floor (0.1, 0.3, 0.5, 0.8).
        ``"n_lamarck_steps"``
            Lamarck refinement steps per genome (0, 1, 3, 5).

        Args:
            enabled: Pass False to disable.
            params:  List of parameter names to tune.  Default: all supported.
        """
        if not enabled:
            self._online_tuning_enabled = False
            self._online_tuning_bandits = {}
            if self._online_tuning_original_struct_floor is not None:
                from yane.core.genome import Genome as _G
                _G._STRUCT_FLOOR = self._online_tuning_original_struct_floor
                self._online_tuning_original_struct_floor = None
            return

        from yane.evolution.online_tuning import UCB1Bandit
        from yane.core.genome import Genome as _G
        if self._online_tuning_original_struct_floor is None:
            self._online_tuning_original_struct_floor = _G._STRUCT_FLOOR

        param_candidates: dict[str, list] = {
            "mutation_rate":    [0.05, 0.15, 0.3, 0.5, 0.8],
            "n_lamarck_steps":  [0, 1, 3, 5, 10],
        }
        if params is None:
            params = list(param_candidates.keys())

        self._online_tuning_enabled = True
        self._online_tuning_params = list(params)
        self._online_tuning_bandits = {}
        for p in params:
            cand = param_candidates.get(p)
            if cand is None:
                raise ValueError(f"Unknown tunable parameter: {p!r}")
            self._online_tuning_bandits[p] = UCB1Bandit(p, cand)

    def _tick_online_tuning(self, iteration: int, max_iterations: int) -> None:
        """Apply bandit selections and record rewards.  Called each generation."""
        if not self._online_tuning_enabled or not self._online_tuning_bandits:
            return

        # Get current best fitness for reward calculation
        try:
            current_best = self._population.get_best().fitness
        except RuntimeError:
            return

        # Apply bandit selections (every generation)
        for param, bandit in self._online_tuning_bandits.items():
            value = bandit.select(iteration, max_iterations)
            self._apply_bandit_value(param, value)

        # Record reward: fitness delta since last call
        if self._online_tuning_last_best is not None:
            delta = current_best - self._online_tuning_last_best
            for bandit in self._online_tuning_bandits.values():
                bandit.update(delta)

        self._online_tuning_last_best = current_best

    def _apply_bandit_value(self, param: str, value: float) -> None:
        """Apply a bandit-selected value to the relevant NeuroEvolution setting."""
        if param == "mutation_rate":
            if self._population is not None:
                Genome._STRUCT_FLOOR = float(value)
        elif param == "n_lamarck_steps":
            self._lamarck.max_steps = max(1, int(value))
            self._lamarck_steps = int(value)

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

    @staticmethod
    def register_activation(name: str, fn: Callable) -> None:
        """Register a custom activation function available at runtime.

        The function is stored in the module-level ``CUSTOM_ACTIVATION_FNS``
        registry so it survives ``configure()`` calls and is discovered by
        ``list_activations()``.

        *fn* should be a named module-level function so that pickling works
        correctly when saving/loading checkpoints.

        Example::

            def my_activation(v: float) -> float:
                return v * v if v > 0 else 0.0

            NeuroEvolution.register_activation("my_act", my_activation)
            yane = NeuroEvolution()
            # Nodes can now use "my_act" as their activation type.
        """
        from yane.util.activation import register_activation as _reg
        _reg(name, fn)

    @staticmethod
    def register_example(plugin: Any) -> None:
        """Register a user-defined evaluator plugin.

        The plugin appears in the GUI example list and is available for API use.

        See ``yane.gui.examples.register_example()`` for details and a
        code example.
        """
        from yane.gui.examples import register_example as _reg
        _reg(plugin)

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
            from torch.utils.tensorboard import SummaryWriter as _SW  # noqa: F401
        except ImportError:
            raise ImportError(
                "TensorBoard logging requires 'torch'. "
                "Install it with:  pip install torch"
            ) from None

    def set_tracking_backend(self, *backends) -> None:
        """Register one or more experiment-tracking backends.

        Each backend receives per-generation metrics, the training config, and
        a ``finish()`` call at the end of ``train()``.  Multiple backends are
        dispatched in registration order; all receive identical data.

        Built-in backends::

            from yane.evolution.tracking import WandbBackend, MlflowBackend
            yane.set_tracking_backend(WandbBackend(project="myproject"))
            yane.set_tracking_backend(MlflowBackend())

        Custom backends only need to implement the
        :class:`~yane.evolution.tracking.TrackingBackend` protocol (duck-typed,
        no inheritance required).

        Calling ``set_tracking_backend()`` with no arguments clears all
        registered backends.
        """
        if not backends:
            self._tracking_backends = []
            return
        for backend in backends:
            self._tracking_backends.append(backend)

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
            "adaptive_pop_schedule": self._adaptive_pop_schedule,
            "n_workers": self._n_workers,
            "target_species": pop._target_species if pop else self._target_species,
            "target_species_min": pop._target_species_min if pop else self._target_species_min,
            "target_species_max": pop._target_species_max if pop else self._target_species_max,
            "compat_tune_interval": pop._compat_tune_interval if pop else self._compat_tune_interval,
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
            "weight_inheritance_enabled": self._weight_inheritance_enabled,
            "weight_blend_alpha": self._weight_blend_alpha,
            "elite_count": self._elite_count,
            "species_elite_count": self._species_elite_count,
            # Stopping criteria
            "min_fitness": self.min_fitness,
            "max_iterations": self.max_iterations,
            "max_evaluations": self._max_evaluations,
            "convergence_spread_eps": self._convergence_spread_eps,
            "convergence_min_stagnation": self._convergence_min_stagnation,
            "adaptive_recovery_enabled": self._adaptive_recovery_enabled,
            "adaptive_recovery_strategies": self._recovery_strategies,
            "adaptive_recovery_cooldown": self._recovery_cooldown,
            "adaptive_recovery_escalate": self._recovery_escalate,
            "adaptive_recovery_diversity_iqr_threshold": self._recovery_diversity_iqr_threshold,
            "adaptive_recovery_injection_frac": self._recovery_injection_frac,
            "adaptive_recovery_early_stopping_patience": self._recovery_early_stopping_patience,
            "adaptive_recovery_warmup": self._recovery_warmup,
            "adaptive_recovery_min_delta": self._recovery_min_delta,
            # Evaluation
            "n_evaluations": self._runner.n_evaluations,
            "eval_aggregation": self._runner.aggregation,
            "eval_sigma_penalty": self._runner.sigma_penalty,
            "anytime_eval_enabled": self._runner.anytime_enabled,
            "anytime_min_evals": self._runner.anytime_min_evals,
            "anytime_max_evals": self._runner.anytime_max_evals,
            "anytime_promotion_frac": self._runner.anytime_promotion_frac,
            "anytime_aggregation": self._runner.anytime_aggregation,
            "eval_middleware_count": len(self._eval_middlewares),
            "checkpoint_policy_enabled": self._checkpoint_policy_enabled,
            "checkpoint_interval": self._checkpoint_interval,
            "checkpoint_keep_best": self._checkpoint_keep_best,
            "checkpoint_max_keep": self._checkpoint_max_keep,
            "checkpoint_path_template": self._checkpoint_path_template,
            "best_checkpoint_path": self._best_checkpoint_path,
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
            # Research features
            "curiosity_enabled": self._curiosity_enabled,
            "curiosity_weight": self._curiosity_weight,
            "curiosity_network_size": self._curiosity_network_size,
            "curiosity_lr": self._curiosity_lr,
            "darts_enabled": self._darts_enabled,
            "darts_prune_threshold": self._darts_prune_threshold,
            "shared_weights_enabled": self._shared_weights_enabled,
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
                ``"curriculum_complete"``, ``"external"``,
                ``"budget_exceeded"``.
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

        # Continual learning: wrap fitness_fn with EWC / replay regularization.
        if self._continual_learner is not None:
            fitness_fn = self._continual_learner.wrap_fitness(fitness_fn, ne=self)

        # Minimal criterion: wrap fitness_fn to penalize non-viable genomes.
        if self._minimal_criterion is not None:
            fitness_fn = self._minimal_criterion.wrap_fitness(fitness_fn)

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
        if self._run_database is None:
            _wj(self._log_run_dir / "config.json", self._config_dict())

        # --- Run tracking ----------------------------------------------------
        import uuid as _uuid
        self._active_run_id = str(_uuid.uuid4())
        if self._run_database is not None:
            try:
                self._run_database.start_run(
                    run_id=self._active_run_id,
                    name=name,
                    seed=self._seed,
                    config=self._config_dict(),
                    experiment_id=self._active_experiment_id,
                )
            except Exception:
                pass

        # --- Logging state ---------------------------------------------------
        _log_interval = max(1, self._population_size // 10)  # log ~10× per generation
        _csv_header = "generation,iteration,best_fitness,mean_fitness,median_fitness,iqr_fitness,species_count,stagnation_count,nodes,connections,validation_fitness"
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

        # Tracking backends: init + log config once per run.
        _tracking_config: dict = {
            "n_inputs":        self._n_inputs,
            "n_outputs":       self._n_outputs,
            "population_size": self._population_size,
            "max_nodes":       self._max_nodes,
            "max_connections": self._max_connections,
            "seed":            self._seed,
            "run_name":        name,
        }
        for _tb_backend in self._tracking_backends:
            try:
                _tb_backend.init(_tracking_config)
                _tb_backend.log_config(_tracking_config)
            except Exception as _tb_err:
                _li("Tracking backend init failed (%s): %s",
                    type(_tb_backend).__name__, _tb_err)

        # Augmentation: select the first pipeline before any evaluations.
        if self._aug_pool is not None:
            self._aug_pool.select()
            self._aug_prev_fit = -float("inf")

        # Budget enforcer: record start time once per run.
        if self._budget_enforcer is not None:
            self._budget_enforcer.start()

        # Hybrid mode: wrap fitness_fn to capture inputs for replay buffer.
        if self._hybrid_mode_enabled and self._hybrid_replay_buffer is not None:
            _rb = self._hybrid_replay_buffer
            _orig_fitness_fn = fitness_fn
            def fitness_fn(g, _fn=_orig_fitness_fn):
                _inputs_seen: list = []
                _orig_fwd = g.forward
                def _capturing_fwd(inp):
                    _inputs_seen.append(list(inp))
                    return _orig_fwd(inp)
                g.__dict__["forward"] = _capturing_fwd
                try:
                    result = _fn(g)
                finally:
                    g.__dict__.pop("forward", None)
                for inp in _inputs_seen:
                    _rb.add(inp)
                return result

        self._n_evaluations_done = 0
        stop_reason: str | None = None
        iterations = 0
        _gen_size = max(1, self._population_size)  # generation boundary for periodic ticks
        _gen_eval_ms: float = 0.0   # accumulated eval time for current generation
        while True:
            if self._island_model is not None:
                self._population = self._island_model.islands[
                    iterations % self._island_model.n_islands
                ]
            genome = self._population.select_for_evaluation()

            result = self._run_evaluations(genome, fitness_fn)
            fitness = self._finalize_fitness(result.fitness, result.elapsed_ms, genome)
            self._population.submit(genome, fitness, result.elapsed_ms)
            _gen_eval_ms += result.elapsed_ms
            iterations += 1
            self._n_evaluations_done += result.n_fitness_calls
            # Phylogeny recording (zero-cost when disabled)
            if self._phylogeny is not None and self._phylogeny.is_enabled:
                _parent_id = (genome._parent_ids[0]
                              if getattr(genome, "_parent_ids", None) else None)
                _generation = iterations // max(1, _gen_size)
                _innov_log = getattr(self._tracker, "_innovation_log", {})
                _genome_innovations = [
                    innov for innov, (_, gid) in _innov_log.items()
                    if gid == genome._genome_id
                ]
                self._phylogeny.record(
                    genome_id=genome._genome_id,
                    parent_id=_parent_id,
                    fitness=fitness,
                    generation=_generation,
                    innovations=_genome_innovations,
                )

            # --- "new_best" event -------------------------------------------
            if self._population._evaluated:
                _cur_best = self._population.get_best()
                if _cur_best.raw_fitness > self._last_best_fitness:
                    self._last_best_fitness = _cur_best.raw_fitness
                    self._event_bus.emit("new_best", {
                        "genome": _cur_best,
                        "fitness": _cur_best.raw_fitness,
                        "iteration": iterations,
                    })

            # Tick adaptive components once per generation
            if iterations % _gen_size == 0:
                if self._darts_enabled and self._population is not None:
                    for _g in self._population._evaluated:
                        _g.update_darts_gates()
                self._tick_transfer_unfreeze(iterations // _gen_size)
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
                if self._island_model is not None:
                    self._island_model.tick()
                if self._online_tuning_enabled:
                    self._tick_online_tuning(iterations, self.max_iterations or 100000)
                self._lamarck.reset_generation_budget()
                # Auto fitness shaping: analyse every 50 generations
                if self._auto_fitness_shaping_enabled and iterations % 50 == 0:
                    _evald = self._population._evaluated
                    if _evald:
                        try:
                            from yane.evolution.fitness_transform import (
                                FitnessLandscapeAnalyzer,
                            )
                            _report = FitnessLandscapeAnalyzer.analyze(_evald)
                            self._auto_fitness_shaping_report = _report
                            _rec = FitnessLandscapeAnalyzer.recommend_transform(_report)
                            if _rec is not None:
                                _rec_name = getattr(_rec, "name", str(_rec))
                                _report.applied_transform = _rec_name
                                self._fitness_transform = _rec
                        except Exception:
                            pass
                # Apply fitness transform to all evaluated genomes
                if self._fitness_transform is not None:
                    _evald = self._population._evaluated
                    if _evald:
                        _raw = [g.raw_fitness for g in _evald]
                        try:
                            _tf = self._fitness_transform(_raw)
                            for _g, _f in zip(_evald, _tf):
                                _g.fitness = _f
                        except Exception:
                            pass
                # Run validation on best genome
                if self._validation_fn is not None:
                    try:
                        _vbest = self._population.get_best()
                        self._last_validation_fitness = self._validation_fn(_vbest)
                    except Exception:
                        pass

            # --- Policy system tick (once per generation, outside heartbeat) ---
            if iterations % _gen_size == 0:
                if self._policy_tick_enabled and self._policy_registry is not None:
                    try:
                        from yane.evolution.policy import TrainingContext
                        _pmem = self.population_memory_info()
                        _pctx = TrainingContext(
                            generation=_pmem.get("generation", iterations // _gen_size),
                            iteration=iterations,
                            best_fitness=_pmem.get("max_fitness", -float("inf")),
                            mean_fitness=_pmem.get("avg_fitness", 0.0),
                            median_fitness=_pmem.get("median_fitness", 0.0),
                            fitness_iqr=_pmem.get("fitness_iqr", 0.0),
                            species_count=_pmem.get("species_count", 0),
                            stagnation_count=_pmem.get("stagnation_count", 0),
                            n_evaluations=self._n_evaluations_done,
                            max_iterations=self.max_iterations or 100000,
                            recovery_events=self._recovery_events,
                            stopped_early=self.stopped_early,
                        )
                        self._policy_registry.tick(_pctx)
                    except Exception:
                        pass

            # --- MetaOptimizer + FeatureGating tick (once per generation) ----
            if iterations % _gen_size == 0:
                _gen_num = iterations // _gen_size
                _cur_best = self._population.get_best()
                _cur_fit = _cur_best.raw_fitness if _cur_best else -float("inf")
                if self._meta_optimizer_enabled:
                    self._tick_meta_optimizer(
                        generation=_gen_num,
                        best_fitness=_cur_fit,
                        eval_ms=_gen_eval_ms,
                    )
                if self._feature_gate_enabled:
                    self._tick_feature_gating(
                        generation=_gen_num,
                        best_fitness=_cur_fit,
                    )
                _gen_eval_ms = 0.0   # reset for next generation

                # Hybrid NEAT: run backprop phase every bp_interval generations.
                if self._hybrid_mode_enabled and _gen_num > 0 and _gen_num % self._hybrid_bp_interval == 0:
                    self._run_hybrid_backprop(_gen_num)

                # Augmentation pool: reward current pipeline, maybe evolve, select next.
                if self._aug_pool is not None:
                    _aug_reward = max(0.0, _cur_fit - getattr(self, "_aug_prev_fit", _cur_fit))
                    self._aug_pool.update_reward(_aug_reward)
                    if self._aug_pool.should_evolve(_gen_num):
                        self._aug_pool.evolve()
                    self._aug_pool.select()   # select pipeline for next generation
                    self._aug_prev_fit = _cur_fit

                # Tracking backends: log once per generation.
                if self._tracking_backends:
                    from yane.evolution.tracking import _scalar_metrics as _scm
                    _tb_mem = self.population_memory_info()
                    _tb_scalars = _scm(_tb_mem)
                    for _tb_backend in self._tracking_backends:
                        try:
                            _tb_backend.log_metrics(_tb_scalars, _gen_num)
                        except Exception:
                            pass

            # --- Periodic CSV logging + heartbeat ---------------------------
            _heartbeat_now = (iterations % 100 == 0)
            _csv_now = (iterations % _log_interval == 0)
            self._maybe_auto_checkpoint(iterations, name)

            if _heartbeat_now:
                # Full diagnostics once; shared by heartbeat log and CSV.
                mem = self.population_memory_info()
                if _use_csv and _csv_now:
                    _wc(_csv_path, _csv_header,
                        f"{mem.get('generation', iterations // _gen_size)},"
                        f"{iterations},"
                        f"{mem.get('max_fitness', 0)},"
                        f"{mem.get('avg_fitness', 0)},"
                        f"{mem.get('median_fitness', 0)},"
                        f"{mem.get('fitness_iqr', 0)},"
                        f"{mem.get('species_count', 0)},"
                        f"{mem.get('stagnation_count', 0)},"
                        f"{mem.get('largest_genome_nodes', 0)},"
                        f"{mem.get('largest_genome_connections', 0)},"
                        f"{mem.get('validation_fitness', '')}")
                if _use_jsonl:
                    _wjl(_jsonl_path, {
                        **mem,
                        "generation": mem.get("generation", iterations // _gen_size),
                        "iteration": iterations,
                    })
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
                        _cb({
                            **mem,
                            "generation": mem.get("generation", iterations // _gen_size),
                            "iteration": iterations,
                        })
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
                # Add validation and anomaly diagnostics to mem
                if self._last_validation_fitness is not None:
                    mem["validation_fitness"] = self._last_validation_fitness
                if self._anomaly_detectors is not None:
                    _reports = self._anomaly_detectors.check_all(mem, iterations)
                    for _r in _reports:
                        self._event_bus.emit("anomaly", {
                            "kind": _r.kind, "message": _r.message,
                            "iteration": _r.iteration, "value": _r.value,
                        })
                        _li("ANOMALY [%s]: %s", _r.kind, _r.message)
                    mem.update(self._anomaly_detectors.get_diagnostics())
                    if _reports:
                        mem["anomaly_kinds"] = [_r.kind for _r in _reports]
                recovery_stop = self._tick_adaptive_recovery(mem, iterations, _li)
                if recovery_stop is not None:
                    stop_reason = recovery_stop
                    break
                if mem.get("stagnation_count", 0) > 0:
                    self._event_bus.emit("stagnation", {
                        "stagnation_count": mem.get("stagnation_count", 0),
                        "iteration": iterations,
                    })
                self._event_bus.emit("generation_end", {**mem, "iteration": iterations})
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
                    f"{iterations // _gen_size},"
                    f"{iterations},"
                    f"{_max_fit},{_avg_fit},{_med_fit},{_iqr_fit},"
                    f"{_pop.species_count},{_pop.stagnation_count},"
                    f"{_max_nodes},{_max_conns},"
                    f"{self._last_validation_fitness if self._last_validation_fitness is not None else ''}")

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

            # Budget: time check is cheap, run every iteration.
            if self._budget_enforcer is not None and self._budget_enforcer.is_time_over():
                stop_reason = "budget_exceeded"
                break

            if iterations % self._resource_check_interval == 0:
                self._enforce_memory_limit()
                while not self._resource_guard.system_ok():
                    time.sleep(0.5)
                # Budget: memory check (heavier, run at same cadence as psutil).
                if self._budget_enforcer is not None:
                    self._budget_enforcer.check_memory()
                    if self._budget_enforcer.degradation.stop_requested:
                        stop_reason = "budget_exceeded"
                        break

        # --- on_stop callback ------------------------------------------------
        if self._island_model is not None:
            best_island = max(
                self._island_model.islands,
                key=lambda pop: pop.get_best().fitness if pop._evaluated else -float("inf"),
            )
            self._population = best_island

        self.stop_reason = stop_reason or "manual"
        self.stopped_early = bool(
            stop_reason and (
                stop_reason.startswith("patience")
                or "all_strategies_exhausted" in stop_reason
            )
        )
        if on_stop is not None:
            try:
                on_stop(self.stop_reason)
            except Exception:
                pass

        # --- End-of-run artefacts -------------------------------------------
        self._write_run_summary(name, stop_reason, iterations, _wj, _li)
        self._event_bus.emit("run_end", {
            "stop_reason": stop_reason or "manual",
            "iterations": iterations,
        })
        # Automatic report export if configured
        if self._report_autosave_template is not None and self._log_run_dir is not None:
            try:
                _report_path = self._report_autosave_template.format(
                    name=name or "run",
                    date=time.strftime("%Y%m%d_%H%M%S"),
                    example=name or "run",
                )
                # If the template contains a path separator, treat it as absolute
                # or relative to CWD; otherwise place it in the run directory.
                if "/" in _report_path or "\\" in _report_path:
                    _dest = Path(_report_path)
                else:
                    _dest = self._log_run_dir / _report_path
                fmt = _dest.suffix.lstrip(".") or self._report_autosave_format or "html"
                if not _dest.suffix:
                    _dest = _dest.with_suffix(f".{fmt}")
                self.export_run_report(
                    str(_dest), fmt=fmt,
                    stop_reason=stop_reason or "manual",
                    iterations=iterations,
                )
            except Exception as _exc:
                _li("Report autosave failed: %s", _exc)
        if self._tensorboard_writer is not None:
            try:
                self._tensorboard_writer.close()
            except Exception:
                pass
            self._tensorboard_writer = None
        for _tb_backend in self._tracking_backends:
            try:
                _tb_backend.finish()
            except Exception:
                pass

        # Post-training pruning
        if self._post_pruning_enabled:
            self._apply_post_training_pruning(fitness_fn)

        # DARTS-Lite post-training pruning
        if self._darts_enabled and self._population is not None:
            best = self._population.get_best()
            if best is not None:
                best.update_darts_gates()  # ensure gates are current
                n_pruned = best.prune_darts_connections(self._darts_prune_threshold)
                if n_pruned:
                    from yane.util.logger import log_info
                    log_info(
                        "DARTS post-training pruning: removed %d connections "
                        "(threshold=%.2f).", n_pruned, self._darts_prune_threshold,
                    )

        # Knowledge Base: auto-learn from this run
        _best_for_kb = self._population.get_best() if self._population else None
        self._auto_kb_learn(
            best_fitness=_best_for_kb.fitness if _best_for_kb else 0.0,
            stop_reason=stop_reason,
        )

        return iterations

    def _apply_post_training_pruning(self, fitness_fn) -> None:
        """Prune best genome post-training; rollback if fitness drop is too large."""
        best = self._population.get_best()
        if best is None or fitness_fn is None:
            return
        pre_fitness = best.fitness
        # Snapshot connections for rollback (prune only removes connections, not nodes)
        saved_conns = {node: list(node.connections) for node in best.nodes}
        n_removed = best.prune(threshold=self._post_pruning_threshold)
        if n_removed == 0:
            return
        compression_rate = best._prune_stats.get("compression_rate", 0.0)
        post_fitness = fitness_fn(best)
        delta = post_fitness - pre_fitness
        if pre_fitness != 0.0:
            drop_frac = (pre_fitness - post_fitness) / abs(pre_fitness)
        else:
            drop_frac = 0.0 if post_fitness >= pre_fitness else 1.0
        if drop_frac > self._post_pruning_max_drop_frac:
            for node in best.nodes:
                node.connections = saved_conns[node]
            best._invalidate_topology()
            best._prune_stats = {
                "connections_removed": n_removed,
                "nodes_removed": 0,
                "fitness_delta": delta,
                "compression_rate": compression_rate,
                "rolled_back": True,
            }
        else:
            best.fitness = post_fitness
            best._prune_stats["fitness_delta"] = delta
            from yane.util.logger import log_info
            log_info(
                "Post-training pruning: removed %d connections (%.1f%%), "
                "fitness %.4f → %.4f (delta %.4f).",
                n_removed, compression_rate * 100, pre_fitness, post_fitness, delta,
            )

    def get_best(self) -> Genome:
        self._ensure_configured()
        return self._population.get_best()

    def get_ensemble(self, k: int = 3) -> list[Genome]:
        """Return the top-k genomes by fitness for ensemble inference."""
        self._ensure_configured()
        return self._population.get_top(k)

    def make_ensemble(self, k: int = 3, mode: str = "mean") -> "EnsembleGenome":
        """Create an ``EnsembleGenome`` wrapper from the top-k evaluated genomes.

        The wrapper provides a unified ``forward()`` interface that aggregates
        outputs from all members.

        Args:
            k: Number of top genomes to include.
            mode: Aggregation strategy — ``"mean"`` (default), ``"vote"``,
                  or ``"weighted"``.

        Returns:
            An ``EnsembleGenome`` instance.
        """
        from yane.evolution.ensemble import EnsembleGenome
        self._ensure_configured()
        members = self._population.get_top(k)
        if not members:
            raise RuntimeError("No evaluated genomes available for ensemble.")
        return EnsembleGenome(members, mode=mode)

    # --- Report / Run-Postmortem --------------------------------------------

    _report_autosave_template: str | None = None

    def set_report_autosave(
        self,
        path_template: str | None,
        format: str | None = None,
    ) -> None:
        """Enable automatic report export when ``train()`` finishes.

        The template may contain ``{name}``, ``{date}``, and ``{example}``
        placeholders which are substituted at export time.

        Example::

            yane.set_report_autosave("{date}_{example}_report.html")

        Pass ``None`` to disable (default).
        """
        self._report_autosave_template = path_template
        if format is not None:
            self._report_autosave_format = format

    def export_run_report(
        self,
        path: str | None = None,
        fmt: str | None = None,
        format: str | None = None,
        *,
        stop_reason: str = "manual",
        iterations: int = 0,
    ) -> str:
        """Export a structured post-training report.

        The report includes:
        - Fitness curve (SVG inline for HTML)
        - Best genome topology and connections
        - Configuration snapshot
        - Recovery events and anomaly diagnostics
        - Runtime statistics

        Args:
            path: Destination file path (``.html``, ``.md``, ``.json``).
                  If omitted the report is returned as a string but not written.
            fmt: Output format — ``"html"`` (default), ``"md"``, or ``"json"``.
            format: Alias for *fmt* (supports legacy callers).

        Returns:
            The rendered report content as a string.
        """
        resolved_fmt = fmt or format or "html"
        from yane.util.report import export_run_report as _export
        return _export(self, path, resolved_fmt,
                       stop_reason=stop_reason, iterations=iterations)

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

    def landscape_pca(self) -> dict:
        """Return a 2-component PCA projection of all evaluated genomes.

        The result contains ``x``, ``y``, ``fitness``, and ``species_id``
        lists, as well as ``explained_var`` for the two components.
        Useful for fitness-landscape visualization in the GUI or external
        tools.

        Returns an empty dict if there are fewer than 2 evaluated genomes.
        """
        self._ensure_configured()
        from yane.evolution.landscape import population_pca
        evald = self._population._evaluated
        if len(evald) < 2:
            return {}
        return population_pca(evald)

    def export_landscape_csv(self, path: str) -> None:
        """Export the current PCA landscape snapshot as CSV."""
        from yane.evolution.landscape import export_landscape_csv
        export_landscape_csv(self.landscape_pca(), path)

    def export_landscape_png(self, path: str, *, width: int = 900, height: int = 640) -> None:
        """Export the current PCA landscape snapshot as a PNG scatterplot."""
        from yane.evolution.landscape import export_landscape_png
        export_landscape_png(self.landscape_pca(), path, width=width, height=height)

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
        info["generation"] = (
            self._population._total_submitted // max(1, self._population.max_size)
        )
        info["evaluations_done"] = self._n_evaluations_done
        if self._adaptive_recovery_enabled:
            checked = max(1, self._recovery_checked)
            info.update({
                "adaptive_recovery_enabled": True,
                "recovery_events": list(self._recovery_events),
                "last_recovery_strategy": (
                    self._recovery_events[-1]["strategy"] if self._recovery_events else None
                ),
                "recovery_success_rate": self._recovery_successes / checked,
                "no_improvement_generations": max(
                    0,
                    info["generation"] - self._recovery_last_improvement_generation,
                ),
                "early_stop_triggered": self.stopped_early,
                "stop_conditions_met": self._recovery_stop_conditions(info),
                "injections_total": self._population._n_diversity_injection,
                "last_injection_generation": (
                    self._recovery_events[-1]["generation"]
                    if self._recovery_events else None
                ),
            })
        if self._last_validation_fitness is not None:
            info["validation_fitness"] = self._last_validation_fitness
        if self._runner.anytime_enabled:
            total = max(1, self._runner.anytime_total_genomes)
            info.update({
                "anytime_eval_enabled": True,
                "anytime_avg_evals_per_genome": self._runner.anytime_total_calls / total,
                "anytime_saved_evals": self._runner.anytime_saved_calls,
                "anytime_promotion_rate": self._runner.anytime_promoted / total,
                "anytime_promotion_frac": self._runner.anytime_promotion_frac,
                "anytime_min_evals": self._runner.anytime_min_evals,
                "anytime_max_evals": self._runner.anytime_max_evals,
                "anytime_promoted_variance": (
                    sum(self._runner.anytime_promoted_variances)
                    / len(self._runner.anytime_promoted_variances)
                    if self._runner.anytime_promoted_variances else 0.0
                ),
            })
        if self._eval_middlewares:
            info["eval_middleware_count"] = len(self._eval_middlewares)
            info["eval_middleware"] = dict(self._eval_middleware_diagnostics)
        if self._checkpoint_policy_enabled:
            info.update({
                "checkpoint_policy_enabled": True,
                "last_auto_checkpoint_iteration": self._last_checkpoint_iteration,
                "rolling_checkpoint_count": len(self._checkpoint_paths),
                "best_checkpoint_path": self._best_checkpoint_path,
            })
        if self._transfer_freeze_records or self._transfer_unfreeze_enabled:
            info.update({
                "transfer_frozen_layers": list(self._transfer_frozen_layers),
                "transfer_frozen_connections": len(self._transfer_freeze_records),
                "transfer_unfreeze_enabled": self._transfer_unfreeze_enabled,
                "transfer_unfreeze_progress": self._transfer_unfreeze_progress,
            })
        # Selection strategy diagnostics
        if self._population is not None:
            strategy = self._population.selection_strategy
            parent_fitnesses = self._population._recent_parent_fitnesses
            all_fitnesses = [g.fitness for g in self._population._evaluated if g.fitness is not None]
            info["selection"] = {
                "strategy": repr(strategy),
                "avg_parent_fitness": (
                    sum(parent_fitnesses) / len(parent_fitnesses)
                    if parent_fitnesses else None
                ),
                "avg_pool_fitness": (
                    sum(all_fitnesses) / len(all_fitnesses)
                    if all_fitnesses else None
                ),
                "n_species_overrides": len(self._population.selection_strategies_by_species),
            }
        # Adaptive population size diagnostics
        if self._adaptive_pop_enabled and self._population is not None:
            pop = self._population
            info.update({
                "adaptive_pop_enabled": True,
                "adaptive_pop_schedule": pop._adaptive_pop_schedule,
                "current_pop_size": pop.max_size,
                "n_pop_size_adjustments": pop._n_pop_size_adjustments,
                "last_resize_trigger": pop._last_resize_trigger or None,
                "pop_size_history": list(pop._pop_size_history[-10:]),
            })
        return info

    def set_target_species(
        self,
        n: int | None = None,
        *,
        n_min: int | None = None,
        n_max: int | None = None,
        tune_interval: int = 1,
        threshold_min: float = 0.01,
        threshold_max: float = 1.5,
    ) -> None:
        """Set the target number or target band of species.

        The adaptive compatibility threshold rises/falls automatically to keep
        the actual species count close to this target. Higher values protect
        more structural niches and help escape local optima (especially for
        XOR-like tasks where intermediate structures are temporarily worse).

        Pass ``0`` to auto-compute from population size: ``sqrt(pop_size)``.
        Pass ``None`` to disable explicit targeting and use static threshold
        behaviour. Use ``n_min``/``n_max`` to keep species in a range instead
        of oscillating around one exact count.

        Default: 5. For small discrete-mapping tasks (binary increment,
        XOR variants): 10–20 works significantly better.
        """
        if n_min is not None or n_max is not None:
            if n_min is None or n_max is None:
                raise ValueError("n_min and n_max must be provided together")
            if n_min < 1 or n_max < n_min:
                raise ValueError("target species band must satisfy 1 <= n_min <= n_max")
            self._target_species = max(1, int(round((n_min + n_max) / 2)))
            self._target_species_min = int(n_min)
            self._target_species_max = int(n_max)
        elif n is None:
            self._target_species = None
            self._target_species_min = None
            self._target_species_max = None
        else:
            self._target_species = max(1, n) if n > 0 else 0
            self._target_species_min = None
            self._target_species_max = None
        self._compat_tune_interval = max(1, int(tune_interval))
        self._compat_threshold_min = max(1e-9, float(threshold_min))
        self._compat_threshold_max = max(self._compat_threshold_min, float(threshold_max))
        if self._population is not None:
            self._apply_speciation_tuning_config(self._population)

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
            from yane.evolution.compatibility import TopologyDistance as _TD
            if type(self._population.compatibility_distance) is _TD:
                only_enabled = metric == "topology_no_disabled"
                self._population.compatibility_distance = _TD(only_enabled=only_enabled)

    # ── Getters ────────────────────────────────────────────────────────────

    def get_config(self) -> dict:
        """Return the full configuration dict (same format as config.json)."""
        return self._config_dict()

    def get_target_species(self) -> int:
        return (self._population._target_species if self._population is not None
                else (self._target_species if self._target_species is not None else 0))

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
        }, codec=self._checkpoint_codec)

    _checkpoint_codec: str = "pickle"

    def set_checkpoint_codec(self, codec_name: str) -> None:
        """Set the codec used for checkpoint serialization.

        Built-in codecs: ``"pickle"`` (default), ``"json"``.

        The codec is stored so that ``save_checkpoint()`` uses it
        automatically.  ``load_checkpoint()`` auto-detects the codec
        from the file header.
        """
        from yane.evolution.codec import get_codec
        get_codec(codec_name)  # validate
        self._checkpoint_codec = codec_name

    def migrate_checkpoint(self, path: str | Path, target_codec: str) -> None:
        """Convert a checkpoint file to a different codec format.

        Args:
            path: Path to an existing checkpoint.
            target_codec: Target codec name (``"pickle"`` or ``"json"``).

        The original file is *not* modified; a new file is written with
        the target codec at ``path.with_suffix(f'.{target_codec}')``.
        """
        from yane.evolution.codec import detect_codec, get_codec
        orig_codec = self._checkpoint_codec
        try:
            self._checkpoint_codec = detect_codec(Path(path).read_bytes())
            self.load_checkpoint(path)
            self._checkpoint_codec = target_codec
            out_path = Path(path).with_suffix(f".{target_codec}")
            self.save_checkpoint(str(out_path))
        finally:
            self._checkpoint_codec = orig_codec

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore population state from a checkpoint file."""
        payload = _ckpt.read(path)
        cfg = payload["config"]
        self._last_checkpoint_compatibility = None
        if self.is_configured:
            report = _ckpt.compatibility_report(
                cfg,
                self._config_dict(),
                stored_hash=payload.get("config_hash"),
            )
            self._last_checkpoint_compatibility = report
            if report["level"] == _ckpt.CompatibilityLevel.BREAKING.value:
                changed = ", ".join(item["path"] for item in report["diff"][:6])
                raise ValueError(
                    "Checkpoint is incompatible with the current configuration "
                    f"({changed}). Load it into a fresh NeuroEvolution instance "
                    "or use warm_start_from_checkpoint() for transfer."
                )
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
        freeze_layers: list[str] | None = None,
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

        if freeze_layers:
            self._transfer_freeze_records = {}
            self._transfer_frozen_layers = []
            self._transfer_unfreeze_progress = 0.0
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
            if freeze_layers:
                self._freeze_genome_layers(genome, freeze_layers)
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
        self._apply_speciation_tuning_config(self._population)
        self._population.elite_count = self._elite_count
        self._population.species_elite_count = self._species_elite_count
        self._population._adaptive_pop_enabled = self._adaptive_pop_enabled
        self._population._adaptive_pop_min = self._adaptive_pop_min
        self._population._adaptive_pop_max = self._adaptive_pop_max
        self._population._adaptive_pop_rate = self._adaptive_pop_rate
        self._population._adaptive_pop_schedule = self._adaptive_pop_schedule
        self._population._weight_blend_alpha = (
            self._weight_blend_alpha if self._weight_inheritance_enabled else -1.0
        )
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

    def set_transfer_unfreeze(
        self,
        enabled: bool = True,
        *,
        start_generation: int = 0,
        duration_generations: int = 25,
    ) -> None:
        """Progressively restore frozen transfer-learning connection rates.

        Frozen connections are those produced by ``load_genome_as_seed`` or
        ``warm_start_from_checkpoint(..., freeze_layers=[...])``. During the
        schedule their mutation and spike rates ramp linearly from zero back to
        their recorded pre-freeze values.
        """
        self._transfer_unfreeze_enabled = bool(enabled)
        self._transfer_unfreeze_start_generation = max(0, int(start_generation))
        self._transfer_unfreeze_generations = max(1, int(duration_generations))
        if not enabled:
            self._transfer_unfreeze_progress = 0.0

    def progressive_unfreeze_transfer(self, progress: float) -> None:
        """Manually set transfer unfreeze progress in ``[0, 1]``."""
        progress = max(0.0, min(1.0, float(progress)))
        self._transfer_unfreeze_progress = progress
        self._apply_transfer_unfreeze_progress(progress)

    def _tick_transfer_unfreeze(self, generation: int) -> None:
        if not self._transfer_unfreeze_enabled or not self._transfer_freeze_records:
            return
        start = self._transfer_unfreeze_start_generation
        duration = self._transfer_unfreeze_generations
        progress = (generation - start) / duration
        self.progressive_unfreeze_transfer(progress)
        if self._transfer_unfreeze_progress >= 1.0:
            self._transfer_unfreeze_enabled = False

    def _apply_transfer_unfreeze_progress(self, progress: float) -> None:
        if self._population is None:
            return
        genomes = list(self._population._evaluated) + list(self._population._unevaluated)
        template = getattr(self._population, "_template", None)
        if template is not None:
            genomes.append(template)
        for genome in genomes:
            for node in genome.nodes:
                for conn in node.connections:
                    rec = self._transfer_freeze_records.get(conn.innovation)
                    if rec is None:
                        continue
                    conn.mutation.shift_rate = rec["shift_rate"] * progress
                    conn.mutation.custom_rate = rec["custom_rate"] * progress
                    conn.mutation.rate_mutation_rate = rec["rate_mutation_rate"] * progress
                    conn.spike_rate = rec["spike_rate"] * progress

    def _freeze_genome_layers(self, genome: Genome, freeze_layers: list[str]) -> None:
        from yane.core.node import NodeType as _NT
        valid = {_NT.INPUT.value, _NT.HIDDEN.value, _NT.OUTPUT.value}
        layers = {str(layer).lower() for layer in freeze_layers}
        unknown = layers - valid
        if unknown:
            raise ValueError(f"Unknown freeze layers: {sorted(unknown)}")
        self._transfer_frozen_layers = sorted(layers)
        for node in genome.nodes:
            for conn in node.connections:
                if node.type.value not in layers and conn.target.type.value not in layers:
                    continue
                if conn.innovation not in self._transfer_freeze_records:
                    self._transfer_freeze_records[conn.innovation] = {
                        "shift_rate": conn.mutation.shift_rate,
                        "custom_rate": conn.mutation.custom_rate,
                        "rate_mutation_rate": conn.mutation.rate_mutation_rate,
                        "spike_rate": conn.spike_rate,
                    }
                conn.mutation.shift_rate = 0.0
                conn.mutation.custom_rate = 0.0
                conn.mutation.rate_mutation_rate = 0.0
                conn.spike_rate = 0.0

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
        """Sanitize + efficiency penalty + hardware penalty. Applied by every submission path."""
        result = finalize_fitness_value(
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
        if self._hw_constraints is not None and genome is not None:
            from yane.evolution.hardware_aware import compute_hardware_metrics, compute_penalty
            try:
                hw_metrics = compute_hardware_metrics(genome, self._hw_constraints)
                result -= compute_penalty(hw_metrics, self._hw_constraints)
            except Exception:
                pass   # hardware estimation is non-critical; never break training
        if genome is not None:
            system = getattr(self, "_safety_system", None)
            if system is not None:
                try:
                    result = system.evaluate(genome, result)
                except Exception:
                    pass  # safety checks must never break training
        return result

    def _run_evaluations(
        self, genome: Genome, fitness_fn: Callable[[Genome], float]
    ) -> EvaluationResult:
        if self._matrix_forward_enabled:
            return self._run_with_matrix_forward(genome, fitness_fn)
        _grouper_active = False
        if self._input_grouping_enabled and getattr(genome, "grouper", None) is not None:
            _grouper_active = True
            _grouper = genome.grouper
            _orig_fwd_grp = genome.forward
            genome.__dict__["forward"] = lambda data: _orig_fwd_grp(_grouper.transform(data))
        _out_grouper_active = False
        if self._output_grouping_enabled and getattr(genome, "out_grouper", None) is not None:
            _out_grouper_active = True
            _out_grouper = genome.out_grouper
            _orig_fwd_out = genome.forward
            genome.__dict__["forward"] = lambda data, _fwd=_orig_fwd_out, _og=_out_grouper: _og.expand(_fwd(data))

        _stdp_active = False
        if self._stdp_enabled:
            from yane.evolution.stdp import (
                genome_has_stdp as _ghs,
                init_stdp_base_weights as _init_stdp,
                restore_stdp_weights as _restore_stdp,
                apply_stdp_update as _apply_stdp,
            )
            if _ghs(genome):
                _stdp_active = True
                _wmin = self._stdp_weight_min
                _wmax = self._stdp_weight_max
                _init_stdp(genome)   # saves _base_weight if not already set
                _orig_fwd_stdp = genome.forward
                _orig_reset_stdp = genome.reset
                def _stdp_fwd(data, _fwd=_orig_fwd_stdp, _g=genome, _mn=_wmin, _mx=_wmax):
                    result = _fwd(data)
                    # The compiled forward path does not update node.value for
                    # input nodes — set them explicitly for pre-synaptic STDP.
                    for _in in _g.input_nodes:
                        _idx = getattr(_in, 'input_index', 0)
                        if _idx < len(data):
                            _in.value = float(data[_idx]) * _in.input_scale
                    _apply_stdp(_g, _mn, _mx)
                    return result
                def _stdp_reset(_reset=_orig_reset_stdp, _g=genome):
                    _reset()
                    _restore_stdp(_g)
                genome.__dict__["forward"] = _stdp_fwd
                genome.__dict__["reset"] = _stdp_reset

        _neuromod_active = False
        if self._neuromodulation_enabled:
            from yane.evolution.neuromodulation import (
                genome_has_modulators as _ghm,
                apply_modulation_to_weights as _apply_mod,
                update_modulation_gains as _update_gains,
                restore_modulation_weights as _restore_mod,
            )
            if _ghm(genome):
                _neuromod_active = True
                _orig_fwd_mod = genome.forward
                _orig_reset_mod = genome.reset
                def _mod_fwd(data, _fwd=_orig_fwd_mod, _g=genome):
                    _apply_mod(_g)     # pre-forward: apply gains from previous call
                    result = _fwd(data)
                    _update_gains(_g)  # post-forward: record new MODULATOR outputs
                    return result
                def _mod_reset(_reset=_orig_reset_mod, _g=genome):
                    _reset()
                    _restore_mod(_g)   # reset: restore base weights + neutral gains
                genome.__dict__["forward"] = _mod_fwd
                genome.__dict__["reset"] = _mod_reset

        _ltc_active = False
        if self._ltc_enabled:
            from yane.evolution.ltc import genome_has_ltc as _ghl, apply_ltc_update as _apply_ltc
            if _ghl(genome):
                _ltc_active = True
                _orig_fwd_ltc = genome.forward
                def _ltc_fwd(data, _fwd=_orig_fwd_ltc, _g=genome):
                    result = _fwd(data)
                    _apply_ltc(_g)
                    return result
                genome.__dict__["forward"] = _ltc_fwd

        _attention_active = False
        if self._attention_enabled and getattr(genome, "attention_block", None) is not None:
            _attention_active = True
            _attn = genome.attention_block
            _orig_fwd_attn = genome.forward
            genome.__dict__["forward"] = lambda data, _fwd=_orig_fwd_attn, _a=_attn: _fwd(_a.forward(data))

        if self._input_transform is not None:
            _t = self._input_transform
            _orig_fwd = genome.forward
            genome.__dict__["forward"] = lambda data: _orig_fwd(_t(data))
        _aug_active = False
        if self._aug_pool is not None:
            _aug_active = True
            _aug_pipeline = self._aug_pool._active
            _aug_rng = self._aug_pool.rng
            _fwd_pre_aug = genome.forward
            genome.__dict__["forward"] = (
                lambda data, _fwd=_fwd_pre_aug, _pl=_aug_pipeline, _r=_aug_rng:
                    _fwd(_pl.apply(list(data), _r))
            )
        _curiosity_active = False
        if self._curiosity_enabled and self._curiosity_module is not None:
            _curiosity_active = True
            _curiosity_recorded: list = []
            _cr = _curiosity_recorded
            _fwd_pre_curiosity = genome.forward
            def _curiosity_fwd(data):
                out = _fwd_pre_curiosity(data)
                _cr.append((list(data), list(out)))
                return out
            genome.__dict__["forward"] = _curiosity_fwd
            _base_fn = fitness_fn
            _mod = self._curiosity_module
            _cw = self._curiosity_weight
            def fitness_fn(g):  # noqa: F821
                _cr.clear()
                base = _base_fn(g)
                # Write task-only score onto the genome being evaluated so that
                # finalize_fitness_value (called for every Lamarck candidate)
                # can record the correct raw_fitness regardless of eval order.
                g._curiosity_task_base = base
                if _cr:
                    total_err = 0.0
                    for inp, out in _cr:
                        total_err += _mod.error(inp, out)
                        _mod.update(inp, out)
                    raw_bonus = total_err / len(_cr)
                    # Cap: bonus ≤ abs(base) so it can't overwhelm a negative task
                    # score or reward numerically unstable (exploding) outputs.
                    bonus = min(raw_bonus, max(1.0, abs(base)))
                    return base + _cw * bonus
                return base
        try:
            original_fn = fitness_fn
            if self._eval_middlewares:
                import inspect
                if not inspect.isgeneratorfunction(fitness_fn):
                    def _wrapped(g: Genome):
                        ctx = EvalContext()
                        value = apply_middleware(g, original_fn, self._eval_middlewares, ctx)
                        self._eval_middleware_diagnostics.update(ctx.diagnostics)
                        return value
                    fitness_fn = _wrapped
            result = self._runner.run(genome, fitness_fn, self._population, self._lamarck)
            _cs = getattr(original_fn, "_component_scores", None)
            if _cs:
                self._eval_middleware_diagnostics["evaluator_components"] = dict(_cs)
            return result
        finally:
            if self._input_transform is not None or _aug_active or _curiosity_active or _grouper_active or _out_grouper_active or _stdp_active or _neuromod_active or _attention_active or _ltc_active:
                genome.__dict__.pop("forward", None)
            if _stdp_active or _neuromod_active:
                genome.__dict__.pop("reset", None)
            if _stdp_active:
                # Restore evolved base weights after evaluation
                from yane.evolution.stdp import restore_stdp_weights as _rs
                _rs(genome)
            if _neuromod_active:
                # Restore modulation state after evaluation
                from yane.evolution.neuromodulation import restore_modulation_weights as _rm
                _rm(genome)
            # Note: _curiosity_task_base is intentionally NOT removed here.
            # finalize_fitness_value() reads and removes it so that raw_fitness
            # records the task-only score even when Lamarck evaluates multiple times.

    def _run_with_matrix_forward(
        self, genome: Genome, fitness_fn: Callable[[Genome], float]
    ) -> EvaluationResult:
        from yane.evolution.matrix_export import is_matrix_compatible, forward_matrix
        patched = False
        if is_matrix_compatible(genome):
            try:
                exported = self._matrix_cache.get(genome)
                _exp = exported
                _t = self._input_transform

                def _matrix_fwd(data):
                    if type(data) is not list:
                        data = [float(x) for x in data]
                    if _t is not None:
                        data = _t(data)
                    return forward_matrix(_exp, data)

                genome.__dict__["forward"] = _matrix_fwd
                patched = True
                self._matrix_hits += 1
            except Exception:
                self._matrix_misses += 1
        else:
            self._matrix_misses += 1
            if self._input_transform is not None:
                _t = self._input_transform
                _orig_fwd = genome.forward
                genome.__dict__["forward"] = lambda data: _orig_fwd(_t(data))
                patched = True
        original_fn = fitness_fn
        try:
            if self._eval_middlewares:
                import inspect
                if not inspect.isgeneratorfunction(fitness_fn):
                    def _wrapped(g: Genome):
                        ctx = EvalContext()
                        value = apply_middleware(g, original_fn, self._eval_middlewares, ctx)
                        self._eval_middleware_diagnostics.update(ctx.diagnostics)
                        return value
                    fitness_fn = _wrapped
            result = self._runner.run(genome, fitness_fn, self._population, self._lamarck)
            _cs = getattr(original_fn, "_component_scores", None)
            if _cs:
                self._eval_middleware_diagnostics["evaluator_components"] = dict(_cs)
            return result
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

    def _recovery_stop_conditions(self, mem: dict) -> list[str]:
        if not self._adaptive_recovery_enabled:
            return []
        generation = mem.get("generation", 0)
        conditions: list[str] = []
        no_improve = generation - self._recovery_last_improvement_generation
        if no_improve >= self._recovery_early_stopping_patience:
            conditions.append("patience_exhausted")
        if mem.get("fitness_iqr", float("inf")) < self._recovery_diversity_iqr_threshold:
            conditions.append("diversity_collapsed")
        if self._recovery_strategy_index >= max(0, len(self._recovery_strategies) - 1):
            conditions.append("all_strategies_exhausted")
        return conditions

    def _tick_adaptive_recovery(self, mem: dict, iterations: int, _li: Callable) -> str | None:
        if not self._adaptive_recovery_enabled or self._population is None:
            return None
        generation = mem.get("generation", iterations // max(1, self._population.max_size))
        best = float(mem.get("max_fitness", -float("inf")))
        if best > self._recovery_best_fitness + self._recovery_min_delta:
            self._recovery_best_fitness = best
            self._recovery_last_improvement_generation = generation

        self._settle_pending_recoveries(best, generation)
        conditions = self._recovery_stop_conditions({**mem, "generation": generation})
        if ("patience_exhausted" in conditions
                and ("diversity_collapsed" in conditions or "all_strategies_exhausted" in conditions)
                and generation >= self._recovery_warmup):
            self.stopped_early = True
            self.stop_reason = "_and_".join(conditions)
            _li("Training stopped: adaptive recovery early stop  reason=%s", self.stop_reason)
            return self.stop_reason

        if generation < self._recovery_warmup:
            return None
        if generation - self._last_recovery_generation < self._recovery_cooldown:
            return None
        trigger = self._recovery_trigger(mem)
        if trigger is None:
            return None
        self._apply_recovery_strategy(trigger, best, generation, _li)
        return None

    def _settle_pending_recoveries(self, best: float, generation: int) -> None:
        remaining: list[dict] = []
        for event in self._pending_recoveries:
            if generation - event["generation"] < self._recovery_cooldown:
                remaining.append(event)
                continue
            self._recovery_checked += 1
            success = best > event["best_before"] + self._recovery_min_delta
            event["success"] = success
            if event.get("strategy") == "lamarck_burst":
                self._lamarck.set_budget(event.get("lamarck_budget_before"))
            if success:
                self._recovery_successes += 1
                if self._recovery_escalate:
                    self._recovery_strategy_index = 0
            elif self._recovery_escalate and self._recovery_strategies:
                self._recovery_strategy_index = min(
                    len(self._recovery_strategies) - 1,
                    self._recovery_strategy_index + 1,
                )
        self._pending_recoveries = remaining

    def _recovery_trigger(self, mem: dict) -> str | None:
        anomaly_kinds = set(mem.get("anomaly_kinds", []))
        if mem.get("fitness_iqr", float("inf")) < self._recovery_diversity_iqr_threshold:
            return "diversity_collapse"
        if anomaly_kinds.intersection({"diversity_collapse", "homogenization", "stuck_speciation"}):
            return ",".join(sorted(anomaly_kinds))
        stagnation = mem.get("stagnation_count", 0)
        threshold = max(1, mem.get("stagnation_threshold", 1))
        if mem.get("species_count", 0) <= 1 and stagnation / threshold >= 0.5:
            return "stuck_speciation"
        return None

    def _apply_recovery_strategy(
        self,
        trigger: str,
        best_before: float,
        generation: int,
        _li: Callable,
    ) -> None:
        if not self._recovery_strategies:
            return
        strategy = self._recovery_strategies[self._recovery_strategy_index]
        pop = self._population
        n = max(1, int(round(pop.max_size * self._recovery_injection_frac)))
        event = {
            "generation": generation,
            "strategy": strategy,
            "trigger": trigger,
            "best_before": best_before,
            "n_genomes": n,
        }
        if strategy == "partial_restart":
            keep = max(1, len(pop._evaluated) - n)
            pop.shrink_to(keep)
            for _ in range(n):
                pop._inject_fresh_genome()
        elif strategy == "lamarck_burst":
            self._recovery_last_lamarck_budget = self._lamarck.budget_per_gen
            base = self._lamarck.budget_per_gen
            self._lamarck.set_budget(max(pop.max_size, (base or 0) * 2, n))
            event["lamarck_budget_before"] = base
            event["lamarck_budget_after"] = self._lamarck.budget_per_gen
        else:
            for _ in range(n):
                pop._inject_fresh_genome()
        self._last_recovery_generation = generation
        self._recovery_events.append(event)
        if len(self._recovery_events) > 200:
            self._recovery_events = self._recovery_events[-200:]
        self._pending_recoveries.append(event)
        _li("Adaptive recovery: strategy=%s trigger=%s generation=%d n=%d",
            strategy, trigger, generation, n)

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

    def _checkpoint_path(
        self,
        iteration: int,
        run_name: str,
        kind: str,
        best_fitness: float,
    ) -> Path:
        generation = iteration // max(1, self._population_size)
        path = Path(self._checkpoint_path_template.format(
            run_name=run_name,
            iteration=iteration,
            generation=generation,
            best_fitness=f"{best_fitness:.6f}",
            kind=kind,
        ))
        if not path.is_absolute() and self._log_run_dir is not None:
            path = self._log_run_dir / path
        return path

    def _maybe_auto_checkpoint(self, iteration: int, run_name: str) -> None:
        if not self._checkpoint_policy_enabled or self._population is None:
            return
        if not self._population._evaluated:
            return
        best = self._population.get_best()
        best_fitness = best.raw_fitness
        if iteration % self._checkpoint_interval == 0:
            path = self._checkpoint_path(iteration, run_name, "rolling", best_fitness)
            self.save_checkpoint(path)
            path_str = str(path)
            if path_str not in self._checkpoint_paths:
                self._checkpoint_paths.append(path_str)
            self._last_checkpoint_iteration = iteration
            self._trim_rolling_checkpoints()
        if self._checkpoint_keep_best and best_fitness > self._last_checkpoint_best_fitness:
            path = self._checkpoint_path(iteration, run_name, "best", best_fitness)
            self.save_checkpoint(path)
            self._best_checkpoint_path = str(path)
            self._last_checkpoint_best_fitness = best_fitness

    def _trim_rolling_checkpoints(self) -> None:
        while len(self._checkpoint_paths) > self._checkpoint_max_keep:
            old = self._checkpoint_paths.pop(0)
            if old == self._best_checkpoint_path:
                continue
            try:
                path = Path(old)
                path.unlink(missing_ok=True)
                Path(str(path) + ".json").unlink(missing_ok=True)
            except Exception:
                pass

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
        if self._run_database is None:
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

        # --- Persist run record to database ----------------------------------
        if self._run_database is not None and self._active_run_id is not None:
            try:
                from yane.util.run_database import load_fitness_history_from_csv
                _csv_path = self._log_run_dir / "fitness_history.csv"
                fitness_history = load_fitness_history_from_csv(_csv_path)
                # Ensure the final best is recorded accurately — the last CSV
                # heartbeat may predate the true final improvement.
                if not fitness_history or fitness_history[-1].get("best_fitness") != best.fitness:
                    fitness_history.append({
                        "iteration": iterations,
                        "best_fitness": best.fitness,
                    })
                # Sanitise mem dict: convert non-JSON-serialisable values to str.
                diagnostics = json.loads(json.dumps(mem, default=str))
                artifacts = {
                    "log_dir": str(self._log_run_dir),
                    "run_log": str(self._log_run_dir / "run.log"),
                    "fitness_history_csv": str(_csv_path),
                    "fitness_history_jsonl": str(
                        self._log_run_dir / "fitness_history.jsonl"
                    ),
                    "best_genome_pkl": str(self._log_run_dir / "best_genome.pkl"),
                }
                self._run_database.finish_run(
                    run_id=self._active_run_id,
                    fitness_history=fitness_history,
                    diagnostics=diagnostics,
                    stop_reason=stop_reason or "manual",
                    artifacts=artifacts,
                )
                self._autosave_report(name, self._active_run_id)
            except Exception:
                pass

    def _run_hybrid_backprop(self, generation: int) -> None:
        """Execute one backprop phase for the hybrid NEAT mode."""
        from yane.util.logger import get_logger
        log = get_logger()
        from yane.evolution.hybrid_neat import run_hybrid_backprop

        teachers = self._population.get_top(self._hybrid_top_k)
        if not teachers:
            return

        # Determine training data
        if self._hybrid_train_data is not None:
            inputs_batch = [list(x) for x, _ in self._hybrid_train_data]
            targets_batch = [list(y) for _, y in self._hybrid_train_data]
        else:
            # Self-supervised: use replay buffer inputs + best genome as teacher
            rb = self._hybrid_replay_buffer
            if rb is None or len(rb) == 0:
                return
            n_sample = min(self._hybrid_bp_batch_size * 4, len(rb))
            inputs_batch = rb.sample(n_sample)
            if not inputs_batch:
                return
            best = self._population.get_best()
            targets_batch = []
            for inp in inputs_batch:
                best.reset()
                try:
                    targets_batch.append([float(v) for v in best.forward(inp)])
                except Exception:
                    targets_batch.append([0.0] * len(best.output_nodes))

        try:
            result = run_hybrid_backprop(
                genomes=teachers,
                inputs_batch=inputs_batch,
                targets_batch=targets_batch,
                bp_epochs=self._hybrid_bp_epochs,
                bp_lr=self._hybrid_bp_lr,
                bp_batch_size=self._hybrid_bp_batch_size,
            )
            log.info(
                "[hybrid] Gen %d: backprop on %d genomes, final losses: %s",
                generation, len(teachers),
                [f"{l:.4f}" for l in result.get("losses", [])],
            )
        except ImportError as e:
            log.warning("[hybrid] Backprop skipped: %s", e)
        except Exception as e:
            log.warning("[hybrid] Backprop error (ignored): %s", e)

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

    # ── Event system ────────────────────────────────────────────────────────

    def on(self, event: str, fn: Callable) -> None:
        """Register *fn* as a handler for training *event*.

        Built-in events: ``"generation_end"``, ``"new_best"``, ``"stagnation"``,
        ``"run_end"``, ``"anomaly"``.  Custom events can be emitted via ``emit()``.
        """
        self._event_bus.on(event, fn)

    def off(self, event: str, fn: Callable) -> None:
        """Unregister *fn* from *event*."""
        self._event_bus.off(event, fn)

    def emit(self, event: str, payload=None) -> None:
        """Emit a custom event (e.g. from inside a fitness function)."""
        self._event_bus.emit(event, payload)

    # ── Anomaly detection ────────────────────────────────────────────────────

    def set_anomaly_detectors(self, detectors=None) -> None:
        """Enable anomaly detection during training.

        Args:
            detectors: List of detector instances (FitnessCollapseDetector, etc.)
                       or None to use DEFAULT_DETECTORS (all four built-in detectors).

        Detectors are checked at every heartbeat (every 100 iterations).
        Detected anomalies are logged and emitted as ``"anomaly"`` events.
        Diagnostics are added to ``population_memory_info()`` under
        ``"anomalies_detected"`` and ``"last_anomaly"``.
        """
        from yane.evolution.anomaly_detection import AnomalyDetectorSet, DEFAULT_DETECTORS
        self._anomaly_detectors = AnomalyDetectorSet(
            detectors if detectors is not None else DEFAULT_DETECTORS
        )

    # ── Fitness transform ────────────────────────────────────────────────────

    def set_fitness_transform(self, transform) -> None:
        """Set a fitness transform applied at each generation boundary.

        *transform* is any callable ``(list[float]) -> list[float]`` that maps
        the raw fitness values of all evaluated genomes to transformed values.
        The transformed values replace ``genome.fitness`` (used for selection);
        ``genome.raw_fitness`` remains unchanged (used for stop conditions).

        Built-in transforms: ``RankTransform``, ``SigmaScaling``,
        ``LinearNormalize``, ``ClipTransform``, ``ChainTransform``.

        Pass ``None`` to disable (default).
        """
        self._fitness_transform = transform

    def set_auto_fitness_shaping(self, enabled: bool = True) -> None:
        """Automatically detect and apply fitness transformations.

        When enabled, the fitness landscape is analysed every 50 generations.
        If the analysis detects strong skewness or plateaus, a suitable
        transform (``RankTransform`` or ``SigmaScaling``) is automatically
        applied to improve selection pressure.

        The analysis results are available via ``population_memory_info()``
        under the key ``"auto_fitness_shaping"``.
        """
        self._auto_fitness_shaping_enabled = enabled
        if not enabled:
            self._fitness_transform = None

    # ── Validation ───────────────────────────────────────────────────────────

    def set_validation_fn(self, fn: Callable | None) -> None:
        """Register a validation function run on the best genome each generation.

        *fn* has the same signature as a fitness function: ``fn(genome) -> float``.
        The result is stored separately and logged in ``population_memory_info()``
        under ``"validation_fitness"`` — it is never used for selection.

        Pass ``None`` to disable.
        """
        self._validation_fn = fn
        self._last_validation_fitness = None

    # ── Config persistence ────────────────────────────────────────────────────

    def save_config(self, path) -> None:
        """Write the full configuration as JSON to *path*."""
        import json
        Path(path).write_text(json.dumps(self._config_dict(), indent=2))

    def load_config(self, path) -> None:
        """Restore numeric training parameters from a JSON config file.

        Only pre-configure parameters are applied (population size, iteration
        limits, Lamarck settings, etc.).  Call ``configure()`` after this to
        rebuild the population with the restored settings.
        """
        import json
        self._load_config_dict(json.loads(Path(path).read_text()))

    def _load_config_dict(self, cfg: dict) -> None:
        """Apply a config dict (as returned by _config_dict()) to this instance."""
        # Reproducibility
        if cfg.get("seed") is not None:
            self.set_seed(cfg["seed"])

        # Population
        if cfg.get("population_size") is not None:
            self.set_population_size(cfg["population_size"])
        if cfg.get("n_workers") is not None:
            self.set_n_workers(cfg["n_workers"])
        if "target_species_min" in cfg and cfg.get("target_species_min") is not None:
            self.set_target_species(
                n_min=cfg["target_species_min"],
                n_max=cfg.get("target_species_max", cfg["target_species_min"]),
                tune_interval=cfg.get("compat_tune_interval", 1),
            )
        elif "target_species" in cfg:
            self.set_target_species(
                cfg.get("target_species"),
                tune_interval=cfg.get("compat_tune_interval", 1),
            )

        # Stopping criteria
        if cfg.get("min_fitness") is not None:
            self.set_min_fitness(cfg["min_fitness"])
        if cfg.get("max_iterations") is not None:
            self.set_max_iterations(cfg["max_iterations"])
        if cfg.get("max_evaluations") is not None:
            self.set_max_evaluations(cfg["max_evaluations"])
        if cfg.get("checkpoint_policy_enabled"):
            self.set_checkpoint_policy(
                interval=cfg.get("checkpoint_interval", 100),
                keep_best=cfg.get("checkpoint_keep_best", True),
                max_keep=cfg.get("checkpoint_max_keep", 5),
                path_template=cfg.get("checkpoint_path_template", "{run_name}_{iteration}.pkl"),
            )
        if cfg.get("adaptive_recovery_enabled"):
            self.set_adaptive_recovery(
                enabled=True,
                strategies=cfg.get("adaptive_recovery_strategies"),
                cooldown=cfg.get("adaptive_recovery_cooldown", 20),
                escalate=cfg.get("adaptive_recovery_escalate", True),
                diversity_iqr_threshold=cfg.get("adaptive_recovery_diversity_iqr_threshold", 1e-4),
                injection_frac=cfg.get("adaptive_recovery_injection_frac", 0.1),
                early_stopping_patience=cfg.get("adaptive_recovery_early_stopping_patience", 500),
                warmup=cfg.get("adaptive_recovery_warmup", 100),
                min_delta=cfg.get("adaptive_recovery_min_delta", 1e-4),
            )

        # Evaluation
        if cfg.get("n_evaluations") is not None:
            self._runner.n_evaluations = cfg["n_evaluations"]
        if cfg.get("eval_aggregation") is not None:
            self._runner.aggregation = cfg["eval_aggregation"]
        if cfg.get("anytime_eval_enabled"):
            self.set_anytime_eval(
                enabled=True,
                min_evals=cfg.get("anytime_min_evals", 1),
                max_evals=cfg.get("anytime_max_evals", 5),
                promotion_frac=cfg.get("anytime_promotion_frac", 0.3),
                aggregation=cfg.get("anytime_aggregation", "mean"),
            )

        # Lamarck — use the public API to avoid touching read-only properties
        lamarck_mode = cfg.get("lamarck_mode") or "hill_climbing"
        lamarck_steps = cfg.get("lamarck_steps") or 0
        lamarck_sigma = cfg.get("lamarck_sigma") or 1.0
        if lamarck_steps and lamarck_steps > 0:
            self.set_lamarck(
                n_steps=lamarck_steps,
                sigma=lamarck_sigma,
                mode=lamarck_mode,
            )

        # Complexity penalty
        np_val = cfg.get("complexity_penalty_nodes", 0.0) or 0.0
        cp_val = cfg.get("complexity_penalty_connections", 0.0) or 0.0
        if np_val or cp_val:
            self.set_complexity_penalty(np_val, cp_val)

        # Elitism
        ec = cfg.get("elite_count")
        sec = cfg.get("species_elite_count")
        if ec is not None or sec is not None:
            self.set_elitism(
                ec if ec is not None else self._elite_count,
                sec if sec is not None else self._species_elite_count,
            )

    # ── Experiment tracking ───────────────────────────────────────────────────

    def set_run_database(self, path: str | Path) -> None:
        """Activate run tracking: save every training run to a SQLite database.

        After calling this, each ``train()`` invocation will automatically
        record the run (config, seed, fitness history, stop reason) to
        *path*.  Call once before the first ``train()``; subsequent calls
        re-open the same (or a different) file.

        Pass ``None`` to disable tracking again.
        """
        if path is None:
            self._run_database = None
            return
        from yane.util.run_database import RunDatabase
        self._run_database = RunDatabase(path)

    def get_run_database(self):
        """Return the active RunDatabase, or None if tracking is not enabled."""
        return self._run_database

    def experiment(self, name: str, tags: list[str] | None = None) -> None:
        """Set the active experiment name for run grouping.

        All subsequent ``train()`` calls will be recorded under this
        experiment.  A ``set_run_database()`` call must precede this.

        Args:
            name: Human-readable experiment name (e.g. ``"XOR-ablation"``).
            tags: Optional list of string tags for filtering.
        """
        if self._run_database is None:
            raise RuntimeError(
                "Call set_run_database(path) before experiment()."
            )
        exp = self._run_database.get_or_create_experiment(name, tags)
        self._active_experiment_id = exp.experiment_id

    def get_active_run_id(self) -> str | None:
        """Return the run_id of the currently active (or most recent) training run."""
        return self._active_run_id

    # ── Selection strategy ────────────────────────────────────────────────────

    def set_selection_strategy(
        self,
        strategy,
        *,
        species_id: int | None = None,
    ) -> None:
        """Set the parent selection strategy.

        Args:
            strategy: Any object implementing the ``SelectionStrategy`` protocol
                (``pick(candidates, score) -> Genome``).  Pass ``None`` to
                reset to the default ``TournamentSelection(k=3)``.
            species_id: When given, the override applies only to the species
                with this ``species_id`` (see ``Species.species_id``).
                When ``None`` (default), the strategy applies globally.

        Built-in strategies (importable from ``yane.evolution.selection_strategy``)::

            TournamentSelection(k=3)   # default
            ElitistSelection(top_frac=0.2)
            FitnessProportional()
            RankSelection()
            NoveltyOnlySelection()
        """
        from yane.evolution.selection_strategy import TournamentSelection as _T
        if species_id is not None:
            self._selection_strategies_by_species[species_id] = strategy
            if self._population is not None:
                self._population.selection_strategies_by_species[species_id] = strategy
        else:
            self._selection_strategy = strategy if strategy is not None else _T(k=3)
            if self._population is not None:
                self._population.selection_strategy = self._selection_strategy

    def set_compatibility_distance(self, metric) -> None:
        """Set the genome-pair compatibility distance used for speciation.

        Args:
            metric: Any callable ``(g1, g2) -> float`` implementing the
                ``DistanceMetric`` protocol.  Pass ``None`` to reset to the
                default ``TopologyDistance()``.

        Built-in metrics (importable from ``yane.evolution.compatibility``)::

            TopologyDistance(only_enabled=False)  # default — standard NEAT
            WeightDistance()                      # avg |Δweight| of shared genes
            ActivationDistance()                  # fraction of differing activations
            ChainMetric([m1, m2], weights=[...])  # weighted sum

        Example — combine topology and weight signals::

            from yane.evolution.compatibility import TopologyDistance, WeightDistance, ChainMetric
            ne.set_compatibility_distance(
                ChainMetric([TopologyDistance(), WeightDistance()], weights=[0.7, 0.3])
            )
        """
        from yane.evolution.compatibility import TopologyDistance as _TD
        self._compatibility_distance = metric if metric is not None else _TD()
        if self._population is not None:
            self._population.compatibility_distance = self._compatibility_distance

    def set_input_transform(self, fn, n_raw_inputs: int | None = None) -> None:
        """Set a preprocessing transform applied to every ``genome.forward()`` call.

        The transform receives the raw network inputs and returns the
        (possibly different-length) vector that is actually passed to the
        network.  This is useful for feature extraction, normalisation, or
        dimensionality reduction without modifying the fitness function.

        Args:
            fn:            Callable ``list[float] -> list[float]``.  Pass
                           ``None`` to remove a previously set transform.
            n_raw_inputs:  Expected length of the raw input vector *before*
                           the transform.  When provided and ``configure()``
                           has already been called, the transform output
                           length is validated against ``n_inputs``.

        Example — normalise raw gym observations::

            ne.set_input_transform(lambda obs: [x / 4.8 for x in obs])
            ne.configure(n_inputs=4, n_outputs=2)
        """
        self._input_transform = fn
        self._n_raw_inputs = n_raw_inputs
        if fn is not None and n_raw_inputs is not None and self._population is not None:
            try:
                dummy_out = fn([0.0] * n_raw_inputs)
                if len(dummy_out) != self._n_inputs:
                    raise ValueError(
                        f"set_input_transform: transform output length {len(dummy_out)}"
                        f" != n_inputs {self._n_inputs} (n_raw_inputs={n_raw_inputs})"
                    )
            except (TypeError, AttributeError):
                pass

    # ── Genome export ─────────────────────────────────────────────────────────

    def export_genome_python(self, path: str | None = None) -> str:
        """Export the best genome as a standalone Python function.

        Returns the source string and optionally writes it to *path*.
        The generated function requires only ``import math``.
        """
        from yane.evolution.genome_export import genome_to_python
        self._ensure_configured()
        src = genome_to_python(self._population.get_best())
        if path is not None:
            Path(path).write_text(src)
        return src

    def export_genome_onnx(
        self,
        path: "str | Path | None" = None,
        opset_version: int = 17,
        unroll_steps: int = 1,
    ) -> "onnx.ModelProto":
        """Export the best genome as an ONNX model.

        Requires the ``onnx`` package (``pip install onnx``).

        Parameters
        ----------
        path :
            Optional file path to save the model.
        opset_version :
            ONNX opset version (default: 17).
        unroll_steps :
            Recurrent unroll depth for cyclic genomes.

        Returns
        -------
        onnx.ModelProto
        """
        from yane.evolution.onnx_export import genome_to_onnx
        self._ensure_configured()
        return genome_to_onnx(
            self._population.get_best(),
            path=path,
            opset_version=opset_version,
            unroll_steps=unroll_steps,
        )

    def export_genome_wasm(
        self,
        path: "str | Path | None" = None,
        title: str = "YANE Network",
        mode: str = "js",
        unroll_steps: int = 1,
    ) -> str:
        """Export the best genome as a standalone HTML/JS file.

        The generated file runs the network forward pass entirely in the
        browser via pure JavaScript — no Emscripten, ONNX, or YANE
        installation needed.

        Parameters
        ----------
        path :
            Optional file path to save the ``.html`` file.
        title :
            Page title shown in the browser.
        mode :
            ``"js"`` (default) for pure-JS transpilation.
            ``"wasm"`` raises ``ImportError`` (requires Emscripten).
        unroll_steps :
            Recurrent unroll depth for cyclic genomes.

        Returns
        -------
        str
            The full HTML source string.
        """
        from yane.evolution.wasm_export import genome_to_html
        self._ensure_configured()
        return genome_to_html(
            self._population.get_best(),
            path=path,
            title=title,
            mode=mode,
            unroll_steps=unroll_steps,
        )

    def export_genome_c_array(
        self,
        path: "str | Path" = ".",
        prefix: str = "yane_net",
    ) -> tuple:
        """Export the best genome as C99 source files for embedded deployment.

        Generates ``{path}/{prefix}.h`` and ``{path}/{prefix}.cc``.
        The code depends only on ``<math.h>`` and is fully C99-compatible.

        Parameters
        ----------
        path :
            Directory to write files into (created if absent).
        prefix :
            Name prefix for the generated files and C identifiers.

        Returns
        -------
        (header_path, source_path):
            Absolute :class:`pathlib.Path` objects for the generated files.
        """
        from yane.evolution.tflite_export import genome_to_c_array
        self._ensure_configured()
        return genome_to_c_array(self._population.get_best(), path=path, prefix=prefix)

    def find_lottery_ticket(
        self,
        fitness_fn,
        target_sparsity: float = 0.5,
        max_fitness_drop: float = 0.05,
        iterations: int = 5,
        lamarck_steps: int = 0,
        lamarck_sigma: float = 0.1,
    ):
        """Find the sparse lottery ticket for the best genome via IMP.

        Uses Iterative Magnitude Pruning on the current best genome.
        The genome is **not** modified in place; call
        :func:`~yane.evolution.sparse_neat.apply_ticket` (or
        ``genome.apply_ticket(ticket)``) to apply the result.

        Parameters
        ----------
        fitness_fn:
            ``(genome) -> float`` fitness function used for evaluation.
        target_sparsity:
            Target fraction of connections to prune (0–1).
        max_fitness_drop:
            Maximum allowed absolute fitness drop from original fitness.
        iterations:
            Number of IMP rounds.
        lamarck_steps:
            Hill-climbing fine-tuning steps per round (0 = disabled).
        lamarck_sigma:
            Step size for Lamarckian refinement.

        Returns
        -------
        LotteryTicket
        """
        from yane.evolution.sparse_neat import find_lottery_ticket as _flt
        self._ensure_configured()
        return _flt(
            self._population.get_best(),
            fitness_fn,
            target_sparsity=target_sparsity,
            max_fitness_drop=max_fitness_drop,
            iterations=iterations,
            lamarck_steps=lamarck_steps,
            lamarck_sigma=lamarck_sigma,
        )

    def export_genome_weights(self) -> dict:
        """Return the best genome's weight matrix and bias vector.

        Returns a dict with keys ``"W"`` (N×N list), ``"b"`` (N list),
        ``"labels"``, ``"n_inputs"``, ``"n_outputs"``.
        """
        from yane.evolution.genome_export import genome_to_numpy_weights
        self._ensure_configured()
        return genome_to_numpy_weights(self._population.get_best())

    # -------------------------------------------------------------------------
    # Probabilistic / Bayesian NEAT
    # -------------------------------------------------------------------------

    def set_probabilistic(
        self,
        enabled: bool = True,
        noise_std: float = 0.05,
        inference_mode: bool = False,
    ) -> None:
        """Enable or disable probabilistic output noise on all genomes.

        When enabled, each ``genome.forward()`` call adds per-output
        Gaussian noise, enabling uncertainty estimation via
        ``genome.bayesian_forward(n=100)``.

        Parameters
        ----------
        enabled:
            Whether probabilistic noise is active.
        noise_std:
            Standard deviation of the per-output Gaussian noise.
        inference_mode:
            When True, forward() is deterministic (no noise).
        """
        from yane.evolution.bayesian_neat import set_probabilistic as _sp
        self._prob_enabled = enabled
        self._prob_noise_std = float(noise_std)
        self._prob_inference_mode = inference_mode
        if self._population is not None:
            all_genomes = (
                list(self._population._evaluated)
                + list(self._population._unevaluated)
            )
            for genome in all_genomes:
                _sp(genome, enabled=enabled, noise_std=noise_std,
                    inference_mode=inference_mode)

    def set_continual_learning(
        self,
        mode: str = "ewc",
        lambda_ewc: float = 0.1,
        replay_weight: float = 0.5,
        replay_buffer_size: int = 500,
        n_progressive_nodes: int = 2,
        progressive: bool = False,
    ) -> "ContinualLearner":
        """Enable Continual / Lifelong Learning NEAT.

        Wraps fitness functions with regularization to prevent catastrophic
        forgetting across multiple sequential tasks.

        Parameters
        ----------
        mode :
            ``"ewc"`` — Elastic Weight Consolidation (weight-change penalty).
            ``"progressive"`` — freeze old weights, expand network for each task.
            ``"replay"`` — memory-replay of old-task examples.
            ``"hybrid"`` — EWC + replay combined.
        lambda_ewc :
            EWC regularization strength (higher = stronger protection).
        replay_weight :
            Memory-replay MSE weighting.
        replay_buffer_size :
            Max stored examples per task.
        n_progressive_nodes :
            New hidden nodes added per task (progressive mode).
        progressive :
            Shortcut: if True, forces ``mode="progressive"``.

        Returns
        -------
        ContinualLearner
        """
        from yane.evolution.continual import ContinualLearner
        if progressive:
            mode = "progressive"
        self._continual_learner = ContinualLearner(
            mode=mode,
            lambda_ewc=lambda_ewc,
            replay_weight=replay_weight,
            replay_buffer_size=replay_buffer_size,
            n_progressive_nodes=n_progressive_nodes,
        )
        return self._continual_learner

    def task_start(self, name: str) -> None:
        """Mark the start of a new continual-learning task.

        Call before :meth:`train` for each new task.  Requires
        :meth:`set_continual_learning` to have been called first.
        """
        if self._continual_learner is None:
            raise RuntimeError(
                "Call set_continual_learning() first."
            )
        self._continual_learner.start_task(name)
        self._task_evaluators_pending_name = name

    def task_finish(
        self,
        evaluator: "Callable | None" = None,
        sample_inputs: "list[list[float]] | None" = None,
    ) -> None:
        """Mark the end of a continual-learning task.

        Call **after** :meth:`train`.  Anchors the current best genome for
        future EWC regularization and optionally builds a memory buffer.

        Parameters
        ----------
        evaluator :
            If provided, registered for :meth:`evaluate_all_tasks`.
        sample_inputs :
            Input vectors to store as examples for memory-replay.
        """
        if self._continual_learner is None:
            raise RuntimeError("Call set_continual_learning() first.")
        self._ensure_configured()
        best = self._population.get_best()
        if best is None:
            return
        self._continual_learner.finish_task(
            best_genome=best,
            best_fitness=best.fitness,
            sample_inputs=sample_inputs,
        )
        name = getattr(self, "_task_evaluators_pending_name", "task")
        if evaluator is not None:
            self._task_evaluators.append((name, evaluator))

    def evaluate_all_tasks(self) -> dict[str, float]:
        """Evaluate the current best genome on all registered task evaluators.

        Returns a dict ``{task_name: fitness}`` for each registered task.
        Register evaluators via :meth:`task_finish`.
        """
        self._ensure_configured()
        best = self._population.get_best()
        if best is None:
            return {}
        results: dict[str, float] = {}
        for name, evaluator in self._task_evaluators:
            try:
                results[name] = float(evaluator(best))
            except Exception:
                results[name] = float("nan")
        return results

    def set_cooperative_population(
        self,
        n_agents: int = 3,
        credit: str = "shared",
        role_specialization: bool = False,
        diversity_weight: float = 0.1,
    ) -> "CooperativeSystem":
        """Configure multi-agent cooperative evolution.

        Use :meth:`train_cooperative` to run the cooperative training loop.

        Parameters
        ----------
        n_agents :
            Number of cooperative agents.
        credit :
            Credit-assignment mode: ``"shared"``, ``"difference"``,
            ``"individual"``, or ``"hierarchical"``.
        role_specialization :
            When True, apply a diversity penalty for similar agents.
        diversity_weight :
            Diversity penalty weight (only used when role_specialization=True).

        Returns
        -------
        CooperativeSystem
        """
        from yane.evolution.cooperative import CooperativeSystem
        self._cooperative_system = CooperativeSystem(
            n_agents=n_agents,
            credit=credit,
            role_specialization=role_specialization,
            diversity_weight=diversity_weight,
        )
        return self._cooperative_system

    def train_cooperative(
        self,
        team_fitness_fn: "Callable",
        n_generations: int = 100,
        n_survivors: int | None = None,
        pop_size: int | None = None,
        probe_inputs: "list[list[float]] | None" = None,
    ) -> "CooperativeResult":
        """Run cooperative co-evolution.

        Requires :meth:`set_cooperative_population` to be called first.
        Creates ``n_agents`` sub-populations and evaluates them as teams.

        Parameters
        ----------
        team_fitness_fn :
            ``(agents: list[Genome]) -> float`` — team-level fitness.
        n_generations :
            Training iterations.
        n_survivors :
            Elites kept per generation.
        pop_size :
            Total agents.  Defaults to ``self._population_size``.
        probe_inputs :
            Inputs for role-similarity measurement.

        Returns
        -------
        CooperativeResult
        """
        from yane.evolution.cooperative import train_cooperative as _train_coop, CooperativeResult
        if not hasattr(self, "_cooperative_system") or self._cooperative_system is None:
            raise RuntimeError("Call set_cooperative_population() first.")
        self._ensure_configured()
        system = self._cooperative_system
        n = pop_size or self._population_size
        agents = [self.next_genome().copy() for _ in range(n)]

        def _mutate(g):
            child = g.copy()
            try:
                child.mutate(self._tracker)
            except Exception:
                pass
            return child

        return _train_coop(
            agents=agents,
            team_fitness_fn=team_fitness_fn,
            mutation_fn=_mutate,
            n_generations=n_generations,
            n_survivors=n_survivors,
            credit=system.credit,
            role_specialization=system.role_specialization,
            diversity_weight=system.diversity_weight,
            probe_inputs=probe_inputs,
            seed=self._seed,
        )

    # -------------------------------------------------------------------------
    # Safety-Constrained Evolution (Safe NEAT)
    # -------------------------------------------------------------------------

    def set_safety_constraints(
        self,
        constraints: "list | None",
        min_safe_frac: float = 0.0,
    ) -> "SafetySystem | None":
        """Configure safety constraints for the evolutionary process.

        When set, the fitness function is wrapped to apply constraints to every
        evaluated genome.  Hard-constraint violations result in ``penalty``
        fitness immediately.  Soft and barrier constraints reduce fitness
        proportionally.

        Parameters
        ----------
        constraints:
            List of :class:`~yane.evolution.safety.SafetyConstraint` objects,
            or ``None`` to disable.
        min_safe_frac:
            Minimum fraction of the population that must be "safe" (no hard
            violations).  When fewer are safe, their fitness is boosted
            slightly to prevent them from being eliminated.

        Returns
        -------
        SafetySystem | None
        """
        if constraints is None:
            self._safety_system = None
            return None
        from yane.evolution.safety import SafetySystem
        self._safety_system = SafetySystem(
            constraints=constraints,
            min_safe_frac=min_safe_frac,
        )
        return self._safety_system

    def _apply_safety_constraints(
        self,
        genome: "Genome",
        raw_fitness: float,
    ) -> float:
        """Apply safety constraints to raw_fitness.  Internal use."""
        system = getattr(self, "_safety_system", None)
        if system is None:
            return raw_fitness
        return system.evaluate(genome, raw_fitness)

    def set_minimal_criterion(
        self,
        criterion_fn: "Callable | None" = None,
        min_viable_frac: float = 0.1,
        penalty: float = -1e6,
        viable_boost_factor: float = 0.5,
    ) -> "MinimalCriterion | None":
        """Filter genomes by a viability criterion before selection.

        Only genomes passing *criterion_fn* are eligible for reproduction.
        When too few genomes are viable (< *min_viable_frac*), the penalty
        is adaptively relaxed to keep the search alive.

        Pass ``None`` to disable the criterion.

        Parameters
        ----------
        criterion_fn :
            ``(genome) -> bool``.  Typically checks the genome's fitness
            (e.g., ``lambda g: g.fitness > -50.0``).  Pass ``None`` to
            disable.
        min_viable_frac :
            Minimum viable fraction before adaptive relaxation triggers.
        penalty :
            Fitness assigned to non-viable genomes.
        viable_boost_factor :
            Penalty multiplier when relaxation is active (< 1 = less severe).

        Returns
        -------
        MinimalCriterion | None
        """
        if criterion_fn is None:
            self._minimal_criterion = None
            return None
        from yane.evolution.minimal_criterion import MinimalCriterion
        self._minimal_criterion = MinimalCriterion(
            criterion_fn=criterion_fn,
            min_viable_frac=min_viable_frac,
            penalty=penalty,
            viable_boost_factor=viable_boost_factor,
        )
        return self._minimal_criterion

    def set_open_ended(
        self,
        mode: str = "novelty_with_criterion",
        archive_size: int = 200,
    ) -> None:
        """Enable open-ended evolution mode combining novelty/curiosity/QD with Minimal Criterion.

        Requires :meth:`set_minimal_criterion` to be called first.

        Parameters
        ----------
        mode :
            ``"novelty_with_criterion"`` — novelty + criterion filter.
            ``"curiosity_with_criterion"`` — curiosity + criterion filter.
            ``"quality_diversity_with_criterion"`` — QD + criterion filter.
        archive_size :
            Novelty archive size (passed to ``set_novelty_search`` when
            mode involves novelty).
        """
        valid = ("novelty_with_criterion", "curiosity_with_criterion",
                 "quality_diversity_with_criterion")
        if mode not in valid:
            raise ValueError(f"mode must be one of {valid}, got {mode!r}")
        self._open_ended_mode = mode
        # Enable the corresponding feature if available
        if "novelty" in mode:
            self.set_novelty_search(enabled=True)
        elif "curiosity" in mode:
            if not self._curiosity_enabled:
                self.set_curiosity(enabled=True)

    def configure_reservoir(
        self,
        n_reservoir: int = 100,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.5,
        leaking_rate: float = 0.3,
        n_inputs: int | None = None,
        n_outputs: int | None = None,
        seed: int | None = None,
    ) -> "ReservoirGenome":
        """Create an Echo State Network reservoir.

        The reservoir is **fixed** after creation (not evolved).  Only the
        readout weights W_out are updated by :meth:`train_reservoir` or via
        evolution.

        Parameters
        ----------
        n_reservoir :
            Number of reservoir neurons.
        spectral_radius :
            Spectral radius of W.  Must be < 1 for the Echo State Property.
        input_scaling :
            Scaling of W_in (input → reservoir).
        leaking_rate :
            Leaky integration rate α (0 < α ≤ 1).
        n_inputs :
            Number of inputs.  Defaults to ``self._n_inputs`` if configured.
        n_outputs :
            Number of outputs.  Defaults to ``self._n_outputs`` if configured.
        seed :
            Seed for deterministic reservoir init.

        Returns
        -------
        ReservoirGenome
        """
        from yane.evolution.reservoir import ReservoirGenome
        n_in = n_inputs or getattr(self, "_n_inputs", 0)
        n_out = n_outputs or getattr(self, "_n_outputs", 0)
        if not n_in or not n_out:
            raise RuntimeError(
                "Call configure(n_inputs, n_outputs) first, or pass "
                "n_inputs/n_outputs explicitly."
            )
        self._reservoir = ReservoirGenome(
            n_inputs=n_in,
            n_reservoir=n_reservoir,
            n_outputs=n_out,
            spectral_radius=spectral_radius,
            input_scaling=input_scaling,
            leaking_rate=leaking_rate,
            seed=seed if seed is not None else self._seed,
        )
        return self._reservoir

    def train_reservoir(
        self,
        inputs_sequence: "list[list[float]]",
        targets_sequence: "list[list[float]]",
        reservoir: "ReservoirGenome | None" = None,
        lambda_ridge: float = 1e-4,
        washout: int = 10,
    ) -> "ReservoirTrainResult":
        """Train reservoir readout weights analytically via Ridge Regression.

        Call :meth:`configure_reservoir` first (or pass *reservoir* explicitly).

        Parameters
        ----------
        inputs_sequence :
            Training inputs (one vector per timestep).
        targets_sequence :
            Corresponding target outputs.
        reservoir :
            Reservoir to train.  Defaults to the one from :meth:`configure_reservoir`.
        lambda_ridge :
            Ridge regularization strength.
        washout :
            Initial timesteps to discard (reservoir warm-up).

        Returns
        -------
        ReservoirTrainResult
        """
        from yane.evolution.reservoir import train_ridge_readout
        r = reservoir or getattr(self, "_reservoir", None)
        if r is None:
            raise RuntimeError("Call configure_reservoir() first.")
        return train_ridge_readout(r, inputs_sequence, targets_sequence,
                                   lambda_ridge=lambda_ridge, washout=washout)

    def meta_train(
        self,
        task_sampler: "Callable",
        meta_iterations: int = 500,
        adaptation_steps: int = 3,
        lamarck_sigma: float = 0.1,
    ) -> "MetaTrainResult":
        """Train a population of quickly-adaptable genomes.

        Implements a gradient-free MAML-like meta-learning loop:

        * **Inner loop** (Lamarck): for each genome evaluation, sample a new
          task and refine the genome with ``adaptation_steps`` hill-climbing
          steps.  The post-adaptation fitness is used as the NEAT fitness
          signal.
        * **Outer loop** (NEAT): evolves genomes that achieve high
          post-adaptation fitness across many task samples.

        Requires :meth:`configure` to have been called first.

        Parameters
        ----------
        task_sampler :
            Callable ``() -> fitness_fn`` that returns a fresh fitness function
            for each new task.
        meta_iterations :
            Maximum total genome evaluations (outer loop).
        adaptation_steps :
            Lamarck hill-climbing steps per inner loop (inner loop depth).
        lamarck_sigma :
            Noise scale for hill-climbing perturbations.

        Returns
        -------
        MetaTrainResult
        """
        from yane.evolution.meta_learning import MetaLearner, MetaTrainResult
        self._ensure_configured()

        learner = MetaLearner(
            adaptation_steps=adaptation_steps,
            lamarck_sigma=lamarck_sigma,
        )
        meta_fn = learner.make_fitness_fn(task_sampler)
        self.set_max_iterations(meta_iterations)
        self.train(meta_fn)

        best = self._population.get_best()
        return MetaTrainResult(
            best_genome=best.copy() if best else self.next_genome(),
            best_meta_fitness=best.fitness if best else -float("inf"),
            adaptation_deltas=learner.adaptation_deltas,
            meta_iterations=meta_iterations,
        )

    def set_adversarial_populations(
        self,
        n_populations: int = 2,
        pairing: str = "round_robin",
        n_matches: int = 10,
        elo_k: float = 32.0,
        seed: int | None = None,
    ) -> "AdversarialSystem":
        """Configure Self-Play / Adversarial Co-Evolution.

        Splits the population into *n_populations* competing groups.
        Use :meth:`train_adversarial` to run the adversarial training loop.

        Parameters
        ----------
        n_populations :
            Number of competing sub-populations (≥ 2).
        pairing :
            Matchmaking strategy: ``"round_robin"`` (all pairs),
            ``"random"`` (n_matches random pairs), or
            ``"best_vs_rest"`` (best genome from each pop plays all others).
        n_matches :
            Matches per genome per generation (``"random"`` mode only).
        elo_k :
            Elo K-factor.
        seed :
            RNG seed.

        Returns
        -------
        AdversarialSystem
            The configured system (use directly or via :meth:`train_adversarial`).
        """
        from yane.evolution.self_play import AdversarialSystem
        self._adversarial_system = AdversarialSystem(
            n_populations=n_populations,
            pairing=pairing,
            n_matches=n_matches,
            elo_k=elo_k,
            seed=seed,
        )
        return self._adversarial_system

    def train_adversarial(
        self,
        game_fn: "Callable",
        n_generations: int = 100,
        n_survivors: int | None = None,
        pop_size: int | None = None,
    ) -> "AdversarialResult":
        """Run adversarial co-evolution.

        Requires :meth:`set_adversarial_populations` to be called first and
        :meth:`configure` for network topology.  The population is evenly
        split across sub-populations.

        Parameters
        ----------
        game_fn :
            ``(genome_a, genome_b) → (score_a, score_b)`` — zero-sum scores
            (score_a + score_b = constant).
        n_generations :
            Number of adversarial generations.
        n_survivors :
            Elites kept per sub-population per generation.
        pop_size :
            Total genomes.  Defaults to ``self._population_size``.

        Returns
        -------
        AdversarialResult
        """
        from yane.evolution.self_play import AdversarialResult, train_adversarial as _train_adv
        if not hasattr(self, "_adversarial_system") or self._adversarial_system is None:
            raise RuntimeError(
                "Call set_adversarial_populations() first."
            )
        self._ensure_configured()
        system = self._adversarial_system
        n = pop_size or self._population_size
        n_pops = system.n_populations
        pop_n = max(2, n // n_pops)

        # Build N sub-populations from fresh genomes
        populations: list[list] = []
        for _ in range(n_pops):
            pops = [self._population.select_for_evaluation().copy()
                    for _ in range(pop_n)]
            populations.append(pops)

        def _mutate(g):
            child = g.copy()
            try:
                child.mutate(self._tracker)
            except Exception:
                pass
            return child

        return _train_adv(
            populations=populations,
            game_fn=game_fn,
            mutation_fn=_mutate,
            n_generations=n_generations,
            n_survivors=n_survivors,
            pairing=system.pairing,
            n_matches=system.n_matches,
            elo_k=system._elo.k_factor,
            seed=self._seed,
        )

    def set_genome_encoding(
        self,
        encoding: str,
        development_steps: int = 5,
        n_genes: int = 20,
        n_nodes: int | None = None,
        seed: int | None = None,
    ) -> "GRNCodec | None":
        """Configure an indirect genome encoding.

        Currently supports ``"grn"`` (Gene Regulatory Network).

        When ``"grn"`` is set, :meth:`configure` additionally creates an
        initial :class:`~yane.evolution.grn_encoding.GRNGenome` prototype
        that can be developed via :meth:`develop_grn`.

        Parameters
        ----------
        encoding :
            ``"grn"`` — Gene Regulatory Network indirect encoding.
        development_steps :
            GRN development rounds (controls phenotype complexity: more
            steps → more connections per gene).
        n_genes :
            Number of GRN genes for a freshly generated prototype.
        n_nodes :
            Node innovation range for random gene generation.
        seed :
            RNG seed.

        Returns
        -------
        GRNCodec | None
        """
        if encoding.lower() == "grn":
            from yane.evolution.grn_encoding import GRNCodec, GRNGenome
            n_in = getattr(self, "_n_inputs", 2)
            n_out = getattr(self, "_n_outputs", 1)
            codec = GRNCodec(n_inputs=n_in, n_outputs=n_out,
                             development_steps=development_steps)
            self._grn_codec = codec
            self._grn_prototype = GRNGenome.random(
                n_genes=n_genes,
                n_nodes=n_nodes or (n_in + n_out + n_genes // 2),
                seed=seed or self._seed,
            )
            return codec
        else:
            raise ValueError(f"Unknown genome encoding: {encoding!r}. Use 'grn'.")

    def develop_grn(
        self,
        grn: "GRNGenome | None" = None,
        max_connections: int | None = None,
    ) -> "Genome":
        """Develop a GRN genotype into a phenotype Genome.

        Requires :meth:`set_genome_encoding` to have been called with
        ``encoding="grn"``.

        Parameters
        ----------
        grn :
            The :class:`~yane.evolution.grn_encoding.GRNGenome` to develop.
            Defaults to the internal prototype.
        max_connections :
            Optional cap on the phenotype's connections.

        Returns
        -------
        Genome
        """
        if not hasattr(self, "_grn_codec") or self._grn_codec is None:
            raise RuntimeError(
                "Call set_genome_encoding('grn') first."
            )
        target = grn or getattr(self, "_grn_prototype", None)
        if target is None:
            raise RuntimeError("No GRN genome provided.")
        return self._grn_codec.develop(target, max_connections=max_connections)

    def configure_hierarchical(
        self,
        n_workers: int = 4,
        selection_mode: str = "hard",
    ) -> "HierarchicalGenome":
        """Build a :class:`~yane.evolution.h_neat.HierarchicalGenome`.

        Creates a manager genome (``n_outputs = n_workers``) and ``n_workers``
        worker genomes sharing the same topology as ``configure()`` was called
        with.  The hierarchy can then be used as a fitness function or evolved
        via repeated mutation/evaluation cycles.

        Call **after** :meth:`configure`.

        Parameters
        ----------
        n_workers :
            Number of sub-policies in the pool.
        selection_mode :
            ``"hard"`` or ``"soft"`` — see
            :class:`~yane.evolution.h_neat.HierarchicalGenome`.

        Returns
        -------
        HierarchicalGenome
        """
        from yane.evolution.h_neat import HierarchicalGenome
        self._ensure_configured()

        n_in = self._n_inputs
        n_out = self._n_outputs
        max_n = self._max_nodes
        max_c = self._max_connections

        # Manager: same n_inputs as workers, n_outputs = n_workers
        mgr_ne = self.__class__(seed=self._seed)
        mgr_ne.configure(n_inputs=n_in, n_outputs=n_workers,
                         max_nodes=max_n, max_connections=max_c)
        manager = mgr_ne.get_best() if mgr_ne.population._evaluated else mgr_ne.next_genome()

        # Workers: same topology as the current configuration
        workers = [self.next_genome().copy() for _ in range(n_workers)]

        return HierarchicalGenome(manager, workers, selection_mode=selection_mode)

    def distill_ensemble(
        self,
        k: int = 5,
        target_nodes: int = 10,
        distillation_steps: int = 500,
        probe_inputs: "list[list[float]] | None" = None,
        n_probes: int = 100,
        sigma: float = 0.1,
        sigma_decay: float = 0.99,
        seed: int | None = None,
    ) -> "DistillationResult":
        """Compress the top-K trained genomes into a compact student genome.

        Distillation runs ``distillation_steps`` hill-climbing steps that
        minimise the MSE between the student's outputs and the ensemble's
        mean outputs on a set of probe inputs.

        Parameters
        ----------
        k :
            Number of top genomes that form the teacher ensemble.
        target_nodes :
            Maximum total node count for the student genome (inputs + hidden
            + outputs).  Lower values → more compression.
        distillation_steps :
            Total hill-climbing steps.  More steps → lower final MSE.
        probe_inputs :
            Fixed probe inputs to supervise distillation.  When *None*,
            ``n_probes`` random inputs in ``[0, 1]^n_inputs`` are generated.
        n_probes :
            Number of random probes when *probe_inputs* is *None*.
        sigma :
            Initial perturbation noise for hill-climbing.
        sigma_decay :
            Multiplicative sigma decay per mini-epoch (0.99 = slow annealing).
        seed :
            RNG seed for reproducibility.

        Returns
        -------
        DistillationResult
            Contains the student genome, final/initial MSE, loss history,
            and compression metrics.

        Example
        -------
        ::

            yane.train(fitness_fn)
            result = yane.distill_ensemble(k=5, target_nodes=8,
                                           distillation_steps=300)
            print(f"Compression ratio: {result.compression_ratio:.2f}×")
        """
        from yane.evolution.distillation import (
            distill_ensemble as _distill,
            _make_student,
            DistillationResult,
        )
        self._ensure_configured()
        teachers = self._population.get_top(k)
        if not teachers:
            raise RuntimeError("Population has no evaluated genomes.  Run train() first.")

        n_inputs = len(teachers[0].input_nodes)
        n_outputs = len(teachers[0].output_nodes)
        rng_seed = seed if seed is not None else (self._seed or 0)
        import random as _rnd
        rng = _rnd.Random(rng_seed)
        student = _make_student(n_inputs, n_outputs, target_nodes, rng)

        return _distill(
            teachers=teachers,
            student=student,
            probe_inputs=probe_inputs,
            distillation_steps=distillation_steps,
            n_probes=n_probes,
            sigma=sigma,
            sigma_decay=sigma_decay,
            seed=rng_seed,
        )

    def export_best_weights_npy(self, path: str | Path) -> None:
        """Save the best genome's weight matrix as a NumPy ``.npy`` file.

        The matrix has shape ``(N, N)`` where N = total nodes.  Row = source,
        column = target.  Also saves a sidecar ``.csv`` with node labels.
        """
        import numpy as np
        weights = self.export_genome_weights()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), np.array(weights["W"], dtype=np.float64))
        # Sidecar CSV with labels
        csv_path = path.with_suffix(".csv")
        csv_path.write_text(
            "index,label\n"
            + "\n".join(f"{i},{lbl}" for i, lbl in enumerate(weights["labels"]))
        )

    def load_genome_as_seed(
        self,
        genome: Genome,
        freeze_layers: list[str] | None = None,
    ) -> None:
        """Use a pre-trained genome as the seed for a new population.

        The genome is adapted to the current I/O dimensions (like
        ``warm_start_from_checkpoint``) and inserted as the initial genome.
        Optionally, connection groups can be frozen (not mutated).

        Args:
            genome: Source genome to use as seed.
            freeze_layers: List of layer types to freeze.
                Supported: ``"input"``, ``"output"``, ``"hidden"``.
        """
        self._ensure_configured()
        genome = genome.copy()
        self._adapt_genome_topology(genome, self._n_inputs, self._n_outputs,
                                    self._tracker, self._stateful)
        genome.max_nodes = self._max_nodes
        genome.max_connections = self._max_connections
        genome.allow_memory = self._stateful
        genome._output_sanitize = self._output_sanitize
        genome._output_fallback = self._output_fallback
        genome._last_species_id = None
        genome._species_stale = True

        self._transfer_freeze_records = {}
        self._transfer_frozen_layers = []
        self._transfer_unfreeze_progress = 0.0
        if freeze_layers:
            self._freeze_genome_layers(genome, freeze_layers)

        self._population._template = genome.copy()
        self._population._unevaluated = [genome]
        self._population._evaluated = []
        self._population._species = []
        self._population._generation = 0
        self._population._stagnation_count = 0

    def fine_tune_genome(
        self,
        genome: Genome,
        fitness_fn: Callable[[Genome], float],
        *,
        n_steps: int = 20,
        load_as_seed: bool = False,
        freeze_layers: list[str] | None = None,
    ) -> Genome:
        """Fine-tune a transferred genome with Lamarck only.

        The topology is adapted to the current task and then kept unchanged;
        only Lamarckian weight refinement is applied. When ``load_as_seed`` is
        true, the refined genome is inserted as the current population seed.
        """
        self._ensure_configured()
        tuned = genome.copy()
        self._adapt_genome_topology(tuned, self._n_inputs, self._n_outputs,
                                    self._tracker, self._stateful)
        tuned.max_nodes = self._max_nodes
        tuned.max_connections = self._max_connections
        tuned.allow_memory = self._stateful
        tuned._output_sanitize = self._output_sanitize
        tuned._output_fallback = self._output_fallback
        baseline = self._finalize_fitness(fitness_fn(tuned), None, tuned)
        tuned.fitness = baseline
        tuned.raw_fitness = baseline
        refined = self._lamarck.refine(tuned, fitness_fn, baseline_fitness=baseline, n_steps=n_steps)
        tuned.fitness = self._finalize_fitness(refined, None, tuned)
        # _finalize_fitness already set tuned.raw_fitness via finalize_fitness_value.
        # Only override if curiosity is not active (curiosity stores task-only score
        # in _curiosity_task_base so finalize_fitness_value uses that as raw_fitness).
        if not hasattr(tuned, '_curiosity_task_base'):
            tuned.raw_fitness = tuned.fitness
        if load_as_seed:
            self.load_genome_as_seed(tuned, freeze_layers=freeze_layers)
        return tuned

    def behaviour_clone(
        self,
        demonstrations: "list[tuple[list[float], list[float]]]",
        n_steps: int = 200,
        sigma: float = 0.05,
        *,
        seed_population: bool = False,
        noise_sigma: float = 0.05,
        freeze_layers: "list[str] | None" = None,
    ) -> "BehaviourCloneResult":
        """Supervised pre-training of the best genome on expert demonstrations.

        Uses Lamarckian hill-climbing to minimise MSE between genome outputs
        and demonstration targets.  Returns a :class:`BehaviourCloneResult`
        with the cloned genome and metrics; call
        ``result.seed_population()`` to replace the evolution population.

        Parameters
        ----------
        demonstrations:
            List of ``(inputs, targets)`` pairs.
        n_steps:
            Total hill-climbing evaluation steps.
        sigma:
            Per-step weight perturbation sigma.
        seed_population:
            If ``True``, immediately seed the population with noisy copies
            of the cloned genome.
        noise_sigma:
            Weight noise added to each population copy (diversity).
        freeze_layers:
            Connection groups to freeze when seeding.

        Returns
        -------
        BehaviourCloneResult

        Note
        ----
        PyTorch/backprop-based cloning and the LunarLander benchmark are
        deferred (require optional torch dependency).
        """
        from yane.evolution.behaviour_cloning import behaviour_clone as _bc
        return _bc(
            self,
            demonstrations,
            n_steps=n_steps,
            sigma=sigma,
            seed_population=seed_population,
            noise_sigma=noise_sigma,
            freeze_layers=freeze_layers,
        )

    # -------------------------------------------------------------------------
    # POET — Paired Open-Ended Trailblazer (Co-Evolution)
    # -------------------------------------------------------------------------

    def train_poet(
        self,
        eval_fn: "Callable",
        initial_env_params: "list[float]",
        n_generations: int = 100,
        archive_size: int = 10,
        env_bounds: "tuple[float, float] | None" = None,
        env_mutation_sigma: float = 0.1,
        lower_bound: float = -float("inf"),
        upper_bound: "float | None" = None,
        transfer_interval: int = 5,
        transfer_k: int = 5,
        env_children_per_gen: int = 2,
        seed: "int | None" = None,
    ) -> "POETResult":
        """Run POET co-evolution of tasks and agents.

        Uses the current best genome as the initial agent.  The agent is
        mutated within each POET pair using the configured Lamarck refiner
        (hill-climbing on the pair's fitness function).

        Parameters
        ----------
        eval_fn:
            ``(agent_genome, env_genome) -> float`` — evaluates the agent.
        initial_env_params:
            Parameter vector for the initial environment.
        n_generations:
            Number of POET step() iterations.
        archive_size:
            Maximum active pairs.
        env_bounds:
            ``(min, max)`` clipping bounds for environment parameters.
        env_mutation_sigma:
            Sigma for environment genome mutation.
        lower_bound:
            Minimum fitness for environment viability (not too hard).
        upper_bound:
            Maximum fitness for viability (``None`` = no upper bound).
        transfer_interval:
            Generations between agent transfer and environment reproduction.
        transfer_k:
            Agents tested per target pair during transfer.
        env_children_per_gen:
            Child environments produced per pair per generation.
        seed:
            Random seed.

        Returns
        -------
        POETResult
        """
        from yane.evolution.poet import (
            EnvironmentCriterion,
            POETResult,
            train_poet as _train_poet,
        )
        self._ensure_configured()
        pop = self._population
        agent = pop.get_best().copy() if pop._evaluated else pop._unevaluated[0].copy()

        criterion = EnvironmentCriterion(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        def _mutate_agent(genome: "Genome") -> "Genome":
            child = genome.copy()
            child.mutate(self._tracker)
            return child

        return _train_poet(
            eval_fn=eval_fn,
            mutate_agent_fn=_mutate_agent,
            initial_env_params=initial_env_params,
            initial_agent=agent,
            n_generations=n_generations,
            archive_size=archive_size,
            env_bounds=env_bounds,
            env_mutation_sigma=env_mutation_sigma,
            criterion=criterion,
            transfer_interval=transfer_interval,
            transfer_k=transfer_k,
            env_children_per_gen=env_children_per_gen,
            seed=seed,
        )

    # -------------------------------------------------------------------------
    # Genome Phylogeny (Stammbaum der Innovationen)
    # -------------------------------------------------------------------------

    def enable_phylogeny(self, max_size: "int | None" = None) -> "PhylogenyTree":
        """Enable genome phylogeny tracking during training.

        When enabled, every evaluated genome is recorded in a
        :class:`~yane.evolution.phylogeny.PhylogenyTree` with its fitness,
        parent, and innovation attribution.  Tracking is disabled by default
        (zero runtime cost).

        Parameters
        ----------
        max_size:
            Maximum number of nodes to retain (oldest roots pruned when
            exceeded).  ``None`` = unlimited.

        Returns
        -------
        PhylogenyTree
            The active tree (empty until :meth:`train` is called).
        """
        from yane.evolution.phylogeny import PhylogenyTree
        if self._phylogeny is None:
            self._phylogeny = PhylogenyTree(max_size=max_size)
        self._phylogeny.enable()
        return self._phylogeny

    def disable_phylogeny(self) -> None:
        """Disable phylogeny recording (existing data retained)."""
        if self._phylogeny is not None:
            self._phylogeny.disable()

    def get_phylogeny(self) -> "PhylogenyTree | None":
        """Return the phylogeny tree, or ``None`` if not enabled."""
        return self._phylogeny

    # -------------------------------------------------------------------------
    # Unified Parameter Registry — Phase 1 of P0 Meta-Adaptive Orchestration
    # -------------------------------------------------------------------------

    def _ensure_param_registry(self) -> ParamRegistry:
        """Return the registry, creating and populating it on first call."""
        if self._param_registry is None:
            self._param_registry = build_default_registry(self)
        return self._param_registry

    def set_param(self, name: str, value: object) -> None:
        """Set a YANE hyperparameter by name.

        Validates the value against the parameter's domain, dispatches to the
        appropriate ``set_*()`` method, and records the change in the registry.

        This is the universal setter used by the MetaOptimizer.  Calling it is
        equivalent to calling the specific ``set_*()`` method directly — it
        never bypasses or overrides the underlying setter.

        Parameters
        ----------
        name:   Dot-namespaced identifier, e.g. ``"lamarck.mode"``.
        value:  New value; must satisfy the parameter's type and domain.

        Raises
        ------
        KeyError:   Unknown parameter name.
        ValueError: Value outside the allowed domain.
        TypeError:  Value has the wrong Python type.

        Examples
        --------
        >>> yane.set_param("lamarck.mode", "cma_es")
        >>> yane.set_param("pop.size", 200)
        >>> yane.set_param("anytime.enabled", True)
        """
        self._ensure_param_registry().dispatch(self, name, value)

    def get_param_space(self) -> dict:
        """Return the full parameter catalog as a plain, serialisable dict.

        Each entry contains: ``type``, ``domain``, ``default``, ``current``,
        ``stage``, ``subsystem``, ``description``, ``impact_history``.

        Returns
        -------
        dict
            Mapping parameter name → parameter info dict, sorted by name.
        """
        return self._ensure_param_registry().get_param_space()

    def get_param_registry(self) -> ParamRegistry:
        """Return the live ``ParamRegistry`` instance.

        Useful for recording fitness impacts, listing parameter names, or
        reading the change log from test code and the MetaOptimizer.
        """
        return self._ensure_param_registry()

    # -------------------------------------------------------------------------
    # Problem Profiler — Phase 2 of P0 Meta-Adaptive Orchestration
    # -------------------------------------------------------------------------

    def profile_problem(
        self,
        evaluator: "Callable",
        n_warmup: int = 50,
    ) -> "ProblemProfile":
        """Profile the fitness landscape before training.

        Runs ``n_warmup`` random genomes through the evaluator (each evaluated
        twice) and returns a ``ProblemProfile`` containing task type, difficulty,
        noise level, and other landscape properties.

        Requires ``configure()`` to have been called first.

        Parameters
        ----------
        evaluator:  Fitness function ``(genome) → float``.
        n_warmup:   Number of random genomes (minimum 5).

        Returns
        -------
        ProblemProfile

        Example
        -------
        >>> yane.configure(n_inputs=4, n_outputs=2)
        >>> profile = yane.profile_problem(my_evaluator, n_warmup=30)
        >>> print(profile.task_type, profile.estimated_difficulty)
        """
        self._ensure_configured()
        from yane.evolution.problem_profiler import ProblemProfile, ProblemProfiler  # noqa: F401
        profile = ProblemProfiler(self).profile(evaluator, n_warmup=n_warmup)
        self._last_problem_profile = profile
        return profile

    # -------------------------------------------------------------------------
    # Knowledge Base — Phase 4 of P0 Meta-Adaptive Orchestration
    # -------------------------------------------------------------------------

    def set_knowledge_base(self, path: "str | Path | None" = None) -> None:
        """Enable the cross-run Knowledge Base.

        Parameters
        ----------
        path: JSON file for persistence.  If the file already exists, its
              entries are loaded immediately.  Pass ``None`` to create an
              in-memory-only KB (not persisted across processes).

        After this call, ``train()`` automatically calls ``kb.learn()`` with
        the run's best fitness and current param-space snapshot.
        """
        from yane.evolution.knowledge_base import KnowledgeBase
        self._knowledge_base = KnowledgeBase(path=path)

    @property
    def knowledge_base(self):
        """The active ``KnowledgeBase`` instance, or ``None`` if not set.

        Use ``set_knowledge_base(path)`` to enable.
        """
        return self._knowledge_base

    def suggest_params(
        self,
        problem_profile: "Any | None" = None,
        top_k: int = 5,
    ) -> "list[dict]":
        """Return parameter suggestions from the Knowledge Base.

        Parameters
        ----------
        problem_profile: A ``ProblemProfile`` object.  If omitted, uses the
                         last profile produced by ``profile_problem()``.
        top_k:           Maximum number of suggestions to return.

        Returns
        -------
        list[dict]  — see ``KnowledgeBase.suggest()`` for the format.

        Raises
        ------
        RuntimeError  — if no KB is configured or no profile is available.
        """
        if self._knowledge_base is None:
            raise RuntimeError(
                "No Knowledge Base configured.  Call set_knowledge_base() first."
            )
        profile = problem_profile or self._last_problem_profile
        if profile is None:
            raise RuntimeError(
                "No problem profile available.  Call profile_problem() first."
            )
        return self._knowledge_base.suggest(profile, top_k=top_k)

    # -------------------------------------------------------------------------
    # MetaOptimizer — Phase 3 of P0 Meta-Adaptive Orchestration
    # -------------------------------------------------------------------------

    def set_meta_optimizer(
        self,
        enabled: bool = True,
        *,
        tune_interval: int = 20,
        max_overhead_pct: float = 5.0,
        plateau_patience: int = 30,
        phase_min_gens: int = 20,
        ucb1_c: float = 2.0,
    ) -> None:
        """Enable the MetaOptimizer for automatic hyperparameter tuning.

        When enabled, the MetaOptimizer is called once per generation during
        ``train()``.  It tunes parameters via UCB1 bandits (categorical) and
        Bayesian Optimisation (continuous), and advances through four
        training phases (EXPLORE → EXPLOIT → REFINE → CONVERGE) based on
        fitness plateau detection.

        Parameters
        ----------
        enabled:          Enable or disable the MetaOptimizer.
        tune_interval:    Tune parameters every N generations (default 20).
        max_overhead_pct: Stop tuning when MetaOptimizer overhead exceeds
                          this fraction of total evaluation time (default 5%).
        plateau_patience: Generations without improvement before a phase
                          transition (default 30).
        ucb1_c:           UCB1 exploration constant (default 2.0).
        """
        self._meta_optimizer_enabled = bool(enabled)
        if enabled:
            from yane.evolution.meta_optimizer import MetaOptimizer
            # Pass the full ParamRegistry so the MetaOptimizer tunes ALL
            # registered parameters, not just the hardcoded subset.
            reg = self._ensure_param_registry() if self.is_configured else None
            self._meta_optimizer_obj = MetaOptimizer(
                tune_interval=tune_interval,
                max_overhead_pct=max_overhead_pct,
                plateau_patience=plateau_patience,
                phase_min_gens=phase_min_gens,
                ucb1_c=ucb1_c,
                seed=self._seed,
                param_registry=reg,
            )
        else:
            self._meta_optimizer_obj = None

    def get_meta_optimizer_diagnostics(self) -> dict:
        """Return diagnostics from the active MetaOptimizer.

        Returns an empty dict if the MetaOptimizer is not enabled.
        """
        if self._meta_optimizer_obj is None:
            return {"enabled": False}
        return self._meta_optimizer_obj.get_diagnostics()

    def _tick_meta_optimizer(
        self,
        generation: int,
        best_fitness: float,
        eval_ms: float = 0.0,
    ) -> None:
        """Internal: called every generation by train()."""
        if not self._meta_optimizer_enabled or self._meta_optimizer_obj is None:
            return
        try:
            self._meta_optimizer_obj.tick(
                generation=generation,
                best_fitness=best_fitness,
                ne=self,
                eval_ms_this_gen=eval_ms,
            )
        except Exception:
            pass  # MetaOptimizer is non-critical; never break training

    def _auto_kb_learn(
        self,
        best_fitness: float,
        stop_reason: str | None,
    ) -> None:
        """Called at the end of train() to record this run in the KB."""
        if self._knowledge_base is None:
            return
        profile = self._last_problem_profile
        if profile is None:
            return
        try:
            params = {
                name: info["current"]
                for name, info in self.get_param_space().items()
                if info["current"] is not None
            }
            self._knowledge_base.learn(
                profile=profile,
                final_params=params,
                final_fitness=best_fitness,
                run_id=self._active_run_id,
            )
        except Exception:
            pass  # KB is non-critical; never break training

    def set_auto_features(
        self,
        enabled: bool = True,
        max_concurrent: int = 3,
        test_interval: int = 50,
        test_duration: "int | None" = None,
        impact_threshold: float = 0.0,
        degradation_patience: int = 30,
        reactivation_delay: int = 100,
        include_heavyweight: bool = False,
    ) -> None:
        """Automatically select and gate research features via UCB1 + Successive Halving.

        Every ``test_interval`` generations an inactive feature is enabled for
        ``test_duration`` generations.  If global best-fitness improved the feature
        stays active; otherwise it is returned to the candidate pool.  Active
        features whose fitness contribution drops are gradually degraded and
        eventually disabled.

        Args:
            enabled:              Pass False to disable feature gating entirely.
            max_concurrent:       Maximum number of features simultaneously active
                                  or under test (default 3).
            test_interval:        Generations between starting new trials (default 50).
            test_duration:        Length of each trial window. Defaults to
                                  ``max(10, test_interval // 2)``.
            impact_threshold:     Minimum absolute fitness improvement for a feature
                                  to be kept active (default 0.0 = any improvement).
            degradation_patience: Consecutive generations with no improvement before
                                  degradation_level increases (default 30).
            reactivation_delay:   Generations after disabling before auto-reactivation
                                  into the candidate pool (default 100).
        """
        self._feature_gate_enabled = bool(enabled)
        if not enabled:
            self._feature_gate = None
            return
        from yane.evolution.feature_gating import FeatureGate, _register_known_features
        fg = FeatureGate(
            max_concurrent=max_concurrent,
            test_interval=test_interval,
            test_duration=test_duration,
            impact_threshold=impact_threshold,
            degradation_patience=degradation_patience,
            reactivation_delay=reactivation_delay,
        )
        _register_known_features(fg, self, include_heavyweight=include_heavyweight)
        self._feature_gate = fg

    def get_feature_gating_diagnostics(self) -> dict:
        """Return diagnostics from the active FeatureGate.

        Returns ``{"enabled": False}`` when feature gating is not configured.
        """
        if self._feature_gate is None:
            return {"enabled": False}
        return self._feature_gate.get_diagnostics()

    def _tick_feature_gating(self, generation: int, best_fitness: float) -> None:
        """Internal: called every generation by train()."""
        if not self._feature_gate_enabled or self._feature_gate is None:
            return
        try:
            self._feature_gate.tick(
                generation=generation,
                best_fitness=best_fitness,
                ne=self,
            )
        except Exception:
            pass  # Feature gating is non-critical; never break training

    def auto_train(
        self,
        evaluator,
        n_inputs: "int | None" = None,
        n_outputs: "int | None" = None,
        target_fitness: "float | None" = None,
        max_time_seconds: "float | None" = None,
        problem_name: "str | None" = None,
        n_warmup: int = 20,
    ) -> "AutoTrainResult":
        """Zero-config training: profile → suggest → configure → train → report.

        Integrates all P0 Meta-Adaptive phases automatically:
        - Phase 2 (Problem Profiler): auto-detects task type, difficulty, noise.
        - Phase 4 (Knowledge Base): suggests params from past runs; learns after.
        - Phase 3 (MetaOptimizer): tunes params continuously during training.
        - Phase 5 (Feature Gating): auto-selects research features.

        Args:
            evaluator:         Fitness function ``f(genome) -> float``.
            n_inputs:          Input count. Required if ``configure()`` not called.
            n_outputs:         Output count. Required if ``configure()`` not called.
            target_fitness:    Stop when best fitness reaches this value.
            max_time_seconds:  Iteration budget estimated from wall-clock time.
                               Default: ~500 generations with auto pop_size.
            problem_name:      Label stored in the auto_config_report.
            n_warmup:          Warmup genomes for the profiler (default 20).

        Returns:
            :class:`AutoTrainResult` with best genome, diagnostics, and a
            human-readable ``auto_config_report``.
        """
        import time as _time
        from yane.evolution.auto_train import (
            apply_cold_start_defaults,
            build_report,
            pick_pop_size,
        )

        t_start = _time.time()

        # ── 1. Configure if needed ──────────────────────────────────────────
        if n_inputs is not None and n_outputs is not None:
            self.configure(n_inputs=n_inputs, n_outputs=n_outputs)
        elif not self.is_configured:
            raise RuntimeError(
                "Call configure(n_inputs, n_outputs) first, or pass "
                "n_inputs/n_outputs to auto_train()."
            )

        # ── 2. Profile the problem ──────────────────────────────────────────
        _t_prof = _time.time()
        profile = self.profile_problem(evaluator, n_warmup=n_warmup)
        _profiling_ms = (_time.time() - _t_prof) * 1000.0

        # ── 3. Pop-size + structural defaults ───────────────────────────────
        pop_size = pick_pop_size(profile)
        self.set_population_size(pop_size)
        # Matrix-forward is a pure speed optimisation (NumPy path for acyclic
        # feedforward genomes, automatic fallback otherwise) — always safe.
        self.set_matrix_forward(True)
        # Target species: classification/regression benefit from more niches
        # (structural diversity key for discrete mappings); RL is fine at 5.
        _species_for_task = {
            "classification": 10,
            "regression":     10,
            "rl_discrete":     5,
            "rl_continuous":   5,
        }
        self.set_target_species(_species_for_task.get(
            getattr(profile, "task_type", ""), 5
        ))
        # Multi-eval: activate when noise makes single-episode fitness
        # unreliable.  n=3 adds 3× cost but yields much cleaner selection
        # signals; n=5 for very noisy environments (noise_level > 0.6).
        _noise = getattr(profile, "noise_level", 0.0)
        if _noise > 0.6:
            self.set_multi_eval(n=5, aggregation="mean")
        elif _noise > 0.3:
            self.set_multi_eval(n=3, aggregation="mean")

        if target_fitness is not None:
            self.set_min_fitness(target_fitness)

        # ── 4. Knowledge Base & parameter suggestion ────────────────────────
        if self._knowledge_base is None:
            self.set_knowledge_base()

        _suggestions: list = []
        try:
            _suggestions = self._knowledge_base.suggest(profile, top_k=3)
        except Exception:
            pass

        _kb_entries = len(self._knowledge_base) if self._knowledge_base else 0
        _kb_suggestions_used = False
        _kb_conf = 0.0
        _applied_params: dict = {}

        if _suggestions and _suggestions[0].get("confidence", 0.0) >= 0.25:
            best_sug = _suggestions[0]
            _kb_conf = best_sug.get("confidence", 0.0)
            _kb_suggestions_used = True
            for pname, pval in (best_sug.get("params") or {}).items():
                try:
                    self.set_param(pname, pval)
                    _applied_params[pname] = pval
                except Exception:
                    pass
        else:
            _applied_params = apply_cold_start_defaults(self, profile)

        # ── Iteration budget (calculated after Lamarck defaults are known) ──
        # Profiling runs without Lamarck; scale up eval_ms to match training.
        if max_time_seconds is not None:
            _lamarck_steps = getattr(self._lamarck, 'n_steps', 0) if self._lamarck else 0
            try:
                _ps = self.get_param_space()
                _lamarck_steps = max(_lamarck_steps, int(_ps.get("lamarck.n_steps", {}).get("current") or 0))
            except Exception:
                pass
            _eval_multiplier = max(1, 1 + _lamarck_steps)
            eval_ms_per_genome = max(1.0, _profiling_ms / max(1, n_warmup * 2)) * _eval_multiplier
            budget_ms = max_time_seconds * 1000.0
            iters = int(budget_ms * 0.9 / eval_ms_per_genome)
            iters = max(pop_size * 10, min(iters, pop_size * 500))
        else:
            iters = pop_size * 500
        self.set_max_iterations(iters)

        # ── 5. MetaOptimizer ────────────────────────────────────────────────
        self.set_meta_optimizer(
            enabled=True,
            tune_interval=5,
            plateau_patience=30,
            phase_min_gens=10,
            max_overhead_pct=10.0,
        )

        # ── 6. Feature Gating ───────────────────────────────────────────────
        # Scale conservatism with expected training length: for short runs
        # (few total generations) test features rarely and briefly so they
        # don't disturb a near-converged population.
        _expected_gens = max(10, iters // max(1, pop_size))
        _test_interval = max(50, _expected_gens // 6)   # ≥50, at most 1/6 of budget
        _test_duration = max(20, _expected_gens // 15)  # ≥20 gens per test
        _degradation_patience = max(60, _expected_gens // 4)
        # Heavyweight features (STDP, attention, neuromodulation, LTC, augmentation)
        # add per-sample computation that can multiply eval time by 10-100×.
        # Only enable them automatically for temporal/RL tasks where they are
        # likely beneficial; for regression/classification keep them disabled.
        _is_temporal = getattr(profile, "task_type", "") in ("rl_discrete", "rl_continuous")
        self.set_auto_features(
            enabled=True,
            max_concurrent=2,
            test_interval=_test_interval,
            test_duration=_test_duration,
            degradation_patience=_degradation_patience,
            include_heavyweight=_is_temporal,
        )

        # ── 6b. Sparse-reward pre-activation ────────────────────────────────
        # For high reward-sparsity tasks most fitness evaluations return ≈ 0,
        # making vanilla selection nearly random.  Curiosity and diversity
        # injection are known to help; pre-activating them avoids waiting
        # 50+ generations for FeatureGating to discover this on its own.
        # Degradation monitoring still applies — they will be disabled if
        # they hurt later performance.
        _sparsity = getattr(profile, "reward_sparsity", 0.0)
        if self._feature_gate is not None and _sparsity > 0.3:
            self._feature_gate.pre_activate("curiosity", self)
            self._feature_gate.pre_activate("diversity_injection", self)

        # ── 7. Train ────────────────────────────────────────────────────────
        self.train(evaluator)

        wall_time = _time.time() - t_start

        # ── 8. Collect results ───────────────────────────────────────────────
        _best = self._population.get_best() if self._population else None
        final_fitness = _best.raw_fitness if _best else -float("inf")
        total_generations = self._n_evaluations_done // max(1, pop_size)
        active_features = (
            self._feature_gate.get_active_features()
            if self._feature_gate is not None else []
        )
        final_params: dict = {}
        try:
            final_params = {
                name: info["current"]
                for name, info in self.get_param_space().items()
                if info["current"] is not None
            }
        except Exception:
            pass

        # ── 9. Build report ──────────────────────────────────────────────────
        report = build_report(
            profile=profile,
            problem_name=problem_name,
            kb_entries=_kb_entries,
            kb_conf=_kb_conf,
            applied_params=_applied_params,
            cold_start=not _kb_suggestions_used,
            meta_diag=self.get_meta_optimizer_diagnostics(),
            feat_diag=self.get_feature_gating_diagnostics(),
            active_features=active_features,
            pop_size=pop_size,
            max_iters=iters,
            wall_time=wall_time,
            total_generations=total_generations,
            final_fitness=final_fitness,
        )

        return AutoTrainResult(
            best_genome=_best,
            final_fitness=final_fitness,
            total_generations=total_generations,
            wall_time=wall_time,
            active_features=active_features,
            final_params=final_params,
            problem_profile=profile,
            auto_config_report=report,
            kb_suggestions_used=_kb_suggestions_used,
            n_kb_suggestions=len(_suggestions),
        )

    def _ensure_configured(self) -> None:
        if not self.is_configured:
            raise RuntimeError("Call configure(n_inputs, n_outputs) first.")

    def _require_current_genome(self) -> Genome:
        if self._current_genome is None:
            raise RuntimeError("Call next_genome() first.")
        return self._current_genome
