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
from yane.util.resource_guard import ResourceGuard
from yane.evolution.curriculum import Curriculum, CurriculumStage  # noqa: F401

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
        self._resource_guard = ResourceGuard()
        self._resource_check_interval: int = 50  # check psutil every N iters (was 1 = ~5% overhead)
        self._population_size: int = 100
        self._n_workers: int = 1
        # Lamarckian refinement
        self._lamarck = LamarckRefiner()
        # Multi-eval + early stopping
        self._runner = EvaluationRunner()
        # Elitism (applied to population on configure())
        self._elite_count: int = 1
        self._species_elite_count: int = 1
        # Fitness sanitizing (disabled by default)
        self._sanitizer = FitnessSanitizer()
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
        self._lamarck.set_explicit(n_steps, sigma)

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
        self._lamarck.set_adaptive(max_steps, top_k, sigma)

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
            "n_workers": self._n_workers,
            "target_species": pop._target_species if pop else None,
            "compat_threshold": pop._compat_threshold if pop else None,
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
            # Efficiency penalty
            "efficiency_penalty": (
                {"max_ms": self._efficiency_penalty.max_ms,
                 "penalty_per_ms": self._efficiency_penalty.penalty_per_ms}
                if self._efficiency_penalty is not None else None
            ),
            # Resource limits
            "resource_check_interval": self._resource_check_interval,
            "memory_limit_gb": self._resource_guard.max_process_gb,
            # Fitness sanitizing
            "sanitize_enabled": self._sanitizer.enabled,
            "sanitize_fallback": self._sanitizer.fallback,
            "sanitize_clip_low": self._sanitizer.clip_low,
            "sanitize_clip_high": self._sanitizer.clip_high,
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
        from yane.util.logger import setup_logging as _setup, write_json as _wj, write_csv as _wc, log_info as _li
        self._log_run_dir = _setup(name)
        _li("Training started  run_name=%s  pop_size=%d  max_iter=%s  min_fitness=%s",
            name, self._population_size, self.max_iterations, self.min_fitness)
        _wj(self._log_run_dir / "config.json", self._config_dict())

        # --- Logging state ---------------------------------------------------
        _log_interval = max(1, self._population_size // 10)  # log ~10× per generation
        _csv_header = "iteration,best_fitness,mean_fitness,median_fitness,iqr_fitness,species_count,stagnation_count,nodes,connections"
        _csv_path = self._log_run_dir / "fitness_history.csv"

        self._n_evaluations_done = 0
        stop_reason: str | None = None
        iterations = 0
        while True:
            genome = self._population.select_for_evaluation()

            result = self._run_evaluations(genome, fitness_fn)
            fitness = self._finalize_fitness(result.fitness, result.elapsed_ms)
            self._population.submit(genome, fitness, result.elapsed_ms)
            iterations += 1
            self._n_evaluations_done += self._runner.n_evaluations

            # --- Periodic CSV logging ---------------------------------------
            if iterations % _log_interval == 0:
                mem = self.population_memory_info()
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

            # --- Periodic heartbeat + crash-safe snapshot (every 100) -------
            if iterations % 100 == 0:
                mem = self.population_memory_info()
                _li("iter=%d  best=%.4f  avg=%.2f  species=%d  stagn=%d  nodes=%d  conns=%d",
                    iterations,
                    mem.get("max_fitness", 0.0),
                    mem.get("avg_fitness", 0.0),
                    mem.get("species_count", 0),
                    mem.get("stagnation_count", 0),
                    mem.get("largest_genome_nodes", 0),
                    mem.get("largest_genome_connections", 0))
                self._write_crash_snapshot(iterations, mem, _li)

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
                stop_reason = self._check_stop_reason(fitness, iterations, _li)
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
        )
        if self._curriculum is not None:
            info.update(self._curriculum.info())
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
        if self._population is not None:
            self._population._interspecies_crossover_rate = rate

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
            "lamarck_n_applied":     self._lamarck.n_applied,
            "lamarck_n_steps_total": self._lamarck.n_steps_total,
            "lamarck_time_ms":       self._lamarck.time_ms,
            "lamarck_n_blocked_top_k": self._lamarck.n_blocked_top_k,
            "n_invalid_fitness":     self._sanitizer.n_invalid,
            "n_clipped_fitness":     self._sanitizer.n_clipped,
            "n_early_stopped":       self._runner.n_early_stopped,
            "early_stopping_n":      self._runner.early_stopping_n,
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
        # Restore cached config so _config_dict() + logs are accurate.
        self._n_inputs           = cfg.get("n_inputs", self._n_inputs)
        self._n_outputs          = cfg.get("n_outputs", self._n_outputs)
        self._max_nodes          = cfg.get("max_nodes", self._max_nodes)
        self._max_connections    = cfg.get("max_connections", self._max_connections)
        self._n_initial_hidden   = cfg.get("n_initial_hidden", self._n_initial_hidden)
        self._stateful           = cfg.get("stateful", self._stateful)
        self._population_size    = cfg.get("population_size", self._population_size)
        self._seed               = cfg.get("seed", self._seed)

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
        self._population.submit(self._current_genome, self._finalize_fitness(fitness, elapsed_ms), elapsed_ms)
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
            self._population.submit(genome, self._finalize_fitness(fitness, elapsed_ms), elapsed_ms)

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

    def _finalize_fitness(self, fitness: float, elapsed_ms: float | None) -> float:
        """Sanitize + efficiency penalty. Applied by every submission path."""
        fitness = self._sanitizer.apply(fitness)
        if self._efficiency_penalty is not None and elapsed_ms is not None:
            fitness = self._efficiency_penalty.apply(fitness, elapsed_ms)
        return fitness

    def _run_evaluations(
        self, genome: Genome, fitness_fn: Callable[[Genome], float]
    ) -> EvaluationResult:
        return self._runner.run(genome, fitness_fn, self._population, self._lamarck)

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
