from __future__ import annotations
import dataclasses
import gc
import json
import math
import random
import statistics
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

def sanitize_fitness(
    value: float,
    fallback: float = 0.0,
    clip_low: float | None = None,
    clip_high: float | None = None,
) -> tuple[float, bool, bool]:
    """Sanitize a raw fitness value.

    Returns (sanitized_value, was_invalid, was_clipped).

    was_invalid: True when value was nan or inf (replaced by fallback).
    was_clipped: True when value was finite but outside [clip_low, clip_high].
    """
    if not math.isfinite(value):
        return fallback, True, False
    clipped = False
    if clip_low is not None and value < clip_low:
        value = clip_low
        clipped = True
    if clip_high is not None and value > clip_high:
        value = clip_high
        clipped = True
    return value, False, clipped


def _aggregate_fitnesses(fitnesses: list[float], aggregation: str, sigma_penalty: float) -> float:
    """Combine multiple fitness values into one.

    aggregation: "mean" | "median" | "min"
    sigma_penalty: subtract sigma_penalty * std from the result (0 = no penalty).
    """
    if len(fitnesses) == 1:
        return fitnesses[0]
    if aggregation == "median":
        result = statistics.median(fitnesses)
    elif aggregation == "min":
        result = min(fitnesses)
    else:
        result = statistics.mean(fitnesses)
    if sigma_penalty > 0.0:
        result -= sigma_penalty * statistics.pstdev(fitnesses)
    return result


from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.evolution.population import Population
from yane.evolution.efficiency_penalty import EfficiencyPenalty
from yane.util.resource_guard import ResourceGuard


@dataclasses.dataclass
class EvaluationResult:
    """Structured result from evaluating a single genome.

    Returned by ``NeuroEvolution._run_evaluations()`` — used internally by the
    training loop and GUI worker paths.  The public API (``submit_fitness``,
    ``submit_fitness_batch``) is not affected.
    """
    genome: Genome
    fitness: float
    elapsed_ms: float
    n_lamarck_steps: int = 0
    stopped_early: bool = False
    raw_fitnesses: list[float] = dataclasses.field(default_factory=list)


def _derive_run_name(fitness_fn: Callable) -> str:
    """Derive a log category name from a fitness function.

    Uses ``fitness_fn.__name__`` unless it is a lambda or other unhelpful
    name — then falls back to ``"training"``.
    """
    raw = getattr(fitness_fn, '__name__', None)
    if raw and raw != '<lambda>':
        return raw
    return "training"


def _compute_fitness_iqr(evaluated: list) -> float:
    """Compute the interquartile range (IQR) of fitness values.

    Returns 0.0 if fewer than 4 genomes are evaluated.
    """
    if len(evaluated) < 4:
        return 0.0
    fits = sorted(g.fitness for g in evaluated)
    n = len(fits)
    q1 = fits[max(0, int(n * 0.25))]
    q3 = fits[min(n - 1, int(n * 0.75))]
    return q3 - q1


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
        self._lamarck_steps: int = 0      # explicit mode: >0 = always-on before eval
        self._lamarck_sigma: float = 1.0  # step-size multiplier on genome.sigma_global
        # Adaptive Lamarck — fires automatically based on stagnation pressure.
        # Active when _lamarck_steps == 0 and _lamarck_max_steps > 0 (default).
        self._lamarck_max_steps: int = 3   # ceiling: steps at full stagnation
        self._lamarck_top_k: float = 0.2   # only refine top 20 % of evaluated pool
        self._lamarck_n_applied: int = 0   # cumulative refinements performed
        self._lamarck_n_steps_total: int = 0  # cumulative hill-climbing steps
        self._lamarck_n_blocked_top_k: int = 0  # refinements skipped by top-k gate
        self._n_evaluations: int = 1
        self._eval_aggregation: str = "mean"
        self._eval_sigma_penalty: float = 0.0
        # Elitism (applied to population on configure())
        self._elite_count: int = 1
        self._species_elite_count: int = 1
        # Fitness sanitizing (disabled by default)
        self._sanitize: bool = False
        self._sanitize_fallback: float = 0.0
        self._sanitize_clip_low: float | None = None
        self._sanitize_clip_high: float | None = None
        self._n_invalid_fitness: int = 0   # nan / inf seen
        self._n_clipped_fitness: int = 0   # values clipped by clip bounds
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
        # Early stopping per genome (generator protocol)
        self._early_stopping_factor: float | None = None   # None = disabled
        self._n_early_stopped: int = 0
        self._early_stopping_n: int | None = None          # calibrated from first complete run
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
        self._early_stopping_factor = factor

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
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        valid = ("mean", "median", "min")
        if aggregation not in valid:
            raise ValueError(f"aggregation must be one of {valid}, got {aggregation!r}")
        self._n_evaluations = n
        self._eval_aggregation = aggregation
        self._eval_sigma_penalty = max(0.0, sigma_penalty)

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
        self._sanitize = True
        self._sanitize_fallback = fallback
        self._sanitize_clip_low = clip_low
        self._sanitize_clip_high = clip_high

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
        self._lamarck_steps = max(0, n_steps)
        self._lamarck_sigma = sigma
        if n_steps > 0:
            self._lamarck_max_steps = 0  # explicit overrides adaptive

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
        self._lamarck_max_steps = max(0, max_steps)
        self._lamarck_top_k = max(0.0, min(1.0, top_k))
        self._lamarck_sigma = sigma
        self._lamarck_steps = 0  # adaptive mode requires explicit to be off

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
            "n_evaluations": self._n_evaluations,
            "eval_aggregation": self._eval_aggregation,
            "eval_sigma_penalty": self._eval_sigma_penalty,
            # Lamarck
            "lamarck_mode": ("explicit" if self._lamarck_steps > 0 else
                             ("adaptive" if self._lamarck_max_steps > 0 else "off")),
            "lamarck_steps": self._lamarck_steps,
            "lamarck_max_steps": self._lamarck_max_steps,
            "lamarck_top_k": self._lamarck_top_k,
            "lamarck_sigma": self._lamarck_sigma,
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
            "sanitize_enabled": self._sanitize,
            "sanitize_fallback": self._sanitize_fallback,
            "sanitize_clip_low": self._sanitize_clip_low,
            "sanitize_clip_high": self._sanitize_clip_high,
        }

    # -------------------------------------------------------------------------
    # Training (automatic loop)
    # -------------------------------------------------------------------------

    def train(
        self,
        fitness_fn: Callable[[Genome], float],
        run_name: str | None = None,
        on_stop: Callable[[str], None] | None = None,
    ) -> int:
        """Run the evolutionary loop.

        Runs until a stop condition is reached or indefinitely if none is set.
        Stop conditions (checked in priority order):
        - ``min_fitness`` — stops when a genome reaches the target fitness.
        - ``max_evaluations`` — stops after N total fitness-function calls.
        - ``max_iterations`` — stops after N genome evaluations.
        - Convergence — stops when IQR < ``fitness_spread_eps`` at full stagnation.

        Automatically pauses when system memory is low and resumes when it recovers.

        Structured logging is automatically set up: a timestamped directory
        under ``logs/<run_name>/`` receives ``run.log``, ``config.json``,
        ``fitness_history.csv`` and ``best_genome.json``.

        Args:
            fitness_fn: Function that evaluates a genome and returns fitness.
            run_name: Category name for logs (default: derived from *fitness_fn*).
            on_stop: Optional callback called with the stop reason string when
                training ends.  Possible values: ``"target_reached"``,
                ``"max_evaluations"``, ``"max_iterations"``, ``"converged"``.
        Returns:
            Number of iterations performed.
        """
        self._ensure_configured()

        # --- Structured logging setup ----------------------------------------
        name = run_name or _derive_run_name(fitness_fn)
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

        lamarck = self._lamarck_steps > 0
        self._n_evaluations_done = 0
        stop_reason: str | None = None
        iterations = 0
        while True:
            genome = self._population.select_for_evaluation()

            if lamarck:
                self._lamarck_refine(genome, fitness_fn)  # refines weights in-place

            result = self._run_evaluations(genome, fitness_fn)
            fitness = self._apply_sanitize(result.fitness)

            if self._efficiency_penalty is not None:
                fitness = self._efficiency_penalty.apply(fitness, result.elapsed_ms)

            self._population.submit(genome, fitness, result.elapsed_ms)
            iterations += 1
            self._n_evaluations_done += self._n_evaluations

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
                # Crash-safe state snapshot — survives segfaults.
                if self._log_run_dir is not None:
                    try:
                        import json
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
                    except Exception:
                        pass

            if self.min_fitness is not None and fitness >= self.min_fitness:
                stop_reason = "target_reached"
                _li("Training stopped: target fitness reached  fitness=%.6f  iterations=%d",
                    fitness, iterations)
                break
            if self._max_evaluations is not None and self._n_evaluations_done >= self._max_evaluations:
                stop_reason = "max_evaluations"
                _li("Training stopped: max evaluations reached  evals=%d  iterations=%d",
                    self._n_evaluations_done, iterations)
                break
            if self.max_iterations is not None and iterations >= self.max_iterations:
                stop_reason = "max_iterations"
                _li("Training stopped: max iterations reached  iterations=%d", iterations)
                break
            if (self._convergence_spread_eps is not None
                    and iterations % max(1, self._population_size // 5) == 0):
                pop = self._population
                stag_threshold = max(1, pop.stagnation_threshold)
                stag_frac = pop.stagnation_count / stag_threshold
                if stag_frac >= self._convergence_min_stagnation:
                    iqr = _compute_fitness_iqr(pop._evaluated)
                    if iqr < self._convergence_spread_eps:
                        stop_reason = "converged"
                        _li("Training stopped: converged  iqr=%.6f  stagnation=%d  iterations=%d",
                            iqr, pop.stagnation_count, iterations)
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
        if self._log_run_dir is not None:
            import pickle
            best = self.get_best()
            pkl_path = self._log_run_dir / "best_genome.pkl"
            pkl_path.write_bytes(pickle.dumps(best))
            # Human-readable summary.
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
                "lamarck_n_applied":       mem.get("lamarck_n_applied", 0),
                "lamarck_n_steps_total":   mem.get("lamarck_n_steps_total", 0),
                "lamarck_n_blocked_top_k": mem.get("lamarck_n_blocked_top_k", 0),
                "n_invalid_fitness": mem.get("n_invalid_fitness", 0),
                "n_clipped_fitness": mem.get("n_clipped_fitness", 0),
            })
            _li("Training finished  best_fitness=%.6f  nodes=%d  connections=%d  iterations=%d  "
                "stop_reason=%s  evals=%d  lamarck_applied=%d  lamarck_blocked_top_k=%d",
                best.fitness, len(best.nodes), best.connection_count, iterations,
                stop_reason or "manual", self._n_evaluations_done,
                mem.get("lamarck_n_applied", 0), mem.get("lamarck_n_blocked_top_k", 0))

        return iterations

    def get_best(self) -> Genome:
        self._ensure_configured()
        return self._population.get_best()

    def get_ensemble(self, k: int = 3) -> list[Genome]:
        """Return the top-k genomes by fitness for ensemble inference."""
        self._ensure_configured()
        return self._population.get_top(k)

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
        info = {
            "total_genomes": len(all_genomes),
            "pop_evaluated":  len(self._population._evaluated),
            "pop_max":        self._population.max_size,
            "total_nodes": total_nodes,
            "total_connections": total_connections,
            "avg_nodes_per_genome": total_nodes / len(all_genomes),
            "avg_connections_per_genome": total_connections / len(all_genomes),
            "largest_genome_nodes": max_nodes,
            "largest_genome_connections": max_connections,
            "species_count":           self._population.species_count,
            "target_species":          self._population._target_species,
            "compat_threshold":        self._population._compat_threshold,
            "stagnation_count":        self._population.stagnation_count,
            "stagnation_threshold":    self._population.stagnation_threshold,
            "since_last_injection":    self._population._since_last_injection,
            "spawn_count":             self._population._spawn_count,
            "dbg_last_adj_n":          self._population._dbg_last_adj_n,
            "dbg_adj_count":           self._population._dbg_adj_count,
            "novelty_weight":          self._population.novelty_weight,
            "efficiency_weight":       self._population.efficiency_weight,
            "min_fitness": min((g.fitness for g in self._population._evaluated), default=0.0),
            "max_fitness": max((g.fitness for g in self._population._evaluated), default=0.0),
            "avg_fitness": (sum(g.fitness for g in self._population._evaluated)
                            / max(1, len(self._population._evaluated))),
            "top5_avg_fitness": (
                sum(f for f in sorted(
                    (g.fitness for g in self._population._evaluated),
                    reverse=True
                )[:5]) / min(5, len(self._population._evaluated))
                if self._population._evaluated else 0.0
            ),
            "median_fitness": (statistics.median(g.fitness for g in self._population._evaluated)
                               if self._population._evaluated else 0.0),
            "fitness_iqr": _compute_fitness_iqr(self._population._evaluated),
            "n_evaluations":    self._n_evaluations,
            "eval_aggregation": self._eval_aggregation,
            # Elitism configuration
            "elite_count":         self._population.elite_count,
            "species_elite_count": self._population.species_elite_count,
            # Fitness sanitizing diagnostics
            "sanitize_enabled":    self._sanitize,
            "n_invalid_fitness":   self._n_invalid_fitness,
            "n_clipped_fitness":   self._n_clipped_fitness,
            # Offspring counters
            "n_crossover":              self._population._n_crossover,
            "n_mutation_only":          self._population._n_mutation_only,
            "n_diversity_injection":    self._population._n_diversity_injection,
            "n_interspecies_crossover": self._population._n_interspecies_crossover,
            "n_early_stopped":          self._n_early_stopped,
            # Lamarck diagnostics
            "lamarck_mode":          "explicit" if self._lamarck_steps > 0 else
                                     ("adaptive" if self._lamarck_max_steps > 0 else "off"),
            "lamarck_n_applied":        self._lamarck_n_applied,
            "lamarck_n_steps_total":    self._lamarck_n_steps_total,
            "lamarck_n_blocked_top_k":  self._lamarck_n_blocked_top_k,
            # Per-species Lamarck diagnostics (for GUI / debugging).
            "lamarck_per_species": [
                {
                    "species_idx": i,
                    "members": len(sp.members),
                    "stagnation_count": sp.stagnation_count,
                    "lamarck_n_applied": sp.lamarck_n_applied,
                    "lamarck_n_steps_total": sp.lamarck_n_steps_total,
                }
                for i, sp in enumerate(self._population._species)
            ],
            # Best genome topology history: list of (total_submitted, n_nodes, n_conn, fitness)
            "best_topology_history": self._population._best_topology_history,
        }

        # Population-wide average mutation rates
        # Averaging across all evaluated genomes gives a sense of where the
        # population's search strategy is converging — e.g. rising sigma_global
        # population-wide signals an exploration phase.
        evaluated = self._population._evaluated
        if evaluated:
            info["pop_avg_sigma_global"] = sum(
                g.sigma_global for g in evaluated) / len(evaluated)
            info["pop_avg_add_node"] = sum(
                g.mutation_add_node.bool_rate for g in evaluated) / len(evaluated)
            info["pop_avg_rem_node"] = sum(
                g.mutation_remove_node.bool_rate for g in evaluated) / len(evaluated)
            info["pop_avg_add_conn"] = sum(
                g.mutation_add_connection.bool_rate for g in evaluated) / len(evaluated)
            info["pop_avg_rem_conn"] = sum(
                g.mutation_remove_connection.bool_rate for g in evaluated) / len(evaluated)
            # Per-node rates (bias shift, activation change)
            all_nodes = [n for g in evaluated for n in g.nodes]
            if all_nodes:
                info["pop_avg_bias_rate"]  = sum(
                    n.mutation_bias.shift_rate for n in all_nodes) / len(all_nodes)
                info["pop_avg_activ_rate"] = sum(
                    n.mutation_activation.custom_rate for n in all_nodes) / len(all_nodes)
            # Per-connection weight rates
            all_conns = [c for g in evaluated for n in g.nodes for c in n.connections]
            if all_conns:
                info["pop_avg_weight_rate"] = sum(
                    c.mutation.shift_rate for c in all_conns) / len(all_conns)

        # Eval-time statistics (computed on demand from current evaluated genomes)
        eval_times = [
            g.eval_time_ms
            for g in self._population._evaluated
            if g.eval_time_ms is not None and math.isfinite(g.eval_time_ms)
        ]
        if eval_times:
            sorted_times = sorted(eval_times)
            n = len(sorted_times)
            p95_idx = min(int(math.ceil(0.95 * n)) - 1, n - 1)
            info["eval_time_mean_ms"]   = statistics.mean(sorted_times)
            info["eval_time_median_ms"] = statistics.median(sorted_times)
            info["eval_time_p95_ms"]    = sorted_times[max(0, p95_idx)]
            info["eval_time_max_ms"]    = sorted_times[-1]

        return info

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
        import pickle
        self._ensure_configured()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "config": self._config_dict(),
            "population": self._population,
            "tracker": self._tracker,
            "lamarck_n_applied": self._lamarck_n_applied,
            "lamarck_n_steps_total": self._lamarck_n_steps_total,
            "lamarck_n_blocked_top_k": self._lamarck_n_blocked_top_k,
            "n_invalid_fitness": self._n_invalid_fitness,
            "n_clipped_fitness": self._n_clipped_fitness,
            "n_early_stopped": self._n_early_stopped,
            "early_stopping_n": self._early_stopping_n,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        tmp.replace(path)

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
        import pickle
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        payload = pickle.loads(path.read_bytes())
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError(f"Unsupported checkpoint format in {path}")
        cfg = payload["config"]
        self._population  = payload["population"]
        self._tracker     = payload["tracker"]
        self._lamarck_n_applied       = payload.get("lamarck_n_applied", 0)
        self._lamarck_n_steps_total   = payload.get("lamarck_n_steps_total", 0)
        self._lamarck_n_blocked_top_k = payload.get("lamarck_n_blocked_top_k", 0)
        self._n_invalid_fitness       = payload.get("n_invalid_fitness", 0)
        self._n_clipped_fitness       = payload.get("n_clipped_fitness", 0)
        self._n_early_stopped         = payload.get("n_early_stopped", 0)
        self._early_stopping_n        = payload.get("early_stopping_n", None)
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
        self._population.submit(self._current_genome, self._apply_sanitize(fitness), elapsed_ms)
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
            self._population.submit(genome, self._apply_sanitize(fitness), elapsed_ms)

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
        """Apply configured sanitizing; count and log anomalies. No-op when disabled."""
        if not self._sanitize:
            return fitness
        clean, invalid, clipped = sanitize_fitness(
            fitness,
            fallback=self._sanitize_fallback,
            clip_low=self._sanitize_clip_low,
            clip_high=self._sanitize_clip_high,
        )
        if invalid:
            self._n_invalid_fitness += 1
            from yane.util.logger import get_logger
            get_logger().warning(
                "sanitize_fitness: invalid value %r replaced with fallback %r "
                "(total invalid: %d)",
                fitness, self._sanitize_fallback, self._n_invalid_fitness,
            )
        elif clipped:
            self._n_clipped_fitness += 1
        return clean

    def _run_evaluations(
        self, genome: Genome, fitness_fn: Callable[[Genome], float]
    ) -> EvaluationResult:
        """Evaluate genome (possibly multiple times) → EvaluationResult.

        Supports two calling conventions for ``fitness_fn``:

        1. **Regular function** — called ``n_evaluations`` times; results are
           aggregated with the configured aggregation strategy.
        2. **Generator function** — called once; each ``yield`` is one episode
           result.  Early stopping (see ``set_early_stopping()``) aborts the
           generator when the running mean drops below the worst-pool threshold.

        After the normal evaluation, adaptive Lamarck refinement fires when
        stagnation pressure is high and the genome ranks in the top fraction of
        the pool — improving weights without touching topology.  Explicit Lamarck
        (_lamarck_steps > 0) is handled in train() instead and skips this path.
        """
        import inspect
        start = time.perf_counter()
        raw: list[float] = []
        stopped_early = False

        if inspect.isgeneratorfunction(fitness_fn):
            gen = fitness_fn(genome)
            N = self._early_stopping_n   # snapshot; None until calibrated
            cumulative = 0.0
            episode_count = 0
            try:
                for k, episode_fitness in enumerate(gen, 1):
                    raw.append(episode_fitness)
                    cumulative += episode_fitness
                    episode_count = k
                    if (self._early_stopping_factor is not None
                            and N is not None
                            and k >= max(1, N // 5)):
                        estimated = cumulative * (N / k)
                        evaluated = self._population._evaluated
                        if evaluated:
                            best = max(g.fitness for g in evaluated)
                            threshold = best - abs(best) * self._early_stopping_factor
                            if estimated < threshold:
                                gen.close()
                                stopped_early = True
                                self._n_early_stopped += 1
                                break
            except StopIteration:
                pass
            # Calibrate N from the first non-stopped complete run.
            if not stopped_early and episode_count > 0 and self._early_stopping_n is None:
                self._early_stopping_n = episode_count
            fitness = _aggregate_fitnesses(raw, self._eval_aggregation, self._eval_sigma_penalty) if raw else 0.0
        elif self._n_evaluations <= 1:
            fitness = fitness_fn(genome)
        else:
            for _ in range(self._n_evaluations):
                raw.append(fitness_fn(genome))
            fitness = _aggregate_fitnesses(
                raw, self._eval_aggregation, self._eval_sigma_penalty
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        n_lamarck_steps = 0
        # Adaptive Lamarck: fires after the baseline is known, only when
        # _lamarck_steps == 0 (explicit mode is off) and stagnation is high.
        if self._lamarck_steps == 0:
            n_steps = self._adaptive_lamarck_steps(genome, fitness)
            if n_steps > 0:
                fitness = self._lamarck_refine(
                    genome, fitness_fn,
                    baseline_fitness=fitness,
                    n_steps=n_steps,
                )
                n_lamarck_steps = n_steps
                self._lamarck_n_applied += 1
                self._lamarck_n_steps_total += n_steps
                # Per-species tracking for diagnostics.
                sp = self._population.get_species_for_genome(genome)
                if sp is not None:
                    sp.lamarck_n_applied += 1
                    sp.lamarck_n_steps_total += n_steps

        return EvaluationResult(
            genome=genome,
            fitness=fitness,
            elapsed_ms=elapsed_ms,
            n_lamarck_steps=n_lamarck_steps,
            stopped_early=stopped_early,
            raw_fitnesses=raw,
        )

    def _lamarck_refine(
        self,
        genome: Genome,
        fitness_fn: Callable[[Genome], float],
        baseline_fitness: float | None = None,
        n_steps: int | None = None,
    ) -> float:
        """Hill-climb weights and biases, return best fitness achieved.

        Only weights and biases are touched — topology (connections, nodes) is
        unchanged, so the compiled forward pass stays valid without rebuilding.
        Each step: perturb → evaluate → keep if better, else revert.

        Args:
            baseline_fitness: known fitness before hill-climbing.  If None a
                              fresh evaluation is performed to establish the baseline.
            n_steps:          override for the number of hill-climbing attempts.
                              Defaults to self._lamarck_steps.
        """
        steps = self._lamarck_steps if n_steps is None else n_steps
        if steps <= 0:
            return baseline_fitness if baseline_fitness is not None else fitness_fn(genome)

        conns = [conn for node in genome.nodes for conn in node.connections if conn.enabled]
        nodes = genome.nodes
        if not conns and not nodes:
            return baseline_fitness if baseline_fitness is not None else fitness_fn(genome)

        sigma = genome.sigma_global * self._lamarck_sigma
        if not (0.0 < sigma < 1e6):
            return baseline_fitness if baseline_fitness is not None else fitness_fn(genome)

        best_fitness = fitness_fn(genome) if baseline_fitness is None else baseline_fitness

        for _ in range(steps):
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

        return best_fitness

    def _adaptive_lamarck_steps(self, genome: Genome, fitness: float) -> int:
        """Compute how many Lamarck steps to apply based on *species* stagnation + top-K.

        Uses the genome's own species stagnation count when available, falling
        back to the global population stagnation.  This targets refinement
        effort at species that are locally stuck rather than spreading it
        uniformly across the whole population.

        Returns 0 if adaptive mode is off or conditions aren't met.
        """
        if self._lamarck_max_steps <= 0 or self._population is None:
            return 0
        pop = self._population

        # --- Per-species stagnation (preferred) -------------------------------
        sp = pop.get_species_for_genome(genome)
        if sp is not None and sp.stagnation_count > 0:
            stag_count = sp.stagnation_count
            stag_threshold = pop.stagnation_threshold
        else:
            # Fallback: global stagnation (e.g. genome not yet assigned).
            stag_count = pop.stagnation_count
            stag_threshold = max(1, pop.stagnation_threshold)

        stag_frac = min(1.0, stag_count / max(1, stag_threshold))
        n_steps = round(stag_frac * self._lamarck_max_steps)
        if n_steps <= 0:
            return 0

        # Top-K gate: only refine genomes that rank in the top fraction of the pool.
        evaluated = pop._evaluated
        if evaluated and self._lamarck_top_k < 1.0:
            k = max(1, int(len(evaluated) * self._lamarck_top_k))
            # List comprehension is faster than a generator here: sorted() exhausts
            # generators lazily (per-yield overhead); a flat float list is cheaper.
            threshold = sorted(
                [g.fitness for g in evaluated], reverse=True
            )[min(k - 1, len(evaluated) - 1)]
            if fitness < threshold:
                self._lamarck_n_blocked_top_k += 1
                # Warn at milestones when the gate consistently blocks at high stagnation.
                # Common cause: n_evaluations=1 with a stochastic environment — stored
                # pool fitness values are biased toward lucky episodes, making the
                # threshold artificially high.  Fix: raise lamarck_top_k or increase
                # n_evaluations.
                if stag_frac >= 1.0 and self._lamarck_n_blocked_top_k in (100, 1000, 10_000):
                    from yane.util.logger import log_warning
                    log_warning(
                        "Lamarck blocked %d× by top-k gate despite full stagnation "
                        "(fitness=%.4f < pool-threshold=%.4f, top_k=%.2f). "
                        "Consider n_evaluations>1 or lamarck_top_k=1.0 for noisy environments.",
                        self._lamarck_n_blocked_top_k, fitness, threshold,
                        self._lamarck_top_k,
                    )
                return 0

        return n_steps

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
