"""Unified Parameter Registry for YANE — Phase 1 of P0 Meta-Adaptive Orchestration.

Every tunable hyperparameter is registered here with its type, domain, default
and a dispatcher that calls the appropriate ``set_*()`` method.  This registry
is the foundation that later phases (Problem Profiler, Knowledge Base,
MetaOptimizer, Feature Gating) build on.

Design rules:
- The registry is a *catalog and tracking layer*, not the ground truth.
  ``NeuroEvolution._*`` attributes remain authoritative.
- Existing ``set_*()`` methods are unchanged; no side-effects added.
- ``set_param(name, value)`` validates, dispatches to ``set_*()``, then
  records the change.
- Pickle-safe: ``__setstate__`` on both classes handles missing fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from yane.neuro_evolution import NeuroEvolution


# ---------------------------------------------------------------------------
# ParamSpec
# ---------------------------------------------------------------------------

@dataclass
class ParamSpec:
    """Describes a single tunable parameter.

    Attributes
    ----------
    name:          Dot-namespaced identifier, e.g. ``"lamarck.mode"``.
    type:          ``"categorical"`` | ``"continuous"`` | ``"integer"`` | ``"boolean"``.
    domain:        For categorical: ``list`` of allowed values.
                   For continuous/integer: ``(min, max)`` tuple (inclusive).
                   For boolean: ``[True, False]``.
    default:       Value used when parameter has never been explicitly set.
    stage:         ``"init"`` (configure-time only), ``"runtime"`` (during train),
                   or ``"both"``.
    subsystem:     Which YANE component owns this parameter.
    description:   Short human-readable description.
    impact_history: Fitness-delta log after each change (filled at runtime).
    """

    name: str
    type: str
    domain: Any
    default: Any
    stage: str
    subsystem: str
    description: str = ""
    impact_history: list[float] = field(default_factory=list)
    _current_value: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._current_value = self.default

    @property
    def current_value(self) -> Any:
        return self._current_value

    @current_value.setter
    def current_value(self, value: Any) -> None:
        self._current_value = value

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.__dict__.setdefault("description", "")
        self.__dict__.setdefault("impact_history", [])
        self.__dict__.setdefault("_current_value", self.__dict__.get("default"))


# ---------------------------------------------------------------------------
# ParamRegistry
# ---------------------------------------------------------------------------

class ParamRegistry:
    """Central catalog of all YANE hyperparameters.

    Usage::

        registry = ne._ensure_param_registry()
        ne.set_param("lamarck.mode", "cma_es")  # validates + dispatches
        space = ne.get_param_space()             # full catalog as dict
    """

    def __init__(self) -> None:
        self._specs: dict[str, ParamSpec] = {}
        self._dispatchers: dict[str, Callable[["NeuroEvolution", Any], None]] = {}
        self._change_log: list[dict] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        spec: ParamSpec,
        dispatcher: Callable[["NeuroEvolution", Any], None] | None = None,
    ) -> None:
        """Register a parameter.

        Parameters
        ----------
        spec:       Fully populated ``ParamSpec``.
        dispatcher: Callable ``(ne, value) → None`` that applies the change to
                    a ``NeuroEvolution`` instance.  May be ``None`` for
                    read-only or future parameters.
        """
        self._specs[spec.name] = spec
        if dispatcher is not None:
            self._dispatchers[spec.name] = dispatcher

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_spec(self, name: str) -> ParamSpec | None:
        return self._specs.get(name)

    def get_current(self, name: str) -> Any:
        spec = self._specs.get(name)
        return spec.current_value if spec is not None else None

    def list_names(self) -> list[str]:
        return sorted(self._specs.keys())

    def get_param_space(self) -> dict[str, dict]:
        """Return the full parameter catalog as a plain, serialisable dict."""
        return {
            name: {
                "type": spec.type,
                "domain": spec.domain,
                "default": spec.default,
                "current": spec.current_value,
                "stage": spec.stage,
                "subsystem": spec.subsystem,
                "description": spec.description,
                "impact_history": list(spec.impact_history),
            }
            for name, spec in sorted(self._specs.items())
        }

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def set_value(self, name: str, value: Any, record: bool = True) -> None:
        """Update current value tracking without dispatching to a setter.

        Use this to record that a parameter was changed by some other means,
        or in tests that want to manipulate registry state directly.
        The value is still validated against the spec's domain.
        """
        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(f"Unknown parameter: {name!r}")
        _validate(spec, value)
        old = spec.current_value
        spec.current_value = value
        if record:
            self._change_log.append({"name": name, "old": old, "new": value})
            if len(self._change_log) > 500:
                self._change_log = self._change_log[-500:]

    def dispatch(self, ne: "NeuroEvolution", name: str, value: Any) -> None:
        """Validate, dispatch to setter, then record the change."""
        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(f"Unknown parameter: {name!r}")
        _validate(spec, value)
        dispatcher = self._dispatchers.get(name)
        if dispatcher is not None:
            dispatcher(ne, value)
        old = spec.current_value
        spec.current_value = value
        self._change_log.append({"name": name, "old": old, "new": value})
        if len(self._change_log) > 500:
            self._change_log = self._change_log[-500:]

    def record_fitness_impact(self, name: str, delta: float) -> None:
        """Append a fitness-delta to the param's impact history."""
        spec = self._specs.get(name)
        if spec is not None:
            spec.impact_history.append(float(delta))
            if len(spec.impact_history) > 200:
                spec.impact_history = spec.impact_history[-200:]

    def get_change_log(self, last_n: int = 20) -> list[dict]:
        return list(self._change_log[-last_n:])

    def get_diagnostics(self) -> dict:
        return {
            "n_params": len(self._specs),
            "recent_changes": self.get_change_log(10),
            "subsystems": sorted({s.subsystem for s in self._specs.values()}),
        }

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.__dict__.setdefault("_specs", {})
        self.__dict__.setdefault("_dispatchers", {})
        self.__dict__.setdefault("_change_log", [])


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _validate(spec: ParamSpec, value: Any) -> None:
    """Raise ValueError / TypeError when value violates the spec's domain."""
    if spec.type == "categorical":
        if value not in spec.domain:
            raise ValueError(
                f"Parameter {spec.name!r}: {value!r} is not in allowed values "
                f"{spec.domain}"
            )
    elif spec.type == "boolean":
        if not isinstance(value, bool):
            raise TypeError(
                f"Parameter {spec.name!r}: expected bool, got {type(value).__name__}"
            )
    elif spec.type in ("continuous", "integer"):
        lo, hi = spec.domain
        if not (lo <= value <= hi):
            raise ValueError(
                f"Parameter {spec.name!r}: {value!r} is outside [{lo}, {hi}]"
            )
        if spec.type == "integer" and not isinstance(value, int):
            raise TypeError(
                f"Parameter {spec.name!r}: expected int, got {type(value).__name__}"
            )
    else:
        raise ValueError(f"Unknown param type {spec.type!r} for {spec.name!r}")


# ---------------------------------------------------------------------------
# Default registry builder
# ---------------------------------------------------------------------------

def build_default_registry(ne: "NeuroEvolution") -> ParamRegistry:
    """Create and return a ``ParamRegistry`` pre-populated with all known YANE
    parameters and their dispatchers for the given ``NeuroEvolution`` instance.
    """
    reg = ParamRegistry()
    _register_lamarck(reg)
    _register_population(reg)
    _register_speciation(reg)
    _register_anytime_eval(reg)
    _register_recovery(reg)
    _register_novelty(reg)
    _register_crossover(reg)
    _register_surrogate(reg)
    _register_curiosity(reg)
    _register_darts(reg)
    _register_shared_weights(reg)
    _register_pruning(reg)
    _register_misc(reg)
    # Layer-3 Research Features
    _register_attention(reg)
    _register_stdp(reg)
    _register_neuromodulation(reg)
    _register_ltc(reg)
    _register_input_grouping(reg)
    _register_output_grouping(reg)
    _register_conv_neat(reg)
    _register_hybrid_mode(reg)
    _register_probabilistic(reg)
    _register_island_model(reg)
    _register_augmentation(reg)
    _register_phylogeny(reg)
    _register_safety(reg)
    return reg


# ---------------------------------------------------------------------------
# Sub-registrations — one function per subsystem
# ---------------------------------------------------------------------------

def _register_lamarck(reg: ParamRegistry) -> None:
    modes = ["hill_climbing", "nes", "sa", "cma_es"]

    reg.register(
        ParamSpec(
            name="lamarck.mode",
            type="categorical",
            domain=modes,
            default="hill_climbing",
            stage="both",
            subsystem="lamarck",
            description="Weight-refinement algorithm used during Lamarck phase.",
        ),
        dispatcher=lambda ne, v: ne.set_lamarck(mode=v),
    )
    reg.register(
        ParamSpec(
            name="lamarck.n_steps",
            type="integer",
            domain=(0, 50),
            default=5,
            stage="both",
            subsystem="lamarck",
            description="Number of Lamarck refinement steps per genome.",
        ),
        dispatcher=lambda ne, v: ne.set_lamarck(n_steps=v),
    )
    reg.register(
        ParamSpec(
            name="lamarck.sigma",
            type="continuous",
            domain=(0.001, 10.0),
            default=1.0,
            stage="both",
            subsystem="lamarck",
            description="Perturbation scale for Lamarck hill-climbing/NES/SA.",
        ),
        dispatcher=lambda ne, v: ne.set_lamarck(sigma=v),
    )
    reg.register(
        ParamSpec(
            name="lamarck.enabled",
            type="boolean",
            domain=[True, False],
            default=True,
            stage="both",
            subsystem="lamarck",
            description="Enable or disable all Lamarck refinement.",
        ),
        dispatcher=lambda ne, v: (
            ne.set_lamarck(n_steps=5) if v else ne.set_lamarck(n_steps=0)
        ),
    )
    reg.register(
        ParamSpec(
            name="lamarck.momentum_enabled",
            type="boolean",
            domain=[True, False],
            default=False,
            stage="both",
            subsystem="lamarck",
            description="Use gradient-based momentum to bias mutation direction.",
        ),
        dispatcher=lambda ne, v: ne.set_lamarck_momentum(enabled=v),
    )


def _register_population(reg: ParamRegistry) -> None:
    reg.register(
        ParamSpec(
            name="pop.size",
            type="integer",
            domain=(10, 5000),
            default=100,
            stage="both",
            subsystem="pop",
            description="Number of genomes in the population.",
        ),
        dispatcher=lambda ne, v: ne.set_population_size(v),
    )
    reg.register(
        ParamSpec(
            name="pop.adaptive_enabled",
            type="boolean",
            domain=[True, False],
            default=False,
            stage="both",
            subsystem="pop",
            description="Enable dynamic population-size adaptation.",
        ),
        dispatcher=lambda ne, v: ne.set_adaptive_pop_size(enabled=v),
    )
    reg.register(
        ParamSpec(
            name="pop.adaptive_min",
            type="integer",
            domain=(10, 1000),
            default=20,
            stage="both",
            subsystem="pop",
            description="Minimum population size during adaptive scheduling.",
        ),
        dispatcher=lambda ne, v: ne.set_adaptive_pop_size(min_pop=v),
    )
    reg.register(
        ParamSpec(
            name="pop.adaptive_max",
            type="integer",
            domain=(20, 5000),
            default=500,
            stage="both",
            subsystem="pop",
            description="Maximum population size during adaptive scheduling.",
        ),
        dispatcher=lambda ne, v: ne.set_adaptive_pop_size(max_pop=v),
    )
    reg.register(
        ParamSpec(
            name="pop.adaptive_schedule",
            type="categorical",
            domain=["linear_decay", "performance_based"],
            default="linear_decay",
            stage="both",
            subsystem="pop",
            description="Schedule strategy for adaptive population-size changes.",
        ),
        dispatcher=lambda ne, v: ne.set_adaptive_pop_size(schedule=v),
    )


def _register_speciation(reg: ParamRegistry) -> None:
    reg.register(
        ParamSpec(
            name="speciation.enabled",
            type="boolean",
            domain=[True, False],
            default=True,
            stage="both",
            subsystem="speciation",
            description="Enable NEAT-style speciation.",
        ),
        dispatcher=lambda ne, v: ne.set_speciation(v),
    )
    reg.register(
        ParamSpec(
            name="speciation.target_n",
            type="integer",
            domain=(1, 100),
            default=5,
            stage="both",
            subsystem="speciation",
            description="Target number of species for self-tuning speciation.",
        ),
        dispatcher=lambda ne, v: ne.set_target_species(v),
    )
    reg.register(
        ParamSpec(
            name="speciation.target_min",
            type="integer",
            domain=(1, 50),
            default=3,
            stage="both",
            subsystem="speciation",
            description="Minimum target species for PI-controller band.",
        ),
        dispatcher=lambda ne, v: ne.set_target_species(n_min=v),
    )
    reg.register(
        ParamSpec(
            name="speciation.target_max",
            type="integer",
            domain=(2, 100),
            default=8,
            stage="both",
            subsystem="speciation",
            description="Maximum target species for PI-controller band.",
        ),
        dispatcher=lambda ne, v: ne.set_target_species(n_max=v),
    )


def _register_anytime_eval(reg: ParamRegistry) -> None:
    reg.register(
        ParamSpec(
            name="anytime.enabled",
            type="boolean",
            domain=[True, False],
            default=False,
            stage="both",
            subsystem="anytime",
            description="Enable adaptive evaluation budgeting.",
        ),
        dispatcher=lambda ne, v: ne.set_anytime_eval(enabled=v),
    )
    reg.register(
        ParamSpec(
            name="anytime.min_evals",
            type="integer",
            domain=(1, 20),
            default=1,
            stage="both",
            subsystem="anytime",
            description="Minimum fitness evaluations for weak genomes.",
        ),
        dispatcher=lambda ne, v: ne.set_anytime_eval(min_evals=v),
    )
    reg.register(
        ParamSpec(
            name="anytime.max_evals",
            type="integer",
            domain=(1, 50),
            default=5,
            stage="both",
            subsystem="anytime",
            description="Maximum fitness evaluations for promoted genomes.",
        ),
        dispatcher=lambda ne, v: ne.set_anytime_eval(max_evals=v),
    )
    reg.register(
        ParamSpec(
            name="anytime.promotion_frac",
            type="continuous",
            domain=(0.05, 0.95),
            default=0.3,
            stage="both",
            subsystem="anytime",
            description="Fraction of genomes promoted to full evaluation.",
        ),
        dispatcher=lambda ne, v: ne.set_anytime_eval(promotion_frac=v),
    )


def _register_recovery(reg: ParamRegistry) -> None:
    reg.register(
        ParamSpec(
            name="recovery.enabled",
            type="boolean",
            domain=[True, False],
            default=False,
            stage="both",
            subsystem="recovery",
            description="Enable adaptive stagnation/diversity recovery.",
        ),
        dispatcher=lambda ne, v: ne.set_adaptive_recovery(enabled=v),
    )
    reg.register(
        ParamSpec(
            name="recovery.cooldown",
            type="integer",
            domain=(5, 500),
            default=20,
            stage="both",
            subsystem="recovery",
            description="Generations to wait between recovery actions.",
        ),
        dispatcher=lambda ne, v: ne.set_adaptive_recovery(cooldown=v),
    )
    reg.register(
        ParamSpec(
            name="recovery.escalate",
            type="boolean",
            domain=[True, False],
            default=True,
            stage="both",
            subsystem="recovery",
            description="Escalate to next strategy if current one fails.",
        ),
        dispatcher=lambda ne, v: ne.set_adaptive_recovery(escalate=v),
    )
    reg.register(
        ParamSpec(
            name="recovery.early_stopping_patience",
            type="integer",
            domain=(10, 10000),
            default=500,
            stage="both",
            subsystem="recovery",
            description="Generations without improvement before early stop.",
        ),
        dispatcher=lambda ne, v: ne.set_adaptive_recovery(
            early_stopping_patience=v
        ),
    )
    reg.register(
        ParamSpec(
            name="recovery.injection_frac",
            type="continuous",
            domain=(0.01, 0.5),
            default=0.1,
            stage="both",
            subsystem="recovery",
            description="Fraction of population replaced during diversity injection.",
        ),
        dispatcher=lambda ne, v: ne.set_adaptive_recovery(injection_frac=v),
    )


def _register_novelty(reg: ParamRegistry) -> None:
    reg.register(
        ParamSpec(
            name="novelty.enabled",
            type="boolean",
            domain=[True, False],
            default=True,
            stage="both",
            subsystem="novelty",
            description="Enable novelty search as selection pressure.",
        ),
        dispatcher=lambda ne, v: ne.set_novelty_search(v),
    )


def _register_crossover(reg: ParamRegistry) -> None:
    reg.register(
        ParamSpec(
            name="crossover.enabled",
            type="boolean",
            domain=[True, False],
            default=True,
            stage="both",
            subsystem="crossover",
            description="Enable sexual reproduction (crossover).",
        ),
        dispatcher=lambda ne, v: ne.set_crossover(v),
    )
    reg.register(
        ParamSpec(
            name="crossover.interspecies_rate",
            type="continuous",
            domain=(0.0, 0.3),
            default=0.0,
            stage="both",
            subsystem="crossover",
            description="Probability of selecting a mate from a different species.",
        ),
        dispatcher=lambda ne, v: ne.set_interspecies_crossover(v),
    )
    reg.register(
        ParamSpec(
            name="crossover.weight_inheritance_enabled",
            type="boolean",
            domain=[True, False],
            default=False,
            stage="both",
            subsystem="crossover",
            description="Blend parent weights during crossover.",
        ),
        dispatcher=lambda ne, v: ne.set_weight_inheritance(enabled=v),
    )
    reg.register(
        ParamSpec(
            name="crossover.weight_blend_alpha",
            type="continuous",
            domain=(0.0, 1.0),
            default=0.7,
            stage="both",
            subsystem="crossover",
            description="Blend factor: 1.0 = take fitter parent weights.",
        ),
        dispatcher=lambda ne, v: ne.set_weight_inheritance(blend_alpha=v),
    )


def _register_surrogate(reg: ParamRegistry) -> None:
    reg.register(
        ParamSpec(
            name="surrogate.enabled",
            type="boolean",
            domain=[True, False],
            default=False,
            stage="both",
            subsystem="surrogate",
            description="Enable fitness surrogate pre-filter.",
        ),
        # surrogate is set via Population.enable_surrogate / NeuroEvolution configure
        dispatcher=None,
    )
    reg.register(
        ParamSpec(
            name="surrogate.frac",
            type="continuous",
            domain=(0.05, 0.9),
            default=0.5,
            stage="both",
            subsystem="surrogate",
            description="Fraction of genomes filtered by surrogate model.",
        ),
        dispatcher=None,
    )


def _register_curiosity(reg: ParamRegistry) -> None:
    reg.register(
        ParamSpec(
            name="curiosity.enabled",
            type="boolean",
            domain=[True, False],
            default=False,
            stage="both",
            subsystem="curiosity",
            description="Enable intrinsic curiosity module.",
        ),
        dispatcher=lambda ne, v: ne.set_curiosity(enabled=v),
    )
    reg.register(
        ParamSpec(
            name="curiosity.weight",
            type="continuous",
            domain=(0.0, 2.0),
            default=0.3,
            stage="both",
            subsystem="curiosity",
            description="Weight of the curiosity bonus added to fitness.",
        ),
        dispatcher=lambda ne, v: ne.set_curiosity(weight=v),
    )


def _register_darts(reg: ParamRegistry) -> None:
    reg.register(
        ParamSpec(
            name="darts.enabled",
            type="boolean",
            domain=[True, False],
            default=False,
            stage="both",
            subsystem="darts",
            description="Enable DARTS-Lite differentiable architecture search.",
        ),
        dispatcher=lambda ne, v: ne.set_darts_mode(enabled=v),
    )
    reg.register(
        ParamSpec(
            name="darts.prune_threshold",
            type="continuous",
            domain=(0.01, 0.99),
            default=0.1,
            stage="both",
            subsystem="darts",
            description="Gate threshold for post-training DARTS pruning.",
        ),
        dispatcher=lambda ne, v: ne.set_darts_mode(prune_threshold=v),
    )


def _register_shared_weights(reg: ParamRegistry) -> None:
    reg.register(
        ParamSpec(
            name="shared_weights.enabled",
            type="boolean",
            domain=[True, False],
            default=False,
            stage="both",
            subsystem="shared_weights",
            description="Enable weight-sharing between connection groups.",
        ),
        dispatcher=lambda ne, v: ne.set_shared_weights(enabled=v),
    )


def _register_pruning(reg: ParamRegistry) -> None:
    reg.register(
        ParamSpec(
            name="pruning.enabled",
            type="boolean",
            domain=[True, False],
            default=False,
            stage="runtime",
            subsystem="pruning",
            description="Enable automatic post-training pruning.",
        ),
        dispatcher=lambda ne, v: ne.set_post_training_pruning(enabled=v),
    )
    reg.register(
        ParamSpec(
            name="pruning.threshold",
            type="continuous",
            domain=(0.0, 0.5),
            default=0.01,
            stage="runtime",
            subsystem="pruning",
            description="Weight magnitude below which connections are pruned.",
        ),
        dispatcher=lambda ne, v: ne.set_post_training_pruning(threshold=v),
    )
    reg.register(
        ParamSpec(
            name="pruning.max_drop_frac",
            type="continuous",
            domain=(0.0, 0.5),
            default=0.02,
            stage="runtime",
            subsystem="pruning",
            description="Max fitness drop fraction before pruning is rolled back.",
        ),
        dispatcher=lambda ne, v: ne.set_post_training_pruning(max_drop_frac=v),
    )


def _register_attention(reg: ParamRegistry) -> None:
    reg.register(ParamSpec(
        name="attention.enabled", type="boolean", domain=[True, False],
        default=False, stage="both", subsystem="attention",
        description="Enable evolvable attention heads on genome inputs.",
    ), dispatcher=lambda ne, v: ne.set_attention(enabled=v))
    reg.register(ParamSpec(
        name="attention.head_dim", type="integer", domain=(2, 32),
        default=4, stage="both", subsystem="attention",
        description="Dimensionality of each attention head.",
    ), dispatcher=lambda ne, v: ne.set_attention(head_dim=v))
    reg.register(ParamSpec(
        name="attention.num_heads", type="integer", domain=(1, 8),
        default=2, stage="both", subsystem="attention",
        description="Number of parallel attention heads.",
    ), dispatcher=lambda ne, v: ne.set_attention(num_heads=v))


def _register_stdp(reg: ParamRegistry) -> None:
    reg.register(ParamSpec(
        name="stdp.enabled", type="boolean", domain=[True, False],
        default=False, stage="both", subsystem="stdp",
        description="Enable Spike-Timing-Dependent Plasticity (STDP).",
    ), dispatcher=lambda ne, v: ne.set_stdp(enabled=v))
    reg.register(ParamSpec(
        name="stdp.weight_min", type="continuous", domain=(-5.0, 0.0),
        default=-1.0, stage="both", subsystem="stdp",
        description="Minimum weight allowed after STDP update.",
    ), dispatcher=lambda ne, v: ne.set_stdp(weight_min=v))
    reg.register(ParamSpec(
        name="stdp.weight_max", type="continuous", domain=(0.0, 5.0),
        default=1.0, stage="both", subsystem="stdp",
        description="Maximum weight allowed after STDP update.",
    ), dispatcher=lambda ne, v: ne.set_stdp(weight_max=v))


def _register_neuromodulation(reg: ParamRegistry) -> None:
    reg.register(ParamSpec(
        name="neuromodulation.enabled", type="boolean", domain=[True, False],
        default=False, stage="both", subsystem="neuromodulation",
        description="Enable dynamic neuromodulation of connection weights.",
    ), dispatcher=lambda ne, v: ne.set_neuromodulation(enabled=v))


def _register_ltc(reg: ParamRegistry) -> None:
    reg.register(ParamSpec(
        name="ltc.enabled", type="boolean", domain=[True, False],
        default=False, stage="both", subsystem="ltc",
        description="Enable Liquid Time-Constant ODE neuron dynamics.",
    ), dispatcher=lambda ne, v: ne.set_ltc(enabled=v))


def _register_input_grouping(reg: ParamRegistry) -> None:
    reg.register(ParamSpec(
        name="input_grouping.enabled", type="boolean", domain=[True, False],
        default=False, stage="init", subsystem="input_grouping",
        description="Enable evolvable input-aggregation groups.",
    ), dispatcher=lambda ne, v: ne.set_input_grouping(enabled=v))


def _register_output_grouping(reg: ParamRegistry) -> None:
    reg.register(ParamSpec(
        name="output_grouping.enabled", type="boolean", domain=[True, False],
        default=False, stage="init", subsystem="output_grouping",
        description="Enable evolvable output-expansion groups.",
    ), dispatcher=lambda ne, v: ne.set_output_grouping(enabled=v))


def _register_conv_neat(reg: ParamRegistry) -> None:
    reg.register(ParamSpec(
        name="conv_neat.enabled", type="boolean", domain=[True, False],
        default=False, stage="init", subsystem="conv_neat",
        description="Enable Convolutional NEAT image-preprocessing stack.",
    ), dispatcher=lambda ne, v: ne.set_conv_neat(enabled=v))
    reg.register(ParamSpec(
        name="conv_neat.n_blocks", type="integer", domain=(1, 6),
        default=2, stage="init", subsystem="conv_neat",
        description="Number of convolutional blocks in the stack.",
    ), dispatcher=lambda ne, v: ne.set_conv_neat(n_blocks=v))


def _register_hybrid_mode(reg: ParamRegistry) -> None:
    reg.register(ParamSpec(
        name="hybrid.enabled", type="boolean", domain=[True, False],
        default=False, stage="both", subsystem="hybrid",
        description="Enable Gradient-NEAT hybrid backprop interleaving.",
    ), dispatcher=lambda ne, v: ne.set_hybrid_mode(enabled=v))
    reg.register(ParamSpec(
        name="hybrid.bp_interval", type="integer", domain=(1, 100),
        default=10, stage="both", subsystem="hybrid",
        description="Generations between backprop refinement rounds.",
    ), dispatcher=lambda ne, v: ne.set_hybrid_mode(bp_interval=v))
    reg.register(ParamSpec(
        name="hybrid.bp_lr", type="continuous", domain=(1e-5, 0.1),
        default=1e-3, stage="both", subsystem="hybrid",
        description="Learning rate for Adam optimizer during backprop.",
    ), dispatcher=lambda ne, v: ne.set_hybrid_mode(bp_lr=v))


def _register_probabilistic(reg: ParamRegistry) -> None:
    reg.register(ParamSpec(
        name="probabilistic.enabled", type="boolean", domain=[True, False],
        default=False, stage="both", subsystem="probabilistic",
        description="Enable Gaussian output noise for uncertainty estimation.",
    ), dispatcher=lambda ne, v: ne.set_probabilistic(enabled=v))
    reg.register(ParamSpec(
        name="probabilistic.noise_std", type="continuous", domain=(0.001, 2.0),
        default=0.05, stage="both", subsystem="probabilistic",
        description="Standard deviation of per-output Gaussian noise.",
    ), dispatcher=lambda ne, v: ne.set_probabilistic(noise_std=v))
    reg.register(ParamSpec(
        name="probabilistic.inference_mode", type="boolean", domain=[True, False],
        default=False, stage="both", subsystem="probabilistic",
        description="When True forward() is deterministic (no noise).",
    ), dispatcher=lambda ne, v: ne.set_probabilistic(inference_mode=v))


def _register_island_model(reg: ParamRegistry) -> None:
    reg.register(ParamSpec(
        name="island.enabled", type="boolean", domain=[True, False],
        default=False, stage="both", subsystem="island",
        description="Enable multi-population island model with migration.",
    ), dispatcher=lambda ne, v: (
        ne.set_island_model(n_islands=2) if v else setattr(ne, "_island_model", None)
    ))
    reg.register(ParamSpec(
        name="island.n_islands", type="integer", domain=(2, 20),
        default=4, stage="init", subsystem="island",
        description="Number of independent island populations.",
    ), dispatcher=lambda ne, v: ne.set_island_model(n_islands=v))
    reg.register(ParamSpec(
        name="island.migrate_interval", type="integer", domain=(1, 200),
        default=20, stage="both", subsystem="island",
        description="Generations between inter-island migrations.",
    ), dispatcher=lambda ne, v: ne.set_island_model(migration_interval=v))
    reg.register(ParamSpec(
        name="island.migrate_count", type="integer", domain=(1, 20),
        default=2, stage="both", subsystem="island",
        description="Number of genomes migrating per event.",
    ), dispatcher=lambda ne, v: ne.set_island_model(migration_count=v))


def _register_augmentation(reg: ParamRegistry) -> None:
    reg.register(ParamSpec(
        name="augmentation.enabled", type="boolean", domain=[True, False],
        default=False, stage="both", subsystem="augmentation",
        description="Enable evolutionary input data augmentation.",
    ), dispatcher=lambda ne, v: ne.set_evolutionary_augmentation(enabled=v))
    reg.register(ParamSpec(
        name="augmentation.pool_size", type="integer", domain=(2, 50),
        default=5, stage="both", subsystem="augmentation",
        description="Number of augmentation pipelines in the pool.",
    ), dispatcher=lambda ne, v: ne.set_evolutionary_augmentation(population_augmentations=v))
    reg.register(ParamSpec(
        name="augmentation.evolve_interval", type="integer", domain=(1, 100),
        default=20, stage="both", subsystem="augmentation",
        description="Generations between augmentation pool evolution steps.",
    ), dispatcher=lambda ne, v: ne.set_evolutionary_augmentation(evolution_interval=v))


def _register_phylogeny(reg: ParamRegistry) -> None:
    reg.register(ParamSpec(
        name="phylogeny.enabled", type="boolean", domain=[True, False],
        default=False, stage="both", subsystem="phylogeny",
        description="Enable genome phylogeny tracking (Stammbaum).",
    ), dispatcher=lambda ne, v: (
        ne.enable_phylogeny() if v else ne.disable_phylogeny()
    ))


def _register_safety(reg: ParamRegistry) -> None:
    reg.register(ParamSpec(
        name="safety.min_safe_frac", type="continuous", domain=(0.0, 1.0),
        default=0.0, stage="both", subsystem="safety",
        description="Minimum fraction of population that must pass hard safety constraints.",
    ), dispatcher=None)  # Dispatcher requires constraint objects; set via /features/safety


def _register_misc(reg: ParamRegistry) -> None:
    reg.register(
        ParamSpec(
            name="diversity_injection.enabled",
            type="boolean",
            domain=[True, False],
            default=True,
            stage="both",
            subsystem="diversity",
            description="Enable stagnation-triggered diversity injection.",
        ),
        dispatcher=lambda ne, v: ne.set_diversity_injection(v),
    )
    reg.register(
        ParamSpec(
            name="online_tuning.enabled",
            type="boolean",
            domain=[True, False],
            default=False,
            stage="both",
            subsystem="online_tuning",
            description="Enable UCB1-bandit online hyperparameter tuning.",
        ),
        dispatcher=lambda ne, v: ne.set_online_tuning(enabled=v),
    )
    reg.register(
        ParamSpec(
            name="matrix_forward.enabled",
            type="boolean",
            domain=[True, False],
            default=False,
            stage="both",
            subsystem="evaluation",
            description="Enable matrix-accelerated forward pass.",
        ),
        dispatcher=lambda ne, v: ne.set_matrix_forward(enabled=v),
    )
