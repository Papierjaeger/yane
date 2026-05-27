"""Adaptive Policy System — unified interface for all adaptive mechanisms.

Policies replace ad-hoc control loops with a standard observe/decide/apply
lifecycle, conflict resolution via priority + conflict_group, and a central
registry with configurable evaluation order.

Usage::

    class MyPolicy:
        name = "my_policy"

        def observe(self, ctx: TrainingContext) -> None:
            ...

        def decide(self, ctx: TrainingContext) -> Action | None:
            if some_condition:
                return Action("adjust_X", value=0.5, priority=10)
            return None

        def apply(self, ctx: TrainingContext, action: Action) -> None:
            ...

    ne.register_policy(MyPolicy())
    ne.set_policy_order(["my_policy", "recovery", "online_tuning"])
"""
from __future__ import annotations

import dataclasses
from collections import OrderedDict
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Action:
    """A typed action proposed by a policy.

    Attributes
    ----------
    name : str
        Human-readable action name (e.g. ``"diversity_boost"``).
    priority : int
        Higher priority wins when two actions share the same conflict_group.
    conflict_group : str
        Actions with the same group exclude each other; only the highest
        priority action in a group is applied.
    payload : dict
        Arbitrary parameters for the action (e.g. ``{"injection_frac": 0.1}``).
    """
    name: str = ""
    priority: int = 0
    conflict_group: str = ""
    payload: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TrainingContext:
    """Snapshot of training state passed to every policy at each tick.

    All fields are read-only — policies should not modify the context.
    """
    generation: int = 0
    iteration: int = 0
    best_fitness: float = -float("inf")
    mean_fitness: float = 0.0
    median_fitness: float = 0.0
    fitness_iqr: float = 0.0
    species_count: int = 0
    stagnation_count: int = 0
    n_evaluations: int = 0
    max_iterations: int = 0
    # Anomaly signals (populated by AnomalyDetectorSet)
    anomalies: list[str] = dataclasses.field(default_factory=list)
    # Recovery diagnostics
    recovery_events: list[dict] = dataclasses.field(default_factory=list)
    stopped_early: bool = False
    # Fitness transform info
    fitness_transform_name: str | None = None
    # Policy diagnostics (populated by the registry)
    last_policy_actions: dict[str, Action] = dataclasses.field(default_factory=dict)
    active_policies: list[str] = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# Policy protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class AdaptivePolicy(Protocol):
    """Interface for pluggable adaptive policies.

    Each policy must expose a ``name`` attribute (used for registration and
    ordering) and implement ``observe``, ``decide``, and ``apply``.
    """

    name: str

    def observe(self, ctx: TrainingContext) -> None:
        """Read training state and update internal statistics.

        Called for every policy before any ``decide()`` call.
        """
        ...

    def decide(self, ctx: TrainingContext) -> Action | None:
        """Examine state and propose an action, or return None.

        Called in policy order.  Actions are collected, conflicts resolved,
        then ``apply()`` is called for the winning actions.
        """
        ...

    def apply(self, ctx: TrainingContext, action: Action) -> None:
        """Execute a previously proposed action.

        Only called for the winning action per conflict_group.
        """
        ...


# ---------------------------------------------------------------------------
# PolicyRegistry — central coordinator
# ---------------------------------------------------------------------------

class PolicyRegistry:
    """Manages policy registration, ordering, conflict resolution, and ticks."""

    def __init__(self) -> None:
        self._policies: dict[str, AdaptivePolicy] = OrderedDict()
        self._enabled: dict[str, bool] = {}
        self._order: list[str] = []
        self._last_actions: dict[str, Action] = {}
        self._rewards: dict[str, list[float]] = {}
        self._conflicts: list[dict] = []
        self._last_best: float = -float("inf")

    def register(self, policy: AdaptivePolicy, enabled: bool = True) -> None:
        """Register a policy.

        If the policy name is already registered, it is replaced.
        """
        name = getattr(policy, "name", type(policy).__name__)
        self._policies[name] = policy
        self._enabled[name] = enabled
        if name not in self._order:
            self._order.append(name)

    def set_order(self, names: list[str]) -> None:
        """Set the evaluation order for all policies.

        Only names that are registered are kept; unknown names are ignored.
        Registered names not in the list are appended at the end.
        """
        ordered: list[str] = []
        for name in names:
            if name in self._policies and name not in ordered:
                ordered.append(name)
        for name in self._policies:
            if name not in ordered:
                ordered.append(name)
        self._order = ordered

    def enable(self, name: str, enabled: bool = True) -> None:
        """Enable or disable a registered policy."""
        if name in self._enabled:
            self._enabled[name] = enabled

    def tick(self, ctx: TrainingContext) -> list[Action]:
        """Run the full observe → decide → resolve → apply cycle.

        Returns the list of applied actions.
        """
        ctx.active_policies = [
            name for name in self._order
            if name in self._enabled and self._enabled[name]
        ]
        ctx.last_policy_actions = dict(self._last_actions)

        # Phase 1: observe all
        for name in ctx.active_policies:
            policy = self._policies[name]
            try:
                policy.observe(ctx)
            except Exception:
                pass

        # Phase 2: decide all → collect actions
        proposed: list[tuple[str, Action]] = []
        for name in ctx.active_policies:
            policy = self._policies[name]
            try:
                action = policy.decide(ctx)
                if action is not None:
                    proposed.append((name, action))
            except Exception:
                pass

        # Phase 3: conflict resolution (by conflict_group, highest priority wins)
        groups: dict[str, tuple[str, Action]] = {}
        for name, action in proposed:
            cg = action.conflict_group or name
            if cg in groups:
                existing_name, existing_action = groups[cg]
                if action.priority > existing_action.priority:
                    groups[cg] = (name, action)
                    self._conflicts.append({
                        "group": cg,
                        "winner": name,
                        "loser": existing_name,
                        "action": action.name,
                    })
            else:
                groups[cg] = (name, action)

        # Phase 4: apply winning actions
        applied: list[Action] = []
        for name, action in groups.values():
            try:
                policy = self._policies[name]
                policy.apply(ctx, action)
                applied.append(action)
                self._last_actions[name] = action
            except Exception:
                pass

        # Track rewards (fitness delta)
        if ctx.best_fitness > self._last_best:
            delta = ctx.best_fitness - self._last_best
            for name in self._last_actions:
                self._rewards.setdefault(name, []).append(delta)
        self._last_best = ctx.best_fitness

        return applied

    @property
    def active_policy_names(self) -> list[str]:
        return [n for n in self._order if self._enabled.get(n, False)]

    def get_diagnostics(self) -> dict:
        return {
            "active_policies": self.active_policy_names,
            "policy_order": list(self._order),
            "policy_enabled": dict(self._enabled),
            "last_policy_actions": {
                k: dataclasses.asdict(v) for k, v in self._last_actions.items()
            },
            "policy_rewards": {
                k: [round(v, 6) for v in vals][-10:]
                for k, vals in self._rewards.items()
            },
            "policy_conflicts": self._conflicts[-10:],
        }
