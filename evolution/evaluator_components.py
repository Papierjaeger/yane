"""Reusable evaluator components for shaped, case-based fitness functions."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from yane.core.genome import Genome


@dataclass(frozen=True)
class StateEncoder:
    """Encode discrete grid/task states into neural-network inputs."""

    mode: str = "scaled"
    sizes: tuple[int, ...] = ()
    scales: tuple[float, ...] = ()

    def encode(self, values: Sequence[int | float]) -> list[float]:
        if self.mode == "scaled":
            if self.scales:
                return [
                    float(v) / scale if scale else float(v)
                    for v, scale in zip(values, self.scales)
                ]
            return [float(v) for v in values]
        if self.mode == "one_hot":
            return self._one_hot(values)
        if self.mode == "mixed":
            encoded = self._one_hot(values)
            encoded.extend(
                float(v) / scale if scale else float(v)
                for v, scale in zip(values, self.scales)
            )
            return encoded
        raise ValueError(f"Unknown state encoder mode: {self.mode!r}")

    @property
    def output_dim(self) -> int:
        if self.mode == "scaled":
            return len(self.scales) if self.scales else len(self.sizes)
        if self.mode == "one_hot":
            return sum(self.sizes)
        if self.mode == "mixed":
            return sum(self.sizes) + (len(self.scales) if self.scales else len(self.sizes))
        raise ValueError(f"Unknown state encoder mode: {self.mode!r}")

    def _one_hot(self, values: Sequence[int | float]) -> list[float]:
        if len(values) != len(self.sizes):
            raise ValueError(f"Expected {len(self.sizes)} values, got {len(values)}")
        out: list[float] = []
        for value, size in zip(values, self.sizes):
            idx = int(value)
            out.extend(1.0 if i == idx else 0.0 for i in range(size))
        return out


@dataclass(frozen=True)
class SubgoalReward:
    """Distance-progress reward toward a dynamic subgoal."""

    subgoal_fn: Callable[[Any], Any]
    distance_fn: Callable[[Any, Any], float]
    progress_weight: float = 1.0
    distance_penalty: float = 0.0

    def score(self, previous_state: Any, next_state: Any) -> float:
        goal = self.subgoal_fn(previous_state)
        prev_d = self.distance_fn(previous_state, goal)
        next_d = self.distance_fn(next_state, goal)
        return self.progress_weight * (prev_d - next_d) - self.distance_penalty * next_d


@dataclass
class GraphPolicyScore:
    """Local action-quality score for deterministic Gym-style discrete MDPs."""

    transitions: dict
    n_actions: int
    terminal_reward: float = 0.0
    goal_score: float = 6.0
    improvement_score: float = 3.0
    equal_score: float = -0.25
    worse_score: float = -1.5
    illegal_score: float = -3.0
    distances: dict[int, float] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        states = list(self.transitions.keys())
        dist = {int(state): float("inf") for state in states}
        changed = True
        while changed:
            changed = False
            for state in states:
                best = dist[state]
                for action in range(self.n_actions):
                    _prob, next_state, reward, done = self.transitions[state][action][0]
                    candidate = 0.0 if done and reward > self.terminal_reward else dist[int(next_state)] + 1.0
                    if candidate < best:
                        best = candidate
                if best < dist[state]:
                    dist[state] = best
                    changed = True
        self.distances = dist

    def action_score(self, state: int, action: int) -> float:
        _prob, next_state, reward, done = self.transitions[state][action][0]
        if done and reward > self.terminal_reward:
            return self.goal_score
        if reward < -9:
            return self.illegal_score
        cur = self.distances.get(int(state), float("inf"))
        nxt = self.distances.get(int(next_state), float("inf"))
        if nxt < cur:
            return self.improvement_score
        if nxt == cur:
            return self.equal_score
        return self.worse_score

    def score_policy(
        self,
        genome: Genome,
        states: Iterable[int],
        encode_state: Callable[[int], list[float]],
    ) -> float:
        total = 0.0
        count = 0
        for state in states:
            out = genome.forward(encode_state(state))
            action = out.index(max(out))
            total += self.action_score(int(state), action)
            count += 1
        return total / max(1, count)


@dataclass(frozen=True)
class MultiStartRollout:
    """Aggregate a rollout function over fixed start cases."""

    cases: tuple[Any, ...]
    rollout_fn: Callable[[Genome, Any], float]
    aggregation: str = "mean"

    def evaluate(self, genome: Genome) -> float:
        values = [self.rollout_fn(genome, case) for case in self.cases]
        if not values:
            return 0.0
        if self.aggregation == "min":
            return min(values)
        if self.aggregation == "max":
            return max(values)
        return sum(values) / len(values)


@dataclass
class EvaluatorSpec:
    """Composable evaluator specification with named weighted components."""

    state_encoder: StateEncoder | None = None
    rollout_cases: tuple[Any, ...] = ()
    component_weights: dict[str, float] = field(default_factory=dict)
    target_fitness: float | None = None
    validation_cases: tuple[Any, ...] = ()
    enabled_components: frozenset[str] | None = None

    def combine(self, components: dict[str, float]) -> float:
        active = (
            {k: v for k, v in components.items() if k in self.enabled_components}
            if self.enabled_components is not None
            else components
        )
        if not self.component_weights:
            return sum(active.values())
        return sum(
            active.get(name, 0.0) * weight
            for name, weight in self.component_weights.items()
            if self.enabled_components is None or name in self.enabled_components
        )

    def with_enabled(self, enabled: frozenset[str] | None) -> "EvaluatorSpec":
        """Return a copy with the given enabled_components set."""
        from dataclasses import replace
        return replace(self, enabled_components=enabled)
