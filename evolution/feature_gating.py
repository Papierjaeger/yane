"""Feature Gating: Successive Halving + UCB1 auto-selection of research features.

Features start INACTIVE. Every ``test_interval`` generations an inactive feature
is chosen via UCB1 and enabled for ``test_duration`` generations. If the global
best-fitness improved by more than ``impact_threshold`` during the window the
feature stays ACTIVE; otherwise it is returned to INACTIVE.

NOTE: Impact is measured as a global fitness-delta proxy — not an isolated A/B
comparison. A feature that is active while fitness improves for any reason will
look beneficial. Keep evaluator behaviour stable during tests or accept the noise.

Active features are watched for degradation: if improvement stays at or below
``impact_threshold`` for ``degradation_patience`` consecutive generations
``degradation_level`` rises by 0.2.  At 0.8 the feature is disabled.

Disabled features re-enter the candidate pool automatically after
``reactivation_delay`` generations, or via :meth:`FeatureGate.reactivate`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class FeatureStatus(Enum):
    INACTIVE = "inactive"   # not active; eligible for UCB1 selection
    TESTING = "testing"     # trial window currently running
    ACTIVE = "active"       # trial showed positive impact; feature is on
    DISABLED = "disabled"   # degraded past threshold; auto-reactivation pending


@dataclass
class FeatureRecord:
    name: str
    enable_fn: Callable[[Any], None]
    disable_fn: Callable[[Any], None]
    degrade_fn: Callable[[Any, float], None] | None = None

    status: FeatureStatus = FeatureStatus.INACTIVE
    degradation_level: float = 0.0

    # UCB1 state
    n_trials: int = 0
    total_reward: float = 0.0
    last_selected_gen: int = -1

    # Trial tracking
    fitness_before: float = field(default=-float("inf"))
    test_start_gen: int = -1
    test_end_gen: int = -1

    # Degradation tracking (while ACTIVE)
    active_since_gen: int = -1
    stagnation_count: int = 0  # consecutive gens with improvement <= impact_threshold

    # Reactivation
    disabled_at_gen: int = -1

    # History (recent impacts)
    impact_history: list = field(default_factory=list)

    def ucb1_score(self, total_selections: int, c: float = 2.0) -> float:
        from yane.evolution.online_tuning import ucb1_score as _ucb1
        mean = self.total_reward / self.n_trials if self.n_trials > 0 else 0.0
        return _ucb1(mean, self.n_trials, total_selections, c=c)

    def update_ucb1(self, reward: float) -> None:
        self.n_trials += 1
        self.total_reward += reward


class FeatureGate:
    """Automatic research-feature selection via Successive Halving + UCB1."""

    def __init__(
        self,
        max_concurrent: int = 3,
        test_interval: int = 50,
        test_duration: int | None = None,
        impact_threshold: float = 0.0,
        degradation_patience: int = 30,
        reactivation_delay: int = 100,
        c_ucb1: float = 2.0,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.test_interval = test_interval
        self.test_duration = (
            test_duration if test_duration is not None else max(10, test_interval // 2)
        )
        self.impact_threshold = impact_threshold
        self.degradation_patience = degradation_patience
        self.reactivation_delay = reactivation_delay
        self.c_ucb1 = c_ucb1

        self._features: dict[str, FeatureRecord] = {}
        self._total_selections: int = 0
        self._last_test_gen: int = -test_interval   # first test fires at gen 0
        self._prev_best_fitness: float = -float("inf")
        self._change_log: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        enable_fn: Callable[[Any], None],
        disable_fn: Callable[[Any], None],
        degrade_fn: Callable[[Any, float], None] | None = None,
    ) -> None:
        """Register a research feature.

        ``enable_fn(ne)``  — called when a trial starts.
        ``disable_fn(ne)`` — called when a trial ends with no impact, or when
                             degradation reaches 0.8.
        ``degrade_fn(ne, degradation_level)`` — optional; called each time
                             degradation_level increases (gradual wind-down).
        """
        if name in self._features:
            raise ValueError(f"Feature {name!r} is already registered.")
        self._features[name] = FeatureRecord(
            name=name,
            enable_fn=enable_fn,
            disable_fn=disable_fn,
            degrade_fn=degrade_fn,
        )

    def tick(self, generation: int, best_fitness: float, ne: Any) -> None:
        """Advance the gating logic by one generation.

        Called once per generation by :meth:`NeuroEvolution._tick_feature_gating`.
        """
        if not self._features:
            return

        improvement = max(0.0, best_fitness - self._prev_best_fitness)
        self._prev_best_fitness = best_fitness

        self._check_reactivation(generation)

        for rec in list(self._features.values()):
            if rec.status == FeatureStatus.TESTING and generation >= rec.test_end_gen:
                self._finish_test(rec, generation, best_fitness, ne)

        self._update_degradation(generation, improvement, ne)

        n_occupied = self._n_active() + self._n_testing()
        slots = self.max_concurrent - n_occupied
        if slots > 0 and (generation - self._last_test_gen) >= self.test_interval:
            candidate = self._select_candidate()
            if candidate is not None:
                self._start_test(candidate, generation, best_fitness, ne)
                self._last_test_gen = generation

    def reactivate(self, name: str) -> None:
        """Force a DISABLED feature back to INACTIVE so UCB1 can re-select it."""
        if name not in self._features:
            raise ValueError(f"Unknown feature: {name!r}")
        rec = self._features[name]
        if rec.status == FeatureStatus.DISABLED:
            rec.status = FeatureStatus.INACTIVE
            rec.degradation_level = 0.0
            rec.stagnation_count = 0
            rec.disabled_at_gen = -1

    def pre_activate(self, name: str, ne: Any) -> bool:
        """Activate a feature immediately, bypassing the test window.

        Used when the problem profiler identifies that a feature is very likely
        beneficial (e.g. curiosity for sparse-reward tasks) and waiting for
        UCB1 selection would waste dozens of generations.

        The feature is marked ACTIVE so FeatureGating skips testing it and
        goes straight to degradation monitoring — it will still be disabled
        automatically if it hurts performance.

        Returns True if the feature was activated, False if unknown or already
        ACTIVE/TESTING.
        """
        if name not in self._features:
            return False
        rec = self._features[name]
        if rec.status not in (FeatureStatus.INACTIVE, FeatureStatus.DISABLED):
            return False
        try:
            rec.enable_fn(ne)
        except Exception:
            return False
        rec.status = FeatureStatus.ACTIVE
        rec.active_since_gen = 0
        rec.stagnation_count = 0
        rec.degradation_level = 0.0
        rec.disabled_at_gen = -1
        self._log("active", rec, 0, "pre-activated by profiler")
        return True

    def get_active_features(self) -> list[str]:
        return [n for n, r in self._features.items() if r.status == FeatureStatus.ACTIVE]

    def get_diagnostics(self) -> dict:
        features = {}
        for name, rec in self._features.items():
            features[name] = {
                "status": rec.status.value,
                "degradation_level": rec.degradation_level,
                "n_trials": rec.n_trials,
                "mean_reward": (
                    rec.total_reward / rec.n_trials if rec.n_trials > 0 else 0.0
                ),
                "stagnation_count": rec.stagnation_count,
            }
        return {
            "enabled": True,
            "n_active": self._n_active(),
            "n_testing": self._n_testing(),
            "max_concurrent": self.max_concurrent,
            "features": features,
            "change_log": self._change_log[-10:],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_reactivation(self, generation: int) -> None:
        for rec in self._features.values():
            if (
                rec.status == FeatureStatus.DISABLED
                and rec.disabled_at_gen >= 0
                and (generation - rec.disabled_at_gen) >= self.reactivation_delay
            ):
                rec.status = FeatureStatus.INACTIVE
                rec.degradation_level = 0.0
                rec.stagnation_count = 0
                # Reset interval clock so a new test doesn't fire in the same tick
                self._last_test_gen = generation
                self._log("inactive", rec, generation, "auto-reactivated after delay")

    def _select_candidate(self) -> FeatureRecord | None:
        candidates = [
            r for r in self._features.values()
            if r.status == FeatureStatus.INACTIVE
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.ucb1_score(self._total_selections, self.c_ucb1))

    def _start_test(
        self, rec: FeatureRecord, generation: int, best_fitness: float, ne: Any
    ) -> None:
        rec.status = FeatureStatus.TESTING
        rec.fitness_before = best_fitness
        rec.test_start_gen = generation
        rec.test_end_gen = generation + self.test_duration
        rec.last_selected_gen = generation
        self._total_selections += 1
        self._log("testing", rec, generation, "trial started")
        try:
            rec.enable_fn(ne)
        except Exception:
            pass  # enable failure must not stop training

    def _finish_test(
        self, rec: FeatureRecord, generation: int, best_fitness: float, ne: Any
    ) -> None:
        # Reset interval clock on test finish so next test waits a full test_interval
        self._last_test_gen = generation
        impact = best_fitness - rec.fitness_before
        if impact > self.impact_threshold:
            # Positive impact → keep active
            reward = min(1.0, max(0.0, impact / (abs(rec.fitness_before) + 1.0)))
            rec.update_ucb1(reward)
            rec.status = FeatureStatus.ACTIVE
            rec.active_since_gen = generation
            rec.stagnation_count = 0
            rec.impact_history.append(impact)
            if len(rec.impact_history) > 50:
                rec.impact_history = rec.impact_history[-50:]
            self._log("active", rec, generation, f"impact={impact:.4f}")
        else:
            # No impact → return to candidate pool
            rec.update_ucb1(0.0)
            rec.status = FeatureStatus.INACTIVE
            self._log("inactive", rec, generation, f"no impact (delta={impact:.4f})")
            try:
                rec.disable_fn(ne)
            except Exception:
                pass

    def _update_degradation(
        self, generation: int, improvement: float, ne: Any
    ) -> None:
        for rec in list(self._features.values()):
            if rec.status != FeatureStatus.ACTIVE:
                continue

            if improvement <= self.impact_threshold:
                rec.stagnation_count += 1
            else:
                rec.stagnation_count = 0

            if rec.stagnation_count >= self.degradation_patience:
                rec.stagnation_count = 0
                rec.degradation_level = min(1.0, rec.degradation_level + 0.2)
                self._log(
                    "active", rec, generation,
                    f"degradation_level={rec.degradation_level:.1f}",
                )
                if rec.degrade_fn is not None:
                    try:
                        rec.degrade_fn(ne, rec.degradation_level)
                    except Exception:
                        pass

                if rec.degradation_level >= 0.8:
                    rec.status = FeatureStatus.DISABLED
                    rec.disabled_at_gen = generation
                    self._log("disabled", rec, generation, "degradation >= 0.8")
                    try:
                        rec.disable_fn(ne)
                    except Exception:
                        pass

    def _n_active(self) -> int:
        return sum(1 for r in self._features.values() if r.status == FeatureStatus.ACTIVE)

    def _n_testing(self) -> int:
        return sum(1 for r in self._features.values() if r.status == FeatureStatus.TESTING)

    def _log(
        self, to_status: str, rec: FeatureRecord, generation: int, reason: str
    ) -> None:
        self._change_log.append({
            "generation": generation,
            "feature": rec.name,
            "to": to_status,
            "reason": reason,
        })
        if len(self._change_log) > 200:
            self._change_log = self._change_log[-200:]


# ------------------------------------------------------------------
# Default feature registration for NeuroEvolution
# ------------------------------------------------------------------

def _register_known_features(fg: FeatureGate, ne: Any, *, include_heavyweight: bool = False) -> None:
    """Register runtime-togglable research features into *fg*.

    Parameters
    ----------
    include_heavyweight:
        When ``False`` (default), only lightweight features are registered —
        those with negligible per-evaluation overhead.  Heavyweight features
        (attention, STDP, neuromodulation, LTC, probabilistic, augmentation)
        add per-sample computation and can multiply eval time by 10-100× on
        large datasets.  Set ``True`` only when the problem is known to be
        compatible (e.g. temporal tasks where STDP is useful).

    Init-stage features (input_grouping, output_grouping, conv_neat) are
    never included; they require a full reconfigure().
    """
    # ── Original 5 ────────────────────────────────────────────────────────────
    fg.register(
        "curiosity",
        enable_fn=lambda _ne: _ne.set_curiosity(enabled=True),
        disable_fn=lambda _ne: _ne.set_curiosity(enabled=False),
    )
    fg.register(
        "darts",
        enable_fn=lambda _ne: _ne.set_darts_mode(enabled=True),
        disable_fn=lambda _ne: _ne.set_darts_mode(enabled=False),
    )
    fg.register(
        "shared_weights",
        enable_fn=lambda _ne: _ne.set_shared_weights(enabled=True),
        disable_fn=lambda _ne: _ne.set_shared_weights(enabled=False),
    )
    fg.register(
        "novelty_search",
        enable_fn=lambda _ne: _ne.set_novelty_search(enabled=True),
        disable_fn=lambda _ne: _ne.set_novelty_search(enabled=False),
    )
    fg.register(
        "diversity_injection",
        enable_fn=lambda _ne: _ne.set_diversity_injection(enabled=True),
        disable_fn=lambda _ne: _ne.set_diversity_injection(enabled=False),
    )

    # ── Heavyweight features (opt-in; can multiply per-eval cost significantly) ──
    # STDP/Neuromodulation: update weights after every forward pass × every sample.
    # Attention: adds matrix multiply of (num_heads × head_dim × n_inputs) per eval.
    # LTC: runs ODE step after each forward call.
    # Augmentation: runs augmentation pipeline per eval.
    # On large datasets (300+ samples) these can cause 10-100× eval slowdown.
    if include_heavyweight:
        fg.register(
            "attention",
            enable_fn=lambda _ne: _ne.set_attention(enabled=True),
            disable_fn=lambda _ne: _ne.set_attention(enabled=False),
        )
        fg.register(
            "ltc",
            enable_fn=lambda _ne: _ne.set_ltc(enabled=True),
            disable_fn=lambda _ne: _ne.set_ltc(enabled=False),
        )
        fg.register(
            "neuromodulation",
            enable_fn=lambda _ne: _ne.set_neuromodulation(enabled=True),
            disable_fn=lambda _ne: _ne.set_neuromodulation(enabled=False),
        )
        fg.register(
            "stdp",
            enable_fn=lambda _ne: _ne.set_stdp(enabled=True),
            disable_fn=lambda _ne: _ne.set_stdp(enabled=False),
        )
        fg.register(
            "probabilistic",
            enable_fn=lambda _ne: _ne.set_probabilistic(enabled=True),
            disable_fn=lambda _ne: _ne.set_probabilistic(enabled=False),
        )
        fg.register(
            "augmentation",
            enable_fn=lambda _ne: _ne.set_evolutionary_augmentation(enabled=True),
            disable_fn=lambda _ne: _ne.set_evolutionary_augmentation(enabled=False),
        )

    # ── Evaluation ─────────────────────────────────────────────────────────────
    fg.register(
        "anytime_eval",
        enable_fn=lambda _ne: _ne.set_anytime_eval(enabled=True),
        disable_fn=lambda _ne: _ne.set_anytime_eval(enabled=False),
    )
    fg.register(
        "adaptive_recovery",
        enable_fn=lambda _ne: _ne.set_adaptive_recovery(enabled=True),
        disable_fn=lambda _ne: _ne.set_adaptive_recovery(enabled=False),
    )
    fg.register(
        "pruning",
        enable_fn=lambda _ne: _ne.set_post_training_pruning(enabled=True),
        disable_fn=lambda _ne: _ne.set_post_training_pruning(enabled=False),
    )

    # ── Deployment ─────────────────────────────────────────────────────────────
    fg.register(
        "surrogate",
        enable_fn=lambda _ne: _ne.set_param("surrogate.enabled", True),
        disable_fn=lambda _ne: _ne.set_param("surrogate.enabled", False),
    )
    fg.register(
        "online_tuning",
        enable_fn=lambda _ne: _ne.set_online_tuning(enabled=True),
        disable_fn=lambda _ne: _ne.set_online_tuning(enabled=False),
    )

    # ── Analysis ───────────────────────────────────────────────────────────────
    fg.register(
        "phylogeny",
        enable_fn=lambda _ne: _ne.enable_phylogeny(),
        disable_fn=lambda _ne: _ne.disable_phylogeny(),
    )

    # ── Island Model — graceful degradation via merge_weakest_island() ─────────
    def _island_enable(ne: Any) -> None:
        """Start with 4 islands (or 2 if population is very small)."""
        n = 4 if (ne._population_size or 100) >= 40 else 2
        try:
            ne.set_island_model(n_islands=n)
        except Exception:
            pass  # e.g. not yet configured — silently skip

    def _island_degrade(ne: Any, level: float) -> None:
        """At each degradation step, eliminate the weakest island.

        degradation_level 0.2 → merge to N-1 islands
        degradation_level 0.4 → merge to N-2 islands
        ...until only 1 island remains, then the disable_fn cleans up.
        """
        im = getattr(ne, "_island_model", None)
        if im is None:
            return
        merged = im.merge_weakest_island()
        if merged and im.n_islands >= 1:
            # Keep ne._population pointing to the island with the best genome
            ne._population = max(im.islands, key=lambda p: (
                p.get_best().fitness if p._evaluated else -float("inf")
            ))
            if im.n_islands == 1:
                # Effectively a single population now — remove the wrapper
                ne._island_model = None

    def _island_disable(ne: Any) -> None:
        """Final cleanup: ensure _island_model is gone."""
        im = getattr(ne, "_island_model", None)
        if im is not None and im.islands:
            # Rescue the best genome across all remaining islands
            best_pop = max(im.islands, key=lambda p: (
                p.get_best().fitness if p._evaluated else -float("inf")
            ))
            ne._population = best_pop
        ne._island_model = None

    fg.register(
        "island_model",
        enable_fn=_island_enable,
        disable_fn=_island_disable,
        degrade_fn=_island_degrade,
    )
