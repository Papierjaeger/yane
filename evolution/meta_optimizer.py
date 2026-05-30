"""MetaOptimizer — Phase 3 of P0 Meta-Adaptive Orchestration.

Continuously tunes YANE's hyperparameters during training using three
complementary strategies:

    a) UCB1-Bandit for categorical parameters (e.g. ``lamarck.mode``).
    b) Bayesian Optimisation (simple GP + EI) for continuous parameters
       (e.g. ``anytime.promotion_frac``, ``lamarck.sigma``).
       Falls back to guided random search when the GP cannot be fit.
    c) Data-driven training phases (EXPLORE → EXPLOIT → REFINE → CONVERGE)
       that transition when a fitness plateau is detected.

Usage::

    yane.configure(n_inputs=4, n_outputs=2)
    yane.set_meta_optimizer(enabled=True, tune_interval=20)
    yane.train(evaluator)          # MetaOptimizer ticks each generation
    print(yane.get_meta_optimizer_diagnostics())
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from yane.neuro_evolution import NeuroEvolution


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------

class TrainingPhase(Enum):
    EXPLORE = "explore"
    EXPLOIT = "exploit"
    REFINE = "refine"
    CONVERGE = "converge"


_PHASE_ORDER = [
    TrainingPhase.EXPLORE,
    TrainingPhase.EXPLOIT,
    TrainingPhase.REFINE,
    TrainingPhase.CONVERGE,
]

# Actions applied at each phase TRANSITION (not every tick).
# Only non-destructive enablements; never disables what the user set.
_PHASE_ACTIONS: dict[TrainingPhase, dict[str, Any]] = {
    TrainingPhase.EXPLORE: {
        # No automatic enables in Explore — just observe and gather data.
    },
    TrainingPhase.EXPLOIT: {
        "anytime.enabled": True,
        "lamarck.enabled": True,
    },
    TrainingPhase.REFINE: {
        "anytime.enabled": True,
        "lamarck.enabled": True,
        "recovery.enabled": True,
    },
    TrainingPhase.CONVERGE: {
        "anytime.enabled": True,
        "lamarck.enabled": True,
        "recovery.enabled": True,
    },
}

# Minimum generations before a plateau is declared (avoids premature transitions)
_PLATEAU_PATIENCE = 30
# Minimum generations before any phase transition is considered
_PHASE_MIN_GENS = 20


# ---------------------------------------------------------------------------
# Simple GP (Bayesian Optimisation, ~200 LOC as spec states)
# ---------------------------------------------------------------------------

def _solve_linear(A: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting.  Solves Ax = b in-place."""
    n = len(b)
    # Build augmented matrix
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        # Partial pivot
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[pivot] = M[pivot], M[col]
        if abs(M[col][col]) < 1e-12:
            continue
        for row in range(col + 1, n):
            f = M[row][col] / M[col][col]
            M[row] = [M[row][j] - f * M[col][j] for j in range(n + 1)]
    # Back-substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(M[i][i]) < 1e-12:
            x[i] = 0.0
        else:
            x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def _normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _normal_cdf(z: float) -> float:
    """Standard normal CDF via Abramowitz & Stegun approximation."""
    if z < 0:
        return 1.0 - _normal_cdf(-z)
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = t * (
        0.319381530
        + t * (-0.356563782
        + t * (1.781477937
        + t * (-1.821255978
        + t * 1.330274429)))
    )
    return 1.0 - _normal_pdf(z) * poly


class SimpleGP:
    """Minimal Gaussian Process with RBF kernel for 1-D Bayesian Optimisation.

    Uses Expected Improvement as the acquisition function.  Falls back
    gracefully when fewer than 2 observations are available.
    """

    def __init__(self, kernel_scale: float = 0.3, noise: float = 0.01) -> None:
        self.kernel_scale = kernel_scale
        self.noise = noise
        self._xs: list[float] = []
        self._ys: list[float] = []

    def observe(self, x: float, y: float) -> None:
        self._xs.append(float(x))
        self._ys.append(float(y))

    def _rbf(self, x1: float, x2: float) -> float:
        return math.exp(-((x1 - x2) ** 2) / (2 * self.kernel_scale ** 2))

    def predict(self, x: float) -> tuple[float, float]:
        """Return (mean, std) at point x."""
        n = len(self._xs)
        if n == 0:
            return 0.0, 1.0
        K = [[self._rbf(self._xs[i], self._xs[j]) for j in range(n)] for i in range(n)]
        for i in range(n):
            K[i][i] += self.noise
        k_star = [self._rbf(x, xi) for xi in self._xs]
        alpha = _solve_linear(K, list(self._ys))
        mu = sum(k * a for k, a in zip(k_star, alpha))
        Kss = self._rbf(x, x)
        v = _solve_linear(K, k_star)
        var = max(0.0, Kss - sum(vi * ki for vi, ki in zip(v, k_star)))
        return mu, math.sqrt(max(0.0, var))

    def expected_improvement(self, x: float, best_y: float, xi: float = 0.01) -> float:
        """EI acquisition function."""
        if len(self._xs) < 2:
            return random.random()  # pure exploration when no data
        mu, sigma = self.predict(x)
        if sigma < 1e-8:
            return 0.0
        Z = (mu - best_y - xi) / sigma
        return (mu - best_y - xi) * _normal_cdf(Z) + sigma * _normal_pdf(Z)

    def suggest(
        self,
        lo: float,
        hi: float,
        n_candidates: int = 20,
    ) -> float:
        """Return the x in [lo, hi] with the highest EI."""
        if not self._xs:
            return lo + random.random() * (hi - lo)
        best_y = max(self._ys)
        candidates = [lo + (hi - lo) * i / max(n_candidates - 1, 1)
                      for i in range(n_candidates)]
        # Add random candidates
        candidates += [lo + random.random() * (hi - lo)
                       for _ in range(n_candidates // 2)]
        best_x = candidates[0]
        best_ei = -1.0
        for c in candidates:
            ei = self.expected_improvement(c, best_y)
            if ei > best_ei:
                best_ei = ei
                best_x = c
        return max(lo, min(hi, best_x))


# ---------------------------------------------------------------------------
# Phase manager (data-driven transitions)
# ---------------------------------------------------------------------------

class PhaseManager:
    """Tracks training progress and decides when to advance to the next phase."""

    def __init__(
        self,
        plateau_patience: int = _PLATEAU_PATIENCE,
        phase_min_gens: int = _PHASE_MIN_GENS,
    ) -> None:
        self._plateau_patience = plateau_patience
        self._phase_min_gens = phase_min_gens
        self._phase = TrainingPhase.EXPLORE
        self._phase_gen: int = 0           # generation when current phase started
        self._best_in_phase: float = -math.inf
        self._stagnation: int = 0
        self._transitions: list[dict] = []

    @property
    def current_phase(self) -> TrainingPhase:
        return self._phase

    def update(self, generation: int, best_fitness: float) -> bool:
        """Update state; return True if a phase transition occurred."""
        if best_fitness > self._best_in_phase + 1e-8:
            self._best_in_phase = best_fitness
            self._stagnation = 0
        else:
            self._stagnation += 1

        elapsed = generation - self._phase_gen
        plateau = self._stagnation >= self._plateau_patience
        min_met = elapsed >= self._phase_min_gens

        if plateau and min_met and self._phase != TrainingPhase.CONVERGE:
            return self._advance(generation, best_fitness)
        return False

    def _advance(self, generation: int, best_fitness: float) -> bool:
        idx = _PHASE_ORDER.index(self._phase)
        if idx + 1 >= len(_PHASE_ORDER):
            return False
        old = self._phase
        self._phase = _PHASE_ORDER[idx + 1]
        self._phase_gen = generation
        self._best_in_phase = best_fitness
        self._stagnation = 0
        self._transitions.append({
            "from": old.value,
            "to": self._phase.value,
            "generation": generation,
        })
        return True

    def get_diagnostics(self) -> dict:
        return {
            "current_phase": self._phase.value,
            "stagnation_gens": self._stagnation,
            "phase_elapsed_gens": 0,  # updated externally
            "transitions": list(self._transitions),
        }


# ---------------------------------------------------------------------------
# Categorical tuner — UCB1 bandit per parameter
# ---------------------------------------------------------------------------

class _CatBandit:
    """UCB1 bandit for a single categorical YANE parameter."""

    def __init__(
        self,
        param_name: str,
        candidates: list,
        c: float = 2.0,
    ) -> None:
        self.param_name = param_name
        self.candidates = list(candidates)
        self.c = c
        self._counts = [0] * len(candidates)
        self._rewards = [0.0] * len(candidates)
        self._total = 0
        self._last_idx: int | None = None

    def select(self) -> Any:
        from yane.evolution.online_tuning import ucb1_score as _ucb1
        self._total += 1
        if 0 in self._counts:
            # Ensure each arm tried at least once
            idx = self._counts.index(0)
        else:
            scores = [
                _ucb1(
                    self._rewards[i] / self._counts[i],
                    self._counts[i],
                    self._total,
                    c=self.c,
                )
                for i in range(len(self.candidates))
            ]
            idx = max(range(len(scores)), key=lambda i: scores[i])
        self._counts[idx] += 1
        self._last_idx = idx
        return self.candidates[idx]

    def update(self, reward: float) -> None:
        if self._last_idx is not None:
            self._rewards[self._last_idx] += max(0.0, reward)

    def best(self) -> Any:
        if not any(self._counts):
            return self.candidates[0]
        idx = max(range(len(self._rewards)),
                  key=lambda i: self._rewards[i] / max(self._counts[i], 1))
        self._last_idx = idx
        return self.candidates[idx]

    def get_diagnostics(self) -> dict:
        return {
            "param": self.param_name,
            "counts": list(self._counts),
            "rewards": [round(r, 4) for r in self._rewards],
            "best": self.best(),
        }


# ---------------------------------------------------------------------------
# Continuous tuner — SimpleGP + EI
# ---------------------------------------------------------------------------

class _ContOptimizer:
    """GP-based optimiser for a single continuous (or integer) parameter."""

    def __init__(
        self,
        param_name: str,
        lo: float,
        hi: float,
        is_integer: bool = False,
    ) -> None:
        self.param_name = param_name
        self.lo = lo
        self.hi = hi
        self.is_integer = is_integer
        self._gp = SimpleGP()
        self._last_val: float | None = None

    def suggest(self) -> Any:
        val = self._gp.suggest(self.lo, self.hi)
        val = max(self.lo, min(self.hi, val))
        if self.is_integer:
            val = int(round(val))
        self._last_val = float(val)
        return val

    def update(self, reward: float) -> None:
        if self._last_val is not None:
            # Allow negative rewards so the GP learns bad regions too
            self._gp.observe(self._last_val, reward)

    def get_diagnostics(self) -> dict:
        return {
            "param": self.param_name,
            "n_observations": len(self._gp._xs),
            "last_val": self._last_val,
        }


# ---------------------------------------------------------------------------
# MetaOptimizer
# ---------------------------------------------------------------------------

class MetaOptimizer:
    """Orchestrates continuous hyperparameter tuning during training.

    Maintains:
    - A ``PhaseManager`` that advances through EXPLORE → EXPLOIT → REFINE → CONVERGE.
    - UCB1 bandits for categorical parameters.
    - GP-based optimisers for continuous parameters.
    - An overhead budget (never spends > ``max_overhead_pct``% of eval time).

    Called via ``tick()`` once per generation inside ``NeuroEvolution.train()``.
    """

    # Fallback hardcoded params used when no registry is provided
    _CAT_PARAMS: dict[str, list] = {
        "lamarck.mode": ["hill_climbing", "nes", "sa", "cma_es"],
    }
    _CONT_PARAMS: dict[str, tuple] = {
        "anytime.promotion_frac": (0.1, 0.9, False),
        "lamarck.sigma": (0.1, 5.0, False),
        "lamarck.n_steps": (1, 20, True),
    }

    # Params to exclude from automatic registry-based tuning.
    #
    # Rule 1 — ALL *.enabled boolean toggles are excluded.
    #   Feature on/off decisions belong to FeatureGating, which tests each
    #   feature in a controlled window and has proper enable/disable callbacks.
    #   Toggling e.g. attention.enabled via set_param() mid-training changes
    #   the genome's effective input dimensionality and breaks novelty vectors.
    #
    # Rule 2 — Init-stage params are excluded (require a full reconfigure).
    #
    # Rule 3 — Infrastructure params managed by other systems are excluded.
    _EXCLUDE: frozenset = frozenset({
        # ── All feature enabled/disabled flags (FeatureGating manages these) ──
        "anytime.enabled",
        "attention.enabled",
        "augmentation.enabled",
        "conv_neat.enabled",
        "crossover.enabled",
        "crossover.weight_inheritance_enabled",
        "curiosity.enabled",
        "darts.enabled",
        "diversity_injection.enabled",
        "hybrid.enabled",
        "input_grouping.enabled",
        "island.enabled",
        "lamarck.enabled",
        "lamarck.momentum_enabled",
        "matrix_forward.enabled",
        "neuromodulation.enabled",
        "novelty.enabled",
        "online_tuning.enabled",
        "output_grouping.enabled",
        "phylogeny.enabled",
        "pop.adaptive_enabled",
        "probabilistic.enabled",
        "probabilistic.inference_mode",
        "pruning.enabled",
        "recovery.enabled",
        "recovery.escalate",
        "shared_weights.enabled",
        "speciation.enabled",
        "stdp.enabled",
        "surrogate.enabled",
        # ── Init-stage: require reconfigure() ──────────────────────────────────
        "conv_neat.n_blocks",
        "island.n_islands",
        "island.migrate_interval",
        "island.migrate_count",
        # ── Infrastructure: managed by other systems ────────────────────────────
        "safety.min_safe_frac",
    })

    def __init__(
        self,
        *,
        tune_interval: int = 20,
        max_overhead_pct: float = 5.0,
        plateau_patience: int = _PLATEAU_PATIENCE,
        phase_min_gens: int = _PHASE_MIN_GENS,
        ucb1_c: float = 2.0,
        seed: int | None = None,
        param_registry: Any = None,
    ) -> None:
        """Create a MetaOptimizer.

        Parameters
        ----------
        param_registry:
            Optional ``ParamRegistry`` instance.  When provided, *all*
            registered parameters (excluding ``_EXCLUDE``) are added to the
            tuning pool automatically — categoricals as UCB1 bandits, booleans
            as 2-arm bandits, integers/continuous as GP optimisers.  Falls back
            to the hardcoded ``_CAT_PARAMS``/``_CONT_PARAMS`` when ``None``.
        """
        self.tune_interval = max(1, int(tune_interval))
        self.max_overhead_pct = max(0.5, float(max_overhead_pct))

        self._rng = random.Random(seed)
        self._phase_mgr = PhaseManager(
            plateau_patience=plateau_patience,
            phase_min_gens=phase_min_gens,
        )

        if param_registry is not None:
            cat_params, cont_params = self._params_from_registry(
                param_registry, ucb1_c
            )
        else:
            cat_params = {
                name: _CatBandit(name, cands, c=ucb1_c)
                for name, cands in self._CAT_PARAMS.items()
            }
            cont_params = {
                name: _ContOptimizer(name, lo, hi, is_int)
                for name, (lo, hi, is_int) in self._CONT_PARAMS.items()
            }

        self._cat: dict[str, _CatBandit] = cat_params
        self._cont: dict[str, _ContOptimizer] = cont_params

        self._generation: int = 0
        self._last_best: float = -math.inf
        self._best_at_last_tune: float = -math.inf  # for interval-delta reward
        self._total_eval_ms: float = 0.0
        self._meta_ms: float = 0.0
        self._n_ticks: int = 0
        self._n_skipped: int = 0
        self._param_change_log: list[dict] = []

    # ------------------------------------------------------------------
    # Registry-based param discovery
    # ------------------------------------------------------------------

    def _params_from_registry(
        self,
        registry: Any,
        ucb1_c: float,
    ) -> tuple[dict, dict]:
        """Build ``_cat`` and ``_cont`` dicts from a ``ParamRegistry``.

        All registered parameters are included except those in ``_EXCLUDE``.
        Boolean params become 2-arm categorical bandits (True/False).
        """
        cat: dict[str, _CatBandit] = {}
        cont: dict[str, _ContOptimizer] = {}

        space = registry.get_param_space()
        for name, spec in space.items():
            # Skip explicitly excluded params AND all *.enabled / *.escalate
            # boolean feature-toggle flags — FeatureGating manages those.
            if name in self._EXCLUDE or name.endswith(".enabled") or name.endswith(".escalate"):
                continue
            ptype = spec.get("type", "")
            domain = spec.get("domain")

            if ptype == "categorical":
                candidates = list(domain) if domain else []
                if len(candidates) >= 2:
                    cat[name] = _CatBandit(name, candidates, c=ucb1_c)

            elif ptype == "boolean":
                cat[name] = _CatBandit(name, [True, False], c=ucb1_c)

            elif ptype in ("continuous", "integer") and domain is not None:
                try:
                    lo, hi = float(domain[0]), float(domain[1])
                    is_int = ptype == "integer"
                    cont[name] = _ContOptimizer(name, lo, hi, is_int)
                except (TypeError, IndexError, ValueError):
                    pass

        return cat, cont

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def tick(
        self,
        generation: int,
        best_fitness: float,
        ne: "NeuroEvolution",
        eval_ms_this_gen: float = 0.0,
    ) -> None:
        """Called once per generation inside ``train()``.

        Parameters
        ----------
        generation:         Current generation number (0-based).
        best_fitness:       Population best fitness this generation.
        ne:                 Live ``NeuroEvolution`` instance (for ``set_param()``).
        eval_ms_this_gen:   Approximate wall time spent on evaluations (ms)
                            — used for overhead tracking.
        """
        self._generation = generation
        self._total_eval_ms += eval_ms_this_gen

        # Update phase (may trigger transition)
        transitioned = self._phase_mgr.update(generation, best_fitness)
        if transitioned:
            self._apply_phase_actions(ne)

        self._last_best = best_fitness

        # Tune params every tune_interval generations
        if generation % self.tune_interval == 0 and generation > 0:
            if self._can_run():
                t0 = time.monotonic()
                # Use improvement since last tune as reward (not just 1-gen delta)
                interval_delta = max(0.0, best_fitness - self._best_at_last_tune)
                self._best_at_last_tune = best_fitness
                self._tune(interval_delta, ne)
                self._meta_ms += (time.monotonic() - t0) * 1000
                self._n_ticks += 1
            else:
                self._n_skipped += 1

    # ------------------------------------------------------------------
    # Parameter optimisation
    # ------------------------------------------------------------------

    def _tune(self, delta: float, ne: "NeuroEvolution") -> None:
        """One round of parameter updates."""
        reg = ne.get_param_registry()
        phase = self._phase_mgr.current_phase

        # ── Categorical: update rewards, select next ────────────────────────
        for name, bandit in self._cat.items():
            # Skip if param not in registry or no dispatcher
            spec = reg.get_spec(name)
            if spec is None:
                continue
            # Update reward for the last selection
            bandit.update(delta)
            # In EXPLOIT/REFINE/CONVERGE: use current best; in EXPLORE: explore
            if phase in (TrainingPhase.EXPLOIT, TrainingPhase.REFINE, TrainingPhase.CONVERGE):
                val = bandit.best()  # exploit
            else:
                val = bandit.select()  # explore
            # Only update if value changed
            if val != reg.get_current(name):
                try:
                    ne.set_param(name, val)
                    self._param_change_log.append({
                        "generation": self._generation,
                        "param": name,
                        "value": val,
                        "delta": delta,
                    })
                except (KeyError, ValueError, TypeError):
                    pass

        # ── Continuous: update rewards, suggest next ────────────────────────
        # Only tune continuous params in EXPLOIT and later
        if phase not in (TrainingPhase.EXPLOIT, TrainingPhase.REFINE, TrainingPhase.CONVERGE):
            return

        for name, opt in self._cont.items():
            spec = reg.get_spec(name)
            if spec is None:
                continue
            opt.update(delta)
            val = opt.suggest()
            # Integer clamping
            if spec.type == "integer":
                val = int(round(val))
            # Domain clamp
            if spec.type in ("continuous", "integer"):
                lo, hi = spec.domain
                val = max(lo, min(hi, val))
            if val != reg.get_current(name):
                try:
                    ne.set_param(name, val)
                    self._param_change_log.append({
                        "generation": self._generation,
                        "param": name,
                        "value": val,
                        "delta": delta,
                    })
                except (KeyError, ValueError, TypeError):
                    pass

    def _apply_phase_actions(self, ne: "NeuroEvolution") -> None:
        """Apply the non-destructive preset for the current phase."""
        actions = _PHASE_ACTIONS.get(self._phase_mgr.current_phase, {})
        for name, value in actions.items():
            try:
                ne.set_param(name, value)
                self._param_change_log.append({
                    "generation": self._generation,
                    "param": name,
                    "value": value,
                    "delta": 0.0,
                    "reason": "phase_transition",
                })
            except (KeyError, ValueError, TypeError):
                pass

    # ------------------------------------------------------------------
    # Overhead self-throttling
    # ------------------------------------------------------------------

    def _can_run(self) -> bool:
        """True if meta overhead is below the configured budget."""
        if self._total_eval_ms < 1.0:
            return True
        overhead_pct = 100.0 * self._meta_ms / (self._total_eval_ms + self._meta_ms)
        return overhead_pct < self.max_overhead_pct

    @property
    def overhead_pct(self) -> float:
        total = self._total_eval_ms + self._meta_ms
        if total < 1e-6:
            return 0.0
        return 100.0 * self._meta_ms / total

    # ------------------------------------------------------------------
    # Phase / state access
    # ------------------------------------------------------------------

    @property
    def current_phase(self) -> TrainingPhase:
        return self._phase_mgr.current_phase

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> dict:
        phase_diag = self._phase_mgr.get_diagnostics()
        return {
            "enabled": True,
            "phase": phase_diag["current_phase"],
            "stagnation_gens": phase_diag["stagnation_gens"],
            "phase_transitions": phase_diag["transitions"],
            "n_ticks": self._n_ticks,
            "n_skipped": self._n_skipped,
            "overhead_pct": round(self.overhead_pct, 2),
            "categorical": {
                name: bandit.get_diagnostics()
                for name, bandit in self._cat.items()
            },
            "continuous": {
                name: opt.get_diagnostics()
                for name, opt in self._cont.items()
            },
            "recent_changes": self._param_change_log[-10:],
        }
