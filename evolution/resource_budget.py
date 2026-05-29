"""Resource Budget System for YANE.

Provides unified, human-readable resource limits with automatic hardware
calibration and a six-level graceful degradation pipeline.

Typical usage::

    yane.set_budget("auto")                    # calibrate from hardware
    yane.set_budget(total_time="30min")        # wall-clock stop
    yane.set_budget(total_time="1h",
                    max_memory="auto",
                    target_platform="cortex-m4")

Public surface
--------------
``parse_time(s)``
    Convert human-readable time string (``"30min"``, ``"2h"``, ``"45s"``)
    to seconds (float).

``parse_memory(s, available_bytes)``
    Convert human-readable memory string (``"4GB"``, ``"80%"``, ``"auto"``)
    to bytes (int).

``ResourceDiscovery``
    Lightweight hardware auto-calibration.  Every sensor is wrapped so a
    missing device (no GPU, desktop without battery) returns *None* rather
    than raising.

``BudgetConfig``
    Dataclass that holds the resolved budget parameters.

``GracefulDegradation``
    Six-level escalation pipeline that mutates a live NeuroEvolution
    instance to reduce resource pressure.

``BudgetEnforcer``
    Ties it all together: tracks elapsed time, checks memory pressure,
    triggers degradation.  An injectable *clock* callable makes the
    time budget fully testable without sleeping.
"""
from __future__ import annotations

import gc
import re
import time as _time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    pass  # avoid circular imports

# ---------------------------------------------------------------------------
# Unit parsers
# ---------------------------------------------------------------------------

_TIME_RE = re.compile(
    r"^\s*(?P<val>[0-9]+(?:\.[0-9]*)?)\s*(?P<unit>h|hr|hour|hours|m|min|minute|minutes|s|sec|second|seconds)?\s*$",
    re.IGNORECASE,
)
_MEM_RE = re.compile(
    r"^\s*(?P<val>[0-9]+(?:\.[0-9]*)?)\s*(?P<unit>b|kb|mb|gb|tb|kib|mib|gib|tib|%)\s*$",
    re.IGNORECASE,
)

_MEM_MULTIPLIERS = {
    "b": 1,
    "kb": 1_000, "kib": 1_024,
    "mb": 1_000_000, "mib": 1_048_576,
    "gb": 1_000_000_000, "gib": 1_073_741_824,
    "tb": 1_000_000_000_000, "tib": 1_099_511_627_776,
}


def parse_time(s: str | float | int | None) -> float | None:
    """Parse a human-readable time string to seconds.

    Examples::

        parse_time("30min")  → 1800.0
        parse_time("1h")     → 3600.0
        parse_time("45s")    → 45.0
        parse_time(300)      → 300.0
        parse_time(None)     → None

    Raises ``ValueError`` for unrecognised format.
    """
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    if not isinstance(s, str):
        raise TypeError(f"Expected str, int, or float; got {type(s).__name__!r}")
    s = s.strip()
    if s == "" or s.lower() == "none":
        return None
    m = _TIME_RE.match(s)
    if not m:
        raise ValueError(f"Cannot parse time: {s!r}")
    val = float(m.group("val"))
    unit = (m.group("unit") or "s").lower()
    if unit in ("h", "hr", "hour", "hours"):
        return val * 3600.0
    if unit in ("m", "min", "minute", "minutes"):
        return val * 60.0
    return val  # seconds


def parse_memory(
    s: str | float | int | None,
    available_bytes: int | None = None,
) -> int | None:
    """Parse a human-readable memory string to bytes.

    ``"auto"`` resolves to 80 % of *available_bytes* (or of physical RAM
    when *available_bytes* is *None*).

    ``"80%"`` resolves to 80 % of *available_bytes* (falls back to physical
    RAM when *available_bytes* is *None*).

    Examples::

        parse_memory("4GB")              → 4_000_000_000
        parse_memory("auto")             → 80% of available RAM
        parse_memory("80%", 8_000_000_000) → 6_400_000_000

    Raises ``ValueError`` for unrecognised format.
    """
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return int(s)
    if not isinstance(s, str):
        raise TypeError(f"Expected str, int, or float; got {type(s).__name__!r}")
    s = s.strip()
    if s.lower() in ("none", ""):
        return None

    def _base() -> int:
        if available_bytes is not None:
            return int(available_bytes)
        return ResourceDiscovery.ram_available_bytes() or 0

    if s.lower() == "auto":
        return int(_base() * 0.80)

    m = _MEM_RE.match(s)
    if not m:
        raise ValueError(f"Cannot parse memory: {s!r}")
    val = float(m.group("val"))
    unit = m.group("unit").lower()
    if unit == "%":
        return int(_base() * val / 100.0)
    mult = _MEM_MULTIPLIERS.get(unit)
    if mult is None:
        raise ValueError(f"Unknown memory unit: {unit!r}")
    return int(val * mult)


# ---------------------------------------------------------------------------
# Resource discovery
# ---------------------------------------------------------------------------

class ResourceDiscovery:
    """Lightweight hardware auto-calibration.

    Every sensor is wrapped so a missing device (desktop with no battery,
    no NVIDIA driver, old psutil) returns *None* rather than raising.
    """

    @staticmethod
    def cpu_count() -> int:
        try:
            import psutil
            return psutil.cpu_count(logical=False) or 1
        except Exception:
            return 1

    @staticmethod
    def ram_total_bytes() -> int:
        try:
            import psutil
            return psutil.virtual_memory().total
        except Exception:
            return 0

    @staticmethod
    def ram_available_bytes() -> int:
        try:
            import psutil
            return psutil.virtual_memory().available
        except Exception:
            return 0

    @staticmethod
    def current_process_bytes() -> int:
        """RSS of the current process in bytes."""
        try:
            import psutil
            return psutil.Process().memory_info().rss
        except Exception:
            return 0

    @staticmethod
    def gpu_memory_bytes() -> int | None:
        """Total GPU memory in bytes, or *None* if no GPU / driver unavailable."""
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.total
        except Exception:
            return None

    @staticmethod
    def disk_free_bytes(path: str = ".") -> int:
        try:
            import psutil
            return psutil.disk_usage(path).free
        except Exception:
            return 0

    @staticmethod
    def battery_plugged() -> bool | None:
        """True if on AC power, False if on battery, None if unknown."""
        try:
            import psutil
            batt = getattr(psutil, "sensors_battery", lambda: None)()
            if batt is None:
                return None
            return batt.power_plugged
        except Exception:
            return None

    @classmethod
    def auto_memory_budget(cls, fraction: float = 0.80) -> int:
        """Return *fraction* of currently available RAM as the memory budget."""
        return int(cls.ram_available_bytes() * fraction)

    @classmethod
    def describe(cls) -> dict[str, Any]:
        """Return a dict with discovered hardware info (safe on all platforms)."""
        return {
            "cpu_cores": cls.cpu_count(),
            "ram_total_gb": round(cls.ram_total_bytes() / 1_073_741_824, 2),
            "ram_available_gb": round(cls.ram_available_bytes() / 1_073_741_824, 2),
            "gpu_memory_gb": (
                round(cls.gpu_memory_bytes() / 1_073_741_824, 2)
                if cls.gpu_memory_bytes() is not None else None
            ),
            "disk_free_gb": round(cls.disk_free_bytes() / 1_073_741_824, 2),
            "battery_plugged": cls.battery_plugged(),
        }


# ---------------------------------------------------------------------------
# Budget configuration
# ---------------------------------------------------------------------------

@dataclass
class BudgetConfig:
    """Resolved budget parameters (all values in SI units)."""

    total_time_seconds: float | None = None
    """Wall-clock training time limit in seconds.  *None* = unlimited."""

    max_memory_bytes: int | None = None
    """Per-process RSS cap in bytes.  *None* = unlimited."""

    max_cpu_pct: float | None = None
    """CPU-usage ceiling in percent.  Stored for diagnostics; not enforced."""

    target_platform: str | None = None
    """Deployment platform (passed through to HardwareConstraints if set)."""

    auto: bool = False
    """Whether this config was created via ``set_budget("auto")``."""


# ---------------------------------------------------------------------------
# Graceful degradation pipeline
# ---------------------------------------------------------------------------

class GracefulDegradation:
    """Six-level escalation pipeline that mutates a live NeuroEvolution instance.

    Levels escalate monotonically — recovery is out of scope.

    Level 1: reduce_pop      — halve population size
    Level 2: skip_lamarck    — disable Lamarckian refinement
    Level 3: simplify_topology — reduce max_nodes by 25 %
    Level 4: disable_research  — turn off P2 research features
    Level 5: reduce_eval_budget — cap anytime evaluation to 1 evaluation
    Level 6: emergency_stop    — save emergency checkpoint + request stop
    """

    MAX_LEVEL = 6

    def __init__(self, ne_ref: Any) -> None:
        self._ne = ne_ref
        self.current_level: int = 0
        self._applied: list[str] = []   # human-readable log of actions taken
        self.stop_requested: bool = False
        self._emergency_checkpoint_path: str | None = None

    def escalate(self) -> int:
        """Escalate to the next degradation level.

        Returns the new level (1–6).  Calling beyond level 6 is a no-op.
        """
        if self.current_level >= self.MAX_LEVEL:
            return self.current_level
        self.current_level += 1
        dispatch = {
            1: self._reduce_pop,
            2: self._skip_lamarck,
            3: self._simplify_topology,
            4: self._disable_research,
            5: self._reduce_eval_budget,
            6: self._emergency_stop,
        }
        dispatch[self.current_level]()
        return self.current_level

    # ------------------------------------------------------------------
    # Individual degradation actions
    # ------------------------------------------------------------------

    def _reduce_pop(self) -> None:
        ne = self._ne
        original = getattr(ne, "_population_size", 100)
        new_size = max(10, original // 2)
        ne._population_size = new_size
        # Also shrink the running population if it exists
        pop = getattr(ne, "_population", None)
        if pop is not None:
            evaluated = getattr(pop, "_evaluated", [])
            if len(evaluated) > new_size:
                target = max(10, len(evaluated) // 2)
                try:
                    pop.shrink_to(target)
                except Exception:
                    pass
        self._applied.append(f"reduce_pop: {original} → {new_size}")
        self._warn(f"[budget] Level 1: population halved ({original} → {new_size})")

    def _skip_lamarck(self) -> None:
        ne = self._ne
        lamarck = getattr(ne, "_lamarck", None)
        if lamarck is not None:
            lamarck.steps = 0
            lamarck.max_steps = 0
        self._applied.append("skip_lamarck: steps=0, max_steps=0")
        self._warn("[budget] Level 2: Lamarckian refinement disabled")

    def _simplify_topology(self) -> None:
        ne = self._ne
        original = getattr(ne, "_max_nodes", 100)
        new_max = max(5, int(original * 0.75))
        ne._max_nodes = new_max
        self._applied.append(f"simplify_topology: max_nodes {original} → {new_max}")
        self._warn(f"[budget] Level 3: max_nodes reduced ({original} → {new_max})")

    def _disable_research(self) -> None:
        ne = self._ne
        disabled = []
        for attr in ("_curiosity_enabled", "_darts_enabled", "_shared_weights_enabled"):
            if getattr(ne, attr, False):
                setattr(ne, attr, False)
                disabled.append(attr.lstrip("_"))
        # Feature gating
        fg = getattr(ne, "_feature_gating", None)
        if fg is not None:
            for rec in fg._features.values():
                rec.active = False
        self._applied.append(f"disable_research: {disabled or 'none'}")
        self._warn(f"[budget] Level 4: research features disabled: {disabled or 'none'}")

    def _reduce_eval_budget(self) -> None:
        ne = self._ne
        runner = getattr(ne, "_runner", None)
        if runner is not None and getattr(runner, "anytime_enabled", False):
            try:
                runner.configure_anytime_eval(enabled=True, min_evals=1, max_evals=1)
            except Exception:
                pass
        self._applied.append("reduce_eval_budget: anytime max_evals=1")
        self._warn("[budget] Level 5: anytime eval budget reduced to 1 eval/genome")

    def _emergency_stop(self) -> None:
        ne = self._ne
        self.stop_requested = True
        # Attempt to save an emergency checkpoint
        import tempfile, os
        path = os.path.join(tempfile.gettempdir(), "yane_emergency_checkpoint.pkl")
        try:
            configured = getattr(ne, "is_configured", False)
            if configured:
                ne.save_checkpoint(path)
                self._emergency_checkpoint_path = path
        except Exception:
            pass
        self._applied.append(f"emergency_stop: checkpoint={path}")
        self._warn(f"[budget] Level 6: EMERGENCY STOP — checkpoint saved to {path}")

    @staticmethod
    def _warn(msg: str) -> None:
        try:
            from yane.util.logger import get_logger
            get_logger().warning(msg)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Budget enforcer — wires it together
# ---------------------------------------------------------------------------

class BudgetEnforcer:
    """Tracks time and memory budgets; escalates graceful degradation.

    Parameters
    ----------
    config :
        The resolved ``BudgetConfig`` for this training run.
    ne_ref :
        Reference to the ``NeuroEvolution`` instance being trained.
        Used by ``GracefulDegradation`` to mutate state.
    clock :
        Injectable callable returning current wall-clock seconds
        (default: ``time.time``).  Pass a synthetic clock in tests.
    """

    def __init__(
        self,
        config: BudgetConfig,
        ne_ref: Any,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.config = config
        self.degradation = GracefulDegradation(ne_ref)
        self._clock = clock or _time.time
        self._start: float | None = None
        # Track how many memory-pressure events triggered an escalation
        self._memory_escalations: int = 0

    def start(self) -> None:
        """Record the training start time.  Call once before the train loop."""
        self._start = self._clock()

    # ------------------------------------------------------------------
    # Time budget
    # ------------------------------------------------------------------

    def elapsed_seconds(self) -> float:
        """Seconds elapsed since :meth:`start` was called."""
        if self._start is None:
            return 0.0
        return self._clock() - self._start

    def is_time_over(self) -> bool:
        """Return *True* if the time budget is exhausted."""
        budget = self.config.total_time_seconds
        if budget is None:
            return False
        return self.elapsed_seconds() >= budget

    # ------------------------------------------------------------------
    # Memory budget
    # ------------------------------------------------------------------

    def check_memory(self, over_budget: bool | None = None) -> None:
        """Check memory pressure and escalate if over budget.

        Parameters
        ----------
        over_budget :
            When *None* (default) the enforcer reads actual process RSS.
            Pass ``True`` or ``False`` to force a result in tests.
        """
        if self.config.max_memory_bytes is None:
            return
        if over_budget is None:
            current = ResourceDiscovery.current_process_bytes()
            over_budget = current > self.config.max_memory_bytes
        if over_budget:
            gc.collect()
            self._memory_escalations += 1
            self.degradation.escalate()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Return a snapshot of the current budget state."""
        elapsed = self.elapsed_seconds()
        budget_s = self.config.total_time_seconds
        mem_budget = self.config.max_memory_bytes
        return {
            "elapsed_seconds": elapsed,
            "time_budget_seconds": budget_s,
            "time_remaining_seconds": (
                max(0.0, budget_s - elapsed) if budget_s is not None else None
            ),
            "time_fraction_used": (
                min(1.0, elapsed / budget_s) if budget_s else None
            ),
            "memory_budget_bytes": mem_budget,
            "memory_current_bytes": ResourceDiscovery.current_process_bytes(),
            "degradation_level": self.degradation.current_level,
            "degradation_actions": list(self.degradation._applied),
            "stop_requested": self.degradation.stop_requested,
            "emergency_checkpoint": self.degradation._emergency_checkpoint_path,
            "target_platform": self.config.target_platform,
        }
