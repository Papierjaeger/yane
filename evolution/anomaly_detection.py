"""Online anomaly detectors for YANE training runs.

Detectors are called at each heartbeat (every 100 iterations) with the
current diagnostics dict and the current iteration count.  Each returns
an AnomalyReport when an anomaly is detected, or None otherwise.

Usage::

    ne = NeuroEvolution()
    ne.set_anomaly_detectors([FitnessCollapseDetector(), DiversityCollapseDetector()])
    # or just:
    ne.set_anomaly_detectors()   # uses DEFAULT_DETECTORS
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class AnomalyReport:
    """A single anomaly detected during training."""
    kind: str       # detector name / short identifier
    message: str    # human-readable description
    iteration: int
    value: float    # the metric value that triggered detection
    threshold: float  # the threshold that was breached


class FitnessCollapseDetector:
    """Fires when best fitness drops by *drop_frac* or more over the last *window* checks."""

    def __init__(self, drop_frac: float = 0.1, window: int = 10) -> None:
        self.drop_frac = drop_frac
        self.window = window
        self._history: list[float] = []

    def check(self, diagnostics: dict, iteration: int) -> AnomalyReport | None:
        best = diagnostics.get("max_fitness", 0.0)
        self._history.append(best)
        if len(self._history) > self.window:
            self._history.pop(0)
        if len(self._history) < self.window:
            return None
        peak = max(self._history)
        if peak <= 0.0:
            return None
        drop = (peak - best) / abs(peak)
        if drop >= self.drop_frac:
            return AnomalyReport(
                kind="fitness_collapse",
                message=(
                    f"Best fitness dropped {drop:.1%} from peak {peak:.4f} to {best:.4f}"
                ),
                iteration=iteration, value=drop, threshold=self.drop_frac,
            )
        return None


class DiversityCollapseDetector:
    """Fires when fitness IQR falls below *min_iqr* (population is homogeneous)."""

    def __init__(self, min_iqr: float = 1e-4, min_pop: int = 10) -> None:
        self.min_iqr = min_iqr
        self.min_pop = min_pop

    def check(self, diagnostics: dict, iteration: int) -> AnomalyReport | None:
        iqr = diagnostics.get("fitness_iqr")
        pop = diagnostics.get("pop_evaluated", 0)
        if iqr is None or pop < self.min_pop or not math.isfinite(iqr):
            return None
        if iqr < self.min_iqr:
            return AnomalyReport(
                kind="diversity_collapse",
                message=(
                    f"Fitness IQR={iqr:.2e} < {self.min_iqr:.2e} "
                    "(population is homogeneous)"
                ),
                iteration=iteration, value=iqr, threshold=self.min_iqr,
            )
        return None


class HomogenizationDetector:
    """Fires when species_count has been ≤ 1 for *window* consecutive heartbeats."""

    def __init__(self, window: int = 5) -> None:
        self.window = window
        self._streak: int = 0

    def check(self, diagnostics: dict, iteration: int) -> AnomalyReport | None:
        sc = diagnostics.get("species_count", 0)
        if sc <= 1:
            self._streak += 1
        else:
            self._streak = 0
        if self._streak >= self.window:
            return AnomalyReport(
                kind="homogenization",
                message=(
                    f"Population collapsed to ≤1 species for "
                    f"{self._streak} consecutive checks"
                ),
                iteration=iteration,
                value=float(self._streak),
                threshold=float(self.window),
            )
        return None


class StuckSpeciationDetector:
    """Fires when species_count is 1 AND stagnation reaches *min_stagnation_frac*."""

    def __init__(self, min_stagnation_frac: float = 0.5) -> None:
        self.min_stagnation_frac = min_stagnation_frac

    def check(self, diagnostics: dict, iteration: int) -> AnomalyReport | None:
        sc = diagnostics.get("species_count", 0)
        stagn = diagnostics.get("stagnation_count", 0)
        th = diagnostics.get("stagnation_threshold", 1) or 1
        frac = stagn / th
        if sc <= 1 and frac >= self.min_stagnation_frac:
            return AnomalyReport(
                kind="stuck_speciation",
                message=(
                    f"1 species + stagnation {stagn}/{th} ({frac:.0%}) "
                    "— training may be stuck"
                ),
                iteration=iteration, value=frac,
                threshold=self.min_stagnation_frac,
            )
        return None


DEFAULT_DETECTORS = [
    FitnessCollapseDetector(),
    DiversityCollapseDetector(),
    HomogenizationDetector(),
    StuckSpeciationDetector(),
]


class AnomalyDetectorSet:
    """Runs a collection of detectors and aggregates results.

    Tracks cumulative counts and the most recent anomaly report for diagnostics.
    """

    def __init__(self, detectors=None) -> None:
        self.detectors = list(detectors if detectors is not None else DEFAULT_DETECTORS)
        self.n_detected: int = 0
        self.last_anomaly: AnomalyReport | None = None

    def check_all(self, diagnostics: dict, iteration: int) -> list[AnomalyReport]:
        reports: list[AnomalyReport] = []
        for det in self.detectors:
            try:
                r = det.check(diagnostics, iteration)
                if r is not None:
                    reports.append(r)
                    self.n_detected += 1
                    self.last_anomaly = r
            except Exception:
                pass
        return reports

    def get_diagnostics(self) -> dict:
        return {
            "anomalies_detected": self.n_detected,
            "last_anomaly": (
                {
                    "kind": self.last_anomaly.kind,
                    "message": self.last_anomaly.message,
                    "iteration": self.last_anomaly.iteration,
                }
                if self.last_anomaly else None
            ),
        }
