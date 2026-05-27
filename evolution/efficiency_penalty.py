class EfficiencyPenalty:
    """Applies a fitness penalty when a genome's evaluation exceeds a time budget."""

    def __init__(self, max_ms: float, penalty_per_ms: float) -> None:
        self.max_ms = max_ms
        self.penalty_per_ms = penalty_per_ms

    def apply(self, fitness: float, elapsed_ms: float) -> float:
        excess = max(0.0, elapsed_ms - self.max_ms)
        return fitness - excess * self.penalty_per_ms
