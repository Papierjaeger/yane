import gc
import psutil

_BYTES_PER_GB = 1_073_741_824
_PROCESS = psutil.Process()


class ResourceGuard:
    """Monitors both system memory and this process's own RAM usage.

    System limit: pauses training when free system memory is low.
    Process limit: actively shrinks the population when this process
                   exceeds its allowed RAM budget.
    """

    def __init__(
        self,
        min_free_gb: float = 2.0,
        max_used_percent: float = 85.0,
        max_process_gb: float | None = None,
    ) -> None:
        self.min_free_gb = min_free_gb
        self.max_used_percent = max_used_percent
        self.max_process_gb = max_process_gb

    def system_ok(self) -> bool:
        mem = psutil.virtual_memory()
        return (
            mem.available / _BYTES_PER_GB >= self.min_free_gb
            and mem.percent <= self.max_used_percent
        )

    def process_over_limit(self) -> bool:
        if self.max_process_gb is None:
            return False
        return _PROCESS.memory_info().rss / _BYTES_PER_GB > self.max_process_gb

    @staticmethod
    def current_free_gb() -> float:
        return psutil.virtual_memory().available / _BYTES_PER_GB

    @staticmethod
    def current_process_gb() -> float:
        return _PROCESS.memory_info().rss / _BYTES_PER_GB
