import time
import psutil

_BYTES_PER_GB = 1_073_741_824


class ResourceGuard:
    """Pauses the training loop when system memory falls below a safe threshold.

    Instead of crashing or being killed by the OS, training simply waits until
    enough memory is available again and then continues.
    """

    def __init__(
        self,
        min_free_gb: float = 2.0,
        max_used_percent: float = 85.0,
        poll_interval_s: float = 0.5,
    ) -> None:
        self.min_free_gb = min_free_gb
        self.max_used_percent = max_used_percent
        self.poll_interval_s = poll_interval_s

    def wait_if_needed(self) -> None:
        """Block until system memory is within safe limits."""
        while True:
            mem = psutil.virtual_memory()
            if mem.available / _BYTES_PER_GB >= self.min_free_gb and mem.percent <= self.max_used_percent:
                return
            time.sleep(self.poll_interval_s)

    @staticmethod
    def current_free_gb() -> float:
        return psutil.virtual_memory().available / _BYTES_PER_GB
