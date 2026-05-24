"""GUI-facing configuration for remote evaluation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RemoteEvaluationConfig:
    enabled: bool = False
    worker_urls: tuple[str, ...] = ()
    token: str = ""
    timeout_s: float = 30.0
    max_retries: int = 2
    batch_size: int = 0

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size or max(1, len(self.worker_urls) * 2)

    @classmethod
    def from_text(
        cls,
        enabled: bool,
        worker_urls_text: str,
        token: str = "",
        timeout_s: float = 30.0,
        max_retries: int = 2,
        batch_size: int = 0,
    ) -> "RemoteEvaluationConfig":
        urls = tuple(
            part.strip()
            for raw in worker_urls_text.replace("\n", ",").split(",")
            for part in [raw]
            if part.strip()
        )
        if enabled and not urls:
            raise ValueError("Remote Evaluation braucht mindestens eine Worker-URL.")
        return cls(
            enabled=enabled,
            worker_urls=urls,
            token=token,
            timeout_s=timeout_s,
            max_retries=max_retries,
            batch_size=batch_size,
        )
