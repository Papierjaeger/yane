"""Experiment-tracking backend protocol and built-in implementations.

Provides a lightweight adapter layer so YANE training runs can be tracked by
WandB, MLflow, or any custom backend without creating hard dependencies on
those libraries.

Usage::

    from yane.evolution.tracking import WandbBackend, MlflowBackend

    yane.set_tracking_backend(WandbBackend(project="my-project"))
    yane.set_tracking_backend(MlflowBackend(experiment_name="xor-test"))
    yane.train(eval_fn)

Multiple backends can be registered simultaneously — each receives the same
metrics and config on every call.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class TrackingBackend(Protocol):
    """Structural protocol for experiment-tracking adapters.

    All methods are called automatically by ``NeuroEvolution.train()``.
    Implementations should swallow transient errors rather than raise so that
    tracking failures never abort training.
    """

    def init(self, config: dict) -> None:
        """Called once at the very start of ``train()``, before any evaluations."""

    def log_config(self, config: dict) -> None:
        """Log training hyperparameters (called once after ``init()``)."""

    def log_metrics(self, metrics: dict, step: int) -> None:
        """Log per-generation scalar metrics.  *step* is the generation index."""

    def log_artifact(self, path: str, artifact_type: str = "model") -> None:
        """Upload a local file as an artifact (e.g. a checkpoint .pkl)."""

    def finish(self) -> None:
        """Called once at the end of ``train()``.  Flush buffers and close."""


def _scalar_metrics(mem: dict) -> dict:
    """Return only the scalar (int/float, non-NaN) entries from *mem*."""
    result: dict = {}
    for k, v in mem.items():
        if isinstance(v, bool):
            result[k] = int(v)
        elif isinstance(v, (int, float)) and v == v:  # exclude NaN
            result[k] = float(v)
    return result


# ---------------------------------------------------------------------------
# WandB backend
# ---------------------------------------------------------------------------

class WandbBackend:
    """WandB experiment-tracking backend.

    Requires the ``wandb`` package::

        pip install wandb

    Authentication: set the ``WANDB_API_KEY`` environment variable or run
    ``wandb login`` once.  Pass ``mode="disabled"`` for offline / CI usage.
    """

    def __init__(
        self,
        project: str = "yane",
        run_name: str | None = None,
        mode: str = "online",
        tags: list[str] | None = None,
        **wandb_kwargs,
    ) -> None:
        self._project = project
        self._run_name = run_name
        self._mode = mode
        self._tags = list(tags) if tags else []
        self._wandb_kwargs = wandb_kwargs
        self._run = None

    @staticmethod
    def _import() :
        try:
            import wandb
            return wandb
        except ImportError as exc:
            raise ImportError(
                "WandbBackend requires the 'wandb' package.  "
                "Install it with:  pip install wandb"
            ) from exc

    def init(self, config: dict) -> None:
        wandb = self._import()
        self._run = wandb.init(
            project=self._project,
            name=self._run_name,
            mode=self._mode,
            tags=self._tags or None,
            **self._wandb_kwargs,
        )

    def log_config(self, config: dict) -> None:
        if self._run is not None:
            self._run.config.update(config)

    def log_metrics(self, metrics: dict, step: int) -> None:
        if self._run is not None:
            try:
                self._run.log({f"yane/{k}": v for k, v in metrics.items()}, step=step)
            except Exception:
                pass

    def log_artifact(self, path: str, artifact_type: str = "model") -> None:
        if self._run is None:
            return
        try:
            wandb = self._import()
            artifact = wandb.Artifact(name=Path(path).stem, type=artifact_type)
            artifact.add_file(path)
            self._run.log_artifact(artifact)
        except Exception:
            pass

    def finish(self) -> None:
        if self._run is not None:
            try:
                self._run.finish()
            except Exception:
                pass
            self._run = None


# ---------------------------------------------------------------------------
# MLflow backend
# ---------------------------------------------------------------------------

class MlflowBackend:
    """MLflow experiment-tracking backend.

    Requires the ``mlflow`` package::

        pip install mlflow

    Tracking server: set the ``MLFLOW_TRACKING_URI`` environment variable.
    When omitted, metrics are written to a local ``mlruns/`` directory.
    """

    def __init__(
        self,
        experiment_name: str = "yane",
        run_name: str | None = None,
        tracking_uri: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        self._experiment_name = experiment_name
        self._run_name = run_name
        self._tracking_uri = tracking_uri
        self._tags = dict(tags) if tags else {}
        self._run = None

    @staticmethod
    def _import():
        try:
            import mlflow
            return mlflow
        except ImportError as exc:
            raise ImportError(
                "MlflowBackend requires the 'mlflow' package.  "
                "Install it with:  pip install mlflow"
            ) from exc

    def init(self, config: dict) -> None:
        mlflow = self._import()
        if self._tracking_uri:
            mlflow.set_tracking_uri(self._tracking_uri)
        mlflow.set_experiment(self._experiment_name)
        self._run = mlflow.start_run(
            run_name=self._run_name,
            tags=self._tags or None,
        )

    def log_config(self, config: dict) -> None:
        if self._run is None:
            return
        mlflow = self._import()
        # MLflow params must be strings; keys and values max 250 chars
        try:
            params = {str(k)[:250]: str(v)[:250] for k, v in config.items()}
            mlflow.log_params(params)
        except Exception:
            pass

    def log_metrics(self, metrics: dict, step: int) -> None:
        if self._run is None:
            return
        mlflow = self._import()
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception:
            pass

    def log_artifact(self, path: str, artifact_type: str = "model") -> None:
        if self._run is None:
            return
        mlflow = self._import()
        try:
            mlflow.log_artifact(path)
        except Exception:
            pass

    def finish(self) -> None:
        if self._run is None:
            return
        mlflow = self._import()
        try:
            mlflow.end_run()
        except Exception:
            pass
        self._run = None
