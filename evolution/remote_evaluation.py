"""Remote/distributed genome evaluation over HTTP.

Architecture
------------
A *master* process (running NeuroEvolution) serialises each Genome to a
base64-encoded pickle and sends it to one or more *remote worker* processes
via ``POST /evaluate``.  Workers run a small FastAPI app that has the user's
fitness function pre-loaded; they unpickle the genome, evaluate it, and return
the numeric fitness.

Security model
--------------
Workers only accept pickle payloads signed with a shared token
(``Authorization: Bearer <token>``).  This prevents arbitrary code execution
via crafted pickles from untrusted senders:

* The token is set via ``RemoteWorkerServer(token=...)`` and
  ``RemoteEvaluationClient(token=...)``.
* **Never** start a worker accessible from a public network without a strong,
  randomly-generated token.
* Workers should run in a private LAN or over an encrypted tunnel (SSH, VPN).
  HTTPS at the application layer is not provided here — add a reverse proxy
  for production deployments.

Usage — master side::

    from yane.evolution.remote_evaluation import RemoteEvaluationClient

    client = RemoteEvaluationClient(
        worker_urls=["http://localhost:8700", "http://worker2:8700"],
        token="secret",
        timeout_s=10.0,
        max_retries=2,
    )
    results = client.evaluate_batch(genomes)  # list[(Genome, float)]
    client.close()

Usage — worker side (run once per machine, with fitness function pre-loaded)::

    from yane.evolution.remote_evaluation import RemoteWorkerServer

    def my_fitness(genome):
        ...
        return fitness_value

    server = RemoteWorkerServer(fitness_fn=my_fitness, token="secret", port=8700)
    server.run()  # blocking; press Ctrl-C to stop
"""
from __future__ import annotations

import base64
import hmac
import pickle
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from yane.core.genome import Genome

try:
    from pydantic import BaseModel as _BaseModel
    from pydantic import Field as _Field

    class _EvalRequest(_BaseModel):
        job_id: str = _Field(default="", max_length=128)
        genome_b64: str = _Field(default="", max_length=64 * 1024 * 1024)
        timeout_s: float = _Field(default=30.0, gt=0.0, le=3600.0)
except ImportError:
    _EvalRequest = None  # type: ignore[assignment,misc]


# ── Protocol data structures ──────────────────────────────────────────────────

@dataclass
class EvalJob:
    """A single evaluation job sent from master to worker."""
    job_id: str
    genome_b64: str  # base64(pickle(genome))
    timeout_s: float = 30.0


@dataclass
class EvalResult:
    """Response from a worker for one evaluation job."""
    job_id: str
    fitness: float | None
    error: str | None
    duration_s: float


# ── Client ────────────────────────────────────────────────────────────────────

@dataclass
class _PendingJob:
    job_id: str
    genome: Genome
    worker_url: str
    attempt: int = 0


class RemoteEvaluationClient:
    """Send genomes to remote workers for evaluation.

    Parameters
    ----------
    worker_urls:
        List of ``http://host:port`` strings for available workers.
    token:
        Shared secret token.  Must match the token configured on each worker.
    timeout_s:
        Per-request HTTP timeout in seconds.
    max_retries:
        How many times to retry a failed request before giving up.
    retry_delay_s:
        Seconds to wait between retries.
    """

    def __init__(
        self,
        worker_urls: list[str],
        token: str = "",
        timeout_s: float = 30.0,
        max_retries: int = 2,
        retry_delay_s: float = 0.5,
    ) -> None:
        if not worker_urls:
            raise ValueError("At least one worker_url is required")
        self.worker_urls = list(worker_urls)
        self.token = token
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_delay_s = retry_delay_s
        self._lock = threading.Lock()
        self._rr_idx = 0
        self._cancelled: set[str] = set()

    # ── public API ────────────────────────────────────────────────────────────

    def evaluate_batch(
        self, genomes: list[Genome]
    ) -> list[tuple[Genome, float]]:
        """Evaluate all *genomes* concurrently; return ``(genome, fitness)`` pairs.

        Genomes whose evaluation fails after all retries are skipped (not
        returned).  Cancelled jobs (via ``cancel(job_id)``) are also skipped.
        """
        import concurrent.futures
        jobs = [_PendingJob(str(uuid.uuid4()), g, self._next_worker()) for g in genomes]

        def _run(job: _PendingJob) -> tuple[Genome, float] | None:
            return self._evaluate_with_retry(job)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.worker_urls) * 2) as pool:
            futures = {pool.submit(_run, j): j for j in jobs}
            results: list[tuple[Genome, float]] = []
            for fut in concurrent.futures.as_completed(futures):
                outcome = fut.result()
                if outcome is not None:
                    results.append(outcome)
        return results

    def cancel(self, job_id: str) -> None:
        """Mark *job_id* as cancelled.  If still in flight it will be ignored on completion."""
        with self._lock:
            self._cancelled.add(job_id)

    def close(self) -> None:
        """Release any held resources (currently a no-op; kept for symmetry with pool APIs)."""

    # ── internals ─────────────────────────────────────────────────────────────

    def _next_worker(self) -> str:
        with self._lock:
            url = self.worker_urls[self._rr_idx % len(self.worker_urls)]
            self._rr_idx += 1
        return url

    def _evaluate_with_retry(
        self, job: _PendingJob
    ) -> tuple[Genome, float] | None:
        for attempt in range(self.max_retries + 1):
            with self._lock:
                if job.job_id in self._cancelled:
                    return None
            try:
                result = self._call_worker(job.worker_url, job.job_id, job.genome)
                if result.error:
                    raise RuntimeError(result.error)
                with self._lock:
                    if job.job_id in self._cancelled:
                        return None
                return (job.genome, float(result.fitness))
            except Exception:
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_s)
                    # try next worker on retry
                    job.worker_url = self._next_worker()
        return None

    def _call_worker(self, url: str, job_id: str, genome: Genome) -> EvalResult:
        """HTTP POST to worker; returns EvalResult or raises on error/timeout."""
        import urllib.request
        import json as _json

        genome_b64 = base64.b64encode(pickle.dumps(genome)).decode()
        payload = _json.dumps({"job_id": job_id, "genome_b64": genome_b64,
                               "timeout_s": self.timeout_s}).encode()

        req = urllib.request.Request(
            f"{url.rstrip('/')}/evaluate",
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s + 5) as resp:
                body = _json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Worker HTTP {exc.code}: {exc.read().decode(errors='replace')}")

        return EvalResult(
            job_id=body["job_id"],
            fitness=body.get("fitness"),
            error=body.get("error"),
            duration_s=body.get("duration_s", 0.0),
        )


# ── Worker server ─────────────────────────────────────────────────────────────

class RemoteWorkerServer:
    """Lightweight FastAPI server that evaluates genomes on behalf of a master.

    The fitness function is provided at construction time and must be importable
    in the worker's Python environment.  The master only sends the genome
    structure — not the fitness function code.

    Parameters
    ----------
    fitness_fn:
        Callable ``(Genome) -> float``.
    token:
        Shared secret; requests without a matching ``Authorization: Bearer``
        header are rejected with HTTP 401.
    host:
        Bind address (default ``"127.0.0.1"`` — localhost only).
        Use ``"0.0.0.0"`` for network access, but only behind a trusted network.
    port:
        TCP port to listen on (default 8700).
    """

    def __init__(
        self,
        fitness_fn: Callable[[Genome], float],
        token: str = "",
        host: str = "127.0.0.1",
        port: int = 8700,
    ) -> None:
        self.fitness_fn = fitness_fn
        self.token = token
        self.host = host
        self.port = port

    def run(self) -> None:
        """Start the worker server (blocking).  Stop with Ctrl-C."""
        try:
            import uvicorn
        except ImportError as exc:
            raise ImportError(
                "uvicorn is required to run the RemoteWorkerServer. "
                "Install it with: pip install uvicorn"
            ) from exc
        app = self._build_app()
        uvicorn.run(app, host=self.host, port=self.port)

    def _build_app(self):
        """Build and return the FastAPI app (separated for testability)."""
        try:
            from fastapi import FastAPI, Header
            from fastapi.responses import JSONResponse
        except ImportError as exc:
            raise ImportError(
                "fastapi is required to run the RemoteWorkerServer. "
                "Install it with: pip install fastapi"
            ) from exc
        if _EvalRequest is None:
            raise ImportError(
                "pydantic is required to run the RemoteWorkerServer. "
                "Install it with: pip install pydantic"
            )

        app = FastAPI(title="YANE Remote Worker")
        fitness_fn = self.fitness_fn
        token = self.token

        @app.post("/evaluate")
        def evaluate(body: _EvalRequest, authorization: str = Header(default="")):
            # Auth check: compare full "Bearer <token>" header value
            if token and not hmac.compare_digest(authorization, f"Bearer {token}"):
                from fastapi import HTTPException
                raise HTTPException(status_code=401, detail="Unauthorized")

            t0 = time.perf_counter()
            try:
                genome_bytes = base64.b64decode(body.genome_b64, validate=True)
                # Security: unpickle only from authenticated senders (token checked above)
                genome = pickle.loads(genome_bytes)
            except Exception as exc:
                return JSONResponse({"job_id": body.job_id, "fitness": None,
                                     "error": f"Deserialise error: {exc}",
                                     "duration_s": time.perf_counter() - t0})

            try:
                import concurrent.futures
                pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = pool.submit(fitness_fn, genome)
                try:
                    fitness = float(future.result(timeout=body.timeout_s))
                    error = None
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    pool.shutdown(wait=False, cancel_futures=True)
                    fitness = None
                    error = f"Evaluation timed out after {body.timeout_s}s"
                except Exception:
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise
                else:
                    pool.shutdown(wait=True, cancel_futures=True)
            except Exception as exc:
                fitness = None
                error = str(exc)

            return JSONResponse({"job_id": body.job_id, "fitness": fitness,
                                 "error": error,
                                 "duration_s": time.perf_counter() - t0})

        @app.get("/health")
        def health():
            return {"ok": True}

        return app
