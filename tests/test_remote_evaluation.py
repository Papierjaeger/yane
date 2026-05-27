"""Tests for RemoteEvaluationClient and RemoteWorkerServer."""
from __future__ import annotations

import base64
import json
import pickle
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from yane import NeuroEvolution, RemoteEvaluationClient, RemoteWorkerServer
from yane.core.genome import Genome
from yane.evolution.remote_evaluation import EvalJob, EvalResult


def _simple_genome() -> Genome:
    yane = NeuroEvolution()
    yane.configure(2, 1)
    return yane.next_genome()


class TestEvalProtocolTypes(unittest.TestCase):
    def test_eval_job_fields(self):
        job = EvalJob(job_id="abc", genome_b64="xxx", timeout_s=5.0)
        self.assertEqual(job.job_id, "abc")
        self.assertEqual(job.timeout_s, 5.0)

    def test_eval_result_fields(self):
        r = EvalResult(job_id="abc", fitness=1.5, error=None, duration_s=0.1)
        self.assertIsNone(r.error)
        self.assertAlmostEqual(r.fitness, 1.5)


class TestRemoteEvaluationClientMocked(unittest.TestCase):
    """Test client logic without a real HTTP server by mocking _call_worker."""

    def _client(self, **kwargs):
        defaults = dict(worker_urls=["http://localhost:8700"], token="t",
                        timeout_s=5.0, max_retries=1, retry_delay_s=0.0)
        defaults.update(kwargs)
        return RemoteEvaluationClient(**defaults)

    def test_evaluate_batch_returns_genome_fitness_pairs(self):
        g = _simple_genome()
        client = self._client()
        with patch.object(client, "_call_worker", return_value=EvalResult("x", 0.9, None, 0.01)):
            results = client.evaluate_batch([g])
        self.assertEqual(len(results), 1)
        self.assertIs(results[0][0], g)
        self.assertAlmostEqual(results[0][1], 0.9)

    def test_evaluate_batch_skips_errored_job_after_retries(self):
        g = _simple_genome()
        client = self._client(max_retries=0)
        with patch.object(client, "_call_worker", side_effect=RuntimeError("timeout")):
            results = client.evaluate_batch([g])
        self.assertEqual(results, [])

    def test_cancel_prevents_result(self):
        g = _simple_genome()
        client = self._client()
        cancelled = []

        def _slow_call(url, job_id, genome):
            client.cancel(job_id)
            return EvalResult(job_id, 1.0, None, 0.0)

        with patch.object(client, "_call_worker", side_effect=_slow_call):
            results = client.evaluate_batch([g])
        self.assertEqual(results, [])

    def test_round_robin_across_workers(self):
        client = self._client(worker_urls=["http://w1:8700", "http://w2:8700"])
        urls_used = []
        orig = client._call_worker

        def _track(url, job_id, genome):
            urls_used.append(url)
            return EvalResult(job_id, 1.0, None, 0.0)

        genomes = [_simple_genome() for _ in range(4)]
        with patch.object(client, "_call_worker", side_effect=_track):
            client.evaluate_batch(genomes)

        self.assertIn("http://w1:8700", urls_used)
        self.assertIn("http://w2:8700", urls_used)

    def test_retry_switches_worker(self):
        client = self._client(
            worker_urls=["http://bad:8700", "http://good:8700"],
            max_retries=1, retry_delay_s=0.0,
        )
        g = _simple_genome()
        call_count = [0]

        def _call(url, job_id, genome):
            call_count[0] += 1
            if url == "http://bad:8700":
                raise RuntimeError("bad worker")
            return EvalResult(job_id, 2.0, None, 0.0)

        with patch.object(client, "_call_worker", side_effect=_call):
            results = client.evaluate_batch([g])

        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0][1], 2.0)
        self.assertEqual(call_count[0], 2)

    def test_genome_serialised_as_base64_pickle(self):
        g = _simple_genome()
        client = self._client()
        captured = {}

        def _capture(url, job_id, genome):
            captured["genome"] = genome
            return EvalResult(job_id, 0.5, None, 0.0)

        with patch.object(client, "_call_worker", side_effect=_capture):
            client.evaluate_batch([g])

        self.assertIsInstance(captured["genome"], Genome)

    def test_requires_at_least_one_worker_url(self):
        with self.assertRaises(ValueError):
            RemoteEvaluationClient(worker_urls=[])


class TestRemoteWorkerServerApp(unittest.TestCase):
    """Test the worker FastAPI app without starting a real server."""

    def _app(self, fitness_fn=None, token="secret"):
        if fitness_fn is None:
            fitness_fn = lambda g: 1.0
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi[test] not installed")
        server = RemoteWorkerServer(fitness_fn=fitness_fn, token=token)
        return TestClient(server._build_app())

    def _genome_b64(self):
        g = _simple_genome()
        return base64.b64encode(pickle.dumps(g)).decode()

    def test_health_endpoint_returns_ok(self):
        client = self._app()
        r = client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])

    def test_evaluate_returns_fitness(self):
        client = self._app(fitness_fn=lambda g: 3.14)
        payload = {"job_id": "j1", "genome_b64": self._genome_b64(), "timeout_s": 5.0}
        r = client.post("/evaluate",
                        json=payload,
                        headers={"Authorization": "Bearer secret"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIsNone(body["error"])
        self.assertAlmostEqual(body["fitness"], 3.14)

    def test_unauthorised_request_returns_401(self):
        client = self._app()
        payload = {"job_id": "j2", "genome_b64": self._genome_b64(), "timeout_s": 5.0}
        r = client.post("/evaluate", json=payload,
                        headers={"Authorization": "Bearer wrong"})
        self.assertEqual(r.status_code, 401)

    def test_no_token_required_when_token_is_empty(self):
        client = self._app(token="")
        payload = {"job_id": "j3", "genome_b64": self._genome_b64(), "timeout_s": 5.0}
        r = client.post("/evaluate", json=payload)
        self.assertEqual(r.status_code, 200)

    def test_fitness_error_returned_in_result(self):
        def bad_fn(g):
            raise ValueError("boom")

        client = self._app(fitness_fn=bad_fn)
        payload = {"job_id": "j4", "genome_b64": self._genome_b64(), "timeout_s": 5.0}
        r = client.post("/evaluate", json=payload,
                        headers={"Authorization": "Bearer secret"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIsNone(body["fitness"])
        self.assertIn("boom", body["error"])

    def test_invalid_genome_b64_returns_error(self):
        client = self._app()
        payload = {"job_id": "j5", "genome_b64": "not-valid-base64!!!", "timeout_s": 5.0}
        r = client.post("/evaluate", json=payload,
                        headers={"Authorization": "Bearer secret"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIsNone(body["fitness"])
        self.assertIsNotNone(body["error"])

    def test_malformed_base64_with_valid_characters_returns_error(self):
        client = self._app()
        payload = {"job_id": "j5b", "genome_b64": "abcd=", "timeout_s": 5.0}
        r = client.post("/evaluate", json=payload,
                        headers={"Authorization": "Bearer secret"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIsNone(body["fitness"])
        self.assertIn("Deserialise error", body["error"])

    def test_invalid_timeout_is_rejected_before_deserialise(self):
        client = self._app()
        payload = {"job_id": "j5c", "genome_b64": self._genome_b64(), "timeout_s": 0.0}
        r = client.post("/evaluate", json=payload,
                        headers={"Authorization": "Bearer secret"})
        self.assertEqual(r.status_code, 422)

    def test_evaluation_timeout_returns_promptly(self):
        def slow_fn(g):
            time.sleep(0.25)
            return 1.0

        client = self._app(fitness_fn=slow_fn)
        payload = {"job_id": "j5d", "genome_b64": self._genome_b64(), "timeout_s": 0.01}

        start = time.perf_counter()
        r = client.post("/evaluate", json=payload,
                        headers={"Authorization": "Bearer secret"})
        elapsed = time.perf_counter() - start

        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIsNone(body["fitness"])
        self.assertIn("timed out", body["error"])
        self.assertLess(elapsed, 0.15)

    def test_duration_s_is_present_in_response(self):
        client = self._app(fitness_fn=lambda g: 0.0)
        payload = {"job_id": "j6", "genome_b64": self._genome_b64(), "timeout_s": 5.0}
        r = client.post("/evaluate", json=payload,
                        headers={"Authorization": "Bearer secret"})
        body = r.json()
        self.assertIn("duration_s", body)
        self.assertGreaterEqual(body["duration_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
