"""Tests for the /configure API endpoint."""
import unittest
import pytest

try:
    from fastapi.testclient import TestClient
    _HAS_TESTCLIENT = True
except ImportError:
    _HAS_TESTCLIENT = False


@pytest.mark.ci
@unittest.skipUnless(_HAS_TESTCLIENT, "fastapi[test] not installed")
class TestConfigureEndpoint(unittest.TestCase):

    def setUp(self):
        from yane.api.server import app, state
        from yane.neuro_evolution import NeuroEvolution
        # Reset server state before each test to avoid cross-test pollution.
        app.dependency_overrides = {}
        state.__init__()
        self.client = TestClient(app)
        self.state = state

    def _post(self, **kwargs):
        return self.client.post("/configure", json=kwargs)

    def test_minimal_configure_returns_200(self):
        r = self._post(n_inputs=2, n_outputs=1)
        self.assertEqual(r.status_code, 200)

    def test_minimal_configure_returns_config_dict(self):
        r = self._post(n_inputs=3, n_outputs=2)
        body = r.json()
        self.assertEqual(body["n_inputs"], 3)
        self.assertEqual(body["n_outputs"], 2)

    def test_configure_population_size(self):
        r = self._post(n_inputs=2, n_outputs=1, population_size=50)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.state._population_size, 50)

    def test_configure_seed(self):
        r = self._post(n_inputs=2, n_outputs=1, seed=42)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.state._seed, 42)

    def test_configure_max_nodes(self):
        r = self._post(n_inputs=2, n_outputs=1, max_nodes=20)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.state._max_nodes, 20)

    def test_configure_max_connections(self):
        r = self._post(n_inputs=2, n_outputs=1, max_connections=30)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.state._max_connections, 30)

    def test_configure_n_initial_hidden(self):
        r = self._post(n_inputs=2, n_outputs=1, n_initial_hidden=3)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.state._n_initial_hidden, 3)

    def test_configure_stateful_false(self):
        r = self._post(n_inputs=2, n_outputs=1, stateful=False)
        self.assertEqual(r.status_code, 200)
        self.assertFalse(self.state._stateful)

    def test_configure_target_species(self):
        r = self._post(n_inputs=2, n_outputs=1, target_species=8)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.state._population._target_species, 8)

    def test_configure_lamarck_steps(self):
        r = self._post(n_inputs=2, n_outputs=1, lamarck_steps=5, lamarck_sigma=0.8)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.state._lamarck.steps, 5)
        self.assertAlmostEqual(self.state._lamarck.sigma, 0.8)

    def test_configure_lamarck_adaptive(self):
        r = self._post(n_inputs=2, n_outputs=1,
                       lamarck_adaptive_max_steps=4,
                       lamarck_adaptive_top_k=0.3)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.state._lamarck.max_steps, 4)
        self.assertAlmostEqual(self.state._lamarck.top_k, 0.3)

    def test_configure_missing_n_inputs_returns_422(self):
        r = self._post(n_outputs=1)
        self.assertEqual(r.status_code, 422)

    def test_configure_missing_n_outputs_returns_422(self):
        r = self._post(n_inputs=2)
        self.assertEqual(r.status_code, 422)

    def test_configure_invalid_n_inputs_zero_returns_422(self):
        r = self._post(n_inputs=0, n_outputs=1)
        self.assertEqual(r.status_code, 422)

    def test_configure_all_params(self):
        r = self._post(
            n_inputs=4, n_outputs=2, max_nodes=50, max_connections=100,
            n_initial_hidden=2, stateful=True, population_size=30,
            target_species=6, lamarck_steps=3, lamarck_sigma=1.2, seed=99,
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.state._n_inputs, 4)
        self.assertEqual(self.state._n_outputs, 2)
        self.assertEqual(self.state._seed, 99)


if __name__ == "__main__":
    unittest.main()
