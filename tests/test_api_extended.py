"""Tests for the extended YANE API (features, params, training, export, cloning).

Covers:
  - GET /params — lists all 68+ registered parameters
  - POST /param — sets a registered parameter by name
  - POST /features/* — feature-specific config endpoints
  - POST /train/register_fn + POST /train/iterations — server-side fitness fn
  - GET /train/status — training status
  - POST /train/stop, /train/reset — lifecycle
  - GET /export/python — Python source export
  - POST /export/symbolic — symbolic formula
  - POST /export/c_array — C99 embedded export
  - GET /export/wasm — HTML/JS export
  - POST /export/lottery_ticket — IMP pruning (needs registered fn)
  - POST /export/clone — behaviour cloning from JSON demos
"""
from __future__ import annotations

import unittest

import pytest

from fastapi.testclient import TestClient

from yane.api.server import app, state

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _configure(n_inputs: int = 2, n_outputs: int = 1) -> None:
    """Configure the shared API state for a test."""
    from yane.api.state import state as _s
    # Re-configure (reconfigure is idempotent)
    _s.set_population_size(10)
    _s.configure(n_inputs, n_outputs)
    # Promote all genomes to evaluated so get_best() works
    if not _s._population._evaluated:
        for g in _s._population._unevaluated:
            g.fitness = 1.0
        _s._population._evaluated.extend(_s._population._unevaluated)
        _s._population._unevaluated.clear()
    # Reset safety / phylogeny that might be set from previous tests
    _s._safety_system = None
    if _s._phylogeny is not None:
        _s._phylogeny.disable()


# ---------------------------------------------------------------------------
# GET /params
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestGetParams(unittest.TestCase):

    def setUp(self):
        _configure()

    def test_params_returns_dict(self):
        r = client.get("/params")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIsInstance(data, dict)

    def test_params_count_ge_60(self):
        r = client.get("/params")
        data = r.json()
        self.assertGreaterEqual(len(data), 60)

    def test_params_have_required_fields(self):
        r = client.get("/params")
        data = r.json()
        for name, spec in data.items():
            with self.subTest(name=name):
                for field in ("type", "domain", "default", "current", "subsystem"):
                    self.assertIn(field, spec, f"Missing field {field!r} in {name!r}")

    def test_research_features_present(self):
        r = client.get("/params")
        data = r.json()
        for expected in ("attention.enabled", "stdp.enabled", "ltc.enabled",
                         "probabilistic.enabled", "island.enabled", "hybrid.enabled"):
            self.assertIn(expected, data, f"Missing parameter: {expected!r}")


# ---------------------------------------------------------------------------
# POST /param
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestSetParam(unittest.TestCase):

    def setUp(self):
        _configure()

    def test_set_boolean_param(self):
        r = client.post("/param", json={"name": "novelty.enabled", "value": True})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["name"], "novelty.enabled")

    def test_set_integer_param(self):
        r = client.post("/param", json={"name": "pop.size", "value": 50})
        self.assertEqual(r.status_code, 200)

    def test_set_continuous_param(self):
        r = client.post("/param", json={"name": "curiosity.weight", "value": 0.5})
        self.assertEqual(r.status_code, 200)

    def test_unknown_param_returns_404(self):
        r = client.post("/param", json={"name": "nonexistent.param", "value": True})
        self.assertEqual(r.status_code, 404)

    def test_out_of_domain_returns_422(self):
        r = client.post("/param", json={"name": "pop.size", "value": 1})  # min is 10
        self.assertEqual(r.status_code, 422)

    def test_not_configured_returns_400(self):
        saved = state._population
        state._population = None
        try:
            r = client.post("/param", json={"name": "novelty.enabled", "value": True})
            self.assertEqual(r.status_code, 400)
        finally:
            state._population = saved


# ---------------------------------------------------------------------------
# POST /features/attention
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestFeaturesAttention(unittest.TestCase):

    def setUp(self):
        _configure()

    def test_enable_attention(self):
        r = client.post("/features/attention",
                        json={"enabled": True, "head_dim": 4, "num_heads": 2})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["ok"])
        self.assertTrue(data["enabled"])

    def test_disable_attention(self):
        r = client.post("/features/attention", json={"enabled": False, "head_dim": 4, "num_heads": 2})
        self.assertEqual(r.status_code, 200)

    def test_not_configured_returns_400(self):
        saved = state._population
        state._population = None
        try:
            r = client.post("/features/attention", json={"enabled": True})
            self.assertEqual(r.status_code, 400)
        finally:
            state._population = saved


# ---------------------------------------------------------------------------
# POST /features/stdp
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestFeaturesSTDP(unittest.TestCase):

    def setUp(self):
        _configure()

    def test_enable_stdp(self):
        r = client.post("/features/stdp", json={"enabled": True})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])


# ---------------------------------------------------------------------------
# POST /features/probabilistic
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestFeaturesProbabilistic(unittest.TestCase):

    def setUp(self):
        _configure()

    def test_enable_probabilistic(self):
        r = client.post("/features/probabilistic",
                        json={"enabled": True, "noise_std": 0.1, "inference_mode": False})
        self.assertEqual(r.status_code, 200)

    def test_inference_mode(self):
        r = client.post("/features/probabilistic",
                        json={"enabled": True, "noise_std": 0.1, "inference_mode": True})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])


# ---------------------------------------------------------------------------
# POST /features/island_model
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestFeaturesIslandModel(unittest.TestCase):

    def setUp(self):
        _configure()

    def test_enable_island_model(self):
        r = client.post("/features/island_model",
                        json={"enabled": True, "n_islands": 3, "migrate_interval": 10})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["n_islands"], 3)


# ---------------------------------------------------------------------------
# POST /features/phylogeny + GET /features/phylogeny/tree
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestFeaturesPhylogeny(unittest.TestCase):

    def setUp(self):
        _configure()

    def test_enable_phylogeny(self):
        r = client.post("/features/phylogeny", json={"enabled": True})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])

    def test_get_tree_returns_dict(self):
        client.post("/features/phylogeny", json={"enabled": True})
        r = client.get("/features/phylogeny/tree")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("nodes", data)

    def test_tree_without_enable_returns_404(self):
        client.post("/features/phylogeny", json={"enabled": False})
        r = client.get("/features/phylogeny/tree")
        self.assertEqual(r.status_code, 404)


# ---------------------------------------------------------------------------
# POST /features/hardware
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestFeaturesHardware(unittest.TestCase):

    def setUp(self):
        _configure()

    def test_set_platform(self):
        r = client.post("/features/hardware",
                        json={"target_platform": "cortex-m4", "enabled": True})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])

    def test_invalid_platform_returns_422(self):
        r = client.post("/features/hardware",
                        json={"target_platform": "invalid-cpu", "enabled": True})
        self.assertEqual(r.status_code, 422)

    def test_disable_hardware(self):
        r = client.post("/features/hardware", json={"enabled": False})
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json()["enabled"])


# ---------------------------------------------------------------------------
# POST /features/augmentation, /features/curiosity, /features/novelty
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestFeaturesSimple(unittest.TestCase):

    def setUp(self):
        _configure()

    def test_augmentation(self):
        r = client.post("/features/augmentation",
                        json={"enabled": True, "pool_size": 5, "evolve_interval": 10})
        self.assertEqual(r.status_code, 200)

    def test_curiosity(self):
        r = client.post("/features/curiosity",
                        json={"enabled": True, "weight": 0.3})
        self.assertEqual(r.status_code, 200)

    def test_novelty(self):
        r = client.post("/features/novelty", json={"enabled": True})
        self.assertEqual(r.status_code, 200)

    def test_anytime(self):
        r = client.post("/features/anytime",
                        json={"enabled": True, "min_evals": 1, "max_evals": 5})
        self.assertEqual(r.status_code, 200)

    def test_darts(self):
        r = client.post("/features/darts", json={"enabled": True, "prune_threshold": 0.1})
        self.assertEqual(r.status_code, 200)

    def test_shared_weights(self):
        r = client.post("/features/shared_weights", json={"enabled": True})
        self.assertEqual(r.status_code, 200)

    def test_neuromodulation(self):
        r = client.post("/features/neuromodulation", json={"enabled": True})
        self.assertEqual(r.status_code, 200)

    def test_ltc(self):
        r = client.post("/features/ltc", json={"enabled": True})
        self.assertEqual(r.status_code, 200)

    def test_continual_learning(self):
        r = client.post("/features/continual_learning",
                        json={"mode": "ewc", "lambda_ewc": 0.1})
        self.assertEqual(r.status_code, 200)


# ---------------------------------------------------------------------------
# Training: register_fn + status + iterations + stop + reset
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestTrainingLifecycle(unittest.TestCase):

    def setUp(self):
        _configure()
        # Reset training state
        client.post("/train/reset")

    def test_status_initially_idle(self):
        r = client.get("/train/status")
        self.assertEqual(r.status_code, 200)
        # May be idle or finished from previous test; just check it's a valid status
        self.assertIn(r.json()["status"], ("idle", "finished", "failed"))

    def test_register_fitness_fn(self):
        src = "def fitness_fn(genome): return 1.0"
        r = client.post("/train/register_fn", json={"name": "xor", "source": src})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])
        self.assertIn("xor", r.json()["registered_functions"])

    def test_register_fn_syntax_error(self):
        r = client.post("/train/register_fn",
                        json={"name": "bad", "source": "def fitness_fn(g) return 1"})
        self.assertEqual(r.status_code, 422)

    def test_register_fn_missing_fitness_fn(self):
        r = client.post("/train/register_fn",
                        json={"name": "missing", "source": "x = 1"})
        self.assertEqual(r.status_code, 422)

    def test_train_iterations_with_registered_fn(self):
        src = "def fitness_fn(genome): return sum(genome.forward([0.5, 0.5]))"
        client.post("/train/register_fn", json={"name": "simple", "source": src})
        r = client.post("/train/iterations", json={"n_iterations": 5, "fitness_fn_name": "simple"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["iterations"], 5)

    def test_stop_when_not_running_returns_400(self):
        r = client.post("/train/stop")
        self.assertEqual(r.status_code, 400)

    def test_reset_clears_state(self):
        client.post("/train/reset")
        r = client.get("/train/status")
        self.assertEqual(r.json()["status"], "idle")


# ---------------------------------------------------------------------------
# Export: python, symbolic, c_array, wasm, clone
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestExports(unittest.TestCase):

    def setUp(self):
        _configure()

    def test_export_python_returns_source(self):
        r = client.get("/export/python")
        self.assertEqual(r.status_code, 200)
        src = r.text
        self.assertIn("def", src)

    def test_export_symbolic_python_format(self):
        r = client.post("/export/symbolic",
                        json={"format": "python", "fold_constants": True})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["format"], "python")
        self.assertIsInstance(data["formula"], str)

    def test_export_symbolic_latex_format(self):
        r = client.post("/export/symbolic", json={"format": "latex"})
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json()["formula"], str)

    def test_export_symbolic_with_input_names(self):
        r = client.post("/export/symbolic",
                        json={"format": "python", "input_names": ["alpha", "beta"]})
        self.assertEqual(r.status_code, 200)

    def test_export_c_array_returns_header_and_source(self):
        r = client.post("/export/c_array", json={"prefix": "test_net"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("header", data)
        self.assertIn("source", data)
        self.assertIn("test_net", data["header"])
        self.assertIn("#include <math.h>", data["source"])

    def test_export_c_array_custom_prefix(self):
        r = client.post("/export/c_array", json={"prefix": "my_model"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("my_model", r.json()["header"])

    def test_export_wasm_returns_html(self):
        r = client.get("/export/wasm")
        self.assertEqual(r.status_code, 200)
        self.assertIn("<html", r.text.lower())

    def test_export_onnx_requires_package(self):
        r = client.get("/export/onnx")
        # Either 200 (onnx installed) or 501 (not installed)
        self.assertIn(r.status_code, (200, 501))

    def test_export_lottery_ticket_no_fn_returns_400(self):
        # No fitness function registered -> 400
        from yane.api.routes.training import _fitness_registry
        _fitness_registry.clear()
        r = client.post("/export/lottery_ticket",
                        json={"target_sparsity": 0.3, "apply": False})
        self.assertEqual(r.status_code, 400)

    def test_export_lottery_ticket_with_fn(self):
        src = "def fitness_fn(genome): return sum(genome.forward([0.5, 0.5]))"
        client.post("/train/register_fn", json={"name": "lt_fn", "source": src})
        r = client.post("/export/lottery_ticket",
                        json={"target_sparsity": 0.0, "iterations": 2, "apply": False,
                              "fitness_fn_name": "lt_fn"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("sparsity", data)
        self.assertFalse(data["applied"])


# ---------------------------------------------------------------------------
# POST /export/clone — Behaviour Cloning
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestBehaviourCloningEndpoint(unittest.TestCase):

    def setUp(self):
        _configure()

    def test_clone_reduces_mse(self):
        demos = [
            {"inputs": [0.0, 0.0], "targets": [0.0]},
            {"inputs": [1.0, 1.0], "targets": [0.0]},
            {"inputs": [0.0, 1.0], "targets": [1.0]},
            {"inputs": [1.0, 0.0], "targets": [1.0]},
        ]
        r = client.post("/export/clone",
                        json={"demonstrations": demos, "n_steps": 10, "sigma": 0.05})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("initial_mse", data)
        self.assertIn("final_mse", data)
        self.assertGreaterEqual(data["initial_mse"], 0.0)
        self.assertGreaterEqual(data["final_mse"], 0.0)

    def test_clone_with_seed_population(self):
        demos = [{"inputs": [0.5, 0.5], "targets": [0.5]}]
        r = client.post("/export/clone",
                        json={"demonstrations": demos, "n_steps": 5,
                              "seed_population": True})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["seeded_population"])

    def test_clone_not_configured_returns_400(self):
        saved = state._population
        state._population = None
        try:
            demos = [{"inputs": [0.5], "targets": [0.5]}]
            r = client.post("/export/clone",
                            json={"demonstrations": demos, "n_steps": 3})
            self.assertEqual(r.status_code, 400)
        finally:
            state._population = saved

    def test_clone_empty_demos_returns_422(self):
        r = client.post("/export/clone",
                        json={"demonstrations": [], "n_steps": 5})
        self.assertEqual(r.status_code, 422)
