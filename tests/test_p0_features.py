"""Tests for P0 features: Event-System, Anomaly Detection, Fitness Transform,
Genome Export, Validation Set, Config Persistence."""
from __future__ import annotations

import json
import math
import types
import tempfile
import os

import pytest


# ---------------------------------------------------------------------------
# Adaptive Evaluation Budgeting
# ---------------------------------------------------------------------------

def test_anytime_eval_promotes_competitive_genomes():
    from yane.evolution.evaluation import EvaluationRunner
    from yane.evolution.lamarck_refiner import LamarckRefiner
    from yane.evolution.population import Population
    from yane.core.genome import Genome

    pop = Population(max_size=10, initial_genome=Genome())
    runner = EvaluationRunner()
    runner.configure_anytime_eval(True, min_evals=1, max_evals=3, promotion_frac=0.5)
    calls = {"n": 0}

    def fitness(_genome):
        calls["n"] += 1
        return 10.0

    result = runner.run(pop.select_for_evaluation(), fitness, pop, LamarckRefiner())
    assert result.n_fitness_calls == 3
    assert calls["n"] == 3
    assert runner.anytime_promoted == 1


def test_anytime_eval_skips_extra_calls_for_weak_genome():
    from yane.evolution.evaluation import EvaluationRunner
    from yane.evolution.lamarck_refiner import LamarckRefiner
    from yane.evolution.population import Population
    from yane.core.genome import Genome

    pop = Population(max_size=10, initial_genome=Genome())
    for score in [10.0, 9.0, 8.0, 7.0, 6.0]:
        g = pop.select_for_evaluation()
        pop.submit(g, score)
    runner = EvaluationRunner()
    runner.configure_anytime_eval(True, min_evals=1, max_evals=4, promotion_frac=0.2)
    result = runner.run(pop.select_for_evaluation(), lambda _g: 0.0, pop, LamarckRefiner())
    assert result.n_fitness_calls == 1
    assert runner.anytime_saved_calls == 3


def test_structured_evaluator_components_encode_and_score_policy():
    from yane.evolution.evaluator_components import GraphPolicyScore, StateEncoder

    encoder = StateEncoder(mode="mixed", sizes=(2, 3), scales=(1.0, 2.0))
    assert encoder.output_dim == 7
    assert encoder.encode((1, 2)) == [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    transitions = {
        0: {0: [(1.0, 1, 0.0, False)], 1: [(1.0, 0, -10.0, False)]},
        1: {0: [(1.0, 1, 0.0, False)], 1: [(1.0, 2, 1.0, True)]},
        2: {0: [(1.0, 2, 0.0, True)], 1: [(1.0, 2, 0.0, True)]},
    }
    scorer = GraphPolicyScore(transitions, n_actions=2, terminal_reward=0.0)
    assert scorer.action_score(1, 1) > scorer.action_score(1, 0)
    assert scorer.action_score(0, 1) < 0.0


def test_neuroevolution_target_species_band_diagnostics():
    from yane.neuro_evolution import NeuroEvolution

    yane = NeuroEvolution()
    yane.set_target_species(n_min=3, n_max=6, tune_interval=4)
    yane.configure(2, 1)
    info = yane.population_memory_info()
    assert info["target_species_min"] == 3
    assert info["target_species_max"] == 6
    assert info["compat_tune_interval"] == 4
    assert info["species_tuning_enabled"] is True


def test_neuroevolution_target_species_none_disables_tuning():
    from yane.neuro_evolution import NeuroEvolution

    yane = NeuroEvolution()
    yane.set_target_species(None)
    yane.configure(2, 1)
    assert yane.population_memory_info()["species_tuning_enabled"] is False


def test_adaptive_recovery_diversity_boost_injects_genomes():
    from yane.neuro_evolution import NeuroEvolution

    yane = NeuroEvolution()
    yane.configure(2, 1)
    yane.set_adaptive_recovery(warmup=0, cooldown=2, injection_frac=0.2)
    pop = yane.population
    for i in range(10):
        g = pop.select_for_evaluation()
        pop.submit(g, 1.0 if i == 0 else 0.0)
    before = pop._n_diversity_injection
    mem = yane.population_memory_info()
    mem.update({"fitness_iqr": 0.0, "generation": 10})
    yane._tick_adaptive_recovery(mem, 100, lambda *_args: None)
    assert pop._n_diversity_injection > before
    assert yane.population_memory_info()["recovery_events"]


def test_adaptive_recovery_cooldown_prevents_repeated_trigger():
    from yane.neuro_evolution import NeuroEvolution

    yane = NeuroEvolution()
    yane.configure(2, 1)
    yane.set_adaptive_recovery(warmup=0, cooldown=10, injection_frac=0.1)
    pop = yane.population
    for i in range(10):
        g = pop.select_for_evaluation()
        pop.submit(g, float(i))
    mem = yane.population_memory_info()
    mem.update({"fitness_iqr": 0.0, "generation": 10})
    yane._tick_adaptive_recovery(mem, 100, lambda *_args: None)
    yane._tick_adaptive_recovery({**mem, "generation": 11}, 110, lambda *_args: None)
    assert len(yane._recovery_events) == 1


def test_adaptive_recovery_escalates_after_failed_cooldown():
    from yane.neuro_evolution import NeuroEvolution

    yane = NeuroEvolution()
    yane.configure(2, 1)
    yane.set_adaptive_recovery(warmup=0, cooldown=2, strategies=["diversity_boost", "partial_restart"])
    pop = yane.population
    for i in range(10):
        g = pop.select_for_evaluation()
        pop.submit(g, 1.0 if i == 0 else 0.0)
    mem = yane.population_memory_info()
    mem.update({"fitness_iqr": 0.0, "generation": 10, "max_fitness": 1.0})
    yane._tick_adaptive_recovery(mem, 100, lambda *_args: None)
    yane._tick_adaptive_recovery({**mem, "generation": 12}, 120, lambda *_args: None)
    assert yane._recovery_strategy_index == 1


def test_adaptive_recovery_guarded_early_stop_requires_signal():
    from yane.neuro_evolution import NeuroEvolution

    yane = NeuroEvolution()
    yane.configure(2, 1)
    yane.set_adaptive_recovery(warmup=0, cooldown=1, early_stopping_patience=3)
    yane._recovery_best_fitness = 0.0
    yane._recovery_last_improvement_generation = 0
    mem = yane.population_memory_info()
    mem.update({"fitness_iqr": 0.0, "generation": 5, "max_fitness": 0.0})
    reason = yane._tick_adaptive_recovery(mem, 50, lambda *_args: None)
    assert reason is not None
    assert yane.stopped_early is True


def test_eval_middleware_lifo_order_and_diagnostics():
    from yane.neuro_evolution import NeuroEvolution

    yane = NeuroEvolution()
    yane.configure(1, 1)
    calls = []

    def first(genome, eval_fn, ctx):
        calls.append("first_before")
        value = eval_fn(genome)
        calls.append("first_after")
        ctx.diagnostics["first"] = True
        return value

    def second(genome, eval_fn, ctx):
        calls.append("second_before")
        value = eval_fn(genome)
        calls.append("second_after")
        ctx.diagnostics["second"] = True
        return value

    yane.add_eval_middleware(first)
    yane.add_eval_middleware(second)
    g = yane.population.select_for_evaluation()
    result = yane._run_evaluations(g, lambda _g: 1.0)
    assert result.fitness == 1.0
    assert calls == ["second_before", "first_before", "first_after", "second_after"]
    info = yane.population_memory_info()
    assert info["eval_middleware"]["first"] is True
    assert info["eval_middleware"]["second"] is True


def test_caching_middleware_invalidates_on_weight_change():
    from yane.evolution.eval_middleware import CachingMiddleware
    from yane.neuro_evolution import NeuroEvolution

    yane = NeuroEvolution()
    yane.configure(1, 1)
    cache = CachingMiddleware(maxsize=8)
    yane.add_eval_middleware(cache)
    g = yane.population.select_for_evaluation()
    from yane.core.connection import Connection
    conn = Connection(g.output_nodes[0], innovation=yane._tracker.get_connection(
        g.input_nodes[0].innovation,
        g.output_nodes[0].innovation,
    ))
    g.input_nodes[0].connections.append(conn)
    g._invalidate_topology()
    calls = {"n": 0}

    def fitness(_g):
        calls["n"] += 1
        return float(calls["n"])

    assert yane._run_evaluations(g, fitness).fitness == 1.0
    assert yane._run_evaluations(g, fitness).fitness == 1.0
    conn.weight += 0.25
    assert yane._run_evaluations(g, fitness).fitness == 2.0
    assert cache.hits == 1
    assert cache.misses == 2


def test_retry_middleware_retries_flaky_evaluator():
    from yane.evolution.eval_middleware import RetryMiddleware
    from yane.neuro_evolution import NeuroEvolution

    yane = NeuroEvolution()
    yane.configure(1, 1)
    yane.add_eval_middleware(RetryMiddleware(n=3, aggregation="mean"))
    g = yane.population.select_for_evaluation()
    calls = {"n": 0}

    def flaky(_g):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("temporary")
        return 4.0

    result = yane._run_evaluations(g, flaky)
    assert result.fitness == 4.0
    assert calls["n"] == 3
    assert yane.population_memory_info()["eval_middleware"]["retry_count"] == 1


# ---------------------------------------------------------------------------
# Event System
# ---------------------------------------------------------------------------

class TestEventBus:
    def test_on_off_emit(self):
        from yane.evolution.events import EventBus
        bus = EventBus()
        received = []
        def handler(p): received.append(p)
        bus.on("test", handler)
        bus.emit("test", 42)
        assert received == [42]
        bus.off("test", handler)
        bus.emit("test", 99)
        assert received == [42]  # handler unregistered

    def test_idempotent_on(self):
        from yane.evolution.events import EventBus
        bus = EventBus()
        calls = []
        fn = lambda p: calls.append(p)
        bus.on("x", fn)
        bus.on("x", fn)  # registering twice should not double-fire
        bus.emit("x", 1)
        assert len(calls) == 1

    def test_exception_in_handler_does_not_propagate(self):
        from yane.evolution.events import EventBus
        bus = EventBus()
        bus.on("x", lambda p: 1 / 0)
        bus.emit("x", None)  # must not raise

    def test_clear_event(self):
        from yane.evolution.events import EventBus
        bus = EventBus()
        hits = []
        bus.on("a", lambda p: hits.append(p))
        bus.on("b", lambda p: hits.append(p))
        bus.clear("a")
        bus.emit("a", 1)
        bus.emit("b", 2)
        assert hits == [2]

    def test_clear_all(self):
        from yane.evolution.events import EventBus
        bus = EventBus()
        hits = []
        bus.on("a", lambda p: hits.append(p))
        bus.clear()
        bus.emit("a", 1)
        assert hits == []

    def test_handler_count(self):
        from yane.evolution.events import EventBus
        bus = EventBus()
        bus.on("x", lambda p: None)
        bus.on("x", lambda p: None)
        assert bus.handler_count("x") == 2
        assert bus.handler_count("y") == 0

    def test_ne_on_off_emit(self):
        """NeuroEvolution.on/off/emit delegate to event bus."""
        from yane.neuro_evolution import NeuroEvolution
        ne = NeuroEvolution()
        received = []
        def handler(p): received.append(p)
        ne.on("my_event", handler)
        ne.emit("my_event", "hello")
        assert received == ["hello"]
        ne.off("my_event", handler)
        ne.emit("my_event", "ignored")
        assert len(received) == 1


# ---------------------------------------------------------------------------
# Anomaly Detection
# ---------------------------------------------------------------------------

class TestAnomalyDetection:
    def _diag(self, max_fitness=1.0, fitness_iqr=0.5, species_count=3,
               stagnation_count=0, stagnation_threshold=20, pop_evaluated=20):
        return {
            "max_fitness": max_fitness,
            "fitness_iqr": fitness_iqr,
            "species_count": species_count,
            "stagnation_count": stagnation_count,
            "stagnation_threshold": stagnation_threshold,
            "pop_evaluated": pop_evaluated,
        }

    def test_fitness_collapse_fires(self):
        from yane.evolution.anomaly_detection import FitnessCollapseDetector
        det = FitnessCollapseDetector(drop_frac=0.1, window=3)
        d = self._diag
        det.check(d(max_fitness=1.0), 1)
        det.check(d(max_fitness=1.0), 2)
        r = det.check(d(max_fitness=0.5), 3)
        assert r is not None
        assert r.kind == "fitness_collapse"

    def test_fitness_collapse_no_fire_when_ok(self):
        from yane.evolution.anomaly_detection import FitnessCollapseDetector
        det = FitnessCollapseDetector(drop_frac=0.5, window=3)
        d = self._diag
        for i in range(3):
            r = det.check(d(max_fitness=1.0), i)
        assert r is None

    def test_diversity_collapse_fires(self):
        from yane.evolution.anomaly_detection import DiversityCollapseDetector
        det = DiversityCollapseDetector(min_iqr=0.01)
        r = det.check(self._diag(fitness_iqr=1e-5), 1)
        assert r is not None
        assert r.kind == "diversity_collapse"

    def test_diversity_collapse_no_fire_when_small_pop(self):
        from yane.evolution.anomaly_detection import DiversityCollapseDetector
        det = DiversityCollapseDetector(min_iqr=0.01, min_pop=50)
        r = det.check(self._diag(fitness_iqr=0.0, pop_evaluated=5), 1)
        assert r is None

    def test_homogenization_fires(self):
        from yane.evolution.anomaly_detection import HomogenizationDetector
        det = HomogenizationDetector(window=3)
        for i in range(3):
            r = det.check(self._diag(species_count=1), i)
        assert r is not None
        assert r.kind == "homogenization"

    def test_homogenization_resets_on_recovery(self):
        from yane.evolution.anomaly_detection import HomogenizationDetector
        det = HomogenizationDetector(window=3)
        det.check(self._diag(species_count=1), 1)
        det.check(self._diag(species_count=5), 2)  # recovery
        r = det.check(self._diag(species_count=1), 3)
        assert r is None  # streak reset

    def test_stuck_speciation_fires(self):
        from yane.evolution.anomaly_detection import StuckSpeciationDetector
        det = StuckSpeciationDetector(min_stagnation_frac=0.5)
        r = det.check(self._diag(species_count=1, stagnation_count=15, stagnation_threshold=20), 1)
        assert r is not None
        assert r.kind == "stuck_speciation"

    def test_anomaly_detector_set(self):
        from yane.evolution.anomaly_detection import AnomalyDetectorSet, FitnessCollapseDetector
        ds = AnomalyDetectorSet([FitnessCollapseDetector(drop_frac=0.1, window=3)])
        d = self._diag
        for i in range(2):
            ds.check_all(d(max_fitness=1.0), i)
        reports = ds.check_all(d(max_fitness=0.0), 3)
        assert len(reports) == 1
        assert ds.n_detected == 1
        assert ds.last_anomaly is not None
        diag = ds.get_diagnostics()
        assert diag["anomalies_detected"] == 1

    def test_ne_set_anomaly_detectors(self):
        from yane.neuro_evolution import NeuroEvolution
        ne = NeuroEvolution()
        ne.set_anomaly_detectors()  # default detectors
        assert ne._anomaly_detectors is not None
        ne.set_anomaly_detectors([])  # empty list
        assert ne._anomaly_detectors is not None


# ---------------------------------------------------------------------------
# Fitness Transform
# ---------------------------------------------------------------------------

class TestFitnessTransform:
    def test_rank_transform(self):
        from yane.evolution.fitness_transform import RankTransform
        t = RankTransform()
        result = t([10.0, 0.0, 5.0])
        # expected ranks: 0.0 → 1/3, 5.0 → 2/3, 10.0 → 3/3
        assert result[1] < result[2] < result[0]
        assert abs(result[0] - 1.0) < 1e-9

    def test_rank_transform_empty(self):
        from yane.evolution.fitness_transform import RankTransform
        assert RankTransform()([]) == []

    def test_sigma_scaling(self):
        from yane.evolution.fitness_transform import SigmaScaling
        t = SigmaScaling()
        result = t([1.0, 2.0, 3.0])
        assert len(result) == 3
        assert all(v >= 0.0 for v in result)

    def test_sigma_scaling_uniform(self):
        from yane.evolution.fitness_transform import SigmaScaling
        result = SigmaScaling()([5.0, 5.0, 5.0])
        assert all(v == 1.0 for v in result)

    def test_linear_normalize(self):
        from yane.evolution.fitness_transform import LinearNormalize
        t = LinearNormalize(0.0, 1.0)
        result = t([0.0, 5.0, 10.0])
        assert abs(result[0] - 0.0) < 1e-9
        assert abs(result[2] - 1.0) < 1e-9

    def test_clip_transform(self):
        from yane.evolution.fitness_transform import ClipTransform
        t = ClipTransform(-1.0, 1.0)
        result = t([-5.0, 0.5, 5.0])
        assert result == [-1.0, 0.5, 1.0]

    def test_chain_transform(self):
        from yane.evolution.fitness_transform import ChainTransform, RankTransform, LinearNormalize
        t = ChainTransform([RankTransform(), LinearNormalize(0.0, 10.0)])
        result = t([3.0, 1.0, 2.0])
        assert len(result) == 3
        assert min(result) >= 0.0
        assert max(result) <= 10.0

    def test_ne_set_fitness_transform(self):
        from yane.neuro_evolution import NeuroEvolution
        from yane.evolution.fitness_transform import RankTransform
        ne = NeuroEvolution()
        ne.set_fitness_transform(RankTransform())
        assert ne._fitness_transform is not None
        ne.set_fitness_transform(None)
        assert ne._fitness_transform is None


# ---------------------------------------------------------------------------
# Genome Export
# ---------------------------------------------------------------------------

class TestGenomeExport:
    def _make_xor_genome(self):
        """Build a simple hand-crafted genome that approximates XOR."""
        from yane.neuro_evolution import NeuroEvolution
        ne = NeuroEvolution(seed=42)
        ne.configure(n_inputs=2, n_outputs=1)
        ne.set_population_size(10)
        # Just return the initial genome for structural tests
        return ne._population._evaluated[0] if ne._population._evaluated else ne._population._unevaluated[0]

    def test_to_python_returns_string(self):
        from yane.evolution.genome_export import genome_to_python
        genome = self._make_xor_genome()
        src = genome_to_python(genome)
        assert isinstance(src, str)
        assert "def forward(inputs):" in src
        assert "import math" in src

    def test_to_python_executes(self):
        from yane.evolution.genome_export import genome_to_python
        genome = self._make_xor_genome()
        src = genome_to_python(genome)
        ns = {}
        exec(compile(src, "<genome>", "exec"), ns)
        result = ns["forward"]([0.0, 0.0])
        assert isinstance(result, list)
        assert len(result) == 1

    def test_to_python_matches_forward(self):
        """Exported function must produce same output as genome.forward()."""
        from yane.evolution.genome_export import genome_to_python
        genome = self._make_xor_genome()
        genome.reset()
        inputs = [0.3, 0.7]
        expected = list(genome.forward(inputs))
        src = genome_to_python(genome)
        ns = {}
        exec(compile(src, "<genome>", "exec"), ns)
        got = ns["forward"](inputs)
        for e, g in zip(expected, got):
            assert abs(e - g) < 1e-6, f"Mismatch: {e} vs {g}"

    def test_to_numpy_weights(self):
        from yane.evolution.genome_export import genome_to_numpy_weights
        genome = self._make_xor_genome()
        result = genome_to_numpy_weights(genome)
        assert "W" in result
        assert "b" in result
        assert "labels" in result
        assert result["n_inputs"] == 2
        assert result["n_outputs"] == 1
        n = len(genome.nodes)
        assert len(result["W"]) == n
        assert len(result["b"]) == n

    def test_ne_export_genome_python(self):
        from yane.neuro_evolution import NeuroEvolution
        ne = NeuroEvolution(seed=7)
        ne.configure(n_inputs=2, n_outputs=1)
        # Manually evaluate one genome
        pop = ne._population
        g = pop._unevaluated[0]
        pop.submit(g, 0.5, None)
        src = ne.export_genome_python()
        assert "def forward" in src

    def test_ne_export_genome_weights(self):
        from yane.neuro_evolution import NeuroEvolution
        ne = NeuroEvolution(seed=7)
        ne.configure(n_inputs=3, n_outputs=2)
        pop = ne._population
        g = pop._unevaluated[0]
        pop.submit(g, 0.5, None)
        w = ne.export_genome_weights()
        assert w["n_inputs"] == 3
        assert w["n_outputs"] == 2

    def test_export_to_file(self):
        from yane.evolution.genome_export import genome_to_python
        genome = self._make_xor_genome()
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name
        try:
            src = genome_to_python(genome)
            with open(path, "w") as f:
                f.write(src)
            assert os.path.exists(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Validation Set
# ---------------------------------------------------------------------------

class TestValidationSet:
    def test_set_validation_fn(self):
        from yane.neuro_evolution import NeuroEvolution
        ne = NeuroEvolution()
        ne.set_validation_fn(lambda g: 1.0)
        assert ne._validation_fn is not None
        ne.set_validation_fn(None)
        assert ne._validation_fn is None
        assert ne._last_validation_fitness is None

    def test_validation_fn_called_during_train(self):
        """Validation function must be called at generation boundary."""
        from yane.neuro_evolution import NeuroEvolution
        ne = NeuroEvolution(seed=42)
        ne.configure(n_inputs=2, n_outputs=1)
        ne.set_population_size(5)
        ne.set_max_iterations(10)
        val_calls = []

        def val_fn(genome):
            val_calls.append(genome)
            return 0.5

        ne.set_validation_fn(val_fn)
        ne.train(lambda g: 0.5)
        # Validation should have been called at least once (at 5-genome generation boundary)
        assert len(val_calls) >= 1

    def test_validation_fitness_in_diagnostics_after_generation(self):
        from yane.neuro_evolution import NeuroEvolution
        ne = NeuroEvolution(seed=42)
        ne.configure(n_inputs=2, n_outputs=1)
        ne.set_population_size(5)
        ne.set_max_iterations(10)
        ne.set_validation_fn(lambda g: 0.99)
        ne.train(lambda g: 0.5)
        assert ne._last_validation_fitness == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# Config Persistence
# ---------------------------------------------------------------------------

class TestConfigPersistence:
    def test_save_config(self):
        from yane.neuro_evolution import NeuroEvolution
        ne = NeuroEvolution(seed=1)
        ne.configure(n_inputs=2, n_outputs=1)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            ne.save_config(path)
            cfg = json.loads(open(path).read())
            assert cfg["n_inputs"] == 2
            assert cfg["n_outputs"] == 1
            assert cfg["seed"] == 1
        finally:
            os.unlink(path)

    def test_load_config_applies_settings(self):
        from yane.neuro_evolution import NeuroEvolution
        cfg = {
            "seed": 42,
            "population_size": 50,
            "max_iterations": 200,
            "n_inputs": 3,
            "n_outputs": 1,
            "min_fitness": 0.9,
            "n_workers": 1,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(cfg, f)
            path = f.name
        try:
            ne = NeuroEvolution()
            ne.load_config(path)
            assert ne._population_size == 50
            assert ne.max_iterations == 200
            assert ne.min_fitness == pytest.approx(0.9)
        finally:
            os.unlink(path)

    def test_save_load_roundtrip(self):
        from yane.neuro_evolution import NeuroEvolution
        ne1 = NeuroEvolution(seed=7)
        ne1.configure(n_inputs=3, n_outputs=2)
        ne1.set_population_size(80)
        ne1.set_max_iterations(500)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            ne1.save_config(path)
            ne2 = NeuroEvolution()
            ne2.load_config(path)
            assert ne2._population_size == 80
            assert ne2.max_iterations == 500
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Events fired during training
# ---------------------------------------------------------------------------

class TestTrainingEvents:
    def _run_short(self, **kwargs):
        from yane.neuro_evolution import NeuroEvolution
        ne = NeuroEvolution(seed=42)
        ne.configure(n_inputs=2, n_outputs=1)
        ne.set_population_size(5)
        ne.set_max_iterations(12)
        for k, v in kwargs.items():
            getattr(ne, k)(v)
        return ne

    def test_run_end_event_fired(self):
        ne = self._run_short()
        received = []
        ne.on("run_end", lambda p: received.append(p))
        ne.train(lambda g: 0.5)
        assert len(received) == 1
        assert "stop_reason" in received[0]
        assert "iterations" in received[0]

    def test_new_best_event_fired(self):
        ne = self._run_short()
        received = []
        ne.on("new_best", lambda p: received.append(p))
        ne.train(lambda g: 0.5)
        assert len(received) >= 1
        assert "fitness" in received[0]
        assert "genome" in received[0]

    def test_generation_end_event_fired(self):
        ne = self._run_short()
        received = []
        ne.on("generation_end", lambda p: received.append(p))
        ne.train(lambda g: 0.5)
        # generation_end fires at heartbeat (every 100 iters); 12 iters → 0 heartbeats
        # No assertion on count, just check no crash

    def test_anomaly_event_via_detector(self):
        """AnomalyDetectorSet fires 'anomaly' event through the bus."""
        from yane.evolution.anomaly_detection import HomogenizationDetector
        ne = self._run_short()
        ne.set_anomaly_detectors([HomogenizationDetector(window=1)])
        anomalies = []
        ne.on("anomaly", lambda p: anomalies.append(p))
        ne.train(lambda g: 0.5)
        # With window=1 and 1 species, anomaly will fire on first heartbeat if any
        # No crash is the main assertion; count may be 0 if no heartbeat hit

    def test_fitness_transform_applied_during_train(self):
        """RankTransform changes genome.fitness but not genome.raw_fitness."""
        from yane.neuro_evolution import NeuroEvolution
        from yane.evolution.fitness_transform import RankTransform
        ne = NeuroEvolution(seed=42)
        ne.configure(n_inputs=2, n_outputs=1)
        ne.set_population_size(5)
        ne.set_max_iterations(10)
        ne.set_fitness_transform(RankTransform())
        ne.train(lambda g: 0.5)
        # After training, fitness values should be rank-transformed (all in (0,1])
        for g in ne._population._evaluated:
            assert 0.0 < g.fitness <= 1.0 + 1e-9
