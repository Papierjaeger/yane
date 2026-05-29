"""Tests for the Cross-Run Knowledge Base (P0 Meta-Adaptive Orchestration, Phase 4).

Covers:
- profile_to_vector(): dimensionality, clamping, task-type one-hot
- _weighted_distance(): correct weighted Euclidean
- _confidence(): increases with count and decreases with distance
- cold_start_suggestion(): correct defaults per task type
- KBEntry: construction and round-trip through JSON
- KnowledgeBase.learn(): entries grow, persistence
- KnowledgeBase.suggest(): k-NN ranking, cold-start fallback, confidence ordering
- KnowledgeBase.best_suggestion(): returns single best
- KnowledgeBase.export_json() / import_json(): merge and replace modes
- Confidence increases with number of similar runs
- NeuroEvolution.set_knowledge_base() / knowledge_base property
- NeuroEvolution.suggest_params()
- NeuroEvolution._auto_kb_learn() fires after train()
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from yane import NeuroEvolution
from yane.evolution.knowledge_base import (
    KBEntry,
    KnowledgeBase,
    _confidence,
    _weighted_distance,
    cold_start_suggestion,
    profile_to_vector,
    TASK_TYPES,
    COLD_START_DEFAULTS,
    DEFAULT_FEATURE_WEIGHTS,
    _N_FEATURES,
)
from yane.evolution.problem_profiler import ProblemProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(
    task_type: str = "classification",
    n_inputs: int = 4,
    n_outputs: int = 1,
    difficulty: float = 0.5,
    noise: float = 0.0,
    temporal: float = 0.0,
) -> ProblemProfile:
    return ProblemProfile(
        task_type=task_type,
        task_type_confidence=0.9,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        estimated_difficulty=difficulty,
        noise_level=noise,
        reward_sparsity=0.1,
        temporal_dependency=temporal,
        state_dim_effective=n_inputs,
        fitness_mean=5.0,
        fitness_std=1.0,
        fitness_min=0.0,
        fitness_max=10.0,
        alternative_types=[],
    )


# ---------------------------------------------------------------------------
# profile_to_vector
# ---------------------------------------------------------------------------

class TestProfileToVector:
    def test_length(self):
        profile = _make_profile()
        vec = profile_to_vector(profile)
        assert len(vec) == _N_FEATURES

    def test_task_type_one_hot_classification(self):
        vec = profile_to_vector(_make_profile(task_type="classification"))
        idx = TASK_TYPES.index("classification")
        assert vec[idx] == 1.0
        assert all(vec[i] == 0.0 for i in range(4) if i != idx)

    def test_task_type_one_hot_rl_continuous(self):
        vec = profile_to_vector(_make_profile(task_type="rl_continuous"))
        idx = TASK_TYPES.index("rl_continuous")
        assert vec[idx] == 1.0

    def test_numeric_components_in_range(self):
        vec = profile_to_vector(_make_profile(noise=1.5, temporal=2.0))
        for v in vec:
            assert math.isfinite(v), f"Non-finite value in vector: {v}"
            assert 0.0 <= v <= 2.0, f"Value {v} unexpectedly large"

    def test_clamping_noise_and_temporal(self):
        vec = profile_to_vector(_make_profile(noise=999.0, temporal=999.0))
        assert vec[7] == pytest.approx(1.0)  # noise_level clamped
        assert vec[8] == pytest.approx(1.0)  # temporal_dependency clamped

    def test_same_profile_same_vector(self):
        p = _make_profile()
        assert profile_to_vector(p) == profile_to_vector(p)

    def test_different_task_types_different_vectors(self):
        v1 = profile_to_vector(_make_profile(task_type="classification"))
        v2 = profile_to_vector(_make_profile(task_type="rl_continuous"))
        assert v1 != v2


# ---------------------------------------------------------------------------
# _weighted_distance
# ---------------------------------------------------------------------------

class TestWeightedDistance:
    def test_identical_vectors_distance_zero(self):
        v = [0.5, 0.3, 1.0]
        assert _weighted_distance(v, v, [1.0, 1.0, 1.0]) == pytest.approx(0.0)

    def test_unit_difference(self):
        v1 = [0.0]
        v2 = [1.0]
        w = [4.0]
        # sqrt(4 * 1) = 2
        assert _weighted_distance(v1, v2, w) == pytest.approx(2.0)

    def test_weights_scale_distance(self):
        v1, v2 = [0.0, 0.0], [1.0, 1.0]
        d1 = _weighted_distance(v1, v2, [1.0, 1.0])
        d2 = _weighted_distance(v1, v2, [4.0, 4.0])
        assert d2 > d1

    def test_different_lengths_handled_gracefully(self):
        v1, v2, w = [0.0, 1.0, 2.0], [0.0, 1.0], [1.0, 1.0, 1.0]
        d = _weighted_distance(v1, v2, w)
        assert math.isfinite(d)


# ---------------------------------------------------------------------------
# _confidence
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_zero_distance_gives_high_confidence(self):
        conf = _confidence(0.0, [0.0, 0.5, 1.0])
        assert conf > 0.5

    def test_max_distance_gives_low_confidence(self):
        conf = _confidence(1.0, [0.0, 0.5, 1.0])
        assert conf < 0.5

    def test_confidence_increases_with_denser_neighborhood(self):
        # Sparse neighbourhood (entries spread out; query far from cluster)
        c1 = _confidence(0.5, [0.1, 0.5, 0.9])
        # Dense neighbourhood (entries tightly clustered; query close)
        c2 = _confidence(0.1, [0.05, 0.1, 0.15])
        assert c2 > c1

    def test_confidence_in_range(self):
        for _ in [1, 3, 10]:
            c = _confidence(0.2, [0.1, 0.2, 0.5])
            assert 0.0 <= c <= 1.0

    def test_all_same_distance_gives_uniform_confidence(self):
        # When all top-K distances are equal, dist_score=0; agreement=1.0 → confidence=0.3
        c1 = _confidence(0.5, [0.5, 0.5, 0.5])
        c2 = _confidence(0.5, [0.5, 0.5, 0.5])
        assert c1 == pytest.approx(c2)   # all entries have same confidence
        assert 0.0 <= c1 <= 1.0


# ---------------------------------------------------------------------------
# cold_start_suggestion
# ---------------------------------------------------------------------------

class TestColdStartSuggestion:
    def test_returns_dict_with_required_keys(self):
        s = cold_start_suggestion("classification")
        assert set(s.keys()) >= {"params", "expected_fitness", "confidence",
                                  "distance", "source", "run_id"}

    def test_source_is_cold_start(self):
        assert cold_start_suggestion("rl_continuous")["source"] == "cold_start"

    def test_confidence_is_zero(self):
        assert cold_start_suggestion("regression")["confidence"] == 0.0

    def test_distance_is_inf(self):
        assert cold_start_suggestion("rl_discrete")["distance"] == float("inf")

    def test_classification_defaults(self):
        params = cold_start_suggestion("classification")["params"]
        assert "lamarck.mode" in params

    def test_rl_continuous_defaults(self):
        params = cold_start_suggestion("rl_continuous")["params"]
        assert params.get("lamarck.mode") == "cma_es"

    def test_unknown_task_type_falls_back(self):
        s = cold_start_suggestion("unknown_type")
        assert s["source"] == "cold_start"
        assert isinstance(s["params"], dict)

    def test_all_task_types_have_defaults(self):
        for t in TASK_TYPES:
            s = cold_start_suggestion(t)
            assert isinstance(s["params"], dict)
            assert len(s["params"]) > 0


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------

class TestKnowledgeBase:
    def test_empty_kb_has_length_zero(self):
        kb = KnowledgeBase()
        assert len(kb) == 0

    def test_learn_adds_entry(self):
        kb = KnowledgeBase()
        kb.learn(_make_profile(), {"lamarck.mode": "cma_es"}, 195.0)
        assert len(kb) == 1

    def test_learn_multiple_entries(self):
        kb = KnowledgeBase()
        for i in range(5):
            kb.learn(_make_profile(difficulty=i * 0.2), {"pop.size": 100 + i}, float(i))
        assert len(kb) == 5

    def test_suggest_empty_returns_cold_start(self):
        kb = KnowledgeBase()
        result = kb.suggest(_make_profile(task_type="classification"))
        assert len(result) == 1
        assert result[0]["source"] == "cold_start"

    def test_suggest_returns_list_of_dicts(self):
        kb = KnowledgeBase()
        for i in range(3):
            kb.learn(_make_profile(), {"lamarck.mode": "cma_es"}, 100.0 + i)
        result = kb.suggest(_make_profile())
        assert isinstance(result, list)
        for item in result:
            assert "params" in item
            assert "expected_fitness" in item
            assert "confidence" in item

    def test_suggest_closer_entry_ranked_first(self):
        kb = KnowledgeBase()
        # Near profile: classification, difficulty=0.5
        kb.learn(_make_profile("classification", difficulty=0.5), {"pop.size": 50}, 90.0)
        # Far profile: rl_continuous, difficulty=0.9
        kb.learn(_make_profile("rl_continuous", difficulty=0.9), {"pop.size": 500}, 50.0)
        result = kb.suggest(_make_profile("classification", difficulty=0.5), top_k=5)
        # First non-cold-start result should be the closer one
        kb_results = [r for r in result if r["source"] == "kb"]
        assert kb_results[0]["params"].get("pop.size") == 50

    def test_suggest_confidence_increases_with_more_entries(self):
        kb = KnowledgeBase()
        profile = _make_profile()
        # 1 entry
        kb.learn(profile, {"pop.size": 50}, 90.0)
        conf1 = next(r["confidence"] for r in kb.suggest(profile) if r["source"] == "kb")
        # 5 entries (all identical profile)
        for _ in range(4):
            kb.learn(profile, {"pop.size": 50}, 90.0)
        conf5 = next(r["confidence"] for r in kb.suggest(profile) if r["source"] == "kb")
        assert conf5 >= conf1

    def test_best_suggestion_returns_single_dict(self):
        kb = KnowledgeBase()
        kb.learn(_make_profile(), {"pop.size": 100}, 80.0)
        best = kb.best_suggestion(_make_profile())
        assert isinstance(best, dict)
        assert "params" in best

    def test_best_suggestion_empty_kb_cold_start(self):
        kb = KnowledgeBase()
        best = kb.best_suggestion(_make_profile(task_type="rl_discrete"))
        assert best["source"] == "cold_start"

    def test_top_k_limits_kb_results(self):
        kb = KnowledgeBase()
        for i in range(10):
            kb.learn(_make_profile(difficulty=i * 0.1), {"pop.size": i * 10}, float(i))
        result = kb.suggest(_make_profile(), top_k=3)
        kb_results = [r for r in result if r["source"] == "kb"]
        assert len(kb_results) <= 3

    def test_clear_empties_entries(self):
        kb = KnowledgeBase()
        kb.learn(_make_profile(), {}, 10.0)
        kb.clear()
        assert len(kb) == 0

    def test_get_diagnostics(self):
        kb = KnowledgeBase()
        d = kb.get_diagnostics()
        assert "n_entries" in d
        assert "feature_weights" in d

    # ── Persistence ─────────────────────────────────────────────────────────

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            kb = KnowledgeBase(path=path)
            kb.learn(_make_profile(), {"lamarck.mode": "cma_es"}, 195.0)
            kb.save()

            kb2 = KnowledgeBase(path=path)
            assert len(kb2) == 1
            assert kb2._entries[0].final_fitness == pytest.approx(195.0)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_export_and_import_json_merge(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            kb1 = KnowledgeBase()
            kb1.learn(_make_profile(), {"pop.size": 100}, 80.0)
            kb1.export_json(path)

            kb2 = KnowledgeBase()
            kb2.learn(_make_profile(), {"pop.size": 200}, 90.0)
            kb2.import_json(path, merge=True)
            assert len(kb2) == 2
        finally:
            Path(path).unlink(missing_ok=True)

    def test_export_and_import_json_replace(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            kb1 = KnowledgeBase()
            kb1.learn(_make_profile(), {"pop.size": 100}, 80.0)
            kb1.export_json(path)

            kb2 = KnowledgeBase()
            kb2.learn(_make_profile(), {"pop.size": 200}, 90.0)
            kb2.learn(_make_profile(), {"pop.size": 300}, 95.0)
            kb2.import_json(path, merge=False)
            assert len(kb2) == 1
        finally:
            Path(path).unlink(missing_ok=True)

    def test_auto_save_on_learn(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            kb = KnowledgeBase(path=path)
            kb.learn(_make_profile(), {"pop.size": 50}, 70.0, auto_save=True)
            raw = json.loads(Path(path).read_text())
            assert len(raw["entries"]) == 1
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_from_nonexistent_path_starts_fresh(self):
        kb = KnowledgeBase(path="/tmp/nonexistent_kb_xyz.json")
        assert len(kb) == 0

    def test_corrupt_file_starts_fresh(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("NOT_JSON{{{")
            path = f.name
        try:
            kb = KnowledgeBase(path=path)
            assert len(kb) == 0
        finally:
            Path(path).unlink(missing_ok=True)

    # ── k-NN correctness ────────────────────────────────────────────────────

    def test_knn_prefers_same_task_type(self):
        kb = KnowledgeBase()
        # 3 classification entries
        for _ in range(3):
            kb.learn(
                _make_profile("classification", difficulty=0.3),
                {"lamarck.mode": "hill_climbing"},
                85.0,
            )
        # 3 rl_continuous entries
        for _ in range(3):
            kb.learn(
                _make_profile("rl_continuous", difficulty=0.8),
                {"lamarck.mode": "cma_es"},
                50.0,
            )
        result = kb.suggest(_make_profile("classification", difficulty=0.3))
        kb_results = [r for r in result if r["source"] == "kb"]
        # The top result should have hill_climbing (from classification runs)
        assert kb_results[0]["params"]["lamarck.mode"] == "hill_climbing"

    def test_knn_distance_ordering(self):
        kb = KnowledgeBase()
        kb.learn(_make_profile(difficulty=0.1), {"pop.size": 10}, 10.0)
        kb.learn(_make_profile(difficulty=0.9), {"pop.size": 90}, 90.0)
        result = kb.suggest(_make_profile(difficulty=0.1))
        kb_results = [r for r in result if r["source"] == "kb"]
        assert kb_results[0]["distance"] <= kb_results[-1]["distance"]


# ---------------------------------------------------------------------------
# NeuroEvolution integration
# ---------------------------------------------------------------------------

class TestNeuroEvolutionKBIntegration:
    def test_set_knowledge_base_creates_kb(self):
        ne = NeuroEvolution(seed=0)
        ne.set_knowledge_base()
        assert ne.knowledge_base is not None

    def test_set_knowledge_base_with_path(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            ne = NeuroEvolution(seed=0)
            ne.set_knowledge_base(path)
            assert ne.knowledge_base is not None
        finally:
            Path(path).unlink(missing_ok=True)

    def test_knowledge_base_property_none_by_default(self):
        ne = NeuroEvolution(seed=0)
        assert ne.knowledge_base is None

    def test_suggest_params_raises_without_kb(self):
        ne = NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1)
        with pytest.raises(RuntimeError, match="Knowledge Base"):
            ne.suggest_params()

    def test_suggest_params_raises_without_profile(self):
        ne = NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1)
        ne.set_knowledge_base()
        with pytest.raises(RuntimeError, match="profile"):
            ne.suggest_params()

    def test_suggest_params_after_profile_problem(self):
        ne = NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1)
        ne.set_knowledge_base()

        def dummy_eval(g):
            return 1.0

        ne.profile_problem(dummy_eval, n_warmup=5)
        result = ne.suggest_params()
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_auto_kb_learn_after_train(self):
        """After train(), the KB should have ≥1 entry if profile_problem was called."""
        ne = NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1)
        ne.set_max_iterations(50)
        ne.set_knowledge_base()

        def xor_eval(g):
            cases = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
            return sum(1 for inp, exp in cases if round(g.forward(inp)[0]) == exp)

        ne.profile_problem(xor_eval, n_warmup=5)
        ne.train(xor_eval)
        assert len(ne.knowledge_base) == 1

    def test_auto_kb_learn_not_called_without_profile(self):
        """Without profile_problem(), train() should not learn into KB."""
        ne = NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1)
        ne.set_max_iterations(30)
        ne.set_knowledge_base()

        def dummy_eval(g):
            return 1.0

        ne.train(dummy_eval)
        # _last_problem_profile is None → nothing learned
        assert len(ne.knowledge_base) == 0

    def test_train_without_kb_does_not_crash(self):
        """train() must work normally when no KB is configured."""
        ne = NeuroEvolution(seed=0)
        ne.configure(n_inputs=2, n_outputs=1)
        ne.set_max_iterations(20)

        def dummy_eval(g):
            return 1.0

        ne.train(dummy_eval)  # should not raise


# ---------------------------------------------------------------------------
# RunDatabase schema migration
# ---------------------------------------------------------------------------

class TestRunDatabaseMigration:
    def test_problem_profile_column_exists(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            from yane.util.run_database import RunDatabase
            db = RunDatabase(path)
            cols = {
                row[1]
                for row in db._db.execute("PRAGMA table_info(runs)").fetchall()
            }
            assert "problem_profile_json" in cols
            db.close()
        finally:
            Path(path).unlink(missing_ok=True)
