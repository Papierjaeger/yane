"""Tests for Curriculum Learning (CurriculumStage, Curriculum, NeuroEvolution.set_curriculum)."""
from __future__ import annotations
import unittest

from yane import NeuroEvolution
from yane.evolution.curriculum import Curriculum, CurriculumStage
from yane.examples.sequence_recall_PI import (
    DECIMAL_PLACES as PI_DECIMAL_PLACES,
    TARGET_FITNESS as PI_TARGET_FITNESS,
    dataset as PI_DATASET,
    make_eval as make_pi_eval,
    make_curriculum_eval as make_pi_curriculum_eval,
    make_curriculum_targets as make_pi_curriculum_targets,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ne(pop_size: int = 10) -> NeuroEvolution:
    ne = NeuroEvolution(seed=0)
    ne.configure(1, 1)
    ne.set_population_size(pop_size)
    return ne


def _always(value: float):
    """Return a fitness function that always yields *value*."""
    def fn(genome):
        return value
    fn.__name__ = f"always_{value}"
    return fn


# ---------------------------------------------------------------------------
# Unit tests for CurriculumStage
# ---------------------------------------------------------------------------

class TestCurriculumStage(unittest.TestCase):

    def test_basic_construction(self):
        fn = _always(1.0)
        s = CurriculumStage(fn, target_fitness=0.9, name="easy")
        self.assertIs(s.fitness_fn, fn)
        self.assertEqual(s.target_fitness, 0.9)
        self.assertEqual(s.name, "easy")

    def test_name_defaults_to_empty_string(self):
        s = CurriculumStage(_always(0.0))
        self.assertEqual(s.name, "")

    def test_non_callable_raises(self):
        with self.assertRaises(TypeError):
            CurriculumStage(42)  # type: ignore


# ---------------------------------------------------------------------------
# Unit tests for Curriculum
# ---------------------------------------------------------------------------

class TestCurriculum(unittest.TestCase):

    def _make(self, *targets):
        stages = [CurriculumStage(_always(float(i)), target_fitness=t)
                  for i, t in enumerate(targets)]
        return Curriculum(stages)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            Curriculum([])

    def test_initial_state(self):
        c = self._make(0.5, 0.8, None)
        self.assertEqual(c.stage_index, 0)
        self.assertEqual(c.stage_count, 3)
        self.assertFalse(c.is_last_stage)

    def test_callable_routes_to_current_stage(self):
        c = self._make(0.5, None)

        class FakeGenome:
            pass

        # Stage 0 always returns 0.0 (index 0, _always(0.0))
        result = c(FakeGenome())
        self.assertAlmostEqual(result, 0.0)

    def test_no_advance_below_target(self):
        c = self._make(0.9, None)
        c.record_fitness(0.5)
        self.assertFalse(c.maybe_advance())
        self.assertEqual(c.stage_index, 0)

    def test_advance_at_target(self):
        c = self._make(0.9, None)
        c.record_fitness(0.95)
        advanced = c.maybe_advance()
        self.assertTrue(advanced)
        self.assertEqual(c.stage_index, 1)
        self.assertEqual(c.n_advances, 1)

    def test_advance_resets_stage_best(self):
        c = self._make(0.5, 0.8, None)
        c.record_fitness(0.9)
        c.maybe_advance()   # → stage 1
        # stage_best must be reset so stage-1 advancement only triggers on fresh data
        c2 = Curriculum(c.stages)
        c2._current = 1
        c2._stage_best = c._stage_best
        self.assertEqual(c._stage_best, float('-inf'))

    def test_last_stage_no_advance(self):
        c = self._make(None)
        c.record_fitness(999.0)
        self.assertFalse(c.maybe_advance())
        self.assertEqual(c.stage_index, 0)

    def test_is_complete_false_on_non_last(self):
        c = self._make(0.5, None)
        c.record_fitness(1.0)
        c.maybe_advance()   # → stage 1 (last)
        # last stage has target_fitness=None → not complete
        self.assertFalse(c.is_complete())

    def test_is_complete_true_when_target_met(self):
        c = self._make(0.5, 0.8)
        c.record_fitness(1.0)
        c.maybe_advance()   # → stage 1
        c.record_fitness(1.0)
        self.assertTrue(c.is_complete())

    def test_info_keys(self):
        c = self._make(0.5, None)
        info = c.info()
        for key in (
            "curriculum_stage_index", "curriculum_stage_count",
            "curriculum_stage_name", "curriculum_stage_target",
            "curriculum_stage_best", "curriculum_stage_evals",
            "curriculum_n_advances",
        ):
            self.assertIn(key, info)


class TestPiCurriculum(unittest.TestCase):

    class _ReplayGenome:
        def __init__(self, outputs):
            self.outputs = list(outputs)
            self.i = 0

        def reset(self):
            self.i = 0

        def forward(self, _inputs):
            value = self.outputs[self.i]
            self.i += 1
            return [value]

    def test_pi_default_target_is_perfect_fitness(self):
        self.assertEqual(PI_TARGET_FITNESS, 0.0)

    def test_pi_curriculum_targets_scale_from_gui_target(self):
        targets = make_pi_curriculum_targets(-0.5, PI_DECIMAL_PLACES)
        self.assertEqual(len(targets), PI_DECIMAL_PLACES)
        self.assertAlmostEqual(targets[0], -0.05)
        self.assertAlmostEqual(targets[-1], -0.5)

    def test_pi_curriculum_protects_previous_outputs(self):
        expected_1 = PI_DATASET[0]["output"][0]
        expected_2 = PI_DATASET[1]["output"][0]
        eval_fn = make_pi_curriculum_eval(
            2,
            protected_digits=1,
            protect_tolerance=0.0,
        )

        prefix_regressed = self._ReplayGenome([expected_1 + 0.1, expected_2])
        new_digit_wrong = self._ReplayGenome([expected_1, expected_2 + 0.1])

        self.assertLess(eval_fn(prefix_regressed), eval_fn(new_digit_wrong))

    def test_pi_fitness_is_zero_for_correct_rounded_digits(self):
        outputs = [sample["output"][0] + 0.01 for sample in PI_DATASET[:PI_DECIMAL_PLACES]]
        genome = self._ReplayGenome(outputs)
        self.assertEqual(make_pi_eval()(genome), 0.0)


# ---------------------------------------------------------------------------
# Integration tests via NeuroEvolution
# ---------------------------------------------------------------------------

class TestNeuroEvolutionCurriculum(unittest.TestCase):

    def test_train_raises_without_fn_and_curriculum(self):
        ne = _ne()
        with self.assertRaises((ValueError, TypeError)):
            ne.train()  # no fitness_fn, no curriculum

    def test_set_curriculum_bad_type_raises(self):
        ne = _ne()
        with self.assertRaises(TypeError):
            ne.set_curriculum(["not_a_stage"])

    def test_set_curriculum_empty_raises(self):
        ne = _ne()
        with self.assertRaises(ValueError):
            ne.set_curriculum([])

    def test_curriculum_stage_property_none_without_curriculum(self):
        ne = _ne()
        self.assertIsNone(ne.curriculum_stage)

    def test_curriculum_stage_property_after_set(self):
        ne = _ne()
        ne.set_curriculum([CurriculumStage(_always(0.0), name="only")])
        self.assertEqual(ne.curriculum_stage, 0)

    def test_single_stage_trains_and_stops_on_target(self):
        ne = _ne(pop_size=20)
        ne.min_fitness = 0.5

        calls = []
        def fn(genome):
            calls.append(1)
            return 1.0   # always satisfies target

        ne.set_curriculum([CurriculumStage(fn, name="trivial")])
        iters = ne.train()
        self.assertGreater(iters, 0)

    def test_two_stage_curriculum_advances(self):
        """Population should re-evaluate and advance through both stages."""
        ne = _ne(pop_size=10)
        ne.set_max_iterations(200)

        advanced = []

        def on_advance(idx, stage):
            advanced.append(idx)

        ne.set_curriculum(
            [
                CurriculumStage(_always(1.0), target_fitness=0.9, name="easy"),
                CurriculumStage(_always(1.0), name="hard"),
            ],
            on_stage_advance=on_advance,
        )
        ne.train()
        # The callback must have fired exactly once (0→1)
        self.assertEqual(advanced, [1])

    def test_curriculum_complete_stop_reason(self):
        """curriculum_complete fires when last stage target is met."""
        ne = _ne(pop_size=10)
        ne.set_max_iterations(500)

        stop_reasons = []

        ne.set_curriculum([
            CurriculumStage(_always(1.0), target_fitness=0.5, name="stage1"),
            CurriculumStage(_always(1.0), target_fitness=0.5, name="stage2"),
        ])
        ne.train(on_stop=lambda r: stop_reasons.append(r))
        self.assertIn("curriculum_complete", stop_reasons)

    def test_population_memory_info_has_curriculum_keys(self):
        ne = _ne(pop_size=10)
        ne.set_curriculum([CurriculumStage(_always(0.0), name="s1")])
        ne.next_genome()
        ne.submit_fitness(0.0)
        info = ne.population_memory_info()
        self.assertIn("curriculum_stage_index", info)
        self.assertIn("curriculum_stage_name", info)
        self.assertEqual(info["curriculum_stage_name"], "s1")

    def test_on_stage_advance_callback_receives_correct_args(self):
        ne = _ne(pop_size=5)
        ne.set_max_iterations(100)

        received = []

        def on_advance(idx, stage):
            received.append((idx, stage.name))

        ne.set_curriculum(
            [
                CurriculumStage(_always(1.0), target_fitness=0.9, name="A"),
                CurriculumStage(_always(0.0), name="B"),
            ],
            on_stage_advance=on_advance,
        )
        ne.train()
        self.assertTrue(len(received) >= 1)
        # First advance must be to index 1, stage name "B"
        self.assertEqual(received[0], (1, "B"))

    def test_population_kept_across_stage_switch(self):
        """After stage advance the population is re-evaluated and get_best() works."""
        ne = _ne(pop_size=10)
        ne.set_max_iterations(60)

        advanced_at = []

        def on_advance(idx, stage):
            # Record which iteration we advanced (get_best() not safe here —
            # reset_stagnation() just moved all genomes to unevaluated)
            advanced_at.append(idx)

        ne.set_curriculum(
            [
                CurriculumStage(_always(1.0), target_fitness=0.9, name="easy"),
                CurriculumStage(_always(0.0), name="hard"),
            ],
            on_stage_advance=on_advance,
        )
        ne.train()
        # Stage switch must have happened
        self.assertTrue(len(advanced_at) >= 1)
        self.assertEqual(advanced_at[0], 1)  # advanced to index 1
        # After training, get_best() must return a valid Genome
        from yane.core.genome import Genome
        self.assertIsInstance(ne.get_best(), Genome)


if __name__ == "__main__":
    unittest.main()
