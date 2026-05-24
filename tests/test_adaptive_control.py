import unittest

from yane import NeuroEvolution
from yane.core.genome import Genome
from yane.evolution.population import Population
from yane.evolution.species import Species
from yane.evolution.adaptive_controller import AdaptiveController, AdaptiveSignals, FeaturePolicy
from yane.evolution.operator_scheduler import OperatorScheduler, ALL_OPS, MIN_WEIGHT, MAX_WEIGHT
from yane.evolution.lamarck_refiner import LamarckRefiner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_population(size: int = 10, n_species: int = 1) -> Population:
    """Create a minimal Population with the given number of species."""
    pop = Population(max_size=size, initial_genome=Genome(), target_species=max(1, n_species))
    genomes = [Genome() for _ in range(n_species)]
    pop._species = []
    for g in genomes:
        sp = Species(g)
        sp.add(g)
        pop._species.append(sp)
    return pop


# ---------------------------------------------------------------------------
# AdaptiveController — signals
# ---------------------------------------------------------------------------

class TestAdaptiveControllerSignals(unittest.TestCase):

    def test_plateau_ratio_zero_on_fresh_population(self):
        ctrl = AdaptiveController()
        pop = _make_population()
        ctrl._compute_signals(pop)
        self.assertEqual(ctrl.signals.plateau_ratio, 0.0)

    def test_plateau_ratio_one_at_full_stagnation(self):
        ctrl = AdaptiveController()
        pop = _make_population()
        pop._stagnation_count = pop.stagnation_threshold
        ctrl._compute_signals(pop)
        self.assertAlmostEqual(ctrl.signals.plateau_ratio, 1.0)

    def test_plateau_ratio_clamped_to_one(self):
        ctrl = AdaptiveController()
        pop = _make_population()
        pop._stagnation_count = pop.stagnation_threshold * 10
        ctrl._compute_signals(pop)
        self.assertLessEqual(ctrl.signals.plateau_ratio, 1.0)

    def test_diversity_score_zero_with_empty_novelty_cache(self):
        ctrl = AdaptiveController()
        pop = _make_population()
        pop._novelty_cache = {}
        ctrl._compute_signals(pop)
        self.assertEqual(ctrl.signals.diversity_score, 0.0)

    def test_diversity_score_from_novelty_cache(self):
        ctrl = AdaptiveController()
        pop = _make_population()
        # Fake novelty values
        g1, g2 = Genome(), Genome()
        pop._novelty_cache = {id(g1): 0.4, id(g2): 0.6}
        ctrl._compute_signals(pop)
        self.assertAlmostEqual(ctrl.signals.diversity_score, 0.5)

    def test_species_stagnation_fraction(self):
        ctrl = AdaptiveController()
        pop = _make_population(n_species=4)
        threshold = pop.stagnation_threshold
        # Mark 2 of 4 species as fully stagnant
        for sp in pop._species[:2]:
            sp.stagnation_count = threshold
        ctrl._compute_signals(pop)
        self.assertAlmostEqual(ctrl.signals.species_stagnation, 0.5)

    def test_n_species_and_target_species(self):
        ctrl = AdaptiveController()
        pop = _make_population(n_species=3)
        ctrl._compute_signals(pop)
        self.assertEqual(ctrl.signals.n_species, 3)


# ---------------------------------------------------------------------------
# AdaptiveController — tick and policy updates
# ---------------------------------------------------------------------------

class TestAdaptiveControllerTick(unittest.TestCase):

    def test_tick_increments_count(self):
        ctrl = AdaptiveController()
        pop = _make_population()
        ctrl.tick(pop)
        self.assertEqual(ctrl._tick_count, 1)
        ctrl.tick(pop)
        self.assertEqual(ctrl._tick_count, 2)

    def test_tick_computes_signals(self):
        ctrl = AdaptiveController()
        pop = _make_population()
        pop._stagnation_count = pop.stagnation_threshold // 2
        ctrl.tick(pop)
        self.assertGreater(ctrl.signals.plateau_ratio, 0.0)

    def test_interspecies_off_mode_skips_update(self):
        ctrl = AdaptiveController()
        pop = _make_population(n_species=2)
        pop._stagnation_count = pop.stagnation_threshold
        # Default mode is "off"
        ctrl.tick(pop)
        self.assertEqual(ctrl.policies["interspecies_crossover"].current_value, 0.0)

    def test_interspecies_adaptive_mode_ramps_at_plateau(self):
        ctrl = AdaptiveController()
        pop = _make_population(n_species=2)
        pop._stagnation_count = pop.stagnation_threshold  # full plateau
        ctrl.configure_interspecies_crossover(mode="adaptive", min_rate=0.01, max_rate=0.2)
        ctrl.tick(pop)
        # With full plateau, pressure=1.0 → current_value should approach max_rate
        self.assertGreater(ctrl.policies["interspecies_crossover"].current_value, 0.1)

    def test_interspecies_adaptive_protection_at_low_success(self):
        ctrl = AdaptiveController()
        pop = _make_population(n_species=2)
        pop._stagnation_count = pop.stagnation_threshold
        ctrl.configure_interspecies_crossover(mode="adaptive", min_rate=0.01, max_rate=0.2)
        # Simulate very low success rate (4 total, 0 improved)
        for _ in range(4):
            ctrl.record_interspecies_offspring(improved=False)
        # Force enough count to trigger protection check
        ctrl._interspecies_offspring_count = 10
        ctrl._interspecies_improvement_count = 0
        ctrl.tick(pop)
        p = ctrl.policies["interspecies_crossover"]
        self.assertEqual(p.last_reason, "adaptive:crossover_low_success")

    def test_get_diagnostics_shape(self):
        ctrl = AdaptiveController()
        pop = _make_population()
        ctrl.tick(pop)
        diag = ctrl.get_diagnostics()
        self.assertIn("controller_tick", diag)
        self.assertIn("signals", diag)
        self.assertIn("policies", diag)
        self.assertIn("lamarck_budget", diag)
        self.assertIn("interspecies_success", diag)
        self.assertIn("recent_decisions", diag)


# ---------------------------------------------------------------------------
# AdaptiveController — Lamarck budget
# ---------------------------------------------------------------------------

class TestAdaptiveControllerLamarckBudget(unittest.TestCase):

    def test_unlimited_budget_always_returns_true(self):
        ctrl = AdaptiveController()
        # Default: _lamarck_gen_budget is None
        for _ in range(1000):
            result = ctrl.consume_lamarck_budget(1)
        self.assertTrue(result)
        self.assertIsNone(ctrl.lamarck_budget_remaining())

    def test_budget_exhausts_at_limit(self):
        ctrl = AdaptiveController()
        ctrl._lamarck_gen_budget = 5
        for _ in range(5):
            self.assertTrue(ctrl.consume_lamarck_budget(1))
        self.assertFalse(ctrl.consume_lamarck_budget(1))

    def test_budget_remaining_decrements(self):
        ctrl = AdaptiveController()
        ctrl._lamarck_gen_budget = 10
        ctrl.consume_lamarck_budget(3)
        self.assertEqual(ctrl.lamarck_budget_remaining(), 7)

    def test_reset_restores_budget(self):
        ctrl = AdaptiveController()
        ctrl._lamarck_gen_budget = 5
        ctrl.consume_lamarck_budget(5)
        self.assertFalse(ctrl.consume_lamarck_budget(1))
        ctrl.reset_lamarck_budget()
        self.assertTrue(ctrl.consume_lamarck_budget(1))


# ---------------------------------------------------------------------------
# AdaptiveController — cross-species success tracking (rolling window)
# ---------------------------------------------------------------------------

class TestAdaptiveControllerInterspeciesSuccess(unittest.TestCase):

    def test_success_rate_zero_at_start(self):
        ctrl = AdaptiveController()
        diag = ctrl.get_diagnostics()
        self.assertEqual(diag["interspecies_success"]["success_rate"], 0.0)

    def test_success_rate_updates(self):
        ctrl = AdaptiveController()
        ctrl.record_interspecies_offspring(improved=True)
        ctrl.record_interspecies_offspring(improved=False)
        diag = ctrl.get_diagnostics()
        self.assertAlmostEqual(diag["interspecies_success"]["success_rate"], 0.5)

    def test_rolling_window_halves_at_50(self):
        ctrl = AdaptiveController()
        for _ in range(50):
            ctrl.record_interspecies_offspring(improved=True)
        # After 50 calls, the 50th call triggers halving
        self.assertLessEqual(ctrl._interspecies_offspring_count, 25)


# ---------------------------------------------------------------------------
# OperatorScheduler — basic weights
# ---------------------------------------------------------------------------

class TestOperatorScheduler(unittest.TestCase):

    def test_default_weights_are_one(self):
        sched = OperatorScheduler()
        for op in ALL_OPS:
            self.assertAlmostEqual(sched.global_weights[op], 1.0)

    def test_record_success_updates_totals(self):
        sched = OperatorScheduler()
        sched.record_success("add_node", improved=True)
        sched.record_success("add_node", improved=False)
        self.assertEqual(sched._global_total["add_node"], 2)
        self.assertEqual(sched._global_success["add_node"], 1)

    def test_unknown_op_ignored(self):
        sched = OperatorScheduler()
        sched.record_success("nonexistent_op", improved=True)
        self.assertNotIn("nonexistent_op", sched._global_total)

    def test_get_weights_for_unknown_species_returns_global(self):
        sched = OperatorScheduler()
        w = sched.get_weights_for_species(species_id=None)
        self.assertEqual(w, sched.global_weights)

    def test_tick_increments_count(self):
        sched = OperatorScheduler()
        pop = _make_population()
        sched.tick(pop)
        self.assertEqual(sched._tick_count, 1)

    def test_tick_does_not_crash_on_empty_population(self):
        sched = OperatorScheduler()
        pop = _make_population()
        pop._evaluated = []
        pop._species = []
        sched.tick(pop)  # should not raise

    def test_high_success_rate_boosts_weight(self):
        sched = OperatorScheduler()
        pop = _make_population()
        # Set success data directly on the population (sync_from_population reads these)
        pop._mutation_total["add_node"] = 20
        pop._mutation_success["add_node"] = 20  # 100% success
        sched.tick(pop)
        self.assertGreater(sched.global_weights["add_node"], 1.0)

    def test_low_success_rate_dampens_weight(self):
        sched = OperatorScheduler()
        pop = _make_population()
        # Set failure data directly on the population
        pop._mutation_total["add_node"] = 20
        pop._mutation_success["add_node"] = 0  # 0% success
        sched.tick(pop)
        self.assertLess(sched.global_weights["add_node"], 1.0)

    def test_weights_stay_within_bounds(self):
        sched = OperatorScheduler()
        pop = _make_population()
        # Extreme success signals via population tracking
        pop._mutation_total["add_node"] = 100
        pop._mutation_success["add_node"] = 100
        for _ in range(20):
            sched.tick(pop)
        for op, w in sched.global_weights.items():
            self.assertGreaterEqual(w, MIN_WEIGHT, f"{op} weight below minimum")
            self.assertLessEqual(w, MAX_WEIGHT, f"{op} weight above maximum")

    def test_plateau_boosts_exploration_ops(self):
        sched = OperatorScheduler()
        pop = _make_population()
        pop._stagnation_count = pop.stagnation_threshold  # full plateau
        sched.tick(pop)
        # add_node should be boosted during stagnation
        self.assertGreaterEqual(sched.global_weights["add_node"], 1.0)

    def test_get_diagnostics_has_required_keys(self):
        sched = OperatorScheduler()
        diag = sched.get_diagnostics()
        for key in ("tick", "global_weights", "qd_pressure", "pruning_pressure", "operator_success_rates"):
            self.assertIn(key, diag)

    def test_per_species_success_tracked(self):
        sched = OperatorScheduler()
        sched.record_success("add_node", improved=True, species_id=1)
        sched.record_success("add_node", improved=False, species_id=1)
        self.assertEqual(sched._species_total[1]["add_node"], 2)
        self.assertEqual(sched._species_success[1]["add_node"], 1)

    def test_apply_to_genome_scales_rates(self):
        sched = OperatorScheduler()
        # Boost add_node weight to max
        sched.global_weights["add_node"] = 3.0
        genome = Genome()
        original_rate = genome.mutation_add_node.bool_rate
        sched.apply_to_genome(genome)
        # Rate should have increased (capped at 0.999)
        self.assertGreater(genome.mutation_add_node.bool_rate, original_rate)


# ---------------------------------------------------------------------------
# LamarckRefiner — budget and per-species eligibility
# ---------------------------------------------------------------------------

class TestLamarckRefinerBudget(unittest.TestCase):

    def test_default_budget_unlimited(self):
        r = LamarckRefiner()
        self.assertIsNone(r.budget_per_gen)
        self.assertIsNone(r.budget_remaining)

    def test_set_budget_stores_value(self):
        r = LamarckRefiner()
        r.set_budget(10)
        self.assertEqual(r.budget_per_gen, 10)

    def test_consume_within_budget(self):
        r = LamarckRefiner()
        r.set_budget(5)
        self.assertTrue(r._consume_budget(3))
        self.assertTrue(r._consume_budget(2))

    def test_consume_exceeds_budget(self):
        r = LamarckRefiner()
        r.set_budget(5)
        r._consume_budget(5)
        result = r._consume_budget(1)
        self.assertFalse(result)

    def test_consume_increments_exhausted_count(self):
        r = LamarckRefiner()
        r.set_budget(2)
        r._consume_budget(2)
        r._consume_budget(1)
        self.assertEqual(r._budget_exhausted_count, 1)

    def test_budget_remaining_decrements(self):
        r = LamarckRefiner()
        r.set_budget(10)
        r._consume_budget(4)
        self.assertEqual(r.budget_remaining, 6)

    def test_budget_remaining_clamped_to_zero(self):
        r = LamarckRefiner()
        r.set_budget(3)
        r._consume_budget(10)
        self.assertEqual(r.budget_remaining, 0)

    def test_reset_generation_budget_clears_used(self):
        r = LamarckRefiner()
        r.set_budget(5)
        r._consume_budget(4)
        r.reset_generation_budget()
        self.assertEqual(r._budget_used_this_gen, 0)
        self.assertTrue(r._consume_budget(4))

    def test_unlimited_always_returns_true(self):
        r = LamarckRefiner()
        for _ in range(1000):
            self.assertTrue(r._consume_budget(1))


class TestLamarckRefinerEligibility(unittest.TestCase):

    def test_all_eligible_by_default(self):
        r = LamarckRefiner()
        self.assertTrue(r.is_eligible_for_species(42))
        self.assertTrue(r.is_eligible_for_species(None))

    def test_restricted_to_set(self):
        r = LamarckRefiner()
        r.set_eligible_species({10, 20})
        self.assertTrue(r.is_eligible_for_species(10))
        self.assertTrue(r.is_eligible_for_species(20))
        self.assertFalse(r.is_eligible_for_species(30))

    def test_none_species_id_eligible_even_with_restriction(self):
        r = LamarckRefiner()
        r.set_eligible_species({10})
        self.assertTrue(r.is_eligible_for_species(None))

    def test_clear_restriction_with_none(self):
        r = LamarckRefiner()
        r.set_eligible_species({10})
        r.set_eligible_species(None)
        self.assertTrue(r.is_eligible_for_species(99))

    def test_record_species_stats_initialises(self):
        r = LamarckRefiner()
        r.record_species_stats(species_id=1, n_steps=3, improved=True)
        self.assertIn(1, r._species_stats)
        self.assertEqual(r._species_stats[1]["applied"], 1)
        self.assertEqual(r._species_stats[1]["steps"], 3)
        self.assertEqual(r._species_stats[1]["improved"], 1)

    def test_record_species_stats_accumulates(self):
        r = LamarckRefiner()
        r.record_species_stats(1, 2, True)
        r.record_species_stats(1, 3, False)
        self.assertEqual(r._species_stats[1]["applied"], 2)
        self.assertEqual(r._species_stats[1]["steps"], 5)
        self.assertEqual(r._species_stats[1]["improved"], 1)


# ---------------------------------------------------------------------------
# NeuroEvolution API — set_lamarck_budget / set_adaptive_control / set_operator_scheduler
# ---------------------------------------------------------------------------

class TestNeuroEvolutionAdaptiveAPI(unittest.TestCase):

    def test_set_lamarck_budget_propagates(self):
        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane.set_lamarck_budget(budget_per_gen=50)
        self.assertEqual(yane._lamarck.budget_per_gen, 50)

    def test_set_adaptive_control_enables(self):
        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane.set_adaptive_control(enabled=True)
        self.assertTrue(yane._adaptive_ctrl_enabled)

    def test_set_adaptive_control_disabled_by_default(self):
        yane = NeuroEvolution()
        yane.configure(2, 1)
        self.assertFalse(yane._adaptive_ctrl_enabled)

    def test_set_operator_scheduler_enabled(self):
        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane.set_operator_scheduler(enabled=True)
        self.assertTrue(yane._operator_scheduler_enabled)

    def test_operator_scheduler_wired_to_population(self):
        yane = NeuroEvolution()
        yane.set_operator_scheduler(enabled=True)
        yane.configure(2, 1)
        self.assertIsNotNone(yane._population._operator_scheduler)

    def test_population_info_has_adaptive_controller_when_enabled(self):
        yane = NeuroEvolution()
        yane.set_adaptive_control(enabled=True)
        yane.configure(2, 1)
        info = yane.population_memory_info()
        self.assertIn("adaptive_controller", info)

    def test_population_info_no_adaptive_controller_when_disabled(self):
        yane = NeuroEvolution()
        yane.configure(2, 1)
        info = yane.population_memory_info()
        self.assertNotIn("adaptive_controller", info)

    def test_population_info_has_operator_scheduler_when_enabled(self):
        yane = NeuroEvolution()
        yane.set_operator_scheduler(enabled=True)
        yane.configure(2, 1)
        info = yane.population_memory_info()
        self.assertIn("operator_scheduler", info)


# ---------------------------------------------------------------------------
# Interspecies crossover — adaptive triggers via population
# ---------------------------------------------------------------------------

class TestAdaptiveInterspeciesCrossover(unittest.TestCase):
    def test_neuro_evolution_configures_adaptive_interspecies_crossover(self):
        yane = NeuroEvolution()
        yane.set_adaptive_interspecies_crossover(min_rate=0.01, max_rate=0.15)
        yane.configure(2, 1)

        info = yane.population_memory_info()

        self.assertEqual(info["interspecies_crossover_mode"], "adaptive")
        self.assertAlmostEqual(info["interspecies_crossover_min"], 0.01)
        self.assertAlmostEqual(info["interspecies_crossover_max"], 0.15)

    def test_adaptive_interspecies_rate_ramps_with_stagnation(self):
        pop = Population(max_size=10, initial_genome=Genome(), target_species=2)
        g1 = Genome()
        g2 = Genome()
        sp1 = Species(g1)
        sp2 = Species(g2)
        sp1.add(g1)
        sp2.add(g2)
        sp2.stagnation_count = pop.stagnation_threshold
        pop._species = [sp1, sp2]
        pop.configure_interspecies_crossover(
            mode="adaptive",
            min_rate=0.01,
            max_rate=0.2,
        )

        rate = pop._adaptive_interspecies_rate()

        self.assertAlmostEqual(rate, 0.2)
        self.assertEqual(pop._interspecies_crossover_last_reason, "adaptive:species_stagnation")

    def test_adaptive_interspecies_rate_zero_with_single_species(self):
        pop = Population(max_size=10, initial_genome=Genome(), target_species=2)
        g = Genome()
        sp = Species(g)
        sp.add(g)
        pop._species = [sp]
        pop.configure_interspecies_crossover(
            mode="adaptive",
            min_rate=0.01,
            max_rate=0.2,
        )

        self.assertEqual(pop._adaptive_interspecies_rate(), 0.0)
        self.assertEqual(pop._interspecies_crossover_last_reason, "adaptive:single_species")

    def test_adaptive_interspecies_novelty_trigger(self):
        """Low mean novelty (< 0.3) should increase interspecies rate."""
        pop = Population(max_size=10, initial_genome=Genome(), target_species=2)
        g1, g2 = Genome(), Genome()
        sp1, sp2 = Species(g1), Species(g2)
        sp1.add(g1)
        sp2.add(g2)
        pop._species = [sp1, sp2]
        pop.configure_interspecies_crossover(
            mode="adaptive",
            min_rate=0.0,
            max_rate=0.2,
        )
        # Seed novelty cache with very low values
        pop._novelty_cache = {id(g1): 0.05, id(g2): 0.05}

        rate = pop._adaptive_interspecies_rate()
        self.assertGreater(rate, 0.0)
        self.assertIn("novelty", pop._interspecies_crossover_last_reason)

    def test_adaptive_interspecies_isolation_trigger(self):
        """Species isolated from interspecies events should get higher rate."""
        pop = Population(max_size=10, initial_genome=Genome(), target_species=2)
        g1, g2 = Genome(), Genome()
        sp1, sp2 = Species(g1), Species(g2)
        sp1.add(g1)
        sp2.add(g2)
        pop._species = [sp1, sp2]
        pop._spawn_count = 200
        pop.configure_interspecies_crossover(
            mode="adaptive",
            min_rate=0.0,
            max_rate=0.2,
        )
        # Both species have been "isolated" (no interspecies events)
        pop._species_last_interspecies = {}

        rate = pop._adaptive_interspecies_rate()
        self.assertGreater(rate, 0.0)
        self.assertIn("isolation", pop._interspecies_crossover_last_reason)

    def test_adaptive_interspecies_protection_dampens_rate(self):
        """Low cross-species success rate should dampen the interspecies rate."""
        pop = Population(max_size=10, initial_genome=Genome(), target_species=2)
        g1, g2 = Genome(), Genome()
        sp1, sp2 = Species(g1), Species(g2)
        sp1.add(g1)
        sp2.add(g2)
        # Force full stagnation to get max pressure without protection
        sp2.stagnation_count = pop.stagnation_threshold
        pop._species = [sp1, sp2]
        pop._stagnation_count = pop.stagnation_threshold
        pop.configure_interspecies_crossover(
            mode="adaptive",
            min_rate=0.0,
            max_rate=0.2,
        )
        # Inject poor success history
        pop._interspecies_n_offspring = 20
        pop._interspecies_n_improved = 2  # 10% success

        rate_protected = pop._adaptive_interspecies_rate()

        # Without protection, a fresh pop at full stagnation would get max_rate
        pop2 = Population(max_size=10, initial_genome=Genome(), target_species=2)
        sp3, sp4 = Species(Genome()), Species(Genome())
        sp3.add(Genome())
        sp4.add(Genome())
        sp4.stagnation_count = pop2.stagnation_threshold
        pop2._species = [sp3, sp4]
        pop2._stagnation_count = pop2.stagnation_threshold
        pop2.configure_interspecies_crossover(
            mode="adaptive", min_rate=0.0, max_rate=0.2
        )

        rate_unprotected = pop2._adaptive_interspecies_rate()
        self.assertLess(rate_protected, rate_unprotected)


# ---------------------------------------------------------------------------
# AdaptiveLamarck modes (already covered but re-anchored in new test class)
# ---------------------------------------------------------------------------

class TestAdaptiveLamarckModes(unittest.TestCase):
    def test_adaptive_lamarck_optimizer_uses_requested_max_steps(self):
        yane = NeuroEvolution()

        yane.set_lamarck_adaptive(max_steps=1, mode="nes")
        self.assertEqual(yane._lamarck.mode, "nes_adaptive")
        self.assertEqual(yane._lamarck.max_steps, 1)

        yane.set_lamarck_adaptive(max_steps=2, mode="sa")
        self.assertEqual(yane._lamarck.mode, "sa_adaptive")
        self.assertEqual(yane._lamarck.max_steps, 2)

        yane.set_lamarck_adaptive(max_steps=4, mode="cma_es")
        self.assertEqual(yane._lamarck.mode, "cma_es_adaptive")
        self.assertEqual(yane._lamarck.max_steps, 4)


if __name__ == "__main__":
    unittest.main()
