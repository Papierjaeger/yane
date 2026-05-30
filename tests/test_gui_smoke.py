import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication


def _app():
    app = QApplication.instance()
    return app or QApplication([])


class TestGUISmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def test_training_left_inspect_and_mainwindow_instantiate(self):
        from yane.gui.tabs.training_tab import TrainingTab
        from yane.gui.panels.left_panel import LeftPanel
        from yane.gui.tabs.inspect_tab import InspectTab
        from yane.gui.window import MainWindow

        widgets = [TrainingTab(), LeftPanel(), InspectTab(), MainWindow()]
        for widget in widgets:
            widget.resize(1000, 700)
            widget.show()
            self.app.processEvents()
            self.assertGreater(widget.width(), 0)
            widget.close()

    def test_advanced_controls_exist(self):
        from yane.gui.tabs.training_tab import TrainingTab

        tab = TrainingTab()
        self.assertTrue(hasattr(tab, "preset_combo"))
        self.assertTrue(hasattr(tab, "chk_remote_eval"))
        self.assertTrue(hasattr(tab, "edit_remote_urls"))
        self.assertTrue(hasattr(tab, "chk_auto_checkpoint"))
        tab.close()

    def test_training_tab_builds_remote_evaluation_config(self):
        from yane.gui.tabs.training_tab import TrainingTab

        tab = TrainingTab()
        tab.chk_remote_eval.setChecked(True)
        tab.edit_remote_urls.setText("http://localhost:8700, http://worker:8700")
        tab.edit_remote_token.setText("secret")
        tab.spin_remote_batch.setValue(8)

        cfg = tab._current_remote_config()

        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.worker_urls, ("http://localhost:8700", "http://worker:8700"))
        self.assertEqual(cfg.token, "secret")
        self.assertEqual(cfg.effective_batch_size, 8)
        tab.close()

    def test_all_examples_have_complete_gui_defaults(self):
        from yane.gui.examples import load_examples

        required_config = {
            "n_workers",
            "multi_eval",
            "aggregation",
            "sigma_penalty",
            "fitness_shaping",
            "multi_objective",
            "quality_diversity",
            "fitness_components",
            "matrix_forward",
            "cppn_substrate",
            "remote_eval",
        }

        for ex in load_examples():
            self.assertTrue(required_config <= set(ex.default_config), ex.name)

    def test_example_defaults_are_applied_to_new_feature_controls(self):
        from yane.gui.tabs.training_tab import TrainingTab

        tab = TrainingTab()
        self.assertEqual(tab._current_example().name, "XOR")
        self.assertTrue(tab.chk_memory.isChecked() is False)
        tab.close()

    def test_training_worker_remote_bootstraps_first_genome(self):
        from yane import NeuroEvolution
        from yane.gui.remote_config import RemoteEvaluationConfig
        from yane.gui.worker import TrainingWorker

        class FakeRemoteClient:
            def __init__(self, **kwargs):
                pass

            def evaluate_batch(self, genomes):
                return [(genome, 1.0) for genome in genomes]

            def close(self):
                pass

        yane = NeuroEvolution()
        yane.configure(1, 1)
        yane.set_min_fitness(1.0)
        worker = TrainingWorker(
            yane,
            lambda render_cb: None,
            remote_config=RemoteEvaluationConfig(enabled=True, worker_urls=("http://fake",)),
        )

        with patch("yane.gui.worker.RemoteEvaluationClient", FakeRemoteClient):
            worker._running = True
            worker._run_remote(0.0)

        self.assertEqual(yane.population.evaluated_count, 1)

    def test_left_panel_accepts_new_diagnostics(self):
        from yane import NeuroEvolution
        from yane.gui.panels.left_panel import LeftPanel

        yane = NeuroEvolution()
        yane.set_multi_objective(weights=(1.0, -0.01), maximize=(True, False))
        yane.configure(1, 1)
        g = yane.next_genome()
        yane.submit_fitness((1.0, 0.0))
        panel = LeftPanel()
        panel.update_genome(g, yane.population_memory_info(), do_heavy=True)
        self.assertEqual(panel.lbl_multi_objective.text().startswith("on"), True)
        panel.close()

    def test_left_panel_accepts_research_feature_diagnostics(self):
        from yane.gui.panels.left_panel import LeftPanel
        from yane.core.genome import Genome

        panel = LeftPanel()
        g = Genome()
        mem = {
            "total_genomes": 1,
            "matrix_forward_hits": 3,
            "matrix_forward_misses": 1,
            "fitness_component_weights": {
                "mode": "adaptive",
                "last_reason": "adaptive:stagnation",
                "weights": {"a": 0.1, "b": 0.2},
            },
            "meta_adaptive_policies": {
                "last_reason": "evolve:improved",
                "global_genes": {
                    "operator_exploration": 1.2,
                    "lamarck_budget": 42,
                    "interspecies_rate": 0.07,
                },
            },
            "module_library": {
                "module_count": 2,
                "total_uses": 3,
                "reuse_rate": 1.5,
            },
        }
        panel.update_genome(g, mem, do_heavy=False)
        self.assertIn("3/1", panel.lbl_matrix_forward.text())
        self.assertIn("adaptive", panel.lbl_fitness_components.text())
        self.assertIn("budget=42", panel.lbl_meta_policy.text())
        self.assertIn("2 modules", panel.lbl_module_library.text())
        panel.close()

    def test_research_diagnostic_formatter(self):
        from yane.gui.research_features import format_research_diagnostics

        labels = format_research_diagnostics({
            "matrix_forward_hits": 2,
            "matrix_forward_misses": 1,
            "meta_adaptive_policies": {
                "last_reason": "evolve:plateau",
                "global_genes": {
                    "operator_exploration": 1.5,
                    "lamarck_budget": 12,
                    "interspecies_rate": 0.03,
                },
            },
        })

        self.assertEqual(labels["matrix_forward"], "2/1")
        self.assertIn("budget=12", labels["meta_policy"])
        self.assertEqual(labels["module_library"], "—")

    def test_research_feature_helpers_apply_yane_controls(self):
        from yane import NeuroEvolution
        from yane.gui.research_features import ResearchFeatureConfig, apply_research_features

        yane = NeuroEvolution()
        yane.configure(2, 1)
        cfg = ResearchFeatureConfig(
            n_inputs=2,
            n_outputs=1,
            max_nodes=10,
            max_connections=20,
            population_size=5,
            target_species=3,
            allow_memory=False,
            output_sanitize=False,
            output_fallback=0.0,
            matrix_forward=True,
            fitness_components=True,
            fitness_component_mode="Adaptiv",
            meta_adaptive=True,
            module_library=True,
            module_insert_rate=0.25,
        )

        apply_research_features(yane, cfg)

        self.assertTrue(yane._matrix_forward_enabled)
        self.assertTrue(yane._operator_scheduler_enabled)
        self.assertIsNotNone(yane.get_fitness_component_weights())
        self.assertIsNotNone(yane.get_meta_adaptive_policies())
        self.assertIsNotNone(yane.get_module_library())
        self.assertAlmostEqual(yane.population._module_insert_rate, 0.25)

    def test_cppn_substrate_helper_replaces_seed_population(self):
        from yane import NeuroEvolution
        from yane.gui.research_features import (
            ResearchFeatureConfig,
            configure_cppn_substrate_population,
        )

        yane = NeuroEvolution(seed=1)
        yane.configure(2, 1, n_initial_hidden=0)
        cfg = ResearchFeatureConfig(
            n_inputs=2,
            n_outputs=1,
            max_nodes=10,
            max_connections=20,
            population_size=7,
            target_species=4,
            allow_memory=False,
            output_sanitize=False,
            output_fallback=0.0,
            cppn_substrate=True,
            cppn_hidden=3,
        )

        configure_cppn_substrate_population(yane, cfg)

        seed = yane.population._unevaluated[0]
        self.assertEqual(yane.population.max_size, 7)
        self.assertEqual(yane.population._target_species, 4)
        self.assertEqual(len(seed.input_nodes), 2)
        self.assertEqual(len(seed.output_nodes), 1)
        self.assertEqual(len(seed.nodes), 6)

    def test_screenshot_layout_smoke(self):
        from yane.gui.window import MainWindow

        with tempfile.TemporaryDirectory() as tmp:
            window = MainWindow()
            for width, height in ((1200, 800), (760, 520)):
                window.resize(width, height)
                window.show()
                self.app.processEvents()
                pixmap = window.grab()
                path = Path(tmp) / f"gui_{width}x{height}.png"
                self.assertTrue(pixmap.save(str(path)))
                self.assertGreater(path.stat().st_size, 0)
            window.close()


class TestGUICrashState(unittest.TestCase):
    """Residual crash-state test that does not depend on removed adaptive widgets."""

    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def test_crash_state_dict_keys_present_in_mem(self):
        """All keys used in the crash-state snapshot must exist in population_memory_info()."""
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(2, 1)
        g = yane.next_genome()
        yane.submit_fitness(1.0)
        mem = yane.population_memory_info()

        crash_keys = [
            "max_fitness", "avg_fitness", "fitness_iqr",
            "species_count", "stagnation_count",
            "largest_genome_nodes", "largest_genome_connections",
            "lamarck_n_applied", "lamarck_mode", "n_invalid_fitness",
        ]
        for key in crash_keys:
            self.assertIn(key, mem, f"crash-state key missing from mem: {key}")


class TestVisualizationWidgets(unittest.TestCase):
    """Smoke tests for Pareto scatter and MAP-Elites heatmap."""

    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def test_pareto_scatter_renders_without_data(self):
        from yane.gui.canvas import ParetoScatter
        w = ParetoScatter()
        w.resize(300, 150)
        w.show()
        self.app.processEvents()
        w.close()

    def test_pareto_scatter_renders_with_data(self):
        from yane.gui.canvas import ParetoScatter
        points = [
            {"objectives": [0.5, 10.0], "fitness": 0.4, "nodes": 5, "connections": 8},
            {"objectives": [0.8, 6.0], "fitness": 0.7, "nodes": 7, "connections": 12},
            {"objectives": [1.0, 4.0], "fitness": 0.9, "nodes": 9, "connections": 15},
        ]
        w = ParetoScatter()
        w.resize(300, 150)
        w.show()
        w.set_points(points)
        self.app.processEvents()
        self.assertEqual(len(w._px), 3)
        w.close()

    def test_map_elites_heatmap_renders_with_data(self):
        from yane.gui.canvas import MapElitesHeatmap
        cells = [
            {"cell": [0, 0], "fitness": 0.1},
            {"cell": [1, 0], "fitness": 0.5},
            {"cell": [1, 1], "fitness": 0.9},
        ]
        w = MapElitesHeatmap()
        w.resize(200, 120)
        w.show()
        w.set_cells(cells)
        self.app.processEvents()
        self.assertEqual(len(w._cell_rects), 3)
        w.close()

    def test_left_panel_has_export_qd_button(self):
        from yane.gui.panels.left_panel import LeftPanel
        panel = LeftPanel()
        self.assertTrue(hasattr(panel, "btn_export_qd"))
        panel.close()

    def test_left_panel_stores_last_qd_cells(self):
        from yane import NeuroEvolution
        from yane.gui.panels.left_panel import LeftPanel
        yane = NeuroEvolution()
        yane.configure(2, 1)
        g = yane.next_genome()
        yane.submit_fitness(1.0)
        mem = yane.population_memory_info()
        panel = LeftPanel()
        panel.update_genome(g, mem, do_heavy=True)
        # Without QD enabled, _last_qd_cells stays None
        self.assertIsNone(panel._last_qd_cells)
        panel.close()


class TestCheckpointMetadataGUI(unittest.TestCase):
    """Tests for _show_checkpoint_metadata: info dialog and reattach warning."""

    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def _make_tab(self):
        from yane.gui.tabs.training_tab import TrainingTab
        tab = TrainingTab()
        tab.resize(1200, 900)
        tab.show()
        self.app.processEvents()
        return tab

    def test_no_crash_when_sidecar_missing(self):
        tab = self._make_tab()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "no_sidecar.pkl"
            path.write_bytes(b"")
            tab._show_checkpoint_metadata(str(path))
        tab.close()

    def test_info_dialog_shown_for_valid_sidecar(self):
        import json
        from unittest.mock import patch
        tab = self._make_tab()
        with tempfile.TemporaryDirectory() as tmp:
            pkl = Path(tmp) / "run.pkl"
            meta = Path(tmp) / "run.pkl.json"
            meta.write_text(json.dumps({
                "version": 2, "created_at": "2026-01-01T00:00:00",
                "config": {"n_inputs": 3, "n_outputs": 1},
                "population_size": 50, "requires_reattach": [],
            }), encoding="utf-8")
            with patch("PySide6.QtWidgets.QMessageBox.information") as mock_info:
                tab._show_checkpoint_metadata(str(pkl))
                mock_info.assert_called_once()
                call_text = mock_info.call_args[0][2]
                self.assertIn("Version", call_text)
                self.assertIn("Pop-Size", call_text)
        tab.close()

    def test_warning_shown_when_reattach_required(self):
        import json
        from unittest.mock import patch
        tab = self._make_tab()
        with tempfile.TemporaryDirectory() as tmp:
            pkl = Path(tmp) / "run.pkl"
            meta = Path(tmp) / "run.pkl.json"
            meta.write_text(json.dumps({
                "version": 2, "created_at": "2026-01-01T00:00:00",
                "config": {}, "population_size": 10,
                "requires_reattach": ["quality_diversity_descriptor"],
            }), encoding="utf-8")
            with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warn, \
                 patch("PySide6.QtWidgets.QMessageBox.information"):
                tab._show_checkpoint_metadata(str(pkl))
                mock_warn.assert_called_once()
                warn_text = mock_warn.call_args[0][2]
                self.assertIn("quality_diversity_descriptor", warn_text)
        tab.close()


class TestAutoTrainGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def test_auto_train_button_exists(self):
        from yane.gui.tabs.training_tab import TrainingTab
        tab = TrainingTab()
        self.assertTrue(hasattr(tab, "btn_auto_train"))
        self.assertTrue(tab.btn_auto_train.isEnabled())
        tab.close()

    def test_left_panel_has_meta_adaptive_labels(self):
        from yane.gui.panels.left_panel import LeftPanel
        panel = LeftPanel()
        for attr in ("lbl_meta_phase", "lbl_meta_overhead", "lbl_meta_ticks",
                     "lbl_meta_changes", "lbl_fg_active", "lbl_fg_status"):
            self.assertTrue(hasattr(panel, attr), f"missing {attr}")
        # group is hidden until P0 is active
        self.assertFalse(panel._meta_grp.isVisible())
        panel.close()

    def test_left_panel_updates_meta_adaptive_from_mem(self):
        from yane.gui.panels.left_panel import LeftPanel
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane.set_population_size(4)
        genome = yane.next_genome()
        yane.submit_fitness(0.5, 1.0)
        genome = yane.get_best().copy()
        mem = yane.population_memory_info()
        mem["meta_optimizer"] = {
            "enabled": True,
            "phase": "EXPLORE",
            "overhead_pct": 2.5,
            "n_ticks": 3,
            "n_skipped": 1,
            "recent_changes": [{"generation": 10, "param": "lamarck.n_steps", "value": 5}],
        }
        mem["feature_gating"] = {
            "enabled": True,
            "n_active": 2,
            "n_testing": 1,
            "features": {
                "curiosity": {"status": "active", "degradation_level": 0.0},
                "lamarck":   {"status": "testing", "degradation_level": 0.3},
            },
        }
        panel = LeftPanel()
        panel.show()
        self.app.processEvents()
        panel.update_genome(genome, mem, do_heavy=False)
        self.app.processEvents()
        self.assertTrue(panel._meta_grp.isVisible())
        self.assertEqual(panel.lbl_meta_phase.text(), "EXPLORE")
        self.assertIn("2.5", panel.lbl_meta_overhead.text())
        self.assertIn("3", panel.lbl_meta_ticks.text())
        self.assertIn("lamarck.n_steps", panel.lbl_meta_changes.text())
        self.assertIn("2", panel.lbl_fg_active.text())
        self.assertIn("curiosity", panel.lbl_fg_status.text())
        panel.close()

    def test_auto_train_worker_profiles_and_signals(self):
        from yane import NeuroEvolution
        from yane.gui.worker import AutoSetupWorker
        from PySide6.QtCore import QCoreApplication

        yane = NeuroEvolution()
        yane.configure(2, 1)
        yane.set_population_size(4)

        def make_eval(render_cb=None):
            def eval_fn(genome):
                return float(sum(genome.forward([0.5, 0.5])))
            return eval_fn

        results = []
        errors = []

        worker = AutoSetupWorker(yane, make_eval, n_warmup=4)
        worker.setup_done.connect(results.append)
        worker.error_occurred.connect(errors.append)
        worker.start()
        worker.wait(10_000)  # max 10 s
        self.app.processEvents()  # deliver queued cross-thread signals

        self.assertEqual(errors, [], errors)
        self.assertEqual(len(results), 1)
        info = results[0]
        self.assertIn("task_type", info)
        self.assertIn("pop_size", info)
        self.assertIn("kb_entries", info)
        # MetaOptimizer and FeatureGating should be configured
        self.assertTrue(yane._meta_optimizer_enabled)
        self.assertIsNotNone(yane._meta_optimizer_obj)
        self.assertTrue(yane._feature_gate_enabled)


class TestFeaturesTabSmoke(unittest.TestCase):
    """Smoke tests for the new FeaturesTab."""

    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def test_features_tab_instantiates(self):
        from yane.gui.tabs.features_tab import FeaturesTab
        tab = FeaturesTab()
        tab.resize(900, 700)
        tab.show()
        self.app.processEvents()
        self.assertGreater(tab.width(), 0)
        tab.close()

    def test_features_tab_has_expected_checkboxes(self):
        from yane.gui.tabs.features_tab import FeaturesTab
        tab = FeaturesTab()
        for attr in (
            "chk_attention", "chk_ltc", "chk_neuromodulation",
            "chk_stdp", "chk_probabilistic",
            "chk_curiosity", "chk_darts", "chk_shared_weights", "chk_augmentation",
            "chk_anytime", "chk_recovery", "chk_pruning",
            "chk_input_grouping", "chk_output_grouping",
            "chk_hardware", "chk_budget", "chk_wandb", "chk_mlflow",
            "chk_phylogeny",
        ):
            self.assertTrue(hasattr(tab, attr), f"Missing widget: {attr}")
        tab.close()

    def test_features_tab_all_off_by_default(self):
        """All feature toggles default to off."""
        from yane.gui.tabs.features_tab import FeaturesTab
        tab = FeaturesTab()
        for attr in (
            "chk_attention", "chk_ltc", "chk_neuromodulation", "chk_stdp",
            "chk_probabilistic", "chk_curiosity", "chk_darts",
            "chk_shared_weights", "chk_augmentation", "chk_anytime",
            "chk_recovery", "chk_pruning", "chk_input_grouping",
            "chk_output_grouping", "chk_hardware", "chk_budget",
            "chk_wandb", "chk_mlflow", "chk_phylogeny",
        ):
            chk = getattr(tab, attr)
            self.assertFalse(chk.isChecked(), f"{attr} should default to off")
        tab.close()

    def test_features_tab_collect_restore_state_roundtrip(self):
        """collect_state / restore_state roundtrip preserves values."""
        from yane.gui.tabs.features_tab import FeaturesTab
        tab = FeaturesTab()
        # Change a few values
        tab.chk_attention.setChecked(True)
        tab.spin_attention_head_dim.setValue(8)
        tab.chk_curiosity.setChecked(True)
        tab.dspin_curiosity_weight.setValue(0.7)
        tab.chk_phylogeny.setChecked(True)
        tab.spin_phylogeny_max.setValue(500)

        state = tab.collect_state()
        tab2 = FeaturesTab()
        tab2.restore_state(state)

        self.assertTrue(tab2.chk_attention.isChecked())
        self.assertEqual(tab2.spin_attention_head_dim.value(), 8)
        self.assertTrue(tab2.chk_curiosity.isChecked())
        self.assertAlmostEqual(tab2.dspin_curiosity_weight.value(), 0.7, places=2)
        self.assertTrue(tab2.chk_phylogeny.isChecked())
        self.assertEqual(tab2.spin_phylogeny_max.value(), 500)
        tab.close()
        tab2.close()

    def test_features_tab_apply_to_ne_all_off_no_crash(self):
        """apply_to_ne with all features off does not crash."""
        from yane import NeuroEvolution
        from yane.gui.tabs.features_tab import FeaturesTab
        tab = FeaturesTab()
        ne = NeuroEvolution(seed=0)
        ne.configure(2, 1)
        tab.apply_to_ne(ne)  # should not raise
        tab.close()

    def test_features_tab_apply_attention_configures_ne(self):
        """Enabling attention in FeaturesTab configures NeuroEvolution correctly."""
        from yane import NeuroEvolution
        from yane.gui.tabs.features_tab import FeaturesTab
        tab = FeaturesTab()
        tab.chk_attention.setChecked(True)
        tab.spin_attention_head_dim.setValue(4)
        tab.spin_attention_num_heads.setValue(2)
        ne = NeuroEvolution(seed=0)
        ne.configure(2, 1)
        tab.apply_to_ne(ne)
        self.assertTrue(ne._attention_enabled)
        tab.close()

if __name__ == "__main__":
    unittest.main()
