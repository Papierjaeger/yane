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
        self.assertGreaterEqual(tab.combo_lamarck_schedule.findText("Adaptiv"), 0)
        self.assertGreaterEqual(tab.combo_lamarck_optimizer.findText("CMA-ES"), 0)
        self.assertGreaterEqual(tab.combo_interspecies_mode.findText("Adaptiv"), 0)
        self.assertTrue(hasattr(tab, "chk_multi_objective"))
        self.assertTrue(hasattr(tab, "chk_quality_diversity"))
        self.assertTrue(hasattr(tab, "preset_combo"))
        self.assertTrue(hasattr(tab, "chk_matrix_forward"))
        self.assertTrue(hasattr(tab, "chk_fitness_components"))
        self.assertTrue(hasattr(tab, "chk_cppn_substrate"))
        self.assertTrue(hasattr(tab, "chk_remote_eval"))
        self.assertTrue(hasattr(tab, "edit_remote_urls"))
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
            "novelty",
            "speciation",
            "crossover",
            "diversity_injection",
            "lamarck_schedule",
            "lamarck_optimizer",
            "lamarck_steps",
        }
        required_adaptive = {
            "adaptive_controller",
            "operator_scheduler",
            "interspecies_mode",
            "interspecies_min_rate",
            "interspecies_max_rate",
            "lamarck_schedule",
            "lamarck_optimizer",
            "lamarck_budget",
            "meta_adaptive",
            "module_library",
            "module_insert_rate",
        }

        for ex in load_examples():
            self.assertTrue(required_config <= set(ex.default_config), ex.name)
            self.assertTrue(required_adaptive <= set(ex.default_adaptive_policies), ex.name)

    def test_example_defaults_are_applied_to_new_feature_controls(self):
        from yane.gui.tabs.training_tab import TrainingTab

        tab = TrainingTab()
        self.assertEqual(tab._current_example().name, "XOR")
        self.assertEqual(tab.spin_workers.value(), 1)
        self.assertTrue(tab.chk_matrix_forward.isChecked())
        self.assertEqual(tab.combo_lamarck_schedule.currentText(), "Explizit")
        self.assertEqual(tab.spin_lamarck.value(), 2)

        reg33_idx = next(
            idx for idx, ex in tab._combo_index_map.items()
            if ex.name == "Regression 3→3"
        )
        tab.example_combo.setCurrentIndex(reg33_idx)
        self.assertTrue(tab.chk_quality_diversity.isChecked())
        self.assertTrue(tab.chk_fitness_components.isChecked())
        self.assertTrue(tab.chk_cppn_substrate.isChecked())
        self.assertTrue(tab.chk_adaptive_ctrl.isChecked())
        self.assertTrue(tab.chk_operator_scheduler.isChecked())
        self.assertTrue(tab.chk_module_library.isChecked())
        self.assertEqual(tab.combo_interspecies_mode.currentText(), "Adaptiv")

        xor_idx = next(idx for idx, ex in tab._combo_index_map.items() if ex.name == "XOR")
        tab.example_combo.setCurrentIndex(xor_idx)
        self.assertFalse(tab.chk_quality_diversity.isChecked())
        self.assertFalse(tab.chk_fitness_components.isChecked())
        self.assertFalse(tab.chk_adaptive_ctrl.isChecked())
        self.assertFalse(tab.chk_module_library.isChecked())
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


class TestGUIAdaptiveSection(unittest.TestCase):
    """Stability tests for the Adaptive Control section added in the major update."""

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

    # --- widget existence ---------------------------------------------------

    def test_adaptive_control_widgets_exist(self):
        tab = self._make_tab()
        for attr in (
            "lbl_interspecies_live",
            "lbl_interspecies_trigger",
            "lbl_interspecies_success",
            "chk_adaptive_ctrl",
            "chk_operator_scheduler",
            "spin_lamarck_budget",
            "lbl_lamarck_budget_used",
            "combo_adaptive_preset",
            "lbl_plateau_ratio",
            "lbl_diversity_score",
            "chk_meta_adaptive",
            "chk_module_library",
            "dspin_module_insert_rate",
        ):
            self.assertTrue(hasattr(tab, attr), f"TrainingTab missing attribute: {attr}")
        tab.close()

    # --- _update_adaptive_labels with various dict shapes -------------------

    def test_update_adaptive_labels_empty_dict_does_not_crash(self):
        tab = self._make_tab()
        tab._update_adaptive_labels({})   # must not raise
        self.app.processEvents()
        tab.close()

    def test_update_adaptive_labels_full_dict(self):
        tab = self._make_tab()
        mem = {
            "interspecies_crossover_current": 0.07,
            "interspecies_crossover_last_reason": "adaptive:global_plateau",
            "interspecies_n_offspring": 20,
            "interspecies_n_improved": 10,
            "lamarck_budget_used": 30,
            "lamarck_budget_per_gen": 100,
            "plateau_ratio": 0.5,
            "adaptive_controller": {
                "signals": {
                    "plateau_ratio": 0.42,
                    "diversity_score": 0.31,
                },
            },
        }
        tab._update_adaptive_labels(mem)
        self.app.processEvents()
        self.assertEqual(tab.lbl_interspecies_live.text(), "0.070")
        self.assertIn("plateau", tab.lbl_interspecies_trigger.text())
        self.assertIn("50.0%", tab.lbl_interspecies_success.text())
        self.assertIn("30/100", tab.lbl_lamarck_budget_used.text())
        self.assertEqual(tab.lbl_plateau_ratio.text(), "0.42")
        self.assertEqual(tab.lbl_diversity_score.text(), "0.31")
        tab.close()

    def test_update_adaptive_labels_no_interspecies_offspring(self):
        tab = self._make_tab()
        mem = {
            "interspecies_n_offspring": 0,
            "interspecies_n_improved": 0,
        }
        tab._update_adaptive_labels(mem)
        self.app.processEvents()
        self.assertEqual(tab.lbl_interspecies_success.text(), "—")
        tab.close()

    def test_update_adaptive_labels_unlimited_budget(self):
        tab = self._make_tab()
        tab._update_adaptive_labels({"lamarck_budget_per_gen": None, "lamarck_budget_used": 5})
        self.app.processEvents()
        self.assertIn("unbegrenzt", tab.lbl_lamarck_budget_used.text())
        tab.close()

    def test_update_adaptive_labels_budget_zero_treated_as_unlimited(self):
        tab = self._make_tab()
        tab._update_adaptive_labels({"lamarck_budget_per_gen": 0, "lamarck_budget_used": 3})
        self.app.processEvents()
        self.assertIn("unbegrenzt", tab.lbl_lamarck_budget_used.text())
        tab.close()

    # --- preset combo interaction -------------------------------------------

    def test_preset_konservativ_disables_adaptive_ctrl(self):
        tab = self._make_tab()
        # Start with adaptive enabled, then switch to Konservativ
        tab.chk_adaptive_ctrl.setChecked(True)
        tab.chk_operator_scheduler.setChecked(True)
        idx = tab.combo_adaptive_preset.findText("Konservativ")
        self.assertGreaterEqual(idx, 0)
        tab.combo_adaptive_preset.setCurrentIndex(idx)
        self.app.processEvents()
        self.assertFalse(tab.chk_adaptive_ctrl.isChecked())
        self.assertFalse(tab.chk_operator_scheduler.isChecked())
        tab.close()

    def test_preset_balanciert_enables_adaptive_ctrl(self):
        tab = self._make_tab()
        idx = tab.combo_adaptive_preset.findText("Balanciert")
        self.assertGreaterEqual(idx, 0)
        tab.combo_adaptive_preset.setCurrentIndex(idx)
        self.app.processEvents()
        self.assertTrue(tab.chk_adaptive_ctrl.isChecked())
        tab.close()

    def test_preset_analysefreundlich_sets_budget(self):
        tab = self._make_tab()
        idx = tab.combo_adaptive_preset.findText("Analysefreundlich")
        self.assertGreaterEqual(idx, 0)
        tab.combo_adaptive_preset.setCurrentIndex(idx)
        self.app.processEvents()
        self.assertGreater(tab.spin_lamarck_budget.value(), 0)
        tab.close()

    def test_preset_aggressiv_enables_scheduler(self):
        tab = self._make_tab()
        idx = tab.combo_adaptive_preset.findText("Aggressiv")
        self.assertGreaterEqual(idx, 0)
        tab.combo_adaptive_preset.setCurrentIndex(idx)
        self.app.processEvents()
        self.assertTrue(tab.chk_operator_scheduler.isChecked())
        tab.close()

    # --- crash-state snapshot format ----------------------------------------

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

    # --- reproducibility: adaptive widget state survives show/hide ----------

    def test_adaptive_labels_update_after_widget_hidden_and_shown(self):
        tab = self._make_tab()
        tab.lbl_plateau_ratio.hide()
        tab.lbl_plateau_ratio.show()
        tab._update_adaptive_labels({"plateau_ratio": 0.77})
        self.app.processEvents()
        self.assertEqual(tab.lbl_plateau_ratio.text(), "0.77")
        tab.close()

    # --- preset file adaptive_policies applied to widgets -------------------

    def test_builtin_adaptive_preset_applies_to_widgets(self):
        """Loading an adaptive profile preset via the main preset combo applies adaptive_policies."""
        from yane.util.presets import PRESET_DIR, load_preset
        from yane.gui.tabs.training_tab import TrainingTab

        preset = load_preset("adaptive_analysefreundlich", preset_dir=PRESET_DIR)
        tab = self._make_tab()
        # Simulate loading via _apply_adaptive_policies directly
        tab._apply_adaptive_policies(preset.adaptive_policies)
        self.app.processEvents()
        self.assertTrue(tab.chk_adaptive_ctrl.isChecked())
        self.assertTrue(tab.chk_operator_scheduler.isChecked())
        self.assertGreater(tab.spin_lamarck_budget.value(), 0)
        self.assertEqual(tab.combo_interspecies_mode.currentText(), "Adaptiv")
        tab.close()

    def test_konservativ_preset_applies_disables_features(self):
        from yane.util.presets import PRESET_DIR, load_preset

        preset = load_preset("adaptive_konservativ", preset_dir=PRESET_DIR)
        tab = self._make_tab()
        tab.chk_adaptive_ctrl.setChecked(True)
        tab.chk_operator_scheduler.setChecked(True)
        tab._apply_adaptive_policies(preset.adaptive_policies)
        self.app.processEvents()
        self.assertFalse(tab.chk_adaptive_ctrl.isChecked())
        self.assertFalse(tab.chk_operator_scheduler.isChecked())
        tab.close()


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


if __name__ == "__main__":
    unittest.main()
