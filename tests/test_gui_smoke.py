import os
import tempfile
import unittest
from pathlib import Path

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
        tab.close()

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


if __name__ == "__main__":
    unittest.main()
