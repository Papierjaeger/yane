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
        self.assertGreaterEqual(tab.combo_lamarck_mode.findText("Explizit CMA-ES"), 0)
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


if __name__ == "__main__":
    unittest.main()
