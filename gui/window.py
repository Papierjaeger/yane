"""Main application window — assembles panels and tabs."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QTabWidget, QFrame, QStatusBar,
)

from yane.gui.panels.left_panel import LeftPanel
from yane.gui.tabs.training_tab import TrainingTab
from yane.gui.tabs.inspect_tab import InspectTab
from yane.gui.tabs.aux_tabs import ServerTab, DebugTab
from yane.gui.tabs.comparison_tab import ComparisonTab

# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

_QSS = """
QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}
QMainWindow { background-color: #181825; }
QTabWidget::pane {
    border: 1px solid #313244;
    border-radius: 6px;
    background: #1e1e2e;
}
QTabBar::tab {
    background: #313244;
    color: #a6adc8;
    padding: 7px 20px;
    border-radius: 4px 4px 0 0;
    margin-right: 2px;
}
QTabBar::tab:selected { background: #45475a; color: #cdd6f4; }
QTabBar::tab:hover    { background: #3b3d54; }
QGroupBox {
    border: 1px solid #313244;
    border-radius: 6px;
    margin-top: 10px;
    padding: 8px;
    font-weight: bold;
    color: #a6adc8;
}
QGroupBox::title { subcontrol-position: top left; left: 10px; top: -7px; }
QPushButton {
    background-color: #45475a;
    color: #cdd6f4;
    border: 1px solid #585b70;
    border-radius: 5px;
    padding: 6px 16px;
}
QPushButton:hover   { background-color: #585b70; }
QPushButton:pressed { background-color: #313244; }
QPushButton:disabled { color: #585b70; background-color: #313244; }
QPushButton#startBtn  { background-color: #40a02b; border-color: #40a02b; color: white; }
QPushButton#startBtn:hover   { background-color: #50b93c; }
QPushButton#startBtn:disabled { background-color: #2a6b1d; color: #6a9b62; }
QPushButton#stopBtn   { background-color: #d20f39; border-color: #d20f39; color: white; }
QPushButton#stopBtn:hover    { background-color: #e8234d; }
QPushButton#stopBtn:disabled { background-color: #7a0a21; color: #a06070; }
QPushButton#pauseBtn  { background-color: #df8e1d; border-color: #df8e1d; color: white; }
QPushButton#pauseBtn:hover   { background-color: #e9a03a; }
QPushButton#pauseBtn:disabled { background-color: #6b4a0d; color: #a07030; }
QPushButton#serverBtn { background-color: #1e66f5; border-color: #1e66f5; color: white; }
QPushButton#serverBtn:hover  { background-color: #3d7ff7; }
QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 8px;
    color: #cdd6f4;
}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #89b4fa; }
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background: #313244;
    selection-background-color: #45475a;
}
QProgressBar {
    border: 1px solid #45475a;
    border-radius: 4px;
    background: #313244;
    text-align: center;
    color: #cdd6f4;
}
QProgressBar::chunk { background-color: #89b4fa; border-radius: 3px; }
QLabel#statValue { color: #89b4fa; font-size: 14px; font-weight: bold; }
QLabel#sectionTitle { color: #a6adc8; font-size: 11px; }
QLabel#mutRate { font-size: 11px; color: #cba6f7; }
QFrame#divider { background: #313244; max-height: 1px; }
QScrollArea { border: none; }
"""


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YANE — Yet Another Neuro Evolution")
        self.resize(1000, 700)
        self.setStyleSheet(_QSS)
        self._best_fitness_for_dot: float = float('-inf')

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._left = LeftPanel()
        root.addWidget(self._left)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("background: #313244;")
        root.addWidget(sep)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        tabs = self._tabs
        self._training_tab    = TrainingTab()
        self._inspect_tab     = InspectTab()
        self._comparison_tab  = ComparisonTab()
        self._server_tab      = ServerTab()
        self._debug_tab       = DebugTab()
        tabs.addTab(self._training_tab,   "  Training  ")
        tabs.addTab(self._inspect_tab,    "  Inspect  ")
        tabs.addTab(self._comparison_tab, "  Vergleich  ")
        tabs.addTab(self._server_tab,     "  API Server  ")
        tabs.addTab(self._debug_tab,      "  Debug  ")

        self._training_tab.genome_updated.connect(self._left.update_genome)
        self._training_tab.genome_updated.connect(self._on_genome_for_inspect)
        self._training_tab.genome_updated.connect(self._debug_tab.on_update)
        self._training_tab.example_changed.connect(self._inspect_tab.set_example)
        self._training_tab.training_started.connect(self._inspect_tab.reset_genome)
        self._training_tab.training_started.connect(self._reset_best_fitness_for_dot)
        self._training_tab.training_started.connect(self._debug_tab.on_training_started)
        self._training_tab.training_started.connect(self._left.sigma_chart.clear)
        tabs.currentChanged.connect(self._on_tab_changed)
        root.addWidget(tabs, stretch=1)

        # Initialise InspectTab with the already-selected example (signal was emitted
        # during TrainingTab.__init__ before the connection existed)
        initial_ex = self._training_tab._current_example()
        if initial_ex:
            self._inspect_tab.set_example(initial_ex)

        self.setCentralWidget(central)

        bar = QStatusBar()
        bar.showMessage("YANE ready — select an example and press Start.")
        self.setStatusBar(bar)

    def _on_genome_for_inspect(self, genome, mem: dict, do_heavy: bool) -> None:
        self._inspect_tab.update_genome(genome, mem)
        if (genome.fitness > self._best_fitness_for_dot
                and self._tabs.currentWidget() is not self._inspect_tab):
            self._best_fitness_for_dot = genome.fitness
            self._tabs.setTabText(1, "  Inspect ●  ")

    def _reset_best_fitness_for_dot(self) -> None:
        self._best_fitness_for_dot = float('-inf')

    def _on_tab_changed(self, index: int) -> None:
        if self._tabs.widget(index) is self._inspect_tab:
            self._tabs.setTabText(1, "  Inspect  ")
        elif self._tabs.widget(index) is self._comparison_tab:
            self._comparison_tab.refresh()

    def closeEvent(self, event) -> None:
        w = self._training_tab._worker
        if w and w.isRunning():
            self.setWindowTitle("YANE — Stopping…")
            w.stop()
            if not w.wait(5000):
                # gc.collect()/malloc_trim still running after 5 s — force-terminate
                from yane.util.logger import get_logger
                get_logger().warning("Worker did not stop within 5 s — terminating forcefully")
                w.terminate()
                w.wait(1000)
        event.accept()
