"""Main application window."""
from __future__ import annotations
import time as _time

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QTabWidget,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QGroupBox, QFormLayout, QProgressBar, QStatusBar, QSizePolicy,
    QFrame, QScrollArea, QMessageBox,
)
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QImage, QPixmap

from yane.gui.canvas import NetworkCanvas, FitnessChart
from yane.gui.worker import TrainingWorker, EpisodeRunner, ServerThread
from yane.gui.examples import load_examples

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
# Helpers
# ---------------------------------------------------------------------------

def _label(text: str, obj_name: str = "") -> QLabel:
    lbl = QLabel(text)
    if obj_name:
        lbl.setObjectName(obj_name)
    return lbl


def _divider() -> QFrame:
    f = QFrame()
    f.setObjectName("divider")
    f.setFrameShape(QFrame.Shape.HLine)
    return f


# ---------------------------------------------------------------------------
# Left panel — network + population stats
# ---------------------------------------------------------------------------

class LeftPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(285)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 4, 8)
        outer.setSpacing(6)

        title = QLabel("Network")
        font = QFont(); font.setPointSize(11); font.setBold(True)
        title.setFont(font)
        outer.addWidget(title)

        self.canvas = NetworkCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setMinimumHeight(160)
        outer.addWidget(self.canvas, stretch=1)

        outer.addWidget(_divider())

        # Everything below the canvas is scrollable so it never gets cut off.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(6)

        # ── Best Genome ────────────────────────────────────────────────────
        best = QGroupBox("Best Genome")
        best_layout = QFormLayout(best)
        best_layout.setSpacing(4)
        self.lbl_nodes        = _label("—", "statValue")
        self.lbl_connections  = _label("—", "statValue")
        self.lbl_best_fit     = _label("—", "statValue")
        self.lbl_shared_fit   = _label("—", "statValue")
        best_layout.addRow("Nodes:",          self.lbl_nodes)
        best_layout.addRow("Connections:",    self.lbl_connections)
        best_layout.addRow("Fitness:",        self.lbl_best_fit)
        best_layout.addRow("Shared fitness:", self.lbl_shared_fit)
        layout.addWidget(best)

        # ── Population ────────────────────────────────────────────────────
        pop = QGroupBox("Population")
        pop_layout = QFormLayout(pop)
        pop_layout.setSpacing(4)
        self.lbl_population  = _label("—", "statValue")
        self.lbl_species     = _label("—", "statValue")
        self.lbl_avg_nodes   = _label("—", "statValue")
        self.lbl_avg_conns   = _label("—", "statValue")
        self.lbl_stagnation    = _label("—", "statValue")
        self.lbl_novelty_weight = _label("—", "statValue")
        pop_layout.addRow("Genomes:",        self.lbl_population)
        pop_layout.addRow("Species:",        self.lbl_species)
        pop_layout.addRow("Avg nodes:",      self.lbl_avg_nodes)
        pop_layout.addRow("Avg conns:",      self.lbl_avg_conns)
        pop_layout.addRow("Stagnation:",     self.lbl_stagnation)
        pop_layout.addRow("Novelty weight:", self.lbl_novelty_weight)
        layout.addWidget(pop)

        # ── Structure mutations (best genome) ─────────────────────────────
        struct = QGroupBox("Structure mutations")
        struct_layout = QFormLayout(struct)
        struct_layout.setSpacing(3)
        self.lbl_rate_add_node  = _label("—", "mutRate")
        self.lbl_rate_rem_node  = _label("—", "mutRate")
        self.lbl_rate_add_conn  = _label("—", "mutRate")
        self.lbl_rate_rem_conn  = _label("—", "mutRate")
        # bypass_connection_prob: when removing a node, probability of creating
        # a shortcut A→B for each A→N→B path to preserve connectivity.
        self.lbl_bypass         = _label("—", "mutRate")
        struct_layout.addRow("Add node:",    self.lbl_rate_add_node)
        struct_layout.addRow("Rem node:",    self.lbl_rate_rem_node)
        struct_layout.addRow("Add conn:",    self.lbl_rate_add_conn)
        struct_layout.addRow("Rem conn:",    self.lbl_rate_rem_conn)
        struct_layout.addRow("Bypass prob:", self.lbl_bypass)
        layout.addWidget(struct)

        # ── Strategy genes (best genome) ──────────────────────────────────
        strat = QGroupBox("Strategy genes")
        strat_layout = QFormLayout(strat)
        strat_layout.setSpacing(3)
        self.lbl_crossover_prob   = _label("—", "mutRate")
        self.lbl_offspring_factor = _label("—", "mutRate")
        self.lbl_species_thresh   = _label("—", "mutRate")
        self.lbl_sigma_global     = _label("—", "mutRate")
        strat_layout.addRow("Crossover prob:",   self.lbl_crossover_prob)
        strat_layout.addRow("Offspring factor:", self.lbl_offspring_factor)
        strat_layout.addRow("Species threshold:", self.lbl_species_thresh)
        strat_layout.addRow("Sigma global:",     self.lbl_sigma_global)
        layout.addWidget(strat)

        # ── Weight / Node rates (avg over best genome) ────────────────────
        wn = QGroupBox("Weight / Node rates  (avg)")
        wn_layout = QFormLayout(wn)
        wn_layout.setSpacing(3)
        self.lbl_weight_rate  = _label("—", "mutRate")
        self.lbl_weight_delta = _label("—", "mutRate")
        self.lbl_bias_rate    = _label("—", "mutRate")
        self.lbl_bias_delta   = _label("—", "mutRate")
        self.lbl_activ_rate   = _label("—", "mutRate")
        wn_layout.addRow("Weight rate:", self.lbl_weight_rate)
        wn_layout.addRow("Weight Δ:",    self.lbl_weight_delta)
        wn_layout.addRow("Bias rate:",   self.lbl_bias_rate)
        wn_layout.addRow("Bias Δ:",      self.lbl_bias_delta)
        wn_layout.addRow("Activation:",  self.lbl_activ_rate)
        layout.addWidget(wn)

        layout.addStretch()
        scroll.setWidget(inner)
        outer.addWidget(scroll)

    def update_genome(self, genome, mem: dict) -> None:
        self.canvas.set_genome(genome)

        # Best genome
        self.lbl_nodes.setText(str(len(genome.nodes)))
        self.lbl_connections.setText(str(genome.connection_count))
        self.lbl_best_fit.setText(f"{genome.fitness:.4f}")
        self.lbl_shared_fit.setText(f"{genome.shared_fitness:.4f}")

        # Population aggregates
        self.lbl_population.setText(str(mem.get("total_genomes", "—")))
        self.lbl_species.setText(str(mem.get("species_count", "—")))
        avg_n = mem.get("avg_nodes_per_genome")
        avg_c = mem.get("avg_connections_per_genome")
        self.lbl_avg_nodes.setText(f"{avg_n:.1f}" if avg_n is not None else "—")
        self.lbl_avg_conns.setText(f"{avg_c:.1f}" if avg_c is not None else "—")
        stag = mem.get("stagnation_count")
        thr  = mem.get("stagnation_threshold")
        if stag is not None and thr is not None:
            self.lbl_stagnation.setText(f"{stag} / {thr}")
        nw = mem.get("novelty_weight")
        if nw is not None:
            self.lbl_novelty_weight.setText(f"{nw:.3f}")

        # Structure mutations
        self.lbl_rate_add_node.setText(f"{genome.mutation_add_node.bool_rate:.4f}")
        self.lbl_rate_rem_node.setText(f"{genome.mutation_remove_node.bool_rate:.4f}")
        self.lbl_rate_add_conn.setText(f"{genome.mutation_add_connection.bool_rate:.4f}")
        self.lbl_rate_rem_conn.setText(f"{genome.mutation_remove_connection.bool_rate:.4f}")
        self.lbl_bypass.setText(f"{genome.bypass_connection_prob:.4f}")

        # Strategy genes
        self.lbl_crossover_prob.setText(f"{genome.crossover_prob:.4f}")
        self.lbl_offspring_factor.setText(f"{genome.offspring_factor:.4f}")
        self.lbl_species_thresh.setText(f"{genome.species_threshold:.4f}")
        self.lbl_sigma_global.setText(f"{genome.sigma_global:.4f}")

        # Weight / Node rates (averaged over all nodes/connections)
        nodes = genome.nodes
        conns = [c for n in nodes for c in n.connections]
        if nodes:
            self.lbl_bias_rate.setText(
                f"{sum(n.mutation_bias.shift_rate for n in nodes) / len(nodes):.4f}")
            self.lbl_bias_delta.setText(
                f"{sum(n.mutation_bias.value_delta for n in nodes) / len(nodes):.4f}")
            self.lbl_activ_rate.setText(
                f"{sum(n.mutation_activation.custom_rate for n in nodes) / len(nodes):.4f}")
        if conns:
            self.lbl_weight_rate.setText(
                f"{sum(c.mutation.shift_rate for c in conns) / len(conns):.4f}")
            self.lbl_weight_delta.setText(
                f"{sum(c.mutation.value_delta for c in conns) / len(conns):.4f}")


# ---------------------------------------------------------------------------
# Gym render widget
# ---------------------------------------------------------------------------

class GymRenderWidget(QLabel):
    """Displays gymnasium render frames (rgb_array numpy arrays)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap: QPixmap | None = None
        self._show_placeholder()

    def update_frame(self, frame) -> None:
        import numpy as np
        arr = np.ascontiguousarray(frame, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return
        h, w, _ = arr.shape
        img = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(img.copy())
        self._rescale()

    def clear_frame(self) -> None:
        self._pixmap = None
        self._show_placeholder()

    def _show_placeholder(self) -> None:
        self.clear()
        self.setText("Enable render and start training to see the environment.")
        self.setStyleSheet("color: #6c7086; font-style: italic; font-size: 11px;")

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)
        self.setStyleSheet("")

    def resizeEvent(self, event) -> None:
        self._rescale()
        super().resizeEvent(event)


# ---------------------------------------------------------------------------
# Training tab
# ---------------------------------------------------------------------------

class TrainingTab(QWidget):
    genome_updated = Signal(object, dict)   # → LeftPanel + InspectTab
    example_changed = Signal(object)        # → InspectTab.set_example
    training_started = Signal()             # → InspectTab.reset_genome
    render_frame = Signal(object)           # numpy array, emitted from worker thread

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._examples = load_examples()
        self._worker: TrainingWorker | None = None
        self._episode_runner: EpisodeRunner | None = None
        self._yane = None
        self._best_genome = None
        self._had_error = False
        self._last_ram_color = ""
        self._run_id = 0

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # --- Config ---
        cfg = QGroupBox("Configuration")
        cfg_form = QFormLayout(cfg)
        cfg_form.setSpacing(6)

        self.example_combo = QComboBox()
        for ex in self._examples:
            self.example_combo.addItem(ex.name)
        self.example_combo.currentIndexChanged.connect(self._on_example_changed)
        cfg_form.addRow("Example:", self.example_combo)

        self.desc_label = _label("", "sectionTitle")
        self.desc_label.setWordWrap(True)
        cfg_form.addRow(self.desc_label)

        self.spin_inputs  = QSpinBox(); self.spin_inputs.setRange(1, 1024)
        self.spin_outputs = QSpinBox(); self.spin_outputs.setRange(1, 256)
        self.spin_nodes   = QSpinBox(); self.spin_nodes.setRange(2, 500); self.spin_nodes.setSpecialValueText("unlimited")
        self.spin_conns   = QSpinBox(); self.spin_conns.setRange(1, 5000); self.spin_conns.setSpecialValueText("unlimited")
        self.spin_pop     = QSpinBox(); self.spin_pop.setRange(2, 1000); self.spin_pop.setValue(100)
        self.dspin_mem    = QDoubleSpinBox(); self.dspin_mem.setRange(0.1, 32.0); self.dspin_mem.setSingleStep(0.5); self.dspin_mem.setValue(2.0); self.dspin_mem.setSuffix(" GB")
        self.dspin_target = QDoubleSpinBox(); self.dspin_target.setRange(-1e9, 1e9); self.dspin_target.setSingleStep(0.1); self.dspin_target.setDecimals(4); self.dspin_target.setSpecialValueText("—")

        cfg_form.addRow("Inputs:",         self.spin_inputs)
        cfg_form.addRow("Outputs:",        self.spin_outputs)
        cfg_form.addRow("Max nodes:",      self.spin_nodes)
        cfg_form.addRow("Max connections:", self.spin_conns)
        cfg_form.addRow("Population:",     self.spin_pop)
        cfg_form.addRow("Memory limit:",   self.dspin_mem)
        cfg_form.addRow("Target fitness:", self.dspin_target)
        layout.addWidget(cfg)

        # --- Controls ---
        ctrl = QWidget()
        ctrl_row = QHBoxLayout(ctrl)
        ctrl_row.setContentsMargins(0, 0, 0, 0)
        self.btn_start = QPushButton("▶  Start"); self.btn_start.setObjectName("startBtn")
        self.btn_pause = QPushButton("⏸  Pause"); self.btn_pause.setObjectName("pauseBtn")
        self.btn_stop  = QPushButton("■  Stop");  self.btn_stop.setObjectName("stopBtn")
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.status_lbl = QLabel("Idle")
        self.btn_start.clicked.connect(self.start_training)
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_stop.clicked.connect(self.stop_training)
        self.btn_render = QPushButton("Render: Off")
        self.btn_render.setCheckable(True)
        self.btn_render.setVisible(False)
        self.btn_render.toggled.connect(self._on_render_toggled)
        self.btn_run_best = QPushButton("▶  Run Best")
        self.btn_run_best.setVisible(False)
        self.btn_run_best.setEnabled(False)
        self.btn_run_best.clicked.connect(self._run_best_episode)
        ctrl_row.addWidget(self.btn_start)
        ctrl_row.addWidget(self.btn_pause)
        ctrl_row.addWidget(self.btn_stop)
        ctrl_row.addWidget(self.btn_render)
        ctrl_row.addWidget(self.btn_run_best)
        ctrl_row.addWidget(self.status_lbl)
        ctrl_row.addStretch()
        layout.addWidget(ctrl)

        # --- Progress ---
        prog = QGroupBox("Progress")
        prog_layout = QVBoxLayout(prog)

        self.lbl_iter    = _label("Iteration: —")
        self.lbl_fitness = _label("Fitness: —")
        prog_layout.addWidget(self.lbl_iter)
        prog_layout.addWidget(self.lbl_fitness)

        ram_row = QWidget()
        ram_layout = QHBoxLayout(ram_row)
        ram_layout.setContentsMargins(0, 0, 0, 0)
        ram_layout.addWidget(QLabel("RAM:"))
        self.ram_bar = QProgressBar()
        self.ram_bar.setRange(0, 100)
        self.ram_bar.setFormat("%v%  (%p%)")
        self.ram_bar.setFixedHeight(18)
        ram_layout.addWidget(self.ram_bar)
        self.lbl_ram = QLabel(f"0.00 / {self.dspin_mem.value():.2f} GB")
        ram_layout.addWidget(self.lbl_ram)
        prog_layout.addWidget(ram_row)

        prog_layout.addWidget(QLabel("Fitness history:"))
        self.chart = FitnessChart()
        prog_layout.addWidget(self.chart)

        layout.addWidget(prog)

        # --- Gym render panel (hidden for non-render examples) ---
        self._render_group = QGroupBox("Environment")
        render_layout = QVBoxLayout(self._render_group)
        self._render_widget = GymRenderWidget()
        render_layout.addWidget(self._render_widget)
        self._score_lbl = _label("Score: —", "statValue")
        self._score_lbl.setVisible(False)
        render_layout.addWidget(self._score_lbl)
        self._render_group.setVisible(False)
        layout.addWidget(self._render_group)

        layout.addStretch()

        self.render_frame.connect(self._render_widget.update_frame)

        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # Init with first example
        self._on_example_changed(0)

        # RAM refresh timer
        self._ram_timer = QTimer()
        self._ram_timer.timeout.connect(self._refresh_ram)
        self._ram_timer.start(1000)

    def _current_example(self):
        idx = self.example_combo.currentIndex()
        if 0 <= idx < len(self._examples):
            return self._examples[idx]
        return None

    def _on_example_changed(self, idx: int) -> None:
        ex = self._current_example()
        if ex is None:
            return
        self.spin_inputs.setValue(ex.n_inputs)
        self.spin_outputs.setValue(ex.n_outputs)
        self.spin_nodes.setValue(ex.max_nodes)
        self.spin_conns.setValue(ex.max_connections)
        self.dspin_target.setValue(ex.target_fitness)
        self.desc_label.setText(ex.description)
        self.btn_render.setVisible(ex.supports_render)
        self.btn_run_best.setVisible(ex.supports_render)
        if not ex.supports_render:
            self.btn_render.setChecked(False)
            self._render_group.setVisible(False)
        self._best_genome = None
        self.btn_run_best.setEnabled(False)
        self.example_changed.emit(ex)

    def _on_render_toggled(self, checked: bool) -> None:
        self.btn_render.setText("Render: On" if checked else "Render: Off")
        self._render_group.setVisible(checked)
        if not checked:
            self._render_widget.clear_frame()

    def start_training(self) -> None:
        ex = self._current_example()
        if ex is None:
            return

        try:
            from yane import NeuroEvolution
            self._yane = NeuroEvolution()
            self._yane.configure(
                n_inputs=self.spin_inputs.value(),
                n_outputs=self.spin_outputs.value(),
                max_nodes=self.spin_nodes.value() or None,
                max_connections=self.spin_conns.value() or None,
                n_initial_hidden=ex.n_initial_hidden,
            )
            self._yane.set_population_size(self.spin_pop.value())
            self._yane.set_resource_limits(max_process_gb=self.dspin_mem.value())
            target = self.dspin_target.value()
            if target > -1e9:
                self._yane.set_min_fitness(target)
            render_cb = None
            if ex.supports_render and self.btn_render.isChecked():
                last_render = 0.0
                _emit = self.render_frame.emit
                def render_cb(frame):
                    nonlocal last_render
                    now = _time.perf_counter()
                    if now - last_render < 1 / 30:
                        return
                    last_render = now
                    _emit(frame)
            evaluate_fn = ex.make_eval(render_cb)
        except Exception as e:
            QMessageBox.critical(self, "Setup Error", str(e))
            return

        # Guard: don't start while the previous worker is still winding down
        if self._worker and self._worker.isRunning():
            return

        self._had_error = False
        self._last_ram_color = ""

        # No parent — let Qt manage lifetime via deleteLater to avoid segfault
        # when Python GC and Qt internal refcount race each other.
        self._run_id += 1
        run_id = self._run_id
        worker = TrainingWorker(self._yane, evaluate_fn)
        worker.finished.connect(worker.deleteLater)
        worker.iteration_done.connect(self._on_iteration)
        worker.error_occurred.connect(self._on_error)
        worker.finished.connect(lambda: self._on_finished(run_id))
        if self._episode_runner and self._episode_runner.isRunning():
            self._episode_runner.stop()
            self._episode_runner = None
        worker.start(QThread.Priority.LowPriority)
        self._worker = worker
        self.training_started.emit()
        self.btn_run_best.setEnabled(False)
        self._render_widget.clear_frame()
        self._score_lbl.setVisible(False)

        self.chart.clear()
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.status_lbl.setText("Training…")

    def _toggle_pause(self) -> None:
        if self._worker is None:
            return
        if self._worker._paused:
            self._worker.resume()
            self.btn_pause.setText("⏸  Pause")
            self.status_lbl.setText("Training…")
        else:
            self._worker.pause()
            self.btn_pause.setText("▶  Resume")
            self.status_lbl.setText("Paused")

    def stop_training(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self.btn_pause.setEnabled(False)
            self.btn_pause.setText("⏸  Pause")
            self.btn_stop.setEnabled(False)
            self.status_lbl.setText("Stopping…")
        else:
            self._reset_training_buttons()
            self.status_lbl.setText("Stopped")

    def _on_iteration(self, iteration: int, fitness: float, best_genome, mem: dict) -> None:
        self.lbl_iter.setText(f"Iteration: {iteration}")
        self.lbl_fitness.setText(f"Current: {fitness:.4f}   Best: {best_genome.fitness:.4f}")
        self.chart.add_point(fitness)
        self.genome_updated.emit(best_genome, mem)
        self._update_ram_bar()
        self.btn_run_best.setEnabled(self.btn_run_best.isVisible())

    def _run_best_episode(self) -> None:
        if self._episode_runner and self._episode_runner.isRunning():
            self._episode_runner.stop()
            return

        ex = self._current_example()
        if ex is None or not ex.supports_render:
            return

        if self._yane is not None:
            try:
                genome = self._yane.get_best().copy()
            except RuntimeError:
                return
        elif self._best_genome is not None:
            genome = self._best_genome.copy()
        else:
            return

        self._render_group.setVisible(True)
        self._render_widget.clear_frame()
        self._score_lbl.setText("Score: running…")
        self._score_lbl.setVisible(True)
        self.btn_run_best.setText("■  Stop Run")

        runner = EpisodeRunner(genome, ex)
        runner.finished.connect(runner.deleteLater)
        runner.frame_ready.connect(self._render_widget.update_frame)
        runner.score_updated.connect(lambda s: self._score_lbl.setText(f"Score: {s:.2f}"))
        runner.finished.connect(self._on_episode_finished)
        runner.start()
        self._episode_runner = runner

    def _on_episode_finished(self) -> None:
        self.btn_run_best.setText("▶  Run Best")
        self._episode_runner = None

    def _reset_training_buttons(self) -> None:
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸  Pause")
        self.btn_stop.setEnabled(False)

    def _on_error(self, msg: str) -> None:
        self._had_error = True
        self._reset_training_buttons()
        self.status_lbl.setText("Error")
        QMessageBox.critical(self, "Training Error", msg)

    def _on_finished(self, run_id: int) -> None:
        if run_id != self._run_id:
            return  # stale signal from a previous training run
        self._reset_training_buttons()
        if not self._had_error:
            if self.status_lbl.text() == "Stopping…":
                self.status_lbl.setText("Stopped")
            else:
                self.status_lbl.setText("Finished ✓")
        self._had_error = False
        if self._yane is not None:
            try:
                best = self._yane.get_best().copy()
                mem = self._yane.population_memory_info()
                if self._best_genome is not None:
                    self._best_genome._clear()
                self._best_genome = best.copy()
                self.btn_run_best.setEnabled(True)
                self.genome_updated.emit(best, mem)
            except Exception:
                pass
        self._worker = None
        self._yane = None

    def _refresh_ram(self) -> None:
        if self._worker and self._worker.isRunning():
            self._update_ram_bar()

    def _update_ram_bar(self) -> None:
        from yane.util.resource_guard import ResourceGuard
        used_gb = ResourceGuard.current_process_gb()
        limit = self.dspin_mem.value()
        pct = min(100, int(used_gb / limit * 100))
        self.ram_bar.setValue(pct)
        self.lbl_ram.setText(f"{used_gb:.2f} / {limit:.2f} GB")
        color = "#d20f39" if pct >= 90 else "#e5c890" if pct >= 70 else "#89b4fa"
        if color != self._last_ram_color:
            self._last_ram_color = color
            self.ram_bar.setStyleSheet(
                f"QProgressBar::chunk {{ background-color: {color}; border-radius: 3px; }}"
            )


# ---------------------------------------------------------------------------
# Inspect tab
# ---------------------------------------------------------------------------

class _TestCaseRow:
    """One persistent row in the test-cases table. Created once, updated in place."""

    _MONO = "font-family: monospace; font-size: 12px;"

    def __init__(self, layout, inputs: list[float], expected: list[float]) -> None:
        self._inputs   = inputs
        self._expected = expected

        row = QWidget()
        rlay = QHBoxLayout(row)
        rlay.setContentsMargins(0, 2, 0, 2)

        in_str  = "[" + ", ".join(f"{v:.1f}" for v in inputs) + "]"
        exp_str = "[" + ", ".join(f"{v:.1f}" for v in expected) + "]"

        self._in_lbl  = QLabel(in_str);  self._in_lbl.setMinimumWidth(120); self._in_lbl.setStyleSheet(self._MONO)
        self._exp_lbl = QLabel(exp_str); self._exp_lbl.setMinimumWidth(80);  self._exp_lbl.setStyleSheet(self._MONO)
        self._out_lbl = QLabel("—");     self._out_lbl.setMinimumWidth(90);  self._out_lbl.setStyleSheet(self._MONO)
        self._tick    = QLabel("?");     self._tick.setFixedWidth(30)
        self._tick.setStyleSheet("color: #585b70; font-size: 16px; font-weight: bold;")

        for w in (self._in_lbl, self._exp_lbl, self._out_lbl, self._tick):
            rlay.addWidget(w)

        layout.addWidget(row)

    def update(self, genome) -> None:
        if genome is None:
            self._out_lbl.setText("—")
            self._tick.setText("?")
            self._tick.setStyleSheet("color: #585b70; font-size: 16px; font-weight: bold;")
            return
        try:
            outputs = genome.forward(self._inputs)
        except Exception:
            self._out_lbl.setText("err")
            self._tick.setText("✗")
            self._tick.setStyleSheet("color: #f38ba8; font-size: 16px; font-weight: bold;")
            return

        correct = all(abs(o - e) < 0.2 for o, e in zip(outputs, self._expected))
        out_str = "[" + ", ".join(f"{v:.3f}" for v in outputs) + "]"
        self._out_lbl.setText(out_str)
        tick, color = ("✓", "#a6e3a1") if correct else ("✗", "#f38ba8")
        self._tick.setText(tick)
        self._tick.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold;")


class InspectTab(QWidget):
    """Shows the best genome's outputs for known test cases and manual inputs."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._genome = None
        self._example = None
        self._test_rows: list[_TestCaseRow] = []

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 14, 14, 14)

        # --- Test cases ---
        self._test_group = QGroupBox("Test Cases — best genome output vs. expected")
        self._test_inner = QVBoxLayout(self._test_group)
        self._placeholder = _label("Select an example to see test cases.", "sectionTitle")
        self._test_inner.addWidget(self._placeholder)
        layout.addWidget(self._test_group)

        # --- Manual test ---
        manual = QGroupBox("Manual Forward Pass")
        manual_layout = QVBoxLayout(manual)

        self._input_widgets: list[QDoubleSpinBox] = []
        self._inputs_form = QWidget()
        self._inputs_form_layout = QFormLayout(self._inputs_form)
        manual_layout.addWidget(self._inputs_form)

        self._no_genome_lbl = _label(
            "Start training first — the best genome will be used here.", "sectionTitle")
        self._no_genome_lbl.setWordWrap(True)
        manual_layout.addWidget(self._no_genome_lbl)

        self.btn_run = QPushButton("▶  Run Forward Pass")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._run_manual)
        manual_layout.addWidget(self.btn_run)

        self._output_group = QGroupBox("Outputs")
        self._output_layout = QFormLayout(self._output_group)
        self._output_labels: list[QLabel] = []
        manual_layout.addWidget(self._output_group)

        layout.addWidget(manual)
        layout.addStretch()

    # ------------------------------------------------------------------

    def set_example(self, example) -> None:
        self._example = example
        self._rebuild_test_rows()
        self._rebuild_input_widgets(
            example.n_inputs  if example else 0,
            example.n_outputs if example else 0,
        )

    def reset_genome(self) -> None:
        if self._genome is not None:
            self._genome._clear()
        self._genome = None
        self._no_genome_lbl.setVisible(True)
        self.btn_run.setEnabled(False)
        for row in self._test_rows:
            row.update(None)

    def update_genome(self, genome, mem: dict) -> None:
        self._no_genome_lbl.setVisible(False)
        if self._genome is not None and self._genome is not genome:
            self._genome._clear()
        self._genome = genome
        self.btn_run.setEnabled(bool(self._input_widgets))
        for row in self._test_rows:
            row.update(genome)

    # ------------------------------------------------------------------

    def _rebuild_test_rows(self) -> None:
        """Called only when the example changes (rare). Rebuilds the table once."""
        self._test_rows.clear()

        # Clear the layout (header + rows)
        while self._test_inner.count():
            item = self._test_inner.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        tc = self._example.test_cases if self._example else None
        if not tc:
            self._placeholder = _label(
                "No fixed test cases for this example." if self._example
                else "Select an example to see test cases.", "sectionTitle")
            self._test_inner.addWidget(self._placeholder)
            return

        # Header
        header = QWidget()
        hlay = QHBoxLayout(header)
        hlay.setContentsMargins(0, 0, 0, 0)
        for txt, w in [("Inputs", 120), ("Expected", 80), ("Output", 90), ("", 40)]:
            lbl = _label(txt, "sectionTitle")
            lbl.setMinimumWidth(w)
            hlay.addWidget(lbl)
        self._test_inner.addWidget(header)

        for inputs, expected in tc:
            row = _TestCaseRow(self._test_inner, inputs, expected)
            self._test_rows.append(row)

    def _rebuild_input_widgets(self, n_inputs: int, n_outputs: int) -> None:
        while self._inputs_form_layout.rowCount():
            self._inputs_form_layout.removeRow(0)
        self._input_widgets.clear()

        for i in range(n_inputs):
            spin = QDoubleSpinBox()
            spin.setRange(-1e6, 1e6)
            spin.setDecimals(4)
            spin.setSingleStep(0.1)
            self._inputs_form_layout.addRow(f"Input {i}:", spin)
            self._input_widgets.append(spin)

        while self._output_layout.rowCount():
            self._output_layout.removeRow(0)
        self._output_labels.clear()

        for i in range(n_outputs):
            lbl = _label("—", "statValue")
            self._output_layout.addRow(f"Output {i}:", lbl)
            self._output_labels.append(lbl)

        self.btn_run.setEnabled(n_inputs > 0 and self._genome is not None)

    def _run_manual(self) -> None:
        if self._genome is None:
            return
        inputs = [w.value() for w in self._input_widgets]
        try:
            outputs = self._genome.forward(inputs)
            for i, lbl in enumerate(self._output_labels):
                lbl.setText(f"{outputs[i]:.5f}" if i < len(outputs) else "—")
        except Exception as e:
            for lbl in self._output_labels:
                lbl.setText(f"Error: {e}")


# ---------------------------------------------------------------------------
# Server tab
# ---------------------------------------------------------------------------

class ServerTab(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._server_thread: ServerThread | None = None

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(14, 14, 14, 14)

        # Controls
        ctrl = QGroupBox("API Server")
        ctrl_layout = QFormLayout(ctrl)
        self.spin_port = QSpinBox()
        self.spin_port.setRange(1024, 65535)
        self.spin_port.setValue(8000)
        ctrl_layout.addRow("Port:", self.spin_port)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_server = QPushButton("Start Server")
        self.btn_server.setObjectName("serverBtn")
        self.btn_server.clicked.connect(self._toggle_server)
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color: #585b70; font-size: 18px;")
        self.status_lbl = QLabel("Offline")
        row_layout.addWidget(self.btn_server)
        row_layout.addWidget(self.status_dot)
        row_layout.addWidget(self.status_lbl)
        row_layout.addStretch()
        ctrl_layout.addRow(row)
        layout.addWidget(ctrl)

        # Endpoints
        ep = QGroupBox("Endpoints")
        ep_layout = QVBoxLayout(ep)
        endpoints = [
            ("POST", "/configure",           "Initialise with n_inputs / n_outputs"),
            ("POST", "/population/next",      "Select next genome for evaluation"),
            ("POST", "/population/fitness",   "Submit fitness for current genome"),
            ("GET",  "/population/status",    "Population size, best fitness"),
            ("GET",  "/population/best",      "Best genome info"),
            ("POST", "/network/inputs",       "Set input values"),
            ("POST", "/network/tick",         "Execute one tick"),
            ("GET",  "/network/outputs",      "Read current outputs"),
            ("POST", "/network/forward",      "Full forward pass"),
            ("POST", "/network/reset",        "Reset network state"),
        ]
        for method, path, desc in endpoints:
            color = "#a6e3a1" if method == "GET" else "#89b4fa"
            lbl = QLabel(f'<span style="color:{color};font-weight:bold;">{method}</span>'
                         f'&nbsp;&nbsp;<tt>{path}</tt>&nbsp;&nbsp;'
                         f'<span style="color:#6c7086;">{desc}</span>')
            lbl.setTextFormat(Qt.TextFormat.RichText)
            ep_layout.addWidget(lbl)
        layout.addWidget(ep)

        layout.addStretch()

    def _toggle_server(self) -> None:
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.stop()
            self._server_thread = None
            self.btn_server.setText("Start Server")
            self.status_dot.setStyleSheet("color: #585b70; font-size: 18px;")
            self.status_lbl.setText("Offline")
        else:
            port = self.spin_port.value()
            self._server_thread = ServerThread(port=port)
            self._server_thread.start()
            self.btn_server.setText("Stop Server")
            self.status_dot.setStyleSheet("color: #a6e3a1; font-size: 18px;")
            self.status_lbl.setText(f"Running on http://127.0.0.1:{port}")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YANE — Yet Another Neuro Evolution")
        self.resize(1000, 700)
        self.setStyleSheet(_QSS)

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

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        self._training_tab = TrainingTab()
        self._inspect_tab  = InspectTab()
        self._server_tab   = ServerTab()
        tabs.addTab(self._training_tab, "  Training  ")
        tabs.addTab(self._inspect_tab,  "  Inspect  ")
        tabs.addTab(self._server_tab,   "  API Server  ")

        self._training_tab.genome_updated.connect(self._left.update_genome)
        self._training_tab.genome_updated.connect(self._inspect_tab.update_genome)
        self._training_tab.example_changed.connect(self._inspect_tab.set_example)
        self._training_tab.training_started.connect(self._inspect_tab.reset_genome)
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

    def closeEvent(self, event) -> None:
        self._training_tab._ram_timer.stop()
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
