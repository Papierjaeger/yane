"""Training tab: start/stop/configure training, live fitness chart, export."""
from __future__ import annotations
import time as _time
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QPushButton, QFrame, QLabel, QSizePolicy, QFormLayout,
    QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QMessageBox, QProgressBar, QTabWidget, QInputDialog, QLineEdit,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QImage, QPixmap

from yane.gui.canvas import FitnessChart, SpeciesChart
from yane.gui.worker import TrainingWorker, EpisodeRunner, AutoSetupWorker
from yane.gui.examples import load_examples
from yane.gui._helpers import _label, _divider, CollapsibleGroup
from yane.gui.remote_config import RemoteEvaluationConfig
from yane.gui.research_features import ResearchFeatureConfig
from yane.gui.training_sections import inline_row
from yane.util.presets import list_presets, load_preset, save_preset

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
    genome_updated = Signal(object, dict, bool)  # genome, mem, do_heavy → LeftPanel + InspectTab
    example_changed = Signal(object)        # → InspectTab.set_example
    training_started = Signal()             # → InspectTab.reset_genome
    ne_ready = Signal(object)               # NeuroEvolution instance → aux_tabs
    render_frame = Signal(object)           # numpy array, emitted from worker thread

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._examples = load_examples()
        self._worker: TrainingWorker | None = None
        self._auto_worker: AutoSetupWorker | None = None
        self._auto_profile_info: dict | None = None
        self._episode_runner: EpisodeRunner | None = None
        self._yane = None
        self._best_genome = None
        self._had_error = False
        self._last_ram_color = ""
        self._run_id = 0
        self._start_time: float = 0.0
        self._last_heavy_update: float = 0.0  # throttle for slow widgets
        self._auto_pop_size: int = 100  # set by AutoSetupWorker after profiling

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
        self._combo_index_map: dict[int, object] = {}  # combo idx → ExampleConfig
        self._build_example_combo()
        self.example_combo.currentIndexChanged.connect(self._on_example_changed)
        cfg_form.addRow("Example:", self.example_combo)

        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Custom", None)
        self._preset_by_index: dict[int, object] = {}
        for preset in list_presets():
            self._preset_by_index[self.preset_combo.count()] = preset
            self.preset_combo.addItem(preset.name, preset)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.btn_save_preset = QPushButton("Save")
        self.btn_save_preset.setToolTip("Save the current GUI settings as a reusable preset.")
        self.btn_save_preset.clicked.connect(self._save_current_preset)
        cfg_form.addRow(
            "Preset:",
            inline_row(self.preset_combo, self.btn_save_preset, stretch_first=True),
        )

        self.desc_label = _label("", "sectionTitle")
        self.desc_label.setWordWrap(True)
        cfg_form.addRow(self.desc_label)

        self.spin_inputs  = QSpinBox(); self.spin_inputs.setRange(1, 1024)
        self.spin_outputs = QSpinBox(); self.spin_outputs.setRange(1, 256)
        self.dspin_mem    = QDoubleSpinBox(); self.dspin_mem.setRange(0.1, 32.0); self.dspin_mem.setSingleStep(0.5); self.dspin_mem.setValue(2.0); self.dspin_mem.setSuffix(" GB")
        self.dspin_target = QDoubleSpinBox(); self.dspin_target.setRange(-1e9, 1e9); self.dspin_target.setSingleStep(0.1); self.dspin_target.setDecimals(4); self.dspin_target.setSpecialValueText("—")

        self.spin_inputs.setToolTip(
            "Anzahl der Input-Neuronen.\n"
            "Muss der Dimension des Beobachtungsvektors (state/observation) entsprechen.\n"
            "Beispiel: CartPole hat 4 Werte → Inputs = 4")
        self.spin_outputs.setToolTip(
            "Anzahl der Output-Neuronen.\n"
            "Muss der Anzahl der Aktionen/Ausgaben entsprechen.\n"
            "Beispiel: CartPole hat 2 Aktionen (links/rechts) → Outputs = 2")
        self.dspin_mem.setToolTip(
            "Maximaler RAM-Verbrauch dieses Prozesses in GB.\n"
            "Bei Überschreitung wird die Population auf die Hälfte geschrumpft\n"
            "(schlechteste Genomes werden entfernt).")
        self.dspin_target.setToolTip(
            "Ziel-Fitness: Training stoppt automatisch wenn ein Genome diesen Wert erreicht.\n"
            "— = kein Zielwert (Training läuft bis du Stop drückst).\n"
            "Beispiel: CartPole gilt als gelöst ab Fitness 500.")

        self.chk_remote_eval = QCheckBox("aktiv")
        self.chk_remote_eval.setChecked(False)
        self.chk_remote_eval.setToolTip(
            "Remote Evaluation aktivieren.\n"
            "Die GUI sendet Genome an RemoteWorkerServer-Instanzen und übernimmt\n"
            "die zurückgelieferte Fitness in die lokale Population.")
        self.edit_remote_urls = QLineEdit()
        self.edit_remote_urls.setPlaceholderText("http://localhost:8700, http://worker:8700")
        self.edit_remote_urls.setToolTip("Kommagetrennte Remote-Worker-URLs.")
        self.edit_remote_token = QLineEdit()
        self.edit_remote_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.edit_remote_token.setPlaceholderText("token")
        self.edit_remote_token.setToolTip("Shared Secret für Authorization: Bearer <token>.")
        self.dspin_remote_timeout = QDoubleSpinBox()
        self.dspin_remote_timeout.setRange(1.0, 3600.0)
        self.dspin_remote_timeout.setSingleStep(1.0)
        self.dspin_remote_timeout.setDecimals(1)
        self.dspin_remote_timeout.setValue(30.0)
        self.dspin_remote_timeout.setSuffix(" s")
        self.spin_remote_retries = QSpinBox()
        self.spin_remote_retries.setRange(0, 10)
        self.spin_remote_retries.setValue(2)
        self.spin_remote_batch = QSpinBox()
        self.spin_remote_batch.setRange(0, 10000)
        self.spin_remote_batch.setValue(0)
        self.spin_remote_batch.setSpecialValueText("Auto")
        self.spin_remote_batch.setToolTip(
            "Remote-Batchgröße. Auto nutzt 2 Jobs pro Worker-URL."
        )

        self.chk_normalize = QCheckBox("aktiv")
        self.chk_normalize.setChecked(True)
        self.chk_normalize.setVisible(False)   # shown only for examples that support it
        self.chk_normalize.setToolTip(
            "Inputs und Outputs vor dem Training auf [0, 1] normalisieren.\n"
            "Empfohlen: Normalisierung verbessert die Fitness-Landschaft erheblich\n"
            "und verhindert dass das Netz gegen die rohe Skala kämpft statt gegen\n"
            "die eigentliche Aufgabe.\n\n"
            "Deaktivieren um das Verhalten ohne Normalisierung zu vergleichen.")

        self.chk_memory = QCheckBox("aktiv")
        self.chk_memory.setChecked(False)
        self.chk_memory.setToolTip(
            "Wenn aktiviert, dürfen Neuronen ihre Werte über mehrere Forward-Passes\n"
            "behalten (persistent value). Die Evolution entscheidet pro Neuron, ob\n"
            "es als Gedächtniszelle agiert.\n\n"
            "Default: aus für Dataset-Aufgaben (XOR, Multiplication, Regression),\n"
            "an für sequenzielle Aufgaben (Pi, Gym-Umgebungen).\n\n"
            "Wenn deaktiviert, können keine Neuronen ihre Werte behalten — das\n"
            "Netz ist rein feedforward auf Schritt-Ebene.")

        self.chk_curriculum = QCheckBox("aktiv")
        self.chk_curriculum.setChecked(False)
        self.chk_curriculum.setVisible(False)   # shown only for examples that support it

        # Evaluator component filter — shown only for examples with named components
        self._eval_components_widget = QWidget()
        self._eval_components_layout = QHBoxLayout(self._eval_components_widget)
        self._eval_components_layout.setContentsMargins(0, 0, 0, 0)
        self._eval_component_checkboxes: list[tuple[str, QCheckBox]] = []
        self._eval_components_widget.setVisible(False)
        self.chk_curriculum.setToolTip(
            "Curriculum Learning: Training in aufsteigend schwieriger werdenden Stufen.\n\n"
            "Pi-Ziffern: Stufe 1 (3 Ziffern) → Stufe 2 (6 Ziffern) → Stufe 3 (10 Ziffern).\n"
            "Die Population wird bei jedem Stufenwechsel behalten, aber neu bewertet.\n\n"
            "Erzwingt sequentielle Ausführung (kein Multiprocessing).\n"
            "Curriculum-Stufe und Fortschritt sind im linken Panel sichtbar.")

        cfg_form.addRow("Inputs:",         self.spin_inputs)
        cfg_form.addRow("Outputs:",        self.spin_outputs)
        cfg_form.addRow("Normalization:", self.chk_normalize)
        cfg_form.addRow("Memory:",        self.chk_memory)
        cfg_form.addRow("Curriculum:",    self.chk_curriculum)
        self._eval_components_label = _label("Eval components:")
        self._eval_components_label.setVisible(False)
        cfg_form.addRow(self._eval_components_label, self._eval_components_widget)
        cfg_form.addRow("Memory limit:",   self.dspin_mem)
        cfg_form.addRow("Target fitness:", self.dspin_target)

        # ── Seed / Stopping / Logging ────────────────────────────────────────
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(-1, 2**31 - 1)
        self.spin_seed.setValue(0)
        self.spin_seed.setSpecialValueText("—")
        self.spin_seed.setToolTip(
            "Zufalls-Seed für reproduzierbare Läufe.\n"
            "— = kein fester Seed (nicht deterministisch).\n"
            "Gleicher Seed = gleiche Startpopulation und Operator-Reihenfolge.")
        cfg_form.addRow("Seed:", self.spin_seed)

        self.spin_max_iter = QSpinBox()
        self.spin_max_iter.setRange(0, 10_000_000)
        self.spin_max_iter.setValue(0)
        self.spin_max_iter.setSpecialValueText("—")
        self.spin_max_iter.setSingleStep(1000)
        self.spin_max_iter.setToolTip(
            "Maximale Iterationen (Generationen).\n"
            "— = unbegrenzt (läuft bis Stop oder Target-Fitness).")
        cfg_form.addRow("Max iterations:", self.spin_max_iter)

        # --- Advanced settings group (collapsed by default) ---
        advance_grp = CollapsibleGroup("Advanced", collapsed=True)

        # ── Checkpoint policy ───────────────────────────────────────────────
        self.chk_auto_checkpoint = QCheckBox("aktiv")
        self.chk_auto_checkpoint.setChecked(False)
        self.chk_auto_checkpoint.setToolTip(
            "Rollierende Auto-Checkpoints während des Trainings.\n"
            "Speichert regelmäßig den Populationszustand.")
        self.spin_ckpt_interval = QSpinBox()
        self.spin_ckpt_interval.setRange(10, 10_000)
        self.spin_ckpt_interval.setValue(100)
        self.spin_ckpt_interval.setEnabled(False)
        self.spin_ckpt_max_keep = QSpinBox()
        self.spin_ckpt_max_keep.setRange(1, 100)
        self.spin_ckpt_max_keep.setValue(5)
        self.spin_ckpt_max_keep.setEnabled(False)
        self.chk_auto_checkpoint.toggled.connect(
            lambda on: (
                self.spin_ckpt_interval.setEnabled(on),
                self.spin_ckpt_max_keep.setEnabled(on),
            )
        )
        advance_grp.addRow(
            "Auto checkpoint:",
            inline_row(
                self.chk_auto_checkpoint, QLabel("every"), self.spin_ckpt_interval,
                QLabel("keep"), self.spin_ckpt_max_keep,
            ),
        )

        advance_grp.addRow(
            "Remote eval:",
            inline_row(self.chk_remote_eval, self.edit_remote_urls, stretch_last=True),
        )
        advance_grp.addRow("Remote token:", self.edit_remote_token)
        advance_grp.addRow(
            "Remote timeout/retry/batch:",
            inline_row(
                self.dspin_remote_timeout,
                QLabel("retries"),
                self.spin_remote_retries,
                QLabel("batch"),
                self.spin_remote_batch,
            ),
        )
        layout.addWidget(cfg)
        layout.addWidget(advance_grp)

        # --- Controls ---
        ctrl = QWidget()
        ctrl_row = QHBoxLayout(ctrl)
        ctrl_row.setContentsMargins(0, 0, 0, 0)
        self.btn_auto_train = QPushButton("▶  Start"); self.btn_auto_train.setObjectName("startBtn")
        self.btn_pause = QPushButton("⏸  Pause"); self.btn_pause.setObjectName("pauseBtn")
        self.btn_stop  = QPushButton("■  Stop");  self.btn_stop.setObjectName("stopBtn")
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.status_lbl = QLabel("Idle")
        self.btn_auto_train.clicked.connect(self.start_training)
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
        ctrl_row.addWidget(self.btn_auto_train)
        ctrl_row.addWidget(self.btn_pause)
        ctrl_row.addWidget(self.btn_stop)
        ctrl_row.addWidget(self.btn_render)
        ctrl_row.addWidget(self.btn_run_best)
        ctrl_row.addWidget(self.status_lbl)
        ctrl_row.addStretch()
        layout.addWidget(ctrl)

        # --- Checkpoint row ---
        chk_row_w = QWidget()
        chk_row = QHBoxLayout(chk_row_w)
        chk_row.setContentsMargins(0, 0, 0, 0)
        chk_row.setSpacing(4)
        self.btn_save_ckpt = QPushButton("💾  Save checkpoint")
        self.btn_load_ckpt = QPushButton("📂  Load checkpoint")
        self.btn_save_ckpt.setEnabled(False)
        self.btn_save_ckpt.setToolTip(
            "Save the current population + InnovationTracker to a .pkl file\n"
            "so training can be resumed later."
        )
        self.btn_load_ckpt.setToolTip(
            "Load a checkpoint file to resume a previous training run.\n"
            "The current population will be replaced."
        )
        self.btn_save_ckpt.clicked.connect(self._save_checkpoint)
        self.btn_load_ckpt.clicked.connect(self._load_checkpoint)
        chk_row.addWidget(self.btn_save_ckpt)
        chk_row.addWidget(self.btn_load_ckpt)
        chk_row.addStretch()
        layout.addWidget(chk_row_w)

        # --- Progress ---
        prog = QGroupBox("Progress")
        prog_layout = QVBoxLayout(prog)

        prog_form = QWidget()
        prog_form_layout = QFormLayout(prog_form)
        prog_form_layout.setSpacing(3)
        prog_form_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_iter    = _label("—", "statValue")
        self.lbl_evals   = _label("—", "statValue")
        self.lbl_fitness = _label("—", "statValue")
        self.lbl_speed   = _label("—", "statValue")
        prog_form_layout.addRow("Generation:",     self.lbl_iter)
        prog_form_layout.addRow("Evaluations:",    self.lbl_evals)
        prog_form_layout.addRow("Current / Best:", self.lbl_fitness)
        prog_form_layout.addRow("Speed:",          self.lbl_speed)
        prog_layout.addWidget(prog_form)

        ram_row = QWidget()
        ram_layout = QHBoxLayout(ram_row)
        ram_layout.setContentsMargins(0, 0, 0, 0)
        ram_layout.addWidget(QLabel("RAM:"))
        self.ram_bar = QProgressBar()
        self.ram_bar.setRange(0, 100)
        self.ram_bar.setFormat("%p%")
        self.ram_bar.setFixedHeight(18)
        ram_layout.addWidget(self.ram_bar)
        self.lbl_ram = QLabel(f"0.00 / {self.dspin_mem.value():.2f} GB")
        ram_layout.addWidget(self.lbl_ram)
        prog_layout.addWidget(ram_row)

        prog_layout.addWidget(QLabel("Fitness history:"))
        self.chart = FitnessChart()
        prog_layout.addWidget(self.chart)

        prog_layout.addWidget(QLabel("Species count:"))
        self.species_chart = SpeciesChart()
        prog_layout.addWidget(self.species_chart)

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


    # Category display order
    _CATEGORY_ORDER = ["Dataset", "Toy Text", "Classic Control", "Box2D", "MuJoCo", "Pixel", "Atari", "Plugins", "Sonstiges"]

    def _build_example_combo(self) -> None:
        """Populate the combo box with group headers and indented example names."""
        from PySide6.QtGui import QStandardItemModel, QStandardItem, QFont, QColor
        model = QStandardItemModel()
        self._combo_index_map.clear()

        # Group examples by category, preserving category order
        groups: dict[str, list] = {}
        for ex in self._examples:
            groups.setdefault(ex.category, []).append(ex)

        header_font = QFont()
        header_font.setBold(True)
        header_color = QColor("#a6adc8")

        combo_idx = 0
        first_example_combo_idx = None

        ordered_categories = list(self._CATEGORY_ORDER)
        ordered_categories.extend(cat for cat in groups if cat not in ordered_categories)
        for cat in ordered_categories:
            if cat not in groups:
                continue
            # Group header — not selectable
            header = QStandardItem(f"  {cat}")
            header.setEnabled(False)
            header.setFont(header_font)
            header.setForeground(header_color)
            model.appendRow(header)
            combo_idx += 1

            for ex in groups[cat]:
                item = QStandardItem(f"    {ex.name}")
                model.appendRow(item)
                self._combo_index_map[combo_idx] = ex
                if first_example_combo_idx is None:
                    first_example_combo_idx = combo_idx
                combo_idx += 1

        self.example_combo.setModel(model)
        if first_example_combo_idx is not None:
            self.example_combo.setCurrentIndex(first_example_combo_idx)

    def _current_example(self):
        return self._combo_index_map.get(self.example_combo.currentIndex())

    def _on_example_changed(self, idx: int) -> None:
        ex = self._current_example()
        if ex is None:
            # User clicked a group header — jump to the next real example
            next_idx = idx + 1
            while next_idx not in self._combo_index_map and next_idx < self.example_combo.count():
                next_idx += 1
            if next_idx in self._combo_index_map:
                self.example_combo.setCurrentIndex(next_idx)
            return
        self.spin_inputs.setValue(ex.n_inputs)
        self.spin_outputs.setValue(ex.n_outputs)
        self.dspin_target.setValue(ex.target_fitness)
        self.desc_label.setText(ex.description)
        # Default memory based on example type; user can override before training.
        self.chk_memory.setChecked(ex.stateful)
        self.btn_render.setVisible(ex.supports_render)
        self.btn_run_best.setVisible(ex.supports_render)
        if not ex.supports_render:
            self.btn_render.setChecked(False)
        self.chk_normalize.setVisible(ex.supports_normalization)
        if ex.supports_normalization:
            self.chk_normalize.setChecked(True)   # default on when switching examples
        self.chk_curriculum.setVisible(ex.make_curriculum is not None)
        if ex.make_curriculum is not None:
            self.chk_curriculum.setChecked(ex.default_curriculum)
        else:
            self.chk_curriculum.setChecked(False)
            self._render_group.setVisible(False)
        # Rebuild evaluator component checkboxes
        self._rebuild_eval_component_checkboxes(ex)
        self._apply_config_dict(ex.default_config)
        self._best_genome = None
        self.btn_run_best.setEnabled(False)
        self.example_changed.emit(ex)

    def _rebuild_eval_component_checkboxes(self, ex) -> None:
        """Show per-component checkboxes when the example supports named components."""
        # Remove old checkboxes
        for _, chk in self._eval_component_checkboxes:
            self._eval_components_layout.removeWidget(chk)
            chk.deleteLater()
        self._eval_component_checkboxes.clear()
        components = getattr(ex, "evaluator_components", None)
        visible = bool(components)
        self._eval_components_widget.setVisible(visible)
        self._eval_components_label.setVisible(visible)
        if components:
            for name in components:
                chk = QCheckBox(name.replace("_", " "))
                chk.setChecked(True)
                chk.setToolTip(
                    f"Komponente '{name}' in die Fitness-Berechnung einbeziehen.\n"
                    "Deaktivieren für Ablations-Vergleiche."
                )
                self._eval_components_layout.addWidget(chk)
                self._eval_component_checkboxes.append((name, chk))

    def _get_enabled_eval_components(self) -> "frozenset[str] | None":
        """Return enabled component names, or None if all are enabled / no components."""
        if not self._eval_component_checkboxes:
            return None
        enabled = frozenset(
            name for name, chk in self._eval_component_checkboxes if chk.isChecked()
        )
        all_names = frozenset(name for name, _ in self._eval_component_checkboxes)
        if enabled == all_names:
            return None  # all enabled = default behaviour
        return enabled if enabled else None

    def _apply_config_dict(self, cfg: dict) -> None:
        """Apply a GUI config dict to widgets; used by presets and example defaults."""
        if "remote_eval" in cfg:
            self.chk_remote_eval.setChecked(bool(cfg["remote_eval"]))
        if "remote_urls" in cfg:
            self.edit_remote_urls.setText(str(cfg["remote_urls"]))
        if "remote_timeout_s" in cfg:
            self.dspin_remote_timeout.setValue(float(cfg["remote_timeout_s"]))
        if "remote_retries" in cfg:
            self.spin_remote_retries.setValue(int(cfg["remote_retries"]))
        if "remote_batch_size" in cfg:
            self.spin_remote_batch.setValue(int(cfg["remote_batch_size"]))
        if "normalize" in cfg and self.chk_normalize.isVisible():
            self.chk_normalize.setChecked(bool(cfg["normalize"]))
        if "memory" in cfg:
            self.chk_memory.setChecked(bool(cfg["memory"]))
        if "curriculum" in cfg and self.chk_curriculum.isVisible():
            self.chk_curriculum.setChecked(bool(cfg["curriculum"]))

    def _on_preset_changed(self, index: int) -> None:
        preset = self._preset_by_index.get(index)
        if preset is None:
            return
        self._apply_config_dict(preset.config)

    def _current_preset_config(self) -> dict:
        return {
            "remote_eval": self.chk_remote_eval.isChecked(),
            "remote_urls": self.edit_remote_urls.text(),
            "remote_timeout_s": self.dspin_remote_timeout.value(),
            "remote_retries": self.spin_remote_retries.value(),
            "remote_batch_size": self.spin_remote_batch.value(),
            "normalize": self.chk_normalize.isChecked(),
            "memory": self.chk_memory.isChecked(),
            "curriculum": self.chk_curriculum.isChecked(),
        }

    def _save_current_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        if not ok or not name.strip():
            return
        path = save_preset(
            name.strip(),
            self._current_preset_config(),
            "Saved from GUI",
        )
        preset = load_preset(path)
        idx = self.preset_combo.count()
        self._preset_by_index[idx] = preset
        self.preset_combo.addItem(preset.name, preset)
        self.preset_combo.setCurrentIndex(idx)
        self.status_lbl.setText(f"Preset saved: {path.name}")

    def _on_render_toggled(self, checked: bool) -> None:
        self.btn_render.setText("Render: On" if checked else "Render: Off")
        self._render_group.setVisible(checked)
        if not checked:
            self._render_widget.clear_frame()

    def _current_research_feature_config(self) -> ResearchFeatureConfig:
        return ResearchFeatureConfig(
            n_inputs=self.spin_inputs.value(),
            n_outputs=self.spin_outputs.value(),
            max_nodes=None,
            max_connections=None,
            population_size=self._auto_pop_size,
            target_species=5,
            allow_memory=self.chk_memory.isChecked(),
            output_sanitize=self._yane._output_sanitize,
            output_fallback=self._yane._output_fallback,
        )

    def _configure_yane_core(self, ex) -> ResearchFeatureConfig:
        from yane import NeuroEvolution

        self._yane = NeuroEvolution()
        seed = self.spin_seed.value()
        if seed != 0:
            self._yane.set_seed(seed if seed > 0 else None)
        self._yane.configure(
            n_inputs=self.spin_inputs.value(),
            n_outputs=self.spin_outputs.value(),
            n_initial_hidden=ex.n_initial_hidden,
            stateful=self.chk_memory.isChecked(),
        )
        return self._current_research_feature_config()

    def _apply_evolution_options(self, research_cfg: ResearchFeatureConfig) -> None:
        # Checkpoint policy
        if self.chk_auto_checkpoint.isChecked():
            self._yane.set_checkpoint_policy(
                interval=self.spin_ckpt_interval.value(),
                keep_best=True,
                max_keep=self.spin_ckpt_max_keep.value(),
                enabled=True,
            )

        n_iter = self.spin_max_iter.value()
        if n_iter > 0:
            self._yane.set_max_iterations(n_iter)
        self._yane.set_resource_limits(max_process_gb=self.dspin_mem.value())
        target = self.dspin_target.value()
        if target > -1e9:
            self._yane.set_min_fitness(target)

    def _render_callback_for_example(self, ex):
        if ex.supports_render and self.btn_render.isChecked():
            return self.render_frame.emit
        return None

    def _current_remote_config(self) -> RemoteEvaluationConfig:
        if self.chk_remote_eval.isChecked() and self.chk_curriculum.isChecked():
            raise ValueError("Remote Evaluation ist mit Curriculum-Learning aktuell nicht kompatibel.")
        return RemoteEvaluationConfig.from_text(
            enabled=self.chk_remote_eval.isChecked(),
            worker_urls_text=self.edit_remote_urls.text(),
            token=self.edit_remote_token.text(),
            timeout_s=self.dspin_remote_timeout.value(),
            max_retries=self.spin_remote_retries.value(),
            batch_size=self.spin_remote_batch.value(),
        )

    def _make_eval_factory(self, ex):
        import functools
        if ex.supports_normalization and not self.chk_normalize.isChecked():
            make_eval_fn = functools.partial(ex.make_eval, normalize=False)
        else:
            make_eval_fn = ex.make_eval
        enabled = self._get_enabled_eval_components()
        if enabled is not None:
            make_eval_fn = functools.partial(make_eval_fn, enabled_components=enabled)
        return make_eval_fn

    def _configure_quality_diversity(self) -> None:
        pass

    def _configure_curriculum(self, ex) -> None:
        if ex.make_curriculum is None or not self.chk_curriculum.isChecked():
            return
        normalize = self.chk_normalize.isChecked() if ex.supports_normalization else True
        target_fitness = (
            self.dspin_target.value()
            if self.dspin_target.value() > -1e9
            else ex.target_fitness
        )
        self._yane.set_curriculum(
            ex.make_curriculum(normalize=normalize, target_fitness=target_fitness)
        )

    def _setup_gui_run_logging(self, ex) -> None:
        from yane.util.logger import setup_logging as _setup_log, write_json as _wj, log_info as _li
        from yane.benchmarks import wire_db, BENCHMARK_DB_PATH

        wire_db(self._yane, f"gui/{ex.name}", BENCHMARK_DB_PATH)

        log_dir = _setup_log(f"gui/{ex.name}")
        self._yane._log_run_dir = log_dir
        self._yane._log_run_name = ex.name
        if self._yane._run_database is None:
            _wj(log_dir / "config.json", self._yane._config_dict())
        if self.preset_combo.currentIndex() in self._preset_by_index:
            preset = self._preset_by_index[self.preset_combo.currentIndex()]
            _wj(log_dir / "preset.json", preset.to_json())
        _li(
            "GUI training started  example=%s  target=%s",
            ex.name,
            self.dspin_target.value() if self.dspin_target.value() > -1e9 else "none",
        )
        self._log_csv_path = log_dir / "fitness_history.csv"
        self._log_csv_header = "generation,iteration,best_fitness,mean_fitness,median_fitness,iqr_fitness,species_count,stagnation_count,nodes,connections,validation_fitness"
        self._log_csv_interval = max(1, self._auto_pop_size // 10)

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
        elif self._auto_worker and self._auto_worker.isRunning():
            self._auto_worker.terminate()
            self._auto_worker = None
            self._reset_training_buttons()
            self.status_lbl.setText("Stopped")
        else:
            self._reset_training_buttons()
            self.status_lbl.setText("Stopped")

    def start_training(self) -> None:
        """Start zero-config training: profile → KB → MetaOptimizer → FeatureGating → Train."""
        ex = self._current_example()
        if ex is None:
            return
        if (self._worker and self._worker.isRunning()) or \
           (self._auto_worker and self._auto_worker.isRunning()):
            return

        try:
            research_cfg = self._configure_yane_core(ex)
            self._apply_evolution_options(research_cfg)
            make_eval_fn = self._make_eval_factory(ex)
            remote_cfg = self._current_remote_config()
            self._configure_quality_diversity()
            self._configure_curriculum(ex)
            self._setup_gui_run_logging(ex)
        except Exception as exc:
            QMessageBox.critical(self, "Setup Error", str(exc))
            return

        self._run_id += 1
        run_id = self._run_id
        self._had_error = False
        self._last_ram_color = ""
        self._auto_profile_info = None

        self.btn_auto_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_lbl.setText("Profiling problem…")
        self.chart.clear()
        self.species_chart.clear()
        self._start_time = _time.perf_counter()
        self._last_heavy_update = 0.0
        self._render_widget.clear_frame()
        self._score_lbl.setVisible(False)

        self.training_started.emit()
        if self._yane is not None:
            self.ne_ready.emit(self._yane)
        self.btn_save_ckpt.setEnabled(True)
        self.btn_run_best.setEnabled(False)

        setup_worker = AutoSetupWorker(self._yane, make_eval_fn)
        setup_worker.finished.connect(setup_worker.deleteLater)
        setup_worker.status_message.connect(self.status_lbl.setText)
        setup_worker.error_occurred.connect(self._on_auto_error)
        setup_worker.setup_done.connect(
            lambda info: self._on_auto_setup_done(info, make_eval_fn, remote_cfg, run_id)
        )
        self._auto_worker = setup_worker
        setup_worker.start(QThread.Priority.LowPriority)

    def _on_auto_setup_done(self, info: dict, make_eval_fn, remote_cfg, run_id: int) -> None:
        self._auto_worker = None
        self._auto_profile_info = info
        self._auto_pop_size = info.get("pop_size", self._auto_pop_size)

        task = info.get("task_type", "?")
        diff = info.get("difficulty", "?")
        pop  = info.get("pop_size", "?")
        kb   = f"KB conf={info.get('kb_conf', 0):.2f}" if info.get("kb_used") else "cold-start"
        self.status_lbl.setText(f"Auto [{task} diff={diff} pop={pop} {kb}] Training…")

        ex = self._current_example()
        render_cb = self._render_callback_for_example(ex) if ex else None

        worker = TrainingWorker(self._yane, make_eval_fn,
                                render_cb=render_cb, remote_config=remote_cfg)
        worker.finished.connect(worker.deleteLater)
        worker.iteration_done.connect(self._on_iteration)
        worker.error_occurred.connect(self._on_error)
        worker.info_message.connect(self.status_lbl.setText)
        worker.workers_resolved.connect(self._on_workers_resolved)
        worker.finished.connect(lambda: self._on_finished(run_id))
        worker.start(QThread.Priority.LowPriority)
        self._worker = worker
        self.btn_pause.setEnabled(True)

    def _on_auto_error(self, msg: str) -> None:
        self._auto_worker = None
        self._auto_profile_info = None
        self._reset_training_buttons()
        self.status_lbl.setText("Error")
        QMessageBox.critical(self, "Auto-Train Setup Error", msg)

    def _on_workers_resolved(self, n: int) -> None:
        pass

    def _on_iteration(self, iteration: int, fitness: float, best_genome, mem: dict) -> None:
        elapsed = _time.perf_counter() - self._start_time
        iter_s = iteration / elapsed if elapsed > 0 else 0.0
        mins, secs = divmod(int(elapsed), 60)
        elapsed_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
        generation = mem.get("generation", iteration // max(1, self._auto_pop_size))
        self.lbl_iter.setText(str(generation))
        self.lbl_evals.setText(str(iteration))
        self.lbl_fitness.setText(f"{fitness:.4f}   /   {best_genome.fitness:.4f}")
        self.lbl_speed.setText(f"{iter_s:.1f} iter/s   {elapsed_str}")
        self.chart.add_point(fitness)

        # --- Periodic CSV logging -------------------------------------------
        csv_path = getattr(self, '_log_csv_path', None)
        if csv_path is not None and iteration % self._log_csv_interval == 0:
            try:
                from yane.util.logger import write_csv
                write_csv(csv_path, self._log_csv_header,
                    f"{generation},"
                    f"{iteration},"
                    f"{mem.get('max_fitness', 0)},"
                    f"{mem.get('avg_fitness', 0)},"
                    f"{mem.get('median_fitness', 0)},"
                    f"{mem.get('fitness_iqr', 0)},"
                    f"{mem.get('species_count', 0)},"
                    f"{mem.get('stagnation_count', 0)},"
                    f"{mem.get('largest_genome_nodes', 0)},"
                    f"{mem.get('largest_genome_connections', 0)},"
                    f"{mem.get('validation_fitness', '')}")
            except Exception as _e:
                from yane.util.logger import log_warning
                log_warning("CSV write failed at iteration %d: %s", iteration, _e)

        # Track iteration count for summary.json.
        self._log_iter_count = iteration

        # --- Crash-safe state snapshot + heartbeat (every 100 iters) -------
        # A segfault kills the process instantly — no atexit, no except.
        # By writing state *before* the crash happens we can at least see
        # the last known iteration, fitness, species, and topology.
        if iteration % 100 == 0:
            try:
                import json
                log_dir = getattr(self, '_log_run_dir', None)
                if log_dir is not None:
                    snap = {
                        "iteration": iteration,
                        "best_fitness": mem.get("max_fitness"),
                        "avg_fitness": mem.get("avg_fitness"),
                        "fitness_iqr": mem.get("fitness_iqr"),
                        "species_count": mem.get("species_count"),
                        "stagnation_count": mem.get("stagnation_count"),
                        "nodes": mem.get("largest_genome_nodes"),
                        "connections": mem.get("largest_genome_connections"),
                        "lamarck_n_applied": mem.get("lamarck_n_applied"),
                        "lamarck_mode": mem.get("lamarck_mode"),
                        "n_invalid_fitness": mem.get("n_invalid_fitness"),
                    }
                    snap_path = log_dir / "_crash_state.json"
                    tmp = snap_path.with_suffix(".tmp")
                    tmp.write_text(json.dumps(snap), encoding="utf-8")
                    tmp.replace(snap_path)

                from yane.util.logger import log_info
                log_info("iter=%d  best=%.4f  avg=%.2f  species=%d  stagn=%d  nodes=%d  conns=%d",
                         iteration,
                         mem.get("max_fitness", 0.0),
                         mem.get("avg_fitness", 0.0),
                         mem.get("species_count", 0),
                         mem.get("stagnation_count", 0),
                         mem.get("largest_genome_nodes", 0),
                         mem.get("largest_genome_connections", 0))
            except Exception as _e:
                from yane.util.logger import log_warning
                log_warning("Crash-state snapshot failed at iteration %d: %s", iteration, _e)

        # Heavy widgets (species chart, weight histogram, network canvas) are
        # throttled to 1 Hz — they involve non-trivial paint work and don't
        # need to update as often as the fitness chart or labels.
        now = _time.perf_counter()
        do_heavy = now - self._last_heavy_update >= 1.0
        if do_heavy:
            self._last_heavy_update = now
            self.species_chart.add_point(mem.get("species_count", 0))

        self.genome_updated.emit(best_genome, mem, do_heavy)
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

    def _save_checkpoint(self) -> None:
        if self._yane is None:
            return
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Checkpoint", "checkpoint.pkl",
            "Checkpoint files (*.pkl);;All files (*)"
        )
        if not path:
            return
        try:
            self._yane.save_checkpoint(path)
            self.status_lbl.setText(f"Saved → {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Save Checkpoint Error", str(e))

    def _load_checkpoint(self) -> None:
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(
                self, "Load Checkpoint",
                "Stop the current training run before loading a checkpoint."
            )
            return
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Checkpoint", "",
            "Checkpoint files (*.pkl);;All files (*)"
        )
        if not path:
            return
        try:
            from yane import NeuroEvolution
            self._yane = NeuroEvolution()
            self._yane.load_checkpoint(path)
            self.btn_save_ckpt.setEnabled(True)
            self.status_lbl.setText(f"Loaded ← {Path(path).name}")
            self._show_checkpoint_metadata(path)
        except Exception as e:
            self._yane = None
            QMessageBox.critical(self, "Load Checkpoint Error", str(e))

    def _show_checkpoint_metadata(self, path: str) -> None:
        """Read the .json sidecar and show metadata; warn if reattach is required."""
        import json
        meta_path = Path(path).with_suffix(".pkl.json")
        if not meta_path.exists():
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return

        requires = meta.get("requires_reattach", [])
        if requires:
            QMessageBox.warning(
                self, "Checkpoint: Callbacks fehlen",
                "Folgende Callbacks muessen nach dem Laden manuell neu verbunden werden:\n\n"
                + "\n".join(f"  • {r}" for r in requires)
                + "\n\nOhne diese Callbacks ist die betroffene Funktionalitaet deaktiviert.",
            )

        cfg = meta.get("config", {})
        pop_size = meta.get("population_size")
        version = meta.get("version", "?")
        created = meta.get("created_at", "?")
        config_hash = meta.get("config_hash", "?")
        lines = [
            f"Version:   {version}",
            f"Erstellt:  {created}",
            f"Config:    {config_hash}",
            f"Pop-Size:  {pop_size if pop_size is not None else '?'}",
        ]
        if cfg:
            n_in = cfg.get("n_inputs", "?")
            n_out = cfg.get("n_outputs", "?")
            lines.append(f"Inputs:    {n_in}")
            lines.append(f"Outputs:   {n_out}")
        compat = meta.get("compatibility") or {}
        if compat:
            lines.append(f"Kompat.:   {compat.get('level', '?')}")
            for item in compat.get("diff", [])[:5]:
                lines.append(
                    f"  {item.get('path')}: "
                    f"{item.get('stored')} -> {item.get('current')}"
                )
        QMessageBox.information(
            self, "Checkpoint-Metadaten", "\n".join(lines)
        )

    def _reset_training_buttons(self) -> None:
        self.btn_auto_train.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸  Pause")
        self.btn_stop.setEnabled(False)

    def _on_error(self, msg: str) -> None:
        self._had_error = True
        self._reset_training_buttons()
        self._update_ram_bar()
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
                mem  = self._yane.population_memory_info()

                # --- Save best genome and summary to log directory ---------
                log_dir = getattr(self._yane, '_log_run_dir', None)
                if log_dir is not None:
                    import pickle
                    try:
                        (log_dir / "best_genome.pkl").write_bytes(pickle.dumps(best))
                        from yane.util.logger import write_json, log_info
                        total_iters = getattr(self, '_log_iter_count', 0)
                        write_json(log_dir / "summary.json", {
                            "run_name": getattr(self._yane, '_log_run_name', "gui"),
                            "stop_reason": "manual",
                            "iterations": total_iters,
                            "best_fitness": best.fitness,
                            "best_nodes": len(best.nodes),
                            "best_connections": best.connection_count,
                            "final_species_count": mem.get("species_count", 0),
                            "final_stagnation": mem.get("stagnation_count", 0),
                            "lamarck_n_applied":       mem.get("lamarck_n_applied", 0),
                            "lamarck_n_steps_total":   mem.get("lamarck_n_steps_total", 0),
                            "lamarck_n_blocked_top_k": mem.get("lamarck_n_blocked_top_k", 0),
                            "n_invalid_fitness": mem.get("n_invalid_fitness", 0),
                            "n_clipped_fitness": mem.get("n_clipped_fitness", 0),
                        })
                        log_info(
                            "GUI training finished  best_fitness=%.6f  nodes=%d  connections=%d  "
                            "iterations=%d  lamarck_applied=%d  lamarck_blocked_top_k=%d",
                            best.fitness, len(best.nodes), best.connection_count, total_iters,
                            mem.get("lamarck_n_applied", 0),
                            mem.get("lamarck_n_blocked_top_k", 0),
                        )
                    except Exception as _e:
                        from yane.util.logger import log_warning
                        log_warning("Failed to write end-of-run artefacts: %s", _e)

                if self._best_genome is not None:
                    self._best_genome._clear()
                self._best_genome = best
                self.btn_run_best.setEnabled(True)
                self.genome_updated.emit(best, mem, True)  # full update on finish
            except Exception:
                pass
        # Auto-train report — build before _yane is cleared
        if self._auto_profile_info is not None and not self._had_error and self._yane is not None:
            try:
                from yane.evolution.auto_train import build_report
                pop_size = self._auto_profile_info.get("pop_size", 0)
                best_fit = self._yane.get_best().fitness if self._yane._population else -float("inf")
                total_gen = getattr(self._yane, "_n_evaluations_done", 0) // max(1, pop_size)
                report = build_report(
                    profile=self._auto_profile_info.get("profile"),
                    problem_name=None,
                    kb_entries=self._auto_profile_info.get("kb_entries", 0),
                    kb_conf=self._auto_profile_info.get("kb_conf", 0.0),
                    applied_params=self._auto_profile_info.get("applied_params", {}),
                    cold_start=self._auto_profile_info.get("cold_start", True),
                    meta_diag=self._yane.get_meta_optimizer_diagnostics(),
                    feat_diag=self._yane.get_feature_gating_diagnostics(),
                    active_features=(self._yane._feature_gate.get_active_features()
                                     if self._yane._feature_gate else []),
                    pop_size=pop_size,
                    max_iters=getattr(self._yane, "_max_iterations", 0) or 0,
                    wall_time=_time.perf_counter() - self._start_time,
                    total_generations=total_gen,
                    final_fitness=best_fit,
                )
                self._show_auto_report(report)
            except Exception:
                pass
            self._auto_profile_info = None

        self._update_ram_bar()
        self._worker = None
        self._yane = None

    def _show_auto_report(self, report: str) -> None:
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle("Auto-Train Report")
        dlg.resize(580, 420)
        lay = QVBoxLayout(dlg)
        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setFontFamily("monospace")
        txt.setPlainText(report)
        lay.addWidget(txt)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(dlg.accept)
        lay.addWidget(btns)
        dlg.exec()

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
