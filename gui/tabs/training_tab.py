"""Training tab: start/stop/configure training, live fitness chart, export."""
from __future__ import annotations
import time as _time
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QPushButton, QFrame, QLabel, QSizePolicy, QFormLayout,
    QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QMessageBox, QProgressBar, QTabWidget,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QImage, QPixmap

from yane.gui.canvas import FitnessChart, SpeciesChart
from yane.gui.worker import TrainingWorker, EpisodeRunner
from yane.gui.examples import load_examples
from yane.gui._helpers import _label, _divider

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
        self._start_time: float = 0.0
        self._last_heavy_update: float = 0.0  # throttle for slow widgets

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

        self.desc_label = _label("", "sectionTitle")
        self.desc_label.setWordWrap(True)
        cfg_form.addRow(self.desc_label)

        self.spin_inputs  = QSpinBox(); self.spin_inputs.setRange(1, 1024)
        self.spin_outputs = QSpinBox(); self.spin_outputs.setRange(1, 256)
        self.spin_nodes   = QSpinBox(); self.spin_nodes.setRange(0, 500); self.spin_nodes.setSpecialValueText("unlimited")
        self.spin_conns   = QSpinBox(); self.spin_conns.setRange(0, 5000); self.spin_conns.setSpecialValueText("unlimited")
        self.spin_pop     = QSpinBox(); self.spin_pop.setRange(2, 1000); self.spin_pop.setValue(100)
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
        self.spin_nodes.setToolTip(
            "Maximale Gesamtzahl an Nodes pro Netz (Input + Hidden + Output).\n"
            "Begrenzt die Netzgröße — kleinere Netze sind schneller aber weniger ausdrucksstark.\n"
            "0 = unbegrenzt (Netz wächst so groß wie nötig).")
        self.spin_conns.setToolTip(
            "Maximale Anzahl an Verbindungen (Synapsen) pro Netz.\n"
            "Begrenzt die Komplexität — weniger Verbindungen = schnellere Forward-Passes.\n"
            "0 = unbegrenzt.")
        self.spin_pop.setToolTip(
            "Anzahl der Genomes in der Population.\n"
            "Mehr Genomes = mehr Diversität und robustere Exploration,\n"
            "aber langsamere Iterationen (jedes Genome muss einmal evaluiert werden).\n"
            "Typisch: 50–200 für einfache Tasks, 200–500 für komplexe.")
        self.dspin_mem.setToolTip(
            "Maximaler RAM-Verbrauch dieses Prozesses in GB.\n"
            "Bei Überschreitung wird die Population auf die Hälfte geschrumpft\n"
            "(schlechteste Genomes werden entfernt).")
        self.dspin_target.setToolTip(
            "Ziel-Fitness: Training stoppt automatisch wenn ein Genome diesen Wert erreicht.\n"
            "— = kein Zielwert (Training läuft bis du Stop drückst).\n"
            "Beispiel: CartPole gilt als gelöst ab Fitness 500.")

        # --- Advanced settings -------------------------------------------------
        self.chk_fitness_shaping = QCheckBox("aktiv")
        self.chk_fitness_shaping.setChecked(False)
        self.chk_fitness_shaping.setToolTip(
            "Rank-basierte Fitness-Transformation (Fitness-Shaping).\n"
            "Ersetzt shared_fitness durch lineare Ränge vor der Selektion.\n"
            "Macht Selektion robust gegen Fitness-Ausreißer und Skalierungsunterschiede.\n"
            "Empfohlen bei Aufgaben mit sehr unterschiedlichen Fitness-Größenordnungen.")

        self.dspin_interspecies = QDoubleSpinBox()
        self.dspin_interspecies.setRange(0.0, 0.2)
        self.dspin_interspecies.setSingleStep(0.01)
        self.dspin_interspecies.setValue(0.0)
        self.dspin_interspecies.setDecimals(2)
        self.dspin_interspecies.setSpecialValueText("—")
        self.dspin_interspecies.setToolTip(
            "Anteil der Crossover-Events, bei denen der zweite Elternteil\n"
            "aus einer anderen Species gewählt wird (Interspecies Crossover).\n"
            "0.0 = nur innerhalb der eigenen Species (Standard)\n"
            "0.05 = 5 % aller Crossovers sind species-übergreifend\n"
            "Hilft, lokale Optima aufzubrechen und Innovationen zu kombinieren.")

        self.dspin_convergence_spread = QDoubleSpinBox()
        self.dspin_convergence_spread.setRange(0.0, 1000.0)
        self.dspin_convergence_spread.setSingleStep(0.1)
        self.dspin_convergence_spread.setValue(0.0)
        self.dspin_convergence_spread.setDecimals(4)
        self.dspin_convergence_spread.setSpecialValueText("—")
        self.dspin_convergence_spread.setToolTip(
            "Konvergenz-Stop: Fitness-IQR-Schwelle.\n"
            "Training stoppt, wenn der Interquartilsabstand (IQR) der Fitness\n"
            "in der Population unter diesen Wert fällt UND die Population voll stagniert.\n"
            "0.0 = deaktiviert.\n"
            "0.01 = typisch für normalisierte Fitness, 1.0 für rohe Werte.")

        self.dspin_convergence_stagnation = QDoubleSpinBox()
        self.dspin_convergence_stagnation.setRange(0.1, 10.0)
        self.dspin_convergence_stagnation.setSingleStep(0.1)
        self.dspin_convergence_stagnation.setValue(1.0)
        self.dspin_convergence_stagnation.setDecimals(1)
        self.dspin_convergence_stagnation.setToolTip(
            "Konvergenz-Stop: Mindest-Stagnation (Faktor).\n"
            "1.0 = volle Stagnation nötig bevor Konvergenz geprüft wird.\n"
            "0.5 = halbe Stagnation reicht (aggressiver).")

        self.dspin_early_stop = QDoubleSpinBox()
        self.dspin_early_stop.setRange(0.0, 10.0)
        self.dspin_early_stop.setSingleStep(0.1)
        self.dspin_early_stop.setValue(0.0)
        self.dspin_early_stop.setDecimals(2)
        self.dspin_early_stop.setSpecialValueText("—")
        self.dspin_early_stop.setToolTip(
            "Early-Stopping-Faktor für Generator-basierte Fitnessfunktionen.\n"
            "Bricht die Evaluierung eines Genoms ab, wenn die extrapolierte\n"
            "Fitness unter best - abs(best) * factor fällt.\n"
            "0.0 = deaktiviert. 1.0 = großzügig (Standard wenn aktiv).\n"
            "Nur relevant wenn die Fitnessfunktion yield verwendet.")

        self.spin_efficiency_max_ms = QDoubleSpinBox()
        self.spin_efficiency_max_ms.setRange(0.0, 10000.0)
        self.spin_efficiency_max_ms.setSingleStep(10.0)
        self.spin_efficiency_max_ms.setValue(0.0)
        self.spin_efficiency_max_ms.setDecimals(1)
        self.spin_efficiency_max_ms.setSpecialValueText("—")
        self.spin_efficiency_max_ms.setToolTip(
            "Effizienzstrafe: Referenzzeit in ms.\n"
            "Genome, die schneller als diese Zeit evaluieren, bekommen\n"
            "einen Bonus bei der Elternauswahl. Langsamere werden bestraft.")

        self.dspin_efficiency_penalty = QDoubleSpinBox()
        self.dspin_efficiency_penalty.setRange(0.0, 1.0)
        self.dspin_efficiency_penalty.setSingleStep(0.001)
        self.dspin_efficiency_penalty.setValue(0.0)
        self.dspin_efficiency_penalty.setDecimals(4)
        self.dspin_efficiency_penalty.setSpecialValueText("—")
        self.dspin_efficiency_penalty.setToolTip(
            "Effizienzstrafe: Strafe pro ms über der Referenzzeit.\n"
            "0.0 = deaktiviert.\n"
            "0.001 = moderat, 0.01 = strikt.")

        self.spin_elite_global = QSpinBox()
        self.spin_elite_global.setRange(0, 50)
        self.spin_elite_global.setValue(1)
        self.spin_elite_global.setToolTip(
            "Globale Elite: Anzahl der Top-Genome, die nie aus der\n"
            "Population entfernt werden (Elitismus). Default 1.\n"
            "Höhere Werte bewahren mehr gute Lösungen, reduzieren aber Diversität.")

        self.spin_elite_species = QSpinBox()
        self.spin_elite_species.setRange(0, 10)
        self.spin_elite_species.setValue(1)
        self.spin_elite_species.setToolTip(
            "Species-Elite: Anzahl der Top-Genome pro Species,\n"
            "die nie entfernt werden. Schützt strukturelle Innovationen\n"
            "auch in kleinen Species. Default 1.")

        self.combo_lamarck_mode = QComboBox()
        self.combo_lamarck_mode.addItems(["Adaptiv (Standard)", "Explizit", "Aus"])
        self.combo_lamarck_mode.setToolTip(
            "Lamarck-Modus:\n\n"
            "Adaptiv (Standard) — läuft automatisch bei Stagnation.\n"
            "  Schrittanzahl steigt von 0 → max_steps proportional zum Stagnationsgrad.\n"
            "  Nur die besten 20 % der Population werden verfeinert.\n"
            "  Kein Setup nötig, empfohlen für die meisten Aufgaben.\n\n"
            "Explizit — feste Schrittzahl vor jeder Fitnessbewertung.\n"
            "  Schrittanzahl rechts wählen.\n"
            "  Gut für Dataset-Aufgaben (XOR, Regression), wo jede Eval billig ist.\n"
            "  Nicht empfohlen für Gym-Umgebungen (jeder Schritt = 1 Episode!).\n\n"
            "Aus — Lamarck komplett deaktiviert.")

        self.spin_lamarck = QSpinBox()
        self.spin_lamarck.setRange(1, 20)
        self.spin_lamarck.setValue(5)
        self.spin_lamarck.setEnabled(False)
        self.spin_lamarck.setToolTip(
            "Anzahl expliziter Hill-Climbing-Schritte pro Genome (nur im Modus 'Explizit').\n\n"
            "Was passiert pro Schritt:\n"
            "  1. Alle Gewichte + Biases mit Gauß-Rauschen perturbieren\n"
            "  2. Fitness messen\n"
            "  3. Besser → behalten,  Schlechter → zurücksetzen\n\n"
            "Schrittgröße = sigma_global des Genoms.\n"
            "3–5 = empfohlen für Regression / Supervised Learning.")

        self.combo_lamarck_mode.currentIndexChanged.connect(self._on_lamarck_mode_changed)

        self.spin_multi_eval = QSpinBox()
        self.spin_multi_eval.setRange(1, 50)
        self.spin_multi_eval.setValue(1)
        self.spin_multi_eval.setSpecialValueText("—")
        self.spin_multi_eval.setToolTip(
            "Mehrfachbewertung: Anzahl Evaluierungen pro Genome.\n\n"
            "Bei stochastischen Umgebungen ist eine einzelne Episode oft zu verrauscht.\n"
            "Mit n > 1 wird jedes Genome n-mal bewertet und die Ergebnisse aggregiert.\n\n"
            "1 = deaktiviert (Standard)\n"
            "3–10 = gut für rauschige Gym-Umgebungen\n\n"
            "Kosten: n-facher Zeitaufwand pro Genome!")

        self.combo_aggregation = QComboBox()
        self.combo_aggregation.addItems(["mean", "median", "min"])
        self.combo_aggregation.setToolTip(
            "Aggregationsmethode für Mehrfachbewertung:\n\n"
            "mean   — Mittelwert (Standard, empfohlen)\n"
            "median — Median; robust gegen einzelne Ausreißer-Episoden\n"
            "min    — Worst-Case; konservativste Wahl, selektiert auf Robustheit")
        self.combo_aggregation.setEnabled(False)

        self.dspin_sigma_penalty = QDoubleSpinBox()
        self.dspin_sigma_penalty.setRange(0.0, 10.0)
        self.dspin_sigma_penalty.setSingleStep(0.1)
        self.dspin_sigma_penalty.setValue(0.0)
        self.dspin_sigma_penalty.setDecimals(2)
        self.dspin_sigma_penalty.setSpecialValueText("—")
        self.dspin_sigma_penalty.setToolTip(
            "Varianz-Strafe bei Mehrfachbewertung.\n\n"
            "Endwert = aggregate(fitness) − sigma_penalty × std(fitness)\n\n"
            "0   = keine Strafe (Standard)\n"
            "0.5 = mäßige Strafe für inkonsistente Genome\n"
            "1.0 = starke Strafe; ein Genome mit std=2 verliert 2 Fitnesspunkte\n\n"
            "Nützlich wenn robuste Policies gewünscht sind, nicht nur gute Mittelwerte.")
        self.dspin_sigma_penalty.setEnabled(False)

        self.spin_multi_eval.valueChanged.connect(self._on_multi_eval_changed)

        import multiprocessing as _mp
        _ncpu = _mp.cpu_count()
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(0, _ncpu)
        self.spin_workers.setValue(0)
        self.spin_workers.setSpecialValueText("Auto")
        self.spin_workers.setToolTip(
            f"Anzahl paralleler Prozesse für die Fitness-Berechnung.\n\n"
            f"Auto (0): Misst die Evaluierungsgeschwindigkeit und wählt\n"
            f"   automatisch die optimale Worker-Anzahl:\n"
            f"   - Zu schnell (<0.5ms/Genome): sequenziell\n"
            f"   - Mittel (1-10ms/Genome): 2–8 Worker\n"
            f"   - Langsam (>10ms/Genome): alle {_ncpu} CPU-Kerne\n\n"
            f"1: Immer sequenziell (kein MP-Overhead).\n"
            f"2–{_ncpu}: Feste Worker-Anzahl.\n\n"
            f"MP lohnt sich nur für langsame Fitness-Funktionen\n"
            f"(Gym mit langen Episoden, MNIST, eigene komplexe Funktionen).\n"
            f"Für XOR/Regression ist 'Auto' immer optimal.\n\n"
            f"Dein System hat {_ncpu} CPU-Kerne.")

        self.spin_species = QSpinBox()
        self.spin_species.setRange(2, 50)
        self.spin_species.setValue(5)
        self.spin_species.setToolTip(
            "Zielanzahl der Arten (Species) in der Population.\n"
            "Der Kompatibilitätsschwellwert wird automatisch angepasst,\n"
            "um diese Anzahl zu erreichen.\n\n"
            "Mehr Arten = mehr Strukturnischen geschützt.\n"
            "Hilfreich für XOR-ähnliche Aufgaben (Regression, binäre Mappings):\n"
            "  5   → Standard (gut für Gym-Environments)\n"
            "  10–20 → besser für diskrete Mappings (XOR, Regression)\n\n"
            "Benchmark: species=20 löst Regression 2→2 in 3/5 Seeds bei 60k it,\n"
            "           species=5  löst sie in 0/5 Seeds bei gleicher Laufzeit.")

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
        self.chk_curriculum.setToolTip(
            "Curriculum Learning: Training in aufsteigend schwieriger werdenden Stufen.\n\n"
            "Pi-Ziffern: Stufe 1 (3 Ziffern) → Stufe 2 (6 Ziffern) → Stufe 3 (10 Ziffern).\n"
            "Die Population wird bei jedem Stufenwechsel behalten, aber neu bewertet.\n\n"
            "Erzwingt sequentielle Ausführung (kein Multiprocessing).\n"
            "Curriculum-Stufe und Fortschritt sind im linken Panel sichtbar.")

        cfg_form.addRow("Inputs:",         self.spin_inputs)
        cfg_form.addRow("Outputs:",        self.spin_outputs)
        cfg_form.addRow("Max nodes:",      self.spin_nodes)
        cfg_form.addRow("Max connections:", self.spin_conns)
        cfg_form.addRow("Population:",     self.spin_pop)
        workers_row = QWidget()
        workers_lay = QHBoxLayout(workers_row)
        workers_lay.setContentsMargins(0, 0, 0, 0)
        workers_lay.addWidget(self.spin_workers)
        self.lbl_workers_active = _label("", "mutRate")
        self.lbl_workers_active.setToolTip(
            "Tatsächlich genutzte Worker-Anzahl während des Trainings.\n"
            "Wird beim Start automatisch bestimmt (Auto-Modus)\n"
            "oder entspricht dem manuell gewählten Wert.")
        workers_lay.addWidget(self.lbl_workers_active)
        cfg_form.addRow("Workers:", workers_row)
        cfg_form.addRow("Target species:", self.spin_species)
        lamarck_row = QWidget()
        lamarck_lay = QHBoxLayout(lamarck_row)
        lamarck_lay.setContentsMargins(0, 0, 0, 0)
        lamarck_lay.setSpacing(4)
        lamarck_lay.addWidget(self.combo_lamarck_mode, stretch=1)
        lamarck_lay.addWidget(self.spin_lamarck)
        cfg_form.addRow("Lamarck:",        lamarck_row)
        cfg_form.addRow("Multi-eval:",     self.spin_multi_eval)
        cfg_form.addRow("Aggregation:",   self.combo_aggregation)
        cfg_form.addRow("Sigma penalty:", self.dspin_sigma_penalty)
        cfg_form.addRow("Normalization:", self.chk_normalize)
        cfg_form.addRow("Memory:",        self.chk_memory)
        cfg_form.addRow("Curriculum:",    self.chk_curriculum)
        cfg_form.addRow("Memory limit:",   self.dspin_mem)
        cfg_form.addRow("Target fitness:", self.dspin_target)

        # --- Advanced settings group ---
        advance_grp = QGroupBox("Advanced")
        advance_form = QFormLayout(advance_grp)
        advance_form.setSpacing(3)
        advance_form.setContentsMargins(0, 0, 0, 0)
        advance_form.addRow("Fitness shaping:",   self.chk_fitness_shaping)
        advance_form.addRow("Interspecies crossover:", self.dspin_interspecies)
        converge_row = QWidget()
        converge_lay = QHBoxLayout(converge_row)
        converge_lay.setContentsMargins(0, 0, 0, 0)
        converge_lay.setSpacing(4)
        converge_lay.addWidget(self.dspin_convergence_spread)
        converge_lay.addWidget(QLabel("×"))
        converge_lay.addWidget(self.dspin_convergence_stagnation)
        advance_form.addRow("Convergence stop (eps × stag):", converge_row)
        advance_form.addRow("Early stop factor:",  self.dspin_early_stop)
        eff_row = QWidget()
        eff_lay = QHBoxLayout(eff_row)
        eff_lay.setContentsMargins(0, 0, 0, 0)
        eff_lay.setSpacing(4)
        eff_lay.addWidget(self.spin_efficiency_max_ms)
        eff_lay.addWidget(QLabel("ms ×"))
        eff_lay.addWidget(self.dspin_efficiency_penalty)
        advance_form.addRow("Efficiency penalty:", eff_row)
        elite_row = QWidget()
        elite_lay = QHBoxLayout(elite_row)
        elite_lay.setContentsMargins(0, 0, 0, 0)
        elite_lay.setSpacing(4)
        elite_lay.addWidget(self.spin_elite_global)
        elite_lay.addWidget(QLabel("global /"))
        elite_lay.addWidget(self.spin_elite_species)
        elite_lay.addWidget(QLabel("per species"))
        advance_form.addRow("Elitism:", elite_row)
        layout.addWidget(cfg)
        layout.addWidget(advance_grp)

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
        self.lbl_fitness = _label("—", "statValue")
        self.lbl_speed   = _label("—", "statValue")
        prog_form_layout.addRow("Iteration:",      self.lbl_iter)
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
    _CATEGORY_ORDER = ["Dataset", "Toy Text", "Classic Control", "Box2D", "Pixel", "Sonstiges"]

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

        for cat in self._CATEGORY_ORDER:
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
        self.spin_nodes.setValue(ex.max_nodes)
        self.spin_conns.setValue(ex.max_connections)
        self.spin_pop.setValue(ex.default_population)
        self.spin_species.setValue(ex.default_target_species)
        self.chk_fitness_shaping.setChecked(ex.default_fitness_shaping)
        if ex.default_lamarck_steps > 0:
            self.combo_lamarck_mode.setCurrentText("Explizit")
            self.spin_lamarck.setValue(ex.default_lamarck_steps)
            self.spin_lamarck.setEnabled(True)
        else:
            self.combo_lamarck_mode.setCurrentText("Adaptiv (Standard)")
            self.spin_lamarck.setEnabled(False)
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
        self._best_genome = None
        self.btn_run_best.setEnabled(False)
        self.example_changed.emit(ex)

    def _on_lamarck_mode_changed(self, index: int) -> None:
        self.spin_lamarck.setEnabled(index == 1)  # only enabled for "Explizit"

    def _on_multi_eval_changed(self, value: int) -> None:
        enabled = value > 1
        self.combo_aggregation.setEnabled(enabled)
        self.dspin_sigma_penalty.setEnabled(enabled)

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
            from yane.util.logger import setup_logging as _setup_log, write_json as _wj, log_info as _li
            self._yane = NeuroEvolution()
            self._yane.configure(
                n_inputs=self.spin_inputs.value(),
                n_outputs=self.spin_outputs.value(),
                max_nodes=self.spin_nodes.value() or None,
                max_connections=self.spin_conns.value() or None,
                n_initial_hidden=ex.n_initial_hidden,
                stateful=self.chk_memory.isChecked(),
            )
            self._yane.set_population_size(self.spin_pop.value())
            # 0 = Auto (worker determines optimal count at runtime)
            self._yane.set_n_workers(self.spin_workers.value())
            self._yane.set_target_species(self.spin_species.value())

            # --- Advanced settings ---
            if self.chk_fitness_shaping.isChecked():
                self._yane.set_fitness_shaping(True)
            is_rate = self.dspin_interspecies.value()
            if is_rate > 0.0:
                self._yane.set_interspecies_crossover(is_rate)
            conv_eps = self.dspin_convergence_spread.value()
            if conv_eps > 0.0:
                self._yane.set_convergence_stop(
                    conv_eps, self.dspin_convergence_stagnation.value())
            esf = self.dspin_early_stop.value()
            if esf > 0.0:
                self._yane.set_early_stopping(esf)
            eff_max = self.spin_efficiency_max_ms.value()
            eff_pen = self.dspin_efficiency_penalty.value()
            if eff_max > 0.0 and eff_pen > 0.0:
                self._yane.set_efficiency_penalty(eff_max, eff_pen)
            self._yane.set_elitism(
                self.spin_elite_global.value(),
                self.spin_elite_species.value(),
            )

            lamarck_mode = self.combo_lamarck_mode.currentText()
            if lamarck_mode == "Explizit":
                self._yane.set_lamarck(n_steps=self.spin_lamarck.value())
            elif lamarck_mode == "Aus":
                self._yane.set_lamarck_adaptive(max_steps=0)
            n_eval = self.spin_multi_eval.value()
            if n_eval > 1:
                self._yane.set_multi_eval(
                    n=n_eval,
                    aggregation=self.combo_aggregation.currentText(),
                    sigma_penalty=self.dspin_sigma_penalty.value(),
                )
            self._yane.set_resource_limits(max_process_gb=self.dspin_mem.value())
            target = self.dspin_target.value()
            if target > -1e9:
                self._yane.set_min_fitness(target)
            render_cb = None
            if ex.supports_render and self.btn_render.isChecked():
                render_cb = self.render_frame.emit

            if ex.supports_normalization and not self.chk_normalize.isChecked():
                import functools
                make_eval_fn = functools.partial(ex.make_eval, normalize=False)
            else:
                make_eval_fn = ex.make_eval

            # Curriculum: build stages and register on yane before worker starts.
            if ex.make_curriculum is not None and self.chk_curriculum.isChecked():
                normalize = self.chk_normalize.isChecked() if ex.supports_normalization else True
                target_fitness = (
                    self.dspin_target.value()
                    if self.dspin_target.value() > -1e9
                    else ex.target_fitness
                )
                stages = ex.make_curriculum(
                    normalize=normalize,
                    target_fitness=target_fitness,
                )
                self._yane.set_curriculum(stages)

            # --- Structured logging for GUI runs ---------------------------
            _log_dir = _setup_log(f"gui/{ex.name}")
            self._yane._log_run_dir = _log_dir
            self._yane._log_run_name = ex.name
            _wj(_log_dir / "config.json", self._yane._config_dict())
            _li("GUI training started  example=%s  pop_size=%d  target=%s",
                ex.name, self.spin_pop.value(),
                self.dspin_target.value() if self.dspin_target.value() > -1e9 else "none")
            self._log_csv_path = _log_dir / "fitness_history.csv"
            self._log_csv_header = "iteration,best_fitness,mean_fitness,median_fitness,iqr_fitness,species_count,stagnation_count,nodes,connections"
            self._log_csv_interval = max(1, self.spin_pop.value() // 10)
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
        worker = TrainingWorker(self._yane, make_eval_fn, render_cb=render_cb)
        worker.finished.connect(worker.deleteLater)
        worker.iteration_done.connect(self._on_iteration)
        worker.error_occurred.connect(self._on_error)
        worker.info_message.connect(self.status_lbl.setText)
        worker.workers_resolved.connect(self._on_workers_resolved)
        worker.finished.connect(lambda: self._on_finished(run_id))
        if self._episode_runner and self._episode_runner.isRunning():
            self._episode_runner.stop()
            self._episode_runner = None
        worker.start(QThread.Priority.LowPriority)
        self._worker = worker
        self.training_started.emit()
        self.btn_save_ckpt.setEnabled(True)
        self.btn_run_best.setEnabled(False)
        self._render_widget.clear_frame()
        self._score_lbl.setVisible(False)

        self.chart.clear()
        self.species_chart.clear()
        self._start_time = _time.perf_counter()
        self._last_heavy_update = 0.0
        self.lbl_workers_active.setText("")   # cleared until workers_resolved fires
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

    def _on_workers_resolved(self, n: int) -> None:
        if n <= 1:
            self.lbl_workers_active.setText("→ sequential")
        else:
            self.lbl_workers_active.setText(f"→ {n} processes")

    def _on_iteration(self, iteration: int, fitness: float, best_genome, mem: dict) -> None:
        elapsed = _time.perf_counter() - self._start_time
        iter_s = iteration / elapsed if elapsed > 0 else 0.0
        mins, secs = divmod(int(elapsed), 60)
        elapsed_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
        self.lbl_iter.setText(str(iteration))
        self.lbl_fitness.setText(f"{fitness:.4f}   /   {best_genome.fitness:.4f}")
        self.lbl_speed.setText(f"{iter_s:.1f} iter/s   {elapsed_str}")
        self.chart.add_point(fitness)

        # --- Periodic CSV logging -------------------------------------------
        csv_path = getattr(self, '_log_csv_path', None)
        if csv_path is not None and iteration % self._log_csv_interval == 0:
            try:
                from yane.util.logger import write_csv
                write_csv(csv_path, self._log_csv_header,
                    f"{iteration},"
                    f"{mem.get('max_fitness', 0)},"
                    f"{mem.get('avg_fitness', 0)},"
                    f"{mem.get('median_fitness', 0)},"
                    f"{mem.get('fitness_iqr', 0)},"
                    f"{mem.get('species_count', 0)},"
                    f"{mem.get('stagnation_count', 0)},"
                    f"{mem.get('largest_genome_nodes', 0)},"
                    f"{mem.get('largest_genome_connections', 0)}")
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
        except Exception as e:
            self._yane = None
            QMessageBox.critical(self, "Load Checkpoint Error", str(e))

    def _reset_training_buttons(self) -> None:
        self.btn_start.setEnabled(True)
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
        self._update_ram_bar()
        self._worker = None
        self._yane = None

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

