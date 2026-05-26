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
from yane.gui.worker import TrainingWorker, EpisodeRunner
from yane.gui.examples import load_examples
from yane.gui._helpers import _label, _divider, CollapsibleGroup
from yane.gui.remote_config import RemoteEvaluationConfig
from yane.gui.research_features import (
    ResearchFeatureConfig,
    apply_research_features,
    configure_cppn_substrate_population,
)
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

        # --- Ablation checkboxes (all default ON) ---
        self.chk_novelty = QCheckBox("aktiv")
        self.chk_novelty.setChecked(True)
        self.chk_novelty.setToolTip(
            "Novelty Search (Ablation).\n"
            "Wenn aktiv: novel Genome bekommen einen Selektionsbonus (0.1–0.5).\n"
            "Wenn deaktiviert: rein fitness-basierte Selektion, kein Neuheitsbonus.\n"
            "Deaktivieren um den Beitrag von Novelty Search zu messen.")

        self.chk_speciation = QCheckBox("aktiv")
        self.chk_speciation.setChecked(True)
        self.chk_speciation.setToolTip(
            "Speciation (Ablation).\n"
            "Wenn aktiv: NEAT-artige Species-Einteilung, Fitness-Sharing, Species-Budget.\n"
            "Wenn deaktiviert: alle Genome in einer einzigen Species, kein Nischen-Schutz.\n"
            "Deaktivieren um den Beitrag von Speciation zu messen.")

        self.chk_crossover = QCheckBox("aktiv")
        self.chk_crossover.setChecked(True)
        self.chk_crossover.setToolTip(
            "Crossover (Ablation).\n"
            "Wenn aktiv: Offspring kann durch Crossover zweier Eltern entstehen.\n"
            "Wenn deaktiviert: alle Offspring entstehen durch Mutation eines Elternteils.\n"
            "Deaktivieren um den Beitrag von sexueller Reproduktion zu messen.")

        self.chk_diversity_injection = QCheckBox("aktiv")
        self.chk_diversity_injection.setChecked(True)
        self.chk_diversity_injection.setToolTip(
            "Diversity Injection (Ablation).\n"
            "Wenn aktiv: bei Stagnation werden frische oder strukturell diverse Genome injiziert.\n"
            "Wenn deaktiviert: kein automatischer Stagnations-Escape.\n"
            "Deaktivieren um den Beitrag der Diversitätsinjektion zu messen.")

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

        self.combo_interspecies_mode = QComboBox()
        self.combo_interspecies_mode.addItems(["Fix", "Adaptiv"])
        self.combo_interspecies_mode.setToolTip(
            "Interspecies-Crossover-Zeitplan.\n"
            "Fix nutzt die eingestellte Rate direkt.\n"
            "Adaptiv erhöht die Live-Rate bei globaler oder Species-Stagnation.")
        self.dspin_interspecies_max = QDoubleSpinBox()
        self.dspin_interspecies_max.setRange(0.0, 0.5)
        self.dspin_interspecies_max.setSingleStep(0.01)
        self.dspin_interspecies_max.setValue(0.2)
        self.dspin_interspecies_max.setDecimals(2)
        self.dspin_interspecies_max.setToolTip(
            "Maximale adaptive Interspecies-Crossover-Rate bei starker Stagnation.")

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

        self.combo_lamarck_schedule = QComboBox()
        self.combo_lamarck_schedule.addItems(["Adaptiv", "Explizit", "Aus"])
        self.combo_lamarck_schedule.setToolTip(
            "Lamarck-Zeitplan.\n"
            "Adaptiv läuft nur bei Stagnation und nur für starke Genome.\n"
            "Explizit läuft mit fester Schrittzahl vor jeder Evaluation.\n"
            "Aus deaktiviert Lamarck vollständig.")

        self.combo_lamarck_optimizer = QComboBox()
        self.combo_lamarck_optimizer.addItems(["Hill-Climbing", "NES", "SA", "CMA-ES"])
        self.combo_lamarck_optimizer.setToolTip(
            "Lokaler Optimierer für Lamarck.\n"
            "Der Optimierer ist unabhängig vom Zeitplan, also auch NES, SA und CMA-ES\n"
            "können adaptiv bei Stagnation laufen.")

        self.spin_lamarck = QSpinBox()
        self.spin_lamarck.setRange(1, 20)
        self.spin_lamarck.setValue(5)
        self.spin_lamarck.setEnabled(False)
        self.spin_lamarck.setToolTip(
            "Anzahl Verfeinerungsschritte pro Genome (für alle Explizit-Modi).\n\n"
            "Hill-Climbing: 1 Eval pro Schritt.\n"
            "NES: 2k+1 Evals für k Schritte (antithetische Paare + Gradient-Step).\n"
            "SA: 1 Eval pro Schritt mit Temperatur-basierter Akzeptanz.\n\n"
            "CMA-ES: mehrere Kandidaten pro Schritt mit Kovarianz-Update.\n\n"
            "Schrittgröße = lamarck_sigma des Genoms.\n"
            "3–5 = empfohlen für Regression / Supervised Learning.")

        self.combo_lamarck_schedule.currentIndexChanged.connect(self._on_lamarck_mode_changed)

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

        self.chk_multi_objective = QCheckBox("aktiv")
        self.chk_multi_objective.setChecked(False)
        self.chk_multi_objective.setToolTip(
            "Multi-Objective Training für GUI-Beispiele.\n"
            "Die Fitnessfunktion wird als Vektor (raw_fitness, complexity) bewertet.\n"
            "Selektion nutzt Pareto-Rang + Crowding-Distance; Logs/Stop nutzen\n"
            "raw_fitness - complexity_weight × complexity.")

        self.dspin_mo_complexity = QDoubleSpinBox()
        self.dspin_mo_complexity.setRange(0.0, 10.0)
        self.dspin_mo_complexity.setSingleStep(0.001)
        self.dspin_mo_complexity.setDecimals(4)
        self.dspin_mo_complexity.setValue(0.01)
        self.dspin_mo_complexity.setToolTip(
            "Gewicht für das zweite Objective 'Komplexität'.\n"
            "Die GUI nutzt connection_count als minimiertes Objective.\n"
            "0.01 bedeutet: Skalarfitness = raw_fitness - 0.01 × connections.")

        self.chk_quality_diversity = QCheckBox("aktiv")
        self.chk_quality_diversity.setChecked(False)
        self.chk_quality_diversity.setToolTip(
            "Quality Diversity / MAP-Elites aktivieren.\n"
            "Das Archiv speichert das beste Genom pro Descriptor-Zelle und nutzt\n"
            "Archiv-Eliten bei Stagnation als zusätzliche Diversity-Injektion.")

        self.combo_qd_descriptor = QComboBox()
        self.combo_qd_descriptor.addItems(["Topology", "Behavior"])
        self.combo_qd_descriptor.setToolTip(
            "Descriptor für MAP-Elites:\n"
            "Topology: (hidden_nodes, connections)\n"
            "Behavior: Outputs auf festen Probeinputs")

        self.chk_matrix_forward = QCheckBox("aktiv")
        self.chk_matrix_forward.setChecked(False)
        self.chk_matrix_forward.setToolTip(
            "Matrix-Forward aktivieren.\n"
            "Kompatible azyklische Feedforward-Genome nutzen automatisch den\n"
            "NumPy-Matrixpfad; inkompatible Genome fallen auf den normalen Forward zurück.")

        self.chk_fitness_components = QCheckBox("aktiv")
        self.chk_fitness_components.setChecked(False)
        self.chk_fitness_components.setToolTip(
            "Optionale Descriptor-/Fitness-Komponenten zur Task-Fitness addieren.\n"
            "Die GUI nutzt kleine Topologie- und Verhaltensterme; im adaptiven Modus\n"
            "werden ihre Gewichte bei Stagnation automatisch verschoben.")

        self.combo_fitness_component_mode = QComboBox()
        self.combo_fitness_component_mode.addItems(["Fix", "Adaptiv"])
        self.combo_fitness_component_mode.setToolTip(
            "Fix hält Komponenten-Gewichte konstant.\n"
            "Adaptiv zeichnet Gewichtshistorie auf und passt Gewichte bei Stagnation an.")

        self.chk_cppn_substrate = QCheckBox("aktiv")
        self.chk_cppn_substrate.setChecked(False)
        self.chk_cppn_substrate.setToolTip(
            "Initialisiert die Population mit einem CPPN-generierten HyperNEAT-Substrat.\n"
            "Danach trainiert das erzeugte Netz als normales YANE-Genome weiter.")

        self.spin_cppn_hidden = QSpinBox()
        self.spin_cppn_hidden.setRange(0, 64)
        self.spin_cppn_hidden.setValue(2)
        self.spin_cppn_hidden.setToolTip("Anzahl Hidden-Nodes in der CPPN-Substratschicht.")

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
        cfg_form.addRow("Max nodes:",      self.spin_nodes)
        cfg_form.addRow("Max connections:", self.spin_conns)
        cfg_form.addRow("Population:",     self.spin_pop)
        self.lbl_workers_active = _label("", "mutRate")
        self.lbl_workers_active.setToolTip(
            "Tatsächlich genutzte Worker-Anzahl während des Trainings.\n"
            "Wird beim Start automatisch bestimmt (Auto-Modus)\n"
            "oder entspricht dem manuell gewählten Wert.")
        cfg_form.addRow("Workers:", inline_row(self.spin_workers, self.lbl_workers_active))
        cfg_form.addRow("Target species:", self.spin_species)
        cfg_form.addRow(
            "Lamarck:",
            inline_row(
                self.combo_lamarck_schedule,
                self.combo_lamarck_optimizer,
                self.spin_lamarck,
            ),
        )
        cfg_form.addRow("Multi-eval:",     self.spin_multi_eval)
        cfg_form.addRow("Aggregation:",   self.combo_aggregation)
        cfg_form.addRow("Sigma penalty:", self.dspin_sigma_penalty)
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

        self.spin_max_eval = QSpinBox()
        self.spin_max_eval.setRange(0, 10_000_000)
        self.spin_max_eval.setValue(0)
        self.spin_max_eval.setSpecialValueText("—")
        self.spin_max_eval.setSingleStep(10000)
        self.spin_max_eval.setToolTip(
            "Maximale Fitness-Evaluations-Aufrufe.\n"
            "— = unbegrenzt.\n"
            "Zählt tatsächliche Fitness-Funktionsaufrufe, nicht nur Iterationen.")
        cfg_form.addRow("Max evaluations:", self.spin_max_eval)

        self.combo_log_format = QComboBox()
        self.combo_log_format.addItems(["csv", "jsonlines", "both"])
        self.combo_log_format.setToolTip(
            "Log-Format für Trainingsläufe.\n"
            "csv: Standard-CSV (Default).\n"
            "jsonlines: JSON pro Heartbeat (komplettes Diagnostics-Dict).\n"
            "both: CSV + JSONL gleichzeitig.")
        cfg_form.addRow("Log format:", self.combo_log_format)

        # --- Advanced settings group (collapsed by default) ---
        advance_grp = CollapsibleGroup("Advanced", collapsed=True)
        advance_grp.addRow("Fitness shaping:",   self.chk_fitness_shaping)
        advance_grp.addRow(
            "Multi-objective:",
            inline_row(self.chk_multi_objective, QLabel("complexity ×"), self.dspin_mo_complexity),
        )
        advance_grp.addRow(
            "Quality diversity:",
            inline_row(self.chk_quality_diversity, self.combo_qd_descriptor),
        )
        advance_grp.addRow(
            "Fitness components:",
            inline_row(self.chk_fitness_components, self.combo_fitness_component_mode),
        )
        advance_grp.addRow(
            "CPPN substrate init:",
            inline_row(self.chk_cppn_substrate, QLabel("hidden"), self.spin_cppn_hidden),
        )
        advance_grp.addRow("Matrix forward:", self.chk_matrix_forward)

        # ── New feature controls ────────────────────────────────────────────
        self.chk_weight_inheritance = QCheckBox("aktiv")
        self.chk_weight_inheritance.setToolTip(
            "Fitness-gewichtetes Blending beim Crossover (blend_alpha=0.7).")
        advance_grp.addRow("Weight inheritance:", self.chk_weight_inheritance)

        self.chk_online_tuning = QCheckBox("aktiv")
        self.chk_online_tuning.setToolTip(
            "UCB1-Bandit-Tuning von mutation_rate und lamarck_steps während des Trainings.")
        advance_grp.addRow("Online tuning:", self.chk_online_tuning)

        self.combo_codec = QComboBox()
        self.combo_codec.addItems(["pickle", "json"])
        self.combo_codec.setToolTip("Checkpoint-Serialisierungsformat.")
        advance_grp.addRow("Checkpoint codec:", self.combo_codec)

        # ── Island model ────────────────────────────────────────────────────
        self.chk_island_model = QCheckBox("aktiv")
        self.chk_island_model.setChecked(False)
        self.chk_island_model.setToolTip(
            "Multi-Population Island Model aktivieren.\n"
            "Jede Insel hat eine eigene Population; regelmäßig migrieren die\n"
            "besten Genome zwischen Inseln.")
        self.spin_n_islands = QSpinBox()
        self.spin_n_islands.setRange(2, 50)
        self.spin_n_islands.setValue(4)
        self.spin_n_islands.setEnabled(False)
        self.spin_migrate_interval = QSpinBox()
        self.spin_migrate_interval.setRange(1, 100)
        self.spin_migrate_interval.setValue(5)
        self.spin_migrate_interval.setEnabled(False)
        self.spin_migrate_count = QSpinBox()
        self.spin_migrate_count.setRange(1, 20)
        self.spin_migrate_count.setValue(3)
        self.spin_migrate_count.setEnabled(False)
        self.chk_island_model.toggled.connect(
            lambda on: (
                self.spin_n_islands.setEnabled(on),
                self.spin_migrate_interval.setEnabled(on),
                self.spin_migrate_count.setEnabled(on),
            )
        )
        advance_grp.addRow(
            "Island model:",
            inline_row(
                self.chk_island_model, QLabel("islands"), self.spin_n_islands,
                QLabel("migrate every"), self.spin_migrate_interval,
                QLabel("count"), self.spin_migrate_count,
            ),
        )

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

        # ── Adaptive population ─────────────────────────────────────────────
        self.chk_adaptive_pop = QCheckBox("aktiv")
        self.chk_adaptive_pop.setChecked(False)
        self.chk_adaptive_pop.setToolTip(
            "Populationsgröße dynamisch anpassen.\n"
            "Wächst bei Entdeckungsphase, schrumpft bei Konvergenz.")
        self.spin_adaptive_pop_min = QSpinBox()
        self.spin_adaptive_pop_min.setRange(2, 500)
        self.spin_adaptive_pop_min.setValue(20)
        self.spin_adaptive_pop_min.setEnabled(False)
        self.spin_adaptive_pop_max = QSpinBox()
        self.spin_adaptive_pop_max.setRange(10, 2000)
        self.spin_adaptive_pop_max.setValue(200)
        self.spin_adaptive_pop_max.setEnabled(False)
        self.chk_adaptive_pop.toggled.connect(
            lambda on: (
                self.spin_adaptive_pop_min.setEnabled(on),
                self.spin_adaptive_pop_max.setEnabled(on),
            )
        )
        advance_grp.addRow(
            "Adaptive pop:",
            inline_row(
                self.chk_adaptive_pop, QLabel("min"), self.spin_adaptive_pop_min,
                QLabel("max"), self.spin_adaptive_pop_max,
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
        ablation_row = QWidget()
        ablation_lay = QHBoxLayout(ablation_row)
        ablation_lay.setContentsMargins(0, 0, 0, 0)
        ablation_lay.setSpacing(8)
        for chk, lbl in (
            (self.chk_novelty,            "Novelty"),
            (self.chk_speciation,         "Speciation"),
            (self.chk_crossover,          "Crossover"),
            (self.chk_diversity_injection, "Diversity inj."),
        ):
            chk.setText(lbl)
            ablation_lay.addWidget(chk)
        ablation_lay.addStretch()
        advance_grp.addRow("Ablation (off = disable):", ablation_row)
        interspecies_row = QWidget()
        interspecies_lay = QHBoxLayout(interspecies_row)
        interspecies_lay.setContentsMargins(0, 0, 0, 0)
        interspecies_lay.setSpacing(4)
        interspecies_lay.addWidget(self.combo_interspecies_mode)
        interspecies_lay.addWidget(self.dspin_interspecies)
        interspecies_lay.addWidget(QLabel("max"))
        interspecies_lay.addWidget(self.dspin_interspecies_max)
        advance_grp.addRow("Interspecies crossover:", interspecies_row)
        converge_row = QWidget()
        converge_lay = QHBoxLayout(converge_row)
        converge_lay.setContentsMargins(0, 0, 0, 0)
        converge_lay.setSpacing(4)
        converge_lay.addWidget(self.dspin_convergence_spread)
        converge_lay.addWidget(QLabel("×"))
        converge_lay.addWidget(self.dspin_convergence_stagnation)
        advance_grp.addRow("Convergence stop (eps × stag):", converge_row)
        advance_grp.addRow("Early stop factor:",  self.dspin_early_stop)
        eff_row = QWidget()
        eff_lay = QHBoxLayout(eff_row)
        eff_lay.setContentsMargins(0, 0, 0, 0)
        eff_lay.setSpacing(4)
        eff_lay.addWidget(self.spin_efficiency_max_ms)
        eff_lay.addWidget(QLabel("ms ×"))
        eff_lay.addWidget(self.dspin_efficiency_penalty)
        advance_grp.addRow("Efficiency penalty:", eff_row)
        elite_row = QWidget()
        elite_lay = QHBoxLayout(elite_row)
        elite_lay.setContentsMargins(0, 0, 0, 0)
        elite_lay.setSpacing(4)
        elite_lay.addWidget(self.spin_elite_global)
        elite_lay.addWidget(QLabel("global /"))
        elite_lay.addWidget(self.spin_elite_species)
        elite_lay.addWidget(QLabel("per species"))
        advance_grp.addRow("Elitism:", elite_row)
        layout.addWidget(cfg)
        layout.addWidget(advance_grp)

        # --- Adaptive Control Section ---
        adaptive_grp = CollapsibleGroup("Adaptive Control", collapsed=True)

        # Interspecies Crossover live display
        self.lbl_interspecies_live = _label("—", "mutRate")
        self.lbl_interspecies_live.setToolTip(
            "Live-Rate des adaptiven Interspecies-Crossovers.\n"
            "Steigt bei Stagnation, niedrigem Novelty oder Species-Isolation.\n"
            "Sinkt wenn Crossover-Offspring schlechter abschneiden als die Eltern.")

        self.lbl_interspecies_trigger = _label("—", "sectionTitle")
        self.lbl_interspecies_trigger.setWordWrap(True)
        self.lbl_interspecies_trigger.setToolTip("Letzter Grund für die adaptive Rate-Anpassung.")

        interspecies_live_row = QWidget()
        interspecies_live_lay = QHBoxLayout(interspecies_live_row)
        interspecies_live_lay.setContentsMargins(0, 0, 0, 0)
        interspecies_live_lay.setSpacing(6)
        interspecies_live_lay.addWidget(self.lbl_interspecies_live)
        interspecies_live_lay.addWidget(QLabel("→"))
        interspecies_live_lay.addWidget(self.lbl_interspecies_trigger, stretch=1)
        adaptive_grp.addRow("Interspecies rate:", interspecies_live_row)

        # Interspecies success diagnostics
        self.lbl_interspecies_success = _label("—", "mutRate")
        self.lbl_interspecies_success.setToolTip(
            "Erfolgsrate Interspecies-Crossover: Anteil der Offspring, die besser\n"
            "als ihre Eltern sind. Niedrig = Protection wird aktiv (Rate wird gesenkt).")
        adaptive_grp.addRow("Cross-species success:", self.lbl_interspecies_success)

        # Adaptive Controller
        self.chk_adaptive_ctrl = QCheckBox("aktiv")
        self.chk_adaptive_ctrl.setChecked(False)
        self.chk_adaptive_ctrl.setToolTip(
            "Aktiviert den zentralen Adaptive Control Layer.\n"
            "Tick pro Generation: Interspecies-Crossover, QD-Druck, Pruning und\n"
            "Lamarck-Budget werden anhand von Plateau, Diversität und Novelty gesteuert.")
        adaptive_grp.addRow("Adaptive Control Layer:", self.chk_adaptive_ctrl)

        # Operator Scheduler
        self.chk_operator_scheduler = QCheckBox("aktiv")
        self.chk_operator_scheduler.setChecked(False)
        self.chk_operator_scheduler.setToolTip(
            "Aktiviert den adaptiven Operator-Scheduler.\n"
            "Mutationsgewichte (Add-Node, Remove, Rewire usw.) werden anhand\n"
            "von Erfolgsraten pro Operator und Species automatisch dosiert.\n"
            "Pruning-Druck steigt bei Komplexitäts-Wachstum + Stagnation.")
        adaptive_grp.addRow("Operator Scheduler:", self.chk_operator_scheduler)

        self.chk_meta_adaptive = QCheckBox("aktiv")
        self.chk_meta_adaptive.setChecked(False)
        self.chk_meta_adaptive.setToolTip(
            "Evolviert Policy-Gene für Operator-Exploration, Lamarck-Budget\n"
            "und Interspecies-Rate global und pro Species.")
        adaptive_grp.addRow("Meta-adaptive policies:", self.chk_meta_adaptive)

        self.chk_module_library = QCheckBox("aktiv")
        self.chk_module_library.setChecked(False)
        self.chk_module_library.setToolTip(
            "Speichert gute Hidden-Module in einer Bibliothek und fügt sie\n"
            "mit einer kleinen Mutationsrate in Offspring ein.")
        self.dspin_module_insert_rate = QDoubleSpinBox()
        self.dspin_module_insert_rate.setRange(0.0, 1.0)
        self.dspin_module_insert_rate.setSingleStep(0.01)
        self.dspin_module_insert_rate.setDecimals(3)
        self.dspin_module_insert_rate.setValue(0.02)
        module_row = QWidget()
        module_lay = QHBoxLayout(module_row)
        module_lay.setContentsMargins(0, 0, 0, 0)
        module_lay.setSpacing(4)
        module_lay.addWidget(self.chk_module_library)
        module_lay.addWidget(QLabel("insert"))
        module_lay.addWidget(self.dspin_module_insert_rate)
        adaptive_grp.addRow("Module library:", module_row)

        # Lamarck budget
        self.spin_lamarck_budget = QSpinBox()
        self.spin_lamarck_budget.setRange(0, 10000)
        self.spin_lamarck_budget.setValue(0)
        self.spin_lamarck_budget.setSpecialValueText("—")
        self.spin_lamarck_budget.setToolTip(
            "Maximale Lamarck-Evaluierungen pro Generation.\n"
            "0 = unbegrenzt (Standard).\n"
            "z.B. 50 = nach 50 Lamarck-Schritten werden weitere Genome\n"
            "in dieser Generation nicht verfeinert.\n"
            "Verhindert, dass Lamarck das gesamte Evaluierungsbudget verbraucht.")
        self.lbl_lamarck_budget_used = _label("—", "mutRate")
        self.lbl_lamarck_budget_used.setToolTip("Verwendete Lamarck-Schritte in der aktuellen Generation.")
        lamarck_budget_row = QWidget()
        lamarck_budget_lay = QHBoxLayout(lamarck_budget_row)
        lamarck_budget_lay.setContentsMargins(0, 0, 0, 0)
        lamarck_budget_lay.setSpacing(4)
        lamarck_budget_lay.addWidget(self.spin_lamarck_budget)
        lamarck_budget_lay.addWidget(QLabel("benutzt:"))
        lamarck_budget_lay.addWidget(self.lbl_lamarck_budget_used)
        adaptive_grp.addRow("Lamarck budget/gen:", lamarck_budget_row)

        # Adaptive Presets
        self.combo_adaptive_preset = QComboBox()
        self.combo_adaptive_preset.addItems([
            "Kein Preset",
            "Konservativ",
            "Balanciert",
            "Aggressiv",
            "Analysefreundlich",
        ])
        self.combo_adaptive_preset.setToolTip(
            "Adaptive Profile:\n\n"
            "Konservativ: Interspecies-Crossover fix, kein Operator-Scheduler.\n"
            "  Gut für stabile, langsame Tasks.\n\n"
            "Balanciert: Adaptiver Interspecies-Crossover (0.01–0.15),\n"
            "  Operator-Scheduler aktiv, moderate Pruning-Pressung.\n\n"
            "Aggressiv: Breite Interspecies-Rate (0.02–0.30),\n"
            "  starker Operator-Scheduler, hoher QD-Druck bei Plateau.\n\n"
            "Analysefreundlich: Alle adaptiven Features aktiv,\n"
            "  aber konservative Grenzen für reproduzierbare Läufe.")
        self.combo_adaptive_preset.currentIndexChanged.connect(self._on_adaptive_preset_changed)
        adaptive_grp.addRow("Adaptives Profil:", self.combo_adaptive_preset)

        # Adaptive signals display
        self.lbl_plateau_ratio = _label("—", "mutRate")
        self.lbl_plateau_ratio.setToolTip("Plateau-Ratio: Stagnation_count / Stagnation_threshold. 1.0 = volle Stagnation.")
        self.lbl_diversity_score = _label("—", "mutRate")
        self.lbl_diversity_score.setToolTip("Diversity-Score: Durchschnittliche Novelty der Population.")
        signals_row = QWidget()
        signals_lay = QHBoxLayout(signals_row)
        signals_lay.setContentsMargins(0, 0, 0, 0)
        signals_lay.setSpacing(6)
        signals_lay.addWidget(QLabel("Plateau:"))
        signals_lay.addWidget(self.lbl_plateau_ratio)
        signals_lay.addWidget(QLabel("  Diversity:"))
        signals_lay.addWidget(self.lbl_diversity_score)
        signals_lay.addStretch()
        adaptive_grp.addRow("Adaptive Signale:", signals_row)

        layout.addWidget(adaptive_grp)

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
        self.spin_nodes.setValue(ex.max_nodes)
        self.spin_conns.setValue(ex.max_connections)
        self.spin_pop.setValue(ex.default_population)
        self.spin_species.setValue(ex.default_target_species)
        self.chk_fitness_shaping.setChecked(ex.default_fitness_shaping)
        if ex.default_lamarck_steps > 0:
            self.combo_lamarck_schedule.setCurrentText("Explizit")
            self.combo_lamarck_optimizer.setCurrentText("Hill-Climbing")
            self.spin_lamarck.setValue(ex.default_lamarck_steps)
            self.spin_lamarck.setEnabled(True)
        else:
            self.combo_lamarck_schedule.setCurrentText("Adaptiv")
            self.combo_lamarck_optimizer.setCurrentText("Hill-Climbing")
            self.spin_lamarck.setEnabled(False)
            self.spin_lamarck.setValue(5)
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
        self.combo_adaptive_preset.setCurrentText("Kein Preset")
        self._apply_adaptive_policies(ex.default_adaptive_policies)
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

    def _on_lamarck_mode_changed(self, index: int) -> None:
        self.spin_lamarck.setEnabled(self.combo_lamarck_schedule.currentText() == "Explizit")

    def _apply_config_dict(self, cfg: dict) -> None:
        """Apply a GUI config dict to widgets; used by presets and example defaults."""
        if "population_size" in cfg:
            self.spin_pop.setValue(int(cfg["population_size"]))
        if "target_species" in cfg:
            self.spin_species.setValue(int(cfg["target_species"]))
        if "n_workers" in cfg:
            self.spin_workers.setValue(int(cfg["n_workers"]))
        if "multi_eval" in cfg:
            self.spin_multi_eval.setValue(int(cfg["multi_eval"]))
        if "aggregation" in cfg:
            self.combo_aggregation.setCurrentText(str(cfg["aggregation"]))
        if "sigma_penalty" in cfg:
            self.dspin_sigma_penalty.setValue(float(cfg["sigma_penalty"]))
        if "fitness_shaping" in cfg:
            self.chk_fitness_shaping.setChecked(bool(cfg["fitness_shaping"]))
        if "multi_objective" in cfg:
            self.chk_multi_objective.setChecked(bool(cfg["multi_objective"]))
        if "multi_objective_complexity_weight" in cfg:
            self.dspin_mo_complexity.setValue(float(cfg["multi_objective_complexity_weight"]))
        if "quality_diversity" in cfg:
            self.chk_quality_diversity.setChecked(bool(cfg["quality_diversity"]))
        if "quality_diversity_descriptor" in cfg:
            self.combo_qd_descriptor.setCurrentText(str(cfg["quality_diversity_descriptor"]))
        if "fitness_components" in cfg:
            self.chk_fitness_components.setChecked(bool(cfg["fitness_components"]))
        if "fitness_component_mode" in cfg:
            self.combo_fitness_component_mode.setCurrentText(str(cfg["fitness_component_mode"]))
        if "matrix_forward" in cfg:
            self.chk_matrix_forward.setChecked(bool(cfg["matrix_forward"]))
        if "cppn_substrate" in cfg:
            self.chk_cppn_substrate.setChecked(bool(cfg["cppn_substrate"]))
        if "cppn_hidden" in cfg:
            self.spin_cppn_hidden.setValue(int(cfg["cppn_hidden"]))
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
        if "novelty" in cfg:
            self.chk_novelty.setChecked(bool(cfg["novelty"]))
        if "speciation" in cfg:
            self.chk_speciation.setChecked(bool(cfg["speciation"]))
        if "crossover" in cfg:
            self.chk_crossover.setChecked(bool(cfg["crossover"]))
        if "diversity_injection" in cfg:
            self.chk_diversity_injection.setChecked(bool(cfg["diversity_injection"]))
        if "convergence_spread" in cfg:
            self.dspin_convergence_spread.setValue(float(cfg["convergence_spread"]))
        if "convergence_stagnation" in cfg:
            self.dspin_convergence_stagnation.setValue(float(cfg["convergence_stagnation"]))
        if "early_stop_factor" in cfg:
            self.dspin_early_stop.setValue(float(cfg["early_stop_factor"]))
        if "efficiency_max_ms" in cfg:
            self.spin_efficiency_max_ms.setValue(float(cfg["efficiency_max_ms"]))
        if "efficiency_penalty" in cfg:
            self.dspin_efficiency_penalty.setValue(float(cfg["efficiency_penalty"]))
        if "elite_global" in cfg:
            self.spin_elite_global.setValue(int(cfg["elite_global"]))
        if "elite_species" in cfg:
            self.spin_elite_species.setValue(int(cfg["elite_species"]))
        if "normalize" in cfg and self.chk_normalize.isVisible():
            self.chk_normalize.setChecked(bool(cfg["normalize"]))
        if "memory" in cfg:
            self.chk_memory.setChecked(bool(cfg["memory"]))
        if "curriculum" in cfg and self.chk_curriculum.isVisible():
            self.chk_curriculum.setChecked(bool(cfg["curriculum"]))
        if "lamarck_schedule" in cfg:
            self.combo_lamarck_schedule.setCurrentText(str(cfg["lamarck_schedule"]))
        if "lamarck_optimizer" in cfg:
            self.combo_lamarck_optimizer.setCurrentText(str(cfg["lamarck_optimizer"]))
        if "lamarck_steps" in cfg:
            self.spin_lamarck.setValue(int(cfg["lamarck_steps"]))
        self._on_lamarck_mode_changed(self.combo_lamarck_schedule.currentIndex())
        self._on_multi_eval_changed(self.spin_multi_eval.value())

    def _on_preset_changed(self, index: int) -> None:
        preset = self._preset_by_index.get(index)
        if preset is None:
            return
        self._apply_config_dict(preset.config)
        if preset.adaptive_policies:
            self._apply_adaptive_policies(preset.adaptive_policies)

    def _current_preset_config(self) -> dict:
        return {
            "population_size": self.spin_pop.value(),
            "target_species": self.spin_species.value(),
            "n_workers": self.spin_workers.value(),
            "multi_eval": self.spin_multi_eval.value(),
            "aggregation": self.combo_aggregation.currentText(),
            "sigma_penalty": self.dspin_sigma_penalty.value(),
            "fitness_shaping": self.chk_fitness_shaping.isChecked(),
            "multi_objective": self.chk_multi_objective.isChecked(),
            "multi_objective_complexity_weight": self.dspin_mo_complexity.value(),
            "quality_diversity": self.chk_quality_diversity.isChecked(),
            "quality_diversity_descriptor": self.combo_qd_descriptor.currentText(),
            "fitness_components": self.chk_fitness_components.isChecked(),
            "fitness_component_mode": self.combo_fitness_component_mode.currentText(),
            "matrix_forward": self.chk_matrix_forward.isChecked(),
            "cppn_substrate": self.chk_cppn_substrate.isChecked(),
            "cppn_hidden": self.spin_cppn_hidden.value(),
            "remote_eval": self.chk_remote_eval.isChecked(),
            "remote_urls": self.edit_remote_urls.text(),
            "remote_timeout_s": self.dspin_remote_timeout.value(),
            "remote_retries": self.spin_remote_retries.value(),
            "remote_batch_size": self.spin_remote_batch.value(),
            "novelty": self.chk_novelty.isChecked(),
            "speciation": self.chk_speciation.isChecked(),
            "crossover": self.chk_crossover.isChecked(),
            "diversity_injection": self.chk_diversity_injection.isChecked(),
            "convergence_spread": self.dspin_convergence_spread.value(),
            "convergence_stagnation": self.dspin_convergence_stagnation.value(),
            "early_stop_factor": self.dspin_early_stop.value(),
            "efficiency_max_ms": self.spin_efficiency_max_ms.value(),
            "efficiency_penalty": self.dspin_efficiency_penalty.value(),
            "elite_global": self.spin_elite_global.value(),
            "elite_species": self.spin_elite_species.value(),
            "normalize": self.chk_normalize.isChecked(),
            "memory": self.chk_memory.isChecked(),
            "curriculum": self.chk_curriculum.isChecked(),
            "lamarck_schedule": self.combo_lamarck_schedule.currentText(),
            "lamarck_optimizer": self.combo_lamarck_optimizer.currentText(),
            "lamarck_steps": self.spin_lamarck.value(),
        }

    def _current_adaptive_policies(self) -> dict:
        return {
            "adaptive_controller": self.chk_adaptive_ctrl.isChecked(),
            "operator_scheduler": self.chk_operator_scheduler.isChecked(),
            "interspecies_mode": self.combo_interspecies_mode.currentText(),
            "interspecies_min_rate": self.dspin_interspecies.value(),
            "interspecies_max_rate": self.dspin_interspecies_max.value(),
            "lamarck_schedule": self.combo_lamarck_schedule.currentText(),
            "lamarck_optimizer": self.combo_lamarck_optimizer.currentText(),
            "lamarck_budget": self.spin_lamarck_budget.value(),
            "meta_adaptive": self.chk_meta_adaptive.isChecked(),
            "module_library": self.chk_module_library.isChecked(),
            "module_insert_rate": self.dspin_module_insert_rate.value(),
        }

    def _apply_adaptive_policies(self, ap: dict) -> None:
        """Apply an adaptive_policies dict to the adaptive control widgets."""
        if "adaptive_controller" in ap:
            self.chk_adaptive_ctrl.setChecked(bool(ap["adaptive_controller"]))
        if "operator_scheduler" in ap:
            self.chk_operator_scheduler.setChecked(bool(ap["operator_scheduler"]))
        if "interspecies_mode" in ap:
            self.combo_interspecies_mode.setCurrentText(str(ap["interspecies_mode"]))
        if "interspecies_min_rate" in ap:
            self.dspin_interspecies.setValue(float(ap["interspecies_min_rate"]))
        if "interspecies_max_rate" in ap:
            self.dspin_interspecies_max.setValue(float(ap["interspecies_max_rate"]))
        if "lamarck_schedule" in ap:
            self.combo_lamarck_schedule.setCurrentText(str(ap["lamarck_schedule"]))
        if "lamarck_optimizer" in ap:
            self.combo_lamarck_optimizer.setCurrentText(str(ap["lamarck_optimizer"]))
        if "lamarck_budget" in ap:
            self.spin_lamarck_budget.setValue(int(ap["lamarck_budget"]))
        if "meta_adaptive" in ap:
            self.chk_meta_adaptive.setChecked(bool(ap["meta_adaptive"]))
        if "module_library" in ap:
            self.chk_module_library.setChecked(bool(ap["module_library"]))
        if "module_insert_rate" in ap:
            self.dspin_module_insert_rate.setValue(float(ap["module_insert_rate"]))

    def _save_current_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        if not ok or not name.strip():
            return
        path = save_preset(
            name.strip(),
            self._current_preset_config(),
            "Saved from GUI",
            adaptive_policies=self._current_adaptive_policies(),
        )
        preset = load_preset(path)
        idx = self.preset_combo.count()
        self._preset_by_index[idx] = preset
        self.preset_combo.addItem(preset.name, preset)
        self.preset_combo.setCurrentIndex(idx)
        self.status_lbl.setText(f"Preset saved: {path.name}")

    def _on_adaptive_preset_changed(self, index: int) -> None:
        preset_name = self.combo_adaptive_preset.currentText()
        if preset_name == "Konservativ":
            self.combo_interspecies_mode.setCurrentText("Fix")
            self.dspin_interspecies.setValue(0.05)
            self.chk_adaptive_ctrl.setChecked(False)
            self.chk_operator_scheduler.setChecked(False)
            self.spin_lamarck_budget.setValue(0)
        elif preset_name == "Balanciert":
            self.combo_interspecies_mode.setCurrentText("Adaptiv")
            self.dspin_interspecies.setValue(0.01)
            self.dspin_interspecies_max.setValue(0.15)
            self.chk_adaptive_ctrl.setChecked(True)
            self.chk_operator_scheduler.setChecked(True)
            self.spin_lamarck_budget.setValue(0)
        elif preset_name == "Aggressiv":
            self.combo_interspecies_mode.setCurrentText("Adaptiv")
            self.dspin_interspecies.setValue(0.02)
            self.dspin_interspecies_max.setValue(0.30)
            self.chk_adaptive_ctrl.setChecked(True)
            self.chk_operator_scheduler.setChecked(True)
            self.spin_lamarck_budget.setValue(0)
        elif preset_name == "Analysefreundlich":
            self.combo_interspecies_mode.setCurrentText("Adaptiv")
            self.dspin_interspecies.setValue(0.01)
            self.dspin_interspecies_max.setValue(0.20)
            self.chk_adaptive_ctrl.setChecked(True)
            self.chk_operator_scheduler.setChecked(True)
            self.spin_lamarck_budget.setValue(100)

    def _on_multi_eval_changed(self, value: int) -> None:
        enabled = value > 1
        self.combo_aggregation.setEnabled(enabled)
        self.dspin_sigma_penalty.setEnabled(enabled)

    def _on_render_toggled(self, checked: bool) -> None:
        self.btn_render.setText("Render: On" if checked else "Render: Off")
        self._render_group.setVisible(checked)
        if not checked:
            self._render_widget.clear_frame()

    def _current_research_feature_config(self) -> ResearchFeatureConfig:
        return ResearchFeatureConfig(
            n_inputs=self.spin_inputs.value(),
            n_outputs=self.spin_outputs.value(),
            max_nodes=self.spin_nodes.value() or None,
            max_connections=self.spin_conns.value() or None,
            population_size=self.spin_pop.value(),
            target_species=self.spin_species.value(),
            allow_memory=self.chk_memory.isChecked(),
            output_sanitize=self._yane._output_sanitize,
            output_fallback=self._yane._output_fallback,
            cppn_substrate=self.chk_cppn_substrate.isChecked(),
            cppn_hidden=self.spin_cppn_hidden.value(),
            matrix_forward=self.chk_matrix_forward.isChecked(),
            fitness_components=self.chk_fitness_components.isChecked(),
            fitness_component_mode=self.combo_fitness_component_mode.currentText(),
            meta_adaptive=self.chk_meta_adaptive.isChecked(),
            module_library=self.chk_module_library.isChecked(),
            module_insert_rate=self.dspin_module_insert_rate.value(),
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
            max_nodes=self.spin_nodes.value() or None,
            max_connections=self.spin_conns.value() or None,
            n_initial_hidden=ex.n_initial_hidden,
            stateful=self.chk_memory.isChecked(),
        )
        self._yane.set_population_size(self.spin_pop.value())
        self._yane.set_n_workers(self.spin_workers.value())
        self._yane.set_target_species(self.spin_species.value())
        research_cfg = self._current_research_feature_config()
        if research_cfg.cppn_substrate:
            configure_cppn_substrate_population(self._yane, research_cfg)
        return research_cfg

    def _apply_evolution_options(self, research_cfg: ResearchFeatureConfig) -> None:
        if self.chk_fitness_shaping.isChecked():
            self._yane.set_fitness_shaping(True)
        self._yane.set_novelty_search(self.chk_novelty.isChecked())
        self._yane.set_speciation(self.chk_speciation.isChecked())
        self._yane.set_crossover(self.chk_crossover.isChecked())
        self._yane.set_diversity_injection(self.chk_diversity_injection.isChecked())
        if self.combo_interspecies_mode.currentText() == "Adaptiv":
            self._yane.set_adaptive_interspecies_crossover(
                min_rate=self.dspin_interspecies.value(),
                max_rate=self.dspin_interspecies_max.value(),
            )
        else:
            self._yane.set_interspecies_crossover(self.dspin_interspecies.value())

        conv_eps = self.dspin_convergence_spread.value()
        if conv_eps > 0.0:
            self._yane.set_convergence_stop(conv_eps, self.dspin_convergence_stagnation.value())
        early_stop_factor = self.dspin_early_stop.value()
        if early_stop_factor > 0.0:
            self._yane.set_early_stopping(early_stop_factor)
        eff_max = self.spin_efficiency_max_ms.value()
        eff_pen = self.dspin_efficiency_penalty.value()
        if eff_max > 0.0 and eff_pen > 0.0:
            self._yane.set_efficiency_penalty(eff_max, eff_pen)
        self._yane.set_elitism(self.spin_elite_global.value(), self.spin_elite_species.value())
        self._apply_lamarck_options()
        self._yane.set_adaptive_control(self.chk_adaptive_ctrl.isChecked())
        self._yane.set_operator_scheduler(self.chk_operator_scheduler.isChecked())
        apply_research_features(self._yane, research_cfg)
        # New feature controls
        self._yane.set_weight_inheritance(enabled=self.chk_weight_inheritance.isChecked())
        self._yane.set_online_tuning(enabled=self.chk_online_tuning.isChecked())
        self._yane.set_checkpoint_codec(self.combo_codec.currentText())

        # Seed / Stopping / Logging
        n_iter = self.spin_max_iter.value()
        if n_iter > 0:
            self._yane.set_max_iterations(n_iter)
        n_eval_max = self.spin_max_eval.value()
        if n_eval_max > 0:
            self._yane.set_max_evaluations(n_eval_max)
        log_fmt = self.combo_log_format.currentText()
        if log_fmt != "csv":
            self._yane.set_log_format(log_fmt)

        # Island model
        if self.chk_island_model.isChecked():
            self._yane.set_island_model(
                n_islands=self.spin_n_islands.value(),
                migration_interval=self.spin_migrate_interval.value(),
                migration_count=self.spin_migrate_count.value(),
            )

        # Checkpoint policy
        if self.chk_auto_checkpoint.isChecked():
            self._yane.set_checkpoint_policy(
                interval=self.spin_ckpt_interval.value(),
                keep_best=True,
                max_keep=self.spin_ckpt_max_keep.value(),
                enabled=True,
            )

        # Adaptive population
        if self.chk_adaptive_pop.isChecked():
            self._yane.set_adaptive_population(
                min_size=self.spin_adaptive_pop_min.value(),
                max_size=self.spin_adaptive_pop_max.value(),
            )

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

    def _apply_lamarck_options(self) -> None:
        optimizer_map = {
            "Hill-Climbing": "hill_climbing",
            "NES": "nes",
            "SA": "sa",
            "CMA-ES": "cma_es",
        }
        optimizer = optimizer_map[self.combo_lamarck_optimizer.currentText()]
        schedule = self.combo_lamarck_schedule.currentText()
        if schedule == "Explizit":
            self._yane.set_lamarck(n_steps=self.spin_lamarck.value(), mode=optimizer)
        elif schedule == "Adaptiv":
            self._yane.set_lamarck_adaptive(mode=optimizer)
        elif schedule == "Aus":
            self._yane.set_lamarck_adaptive(max_steps=0)
        budget = self.spin_lamarck_budget.value()
        self._yane.set_lamarck_budget(budget if budget > 0 else None)

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

        if not self.chk_multi_objective.isChecked():
            return make_eval_fn

        weight = self.dspin_mo_complexity.value()
        self._yane.set_multi_objective(
            enabled=True,
            weights=(1.0, -weight),
            maximize=(True, False),
        )
        base_make_eval_fn = make_eval_fn

        def _make_mo_eval(render_cb=None, _base=base_make_eval_fn):
            base_eval = _base(render_cb)

            def _eval(genome):
                raw = base_eval(genome)
                return (raw, float(genome.connection_count))

            return _eval

        return _make_mo_eval

    def _configure_quality_diversity(self) -> None:
        if not self.chk_quality_diversity.isChecked():
            return
        if self.combo_qd_descriptor.currentText() == "Behavior":
            from yane.evolution.quality_diversity import descriptor_from_outputs
            import random
            rng = random.Random(42)
            probes = [
                [rng.uniform(-1.0, 1.0) for _ in range(self.spin_inputs.value())]
                for _ in range(2)
            ]
            bins = tuple(8 for _ in range(max(1, self.spin_outputs.value() * len(probes))))
            ranges = tuple((-1.0, 1.0) for _ in bins)
            self._yane.set_quality_diversity(
                descriptor_from_outputs(probes),
                bins=bins,
                ranges=ranges,
                max_cells=500,
            )
            return

        self._yane.set_quality_diversity(
            descriptor_fn=lambda g: (
                float(max(0, len(g.nodes) - len(g.input_nodes) - len(g.output_nodes))),
                float(g.connection_count),
            ),
            bins=(12, 16),
            ranges=((0.0, float(max(1, self.spin_nodes.value() or 100))),
                    (0.0, float(max(1, self.spin_conns.value() or 200)))),
            max_cells=500,
        )

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
            "GUI training started  example=%s  pop_size=%d  target=%s",
            ex.name,
            self.spin_pop.value(),
            self.dspin_target.value() if self.dspin_target.value() > -1e9 else "none",
        )
        self._log_csv_path = log_dir / "fitness_history.csv"
        self._log_csv_header = "generation,iteration,best_fitness,mean_fitness,median_fitness,iqr_fitness,species_count,stagnation_count,nodes,connections,validation_fitness"
        self._log_csv_interval = max(1, self.spin_pop.value() // 10)

    def start_training(self) -> None:
        ex = self._current_example()
        if ex is None:
            return

        try:
            research_cfg = self._configure_yane_core(ex)
            self._apply_evolution_options(research_cfg)
            render_cb = self._render_callback_for_example(ex)
            make_eval_fn = self._make_eval_factory(ex)
            remote_cfg = self._current_remote_config()
            self._configure_quality_diversity()
            self._configure_curriculum(ex)
            self._setup_gui_run_logging(ex)
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
        worker = TrainingWorker(
            self._yane,
            make_eval_fn,
            render_cb=render_cb,
            remote_config=remote_cfg,
        )
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
        if self._yane is not None:
            self.ne_ready.emit(self._yane)
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
        generation = mem.get("generation", iteration // max(1, self.spin_pop.value()))
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
        self._update_adaptive_labels(mem)

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

    def _update_adaptive_labels(self, mem: dict) -> None:
        """Update live display of adaptive control diagnostics."""
        try:
            # Interspecies crossover
            current_rate = mem.get("interspecies_crossover_current", None)
            if current_rate is not None:
                self.lbl_interspecies_live.setText(f"{current_rate:.3f}")
            reason = mem.get("interspecies_crossover_last_reason", "—")
            self.lbl_interspecies_trigger.setText(reason)

            # Cross-species success
            n_offspring = mem.get("interspecies_n_offspring", 0)
            n_improved = mem.get("interspecies_n_improved", 0)
            if n_offspring > 0:
                rate = n_improved / n_offspring
                self.lbl_interspecies_success.setText(
                    f"{rate:.1%}  ({n_improved}/{n_offspring})"
                )
            else:
                self.lbl_interspecies_success.setText("—")

            # Lamarck budget
            budget_used = mem.get("lamarck_budget_used", 0)
            budget_limit = mem.get("lamarck_budget_per_gen", None)
            if budget_limit is not None and budget_limit > 0:
                self.lbl_lamarck_budget_used.setText(f"{budget_used}/{budget_limit}")
            else:
                self.lbl_lamarck_budget_used.setText(f"{budget_used} (unbegrenzt)")

            # Adaptive signals from AdaptiveController
            ctrl = mem.get("adaptive_controller", {})
            signals = ctrl.get("signals", {}) if ctrl else {}
            plateau = signals.get("plateau_ratio", mem.get("plateau_ratio", 0.0))
            diversity = signals.get("diversity_score", 0.0)
            self.lbl_plateau_ratio.setText(f"{plateau:.2f}")
            self.lbl_diversity_score.setText(f"{diversity:.2f}")
        except Exception as _e:
            from yane.util.logger import log_warning
            log_warning("_update_adaptive_labels failed: %s", _e)

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
