"""Left panel: collapsible stats/controls sidebar shown alongside the tab area."""
from __future__ import annotations

import json

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QPushButton, QFrame, QLabel, QSizePolicy, QFormLayout,
    QFileDialog, QMessageBox,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from yane.gui.canvas import (
    NetworkCanvas, FitnessHistogram, MutationRateChart, WeightHistogram,
    ParetoScatter, MapElitesHeatmap,
)
from yane.gui._helpers import _label, _divider, CollapsibleGroup
from yane.gui.research_features import format_research_diagnostics

_CollapsibleGroup = CollapsibleGroup  # local alias kept for backward compat


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
        best = _CollapsibleGroup("Best Genome", collapsed=False)
        self.lbl_nodes        = _label("—", "statValue")
        self.lbl_connections  = _label("—", "statValue")
        self.lbl_best_fit     = _label("—", "statValue")
        self.lbl_shared_fit   = _label("—", "statValue")
        self.lbl_efficiency   = _label("—", "statValue")
        self.lbl_activations  = _label("—", "mutRate")
        self.lbl_activations.setWordWrap(True)
        self.lbl_nodes.setToolTip(
            "Gesamtzahl der Knoten im besten Netz\n(Input-Nodes + Hidden-Nodes + Output-Nodes)")
        self.lbl_connections.setToolTip(
            "Anzahl der gerichteten Verbindungen (Synapsen) im besten Netz.\n"
            "Jede Verbindung hat ein evolviertes Gewicht.")
        self.lbl_best_fit.setToolTip(
            "Rohfitness des besten Genoms — der Wert den deine evaluate()-Funktion zurückgibt.\n"
            "Wird nie geteilt oder angepasst.")
        self.lbl_shared_fit.setToolTip(
            "Geteilte Fitness = Rohfitness ÷ Anzahl Genomes in derselben Art (fitness sharing).\n"
            "Verhindert dass eine dominante Art alle anderen verdrängt.\n"
            "Wird für die Selektion verwendet.")
        self.lbl_efficiency.setToolTip(
            "Relative Auswertungsgeschwindigkeit des besten Genoms.\n"
            "1.0 = schnellstes gemessenes Genom in der aktuellen Population,\n"
            "0.0 = langsamstes. Die Rohfitness bleibt davon unverändert.")
        self.lbl_activations.setToolTip(
            "Aktivierungsfunktionen aller Nodes im besten Netz (Kürzel:Anzahl).\n"
            "SIG=Sigmoid  LIN=Linear  TAN=Tanh  REL=ReLU  SQU=Square (x²)\n"
            "ABS=Betrag   GAU=Gauß    CUB=Kubik  SIN=Sinus  COS=Kosinus\n"
            "ELU=ELU  SWI=Swish  SOF=Softplus  LEA=Leaky ReLU  BIN=Binär\n"
            "Zeigt was das Netz selbst entdeckt hat.")
        best.addRow("Nodes:",          self.lbl_nodes)
        best.addRow("Connections:",    self.lbl_connections)
        best.addRow("Fitness:",        self.lbl_best_fit)
        best.addRow("Shared fitness:", self.lbl_shared_fit)
        best.addRow("Efficiency:",     self.lbl_efficiency)
        best.addRow("Activations:",    self.lbl_activations)
        layout.addWidget(best)

        # ── Population ────────────────────────────────────────────────────
        pop = _CollapsibleGroup("Population", collapsed=False)
        self.lbl_population  = _label("—", "statValue")
        self.lbl_species     = _label("—", "statValue")
        self.lbl_avg_nodes   = _label("—", "statValue")
        self.lbl_avg_conns   = _label("—", "statValue")
        self.lbl_pop_min     = _label("—", "mutRate")
        self.lbl_pop_avg     = _label("—", "mutRate")
        self.lbl_top5_avg    = _label("—", "mutRate")
        self.lbl_pop_max     = _label("—", "mutRate")
        self.lbl_stagnation      = _label("—", "statValue")
        self.lbl_next_injection  = _label("—", "statValue")
        self.lbl_novelty_weight  = _label("—", "statValue")
        self.lbl_eff_weight      = _label("—", "statValue")
        self.lbl_population.setToolTip("Gesamtzahl der Genomes (evaluiert + noch ausstehend)")
        self.lbl_species.setToolTip(
            "Anzahl der Arten (Species).\n"
            "Strukturell ähnliche Genomes werden gruppiert — Innovation ist\n"
            "innerhalb einer Art geschützt, bevor sie gegen andere antritt.")
        self.lbl_avg_nodes.setToolTip("Durchschnittliche Knotenanzahl über alle Genomes der Population")
        self.lbl_avg_conns.setToolTip("Durchschnittliche Verbindungsanzahl über alle Genomes der Population")
        _spread_tip = ("Fitnessspanne der gesamten Population.\n"
                       "Kleine Spanne = Population konvergiert.\n"
                       "Große Spanne = viel Diversität.")
        self.lbl_pop_min.setToolTip(_spread_tip)
        self.lbl_pop_avg.setToolTip(_spread_tip)
        self.lbl_top5_avg.setToolTip(
            "Durchschnittliche Fitness der Top-5-Genome (nach Fitness sortiert).\n"
            "Zeigt die Leistung der Elite unabhängig vom Populations-Durchschnitt.")
        self.lbl_pop_max.setToolTip(_spread_tip)
        self.lbl_stagnation.setToolTip(
            "Wie viele Iterationen seit der letzten Fitness-Verbesserung.\n"
            "Bei Stagnation wird automatisch Diversität in die Population injiziert\n"
            "(neue zufällige Genomes oder Mutationen des besten).")
        self.lbl_next_injection.setToolTip(
            "Fortschritt bis zur nächsten Diversitäts-Injektion.\n"
            "Format: since / threshold  (noch verbleibende Iterationen)")
        self.lbl_novelty_weight.setToolTip(
            "Wie stark Neuartigkeit (novelty) bei der Selektion gewichtet wird.\n"
            "Steigt von 0.1 → 0.5 je länger die Population stagniert.\n"
            "Zwingt Evolution neue Verhaltensweisen zu erkunden statt zu optimieren.")
        self.lbl_eff_weight.setToolTip(
            "Wie stark Effizienz aktuell in die Elternauswahl einfließt.\n"
            "Hoch bei Fortschritt, niedrig bei Stagnation.\n"
            "Dadurch werden schnelle Netze bevorzugt, solange die Aufgabe gut vorankommt.")
        self.lbl_invalid_fitness = _label("—", "mutRate")
        self.lbl_clipped_fitness = _label("—", "mutRate")
        self.lbl_invalid_fitness.setToolTip(
            "Anzahl Fitnesswerte, die nan oder inf waren und durch den Fallback-Wert\n"
            "ersetzt wurden (nur aktiv wenn set_fitness_sanitizing() gesetzt ist).")
        self.lbl_clipped_fitness.setToolTip(
            "Anzahl Fitnesswerte, die am konfigurierten clip_low / clip_high begrenzt wurden.")
        pop.addRow("Genomes:",           self.lbl_population)
        pop.addRow("Species:",           self.lbl_species)
        pop.addRow("Avg nodes:",         self.lbl_avg_nodes)
        pop.addRow("Avg conns:",         self.lbl_avg_conns)
        pop.addRow("Min fitness:",       self.lbl_pop_min)
        pop.addRow("Avg fitness:",       self.lbl_pop_avg)
        pop.addRow("Top-5 avg:",         self.lbl_top5_avg)
        pop.addRow("Max fitness:",       self.lbl_pop_max)
        pop.addRow("Stagnation (total):", self.lbl_stagnation)
        pop.addRow("Next injection:",    self.lbl_next_injection)
        pop.addRow("Novelty weight:",    self.lbl_novelty_weight)
        pop.addRow("Efficiency weight:", self.lbl_eff_weight)
        pop.addRow("Invalid fitness:",   self.lbl_invalid_fitness)
        pop.addRow("Clipped fitness:",   self.lbl_clipped_fitness)
        self.lbl_plateau = _label("—", "mutRate")
        self.lbl_plateau.setToolTip(
            "Plateau-Verhältnis: stagnation_count / stagnation_threshold.\n"
            "0.0 = frischer Fortschritt, 1.0 = voll stagniert.\n"
            "Zeigt, wie nah die Population an einem Plateau ist.")
        pop.addRow("Plateau ratio:",     self.lbl_plateau)
        self.lbl_jump_rate = _label("—", "mutRate")
        self.lbl_jump_rate.setToolTip(
            "Anteil der Evaluierungen, bei denen ein neues Fitness-Maximum erreicht wurde.\n"
            "Hoher Wert = Population verbessert sich noch schnell.\n"
            "Niedriger Wert = Optimierungsfortschritt verlangsamt sich.")
        pop.addRow("Jump rate:",         self.lbl_jump_rate)
        self.lbl_pop_size_adjustments = _label("—", "mutRate")
        self.lbl_pop_size_adjustments.setToolTip(
            "Wie oft die Populationsgröße durch Adaptive Population Sizing angepasst wurde\n"
            "(nur aktiv wenn set_adaptive_population() gesetzt ist).")
        pop.addRow("Pop size adj.:",     self.lbl_pop_size_adjustments)
        self.lbl_multi_objective = _label("—", "mutRate")
        self.lbl_multi_objective.setToolTip(
            "Multi-Objective-Status. Wenn aktiv, nutzt die Selektion Pareto-Rang\n"
            "und Crowding-Distance statt nur einer skalarisierten Fitness.")
        pop.addRow("Multi-objective:", self.lbl_multi_objective)
        self.fitness_hist = FitnessHistogram()
        self.fitness_hist.setToolTip(
            "Fitness-Verteilung der evaluierten Population.\n"
            "Zeigt, ob die Fitness breit gestreut (Exploration)\n"
            "oder eng gebündelt (Konvergenz) ist.")
        pop.addRow("Fitness dist:",      self.fitness_hist)
        layout.addWidget(pop)

        # ── Lamarck ───────────────────────────────────────────────────────
        lamarck_grp = _CollapsibleGroup("Lamarck Refinement", collapsed=True)
        self.lbl_lamarck_mode    = _label("—", "statValue")
        self.lbl_lamarck_applied = _label("—", "mutRate")
        self.lbl_lamarck_steps   = _label("—", "mutRate")
        self.lbl_lamarck_mode.setToolTip(
            "Aktueller Lamarck-Modus:\n"
            "  adaptive      — Hill-Climbing, automatisch bei Stagnation\n"
            "  explicit      — Hill-Climbing, feste Schrittzahl vor jeder Eval\n"
            "  nes_explicit  — NES (antithetische Paare), immer aktiv\n"
            "  nes_adaptive  — NES, automatisch bei Stagnation\n"
            "  sa_explicit   — Simulated Annealing, immer aktiv\n"
            "  sa_adaptive   — Simulated Annealing, automatisch bei Stagnation\n"
            "  cma_es_explicit/adaptive — volle-Kovarianz-CMA-ES\n"
            "  off           — deaktiviert")
        self.lbl_lamarck_applied.setToolTip(
            "Anzahl der Genome, für die Lamarck-Verfeinerung durchgeführt wurde\n"
            "(kumulativ seit Trainingsstart).")
        self.lbl_lamarck_steps.setToolTip(
            "Gesamtzahl der Verfeinerungsschritte (kumulativ).\n"
            "Hill-Climbing: 1 Eval/Schritt.  NES: 2k+1 Evals/Schritt.  SA: 1 Eval/Schritt.")
        lamarck_grp.addRow("Mode:",        self.lbl_lamarck_mode)
        lamarck_grp.addRow("Applied:",     self.lbl_lamarck_applied)
        lamarck_grp.addRow("Steps total:", self.lbl_lamarck_steps)
        self.lbl_lamarck_per_species = _label("—", "mutRate")
        self.lbl_lamarck_per_species.setToolTip(
            "Mittlere Lamarck-Steps pro Species (nur Species mit >0 Mitgliedern).\n"
            "Zeigt, ob die Lamarck-Last gleichmäßig oder auf wenige Species konzentriert ist.")
        lamarck_grp.addRow("Ø steps/species:", self.lbl_lamarck_per_species)
        self.lbl_lamarck_blocked = _label("—", "mutRate")
        self.lbl_lamarck_blocked.setToolTip(
            "Anzahl der Fälle, in denen Lamarck durch den top-k-Filter geblockt wurde.\n"
            "Ein hoher Wert bei null 'Applied' deutet auf ein zu enges top-k-Gate hin —\n"
            "typisch bei stochastischen Umgebungen mit n_evaluations=1.\n"
            "Abhilfe: lamarck_top_k erhöhen oder n_evaluations > 1 setzen.")
        lamarck_grp.addRow("Blocked (top-k):", self.lbl_lamarck_blocked)
        self.lbl_lamarck_time = _label("—", "mutRate")
        self.lbl_lamarck_time.setToolTip(
            "Kumulative Zeit in Lamarck-Refinement (ms).\n"
            "Zeigt den Overhead des Hill-Climbings.")
        lamarck_grp.addRow("Time total (ms):", self.lbl_lamarck_time)
        self.lbl_lamarck_sigma = _label("—", "mutRate")
        self.lbl_lamarck_sigma.setToolTip(
            "Lamarck-Sigma des besten Genoms — evolvierbarer Schrittweitenparameter\n"
            "unabhängig von sigma_global. Steuert, wie groß die Gewichtsstörungen\n"
            "beim Hill-Climbing ausfallen.")
        lamarck_grp.addRow("Best σ_lamarck:", self.lbl_lamarck_sigma)
        layout.addWidget(lamarck_grp)

        # ── Sicherheitslimits ─────────────────────────────────────────────
        safe_grp = _CollapsibleGroup("Sicherheitslimits", collapsed=True)
        self.lbl_n_weight_clipped = _label("—", "mutRate")
        self.lbl_n_bias_clipped   = _label("—", "mutRate")
        self.lbl_n_output_sanitized = _label("—", "mutRate")
        self.lbl_n_weight_clipped.setToolTip(
            "Kumulative Anzahl der Gewichte, die durch Weight-Clipping beschnitten wurden\n"
            "(nur aktiv wenn set_weight_clipping() gesetzt ist).")
        self.lbl_n_bias_clipped.setToolTip(
            "Kumulative Anzahl der Biases, die durch Weight-Clipping beschnitten wurden.")
        self.lbl_n_output_sanitized.setToolTip(
            "Kumulative Anzahl der NaN/Inf-Ausgaben, die durch Output-Sanitizing ersetzt wurden\n"
            "(nur aktiv wenn set_output_sanitizing() gesetzt ist).")
        safe_grp.addRow("W clipped:",       self.lbl_n_weight_clipped)
        safe_grp.addRow("B clipped:",       self.lbl_n_bias_clipped)
        safe_grp.addRow("Output sanitized:", self.lbl_n_output_sanitized)
        layout.addWidget(safe_grp)

        # ── Quality Diversity ─────────────────────────────────────────────
        self._last_qd_cells: list[dict] | None = None
        qd_grp = _CollapsibleGroup("Quality Diversity", collapsed=True)
        self.lbl_qd_enabled = _label("—", "statValue")
        self.lbl_qd_cells = _label("—", "mutRate")
        self.lbl_qd_coverage = _label("—", "mutRate")
        self.lbl_qd_updates = _label("—", "mutRate")
        self.lbl_qd_injections = _label("—", "mutRate")
        self.lbl_qd_enabled.setToolTip("Status des MAP-Elites-Archivs.")
        self.lbl_qd_cells.setToolTip("Anzahl belegter Archivzellen.")
        self.lbl_qd_coverage.setToolTip("Anteil belegter Zellen relativ zum Descriptor-Gitter.")
        self.lbl_qd_updates.setToolTip("Wie oft ein Archiv-Cell-Elite neu gesetzt oder ersetzt wurde.")
        self.lbl_qd_injections.setToolTip("Wie oft ein Archiv-Elite als Diversity-Injektion genutzt wurde.")
        qd_grp.addRow("Enabled:", self.lbl_qd_enabled)
        qd_grp.addRow("Cells:", self.lbl_qd_cells)
        qd_grp.addRow("Coverage:", self.lbl_qd_coverage)
        qd_grp.addRow("Updates:", self.lbl_qd_updates)
        qd_grp.addRow("Injections:", self.lbl_qd_injections)
        self.qd_heatmap = MapElitesHeatmap()
        qd_grp.addWidget(self.qd_heatmap)
        self.btn_export_qd = QPushButton("Export QD Archive (JSON)")
        self.btn_export_qd.setToolTip("Aktuelles MAP-Elites-Archiv als JSON-Datei speichern.")
        self.btn_export_qd.clicked.connect(self._export_qd_archive)
        qd_grp.addWidget(self.btn_export_qd)
        layout.addWidget(qd_grp)

        pareto_grp = _CollapsibleGroup("Pareto Front", collapsed=True)
        self.pareto_scatter = ParetoScatter()
        pareto_grp.addWidget(self.pareto_scatter)
        layout.addWidget(pareto_grp)

        research_grp = _CollapsibleGroup("Research Features", collapsed=True)
        self.lbl_matrix_forward = _label("—", "mutRate")
        self.lbl_fitness_components = _label("—", "mutRate")
        self.lbl_meta_policy = _label("—", "mutRate")
        self.lbl_module_library = _label("—", "mutRate")
        self.lbl_matrix_forward.setToolTip("Matrix-Forward Hits/Misses für kompatible Genome.")
        self.lbl_fitness_components.setToolTip("Aktuelle adaptive/fixe Fitness-Komponenten-Gewichte.")
        self.lbl_meta_policy.setToolTip("Meta-adaptive Policy-Gene und letzter Update-Grund.")
        self.lbl_module_library.setToolTip("Modulbibliothek: gespeicherte Module und Wiederverwendung.")
        research_grp.addRow("Matrix forward:", self.lbl_matrix_forward)
        research_grp.addRow("Fitness comps:", self.lbl_fitness_components)
        research_grp.addRow("Meta policy:", self.lbl_meta_policy)
        research_grp.addRow("Module library:", self.lbl_module_library)
        layout.addWidget(research_grp)

        # ── Curriculum ────────────────────────────────────────────────────
        cur_grp = _CollapsibleGroup("Curriculum", collapsed=False)
        self.lbl_cur_stage  = _label("—", "statValue")
        self.lbl_cur_name   = _label("—", "mutRate")
        self.lbl_cur_target = _label("—", "mutRate")
        self.lbl_cur_best   = _label("—", "mutRate")
        self.lbl_cur_stage.setToolTip(
            "Aktuelle Stufe des Curriculums (Index / Gesamt).\n"
            "Beim Stufenwechsel wird die Population neu bewertet.")
        self.lbl_cur_name.setToolTip("Name der aktuellen Stufe.")
        self.lbl_cur_target.setToolTip(
            "Ziel-Fitness der aktuellen Stufe — bei Erreichen wird\n"
            "automatisch zur nächsten Stufe gewechselt.")
        self.lbl_cur_best.setToolTip(
            "Bestes Fitness-Ergebnis in der aktuellen Stufe\n"
            "(nur Genome gezählt, die auf dieser Stufe bewertet wurden).")
        cur_grp.addRow("Stage:",  self.lbl_cur_stage)
        cur_grp.addRow("Name:",   self.lbl_cur_name)
        cur_grp.addRow("Target:", self.lbl_cur_target)
        cur_grp.addRow("Best:",   self.lbl_cur_best)
        self._cur_grp = cur_grp
        cur_grp.setVisible(False)   # only shown when curriculum is active
        layout.addWidget(cur_grp)

        # ── Evaluation timing ─────────────────────────────────────────────
        eval_grp = _CollapsibleGroup("Evaluation timing", collapsed=True)
        self.lbl_eval_mean   = _label("—", "statValue")
        self.lbl_eval_median = _label("—", "mutRate")
        self.lbl_eval_p95    = _label("—", "mutRate")
        self.lbl_eval_max    = _label("—", "mutRate")
        _eval_tip = (
            "Auswertungszeiten über alle aktuell evaluierten Genomes.\n"
            "Bei Mehrfachbewertung (n_evaluations > 1) ist dies die Summe aller Durchläufe.")
        self.lbl_eval_mean.setToolTip(_eval_tip)
        self.lbl_eval_median.setToolTip(_eval_tip)
        self.lbl_eval_p95.setToolTip(
            "95. Perzentil der Eval-Zeiten — zeigt die obere Grenze für 95% der Genomes.")
        self.lbl_eval_max.setToolTip(
            "Langsamste Eval-Zeit in der Population — möglicher Performance-Ausreißer.")
        eval_grp.addRow("Mean:",   self.lbl_eval_mean)
        eval_grp.addRow("Median:", self.lbl_eval_median)
        eval_grp.addRow("p95:",    self.lbl_eval_p95)
        eval_grp.addRow("Max:",    self.lbl_eval_max)
        layout.addWidget(eval_grp)

        # ── Offspring counters ────────────────────────────────────────────
        off_grp = _CollapsibleGroup("Offspring (cumulative)", collapsed=True)
        self.lbl_n_crossover  = _label("—", "mutRate")
        self.lbl_n_mutation   = _label("—", "mutRate")
        self.lbl_n_injection  = _label("—", "mutRate")
        self.lbl_interspecies_rate = _label("—", "mutRate")
        self.lbl_interspecies_rate.setToolTip(
            "Live-Rate fuer Interspecies-Crossover. Im adaptiven Modus wird sie\n"
            "aus Stagnation und Species-Druck abgeleitet.")
        self.lbl_n_crossover.setToolTip(
            "Anzahl Offspring, die durch Kreuzung zweier Elternteile entstanden sind.")
        self.lbl_n_mutation.setToolTip(
            "Anzahl Offspring, die durch reine Mutation (Kopie + Mutation) entstanden sind.")
        self.lbl_n_injection.setToolTip(
            "Anzahl Genome, die per Diversitäts-Injektion (Stagnation oder Topologie-Stagnation)\n"
            "in die Population eingebracht wurden.")
        off_grp.addRow("Crossover:", self.lbl_n_crossover)
        off_grp.addRow("Mutation:",  self.lbl_n_mutation)
        off_grp.addRow("Injection:", self.lbl_n_injection)
        off_grp.addRow("Interspecies rate:", self.lbl_interspecies_rate)
        layout.addWidget(off_grp)

        # ── Structure mutations (best genome) ─────────────────────────────
        struct = _CollapsibleGroup("Structure mutations", collapsed=True)
        self.lbl_rate_add_node  = _label("—", "mutRate")
        self.lbl_rate_rem_node  = _label("—", "mutRate")
        self.lbl_rate_add_conn  = _label("—", "mutRate")
        self.lbl_rate_rem_conn  = _label("—", "mutRate")
        self.lbl_bypass         = _label("—", "mutRate")
        self.lbl_rate_add_node.setToolTip(
            "Wahrscheinlichkeit pro Mutation, eine neue Hidden-Node einzufügen.\n"
            "NEAT-Prinzip: eine bestehende Verbindung A→B wird aufgeteilt in A→N→B.\n"
            "Das Netzwerk-Verhalten bleibt dabei erhalten.")
        self.lbl_rate_rem_node.setToolTip(
            "Wahrscheinlichkeit pro Mutation, eine zufällige Hidden-Node zu entfernen.\n"
            "Für jede Verbindung A→N→B wird mit 'Bypass prob' eine Direktverbindung A→B erzeugt.")
        self.lbl_rate_add_conn.setToolTip(
            "Wahrscheinlichkeit pro Mutation, eine neue Verbindung hinzuzufügen.\n"
            "Zyklen (Schleifen) sind erlaubt — das Netz kann damit Gedächtnis entwickeln.")
        self.lbl_rate_rem_conn.setToolTip(
            "Wahrscheinlichkeit pro Mutation, eine zufällige Verbindung zu entfernen.")
        self.lbl_bypass.setToolTip(
            "Beim Entfernen einer Node N: Wahrscheinlichkeit, für jedes Paar A→N→B\n"
            "eine Direktverbindung A→B zu erzeugen (Bypass).\n"
            "Bypass-Gewicht = w(A→N) × w(N→B) — erhält den Signalfluss näherungsweise.")
        struct.addRow("Add node:",    self.lbl_rate_add_node)
        struct.addRow("Rem node:",    self.lbl_rate_rem_node)
        struct.addRow("Add conn:",    self.lbl_rate_add_conn)
        struct.addRow("Rem conn:",    self.lbl_rate_rem_conn)
        struct.addRow("Bypass prob:", self.lbl_bypass)
        layout.addWidget(struct)

        # ── Mutation success rates (population-wide) ──────────────────────
        mut_succ = _CollapsibleGroup("Mutation success", collapsed=True)
        self._mut_success_labels: dict[str, QLabel] = {}
        for mt in ("add_node", "remove_node", "add_connection", "remove_connection",
                    "rewire", "disable", "enable"):
            lbl = _label("—", "mutRate")
            lbl.setToolTip(
                f"Erfolgsrate der Mutation '{mt}': Anteil der Anwendungen,\n"
                f"die zu einer Fitness-Verbesserung führten.\n"
                f"Zeigt, welche Strukturänderungen tatsächlich helfen.")
            self._mut_success_labels[mt] = lbl
            mut_succ.addRow(f"{mt}:", lbl)
        layout.addWidget(mut_succ)

        # ── Strategy genes (best genome) ──────────────────────────────────
        strat = _CollapsibleGroup("Strategy genes", collapsed=True)
        self.lbl_crossover_prob   = _label("—", "mutRate")
        self.lbl_offspring_factor = _label("—", "mutRate")
        self.lbl_compat_thresh    = _label("—", "mutRate")
        self.lbl_sigma_global     = _label("—", "mutRate")
        self.lbl_crossover_prob.setToolTip(
            "Wahrscheinlichkeit, dass ein Offspring durch Kreuzung (Crossover) zweier\n"
            "Elternteile entsteht (statt reiner Mutation).\n"
            "Evolviert sich selbst — kein manuelles Tuning nötig.")
        self.lbl_offspring_factor.setToolTip(
            "Relativer Fortpflanzungsvorteil: höher = öfter als Elternteil ausgewählt.\n"
            "Genomes mit höherem Faktor dominieren die Selektion stärker.\n"
            "Evolviert sich selbst.")
        self.lbl_compat_thresh.setToolTip(
            "Globale Kompatibilitätsschwelle für die Artzuordnung.\n"
            "Zwei Genomes landen in derselben Art wenn ihre Distanz < Schwelle.\n"
            "Wird automatisch angepasst: steigt wenn zu viele Arten entstehen,\n"
            "sinkt wenn zu wenige.")
        self.lbl_sigma_global.setToolTip(
            "Globale Schrittgröße für Gewichts- und Bias-Mutationen (ähnlich CMA-ES).\n"
            "Kleiner = feineres Tuning, größer = weiterer Sprung im Gewichtsraum.\n"
            "Evolviert sich selbst und wird auch von Lamarck als Suchradius genutzt.")
        strat.addRow("Crossover prob:",   self.lbl_crossover_prob)
        strat.addRow("Offspring factor:", self.lbl_offspring_factor)
        strat.addRow("Compat threshold:", self.lbl_compat_thresh)
        strat.addRow("Sigma global:",     self.lbl_sigma_global)
        layout.addWidget(strat)

        # ── Input scales (best genome) ────────────────────────────────────
        iscale = _CollapsibleGroup("Input Scales  (×)", collapsed=True)
        self.lbl_input_scales = _label("—", "mutRate")
        self.lbl_input_scales.setWordWrap(True)
        self.lbl_input_scales.setToolTip(
            "Skalierungsfaktoren für jeden Input-Node — vom Netz selbst erlernt.\n"
            "In0: ×0.12 bedeutet: roher Input-Wert wird intern mit 0.12 multipliziert.\n"
            "Ermöglicht automatische Normalisierung ohne manuelles Preprocessing.\n"
            "Startet bei ×1.0 (neutral) und evolviert sich mit der Zeit.")
        iscale.addRow(self.lbl_input_scales)
        layout.addWidget(iscale)

        # ── Output scales (best genome) ───────────────────────────────────
        oscale = _CollapsibleGroup("Output Scales  (×)", collapsed=True)
        self.lbl_output_scales = _label("—", "mutRate")
        self.lbl_output_scales.setWordWrap(True)
        self.lbl_output_scales.setToolTip(
            "Skalierungsfaktoren für jeden Output-Node — vom Netz selbst erlernt.\n"
            "Out0: ×2.5 bedeutet: der aktivierte Output-Wert wird mit 2.5 multipliziert.\n"
            "Ermöglicht automatische Anpassung an den erwarteten Aktionsbereich.\n"
            "Analogon zu Input Scales, aber für die Ausgabe.")
        oscale.addRow(self.lbl_output_scales)
        layout.addWidget(oscale)

        # ── Weight / Node rates (avg over best genome + population) ─────────
        wn = _CollapsibleGroup("Mutation rates  (best / pop avg)", collapsed=True)
        self.lbl_weight_rate  = _label("—", "mutRate")
        self.lbl_weight_delta = _label("—", "mutRate")
        self.lbl_bias_rate    = _label("—", "mutRate")
        self.lbl_bias_delta   = _label("—", "mutRate")
        self.lbl_activ_rate   = _label("—", "mutRate")
        self.lbl_pop_sigma    = _label("—", "mutRate")
        self.lbl_pop_weight_rate = _label("—", "mutRate")
        self.lbl_pop_bias_rate   = _label("—", "mutRate")
        self.lbl_pop_activ_rate  = _label("—", "mutRate")
        self.lbl_weight_rate.setToolTip(
            "Durchschnittliche Wahrscheinlichkeit, dass ein Gewicht pro Mutation verändert wird.\n"
            "Selbst-adaptiv: passt sich über Generationen an (shift_rate).")
        self.lbl_weight_delta.setToolTip(
            "Durchschnittliche Schrittgröße der Gewichtsänderungen (value_delta × sigma_global).\n"
            "Selbst-adaptiv: kleiner wenn gut konvergiert, größer bei Exploration.")
        self.lbl_bias_rate.setToolTip(
            "Durchschnittliche Wahrscheinlichkeit, dass ein Bias pro Mutation verändert wird.\n"
            "Der Bias verschiebt die Aktivierungsfunktion auf der x-Achse.")
        self.lbl_bias_delta.setToolTip(
            "Durchschnittliche Schrittgröße der Bias-Änderungen.\n"
            "Selbst-adaptiv wie die Gewichts-Schrittgröße.")
        self.lbl_activ_rate.setToolTip(
            "Durchschnittliche Wahrscheinlichkeit, dass eine Node ihre\n"
            "Aktivierungsfunktion pro Mutation wechselt (custom_rate).")
        self.lbl_pop_sigma.setToolTip(
            "Durchschnittlicher sigma_global-Wert über die gesamte evaluierte Population.\n"
            "Höher = Population exploriert global (große Schritte), niedriger = Feintuning.")
        self.lbl_pop_weight_rate.setToolTip(
            "Durchschnittliche Weight-shift-Rate über alle Verbindungen der gesamten Population.")
        self.lbl_pop_bias_rate.setToolTip(
            "Durchschnittliche Bias-shift-Rate über alle Nodes der gesamten Population.")
        self.lbl_pop_activ_rate.setToolTip(
            "Durchschnittliche Aktivierungswechsel-Rate über alle Nodes der gesamten Population.")
        wn.addRow("Weight rate:", self.lbl_weight_rate)
        wn.addRow("Weight Δ:",    self.lbl_weight_delta)
        wn.addRow("Bias rate:",   self.lbl_bias_rate)
        wn.addRow("Bias Δ:",      self.lbl_bias_delta)
        wn.addRow("Activation:",  self.lbl_activ_rate)
        wn.addRow("Pop σ:",       self.lbl_pop_sigma)
        wn.addRow("Pop wt rate:", self.lbl_pop_weight_rate)
        wn.addRow("Pop bias:",    self.lbl_pop_bias_rate)
        wn.addRow("Pop activ:",   self.lbl_pop_activ_rate)
        self.sigma_chart = MutationRateChart()
        self.sigma_chart.setToolTip(
            "Verlauf der populationsweiten Mutationsraten.\n"
            "Grün: sigma_global (Schrittweite)  Blau: Weight-shift-Rate.")
        wn.addWidget(self.sigma_chart)
        layout.addWidget(wn)

        # ── Weight distribution histogram ─────────────────────────────────
        whist = _CollapsibleGroup("Weight Distribution", use_form=False, collapsed=False)
        whist.setToolTip(
            "Histogramm der Verbindungsgewichte im besten Netz.\n"
            "Schmal = Gewichte sind konvergiert (ähnliche Werte).\n"
            "Breit = viel Variation, Netz exploriert noch.\n"
            "Die gestrichelte Linie zeigt Gewicht = 0.")
        self.weight_hist = WeightHistogram()
        whist.addWidget(self.weight_hist)
        layout.addWidget(whist)

        layout.addStretch()
        scroll.setWidget(inner)
        outer.addWidget(scroll)

    def update_genome(self, genome, mem: dict, do_heavy: bool = True) -> None:
        if do_heavy:
            self.canvas.set_genome(genome)

        # Best genome
        self.lbl_nodes.setText(str(len(genome.nodes)))
        self.lbl_connections.setText(str(genome.connection_count))
        self.lbl_best_fit.setText(f"{genome.fitness:.4f}")
        self.lbl_shared_fit.setText(f"{genome.shared_fitness:.4f}")
        if genome.eval_time_ms is not None:
            self.lbl_efficiency.setText(
                f"{genome.efficiency_score:.3f}  ({genome.eval_time_ms:.2f} ms)"
            )
        else:
            self.lbl_efficiency.setText("—")

        # Activation distribution (abbreviation ×count)
        from collections import Counter
        act_counts = Counter(n.activation.name[:3] for n in genome.nodes)
        self.lbl_activations.setText(
            "  ".join(f"{k}:{v}" for k, v in sorted(act_counts.items(), key=lambda x: -x[1]))
            or "—"
        )

        # Population aggregates
        self.lbl_population.setText(str(mem.get("total_genomes", "—")))
        self.lbl_species.setText(str(mem.get("species_count", "—")))
        avg_n = mem.get("avg_nodes_per_genome")
        avg_c = mem.get("avg_connections_per_genome")
        self.lbl_avg_nodes.setText(f"{avg_n:.1f}" if avg_n is not None else "—")
        self.lbl_avg_conns.setText(f"{avg_c:.1f}" if avg_c is not None else "—")
        for key, lbl in (("min_fitness",    self.lbl_pop_min),
                         ("avg_fitness",    self.lbl_pop_avg),
                         ("top5_avg_fitness", self.lbl_top5_avg),
                         ("max_fitness",    self.lbl_pop_max)):
            v = mem.get(key)
            lbl.setText(f"{v:.4f}" if v is not None else "—")
        stag  = mem.get("stagnation_count")
        thr   = mem.get("stagnation_threshold")
        since = mem.get("since_last_injection")
        if stag is not None:
            self.lbl_stagnation.setText(str(stag))
        if since is not None and thr is not None:
            remaining = max(0, thr - since)
            self.lbl_next_injection.setText(f"{since} / {thr}  ({remaining} left)")
        nw = mem.get("novelty_weight")
        if nw is not None:
            self.lbl_novelty_weight.setText(f"{nw:.3f}")
        ew = mem.get("efficiency_weight")
        if ew is not None:
            self.lbl_eff_weight.setText(f"{ew:.3f}")
        if mem.get("sanitize_enabled"):
            self.lbl_invalid_fitness.setText(str(mem.get("n_invalid_fitness", 0)))
            self.lbl_clipped_fitness.setText(str(mem.get("n_clipped_fitness", 0)))
        else:
            self.lbl_invalid_fitness.setText("—")
            self.lbl_clipped_fitness.setText("—")

        # Plateau ratio
        pr = mem.get("plateau_ratio")
        if pr is not None:
            self.lbl_plateau.setText(f"{pr:.2f}")
        else:
            self.lbl_plateau.setText("—")

        # Jump rate
        jr = mem.get("jump_rate")
        if jr is not None:
            self.lbl_jump_rate.setText(f"{jr:.4f}")
        else:
            self.lbl_jump_rate.setText("—")

        # Adaptive population size adjustments
        adj = mem.get("n_pop_size_adjustments")
        if adj is not None and mem.get("adaptive_pop_enabled"):
            self.lbl_pop_size_adjustments.setText(str(adj))
        else:
            self.lbl_pop_size_adjustments.setText("—")
        if mem.get("multi_objective_enabled"):
            maximize = mem.get("multi_objective_maximize")
            self.lbl_multi_objective.setText(f"on {maximize}" if maximize else "on")
        else:
            self.lbl_multi_objective.setText("—")

        # Safety limits (weight clipping + output sanitizing)
        self.lbl_n_weight_clipped.setText(str(mem.get("n_weight_clipped", 0)))
        self.lbl_n_bias_clipped.setText(str(mem.get("n_bias_clipped", 0)))
        self.lbl_n_output_sanitized.setText(str(mem.get("n_output_sanitized", 0)))

        # Quality Diversity
        if mem.get("quality_diversity_enabled"):
            self.lbl_qd_enabled.setText("on")
            self.lbl_qd_cells.setText(str(mem.get("quality_diversity_cells", 0)))
            cov = mem.get("quality_diversity_coverage", 0.0)
            self.lbl_qd_coverage.setText(f"{cov:.4f}")
            self.lbl_qd_updates.setText(str(mem.get("n_quality_diversity_updates", 0)))
            self.lbl_qd_injections.setText(str(mem.get("n_quality_diversity_injections", 0)))
            cells_data = mem.get("quality_diversity_cells_data")
            if cells_data is not None:
                self._last_qd_cells = cells_data
            if do_heavy:
                self.qd_heatmap.set_cells(cells_data)
        else:
            self.lbl_qd_enabled.setText("—")
            self.lbl_qd_cells.setText("—")
            self.lbl_qd_coverage.setText("—")
            self.lbl_qd_updates.setText("—")
            self.lbl_qd_injections.setText("—")
            if do_heavy:
                self.qd_heatmap.set_cells(None)

        if do_heavy:
            self.pareto_scatter.set_points(mem.get("pareto_points"))

        research = format_research_diagnostics(mem)
        self.lbl_matrix_forward.setText(research["matrix_forward"])
        self.lbl_fitness_components.setText(research["fitness_components"])
        self.lbl_meta_policy.setText(research["meta_policy"])
        self.lbl_module_library.setText(research["module_library"])

        # Fitness histogram
        fh = mem.get("fitness_histogram")
        self.fitness_hist.set_data(fh)

        # Eval-time statistics
        for key, lbl in (
            ("eval_time_mean_ms",   self.lbl_eval_mean),
            ("eval_time_median_ms", self.lbl_eval_median),
            ("eval_time_p95_ms",    self.lbl_eval_p95),
            ("eval_time_max_ms",    self.lbl_eval_max),
        ):
            v = mem.get(key)
            lbl.setText(f"{v:.2f} ms" if v is not None else "—")

        # Offspring counters
        for key, lbl in (
            ("n_crossover",           self.lbl_n_crossover),
            ("n_mutation_only",       self.lbl_n_mutation),
            ("n_diversity_injection", self.lbl_n_injection),
        ):
            v = mem.get(key)
            lbl.setText(str(v) if v is not None else "—")
        is_rate = mem.get("interspecies_crossover_current")
        is_mode = mem.get("interspecies_crossover_mode")
        is_reason = mem.get("interspecies_crossover_last_reason")
        self.lbl_interspecies_rate.setText(
            f"{is_rate:.3f} ({is_mode})" if is_rate is not None else "—"
        )
        if is_reason:
            self.lbl_interspecies_rate.setToolTip(str(is_reason))

        # Curriculum stats
        cur_idx = mem.get("curriculum_stage_index")
        if cur_idx is not None:
            self._cur_grp.setVisible(True)
            count = mem.get("curriculum_stage_count", "?")
            self.lbl_cur_stage.setText(f"{cur_idx + 1} / {count}")
            self.lbl_cur_name.setText(mem.get("curriculum_stage_name") or "—")
            target = mem.get("curriculum_stage_target")
            self.lbl_cur_target.setText(f"{target:.4f}" if target is not None else "—")
            best = mem.get("curriculum_stage_best")
            self.lbl_cur_best.setText(f"{best:.4f}" if best is not None else "—")
        else:
            self._cur_grp.setVisible(False)

        # Lamarck stats
        lamarck_mode = mem.get("lamarck_mode")
        if lamarck_mode is not None:
            self.lbl_lamarck_mode.setText(lamarck_mode)
        n_applied = mem.get("lamarck_n_applied")
        if n_applied is not None:
            self.lbl_lamarck_applied.setText(str(n_applied))
        n_steps_total = mem.get("lamarck_n_steps_total")
        if n_steps_total is not None:
            self.lbl_lamarck_steps.setText(str(n_steps_total))
        per_species = mem.get("lamarck_per_species")
        if per_species is not None:
            non_empty = [s for s in per_species if s["members"] > 0]
            if non_empty:
                avg = sum(s["lamarck_n_steps_total"] for s in non_empty) / len(non_empty)
                self.lbl_lamarck_per_species.setText(f"{avg:.1f}")
            else:
                self.lbl_lamarck_per_species.setText("—")
        n_blocked = mem.get("lamarck_n_blocked_top_k")
        if n_blocked is not None:
            self.lbl_lamarck_blocked.setText(str(n_blocked))
        lt = mem.get("lamarck_time_ms")
        if lt is not None:
            self.lbl_lamarck_time.setText(f"{lt:.1f}")
        ls = getattr(genome, 'lamarck_sigma', None)
        if ls is not None:
            self.lbl_lamarck_sigma.setText(f"{ls:.4f}")

        # Structure mutations
        self.lbl_rate_add_node.setText(f"{genome.mutation_add_node.bool_rate:.4f}")
        self.lbl_rate_rem_node.setText(f"{genome.mutation_remove_node.bool_rate:.4f}")
        self.lbl_rate_add_conn.setText(f"{genome.mutation_add_connection.bool_rate:.4f}")
        self.lbl_rate_rem_conn.setText(f"{genome.mutation_remove_connection.bool_rate:.4f}")
        self.lbl_bypass.setText(f"{genome.bypass_connection_prob:.4f}")

        # Mutation success rates (population-wide)
        m_succ = mem.get("mutation_success", {})
        m_total = mem.get("mutation_total", {})
        for mt, lbl in self._mut_success_labels.items():
            total = m_total.get(mt, 0)
            succ = m_succ.get(mt, 0)
            if total > 0:
                lbl.setText(f"{succ / total:.0%} ({succ}/{total})")
            else:
                lbl.setText("—")

        # Strategy genes
        self.lbl_crossover_prob.setText(f"{genome.crossover_prob:.4f}")
        self.lbl_offspring_factor.setText(f"{genome.offspring_factor:.4f}")
        ct = mem.get("compat_threshold")
        self.lbl_compat_thresh.setText(f"{ct:.4f}" if ct is not None else "—")
        self.lbl_sigma_global.setText(f"{genome.sigma_global:.4f}")

        # Input scales
        inp_nodes = genome.input_nodes
        if inp_nodes:
            self.lbl_input_scales.setText(
                "  ".join(f"In{i}: ×{n.input_scale:.3f}" for i, n in enumerate(inp_nodes))
            )
        else:
            self.lbl_input_scales.setText("—")

        # Output scales
        out_nodes = genome.output_nodes
        if out_nodes:
            self.lbl_output_scales.setText(
                "  ".join(f"Out{i}: ×{n.output_scale:.3f}" for i, n in enumerate(out_nodes))
            )
        else:
            self.lbl_output_scales.setText("—")

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
            if do_heavy:
                self.weight_hist.set_weights([c.weight for c in conns])
        elif do_heavy:
            self.weight_hist.set_weights([])

        # Population-wide avg rates
        def _fmt_pop(key: str) -> str:
            v = mem.get(key)
            return f"{v:.4f}" if v is not None else "—"
        self.lbl_pop_sigma.setText(_fmt_pop("pop_avg_sigma_global"))
        self.lbl_pop_weight_rate.setText(_fmt_pop("pop_avg_weight_rate"))
        self.lbl_pop_bias_rate.setText(_fmt_pop("pop_avg_bias_rate"))
        self.lbl_pop_activ_rate.setText(_fmt_pop("pop_avg_activ_rate"))

        if do_heavy:
            sigma = mem.get("pop_avg_sigma_global")
            wt = mem.get("pop_avg_weight_rate")
            if sigma is not None and wt is not None:
                self.sigma_chart.add_point(sigma, wt)

    def _export_qd_archive(self) -> None:
        if not self._last_qd_cells:
            QMessageBox.information(self, "QD Export", "Kein QD-Archiv verfügbar.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "QD-Archiv exportieren", "qd_archive.json",
            "JSON (*.json);;CSV (*.csv)"
        )
        if not path:
            return
        try:
            if path.endswith(".csv"):
                import csv
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    first = self._last_qd_cells[0]
                    n_dims = len(first.get("cell", []))
                    header = [f"cell_{i}" for i in range(n_dims)] + ["fitness"]
                    writer.writerow(header)
                    for c in self._last_qd_cells:
                        writer.writerow(list(c.get("cell", [])) + [c.get("fitness", "")])
            else:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self._last_qd_cells, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Export fehlgeschlagen", str(e))


# ---------------------------------------------------------------------------
# Gym render widget
# ---------------------------------------------------------------------------
