# Tasks: YANE staerker machen

Diese Datei ist die aktuelle Roadmap fuer YANE. Offene und neue Tasks stehen
oben. Abgeschlossene Arbeit ist weiter unten nur noch kompakt zusammengefasst.

## Status

**Aktueller Stand:** Alle P0- und P1-Tasks abgeschlossen. Teststand: `699 passed`.

- Core-Evolution, Speciation, Mutation, Worker-Pipeline, GUI, API, Logging, Checkpoints: implementiert.
- Multi-Objective, Quality Diversity, CMA-ES, Backprop-/Matrix-Bausteine, Presets, Benchmark-Gates: implementiert.
- AdaptiveController, OperatorScheduler, Lamarck-Budget, Interspecies-Trigger (Novelty/Isolation/Schutz): implementiert.
- Adaptive Benchmark-Suite, GUI-Stability-Guard, Preset-Schema v2 mit `adaptive_policies`: implementiert.
- Matrix-Forward-Integration, Checkpoint-State fuer Adaptive-Objekte: implementiert.
- Naechster Schwerpunkt: Checkpoint-Format weiter haerten, P2-Forschungsfeatures.

## Legende

- `P0`: hoher Hebel, nah am aktuellen Code, direkt nuetzlich
- `P1`: wichtiger Ausbau, mittlerer Aufwand
- `P2`: experimentell, Forschungsarbeit oder groesserer Umbau
- ✅: erledigt
- ⚡: teilweise erledigt
- 🔲: offen

---

## Offene Tasks

### P1 ⚡ Checkpoint-Format langfristig haerten

Checkpoint v2 mit Migration, Metadata-Sidecar und Adaptive-State ist
implementiert. Fuer langfristige Robustheit fehlen noch einige Punkte.

Aufgaben:

- Kleine Fixture-Checkpoints fuer v1/v2 in Tests versionieren.
- JSON-Metadaten in GUI/API sichtbar machen.
- Optional: getrennte Speicherung von Config, Tracker, Population, QD-Archiv.
- Import-Warnungen fuer fehlende Descriptor-Callbacks in GUI anzeigen.
- Dokumentieren, welche Teile Pickle bleiben und warum.

Bereits erledigt:

- Checkpoint-State fuer `AdaptiveController` und `OperatorScheduler`. ✅

Nutzen:

- Alte Laeufe bleiben langfristig nutzbar.
- Checkpoints werden besser debugbar.

### P1 🔲 Remote/Distributed Evaluation konkretisieren

`AsyncEvaluationQueue` ist ein lokaler Baustein. Remote-Auswertung ist noch
nicht produktiv nutzbar.

Aufgaben:

- Remote-Worker-Protokoll entwerfen: Job, Genome-Payload, Result, Error, Timeout.
- HTTP- oder WebSocket-Prototyp bauen.
- Retry/Timeout/Cancel-Policy implementieren.
- Security-Grenzen dokumentieren: keine fremden Pickles ungeprueft laden.
- Benchmark gegen lokales Multiprocessing.

Nutzen:

- Lange Simulationen koennen auf mehrere Prozesse oder Maschinen verteilt werden.
- Saubere Grundlage fuer Cluster- oder Server-Experimente.

### P2 🔲 Evolvierbare Descriptor-Gewichte

Descriptor-Registry und Fitness-Komponenten sind vorhanden. Evolvierbare oder
adaptive Gewichtung ist noch Forschungsarbeit.

Aufgaben:

- Gewichtshistorie fuer Fitness-Komponenten speichern.
- Adaptive Gewichtung bei Stagnation testen.
- Descriptor-Kombinationen per Ablation benchmarken.
- Mechanismus gegen Descriptor-Collapse entwerfen.

### P2 🔲 Meta-adaptive Policies evolvieren

Wenn die handgebauten adaptiven Policies stabil sind, koennen ihre Parameter
selbst zum Evolutionsobjekt werden.

Aufgaben:

- Policy-Gene fuer Operator-Scheduler, Lamarck-Budget und Interspecies-Rate modellieren.
- Policy-Gene pro Species und global vergleichen.
- Sicherheitsgrenzen fuer extreme Policies einbauen.
- Meta-Ablation: feste Policy vs handadaptive Policy vs evolvierte Policy.

### P2 🔲 Modul-Crossover und Modulbibliothek

Module koennen erkannt und dupliziert werden. Wiederverwendung ueber Genome und
Laeufe hinweg ist noch offen.

Aufgaben:

- Modul-Crossover zwischen kompatiblen Subgraphen erforschen.
- Gute Module in einer Bibliothek speichern.
- Mutationsoperator: Modul aus Bibliothek einfuegen.
- Diagnostics: Modulhaeufigkeit und Wiederverwendungsrate.

### P2 🔲 Evolvierbare CPPNs

Indirekte Kodierung kann Verbindungen aus Koordinaten erzeugen, aber die
CPPN-Funktion selbst ist noch nicht evolvierbar.

Aufgaben:

- CPPN-Genome als eigene kleine Netzklasse oder normales YANE-Genome modellieren.
- Weight-Pattern aus CPPN-Outputs erzeugen.
- HyperNEAT-artige Substrate fuer Inputs/Outputs definieren.
- Benchmark gegen direkte Kodierung auf regelmaessigen Aufgaben.

---

## Abgeschlossen

### ✅ P0 Adaptive Control Layer einfuehren

- `AdaptiveController` mit einheitlichen Signalen (Plateau, Fitness-Trend, Diversity, Species-Stagnation, Eval-Kosten, Komplexitaet).
- Policy-Format `off` / `fixed` / `adaptive` / `auto` fuer alle Features.
- `PolicyDecision`-Recorder fuer Diagnostics.
- Integration in `neuro_evolution.py` und `diagnostics.py`.
- API: `set_adaptive_control()`, `get_adaptive_controller()`.

### ✅ P0 Lamarck-Modi adaptiv vereinheitlichen

- Lamarck-Optimierer (Hill-Climb, NES, SA, CMA-ES) × Zeitplan (aus, explizit, adaptiv) in GUI klar getrennt.
- Per-Species-Eligibility: `set_lamarck_per_species()`, `LamarckRefiner.set_eligible_species()`.
- Kostenbudget: `set_lamarck_budget()`, `_consume_budget()`, `reset_generation_budget()`.
- Diagnostics: Modus, `n_improved`, `budget_used`, `budget_exhausted_count`, `species_stats`.

### ✅ P0 GUI fuer adaptive Features eindeutig machen

- Eigene Sektion `Adaptive Control` (CollapsibleGroup) mit Live-Labels.
- Vier Presets: Konservativ, Balanciert, Aggressiv, Analysefreundlich.
- Crash-Guard um `_update_adaptive_labels` (try/except + log_warning).
- 12 GUI-Smoke-Tests fuer Widgets, Labels und Preset-Interaktion.

### ✅ P0 Interspecies-Crossover adaptiv machen

- Adaptive Rate: Stagnations-, Novelty- und Isolation-Trigger, Schutzregel bei schlechter Erfolgsrate.
- Diagnostics: aktuelle Rate, Modus, Min/Max, letzter Trigger, Crossover-Erfolgsrate, Nachkommen-Fitness.
- GUI: Fix / Adaptiv mit Live-Rate und letztem Trigger.

### ✅ P0 Adaptive Operator-Scheduler

- `OperatorScheduler` mit globalen und per-Species-Gewichten fuer alle Mutations-Operatoren.
- Adaptiver QD-Druck und Pruning-Druck.
- `sync_from_population()`, `tick()`, `apply_to_genome()`, `get_diagnostics()`.
- API: `set_operator_scheduler()`, `get_operator_scheduler()`.

### ✅ P1 Adaptive Benchmark-Suite

- `benchmarks/adaptive_suite.py`: 7 Konfigurationen (baseline → full_adaptive) auf XOR und CartPole.
- Metriken: Loesung, Iterationen, Wall-Time, Best-Fitness, adaptive Diagnostics.

### ✅ P1 GUI-Stability-Analyse

- Crash-State-Snapshot alle 100 Iterationen nach `_crash_state.json`.
- `ResourceGuard` fuer System- und Prozess-RAM.
- Crash-Guard in `_update_adaptive_labels`.
- Tests: alle Crash-State-Keys im `population_memory_info()`-Dict vorhanden.

### ✅ P1 Preset-System fuer adaptive Profile erweitern

- `ExperimentPreset.adaptive_policies: dict` mit optionalem JSON-Abschnitt (Schema v2, rueckwaertskompatibel).
- 4 Preset-Dateien in `presets/`: `adaptive_konservativ`, `adaptive_balanciert`, `adaptive_aggressiv`, `adaptive_analysefreundlich`.
- `_current_adaptive_policies()` und `_apply_adaptive_policies(ap)` in `TrainingTab`.
- `_save_current_preset()` persistiert adaptive Einstellungen, `_on_preset_changed()` befuellt adaptive Widgets.
- 11 neue Tests in `test_presets.py`, 2 neue GUI-Smoke-Tests.

### ✅ P1 Release-Cleanup und API-Konsistenz

- `__init__.py`: `AdaptiveController`, `AdaptiveSignals`, `FeaturePolicy`, `OperatorScheduler` zu `__all__` hinzugefuegt.
- README: Abschnitt Adaptive Control Layer + Operator Scheduler, Projektstruktur, Presets-Abschnitt mit `adaptive_policies`-Tabelle, Status-Sektion aktualisiert.
- API-Namenskonsistenz bestaetigt.

### ✅ P1 Pareto- und MAP-Elites-Visualisierung polishen

- `ParetoScatter`: Achsenbeschriftung (Min/Max-Ticks), Punkte farbig nach Fitness (blau→gruen), Hover-Tooltip mit Objectives/Fitness/Nodes/Connections.
- `MapElitesHeatmap`: Hover-Tooltip mit Zell-Koordinaten und Fitness, Fitness-Range in Titelzeile.
- `LeftPanel`: Export-Button fuer QD-Archiv (JSON und CSV), `_last_qd_cells` fuer spaetere Exporte.
- 5 neue GUI-Smoke-Tests.

### ✅ P1 Checkpoint: AdaptiveController und OperatorScheduler State

- `save_checkpoint()` speichert `adaptive_ctrl`, `adaptive_ctrl_enabled`, `operator_scheduler`, `operator_scheduler_enabled`.
- `load_checkpoint()` stellt Zustand wieder her; `_operator_scheduler` wird neu an die Population verdrahtet.
- Rueckwaertskompatibel: Alte Checkpoints ohne diese Keys laden mit `enabled=False`.
- 6 neue Tests in `test_checkpoint_migration.py`.

### ✅ P1 Matrix-Forward automatisch im Training nutzen

- `set_matrix_forward(enabled=True)` aktiviert transparente Matrix-Beschleunigung per `genome.forward()`.
- Automatischer Fallback bei inkompatiblen Genomen (Zyklen, Memory-Nodes, unbekannte Aktivierung).
- Diagnostics: `matrix_forward_hits` und `matrix_forward_misses` in `population_memory_info()`.
- 6 neue Tests in `test_matrix_export.py`.
