# Tasks: YANE staerker machen

Diese Datei ist die aktuelle Roadmap fuer YANE. Offene und neue Tasks stehen
oben. Abgeschlossene Arbeit ist weiter unten nur noch kompakt zusammengefasst.

## Status

**Aktueller Stand:** Alle P0-Bausteine fertig. Die grossen P1-Architekturthemen
aus diesem Loop sind weitgehend implementiert (Adaptive Policy System,
RunDatabase/Experiment-Tracking, Selektionsstrategien, Middleware, Reports,
Inselmodell, Surrogates, Online-Tuning, Weight-Inheritance). P2-Features
Lamarck-Momentum, Post-Training Pruning, Population-Size-Adaptation,
Intrinsic Curiosity, DARTS-Lite und Shared Weights vollstaendig abgeschlossen.
Teststand: `1153 passed, 1 skipped`.

> **Roadmap-Erweiterung (2026-05-26):** 26 neue Features geplant — 5 P1
> (Benchmarking-Suite, WandB/MLflow, Interactive Evolution, Hardware-Aware,
> Data Augmentation) und 21 P2 (ONNX-Export, Distillation, Gradient-Hybrid,
> WASM-Export, Attention-Heads, LTC-Nodes, Temporal Speciation, Self-Play,
> H-NEAT, GRN-Encoding, Developmental NEAT, Continual Learning, Meta-Learning,
> Reservoir Computing, Open-Endedness, Multi-Agent Cooperation, Bayesian NEAT,
> Safe NEAT, Sparse NEAT, TFLite-Export, Symbolic Regression).

> **Roadmap-Korrektur:** Mehrere P2-Forschungsfeatures besitzen aktuell nur
> isolierte Experimental-/Spike-Bausteine. Sie sind unten als `⚡` oder `□`
> markiert, bis sie sauber in `Genome`, `Population`, `NeuroEvolution`,
> Checkpoints, Crossover und Tests integriert sind.

- Core-Evolution, Speciation, Mutation, Worker-Pipeline, GUI, API, Logging, Checkpoints: implementiert.
- Multi-Objective, Quality Diversity, CMA-ES, Backprop-/Matrix-Bausteine, Presets, Benchmark-Gates: implementiert.
- AdaptiveController, OperatorScheduler, Lamarck-Budget, Interspecies-Trigger (Novelty/Isolation/Schutz): implementiert.
- Adaptive Benchmark-Suite, GUI-Stability-Guard, Preset-Schema v2 mit `adaptive_policies`: implementiert.
- Matrix-Forward-Integration, Checkpoint-State fuer Adaptive-Objekte: implementiert.
- Checkpoint gehaertet: Fixture-Dateien, JSON-Metadaten in GUI+API, Pickle-Dokumentation: implementiert.
- Remote/Distributed Evaluation: HTTP-Protokoll, Client/Worker, Retry/Cancel, Benchmark: implementiert.
- P2-Forschungsfeatures: Modulbibliothek, CPPN, Meta-adaptive Policies und evolvierbare Descriptor-Gewichte sind implementiert; DARTS-Lite, Intrinsic Curiosity und Shared Weights vollstaendig integriert; STDP, Neuromodulation, Input-/Output-Gruppierung, Conv-NEAT und ES-HyperNEAT sind noch Spikes/offen.
- raw_fitness-Fix (Fitness-Komponenten verschmutzen nicht mehr genome.fitness fuer Ziel-Check und Diagnostics): implementiert.
- Event-System, Anomalie-Detektion, Fitness-Transformer, Genome-Export, Validierungs-Set, Konfigurationspersistenz, Gym-Inspect-Verbesserung: implementiert.
- Naechste Schwerpunkte: P2-Research-Spikes ehrlich zu vollstaendigen Features ausbauen oder als experimentell belassen; Checkpoint-Kompatibilitaets-Diff haerten; 26 neue P1/P2-Features (Benchmark-Suite, WandB/MLflow, ONNX/WASM/TFLite-Export, Distillation, Gradient-Hybrid, Attention, LTC, Temporal Speciation, Self-Play, H-NEAT, GRN-Encoding, Developmental NEAT, Continual/Meta-Learning, Reservoir Computing, Open-Endedness, Multi-Agent Cooperation, Interactive/Bayesian/Safe/Sparse/Hardware-Aware NEAT, Data Augmentation, Symbolic Regression).

## Legende

- `P0`: hoher Hebel, nah am aktuellen Code, direkt nuetzlich
- `P1`: wichtiger Ausbau, mittlerer Aufwand
- `P2`: experimentell, Forschungsarbeit oder groesserer Umbau
- ✓: erledigt
- ⚡: teilweise erledigt
- □: offen

---

## Architektur-Leitlinie

YANE ist in drei konzeptionelle Schichten gegliedert. Neue Features sollen klar einer Schicht zugeordnet werden.

**Layer 1 — Stable Core**
Population, Genome, Speciation, Mutation, Evaluation, Checkpoints.
Kleine, testbare, rueckwaertskompatible Basis. Aenderungen hier erfordern besondere Sorgfalt; keine experimentellen Abhaengigkeiten.

**Layer 2 — Adaptive Systems**
Recovery, Auto-Tuning, Scheduling, Surrogates, Diversity-Systeme, Policy-Orchestrierung.
Baut auf Layer 1 auf; nutzt ausschliesslich oeffentliche APIs. Ziel ist das Adaptive Policy System als einheitlicher Orchestrierungs-Punkt.

**Layer 3 — Research Features**
DARTS, STDP, Neuromodulation, Curiosity, ES-HyperNEAT, Input-/Output-Gruppierung, Convolutional NEAT, Attention-Heads, LTC-Nodes, ONNX/WASM/TFLite-Export, Distillation, Gradient-NEAT-Hybrid, Self-Play, H-NEAT, GRN-Encoding, Developmental NEAT, Continual Learning, Meta-Learning, Reservoir Computing, Open-Endedness, Multi-Agent Cooperation, Bayesian NEAT, Safe NEAT, Sparse NEAT, Symbolic Regression.
Research-Features werden modular und per API an-/abschaltbar implementiert. Sie koennen in Genome, Population und NeuroEvolution integriert sein, muessen aber standardmaessig deaktiviert sein und duerfen bei Deaktivierung keine Laufzeitkosten verursachen.

Ziel: Layer 1 wartbar halten; Layer 2 durch das Policy-Interface erweiterbar machen; Layer 3 vollstaendig integriert aber sicher deaktivierbar.

---

## Offene Tasks

---

### ✓ P0 Strukturierte Evaluator-Komponenten / Task-Curriculum

Mehrere Langtests zeigen: manche Umgebungen scheitern nicht an NEAT selbst,
sondern an einer zu flachen oder zu monolithischen Fitnessfunktion. Taxi,
CliffWalking und FrozenLake werden erst robust, wenn State-Encoding,
Teilaufgaben und lokale Aktionssignale explizit modelliert werden. Diese
Logik steckt derzeit direkt in `gui/examples.py`; sie soll als wiederverwendbares
Evaluator-Baukastenmodell verfuegbar werden.

**Aktueller Stand:** Vollstaendig implementiert. `EvaluatorSpec` mit
`enabled_components` erlaubt Ablations-Vergleiche; CliffWalking
(`rollout_score`, `oracle_score`), FrozenLake (`rollout_score`, `subgoal_score`)
und Taxi (`policy_score`, `rollout_score`, `subgoal_score`) nutzen alle
`EvaluatorSpec.combine()`. GUI-Komponentenschalter (Checkboxen pro Evaluator-
Komponente) erlauben Ein-/Ausschalten fuer Vergleichslaeufe. Diagnostics
schreiben Komponentenwerte via `_component_scores` in
`_eval_middleware_diagnostics["evaluator_components"]`. Ablations-Tests und
MultiStartRollout-Tests abgedeckt.

**Ziel:** Beispiele sollen Fitness aus klar benannten Komponenten zusammensetzen:
Rollout-Erfolg, Subgoal-Fortschritt, lokale Policy-Qualitaet, illegal-action-
Penalty, State-Coverage und optional Curriculum-Stufen.

- `EvaluatorSpec`: beschreibt `state_encoder`, `rollout_cases`, `subgoals`,
  `component_weights`, `target_fitness` und `validation_cases`.
- `StateEncoder`: eingebaute Encoder `scaled`, `one_hot`, `mixed`; Beispiele
  koennen damit ohne handgeschriebene Input-Dimension-Drift konfiguriert werden.
- `SubgoalReward`: generische Komponente fuer Distanz zum naechsten Subziel
  (Manhattan, Graph-Distanz, benutzerdefiniert).
- `GraphPolicyScore`: fuer diskrete MDPs mit bekanntem `env.P`; bewertet lokale
  Aktionen anhand kuerzester Pfade zum Ziel ohne vollstaendige Episode erzwingen
  zu muessen.
- `MultiStartRollout`: feste oder gesampelte Startzustandsliste; Fitness ist
  aggregiert ueber Cases, optional mit separatem Validation-Set.
- `CurriculumSpec`: Teilaufgaben werden der Reihe nach freigeschaltet
  (z.B. Taxi: Navigation zu Passagier → Pickup → Navigation zum Ziel → Dropoff).
- Diagnostics: Komponenten-Scores pro Generation (`rollout_score`,
  `policy_score`, `subgoal_score`, `illegal_action_rate`, `case_success_rate`).
- GUI: Beispiele zeigen die aktiven Komponenten und erlauben optionales
  Ein-/Ausschalten fuer Vergleichslaeufe.

**Akzeptanzkriterien:**

- ✓ FrozenLake, CliffWalking und Taxi nutzen `EvaluatorSpec` statt handverteilter
  Speziallogik im Evaluator.
- Taxi loest in einem 10-Minuten-Benchmark mindestens 3/3 Seeds mit Default-
  Einstellungen.
- FrozenLake loest in einem 10-Minuten-Benchmark mindestens 3/3 Seeds.
- CliffWalking loest in einem 30-Minuten-Benchmark mindestens 3/3 Seeds.
- ✓ Ein Ablations-Benchmark zeigt Rollout-only vs. Subgoal vs. GraphPolicyScore.
- ✓ Tests decken State-Encoding-Dimensionen, GraphPolicyScore und MultiStart-
  Aggregation ab.

**Beziehung zu P1 Evaluation-Middleware:** Dieser Task ist die konkrete
Beispiel-/Fitness-Ebene. Der P1 Middleware-Stack ist spaeter der allgemeine
technische Rahmen, um diese Komponenten beliebig zu komponieren, zu cachen und
zu protokollieren.

---

### ✓ P0 Self-Tuning Speziation (Automatischer Kompatibilitaetsschwellenwert)

Die Spezies-Anzahl schwankt stark wenn der Kompatibilitaetsschwellenwert fest eingestellt ist; zu viele Spezies verschwenden Budget, zu wenige unterdruecken Diversitaet.

**Aktueller Stand:** Implementiert. `set_target_species(n)`,
`set_target_species(n_min=..., n_max=..., tune_interval=...)` und
`set_target_species(None)` sind verfuegbar. Diagnostics enthalten Zielband,
Tune-Intervall, letzten Anpassungsschritt und Trend.

- ✓ `NeuroEvolution.set_target_species(n_min=4, n_max=8, tune_interval=10)`: YANE passt den Schwellenwert alle N Schritte an um die Zielbreite einzuhalten.
- ✓ PI-Regler: Schwellenwert steigt wenn zu viele Spezies vorhanden, sinkt wenn zu wenige; begrenzt durch `[threshold_min, threshold_max]`.
- ✓ Diagnostics: aktueller Schwellenwert, letzter Anpassungsschritt, Anpassungs-Trend.
- ✓ `set_target_species(None)` deaktiviert Self-Tuning.
- ✓ Tests: Zielband-Konfiguration, Deaktivierung und Anpassungsverhalten.

### ✓ P0 Adaptive Recovery System

Bei erkannter Stagnation oder Diversitaets-Kollaps laeuft YANE ohne Gegenmasnahme weiter. Die Anomalie-Detektoren erkennen das Problem, handeln aber nicht. Gleichzeitig darf ein naives Patience-Kriterium NEAT-typische Stepping-Stone-Phasen nicht abwuergen: lange Stagnation vor einem Fitness-Sprung ist normaler NEAT-Ablauf. Dieses System vereint Recovery, Diversity-Injection und Early Stopping in einer einzigen, eskalierenden Regelkette.

**Aktueller Stand:** Implementiert via `NeuroEvolution.set_adaptive_recovery(...)`.
Recovery nutzt vorhandene Population-Mechanismen: Diversity-Injektion,
Teil-Restart ueber Best-Only-Shrink plus Fresh-Genome-Injection und temporaerer
Lamarck-Budget-Burst. Cooldown, Eskalation, konservatives Early Stopping,
persistente Stop-Attribute und Diagnostics sind vorhanden.

*Hinweis: `iterations` in YANE zaehlt einzelne Genome-Evaluierungen, nicht Generationen; 1 Generation = `pop_size` Iterationen. Bei pop_size=100 entsprechen 50.000 Iterationen nur 500 Generationen. Der `patience`-Parameter bezieht sich auf Generationen.*

*Dieses System wird spaeter als Policy-Implementierung unter dem Adaptive Policy System (P1) modelliert.*

**API:**

- ✓ `NeuroEvolution.set_adaptive_recovery(enabled=True, strategies=["diversity_boost", "partial_restart", "lamarck_burst"], cooldown=20, escalate=True, diversity_iqr_threshold=1e-4, injection_frac=0.1, early_stopping_patience=500, warmup=100, min_delta=1e-4)`.

**Trigger (in Reihenfolge):**

1. ✓ `DiversityCollapseDetector`/IQR-Signal: Injektions-Phase mit `injection_frac * pop_size` neuen Genomen.
2. ✓ `HomogenizationDetector` oder `StuckSpeciationDetector`: Recovery-Phase mit konfiguriertem `strategy`.
3. ✓ Wenn Recovery nach `cooldown`-Generationen keinen Fitness-Delta > `min_delta` erzeugt und `escalate=True`: naechste Strategie in der Liste versuchen.

**Recovery-Strategien:**

- ✓ `diversity_boost`: mehrere diverse Genome einschleusen.
- ✓ `partial_restart`: schwache Populationsteile entfernen und durch neue Genome ersetzen.
- ✓ `lamarck_burst`: Lamarck-Budget temporaer erhoehen um lokales Optimum zu verlassen.

**Cooldown und Eskalation:**

- ✓ Nach jeder Recovery-Massnahme mindestens `cooldown` Generationen Pause vor erneutem Trigger.
- ✓ Bei `escalate=True` werden Strategien der Reihe nach eskaliert wenn keine Verbesserung eintritt.
- ✓ Sind alle Strategien erschoepft oder Diversity kollabiert und es gibt keine Verbesserung seit `early_stopping_patience` Generationen: Trainings-Stop.

**Early Stopping (letzte Stufe, kein naives Patience-Kriterium):**

- ✓ Stop wird nur ausgeloest wenn `patience` Generationen ohne Verbesserung um `min_delta` UND mindestens ein weiteres Signal: Diversity kollabiert ODER alle Recovery-Strategien erschoepft und erfolglos.
- ✓ `warmup`-Generationen werden immer vollstaendig abgewartet bevor Early Stopping greifen kann.
- Optional: statistischer Trend-Test (Mann-Whitney U ueber beide Haelften des Patience-Fensters) via `use_stat_test=True`.
- Hoher Default (500 Generationen = 50k Iterationen bei pop_size=100) verhindert vorzeitiges Abwuergen von Stepping-Stone-Phasen.

**Persistente Attribute nach Training:**

- ✓ `yane.stopped_early: bool`
- ✓ `yane.stop_reason: str` (z.B. `"patience_exhausted_and_diversity_collapsed"`, `"patience_exhausted_and_all_strategies_exhausted"`)

**Diagnostics:**

- `recovery_events`: Liste aller Recovery-Massnahmen mit Generation und Strategie.
- `last_recovery_strategy`: zuletzt verwendete Strategie.
- `recovery_success_rate`: Anteil Recoveries mit Fitness-Delta > `min_delta` nach `cooldown`.
- `no_improvement_generations`: aktuelle Stagnations-Generationenanzahl.
- `early_stop_triggered: bool`, `stop_conditions_met: list[str]`.
- `injections_total`, `last_injection_generation`, `iqr_before_injection`, `iqr_after_injection`.

**Tests:**

- `HomogenizationDetector`-Trigger loest korrekte Strategie aus.
- Cooldown verhindert Mehrfach-Trigger innerhalb von N Generationen.
- Erfolgreiche Recovery: Fitness-Delta > `min_delta` nach `cooldown` → `recovery_success_rate` steigt.
- Erfolglose Recovery + Eskalation: naechste Strategie wird nach `cooldown` aktiviert.
- Stepping-Stone-Szenario: Fitness springt nach langer Pause → kein vorzeitiger Stop.
- Early Stopping nur wenn Patience UND Diversity-Bedingung erfuellt; `warmup` wird respektiert.
- Diversity-Injektion: korrekte Anzahl Genome ersetzt; IQR steigt nach Injektion.

### ✓ P0 Anytime-Evaluation (Adaptives Evaluations-Budget)

Jedes Genom wird gleich oft evaluiert, egal wie schlecht es ist; teure Umgebungen (z.B. CartPole) verschwenden Budget auf schwache Genome.

**Aktueller Stand:** Implementiert via
`NeuroEvolution.set_anytime_eval(enabled=True, min_evals=1, max_evals=5,
promotion_frac=0.3, aggregation="mean")`. Der Runner bewertet alle Genome mit
Minimalbudget und promotet konkurrenzfaehige Genome anhand des aktuellen
Populationsquantils. Diagnostics melden durchschnittliche Evals, gesparte Evals,
Promotion-Rate und Varianz promoteter Genome.

> **Architektur: Adaptive Evaluation Budgeting** — Anytime-Evaluation (P0) und Fitness-Surrogate (P1) teilen denselben Unterbau: Budget-Verwaltung per Promotion-Fraktion, Diagnostics-Schema (gesparte Evals, Eval-Varianz, Promotion-Rate), konfigurierbares Aggregations-Fn. Anytime benutzt mehrfache echte Evaluierungen; Surrogate filtert vorab per Modell-Vorhersage. Beide Mechanismen koennen gleichzeitig aktiv sein: Surrogate filtert zuerst, Anytime bewertet die promotierten Genome mehrfach.

- ✓ `NeuroEvolution.set_anytime_eval(enabled=True, min_evals=1, max_evals=5, promotion_frac=0.3)`.
- ✓ Phase 1: alle Genome werden schnell evaluiert.
- ✓ Phase 2: obere `promotion_frac` werden bis `max_evals` evaluiert; Fitness wird aggregiert.
- ✓ Konfigurierbares Aggregations-Fn: `mean`, `median`, `min`, `max`.
- ✓ Diagnostics: durchschnittliche Evals pro Genome, gesparte Evals, Fitness-Varianz der promovierten Genome, Promotion-Rate.
- ✓ Tests: kompetitive Genome werden promotet; schwache Genome sparen Zusatz-Evals.

---

### ✓ P1 Adaptive Policy System

Verschiedene Adaptive-Mechanismen (Recovery, Diversity, Pop-Resize, Lamarck Burst, Fitness Shaping, Online Tuning, Surrogate-Gating) sind derzeit als separate Regelkreise implementiert.

**Aktueller Stand:** Vollstaendig implementiert. `PolicyRegistry` mit `register()`,
`set_order()`, `tick()` (observe → decide → apply), `Action`-Typ mit
`priority`/`conflict_group`, `TrainingContext`. Integration in `train()` per
Generation. `register_policy()`/`set_policy_order()`/`get_policy_diagnostics()`
API. Konfliktaufloesung: hoehere Priority gewinnt pro conflict_group. 7 Tests.

### ✓ P1 Modular Compatibility Distance / Genome Descriptor

Die aktuelle Kompatibilitaets-Distanz in der NEAT-Speziation ist auf Verbindungsanzahl und Gewichtsdifferenz beschraenkt.

**Aktueller Stand:** Vollstaendig implementiert (bereits vor diesem Loop). `DistanceMetric`-Protokoll, `TopologyDistance`, `WeightDistance`, `ActivationDistance`, `ChainMetric`. `set_compatibility_distance()` API. Tests in `test_diagnostics_features.py`.

### ✓ P1 Experiment Tracking / Run Database

Ausgangslage: Es gab kein zentrales `Run`-Objekt. Erkenntnisse aus
verschiedenen Konfigurationen mussten manuell aus CSV-Logs zusammengesetzt
werden; reproduzierbare Re-Runs waren ohne manuelle Konfigurationsdokumentation
nicht moeglich.

**Aktueller Stand:** Vollstaendig implementiert. `Run`, `Experiment` und
`RunDatabase` in `util/run_database.py`, SQLite-Persistenz, automatische Run-
Erfassung via `set_run_database()`, `experiment()`-Kontext, `start_run()` /
`finish_run()`, `load_run()`, `list_runs()` und `reproduce_run()`. `RunRecord`
bleibt als CSV-/History-Layer fuer bestehende Logs erhalten.

*Dieser Task ist der Daten-Layer. Generationsreport (Export-Layer) und Lernkurven-Vergleich (GUI-Layer) bauen darauf auf.*

**Datenmodell:**

```python
class Run:
    run_id: str       # UUID
    seed: int
    config: dict      # vollstaendige NeuroEvolution-Konfiguration (serialisierbar)
    fitness_history: list[dict]  # pro Generation: best, mean, iqr, species_count, ...
    diagnostics: dict  # Anomalien, Recovery-Events, Policy-Aktionen, Stop-Reason
    artifacts: dict   # Pfade zu Checkpoints, Plots, Reports
    start_time: str
    end_time: str
    stop_reason: str

class Experiment:
    experiment_id: str
    name: str
    runs: list[Run]
    tags: list[str]

class RunDatabase:
    def save_run(run: Run) -> None: ...
    def load_run(run_id: str) -> Run: ...
    def list_runs(experiment_id=None, tags=None) -> list[Run]: ...
    def compare_runs(run_ids: list[str]) -> ComparisonReport: ...
    def reproduce_run(run_id: str) -> NeuroEvolution: ...  # laedt Konfig + Seed
```

- `yane.set_run_database(path)`: aktiviert Tracking in SQLite oder JSON-Verzeichnis.
- `yane.experiment(name, tags=[])`: erstellt/waehlt ein Experiment als Kontext.
- Jeder Trainingslauf speichert sich automatisch als `Run`; `run_id` wird beim Start generiert.
- Export: JSON pro Run, SQLite fuer Abfragen.
- `RunDatabase.reproduce_run(run_id)`: rekonstruiert `NeuroEvolution`-Instanz aus gespeicherter Konfig und Seed.

**Tests:**

- Run wird gespeichert und vollstaendig geladen (Fitness-History, Konfig, Artefakte).
- `reproduce_run` erzeugt identische Konfiguration mit gespeichertem Seed.
- `compare_runs` gibt korrekte statistische Zusammenfassung ueber N Runs.
- Ohne `set_run_database` kein Performance-Overhead (kein automatisches Tracking).

### ✓ P1 Selektionsstrategie als Plugin

Ausgangslage: Die Selektionsstrategie (Tournament) war fest in `population.py`
verdrahtet; ein Austausch war ohne Code-Aenderung nicht moeglich.

**Aktueller Stand:** Vollstaendig implementiert. `SelectionStrategy`-Protokoll
in `evolution/selection_strategy.py`, eingebaute Strategien
`TournamentSelection`, `ElitistSelection`, `FitnessProportionalSelection`,
`RankSelection` und `NoveltyOnlySelection`, `set_selection_strategy()` API,
per-Species-Override und Diagnostics fuer aktive Strategie/Selektionsqualitaet.

- `SelectionStrategy`-Protokoll: `select(population, n) -> list[Genome]`.
- Eingebaute Strategien: `TournamentSelection(k=3)`, `ElitistSelection(top_frac=0.2)`, `FitnessProportional()`, `NoveltyOnlySelection()`, `RankSelection()`.
- `NeuroEvolution.set_selection_strategy(strategy)`.
- Per-Species-Ueberschreibung: `set_selection_strategy(strategy, species_id=...)`.
- Diagnostics: aktive Strategie, Durchschnittsfitness der selektierten vs. nicht-selektierten Genome.
- Tests: jede Strategie gibt korrekte Anzahl Genome zurueck; Fitness-Proportional ist stochastisch korrekt.

### ✓ P1 Evaluation-Middleware-Stack

Evaluatoren sind einfache Callables ohne Kompositionsmechanismus; Caching, Normierung und Noise-Injection muessen pro Evaluator manuell implementiert werden.

**Aktueller Stand:** Vollstaendig implementiert in `evolution/eval_middleware.py`.
`NeuroEvolution.add_eval_middleware()` und `clear_eval_middleware()` sind
verfuegbar; Middleware laeuft in LIFO-Reihenfolge. Eingebaut sind
`CachingMiddleware`, `TimingMiddleware`, `RetryMiddleware`,
`NoiseMiddleware`, `ComponentMiddleware` und `CaseBatchMiddleware` inklusive
Diagnostics. Eine eigene GUI-Detailansicht fuer Komponentenwerte ist noch nicht
separat ausgebaut, die Middleware-API selbst ist abgeschlossen.

- ✓ `EvalMiddleware`-Protokoll: `__call__(genome, eval_fn, ctx) -> float`.
- ✓ Eingebaute Middleware: `CachingMiddleware(maxsize=512)` (Genome-Hash → Fitness), `TimingMiddleware` (Eval-Zeit pro Genom), `RetryMiddleware(n=3, aggregation="mean")`.
- ✓ `NoiseMiddleware(sigma=0.05, n_samples=3, aggregation="mean")` (Weight-Perturbation).
- ✓ `ComponentMiddleware`: fuehrt mehrere benannte Fitness-Komponenten aus und
  schreibt Rohwerte + gewichteten Gesamtwert in Diagnostics.
- ✓ `CaseBatchMiddleware`: evaluiert feste Case-Listen (Train/Validation) mit
  konfigurierbarer Aggregation; Grundlage fuer MultiStart- und Curriculum-
  Beispiele.
- ✓ `yane.add_eval_middleware(mw)` haengt in die Kette ein.
- ✓ Middleware-Reihenfolge: LIFO (zuletzt hinzugefuegt = aeussere Schicht).
- ✓ Diagnostics: Cache-Hit-Rate, Durchschnittliche Eval-Zeit, Retry-Rate,
  Komponenten-Rohwerte, Komponenten-Gewichte, Case-Erfolgsraten.
- ✓ Tests: Middleware-Reihenfolge korrekt; Komponentenwerte bleiben getrennt
  sichtbar; Cache nutzt Genome-Fingerprint und invalidiert bei Gewichts-/Topologie-
  Aenderung; Retry wiederholt instabile Evaluatoren; Validation-Cases
  beeinflussen die Selektion nicht.

### ✓ P1 Generationsreport / Run-Postmortem

Nach einem Trainingslauf gibt es keine strukturierte Zusammenfassung; Erkenntnisse muessen manuell aus Logs und CSV herausgezogen werden.

**Aktueller Stand:** Vollstaendig implementiert. `util/report.py` mit
`export_run_report()` und drei Formaten: HTML (self-contained, inline SVG),
JSON (maschinenlesbar), Markdown (README-geeignet). SVG-Fitnesskurve mit
Best/Mean/Validation-Linien und Legende. Recovery-Events-Tabelle. Beste-Genom-
Topologie (Nodes + Connections). Config-Dump. `set_report_autosave()` fuer
automatischen Export nach `train()`. Keine RunDatabase-Abhaengigkeit.

*Export-Layer: baut nicht auf RunDatabase auf (liefert sofort nach train()).*

- ✓ `yane.export_run_report(path, fmt="html")` generiert Bericht nach `train()`.
- ✓ Fitness-Kurve (SVG-Inline, self-contained), beste Genome (Topologie + Score),
  Konfiguration, Recovery-Events, Runtime-Statistiken.
- ✓ HTML self-contained (kein externes CSS/JS); JSON maschinenlesbar.
- ✓ `yane.set_report_autosave(path_template)` mit `{name}`, `{date}`, `{example}`-Platzhaltern.
- ✓ 7 Tests (test_logging.py) + 4 Tests (test_diagnostics_features.py): HTML/JSON/MD-Formate,
  Autosave, invalid-format, Datei-Schreiben.

### ✓ P1 Automatisches Checkpoint-Rolling (Retention-Policy und Best-Tracking)

Checkpoints werden nur manuell gespeichert; der beste Zustand kann zwischen manuellen Saves verloren gehen und alte Checkpoints haeufen sich unbegrenzt an.

**Aktueller Stand:** Implementiert via
`NeuroEvolution.set_checkpoint_policy(interval=..., keep_best=True,
max_keep=..., path_template=...)`. Rolling-Checkpoints werden waehrend
`train()` geschrieben, alte Rolling-Dateien inklusive JSON-Sidecar werden nach
Retention entfernt, und der beste Checkpoint ist ueber
`get_best_checkpoint_path()` abrufbar.

- ✓ `NeuroEvolution.set_checkpoint_policy(interval=100, keep_best=True, max_keep=5, path_template="{run_name}_{kind}_{iteration}.pkl")`.
- ✓ `interval`: automatisches Speichern alle N Iterationen parallel zum Training.
- ✓ `keep_best=True`: Extra-Checkpoint wird immer geschrieben wenn neue Best-Fitness erreicht wird.
- ✓ `max_keep=N`: die aeltesten Rolling-Checkpoints werden geloescht sobald das Limit ueberschritten wird; Best-Checkpoint bleibt erhalten.
- ✓ `yane.get_best_checkpoint_path()` → Pfad zum besten gespeicherten Checkpoint.
- ✓ Diagnostics: letzter Auto-Save, Anzahl vorhandener Checkpoints, Pfad des Best-Checkpoints.
- ✓ Tests: Auto-Save nach Intervall; Retention entfernt alte Rollover-Dateien.

### ✓ P1 Generationsanzeige in der GUI (statt Iterationen)

Die GUI zeigt unter "Iteration:" die Anzahl einzelner Genome-Evaluierungen, nicht Generationen. Das ist irrefuehrend: mit pop_size=100 sieht der Nutzer "50000" und denkt er haette 50.000 Generationen trainiert — tatsaechlich waren es 500.

**Aktueller Stand:** Implementiert. `population_memory_info()` enthaelt
`generation`; GUI zeigt "Generation" und separate "Evaluations"; CSV/JSONL
enthalten `generation`, und CSV enthaelt `validation_fitness`.

- Neuen Zaehler `generation = iteration // pop_size` im Training-Loop fuehren; in `population_memory_info()` als `"generation"` Key eintragen.
- GUI Training-Tab: "Iteration:" Label umbenennen zu "Generation:" und den Generations-Zaehler anzeigen; Rohe Evaluierungsanzahl als Tooltip oder sekundaeres Label ("50.000 Evaluierungen").
- CSV-Log und JSONL-Log: Spalte `generation` hinzufuegen (vor oder neben `iteration`).
- `max_iterations` Semantik bleibt unveraendert (zaehlt Evaluierungen); optionales `max_generations` als Alias (`max_generations * pop_size` → `max_iterations`).
- Tests: `population_memory_info()["generation"]` ist korrekt; GUI-Label zeigt Generations-Zaehler; CSV enthaelt `generation`-Spalte.

### ✓ P1 Hybrid Feature-Extractor API (Input-Vorverarbeitung)

NEAT eignet sich schlecht fuer hochdimensionale Eingaben (Bilder, Sensorrauschen), weil der Suchraum explodiert. Mit einer vorgeschalteten Transformation — CNN-Layer, PCA, Autoencoder — reduziert NEAT den Suchraum auf das semantisch relevante.

**Aktueller Stand:** Vollstaendig implementiert (bereits vor diesem Loop). `set_input_transform(fn)` mit Dimensionalitaets-Validierung, `_run_evaluations()`- und `_run_with_matrix_forward()`-Integration. 6 Tests in `test_diagnostics_features.py`.

### ✓ P1 Fitness-Landscape-Visualisierung (PCA / t-SNE)

Konvergenzverhalten und Populationsstruktur sind nur ueber Zahlen erkennbar.

**Aktueller Stand:** Vollstaendig implementiert. `GenomeDescriptor`
(12D-Featurevektor aus Topologie/Gewichten/Biases), `population_pca()` via
Power-Iteration/SVD (keine externen Abhaengigkeiten) und `landscape_pca()` API
auf `NeuroEvolution` sind vorhanden und getestet. CSV- und PNG-Export sind als
Core-Helper und `NeuroEvolution`-Wrapper verfuegbar. Die GUI zeigt den Snapshot
als Scatterplot und kann die Daten als PNG/CSV exportieren.

### ✓ P1 Populations-Filter und -Aggregatoren API

Analyse von Populationszustaenden erfordert direkten Zugriff auf interne Listen; keine saubere funktionale API.

**Aktueller Stand:** Implementiert fuer evaluierte Genome:
`filter`, `map`, `reduce`, `group_by`, `top_k`.

- `population.filter(fn: Genome -> bool) -> list[Genome]`: selektiert Genome nach Praedikat.
- `population.map(fn: Genome -> T) -> list[T]`: transformiert Genome zu beliebigen Werten.
- `population.reduce(fn: (acc, Genome) -> acc, init) -> acc`: Fold ueber Population.
- `population.group_by(fn: Genome -> K) -> dict[K, list[Genome]]`: Gruppierung (z. B. nach Species oder Aktivierungstyp).
- `population.top_k(k, key=lambda g: g.fitness) -> list[Genome]`.
- Alle Methoden mit Typ-Annotationen; kein Overhead wenn nicht verwendet.
- Tests: Filter, Map, Group-by auf kleiner Test-Population; Top-k-Reihenfolge korrekt.

### ✓ P1 Connection-Weight-Histogramm und Gewichtsgesundheit

Pathologien wie Vanishing/Exploding Weights oder symmetrische Gewichtsverteilungen sind in den Diagnostics unsichtbar.

**Aktueller Stand:** Vollstaendig implementiert. `WeightHistogram`-Widget in
`gui/canvas.py`. `export_best_weights_npy()` API. N-Generationen-Streak-Tracking
ist in Diagnostics vorhanden; die GUI zeigt Histogramm/Gesundheitswerte, aber
nicht jedes interne Warn-Streak-Detail.

### ✓ P1 Erweiterbare Aktivierungsfunktionen

Das Aktivierungsset (sigmoid, tanh, relu, leaky_relu, swish, linear) ist im Code fest verdrahtet.

**Aktueller Stand:** Vollstaendig implementiert. `CUSTOM_ACTIVATION_FNS`-Dict,
`register_activation()`, `resolve_activation_fn()`, `list_activations()`.
`Node.activation` akzeptiert `ActivationType | str`; Mutation ueberspringt
custom-Activations. GELU, Mish, SiLU als eingebaute Erweiterungen (skalar +
batch). Pickle-Roundtrip funktioniert durch Rekonstruktion von `_activate_fn`
in `__setstate__`. Matrix-Export unterstuetzt custom Namen.

- ✓ `NeuroEvolution.register_activation(name, fn)` registriert benutzerdefinierte Aktivierungen zur Laufzeit.
- ✓ Registrierte Funktionen werden im Checkpoint gespeichert (Name-Stichwort reicht, da Funktionen beim Laden registriert sein muessen).
- ✓ GELU, Mish, SiLU als eingebaute Erweiterungen (skalar + batch).
- ✓ Kein Breaking Change: `ActivationType.RELU` funktioniert weiterhin.
- ✓ 14 Tests: Registry-API, Node-Property, Forward, Mutation-Schutz, GELU/Mish/SiLU-Werte, Pickle-Roundtrip.

### ✓ P1 Multi-Population Inselmodell

**Aktueller Stand:** Vollstaendig implementiert. `IslandModel` in `evolution/islands.py` mit N unabhaengigen `Population`-Instanzen, periodischer Migration der Top-Genome zwischen zufaellig gepaarten Inseln. `set_island_model()` API. Diagnostics: Fitness/Stagnation/Species pro Insel, Migrations-Events. 5 Tests.

### ✓ P1 Hyperparameter-Suche

**Aktueller Stand:** Vollstaendig implementiert. `hyperparameter_search()` in `evolution/hyperparameter_search.py`. Grid-Search und Random-Search ueber Parameter-Grid. Mehrere Seeds pro Konfiguration. Ranking nach medianer Fitness. RunDatabase-Integration. 3 Tests.

### ✓ P1 Ensemble-Bewertung und -Deployment

Ein einzelnes Genom ist stochastisch; ein Ensemble aus den Top-K ist robuster.

**Aktueller Stand:** Vollstaendig implementiert. `EnsembleGenome`-Wrapper mit
`forward()`, `to_python()`. Strategien: mean, vote, weighted (fitness-gewichtet).
`make_ensemble(k, mode)` API. Export erzeugt standalone Python mit
`memberN_forward()` + `ensemble_forward()`. 7 Tests.

### ✓ P1 Strukturierte / maschinenlesbare Protokollierung

**Vollstaendig implementiert.** `set_log_format("jsonlines"|"csv"|"both")`,
`set_tensorboard_logdir(path)` und `set_log_callbacks(on_generation=fn)`
existieren und funktionieren. CSV enthaelt `validation_fitness`-Spalte sowohl
im teuren als auch im billigen Log-Pfad; JSONL enthaelt die erweiterten
Diagnostics.

### ✓ P1 Erweiterte Genome-Analyse im Inspect (Sensitivitaet / Attribution)

Nutzer sehen Ausgaben, aber nicht WARUM das Genom so entscheidet.

**Aktueller Stand:** Vollstaendig implementiert (bereits vor diesem Loop). `genome.sensitivity_analysis()` und `genome.dead_nodes()` in `core/genome.py`. `SensitivityChart`-Widget in `gui/canvas.py`. Tests in `test_diagnostics_features.py`.

### ✓ P1 Plugin-System fuer benutzerdefinierte Evaluatoren

Ausgangslage: Eigene Umgebungen einzubinden erforderte direktes Editieren der
`examples.py`.

**Aktueller Stand:** Vollstaendig implementiert. `PLUGIN_EXAMPLES`-Liste,
`register_example(plugin)`, `load_plugins_from_directory()` fuer explizites
Laden und `autoload_user_plugins()` als opt-in Autoload aus `~/.yane/plugins/`.
Plugin-Dateien definieren eine `register(reg)`-Funktion.
`NeuroEvolution.register_example()` als Static-Method-API. 7 Tests.

### ✓ P1 Lernkurven-Vergleich (mehrere Runs)

Einzelne Runs sind nicht repraesentativ; Vergleich verschiedener Konfigurationen ist nicht moeglich.

**Aktueller Stand:** Implementiert. `ComparisonTab` in
`gui/tabs/comparison_tab.py` laedt Runs, zeichnet ueberlagerte Fitness-Kurven,
berechnet Statistik inklusive Median/IQR-Baendern fuer vergleichbare Runs und
exportiert PNG/CSV. Baut auf Experiment Tracking und `RunRecord`-Kompatibilitaet
auf.

*GUI-Layer: laedt mehrere `Run`-Objekte aus der RunDatabase und visualisiert sie vergleichend.*

- GUI: „Vergleich"-Ansicht mit ueberlagerten Fitness-Kurven (bis zu 4 Runs, farbcodiert).
- Statistische Zusammenfassung: Median, 25./75.-Perzentil ueber N Wiederholungen derselben Konfig.
- Export: Vergleichs-Plot als PNG, Rohdaten als CSV.

### ✓ P1 Fitness-Surrogate-Modell (Billigfilter vor teurer Evaluierung)

**Aktueller Stand:** Vollstaendig implementiert. `FitnessSurrogate` in `evolution/surrogate.py`. Lineares Modell (OLS mit Ridge) ueber 12D-Genome-Deskriptor-Vektoren. Warmup-Phase, adaptive Filterung der unteren `surrogate_frac`. Spearman-Rho-Diagnostik. 5 Tests.

### ✓ P1 Automatische Fitness-Shaping-Erkennung

Die Fitness-Landschaft (sparse, plateau, skewed) ist unsichtbar; der Nutzer muss manuell entscheiden welche Fitness-Transformationen noetig sind.

**Aktueller Stand:** Vollstaendig implementiert. `FitnessLandscapeAnalyzer` mit
`analyze()` → `FitnessLandscapeReport` (Sparsity, Plateau, Skewness, Cluster-
Separability) und `recommend_transform()`. Automatische Anwendung alle 50
Generationen via `set_auto_fitness_shaping()`. 8 Tests.

### ✓ P1 Online-Hyperparameter-Adaptation (Bandit-Tuning waehrend Training)

Statische Hyperparameter sind fuer alle Trainingsphasen gleich.

**Aktueller Stand:** Vollstaendig implementiert. `UCB1Bandit`-Klasse in
`evolution/online_tuning.py`. `set_online_tuning()` mit Parametern
`mutation_rate` und `n_lamarck_steps`. UCB1-Formel mit exploration/
exploitation-Phase. Fitness-Delta als Reward. 8 Tests.

### ✓ P1 Weight-Inheritance beim Crossover (Lamarck-informierte Gewichts-Initialisierung)

Beim Crossover werden Gewichte neuer Verbindungen (die nur ein Elternteil hat) zufaellig initialisiert; Lamarck-optimierte Elterngewichte werden nicht weitervererbt.

**Aktueller Stand:** Vollstaendig implementiert. `NeuroEvolution.set_weight_inheritance()` mit `blend_alpha`, Propagation zur Population auf configure/checkpoint/API. Matching-Connections blended: `blend_alpha * w_fitter + (1-blend_alpha) * w_weaker`. Default -1.0 = 50/50-Zufall (unchanged). Bias ebenso blended. Aktivierung bleibt 50/50 (Enum).

- ✓ `NeuroEvolution.set_weight_inheritance(enabled=True, blend_alpha=0.7)`: `blend_alpha` steuert Gewichtung des besseren Elternteils.
- ✓ Tests: Blend liegt im erwarteten Bereich; `blend_alpha=0` = schwaecheres Elterngewicht; `blend_alpha=1` = fitteres Elterngewicht; negative Alpha = 50/50-Zufall; API-Fehlerbehandlung; Config-Dict-Serialisierung; Populations-Propagation.

---

### ✓ P2 Genome-Codec-Protokoll (austauschbare Serialisierung)

**Aktueller Stand:** Vollstaendig implementiert (vorheriger Loop). `GenomeCodec`-Protokoll, `PickleCodec`, `JsonCodec`, `set_checkpoint_codec()`, `migrate_checkpoint()`, `detect_codec()`. 9 Tests.

### ✓ P2 Konfigurationsversionierung und Kompatibilitaets-Check

Checkpoints enthalten die Population, aber nicht den vollstaendigen Zustand der Konfiguration; spaeters Nachladen kann zu stillem Fehlverhalten fuehren.

**Aktueller Stand:** Vollstaendig implementiert. Checkpoint-Metadaten enthalten
Version, Topologie-Daten, Konfigurations-Hash und Konfigurationsdaten.
`CompatibilityLevel`, `compatibility_report()` und `check_compatibility()`
weisen Hash-Abweichungen als strukturierten Diff aus und unterscheiden
`EXACT`, `COMPATIBLE` und `BREAKING`. Neue Checkpoints speichern den Hash auch
im Payload; `read()` validiert ihn und `load_checkpoint()` blockt beim Laden in
eine bereits konfigurierte, inkompatible Instanz. GUI-Metadaten zeigen den
Konfigurations-Hash; `python -m yane.checkpoint --diff old.pkl new.pkl` gibt
einen maschinenlesbaren Diff aus.

- ✓ Jede Konfiguration erhaelt einen deterministischen Konfigurations-Hash (SHA-256 ueber kanonisches JSON).
- ✓ Beim `load_checkpoint()`: Hash wird validiert; bei Abweichung erscheint strukturierter Diff der geaenderten Felder.
- ✓ `CompatibilityLevel`: `EXACT` (identisch), `COMPATIBLE` (nur unkritische Felder geaendert), `BREAKING` (Inputs/Outputs/Topologie-Constraints geaendert).
- ✓ GUI: Checkpoint-Metadaten zeigen Konfig-Hash und Aenderungs-Diff, falls vorhanden; `BREAKING` wird beim Load als Fehler blockiert.
- ✓ CLI: `python -m yane.checkpoint --diff old.pkl new.pkl` zeigt Konfigurations-Unterschiede.

### ⚡ P2 Transfer Learning / Genome Fine-Tuning

Wissen aus einem trainierten Genom soll auf eine neue Aufgabe uebertragen werden.

**Aktueller Stand:** Teilweise implementiert. `warm_start_from_checkpoint(path)`
laedt Genome aus einem Checkpoint und passt Eingabe/Ausgabe-Dimension an.
`load_genome_as_seed()` seeded die Population tatsaechlich mit einem gegebenen
Genom. Freeze speichert die urspruenglichen Mutations-/Spike-Raten pro
Connection-Innovation und `set_transfer_unfreeze()` kann eingefrorene Teile
waehrend des Trainings schrittweise entsperren. `fine_tune_genome()` bietet
Lamarck-only-Feinabstimmung ohne Topologie-Aenderung; `behaviour_clone()` kann
das geklonte Genom direkt als Population-Seed setzen. Transfer-Benchmarks
fehlen noch.

- ✓ `yane.load_genome_as_seed(genome, freeze_layers=[...])`: bestimmte Verbindungsgruppen koennen eingefroren werden.
- ✓ Lamarck-Feinabstimmung auf neuer Aufgabe ohne Topologie-Aenderung als erste Phase.
- ✓ Dann schrittweise Entsperren eingeforener Teile (progressive unfreeze).
- Benchmark: Transfer CartPole → LunarLander vs. Training from scratch.

### ⚡ P2 Offene Evolution / Co-Evolution von Aufgabe und Agent (POET-aehnlich)

**Aktueller Stand:** Experimenteller Spike in `evolution/experimental.py`.
`CoevolutionPool` und `CoevolutionPair` modellieren einfache Agent-/Environment-
Paare. Eine vollstaendige POET-aehnliche Integration mit `EnvironmentGenome`,
Archiv, Survival-Regeln, Transfer zwischen Aufgaben und Trainingsloop fehlt.

### ✓ P2 YANE → PyTorch-Bruecke (NAS + Feinabstimmung)

**Aktueller Stand:** Vollstaendig implementiert. `genome_to_torch_module()` in `evolution/torch_bridge.py` exportiert Genome als `torch.nn.Module` mit exakter Topologie und Gewichten. `forward_with_torch()` mit Fallback. Memory-Knoten via GRUCell. Test (skipped ohne torch).

### ⚡ P2 Genome-Phylogenie (Stammbaum der Innovationen)

**Aktueller Stand:** Teilweise implementiert. Genome tragen IDs und Parent-IDs;
`InnovationTracker` kann Crossover, Innovationen und Ahnenketten erfassen.
Fitness-Delta-Attribution pro Innovation, Baum-/Graph-Analyse, Export und GUI-
Visualisierung fehlen noch.

### ⚡ P2 Verhaltensklonierung als Warm-Start

Evolution braucht viele Iterationen bis zu brauchbaren Loesungen; Demonstrationen koennen das beschleunigen.

**Aktueller Stand:** Teilweise implementiert. `behaviour_clone()` optimiert ein
Genom mit Demonstrationspaaren ueber eine einfache lokale Suche und gibt das
gekloente Genom zurueck. Es wird noch nicht automatisch als Population-Seed
verdrahtet; Backprop-/Torch-Training, Lamarck-Integration und Benchmarks fehlen.

- `yane.behaviour_clone(demonstrations, n_steps)`: supervised Vortraining des besten Genoms auf Demonstrations-Daten via Lamarck/Backprop.
- Demonstrationen als Liste von `(inputs, outputs)`-Paaren; kein RL-Umgebungsformat noetig.
- Geklontes Genom wird als initiales Seed fuer die Population verwendet.
- Benchmark: BC-Warm-Start vs. random-init auf LunarLander.

### ✓ P2 Population-Size-Adaptation (Dynamische Pop-Groesse)

Die Populationsgroesse ist nach Start fest; in fruehen Phasen ist eine grosse Population fuer Exploration noetig, spaeter waere eine kleine Population effizienter.

**Aktueller Stand:** Vollstaendig implementiert. `set_adaptive_pop_size(min_pop, max_pop, schedule="performance_based"|"linear_decay", growth_rate, enabled)` als spec-konforme API. `linear_decay` senkt die Groesse monoton (mit optionalem `_adaptive_pop_total_gens` aus `max_iterations`), `performance_based` nutzt Stagnations- und Spezies-Diversitaetssignale. Debounce-Fix: `_last_pop_adjust_spawn`-Delta statt `% max_size` (funktioniert korrekt wenn `max_size` sich aendert). Diagnostics in `population_memory_info()`: `current_pop_size`, `last_resize_trigger`, `pop_size_history`. 8 neue Tests in `test_adaptive_population.py`.

- `NeuroEvolution.set_adaptive_pop_size(min_pop=20, max_pop=500, schedule="linear_decay"|"performance_based")`.
- `linear_decay`: Pop-Groesse nimmt linear von `max_pop` zu `min_pop` ueber den Trainingsverlauf ab.
- `performance_based`: Pop-Groesse sinkt wenn Konvergenzrate hoch ist, steigt bei Stagnation.
- Groesse wird nur an Generationsgrenzen geaendert; Ueberschuss-Genome werden per Fitness-Selection eliminiert.
- Diagnostics: aktuelle Pop-Groesse, letzter Resize-Trigger, Groessen-Historie.
- Tests: Pop-Groesse bleibt in `[min_pop, max_pop]`; bei `linear_decay` sinkt sie monoton; kein Resize innerhalb einer Generation.

### ✓ P2 Gradient-gesteuerte Mutations-Richtung (Lamarck-Momentum)

Lamarck-Gradienten werden nach jedem Refinement-Schritt verworfen; sie koennten die Mutations-Richtung fuer die naechste Generation informieren.

**Aktueller Stand:** `LamarckRefiner` (Hill-Climb, NES, SA, CMA-ES) berechnet Gewichts-Deltas intern fuer das Refinement, gibt sie aber nach Abschluss nicht weiter. Mutation und Lamarck sind zwei komplett unabhaengige Mechanismen ohne Informationsfluss.

- `LamarckRefiner` speichert den Parameteraenderungs-Vektor (Gradient-Schaetzung) des letzten Schritts.
- Beim Mutieren: mit Wahrscheinlichkeit `momentum_prob` wird die Mutations-Richtung mit dem gespeicherten Momentum gewichtet.
- Momentum-Decay: Gradient-Information verfaellt exponentiell (`decay=0.9`).
- `NeuroEvolution.set_lamarck_momentum(enabled=True, momentum_prob=0.3, decay=0.9)`.
- Benchmark: Mit vs. ohne Momentum auf Symbolic-Regression und CartPole (Konvergenzgeschwindigkeit).
- Tests: Momentum-Vektor wird nach Lamarck-Schritt aktualisiert; bei `momentum_prob=0` kein Einfluss auf Mutation.

### ✓ P2 Automatisches Post-Training Pruning (Netzwerk-Komprimierung)

Trainierte Genome sind oft ueberdimensioniert; unnoetige Verbindungen und tote Knoten koennen ohne Leistungsverlust entfernt werden.

**Aktueller Stand:** Vollstaendig implementiert. `set_post_training_pruning(enabled, threshold, max_drop_frac)` in `NeuroEvolution`. Nach `train()` wird das beste Genom gepruned; Fitness-Rollback wenn Drop > `max_drop_frac`. `genome.prune()` und `genome.compress()` fuellen `_prune_stats` mit echten Werten. `genome.prune_stats()` gibt eine unabhaengige Kopie zurueck. 12 Tests in `test_post_training_pruning.py`.

- `genome.prune(threshold=0.01, method="weight"|"activation_frequency")` entfernt Verbindungen unter Schwellenwert.
- `genome.compress(target_size)`: iteratives Pruning bis Zielgroesse erreicht (kleinstes Gewicht zuerst).
- `NeuroEvolution.set_post_training_pruning(enabled=True, threshold=0.01, max_drop_frac=0.02)`: automatisches Pruning nach `run_end`.
- Evaluierung: gepruntes Genom wird einmal neu bewertet; wenn Fitness-Drop groesser als `max_drop_frac` wird Pruning rueckgaengig gemacht.
- `genome.prune_stats()`: Anzahl entfernter Verbindungen/Knoten, Fitness-Delta, Komprimierungsrate.
- Tests: `prune()` entfernt Verbindungen unter Schwellenwert; `compress()` erreicht Zielgroesse; Fitness-Check aktiviert Rollback bei zu grossem Drop.

---

### ✓ P2 Differenzierbare Topologie-Suche (DARTS-Lite)

**Abgeschlossen:** Gate-Werte per Connection (`_darts_gates: dict[int, float]`)
werden jede Generation aus `sigmoid(|weight| * 2)` aktualisiert. Post-Training
Pruning entfernt Connections mit Gate < threshold vom besten Genom.
`NeuroEvolution.set_darts_mode(enabled, prune_threshold)` steuert das Feature.
Gates werden bei `copy()` und `crossover()` vererbt. Tests in
`tests/test_darts.py` (12 Tests).

### ✓ P2 Intrinsische Belohnung / Curiosity-Modul

**Abgeschlossen:** `IntrinsicCuriosityModule` (2-Layer-Vorhersagenetz
n_inputs → hidden → n_outputs) ist in `_run_evaluations()` integriert.
`genome.forward()` wird instrumentiert um Input/Output-Paare zu erfassen;
Vorhersagefehler wird als Bonus zur Fitness addiert. Online-SGD mit
Gradient-Clipping aktualisiert das Vorhersagenetz nach jeder Evaluation.
`NeuroEvolution.set_curiosity(enabled, weight, network_size, lr)` steuert das
Feature. Tests in `tests/test_curiosity.py` (14 Tests).

### ⚡ P2 Synaptische Plastizitaet (STDP / Hebbsches Lernen)

Genome lernen derzeit nur durch Evolution, nicht durch Erfahrung innerhalb einer Episode.

**Aktueller Stand:** Experimenteller Spike. `STDPRule` existiert als isolierte
Regel fuer Gewichtsanpassungen. Genome speichern aber noch keine evolvierbaren
Plastizitaetskoeffizienten, `genome.forward()` veraendert Gewichte nicht
episoden-lokal und `genome.reset()` verwaltet keine Basis-/Arbeitsgewichte.

- Knoten/Verbindungen koennen evolvierte Hebb-Regel-Koeffizienten (A, B, C, D) tragen.
- Gewichte werden waehrend `genome.forward()` nach der STDP-Regel angepasst (intra-lifetime-learning).
- `genome.reset()` setzt Gewichte auf Basiswerte zurueck (Plastizitaet ist episoden-lokal).
- Benchmark: STDP vs. Lamarck auf Aufgaben mit veraenderlicher Umgebung (z. B. wechselnde XOR-Eingaenge).

### ⚡ P2 Neuromodulation

Modulatorische Signale erlauben kontextabhaengige Gewichtung ganzer Verbindungsgruppen.

**Aktueller Stand:** Experimenteller Spike. `Neuromodulator` existiert als
isolierter Skalierungshelfer. Es gibt noch keinen `MODULATOR`-Knotentyp, keine
evolvierbaren Modulationskanten und keine Integration in `genome.forward()`.

- Sonderknotentyp `Modulator`: sein Ausgabe-Wert skaliert eingehende Verbindungen anderer Knoten.
- Evolvierbar: welcher Knoten moduliert, welche Verbindungen beeinflusst werden, Staerke.
- Anwendung: schnelle Anpassung an wechselnde Aufgaben (Multi-Task-Szenarien).
- Benchmark: Modulation vs. kein Modulation auf einem Aufgaben-Wechsel-Szenario.

### ⚡ P2 Evolutionaere Input-Gruppierung (Evolvable Input Aggregation Layer)

NEAT behandelt jeden Input als unabhaengigen Knoten; bei hochdimensionalen Eingaben (Sensoren, Pixel-Gruppen) entstehen so viele Verbindungen, dass der Suchraum explodiert. Ein evolvierbarer Aggregations-Layer vor dem eigentlichen NEAT-Netz reduziert die Anzahl der effektiven Input-Knoten durch gelerntes Zusammenfassen von Inputs — ohne dass der Nutzer vorher wissen muss welche Inputs zusammengehoeren.

Verwandte Konzepte: Capsule Networks (Hinton 2017) gruppieren semantisch aequivalente Features; CoSyNE co-evolviert Eingabe-Gewichte; ES-HyperNEAT bestimmt Knoten-Positionen im Substrat automatisch. Das Neue hier ist die Kombination: evolvierbare Gruppen mit evolvierbaren Aggregations-Operatoren als dynamisch wachsender Pre-Layer, der vollstaendig in NEAT-Mutations- und Crossover-Logik integriert ist.

**Aktueller Stand:** Experimenteller Spike. `InputGrouping` kann externe Inputs
standalone reduzieren und aggregieren. Es ist aber kein Teil von `Genome`, wird
nicht gecrossovert oder gecheckpointet, erzeugt keine dynamischen Input-Knoten
und hat keine `NeuroEvolution.set_input_grouping()`-Integration.

**Design: `InputGroup` und `InputGrouper`**

- `InputGroup`: `members: list[int]` (Raw-Input-Indices), `aggregation: AggType` (mean / max / sum / weighted_sum), `weights: list[float]` (nur fuer weighted_sum, evolvierbar), `enabled: bool`.
- `InputGrouper`: Liste von `InputGroup`; ist Teil des `Genome`-Objekts (wird beim Checkpoint mitgespeichert und gecrossoverd).
- Forward-Pass: `grouper.transform(raw_inputs) -> list[float]` — liefert einen Wert pro Gruppe als effektive Inputs fuer das NEAT-Netz.

**Mutations-Operatoren (evolutionaer, nicht datengetrieben):**

- `create_group(indices)`: Auswahl zunaechst zufaellig (kein Korrelations-Algorithmus — das Lernen passiert durch Selektion, nicht durch Initialisierung).
- `split_group(group)`: Gruppe in zwei Teilgruppen aufteilen; erfordert neuen Input-Knoten im Genome.
- `merge_groups(g1, g2)`: Zwei Gruppen zusammenfassen; loescht einen Input-Knoten.
- `add_input_to_group(raw_index, group)`: Einzelnen Input in bestehende Gruppe verschieben.
- `remove_input_from_group(raw_index, group)`: Input aus Gruppe herausloesen; bei Gruppe-Groesse 1 → Gruppe deaktivieren.
- `change_aggregation(group, new_agg)`: Aggregations-Funktion wechseln.

**Dynamische Input-Knoten-Anzahl (harter Teil):**

- Wenn `split_group` einen neuen Knoten erzeugt oder `merge_groups` einen loescht, passt der Grouper die Anzahl der Input-Knoten im Genome automatisch an (analog zu `warm_start_from_checkpoint`-Logik).
- Crossover: zwei Eltern-Genome koennen unterschiedliche Grouper-Strukturen haben; Alignment erfolgt per Gruppen-Innovation-Nummer (analog zu Connection-Innovations).
- `NeuroEvolution.set_input_grouping(enabled=True, initial_groups=None)`: aktiviert den Layer; `initial_groups=None` → jeder Input startet als eigene Gruppe (Nullhypothese: kein Gruppiervorteil).

**Abgrenzung:**

- Hybrid Feature-Extractor (P1-Task): fest vom Nutzer vorgegeben, nicht evolviert.
- Shared Weights (P2-Task): teilt Gewichte zwischen Verbindungen, gruppiert keine Inputs.
- HyperNEAT: erzeugt raeumliche Gewichtsmuster, bestimmt aber nicht welche Inputs zusammengefasst werden.

**Benchmark:** Input-Gruppierung vs. kein Grouper auf CartPole (4 Inputs → Baseline), auf einem Sensor-Array-Task mit 50 korrelierten Inputs (Gruppen sollten konvergieren), auf MNIST mit Pixel-Gruppierung (784 → ~50 Gruppen als Vorstufe zu Convolutional NEAT).

- Tests: `transform()` produziert korrekte Ausgabe-Dimension; `split_group()` fuegt korrekten Input-Knoten hinzu; Crossover zweier Genome mit unterschiedlichen Groupern laeuft ohne Fehler; Checkpoint-Round-Trip erhaelt Gruppen.

### ⚡ P2 Evolutionaere Output-Gruppierung (Evolvable Output Synergy Layer)

Symmetrisch zur Input-Gruppierung — aber die externe Schnittstelle bleibt unveraendert. Von aussen sieht das Genome weiterhin N Ausgabe-Kanaele; intern werden Outputs die immer gemeinsam aktiviert werden unter einem geteilten Proto-Output-Knoten zusammengefasst. Das NEAT-Netz evolviert nur noch K < N Proto-Knoten, der `OutputGrouper` expandiert diese transparent zurueck auf die N extern erwarteten Ausgabe-Werte. Weder die Fitness-Funktion noch der Nutzer-Code muss geaendert werden.

Beispiel: Aktion 3 und Aktion 7 werden in einer Aufgabe stets gleichzeitig ausgefuehrt. Ohne Gruppierung evolviert NEAT zwei separate Output-Nodes mit eigenen Verbindungsbaumen — doppelter Suchaufwand. Mit Gruppierung teilen sich beide einen Proto-Knoten; das Netz lernt die Synergy einmal, der Grouper verteilt sie auf beide Slots.

Anwendungsfaelle: hochdimensionale Steuerung (Roboterarm mit 20 Gelenken, von denen 4 Gruppen immer synchron feuern), multi-label Klassifikation mit korrelierten Klassen, jede Aufgabe mit korrelierten Ausgabe-Kanaelen.

**Aktueller Stand:** Experimenteller Spike. `OutputGrouping` kann Proto-Outputs
standalone auf externe Outputs expandieren. Es ist aber kein Teil von `Genome`,
veraendert keine interne Output-Knoten-Anzahl, wird nicht gecrossovert oder
gecheckpointet und ist nicht in `genome.forward()`/`genome_to_python()`
integriert.

**Design: `OutputGroup` und `OutputGrouper`**

- `OutputGroup`: `targets: list[int]` (externe Ausgabe-Slot-Indices), `expansion: ExpType` (copy / scale / affine), `weights: list[float]` (Skalierungs- oder Affin-Koeffizienten je Ziel-Slot, evolvierbar), `enabled: bool`.
- `OutputGrouper`: Liste von `OutputGroup`; Teil des `Genome`-Objekts (Checkpoint und Crossover analog zur Input-Gruppierung).
- Forward-Pass: NEAT erzeugt K Proto-Output-Werte → `grouper.expand(proto_outputs) -> list[float]` der Laenge N (unveraenderte externe Ausgabe-Dimension).
- Externe Nutzung: `genome.forward(inputs)` gibt weiterhin N Werte zurueck — die Gruppierung ist vollstaendig intern.

**Expansions-Typen:**

- `copy`: alle Ziel-Slots erhalten denselben Proto-Wert (staerkste Synergy-Annahme, ein Knoten fuer identische Aktionen).
- `scale`: Ziel-Slot i erhaelt `proto * weights[i]` (proportionale Verteilung, evolvierbare Skalierung).
- `affine`: Ziel-Slot i erhaelt `proto * weights[i] + biases[i]` (feinste Steuerung; Bias evolvierbar).

**Mutations-Operatoren (evolutionaer, nicht datengetrieben):**

- `create_group(output_indices)`: Mehrere externe Slots unter einem neuen Proto-Knoten zusammenfassen; reduziert interne Output-Node-Anzahl um `len(indices) - 1`.
- `split_group(group)`: Proto-Knoten in zwei unabhaengige Knoten aufteilen; erhoeh interne Output-Knoten-Anzahl um 1.
- `merge_groups(g1, g2)`: Zwei Proto-Knoten verschmelzen; reduziert interne Output-Knoten-Anzahl um 1.
- `add_output_to_group(slot_index, group)`: Externen Ausgabe-Slot einer bestehenden Gruppe zuordnen.
- `remove_output_from_group(slot_index, group)`: Ausgabe-Slot aus Gruppe herausloesen; bei Gruppen-Groesse 1 → direkter Pass-Through (kein Expansions-Overhead).
- `change_expansion(group, new_exp)`: Expansions-Typ wechseln.

**Dynamische interne Output-Knoten-Anzahl (harter Teil):**

- `create_group` / `merge_groups` reduzieren, `split_group` erhoeh die Anzahl der NEAT-internen Output-Nodes — analoges Problem zur Input-Gruppierung.
- Crossover: Eltern koennen unterschiedliche Grouper-Strukturen haben; Alignment per Gruppen-Innovation-Nummer.
- `genome_to_python()` emittiert den Expand-Block nach den Proto-Output-Nodes; externe Ausgabe-Dimension bleibt N.
- `NeuroEvolution.set_output_grouping(enabled=True, initial_groups=None)`: `initial_groups=None` → jeder externe Slot startet als eigene Gruppe (Nullhypothese: kein Synergy-Effekt).

**Zusammenspiel mit Input-Gruppierung:**

Beide Layer koennen gleichzeitig aktiv sein: `InputGrouper.transform(raw_inputs) → NEAT (K_in Inputs, K_out Outputs) → OutputGrouper.expand(proto_outputs) → actual_outputs`. Die externe Schnittstelle zeigt weiterhin N_in Inputs und N_out Outputs; NEAT arbeitet intern auf einem komprimierten Raum.

**Abgrenzung:**

- Evolutionaere Input-Gruppierung (P2-Task): Aggregation (viele externe Inputs → wenige interne), externe Input-Dimension erscheint unveraendert.
- Shared Weights (P2-Task): teilt Gewichtswerte zwischen Verbindungen, aendert weder Input- noch Output-Dimension.
- Ensemble-Bewertung (P1-Task): kombiniert *mehrere Genome* — Output-Gruppierung operiert *innerhalb eines Genoms*.

**Benchmark:** BipedalWalker (4 unkorrelierte Outputs → Null-Hypothese: kein Grouping entsteht), synthetischer Synergy-Task (20 Outputs, 5 Gruppen mit bekannten Korrelationen → Gruppen sollten konvergieren), multi-label Klassifikation mit korrelierten Klassen.

- Tests: `expand()` gibt stets N Werte zurueck; `genome.forward()` gibt unveraendert N Werte zurueck; `split_group()` erhoeh interne Output-Knoten-Anzahl um 1; Crossover zweier Genome mit unterschiedlichen OutputGroupern laeuft fehlerfrei; `genome_to_python()` erzeugt korrekten Expand-Block.

### ✓ P2 Shared Weights (Weight-Sharing zwischen Verbindungen)

**Abgeschlossen:** `Connection.weight_group: str | None` (neuer Slot) weist
Verbindungen einer Gruppe zu. `genome.weight_groups: dict[str, float]` haelt
den kanonischen Gewichtswert pro Gruppe; `genome.sync_shared_weights()` pusht
ihn zu allen Mitgliedern. `genome.set_weight_group(conn, group_id)` legt
Gruppen an. `genome.get_lamarck_connections()` dedupliziert repraesentative
Verbindungen fuer alle 4 Lamarck-Modi; `genome._sync_groups_from_reps()`
synchronisiert nach jedem Schritt. `genome.mutate()` resynct nach jedem
Mutations-Durchlauf. `copy()`, `crossover()` und `__setstate__` sind
rueckwaertskompatibel. `NeuroEvolution.set_shared_weights(enabled)`.
Tests in `tests/test_shared_weights.py` (24 Tests).

### ⚡ P2 Convolutional NEAT (CoDeepNEAT-inspiriert)

NEAT sucht verbindungsweise; fuer Bildverarbeitung ist die sinnvolle Sucheinheit ein Conv-Block (Filter, Stride, Channels), nicht eine einzelne Gewichtsverbindung.

**Aktueller Stand:** Experimenteller Spike. `ConvModule` kann einfache
Faltungs-/Patch-Operationen standalone ausfuehren. Es gibt noch keinen
`NodeType.CONV2D`, keine `add_conv_block`-Mutation, kein `forward_image()`,
keine Shared-Weight-Integration und keinen Python-Export fuer Conv-Knoten.

- Neuer Knotentyp `CONV2D` mit evolvierbaren Parametern: `kernel_size`, `stride`, `out_channels`, `activation`.
- Weight-Sharing automatisch: alle raeumlichen Positionen eines Filters teilen dasselbe Gewicht (baut auf Shared-Weights-Task auf).
- Mutations-Operator `add_conv_block`: fuegt einen vollstaendigen Conv-Block (CONV2D + optionaler MaxPool-Knoten) als Einheit hinzu.
- `genome.forward_image(pixels, height, width, channels)`: korrekter Faltungs-Forward-Pass.
- `genome_to_python()` unterstuetzt CONV2D-Knoten (erzeugt `for`-Schleifen statt einzelner Verbindungen).
- Benchmark: Convolutional NEAT vs. HyperNEAT vs. flaches NEAT auf MNIST (Accuracy nach fester Evaluierungs-Anzahl).

### □ P2 ES-HyperNEAT (Evolvable Substrate HyperNEAT)

Das aktuelle HyperNEAT-Substrat wird vom Nutzer manuell als Gitter-Koordinaten definiert; die CPPN weiss nicht wo sinnvolle Knoten-Positionen im geometrischen Raum liegen.

**Aktueller Stand:** HyperNEAT-Bausteine fuer feste Substrate sind vorhanden,
und experimentelle Substrate koennen beschrieben werden. ES-HyperNEAT im
eigentlichen Sinn fehlt: keine CPPN-varianzbasierte Quadtree-Suche, keine
evolvierbaren Substrat-Koordinaten, kein `evolve_substrate=True`-Flow und keine
Benchmarks.

- ES-HyperNEAT: CPPN-Output-Varianz bestimmt automatisch ob an einer Koordinate ein Knoten sinnvoll ist (hohe lokale Varianz → Knoten platzieren).
- `hyperneat_substrate(evolve=True)`: Substrat-Koordinaten werden nicht vorgegeben sondern per Quadtree aus der CPPN-Aktivierungslandschaft abgeleitet.
- Geometrische Biase bleiben erhalten: Eingabe-Knoten unten, Ausgabe-Knoten oben; CPPN kodiert raeumliche Beziehungen.
- `generate_genome_from_cppn(cppn, substrate, evolve_substrate=True)`: erweiterte Signatur.
- Benchmark: ES-HyperNEAT vs. festes Substrat auf einem 2D-Navigations-Task (Maze) und MNIST.

---

### □ P1 Automated Regression Benchmarking Suite (CI-faehige Benchmark-Pipeline)

Benchmarks werden derzeit manuell gestartet und die Ergebnisse nur informell
in `benchmarks/` abgelegt. Es gibt keine automatisierte Erkennung von
Regressionen zwischen Code-Aenderungen.

**Ziel:** Eine CI-faehige Benchmark-Suite, die automatisch Regressionen erkennt,
Reports generiert und historische Trends verfolgt.

**Design: `BenchmarkRunner` und `RegressionDetector`**

- `BenchmarkRunner`: liest eine `benchmarks.yaml`-Konfiguration mit
  Beispielen, Seeds, Timeouts, Target-Fitness und Vergleichsbasislinie.
- `RegressionDetector`: vergleicht aktuelle Runs mit gespeicherten
  Baseline-Werten (Median-Fitness, Konvergenz-Iterationen, Erfolgsrate).
  Nutzt Mann-Whitney-U-Test fuer statistische Signifikanz (p < 0.05).
- `run_benchmark_suite(path="benchmarks.yaml")`: fuehrt alle definierten
  Benchmarks aus und gibt einen `BenchmarkReport` zurueck.
- `RegressionSeverity`: `NONE`, `MINOR` (<5% Verschlechterung), `MAJOR`
  (5-20%), `CRITICAL` (>20% oder Erfolgsrate bricht ein).
- `baseline/`-Verzeichnis: speichert Referenzwerte pro Benchmark als JSON.
  `--update-baseline` Flag aktualisiert sie nach manueller Pruefung.

**CI-Integration:**

- Exit-Code: 0 bei `NONE`/`MINOR`, 1 bei `MAJOR`, 2 bei `CRITICAL`.
- `python -m yane.benchmarks --ci` fuer CI-Pipelines.
- JSON-Report auf stdout fuer einfaches Parsen in GitHub Actions/GitLab CI.
- Optionale GitHub-Issue-Erstellung bei `CRITICAL`-Regression (via CLI-Flag).

**Trend-Tracking:**

- `benchmarks/history/` speichert Zeitreihen pro Benchmark (Datum, Commit,
  Median-Fitness, Konvergenz-Iterationen).
- `benchmark_trend(example_name) → TrendReport` mit Visualisierung (ASCII-Plot
  oder PNG via Matplotlib, falls verfuegbar).

**Akzeptanzkriterien:**

- Alle existierenden Benchmarks in `benchmarks/` sind in der Suite
  konfigurierbar.
- Regression-Erkennung meldet korrekt eine absichtlich verschlechterte
  Mutation (z. B. `sigma_global` auf Extremwert).
- CI-Mode gibt korrekten Exit-Code.
- Baseline-Update ueberschreibt nur mit explizitem Flag.
- Trend-Tracking speichert und laedt historische Daten korrekt.
- Tests: RegressionDetector mit synthetischen Fitness-Verlaeufen;
  CI-Exit-Codes; Baseline-Laden/Speichern.

---

### □ P1 WandB / MLflow Integration (Experiment-Tracking-Backends)

TensorBoard-Support existiert bereits (`set_tensorboard_logdir()`), aber viele
Teams nutzen WandB (Weights & Biases) oder MLflow fuer Experiment-Tracking,
Hyperparameter-Vergleiche und Team-Kollaboration.

**Ziel:** Alternative Tracking-Backends, die parallel zu TensorBoard und
RunDatabase laufen koennen.

**Design: `TrackingBackend`-Protokoll**

```python
class TrackingBackend:
    def init(self, config: dict) -> None: ...
    def log_metrics(self, metrics: dict, step: int) -> None: ...
    def log_artifact(self, path: str, artifact_type: str) -> None: ...
    def log_config(self, config: dict) -> None: ...
    def finish(self) -> None: ...
```

- `NeuroEvolution.set_tracking_backend(backend)` registriert einen
  TrackingBackend; mehrere Backends sind gleichzeitig moeglich (wandb +
  tensorboard + mlflow).
- `WandbBackend`: nutzt `wandb.init()`, `wandb.log()`,
  `wandb.log_artifact()`. API-Key aus Umgebungsvariable `WANDB_API_KEY`.
- `MlflowBackend`: nutzt `mlflow.start_run()`, `mlflow.log_metrics()`,
  `mlflow.log_artifact()`. Tracking-URI aus Umgebungsvariable
  `MLFLOW_TRACKING_URI`.
- Generations-Metriken (Best-Fitness, Mean-Fitness, Species-Count, IQR,
  Novelty, etc.) werden automatisch geloggt.
- Checkpoint-Dateien werden als Artifacts hochgeladen (optional).
- Run-Konfiguration wird als `config`-Dict geloggt.
- `finish()` wird automatisch am Ende von `train()` aufgerufen.

**Abgrenzung zu RunDatabase:**

- RunDatabase ist der lokale, YANE-eigene Persistenz-Layer (SQLite).
- Tracking-Backends sind externe Cloud-/Team-Dienste fuer Visualisierung und
  Kollaboration.
- Beide koennen parallel genutzt werden.

**Akzeptanzkriterien:**

- `WandbBackend` loggt Metriken ohne Fehler (Test mit `wandb.init(mode="disabled")`).
- `MlflowBackend` loggt Metriken in lokales Tracking-Verzeichnis.
- Mehrere Backends gleichzeitig aktiv: alle erhalten dieselben Metriken.
- Fehlende optionale Abhaengigkeiten (`wandb`, `mlflow`) werden mit klarem
  ImportError gemeldet, nicht als harte Abhaengigkeit.
- Tests: Mock-Backend verifiziert korrekte Metrik-Uebergabe; ImportError bei
  fehlenden Paketen; Multi-Backend-Dispatch.

---

### □ P2 Genome-to-ONNX-Export (Produktions-Deployment)

Trainierte Genome sind nur innerhalb von YANE/Python nutzbar. ONNX (Open Neural
Network Exchange) ist der Standard fuer plattformunabhaengiges Deployment und
wird von Inferenz-Engines wie ONNX Runtime, TensorRT, OpenVINO und WebML
unterstuetzt.

**Ziel:** Genome als ONNX-Modell exportieren, das ohne YANE-Abhaengigkeiten in
Produktion laeuft.

**Design: `genome_to_onnx()` in `evolution/onnx_export.py`**

```python
genome.export_onnx(path="model.onnx", input_name="input",
                   output_name="output", opset_version=17)
```

- ONNX-Graph-Repraesentation: Input-Nodes → ONNX-Inputs, Output-Nodes →
  ONNX-Outputs, Hidden-Nodes → ONNX-Zwischenknoten.
- Jede Connection wird als ONNX-Operation abgebildet: `Mul(weight)` →
  `Add(bias)` → `Activation`.
- Topologische Sortierung des Genoms bestimmt die ONNX-Graphen-Reihenfolge.
- Aktivierungsfunktionen werden auf ONNX-Aequivalente gemappt (sigmoid →
  `Sigmoid`, relu → `Relu`, tanh → `Tanh`, etc.).
- Zyklische Netze: ONNX unterstuetzt keine Zyklen; der Export flacht den BFS-
  Forward-Pass in eine feste Anzahl entrollter Schritte ab (konfigurierbar,
  Default: 5). Alternativ: Abbruch mit klarer Fehlermeldung wenn
  `allow_cyclic=False`.
- `stateful`/Memory-Nodes: entrollte Zeitschritte mit State-Feedback.
- Custom-Aktivierungen: werden als `identity` + externe Post-Processing-
  Dokumentation exportiert (ONNX kann keine beliebigen Python-Funktionen).

**API:**

```python
yane.export_onnx("model.onnx", opset_version=17, allow_cyclic=True,
                 unroll_steps=5)
```

- `genome_to_onnx()` als standalone-Funktion fuer Checkpoint-Nachbearbeitung.
- `genome.export_onnx()` als Genome-Methode.
- ONNX-Modell-Validierung via `onnx.checker.check_model()` vor dem Schreiben.

**Benchmark:**

- Inferenzzeit ONNX Runtime vs. YANE-Forward auf XOR, CartPole und BipedalWalker
  (Erwartung: ONNX 2-10x schneller).
- Byte-Groesse des exportierten ONNX-Modells vs. Pickle-Checkpoint.

**Abgrenzung:**

- `genome_to_python()`: Python-Code-Export, kein standardisiertes Format.
- `torch_bridge.py`: PyTorch-spezifisch, kein Deployment-Format.
- ONNX-Export: plattformunabhaengig, standardisiert, optimierbar.

**Akzeptanzkriterien:**

- XOR-Genom nach ONNX exportiert: ONNX-Inferenz liefert identische Outputs
  (Toleranz 1e-5) wie `genome.forward()`.
- ONNX-Modell besteht `onnx.checker.check_model()`.
- Zyklisches Genom mit `unroll_steps=3` exportiert: Outputs innerhalb 1e-3
  Toleranz.
- `genome_to_python()`-Roundtrip: exportiertes ONNX-Modell reproduziert
  korrektes Verhalten.
- Tests: ONNX-Export fuer XOR, lineares Netz, rekurrentes Netz, custom
  Aktivierungen; ONNX-Validierung; Roundtrip-Konsistenz.

---

### □ P2 Population Distillation (Ensemble → kompaktes Einzelgenom)

Ein Ensemble aus Top-K-Genomen ist robuster als ein einzelnes Genom, aber
teurer in der Inferenz. Distillation uebertraegt das Wissen des Ensembles in
ein einzelnes, kompaktes Genom — aequivalent zu Knowledge Distillation im
Deep Learning.

**Ziel:** Ein kompaktes Student-Genom, das die Ensemble-Ausgaben moeglichst
genau reproduziert, mit deutlich weniger Nodes/Connections.

**Design: `distill_ensemble()` in `evolution/distillation.py`**

```python
student = yane.distill_ensemble(k=5, target_nodes=10,
                                target_connections=30,
                                distillation_steps=500,
                                temperature=2.0)
```

**Ablauf:**

1. Ensemble aus Top-K-Genomen erstellen (`get_ensemble(k)`).
2. Student-Genom mit `target_nodes` und `target_connections` initialisieren
   (kleiner als das groesste Ensemble-Mitglied).
3. Distillation-Loop: feste Liste von Probe-Inputs (generiert aus der
   Evaluator-Domain oder zufaellig gesampelt) durch Ensemble UND Student
   forwarden.
4. Loss: MSE zwischen Ensemble-Output und Student-Output, plus optionale
   Topologie-Komplexitaetsstrafe.
5. Student-Gewichte via Lamarck-Refinement (Hill-Climb oder CMA-ES)
   optimieren — kein Backprop, bleibt im YANE-Paradigma.
6. Optional: Knowledge-Distillation-Temperatur: Ensemble-Outputs werden mit
   Softmax/Temperatur geglaettet bevor der Student darauf trainiert wird.
7. Student-Genom wird zurueckgegeben; kann als Seed fuer weitere Evolution
   oder direkt als Deployment-Modell verwendet werden.

**API:**

```python
yane.distill_ensemble(k=5, target_nodes=None, target_connections=None,
                      distillation_steps=500, temperature=2.0,
                      probe_inputs=None)
```

- `target_nodes=None`: automatisch 50% der durchschnittlichen Ensemble-
  Mitglieds-Groesse.
- `probe_inputs=None`: generiert 100 zufaellige Inputs im Bereich `[-1, 1]`
  (uebersteuerbar fuer domain-spezifische Inputs).

**Diagnostics:**

- `distillation_loss_history`: MSE pro Distillation-Schritt.
- `compression_ratio`: Student-Nodes / durchschnittliche Ensemble-Mitglied-Nodes.
- `output_correlation`: Pearson-Korrelation zwischen Ensemble- und Student-Outputs.

**Benchmark:**

- Distillation auf CartPole: Student mit 50% der Nodes erreicht >95% der
  Ensemble-Performance.
- Distillation auf XOR: Student mit 3 Nodes erreicht identische Accuracy.

**Akzeptanzkriterien:**

- Distillation auf XOR: Student (3 Nodes) loest XOR exakt.
- Student ist kleiner als das durchschnittliche Ensemble-Mitglied
  (`compression_ratio < 1.0`).
- Distillation-Loss sinkt monoton ueber die Schritte.
- `distill_ensemble()` funktioniert ohne `train()`-Kontext (Standalone).
- Tests: Distillation auf XOR; Kompressionsrate; Output-Korrelation;
  API-Fehlerbehandlung (k > Populationsgroesse).

---

### □ P2 Gradient-NEAT-Hybrid-Modus (Backprop + Evolution interleaved)

NEAT evolviert Topologie und Gewichte gemeinsam — langsam aber gruendlich.
Backprop (via `torch_bridge`) trainiert Gewichte schnell, aber auf fester
Topologie. Ein Hybrid-Modus kombiniert beide: Evolution schlaegt neue
Topologien vor, Backprop trainiert die Gewichte effizient.

**Ziel:** `train()` mit hybridem Modus, der zwischen Evolutions- und
Backprop-Phasen wechselt.

**Design: `set_hybrid_mode()` in `NeuroEvolution`**

```python
yane.set_hybrid_mode(enabled=True, bp_interval=10, bp_epochs=50,
                     bp_lr=0.01, bp_batch_size=32)
```

**Ablauf:**

1. Normaler NEAT-Trainingsloop laeuft.
2. Alle `bp_interval` Generationen: Top-K-Genome werden via
   `genome_to_torch_module()` in PyTorch-Module konvertiert.
3. Backprop-Phase: jedes Top-Genom wird fuer `bp_epochs` Epochen mit Adam
   auf den Trainingsdaten trainiert (via `torch_bridge`).
4. Trainierte Gewichte werden zurueck ins Genom geschrieben
   (`torch_module_to_genome()`).
5. Verbesserte Genome werden in die Population zurueckgegeben; Evolution
   kann neue Topologien darauf aufbauen.
6. Optional: nur Gewichte werden per Backprop trainiert, Topologie bleibt
   waehrend BP-Phase eingefroren.

**Datenbereitstellung:**

- `set_training_data(inputs, targets)`: registriert Trainingsdaten fuer die
  BP-Phase. Nur fuer Dataset-Aufgaben (XOR, Regression, Klassifikation).
- Fuer Gym-Umgebungen: `set_replay_buffer(capacity=10000)`: speichert
  (State, Action, Reward)-Transitionen waehrend normaler Evaluation; Backprop
  trainiert auf gesampelten Batches aus dem Replay-Buffer (Behavior Cloning
  auf die eigenen besten Aktionen).
- Ohne Daten: Hybrid-Modus deaktiviert sich mit Warning.

**Diagnostics:**

- `bp_loss_history`: Trainings-Loss pro Backprop-Epoche.
- `bp_improvement`: Fitness-Delta vor/nach Backprop-Phase.
- `hybrid_phase`: aktuelle Phase (`"evolution"` oder `"backprop"`).

**Limitationen:**

- Nur fuer `torch`-verfuegbare Umgebungen (optionaler Import).
- Zyklische Netze: `genome_to_torch_module()` flacht Zyklen in entrollte
  Schritte ab (analog ONNX-Export).
- Memory-Nodes: GRUCell-Approximation aus `torch_bridge` wird genutzt.

**Benchmark:**

- XOR: Hybrid vs. reiner NEAT vs. reiner Backprop (Konvergenz-Iterationen).
- CartPole: Hybrid mit Replay-Buffer vs. reiner NEAT.
- Erwartung: Hybrid konvergiert schneller als reiner NEAT auf
  gradientenfreundlichen Problemen.

**Akzeptanzkriterien:**

- XOR: Hybrid-Modus konvergiert in <50% der Iterationen von reinem NEAT.
- Gewichte nach BP-Phase sind im Genom persistent (ueberleben Crossover).
- Ohne PyTorch: klarer ImportError, kein Crash.
- `set_training_data()` akzeptiert numpy-Arrays und Python-Listen.
- Tests: Hybrid auf XOR; Gewichts-Persistenz; Fehlerbehandlung ohne Daten;
  Replay-Buffer-Sampling.

---

### □ P2 WebAssembly-Export (Browser-Deployment)

Neben ONNX fuer Server-/Edge-Deployment ist WebAssembly (WASM) der Standard
fuer Browser-basierte Inferenz. Ein WASM-Export ermoeglicht trainierte YANE-
Genome direkt im Browser auszufuehren — ohne Python, ohne Server.

**Ziel:** `genome_to_wasm()` generiert eine standalone HTML/JS/WASM-Datei, die
das Genom im Browser ausfuehrt.

**Design: `genome_to_wasm()` in `evolution/wasm_export.py`**

```python
genome.export_wasm("model.html", inline_wasm=True)
```

**Export-Strategie (absteigend nach Praktikabilitaet):**

1. **Python→C→WASM (empfohlen):** `genome_to_python()` erzeugt Python-Code;
  dieser wird mit einem simplen Python→C-Transpiler (nur arithmetische
  Operationen, keine Python-Stdlib) in C uebersetzt; Emscripten kompiliert C
  zu WASM. Limitierung auf azyklische Netze (oder entrollte Zyklen).
2. **ONNX→WASM (Fallback):** ONNX-Modell via ONNX Runtime Web ausfuehren.
  Erfordert `onnx.js` als Dependency im Browser.
3. **Pure-JS-Transpilation (einfachste):** `genome_to_python()`→JavaScript
  uebersetzen. Keine WASM-Abhaengigkeit, aber langsamer.

**Output-Formate:**

- `model.html`: standalone HTML mit eingebettetem WASM und JS-Glue-Code.
  Oeffnet im Browser und zeigt eine simple Test-UI (Input-Felder → Output).
- `model.wasm` + `model.js`: getrennte Dateien fuer Einbettung in eigene
  Webprojekte.
- `model.js` (Pure-JS-Fallback): wenn Emscripten nicht verfuegbar.

**Unterstuetzte Features:**

- Azyklische Netze: voll unterstuetzt.
- Zyklische Netze: entrollt mit konfigurierbaren Schritten.
- Memory/Stateful: State-Variable im JS-Code, Reset-Button in HTML.
- Aktivierungsfunktionen: alle eingebauten Funktionen werden nach JS/WASM
  uebersetzt. Custom-Aktivierungen: Fehler mit Hinweis.

**API:**

```python
yane.export_wasm("model.html", mode="wasm", allow_cyclic=True,
                 unroll_steps=5, inline=True)
```

- `mode`: `"wasm"` (Emscripten), `"onnx-web"` (ONNX Runtime Web),
  `"pure-js"` (JavaScript).
- `inline=True`: alles in eine HTML-Datei packen.

**Benchmark:**

- Inferenzzeit WASM vs. Pure-JS vs. YANE-Python auf XOR (1000 Forward-Paesse).
- Dateigroesse der Export-Formate.

**Akzeptanzkriterien:**

- XOR-Genom als `.html` exportiert: im Browser geoeffnet, Forward-Pass
  liefert identische Outputs (Toleranz 1e-5).
- Pure-JS-Modus funktioniert ohne Emscripten-Installation.
- Zyklisches Netz mit `unroll_steps=3` exportiert: Outputs innerhalb 1e-3.
- Memory-Node: State persistiert ueber mehrere `forward()`-Aufrufe im
  Browser.
- Tests: WASM/Pure-JS-Export fuer XOR; Output-Vergleich Python↔WASM↔JS;
  Memory-Persistenz im simulierten Browser-Kontext (node.js oder Mock).

---

### □ P2 Evolvable Attention Heads (Transformer-inspirierte Architektursuche)

Attention-Mechanismen sind der Kern moderner Transformer-Architekturen. Ein
evolvierbarer Attention-Node-Typ erlaubt YANE, Attention-Strukturen zu
entdecken — ohne dass der Nutzer sie manuell designen muss.

**Ziel:** `NodeType.ATTENTION` als neuer Knotentyp, der Key/Query/Value aus
seinen eingehenden Verbindungen berechnet und Attention-gewichtete Outputs
produziert.

**Design: `NodeType.ATTENTION`**

- Aufbau eines Attention-Heads: Q, K, V werden aus den eingehenden
  Connections berechnet (jeweils durch eine Gewichtsmatrix).
- `head_dim`: Dimensionalitaet des Attention-Raums (evolvierbar).
- `num_heads`: Anzahl paralleler Attention-Koepfe (evolvierbar, Default: 1).
- Forward: `softmax(Q @ K^T / sqrt(head_dim)) @ V` → Output.
- Multi-Head: Outputs aller Koepfe werden konkateniert und linear
  projiziert.
- Positional Encoding: optional, ueber eine spezielle `POS_ENCODING`-Input-
  Node oder als evolvierbarer Bias auf den Inputs.

**Integration in `genome.forward()`:**

- `ATTENTION`-Node erwartet eine feste Anzahl eingehender Connections, die
  als Sequenz interpretiert werden.
- Inputs werden in `head_dim` grosse Bloecke gruppiert (eine Connection pro
  Q/K/V-Dimension).
- Wenn `num_heads > 1`: Output-Dimension = `num_heads * head_dim`.
- Attention-Nodes koennen mit normalen Nodes verbunden werden; das Netz
  kann Attention und normale Verarbeitung mischen.

**Mutations-Operatoren:**

- `add_attention_head()`: fuegt einen `ATTENTION`-Node mit `head_dim=4` und
  `num_heads=1` hinzu.
- `mutate_head_dim(delta)`: aendert `head_dim` um ±1.
- `mutate_num_heads(delta)`: aendert `num_heads` um ±1 (begrenzt durch
  verfuegbare Input-Connections).
- Standard-Mutationen (`add_connection`, `remove_node`, etc.) funktionieren
  weiterhin auf Attention-Nodes.

**Layer-3-Integration:**

- `NeuroEvolution.set_attention(enabled=True)` aktiviert Attention-Nodes.
- Bei `enabled=False`: `ATTENTION`-Nodes werden nicht erzeugt, existierende
  werden ignoriert (keine Laufzeitkosten).
- `genome_to_python()` emittiert Attention-Forward-Code.
- `genome_to_torch_module()` mappt auf `nn.MultiheadAttention`.

**Abgrenzung:**

- Memory/Stateful: persistente Werte ueber Zeit.
- Attention: gewichtete Aggregation ueber raeumliche/sequenzielle Positionen.
- Beide koennen kombiniert werden (Attention ueber Memory-Zellen).

**Benchmark:**

- Sequenz-Copy-Task (LSTM-klassisch): Attention-NEAT vs. normaler NEAT.
- Symbolic-Regression mit Attention ueber Input-Features.
- Mini-Image-Classification (8x8 MNIST): Attention ueber Pixel-Patches.

**Akzeptanzkriterien:**

- Attention-Node produziert korrekte Softmax-gewichtete Outputs
  (Unit-Test mit bekannten Q/K/V-Werten).
- `add_attention_head()` fuegt funktionalen Attention-Node hinzu.
- `genome.forward()` mit Attention-Node liefert deterministische Outputs.
- Crossover: Genome mit unterschiedlichen Attention-Konfigurationen kreuzbar.
- Checkpoint-Roundtrip erhaelt `head_dim` und `num_heads`.
- Tests: Attention-Mathe; Multi-Head-Konkatenation; Crossover-Kompatibilitaet;
  Checkpoint-Persistenz; genome_to_python mit Attention.

---

### □ P2 Liquid Time-Constant (LTC) Nodes (ODE-basierte Neuronen)

Liquid Time-Constant Networks (Hasani et al. 2021) ersetzen klassische
Aktivierungsfunktionen durch eine zeitkontinuierliche ODE: jedes Neuron hat
eine learnbare Zeitkonstante τ, die bestimmt wie schnell es auf Inputs
reagiert. Das ist eine natuerliche Erweiterung des existierenden
Memory/Stateful-Systems.

**Ziel:** `NodeType.LTC` als neuer Knotentyp mit ODE-basierter Dynamik,
komplementaer zu `persist_value`-basiertem Memory.

**Design: `NodeType.LTC`**

- ODE: `dx/dt = -(1/τ) * x + f(input, bias)` wobei `f` eine nichtlineare
  Aktivierungsfunktion ist und `τ` die Zeitkonstante.
- Diskrete Approximation (Forward Euler): `x_{t+1} = x_t + dt * (-x_t/τ +
  activation(sum(inputs) + bias))`.
- `dt`: Zeitschritt, Default `0.1` (evolvierbar).
- `tau`: Zeitkonstante, Default `1.0` (evolvierbar). Kleine τ → schnelle
  Reaktion. Grosse τ → langsame, integrierende Dynamik.
- `activation`: eingebaute Aktivierungsfunktion (evolvierbar, wie normale
  Nodes).
- `persist_value` ist bei LTC-Nodes implizit immer `True` (der State ist
  die ODE-Variable `x`).
- `genome.reset()` setzt `x = 0.0`.

**Forward-Pass-Integration:**

- LTC-Nodes feuern im topologischen Order (azyklisch) oder BFS-Wellen
  (zyklisch), genau wie normale Nodes.
- Der ODE-Schritt wird einmal pro `genome.forward()` oder `genome.tick()`
  ausgefuehrt — ein Forward-Pass = ein ODE-Integrationsschritt.
- Fuer mehrere Integrationsschritte pro Forward: `genome.forward(data,
  ode_steps=5)` fuehrt 5 Euler-Schritte aus bevor Outputs gelesen werden.

**Mutations-Operatoren:**

- `add_ltc_node()`: fuegt einen `LTC`-Node mit Default-τ=1.0 und dt=0.1
  hinzu.
- `mutate_tau(node, delta)`: aendert τ um einen gaußschen Schritt.
- `mutate_dt(node, delta)`: aendert dt um einen gaußschen Schritt.
- Normal-Node → LTC-Node: Mutations-Operator, der `persist_value=True` in
  LTC-Dynamik umwandelt.

**Layer-3-Integration:**

- `NeuroEvolution.set_ltc(enabled=True)` aktiviert LTC-Nodes.
- Bei `enabled=False`: keine LTC-Nodes, keine ODE-Kosten.
- `genome_to_python()` emittiert Euler-Integrationsschleife fuer LTC-Nodes.
- `genome_to_torch_module()` mappt auf `nn.RNN`-aehnliche Zellen oder
  benutzerdefinierte ODE-Forward-Funktion.

**Abgrenzung:**

- `stateful`/`persist_value`: harter Wert-Erhalt ohne Dynamik.
- LTC: kontinuierliche, zeitabhaengige Zustandsubergang mit τ-Steuerung.
- CT-RNN (Continuous-Time RNN): aequivalent, LTC ist die moderne Variante
  mit stabilitaetsgarantierter ODE.

**Benchmark:**

- Sine-Wave-Prediction: LTC vs. normaler NEAT mit Memory-Nodes (MSE ueber
  Zeit).
- BipedalWalker mit LTC-Nodes vs. normalen persistierenden Nodes.
- CartPole-Swingup (partielle Observability): LTC vs. GRU/LSTM-Vergleich.

**Akzeptanzkriterien:**

- LTC-Node mit τ→∞ naehert sich konstantem State (integriert ohne Leak).
- LTC-Node mit τ→0 naehert sich instantanem Node (kein Memory).
- `genome.reset()` setzt LTC-State korrekt zurueck.
- `ode_steps > 1` produziert unterschiedliche Outputs als `ode_steps=1`
  (Integration sichtbar).
- Crossover: Genome mit LTC- und normalen Nodes kreuzbar.
- Checkpoint-Roundtrip erhaelt τ und dt.
- Tests: ODE-Mathe (Euler-Schritt); Reset-Verhalten; τ-Extremwerte;
  Crossover-Kompatibilitaet; genome_to_python mit LTC.

---

### □ P2 Temporal Speciation (Verhaltensbasierte Spezies-Bildung)

Die aktuelle Speziation gruppiert Genome nach statischer Topologie-
Aehnlichkeit (Excess/Disjoint-Gene, Gewichtsdifferenz). In rekurrenten oder
Stateful-Netzen sagt die Topologie aber wenig ueber das tatsaechliche
Verhalten ueber die Zeit aus. Temporal Speciation gruppiert Genome nach
aehnlichen Trajektorien — relevant fuer rekurrente Netze, LTC-Nodes und
partielle Observability.

**Ziel:** `TemporalDistance` als alternative Kompatibilitaetsmetrik, die
verhaltensbasiert gruppiert.

**Design: `TemporalDistance` in `evolution/compatibility.py`**

```python
yane.set_compatibility_distance(TemporalDistance(
    n_rollouts=5, rollout_len=20, time_weight=0.5
))
```

- Behavior Descriptor pro Genom: Trajektorie ueber `rollout_len` Schritte
  mit `n_rollouts` verschiedenen Startzustaenden (fixed random seeds).
- Distanz zwischen zwei Genomen: Dynamic Time Warping (DTW) ueber die
  Output-Trajektorien, gewichtet mit `time_weight`.
- `TemporalDistance` implementiert das `DistanceMetric`-Protokoll (bestehend
  aus P1 Modular Compatibility Distance).
- Kombinierbar mit `TopologyDistance` via `ChainMetric`:
  `ChainMetric([TopologyDistance(), TemporalDistance()], weights=[0.5, 0.5])`.
- Caching: Trajektorien werden nur bei Strukturaenderung neu berechnet
  (nicht bei jeder Spezies-Zuordnung).

**Benchmark:**

- Rekurrenter Sequenz-Task: Temporal vs. Topologie-Speziation
  (Konvergenzgeschwindigkeit).
- Partielle-Observability-Umgebung: Temporal-Speziation gruppiert Genome
  mit aehnlichem Verhalten auch bei unterschiedlicher Topologie.

**Akzeptanzkriterien:**

- `TemporalDistance` implementiert korrekt das `DistanceMetric`-Protokoll.
- DTW-Distanz zwischen identischen Trajektorien ist 0.
- `ChainMetric` mit `TemporalDistance` + `TopologyDistance` funktioniert.
- Caching: Trajektorie wird nicht neu berechnet wenn Topologie unveraendert.
- Tests: DTW-Mathe; DistanceMetric-Protokoll-Kompatibilitaet; Chain-
  Integration; Caching-Invalidierung; Template-Genome haben definierte
  TemporalDistance.

---

### □ P2 Self-Play / Adversarial Populations (Kompetitive Co-Evolution)

Anders als Co-Evolution (POET-aehnlich, Agent+Environment gemeinsam),
modelliert Self-Play zwei oder mehr Sub-Populationen die *gegeneinander*
antreten. Der Fitnessgewinn des einen ist der Fitnessverlust des anderen —
ein Nullsummenspiel. Anwendung: Spiele (Schach, Go), adversariales Training,
competitive Multi-Agent-Szenarien.

**Ziel:** `set_adversarial_populations()` teilt die Population in zwei
gegnerische Sub-Populationen und laesst sie gegeneinander antreten.

**Design: `AdversarialPopulation` in `evolution/adversarial.py`**

```python
yane.set_adversarial_populations(n_populations=2, pairing="round_robin",
                                 shared_species=False)
```

- `n_populations`: Anzahl gegnerischer Sub-Populationen (Default: 2).
- Jede Sub-Population hat eigene Spezies, eigenen Genpool und eigenen
  Mutationsdruck.
- `pairing="round_robin"`: jedes Genom aus Pop A spielt gegen jedes Genom
  aus Pop B (oder Stichprobe bei grossen Populationen).
- `pairing="top_vs_top"`: nur die besten K Genome jeder Population spielen
  gegeneinander.
- `pairing="random"`: zufaellige Paarungen pro Generation.
- `shared_species=False`: jede Sub-Population hat eigene Spezies.
  `shared_species=True`: globale Spezies ueber alle Sub-Populationen hinweg.
- Fitness: `fitness_a = score_a - score_b`, `fitness_b = score_b - score_a`
  (Nullsumme). Optional: `fitness_a = score_a` (absolute Leistung).

**Evaluator-Schnittstelle:**

```python
def adversarial_evaluate(genome_a, genome_b):
    score_a = simulate_match(genome_a, genome_b)
    score_b = -score_a  # Nullsumme
    return score_a, score_b
```

- `yane.set_adversarial_evaluator(adversarial_evaluate)` registriert die
  Evaluierungsfunktion.
- YANE managed das Pairing und die Fitness-Zuweisung automatisch.

**Diagnostics:**

- `elo_ratings`: Elo-Rating pro Genom (oder pro Sub-Population), aktualisiert
  nach jeder Generation.
- `population_balance`: Fitness-Verhaeltnis zwischen Sub-Populationen.
- `arms_race_indicator`: Anstieg der durchschnittlichen Fitness in BEIDEN
  Populationen (Indikator fuer gesundes adversariales Training).

**Abgrenzung zu Co-Evolution (POET):**

- POET: Agent und Environment co-evolvieren kooperativ; Environment wird
  schwerer wenn Agent besser wird.
- Self-Play: Zwei+ Agenten konkurrieren direkt; Fitness ist relativ.
- Beide koennen parallel existieren (z. B. Self-Play innerhalb eines
  POET-Environments).

**Benchmark:**

- Tic-Tac-Toe: Self-Play-NEAT vs. zufaelliger Gegner (Spielstaerke nach N
  Generationen).
- Einfaches Competitive-Spiel (z. B. Pursuit-Evasion): Self-Play vs. Single-
  Population.
- Erwartung: Self-Play produziert robustere Strategien als Training gegen
  statischen Gegner.

**Akzeptanzkriterien:**

- Zwei Sub-Populationen evolvieren mit getrennten Spezies.
- Fitness ist korrekt Nullsumme (Summe ueber alle Genome ≈ 0).
- Elo-Ratings steigen in beiden Populationen bei gesundem Arms Race.
- `shared_species=True`: Genome aus beiden Populationen in denselben Spezies.
- Crossover nur innerhalb der eigenen Sub-Population.
- Tests: Nullsummen-Fitness; getrennte Spezies; Elo-Update; Pairing-
  Mechanismen; Crossover-Isolation.

---

### □ P2 Hierarchical NEAT (H-NEAT) — Mehrstufige Policy-Architektur

Komplexe Aufgaben zerfallen natuerlich in Hierarchien: Strategie→Taktik→Aktion.
Ein flaches NEAT-Netz muss alle Ebenen gleichzeitig lernen. H-NEAT evolviert
explizit hierarchische Strukturen: High-Level-Genome waehlen Sub-Policies aus,
Low-Level-Genome fuehren sie aus.

**Ziel:** Ein zweistufiges System, bei dem ein Manager-Genom aus einem Pool von
Sub-Policies die passende auswaehlt und ein Worker-Genom die feingranularen
Aktionen steuert.

**Design: `HierarchicalNEAT` in `evolution/hierarchical.py`**

```
High-Level Manager (waehlt Sub-Policy pro Zeitschritt)
    ├── Sub-Policy Pool (N Low-Level Genome)
    │   ├── Worker-1: "navigate"
    │   ├── Worker-2: "attack"
    │   ├── Worker-3: "defend"
    │   └── ...
    └── Output: aggregiert oder direkt Worker-Aktion
```

- `ManagerGenome`: erhaelt Umgebungszustand, gibt Sub-Policy-Index aus
  (one-hot oder softmax via Output-Nodes).
- `WorkerGenome`: erhaelt Umgebungszustand, gibt konkrete Aktion aus.
- Sub-Policy-Pool: Menge von `WorkerGenome`-Instanzen, die durch Crossover
  und Mutation evolvieren wie eine normale Population.
- Forward-Pass: `manager.forward(state) → sub_idx` →
  `workers[sub_idx].forward(state) → action`.
- Sub-Policies teilen keine Connections, sind unabhaengige Genome.

**Mutations-Operatoren:**

- `add_sub_policy()`: neues Worker-Genom aus Template erstellen.
- `remove_sub_policy(idx)`: Worker loeschen.
- `split_sub_policy(idx)`: Worker durch zwei spezialisierte Worker ersetzen
  (kopieren + strukturmutieren).
- `merge_sub_policies(i, j)`: zwei Worker per Crossover zu einem kombinieren.
- Manager und Worker durchlaufen normale NEAT-Mutation unabhaengig.

**Koordination:**

- `selection_mode`: `"hard"` (eine Sub-Policy aktiv) oder `"soft"` (gewichtete
  Mischung aller Sub-Policies).
- `switching_cost`: Penalty fuer haeufiges Wechseln (evo-freundliches
  Verhalten).
- Manager und Worker werden gemeinsam evaluiert; Fitness wird an beide
  zurueckgegeben (Shared Credit Assignment).

**Diagnostics:**

- `sub_policy_usage`: Haeufigkeit jeder Sub-Policy pro Episode.
- `switching_rate`: Sub-Policy-Wechsel pro Zeitschritt.
- `sub_policy_fitness`: Fitness-Attribution pro Sub-Policy.

**Benchmark:**

- Hierarchical-Task (z. B. „zuerst navigieren, dann Objekt manipulieren"):
  H-NEAT vs. flaches NEAT (Konvergenzgeschwindigkeit, End-Fitness).
- Meta-World (ML1): H-NEAT vs. normales NEAT.

**Akzeptanzkriterien:**

- Manager waehlt unterschiedliche Sub-Policies fuer unterschiedliche
  Umgebungszustaende.
- Sub-Policy-Pool waechst/schrumpft durch Mutation.
- Crossover funktioniert auf Manager + allen Workern.
- Checkpoint speichert/ladt komplette Hierarchie.
- Tests: Manager-Output-Range; Sub-Policy-Selektion; Pool-Mutation;
  Checkpoint-Roundtrip.

---

### □ P2 Gene Regulatory Network (GRN) Encoding

Direkte Topologie-Kodierung (jedes Gen = eine Connection) skaliert schlecht bei
grossen Netzen. Eine indirekte Kodierung via GRN — inspiriert durch biologische
Genregulation — komprimiert das Genom: wenige regulatorische Gene steuern die
Entwicklung vieler Connections und Nodes.

**Ziel:** `GRNCodec` als alternative Genom-Kodierung, die ein kompaktes
GRN-Genom (wenige Gene) in ein volles NEAT-Netzwerk (viele Connections)
entfaltet.

**Design: `GRNCodec` in `evolution/grn_encoding.py`**

- GRN-Gen: `(input_gene, output_gene, weight, activation, regulatory_sites[])`.
- Jedes GRN-Gen hat eine `concentration` (Aktivierungslevel) und
  reguliert andere Gene via Aktivierungs-/Hemmungs-Seiten.
- Entwicklung (Genotyp→Phaenotyp): GRN wird fuer N Entwicklungsschritte
  simuliert; am Ende werden Gene ueber einem Schwellwert als Connections
  im resultierenden Genom realisiert.
- Entwicklungs-Schritte: `grn.tick()` propagiert Konzentrationen zwischen
  Genen gemaess ihrer regulatorischen Verbindungen.
- Ergebnis: ein `Genome` mit Nodes und Connections, das normal evaluiert wird.

**Vorteile der GRN-Kodierung:**

- Kompaktheit: 20 GRN-Gene koennen >1000 Connections enkodieren.
- Modularitaet: regulatorische Module enkodieren funktionale Einheiten.
- Wiederholungsmuster: ein GRN-Gen kann mehrere aequivalente Connections
  erzeugen (Symmetrie).
- Robustheit: Mutation eines GRN-Gens aendert mehrere Connections konsistent.

**Mutations-Operatoren (auf GRN-Ebene):**

- `add_regulatory_gene()`: neues Gen einfuegen.
- `remove_regulatory_gene()`: Gen entfernen.
- `add_regulatory_site(gene, target)`: neue regulatorische Verbindung.
- `mutate_gene_concentration()`: Konzentration gaußsch mutieren.
- `duplicate_gene()`: Genduplikation mit anschliessender Divergenz.

**Integration:**

- `NeuroEvolution.set_genome_encoding("grn", development_steps=5)`.
- `NeuroEvolution.set_genome_encoding("direct")`: Standard-NEAT-Kodierung.
- GRN-Codec implementiert `GenomeCodec`-Protokoll (P2-Task).
- `genome.get_grn()` gibt das GRN-Objekt zurueck (None bei direkter Kodierung).

**Benchmark:**

- Retina-Problem (8x8 visueller Input): GRN vs. direkte Kodierung
  (Netzwerkgroesse, Konvergenz-Iterationen).
- Modularer Task (z. B. 4 unabhaengige XOR-Subprobleme): Modularitaet der
  GRN-Loesung.

**Akzeptanzkriterien:**

- GRN mit 20 Genen enkodiert ein Genom mit >100 Connections nach Entwicklung.
- `development_steps` beeinflusst die resultierende Netzwerkgroesse.
- GRN-Mutationen propagieren korrekt in den Phaenotyp.
- Crossover zweier GRN-Genome funktioniert (Alignment der Gene).
- Checkpoint-Roundtrip erhaelt GRN-Struktur.
- Tests: GRN-Entwicklung; Gen-Konzentrations-Dynamik; Phaenotyp-Groessen-
  Korrelation; Crossover-Kompatibilitaet; GenomeCodec-Protokoll.

---

### □ P2 Developmental NEAT — Ontogenese waehrend Evaluation

Das Genom veraendert sich NUR durch Evolution zwischen Generationen — nicht
waehrend einer Episode. Developmental NEAT erlaubt dem Genom, sich *innerhalb*
einer Evaluation zu entwickeln: Connections koennen aktiviert/deaktiviert
werden, gesteuert durch evolvierte Entwicklungsregeln.

**Ziel:** `genome.developmental_forward()` aendert die Topologie waehrend des
Forward-Passes basierend auf evolvierbaren Entwicklungsgenen.

**Design: `DevelopmentalGenome` in `evolution/developmental.py`**

- Entwicklungsgene: jedes Genom traegt zusaetzlich eine Menge von
  `DevelopmentalRule`-Objekten.
- `DevelopmentalRule`: `trigger_condition` (z. B. „Node-3-Aktivierung > 0.5"),
  `action` (z. B. `ADD_CONNECTION(n3, n7)`, `ENABLE(n5)`, `DISABLE(n2→n8)`).
- `genome.developmental_forward(inputs, allow_development=True)`:
  1. Normaler Forward-Pass.
  2. Nach jedem Forward: pruefe alle DevelopmentalRules.
  3. Feuere Regeln deren Bedingung erfuellt ist.
  4. Resultierende Strukturaenderungen sind *episoden-lokal*.
  5. `genome.reset()` setzt auf Basis-Topologie zurueck.
- `genome.freeze_development()`: deaktiviert alle Entwicklungsregeln
  (eingefrorenes Netz fuer Deployment).
- Entwicklungsregeln sind NICHT episoden-lokal: sie evolvieren normal
  zwischen Generationen.

**Mutations-Operatoren:**

- `add_developmental_rule()`: neue Regel mit zufaelliger Bedingung/Aktion.
- `remove_developmental_rule()`: Regel entfernen.
- `mutate_rule_condition()`: Bedingungsschwellwert aendern.
- `mutate_rule_action()`: Aktionstyp wechseln.
- Regeln werden wie Connections ueber Innovation-Numbers getrackt.

**Anwendungsfaelle:**

- Adaptive Agenten, die in einer Episode neue Faehigkeiten entwickeln.
- Continual-Learning ohne explizites Re-Training.
- Exploration: Entwicklung erzeugt Verhaltensvarianz innerhalb einer Episode.

**Benchmark:**

- Wechselnde-Umgebung-Task (Umgebung aendert sich alle 50 Schritte):
  Developmental vs. normaler NEAT (Adaptionsgeschwindigkeit).
- Open-Ended-Maze: Agent entdeckt neuen Bereich → entwickelt neue Connections.

**Akzeptanzkriterien:**

- `developmental_forward()` fuegt waehrend einer Episode tatsaechlich
  Connections hinzu (Topologie aendert sich).
- `genome.reset()` stellt Basis-Topologie wieder her.
- `freeze_development()` unterdrueckt alle Regel-Anwendungen.
- Entwicklungsregeln werden korrekt vererbt und mutiert.
- Checkpoint-Roundtrip erhaelt alle DevelopmentalRules.
- Tests: Regel-Trigger; Episoden-Lokalitaet der Aenderungen;
  freeze-Entwicklung; Vererbung; Checkpoint-Persistenz.

---

### □ P2 Continual / Lifelong Learning NEAT

Ein Genom das sequenziell mehrere Aufgaben lernt, ueberschreibt frueheres
Wissen (Catastrophic Forgetting). Continual-Learning-NEAT schuetzt wichtige
Connections und expandiert das Netzwerk selektiv — inspiriert von Elastic
Weight Consolidation (EWC) und Progressive Neural Networks.

**Ziel:** `train()` mit aufgabenweisem Training: Genom loest Aufgabe 1, dann
Aufgabe 2, ..., und behaelt Performance auf allen vorherigen Aufgaben.

**Design: `ContinualLearning` in `evolution/continual.py`**

```python
yane.set_continual_learning(mode="ewc", lambda_ewc=0.1,
                            progressive=True)
```

**Modi:**

1. **EWC-Mode:** Nach jeder Aufgabe werden Fisher-Information-Matrix und
   finale Gewichte gespeichert. Fitness = Aufgaben-Fitness − λ *
   Σ_i F_i * (θ_i − θ*_i)² (Strafe fuer Gewichtsabweichung von vorherigen
   Aufgaben).
2. **Progressive-Mode:** Neue Aufgabe erhaelt neue Hidden-Nodes (laterale
   Expansion). Bestehende Connections werden eingefroren. Neue Connections
   duerfen von alten Nodes lesen, aber nicht umgekehrt.
3. **Memory-Replay:** Periodische Re-Evaluation auf gespeicherten Samples
   frueherer Aufgaben (Replay-Buffer). Keine Fisher-Matrix noetig.
4. **Hybrid:** EWC + Progressive + Replay kombiniert.

**API:**

```python
yane.set_continual_learning(mode="hybrid", lambda_ewc=0.1,
                            progressive=True, replay_size=200)

# Aufgabe 1
yane.configure(n_inputs=4, n_outputs=2)
yane.task_start("cartpole")
yane.train(cartpole_evaluator)

# Aufgabe 2 (ohne configure!)
yane.task_start("lunarlander")
yane.train(lunarlander_evaluator)

# Test
print(yane.evaluate_all_tasks())
# {"cartpole": 195.0, "lunarlander": 120.0}
```

**Diagnostics:**

- `forgetting_rate`: Fitness-Delta pro Aufgabe nach neuem Training.
- `forward_transfer`: Verbesserung auf neuer Aufgabe durch vorheriges Wissen.
- `frozen_connections`: Anzahl eingefrorener Connections.
- `task_memory_usage`: Speicherverbrauch pro Aufgabe (Replay-Buffer,
  Fisher-Matrix).

**Benchmark:**

- 3-Gym-Sequenz (CartPole → Acrobot → MountainCar): Forgetting-Rate vs.
  normales Neu-Training.
- Permuted-MNIST (10 Permutationen sequenziell): Accuracy auf allen Tasks.

**Akzeptanzkriterien:**

- Nach Aufgabe 2: Fitness auf Aufgabe 1 ≥ 90% der urspruenglichen Fitness.
- Progressive-Mode: neue Connections lesen von alten Nodes, nicht umgekehrt.
- `task_start("name")` speichert Fisher-Matrix/Replay-Buffer automatisch.
- `evaluate_all_tasks()` gibt korrekte Fitness pro Aufgabe zurueck.
- Tests: EWC-Penalty-Berechnung; Progressive-Expansion; Replay-Buffer;
  Forgetting-Rate; Multi-Task-Evaluation.

---

### □ P2 Meta-Learning NEAT — Few-Shot Adaptation

Statt ein Genom fuer EINE Aufgabe zu optimieren, wird es fuer schnelle
*Adaptionsfaehigkeit* optimiert: das Genom soll sich in wenigen
Evaluierungs-Schritten an eine neue, unbekannte Aufgabe anpassen koennen.

**Ziel:** `meta_train()` evolviert Genome, die nach 2-3 Episoden auf einer
neuen Aufgabe gut performen — MAML-inspiriert (Model-Agnostic Meta-Learning),
aber evolutionaer.

**Design: `MetaLearningNEAT` in `evolution/meta_learning.py`**

```python
yane.meta_train(task_sampler, meta_iterations=1000,
                adaptation_steps=3, inner_lr=0.01)
```

**Ablauf:**

1. `task_sampler()` gibt eine zufaellige Aufgabe zurueck (z. B. XOR mit
   zufaelliger Input-Permutation, CartPole mit variierender Gravitation).
2. Inner Loop: Genom wird fuer `adaptation_steps` mit Lamarck-Refinement
   an die Aufgabe angepasst (nur Gewichte, keine Topologie).
3. Fitness = Performance NACH Adaptation (nicht vorher!).
4. Outer Loop: normale NEAT-Evolution selektiert Genome mit hoher
   Post-Adaptation-Fitness.

**Implementation:**

- Lamarck-Refinement als Adaptations-Mechanismus (kein Backprop — bleibt im
  YANE-Paradigma).
- `adaptation_steps=3` mit Hill-Climb oder CMA-ES auf der aktuellen Aufgabe.
- `inner_lr`: Lernrate fuer Lamarck-Adaptation.
- Fitness wird nur nach der Adaptation gemessen; Zwischenwerte in Diagnostics.
- Meta-Training kann auf denselben Aufgaben wiederholt werden (wenige
  Evaluierungen pro Adaptation).

**Anwendungsfaelle:**

- Robotik: Roboter soll sich in 2-3 Versuchen an neue Untergruende anpassen.
- Spiel-Agent: Generalist der schnell neue Level lernt.
- Scientific Discovery: schnelle Anpassung an neue Messdaten.

**Diagnostics:**

- `pre_adaptation_fitness`: Fitness vor Inner-Loop-Adaptation.
- `post_adaptation_fitness`: Fitness nach Inner-Loop-Adaptation.
- `adaptation_delta`: Fitness-Verbesserung durch Adaptation.
- `meta_generalization`: Performance auf gehaltenen Test-Aufgaben.

**Benchmark:**

- Sinus-Regression (MAML-Standard): Meta-NEAT vs. normaler NEAT
  (Post-Adaption-MSE nach 3 Schritten).
- Gym-Task-Familie (CartPole mit variierender Laenge/Masse): Few-Shot-Adaption.

**Akzeptanzkriterien:**

- Post-Adaptation-Fitness > Pre-Adaptation-Fitness (Adaptation wirkt).
- Meta-trainiertes Genom adaptiert schneller als zufaellig initialisiertes.
- `adaptation_steps=0` deaktiviert Inner Loop (normales Training).
- `task_sampler` wird korrekt in jeder Meta-Iteration aufgerufen.
- Tests: Inner-Loop-Lamarck; Adaptation-Delta; Meta-vs-Normal-Vergleich;
  Task-Sampler-Integration.

---

### □ P2 Evolutionary Reservoir Computing

Reservoir Computing (Echo State Networks, Liquid State Machines) trennt ein
festes, zufaelliges rekurrentes Reservoir von einer evolvierbaren Readout-
Schicht. Extrem schnell: nur Readout-Gewichte evolvieren, keine Struktur-
Mutation noetig.

**Ziel:** `ReservoirGenome` mit fixiertem Reservoir und evolvierbaren
Readout-Connections — effizient fuer Echtzeit-Zeitreihen und Edge-Devices.

**Design: `ReservoirGenome` in `evolution/reservoir.py`**

```python
yane.configure_reservoir(n_reservoir=100, spectral_radius=0.9,
                          input_scaling=0.5, leaking_rate=0.3,
                          connectivity=0.1)
```

- Reservoir: `n_reservoir` Hidden-Nodes mit zufaelliger, fixierter rekurrenter
  Verdrahtung (`connectivity`-Anteil).
- `spectral_radius`: Skalierung der rekurrenten Gewichtsmatrix (<1 garantiert
  Echo-State-Property).
- `leaking_rate`: Leaky-Integrator (1.0 = kein Leak, 0.0 = kein Update).
- Readout: Connections von Reservoir-Nodes zu Output-Nodes. Nur DIESE sind
  evolvierbar.
- Keine Strukturmutationen (kein add_node, add_connection) → stark
  vereinfachte Evolution.
- `genome.forward(inputs)`: Reservoir-State wird einmal pro Forward
  aktualisiert; Readout linear oder mit Aktivierung.
- Optional: `RidgeRegressionReadout` statt evolutionaerem Readout
  (analytische Loesung via Pseudoinverse).

**Vorteile:**

- Sehr schnell: kein Crossover, keine Speziation, keine Topologie-Suche.
- Deterministisch: gleiche Seeds → gleiche Ergebnisse.
- Geringer Speicher: fixierte Topologie, nur Readout-Gewichte im Checkpoint.
- Ideal fuer Zeitreihen-Prognose, Signalverarbeitung, Embedded-Systeme.

**Integration:**

- `NeuroEvolution.set_reservoir_mode(enabled=True)`: deaktiviert Struktur-
  mutationen, Speziation und Novelty Search.
- `genome.get_reservoir_state()`: liest aktuellen Reservoir-Aktivierungsvektor
  (fuer externe Analyse).
- Ridge-Readout via `np.linalg.lstsq`: eine Zeile Code, keine Evolution noetig.

**Benchmark:**

- Mackey-Glass-Zeitreihe: Reservoir vs. normaler NEAT (MSE, Trainingszeit).
- NARMA-10 (nichtlineare Zeitreihe): Reservoir mit Ridge vs. evolutionaer.
- CartPole mit Reservoir (State→Reservoir→Action): Performance vs. normaler NEAT.

**Akzeptanzkriterien:**

- Reservoir-State ist deterministisch bei gleichem Seed.
- `spectral_radius < 1`: Echo-State-Property erfuellt (kein Chaos).
- Readout-Gewichte evolvieren korrekt (nur Readout aendert sich).
- Ridge-Readout loest XOR-nahen Task ohne Evolution.
- Tests: Reservoir-State-Determinismus; Echo-State-Property-Test;
  Readout-Evolution; Ridge-Loesung; Checkpoint (nur Readout).

---

### □ P2 Open-Ended Evolution / Minimal Criterion

Novelty Search allein driftet in Bereiche die „neu" aber sinnlos sind
(z. B. statische Netze mit konstantem Output). Ein *minimal criterion*
begrenzt die Suche auf Genome, die eine Mindestleistung erfuellen — und
erlaubt darueber hinaus uneingeschraenkte Exploration.

**Ziel:** `set_minimal_criterion(fn)` filtert Genome vor Selektion: wer das
Kriterium nicht erfuellt, wird aus der Population entfernt.

**Design: `MinimalCriterion` in `evolution/openended.py`**

```python
yane.set_minimal_criterion(lambda g: g.fitness > -50.0)
yane.set_open_ended(mode="novelty_with_criterion", archive_size=200)
```

**Modi:**

- `novelty_with_criterion`: Novelty Search + Minimal Criterion. Nur Genome
  die das Kriterium erfuellen kommen ins Novelty-Archiv und werden zur
  Fortpflanzung zugelassen.
- `curiosity_with_criterion`: Curiosity-Modul (bereits implementiert) +
  Minimal Criterion.
- `quality_diversity_with_criterion`: MAP-Elites + Minimal Criterion.
- `elite_with_exploration`: Top-Elite belegt Plaetze; Rest wird durch
  Novelty gefuellt.

**Verhalten bei Kriteriumsverletzung:**

- Genome die das Kriterium nicht erfuellen: werden markiert (`viable=False`),
  nicht zur Fortpflanzung zugelassen, aber in Diagnostics gezaehlt.
- `min_viable_frac` (Default: 0.1): wenn weniger als 10% der Population
  viable sind → Kriterium temporaer lockern (adaptive Schwelle).
- `viable_boost`: viable Genome erhalten Bonus im Selection-Score.

**Anwendungsfaelle:**

- Kreative Problemloesung: „loese XOR zu 80%, aber WIE ist egal".
- Exploration in High-Dimensional-Raeumen mit Qualitaetssicherung.
- Quality Diversity: MAP-Elites mit Minimal-Criterion-Boden.

**Diagnostics:**

- `viable_fraction`: Anteil Genome die Kriterium erfuellen.
- `criterion_threshold`: aktueller Schwellwert (bei adaptiver Lockerung).
- `archive_viable_ratio`: Anteil viable Genome im Novelty-Archiv.
- `exploration_frontier`: max. Novelty unter viable Genomen.

**Benchmark:**

- Maze-Navigation: Novelty+MinCriterion vs. reine Novelty vs. Fitness-basiert.
- Bildgenerierung (CPPN): MinCriterion (z. B. „mindestens 3 verschiedene
  Farben") vs. reine Novelty.

**Akzeptanzkriterien:**

- Genome unter Kriterium werden nicht zur Fortpflanzung zugelassen.
- Adaptive Lockerung greift wenn `viable_frac < min_viable_frac`.
- `viable=True`-Genome haben hoeheren Selection-Score.
- Kriterium kann zur Laufzeit geaendert werden.
- Tests: Kriteriums-Filter; adaptive Schwelle; viable-Boost;
  Archiv-Integration; open-ended-Modus-Umschaltung.

---

### □ P2 Multi-Agent Cooperation (kooperativ, nicht adversarial)

Anders als Self-Play (kompetitiv): mehrere Genome muessen *kooperieren*, um
eine gemeinsame Aufgabe zu loesen. Fitness ist geteilt oder kombinatorisch.

**Ziel:** `set_cooperative_population(n_agents)` trainiert N Genome gemeinsam
an einer kooperativen Aufgabe.

**Design: `CooperativePopulation` in `evolution/cooperative.py`**

```python
yane.set_cooperative_population(n_agents=3, credit="shared")
```

**Credit-Assignment-Modi:**

- `shared`: alle Agenten erhalten dieselbe Global-Fitness (Team-Performance).
- `difference`: Fitness = Team-Fitness − Team-Fitness_ohne_Agent (Shapley-
  approximiert, aufwendig).
- `individual`: jeder Agent hat eigene Fitness-Komponenten innerhalb der
  Team-Evaluation (z. B. individueller Score + Team-Bonus).
- `hierarchical`: ein Lead-Agent erhaelt Global-Fitness, andere erhalten
  lokale Hilfs-Fitness.

**Evaluator-Schnittstelle:**

```python
def cooperative_evaluate(agents: list[Genome]) -> list[float]:
    # agents[0], agents[1], agents[2] steuern gemeinsam
    team_score = simulate_team(agents)
    return [team_score] * len(agents)  # shared credit
```

**Rollenzuweisung:**

- `role_specialization=True`: Agenten evolvieren automatisch in
  spezialisierte Rollen durch unterschiedliche Fitness-Landschaften.
- `role_diversity_bonus`: Penalty wenn zwei Agenten identisches Verhalten
  zeigen (fördert Spezialisierung).
- `role_tags`: optionale benutzerdefinierte Rollen („Angreifer",
  „Verteidiger", „Sammler").

**Diagnostics:**

- `role_similarity`: durchschnittliche Verhaltensaehnlichkeit zwischen
  Agenten (niedrig = gute Spezialisierung).
- `cooperation_index`: Korrelation zwischen Team-Fitness und Agent-
  Austausch (misst gegenseitige Abhaengigkeit).
- `free_rider_detection`: Agenten die deutlich unter Team-Durchschnitt
  performen.

**Benchmark:**

- Multi-Agent-Particle-Environments (MPE): kooperative Navigation.
- Team-Task (2 Agenten bewegen gemeinsam ein grosses Objekt): Kooperation
  vs. Einzelagent.

**Akzeptanzkriterien:**

- N Agenten erhalten korrekte Fitness nach `shared`/`difference`/`individual`.
- `role_similarity` sinkt ueber Generationen (Spezialisierung).
- Crossover nur innerhalb der eigenen Rolle (kein cross-role Crossover).
- Checkpoint speichert/ladt alle N Agenten.
- Tests: Credit-Assignment-Modi; Rollen-Spezialisierung; Diversity-Bonus;
  Free-Rider-Erkennung.

---

### □ P1 Interactive Evolution — Human-in-the-Loop

Nicht jede Aufgabe hat eine objektive Fitness-Funktion. Bei kreativen oder
subjektiven Aufgaben (Design, Kunst, Musik, Praeferenzlernen) bewertet der
Mensch die Genome — interaktiv ueber die GUI.

**Ziel:** `InteractiveEvaluator` in der GUI, die menschliches Feedback
(Favoriten, Rankings, Slider) als Fitness-Signal verwendet.

**Design: `InteractiveEvaluator` in `gui/interactive_eval.py`**

**Feedback-Modi:**

1. **Paarweiser Vergleich:** GUI zeigt zwei Genome nebeneinander
   (Output-Visualisierung oder Verhaltens-Animation). Nutzer waehlt das
   bessere aus. Elo-Rating wird als Fitness verwendet.
2. **Rating:** Nutzer bewertet jedes Genom auf einer Skala (1-5 Sterne,
   Slider 0-100).
3. **Ranking:** Nutzer ordnet K Genome nach Praeferenz. Rank wird in
   Fitness umgerechnet (1. Platz = K, letzter Platz = 1).
4. **Implicit:** Nutzerverhalten (Verweildauer, Interaktionen) wird als
   implizites Fitness-Signal verwendet.

**Ablauf:**

1. Population generiert Kandidaten.
2. GUI praesentiert Genom zur Bewertung (z. B. Output-Bild, Animation,
   Sound, Text).
3. Mensch gibt Feedback.
4. Feedback wird in numerische Fitness umgerechnet.
5. Population aktualisiert, naechste Generation.
6. Optional: `surrogate_model=True` — ein Surrogate-Modell lernt die
   menschlichen Praeferenzen und evaluiert Genome automatisch zwischen
   menschlichen Feedback-Runden (reduziert Nutzer-Ermuedung).

**GUI-Integration:**

- Neuer Tab oder Panel: "Interactive Evolution".
- Fortschrittsbalken: „Bewerte 10 von 50 Genomen dieser Generation".
- Visualisierung des aktuellen Genoms (z. B. CPPN-Bild, Netzwerk-Topologie,
  Gym-Rendering).
- Speichern/Laden des menschlichen Praeferenzmodells (Surrogate).

**API:**

```python
yane.set_interactive_evaluation(mode="pairwise", surrogate_model=True,
                                surrogate_update_interval=5)
yane.submit_feedback(genome_id, rating)  # programmatisch
```

**Anwendungsfaelle:**

- CPPN-Bilderzeugung: Mensch waehlt aesthetisch ansprechende Muster.
- Spiel-Agent-Verhalten: Mensch bewertet „menschliches" Verhalten.
- Produktdesign: Evolutionaerer Design-Optimierer mit menschlichem Feedback.

**Benchmark:**

- CPPN-Bild-Evolution: 10 menschliche Bewertungen → Surrogate uebernimmt →
  Bildqualitaet nach 50 automatischen Generationen.
- Simulierter Mensch (synthetische Praeferenzfunktion) fuer reproduzierbare
  Tests.

**Akzeptanzkriterien:**

- Paarweiser Vergleich produziert konsistente Elo-Ratings (Test mit
  simuliertem Nutzer).
- Surrogate-Modell reduziert Anzahl benoetigter menschlicher Bewertungen
  (mit simuliertem Nutzer messbar).
- GUI zeigt mindestens zwei Genome gleichzeitig an.
- `submit_feedback()` aktualisiert Fitness korrekt.
- Tests: Elo-Update; Surrogate-Vorhersage; Feedback→Fitness-Konvertierung;
  Modus-Wechsel.

---

### □ P2 Probabilistic / Bayesian NEAT

Klassische NEAT-Nodes geben Punktwerte aus. Bayesian NEAT erweitert Nodes um
eine Varianz-Ausgabe: `y = μ(x), σ(x)`. Das Genom lernt nicht nur WAS die
richtige Aktion ist, sondern auch WIE SICHER es sich ist.

**Ziel:** `NodeType.PROBABILISTIC` mit dualem Output (μ, σ) und Uncertainty-
gewichteter Fitness.

**Design: `ProbabilisticNode` via `NodeType.PROBABILISTIC`**

- Jeder probabilistische Node hat ZWEI Aktivierungswerte: `mu` (Mittelwert)
  und `log_sigma` (Log-Standardabweichung, via `softplus` garantiert >0).
- Forward-Pass waehrend Training: `output = mu + sigma * epsilon` (epsilon ~
  N(0,1), Reparameterization Trick).
- Forward-Pass waehrend Inferenz/Deployment: `output = mu` (deterministisch).
- `bayesian_forward()`: fuehrt N Samples aus und gibt `(mean, std)` zurueck.

**Fitness-Berechnung mit Uncertainty:**

- `fitness = task_fitness - uncertainty_penalty * mean(sigma)` (belohnt
  kalibrierte Sicherheit).
- `fitness = mean(task_fitness over N samples)` (robuster, reduziert
  Overfitting auf Noise).
- Optional: `Expected Improvement`-basierte Fitness fuer Bayesian
  Optimization-Anwendungen.

**Mutations-Operatoren:**

- `add_probabilistic_node()`: normalen Node in probabilistischen Node
  umwandeln (oder neuen hinzufuegen).
- `mutate_sigma_prior(node)`: Log-Sigma-Bias mutieren.
- Probabilistische Node-Connections tragen doppelte Gewichte: eins fuer μ,
  eins fuer σ (oder geteilt).

**Layer-3-Integration:**

- `NeuroEvolution.set_probabilistic(enabled=True, n_samples=5)`.
- Bei `enabled=False`: probabilistische Nodes verhalten sich wie normale
  Nodes (μ-Only).
- `genome.forward()` im Training: stochastisch; in Deployment: deterministisch
  (via `genome.set_inference_mode(True)`).
- Diagnostics: durchschnittliches σ pro Layer, Uncertainty-Kalibrierung
  (sollte mit tatsaechlichem Fehler korrelieren).

**Anwendungsfaelle:**

- Risiko-averse Entscheidungen (Finanz-Trading, Robotik-Safety).
- Active Learning: Genom sagt „ich bin unsicher" → mehr Daten sammeln.
- Bayesian Optimization: Genom als Surrogate-Modell mit Uncertainty.

**Benchmark:**

- Noisy-XOR (Inputs mit Gauss-Noise): Bayesian vs. normaler NEAT (Robustheit).
- UCI-Regression mit Uncertainty: Kalibrierung (σ korreliert mit |pred -
  true|).

**Akzeptanzkriterien:**

- `bayesian_forward(n=100)` reduziert Varianz des Outputs im Vergleich zu
  n=1.
- `inference_mode=True`: deterministischer Output (σ=0 effektiv).
- Uncertainty-Penalty verhindert Overconfidence auf noisy data.
- Crossover: probabilistische und normale Nodes kreuzbar.
- Tests: Reparameterization-Sampling; Inference-Mode; Sigma-Output-Range;
  Uncertainty-Kalibrierung; Crossover-Kompatibilitaet.

---

### □ P2 Safety-Constrained Evolution (Safe NEAT)

In sicherheitskritischen Anwendungen (Robotik, autonomes Fahren, Medizin)
duerfen bestimmte Aktionen oder Zustaende NIE auftreten. Safe NEAT evolviert
Genome unter harten Nebenbedingungen: Verletzungen → Evaluation sofort
abgebrochen, Fitness = −∞.

**Ziel:** `set_safety_constraints(constraints)` definiert unverletzbare Regeln;
Genome die sie brechen werden eliminiert.

**Design: `SafetyConstraint` in `evolution/safety.py`**

```python
yane.set_safety_constraints([
    SafetyConstraint("joint_limit",
                     check=lambda state, action: max(action) < 1.0,
                     penalty=-1000.0),
    SafetyConstraint("collision",
                     check=lambda state, action: state[0] > 0.0,
                     mode="hard"),  # bricht Evaluation sofort ab
])
```

**Constraint-Typen:**

- `SafetyConstraint(mode="hard")`: bei Verletzung → Evaluation abbrechen,
  Fitness = `penalty` (Default: −∞).
- `SafetyConstraint(mode="soft")`: Fitness-Reduktion proportional zur
  Verletzungshaeufigkeit.
- `SafetyConstraint(mode="barrier")`: logarithmische Barriere-Funktion
  (stetig, differenzierbar, Lamarck-kompatibel).

**Constraint-Check:**

- Vor jeder Aktion: `check(state, action)` wird aufgerufen.
- `mode="hard"`: bei erster Verletzung → `raise SafetyViolation`.
- `mode="soft"`: Zaehler fuer Verletzungen; Fitness = base_fitness −
  penalty * violation_count.
- `mode="barrier"`: Fitness += barrier_scale * log(safety_margin).

**Population-Management:**

- `min_safe_frac` (Default: 0.5): wenn weniger als 50% der Population
  sicher sind → `safety_boost`-Modus (sichere Genome erhalten Bonus).
- `safety_archive`: speichert sicher getestete Genome und deren
  Constraint-Verifikation.
- `safety_mutation_budget`: unsichere Genome duerfen mehr mutieren
  (groessere Suchsprünge in sichere Regionen).

**Diagnostics:**

- `safety_violations`: Haeufigkeit pro Constraint-Typ.
- `safe_fraction`: Anteil sicherer Genome in der Population.
- `safety_margin`: minimaler Abstand zur Constraint-Grenze.
- `violation_heatmap`: welche Zustaende/Aktionen verletzen Constraints
  (fuer Debugging).

**Benchmark:**

- CartPole mit Action-Limit (Aktion < 0.5): Safe NEAT vs. normaler NEAT
  mit Penalty.
- Robot-Arm: Joint-Limits (Hard Constraints) — Safe NEAT haelt Limits ein,
  normaler NEAT nicht.
- Grid-World mit Lava-Zellen: Safe-NEAT lernt Lava zu vermeiden.

**Akzeptanzkriterien:**

- Hard-Constraint-Verletzung → Fitness = penalty, keine weiteren
  Evaluierungs-Schritte.
- Soft-Constraint: Fitness-Degradation proportional zur Verletzungshaeufigkeit.
- `min_safe_frac`-Mechanismus schuetzt sichere Genome vor Verdraengung.
- Constraints sind serialisierbar (Checkpoint-Kompatibilitaet).
- Tests: Hard-Constraint-Abbruch; Soft-Constraint-Penalty; Barrier-Modus;
  Safe-Fraction-Schutz; Constraint-Serialisierung.

---

### □ P1 Hardware-Aware NEAT (Deployment-Constraint-Evolution)

Genome werden in der Cloud oder auf lokaler Workstation evolviert, aber auf
Edge-Geraeten deployed. YANE soll Genome direkt unter Hardware-Constraints
evolvieren: maximale FLOPs, Speicher, Latenz, Energie.

**Ziel:** `set_hardware_constraints()` erweitert `efficiency_score` um reale
HW-Metriken und bestraft Genome die HW-Budget ueberschreiten.

**Design: `HardwareConstraint` in `evolution/hardware_aware.py`**

```python
yane.set_hardware_constraints(
    max_flops=1_000_000,      # 1M FLOPs pro Forward-Pass
    max_memory_bytes=4096,    # 4KB
    max_latency_us=100,       # 100µs
    target_platform="cortex-m4"
)
```

**Metrik-Berechnung (deterministisch, aus Genom-Topologie):**

- `flops_per_forward()`: Zaehlt Multiplikationen + Additionen pro Forward.
  n_inputs→hidden: n_connections Multiplikationen + n_nodes Additionen.
- `memory_bytes()`: `n_nodes * sizeof(Node) + n_connections *
  sizeof(Connection)`. Konstanten aus tatsaechlicher Python-Struct-Groesse
  oder Zielplattform-Definition.
- `latency_us()`: Schaetzung basierend auf FLOPs / Zielplattform-MHz *
  Safety-Factor.

**Integration:**

- `genome.hardware_profile()`: gibt `HardwareProfile(nodes, connections,
  flops, memory, latency)` zurueck.
- Fitness-Penalty: `fitness -= hw_penalty_weight * max(0, (actual -
  constraint) / constraint)` → proportionale Strafe fuer Ueberschreitung.
- `constraint_violation`: binaer (0/1), wird in Diagnostics gefuehrt.
- GUI zeigt Hardware-Profil des besten Genoms.

**Zielplattform-Profile:**

- Vordefinierte Profile: `"cortex-m4"`, `"cortex-m7"`, `"esp32"`,
  `"raspberry-pi-zero"`, `"raspberry-pi-4"`, `"desktop"`.
- Profile definieren MHz, SRAM, FPU-Verfuegbarkeit, Energiekosten pro FLOP.
- Nutzerdefinierte Profile via Dict.

**Diagnostics:**

- `hw_flops_actual`: tatsaechliche FLOPs des besten Genoms.
- `hw_memory_actual`: tatsaechlicher Speicherverbrauch.
- `hw_constraint_violations`: Anteil Genome die HW-Budget ueberschreiten.
- `hw_pareto_front`: Fitness-vs-FLOPs-Kurve der Population.

**Benchmark:**

- XOR unter Cortex-M4-Constraints (max. 10K FLOPs): HW-aware NEAT findet
  kompakte Loesung.
- CartPole unter ESP32-Constraints: Fitness-vs-FLOPs-Tradeoff.

**Akzeptanzkriterien:**

- `hardware_profile()` gibt korrekte FLOPs/Memory/Latenz zurueck.
- Fitness-Penalty ist proportional zur Constraint-Ueberschreitung.
- Vordefinierte Profile laden korrekte Werte.
- `hw_pareto_front` enthaelt nicht-dominierte Genome.
- Tests: FLOPs-Zaehlung; Memory-Schaetzung; Penalty-Berechnung;
  Plattform-Profile; Pareto-Front.

---

### □ P2 Sparse NEAT / Lottery Ticket Hypothesis

Die Lottery Ticket Hypothesis besagt: in zufaellig initialisierten Netzen
existieren „glueckliche" Subnetzwerke die allein genauso gut performen wie
das ganze Netz. Sparse NEAT findet diese Subnetzwerke durch iteratives
Magnitude Pruning.

**Ziel:** `genome.find_lottery_ticket()` identifiziert das sparseste
Subnetzwerk das ≥99% der Original-Fitness haelt.

**Design: `LotteryTicketSearch` in `evolution/lottery_ticket.py`**

```python
ticket = genome.find_lottery_ticket(target_sparsity=0.1,
                                     max_fitness_drop=0.01,
                                     iterations=5)
```

**Algorithmus (Iterative Magnitude Pruning):**

1. Trainiere Genom normal (NEAT + Lamarck).
2. Entferne p% der Connections mit kleinstem |Gewicht|.
3. Fine-tune das geprunte Genom (Lamarck, N Schritte).
4. Wenn Fitness-Drop < `max_fitness_drop`: weiter zu Schritt 2.
5. Wenn Fitness-Drop > `max_fitness_drop`: letztes gueltiges Ticket
   wiederherstellen.
6. Wiederhole bis `target_sparsity` erreicht oder Drop zu gross.

**Unterschied zu Post-Training Pruning (P2-bereits-implementiert):**

- Post-Training Pruning: eine Runde, entfernt schwache Connections.
- Lottery Ticket: ITERATIV, mit Re-Training zwischen Runden, findet
  minimale Struktur.
- Lottery Ticket erkennt „glueckliche" Initialisierungen (nicht nur
  schwache Connections).

**API:**

```python
ticket = genome.find_lottery_ticket(target_sparsity=0.05,
                                     max_fitness_drop=0.02,
                                     iterations=10,
                                     method="magnitude")
genome.apply_ticket(ticket)  # ersetzt Genom durch Ticket
```

- `method="magnitude"`: Standard-IMP (nach |Gewicht|).
- `method="gradient"`: Gewichte mit kleinstem Lamarck-Gradient entfernen
  (teurer, aber genauer).
- `method="random"`: Baseline (Random Pruning).

**Diagnostics:**

- `ticket_sparsity`: finale Sparsity (behaltene Connections / urspruengliche).
- `ticket_fitness_drop`: finaler Fitness-Verlust.
- `ticket_rounds`: Anzahl IMP-Iterationen.
- `ticket_mask`: binaere Maske (welche Connections behalten wurden).

**Benchmark:**

- XOR: Lottery Ticket mit <50% Connections vs. Post-Training Pruning.
- CartPole: Ticket mit 30% Connections haelt >90% Performance.
- Vergleich IMP vs. Magnitude-Pruning vs. Random.

**Akzeptanzkriterien:**

- `find_lottery_ticket()` findet Ticket mit `target_sparsity` oder bricht
  mit Begruendung ab.
- Ticket-Fitness ≥ Original-Fitness − `max_fitness_drop`.
- IMP-Methode findet sparsere Tickets als Random (p < 0.05 ueber 10 Runs).
- Ticket ist serialisierbar und als eigenes Genom speicherbar.
- Tests: IMP-Iteration; Ticket-Sparsity; Fitness-Drop-Check;
  Methoden-Vergleich; Ticket-Serialisierung.

---

### □ P1 Evolutionary Data Augmentation

Bei kleinen Trainingsdatensaetzen overfittet NEAT. Evolutionaere Data
Augmentation findet automatisch Augmentierungen die die Generalisierung
verbessern — die Augmentierungsparameter sind selbst evolvierbar.

**Ziel:** `set_evolutionary_augmentation()` evolviert Augmentierungs-Pipelines
parallel zur Genom-Evolution.

**Design: `AugmentationPipeline` in `evolution/augmentation.py`**

```python
yane.set_evolutionary_augmentation(enabled=True,
    augmentation_space=[
        "gaussian_noise", "dropout_noise", "scaling",
        "translation", "mixup", "cutout"
    ],
    population_augmentations=10
)
```

- `AugmentationGenome`: evolvierbares Genom das eine Augmentierungs-Pipeline
  repraesentiert (nicht mit `Genome` zu verwechseln — ist ein separates,
  kleines Genom nur fuer Augmentierung).
- Jedes Augmentierungs-Gen: `(type, probability, magnitude)`.
- Pipeline: mehrere Augmentierungen sequenziell oder parallel anwendbar.
- Co-Evolution: Augmentierungsgenome evolvieren parallel zur NEAT-Population.
  Fitness = NEAT-Test-Set-Performance (je besser das augmentierte Training,
  desto fitter die Augmentierung).
- Jedes NEAT-Training verwendet ein Augmentierungsgenom aus dem Pool.

**Augmentierungstypen:**

- `gaussian_noise`: Input + N(0, σ²). σ evolvierbar.
- `dropout_noise`: zufaellig Input-Dimensionen auf 0 setzen. Dropout-Rate
  evolvierbar.
- `scaling`: Input * factor. Factor evolvierbar.
- `translation`: Input + offset. Offset evolvierbar.
- `mixup`: zwei Samples mischen: `λ*sample_a + (1-λ)*sample_b`. λ aus
  Beta-Verteilung, α evolvierbar.
- `cutout`: rechteckige Regionen im Input auf 0 setzen.
- `label_smoothing`: Target-Wert glaetten (für Klassifikation).

**Evolutions-Mechanik:**

- Augmentierungsgenome haben eigene Population (Groesse:
  `population_augmentations`).
- Selektion: Tournament-Selection basierend auf NEAT-Test-Set-Fitness.
- Crossover: Augmentierungs-Pipelines kreuzen (Gene austauschen).
- Mutation: Augmentierungsparameter gaußsch mutieren, Typ wechseln,
  hinzufuegen/entfernen.

**Diagnostics:**

- `best_augmentation_pipeline`: finale Pipeline-Parameter.
- `augmentation_impact`: Fitness-Delta mit vs. ohne Augmentierung.
- `augmentation_diversity`: Pipeline-Vielfalt (wie viele verschiedene Typen).

**Benchmark:**

- Small-Data-XOR (nur 4 Trainingssamples): Augmentation vs. ohne.
- UCI-Datasets mit <100 Samples: Generalisierungs-Test-Set-Performance.

**Akzeptanzkriterien:**

- Augmentierungs-Pipeline-Parameter veraendern sich ueber Generationen.
- Test-Set-Fitness mit Augmentierung > ohne Augmentierung (bei Small Data).
- Augmentierungs-Crossover produziert valide Pipelines.
- Augmentierungsgenome sind unabhaengig von NEAT-Genomen (getrennte Pools).
- Tests: Pipeline-Forward; Augmentierungs-Mutation; Crossover; Fitness-
  Attribution; Small-Data-Benchmark.

---

### □ P2 Genome-to-TFLite / Embedded Export

Neben ONNX und WASM: TensorFlow Lite Micro ist der Standard fuer Embedded-
Inferenz auf Mikrocontrollern (Arduino, ESP32, Cortex-M). TFLite bietet
8-Bit-Quantisierung und laeuft ohne Heap-Allocation.

**Ziel:** `genome_to_tflite()` exportiert Genome als quantisiertes TFLite-
Modell fuer Embedded-Deployment.

**Design: `genome_to_tflite()` in `evolution/tflite_export.py`**

```python
genome.export_tflite("model.tflite", quantization="int8",
                     representative_dataset=calibration_data)
```

**Export-Pipeline:**

1. `genome_to_torch_module()` → PyTorch-Modul.
2. `torch.onnx.export()` → ONNX-Modell.
3. `onnx2tf` oder direkte TFLite-Conversion → TFLite-Modell.
4. Optional: Quantisierung (FP16, INT8, Hybrid).
5. Optional: `representative_dataset` fuer INT8-Kalibrierung.

Alternative (wenn torch nicht verfuegbar): direkte TFLite-Modellkonstruktion
via `tflite.Model`-API (aufwendiger aber keine Abhaengigkeit).

**Embedded-spezifische Optimierungen:**

- `arena_size`: TFLite-Arena-Groesse in Bytes (fuer Allocator-Planung).
- `remove_dynamic_ops`: alle Operationen muessen statisch sein (keine
  dynamischen Shapes, kein `tick()`-Modus).
- `inlining`: kleine Aktivierungsfunktionen direkt in den Graphen inlinen.
- Nur azyklische Netze unterstuetzt (zyklische → entrollt).

**Quantisierungs-Modi:**

- `"fp32"`: keine Quantisierung, volle Genauigkeit.
- `"fp16"`: halbe Genauigkeit (2x kleinere Groesse).
- `"int8"`: 8-Bit-Ganzzahl (4x kleinere Groesse, schnellere Inferenz).
- `"hybrid"`: Gewichte int8, Aktivierungen fp32 (beste Balance).

**Output-Formate:**

- `model.tflite`: FlatBuffer-Datei.
- `model.cc`: C-Quellcode-Array (`const unsigned char model[] = {...}`) zum
  Einbetten in Firmware.
- `model.h`: C-Header mit Modell-Groesse.

**Validierung:**

- `validate_tflite()`: TFLite-Inferenz mit Test-Inputs → Output-Vergleich
  mit YANE-Forward (Toleranz: 1e-3 fuer fp32, 1e-1 fuer int8).

**Benchmark:**

- Inferenzzeit TFLite (ARM Cortex-M Simulator) vs. YANE-Forward.
- Modellgroesse: Pickle vs. TFLite-fp32 vs. TFLite-int8.

**Akzeptanzkriterien:**

- XOR-Genom als TFLite exportiert: TFLite-Inferenz innerhalb 1e-3 (fp32).
- INT8-quantisiertes Modell: Outputs innerhalb 5% relativen Fehlers.
- `model.cc`-Export kompilierbar (syntaktisch korrektes C-Array).
- Azyklisches Netz: voll unterstuetzt; zyklisches → klarer Fehler.
- Tests: TFLite-Export; Quantisierungs-Vergleich; C-Array-Export;
  Validierungsfunktion.

---

### □ P2 Symbolic Regression Export (Genom → mathematische Formel)

Statt eines numerischen Blackbox-Netzwerks: `genome_to_symbolic()` extrahiert
eine analytische, menschenlesbare Formel aus dem Genom. Aktivierungsfunktionen
werden durch ihre mathematische Notation ersetzt.

**Ziel:** Ein Genom als geschlossene mathematische Formel exportieren —
interpretierbar, ueberpruefbar, in Paper/Publikationen verwendbar.

**Design: `genome_to_symbolic()` in `evolution/symbolic_export.py`**

```python
formula = genome.to_symbolic(input_names=["x", "y"],
                              output_names=["f"],
                              format="latex")
print(formula)
# f(x,y) = sin(0.5*x - 0.3*y) + tanh(0.8*x + 0.2)
```

**Algorithmus:**

1. Topologische Sortierung des Genoms.
2. Jeder Node wird in einen mathematischen Ausdruck uebersetzt:
   - Input-Node: `input_names[i]` (z. B. `"x"`, `"y"`).
   - Hidden/Output-Node: `activation(bias + Σ weight_i * source_i)`.
   - Rekurrente Verbindungen: entrollt oder als rekursive Referenz
     (`f_{t-1}`).
3. Common-Subexpression-Elimination: wiederverwendete Teilausdruecke werden
   als `h_1 = ...; h_2 = ...` definiert.
4. Konstanten-Faltung: `0.0 * x` → entfernt, `1.0 * x` → `x`, `x + 0.0` → `x`.
5. Symbolische Vereinfachung (optional, via `sympy`): `sin(x)² + cos(x)²` → `1`.

**Ausgabe-Formate:**

- `"latex"`: LaTeX-Formel (`f(x) = \sin(0.5x - 0.3y) + \tanh(0.8x + 0.2)`).
- `"python"`: Python-Lambda (`lambda x, y: math.sin(0.5*x - 0.3*y) + ...`).
- `"sympy"`: SymPy-Expression (weiterverwendbar).
- `"text"`: einfache ASCII-Darstellung.

**Einschraenkungen:**

- Nur azyklische Netze: voll unterstuetzt.
- Zyklische Netze: entrollt mit `unroll_steps` → Formel mit Index `_{t-1}`.
- Custom-Aktivierungen: Export als Funktionsname (z. B. `gelu(x)`).
- Grosse Netze (>100 Nodes): Formel wird unuebersichtlich → Warnung, CSE
  erzwingen.

**Anwendungsfaelle:**

- Physik-Simulation: Genom entdeckt physikalisches Gesetz → Formel-Export
  zeigt es.
- Interpretierbare KI: Formel statt Blackbox.
- Wissenschaftliche Publikationen: Genom-Ergebnisse als Gleichung zitierbar.

**Benchmark:**

- Symbolic-Regression-Tasks (Nguyen, Koza): Genom-Formel vs. Ground-Truth-
  Formel (Edit-Distanz, R²).
- Physik-Datensatz: entdeckte Formel vs. bekanntes Gesetz.

**Akzeptanzkriterien:**

- `to_symbolic()` gibt syntaktisch korrekte Formel im gewuenschten Format.
- Symbolic-Output evaluiert zu identischen Werten wie `genome.forward()`
  (Toleranz 1e-6).
- Konstanten-Faltung entfernt `0.0`, `1.0` Multiplikationen.
- LaTeX-Output ist kompilierbar (valide LaTeX-Syntax).
- CSE reduziert Formel-Laenge bei Netzwerken mit Shared Subgraphs.
- Tests: Format-Korrektheit; Output-Aequivalenz; Konstanten-Faltung;
  LaTeX-Validierung; CSE-Reduktion.
