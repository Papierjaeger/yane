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
- Naechste Schwerpunkte: P2-Research-Spikes ehrlich zu vollstaendigen Features ausbauen oder als experimentell belassen; Checkpoint-Kompatibilitaets-Diff haerten.

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
DARTS, STDP, Neuromodulation, Curiosity, ES-HyperNEAT, Input-/Output-Gruppierung, Convolutional NEAT.
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
