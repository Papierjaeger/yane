# Tasks: YANE staerker machen

Diese Datei ist die aktuelle Roadmap fuer YANE. Offene und neue Tasks stehen
oben. Abgeschlossene Arbeit ist weiter unten nur noch kompakt zusammengefasst.

## Status

**Aktueller Stand:** Alle P0-Bausteine fertig. Zuletzt: Experiment Tracking (RunDatabase/SQLite, Run-Objekt, reproduce_run, experiment() Kontext, 10 neue Tests) und Selektionsstrategie als Plugin (SelectionStrategy-Protokoll, TournamentSelection/ElitistSelection/FitnessProportional/RankSelection/NoveltyOnlySelection, per-Species-Override, Diagnostics, 14 neue Tests). Offene P1-Schwerpunkte: Adaptive Policy System. Teststand: `867 passed`.

- Core-Evolution, Speciation, Mutation, Worker-Pipeline, GUI, API, Logging, Checkpoints: implementiert.
- Multi-Objective, Quality Diversity, CMA-ES, Backprop-/Matrix-Bausteine, Presets, Benchmark-Gates: implementiert.
- AdaptiveController, OperatorScheduler, Lamarck-Budget, Interspecies-Trigger (Novelty/Isolation/Schutz): implementiert.
- Adaptive Benchmark-Suite, GUI-Stability-Guard, Preset-Schema v2 mit `adaptive_policies`: implementiert.
- Matrix-Forward-Integration, Checkpoint-State fuer Adaptive-Objekte: implementiert.
- Checkpoint gehaertet: Fixture-Dateien, JSON-Metadaten in GUI+API, Pickle-Dokumentation: implementiert.
- Remote/Distributed Evaluation: HTTP-Protokoll, Client/Worker, Retry/Cancel, Benchmark: implementiert.
- P2-Forschungsfeatures: Modulbibliothek, CPPN, Meta-adaptive Policies, Evolvierbare Descriptor-Gewichte: implementiert.
- raw_fitness-Fix (Fitness-Komponenten verschmutzen nicht mehr genome.fitness fuer Ziel-Check und Diagnostics): implementiert.
- Event-System, Anomalie-Detektion, Fitness-Transformer, Genome-Export, Validierungs-Set, Konfigurationspersistenz, Gym-Inspect-Verbesserung: implementiert.
- Naechste Schwerpunkte: P1 Architektur-Tasks (Adaptive Policy System, Experiment Tracking/Run-DB, Selektionsstrategie als Plugin).

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
Experimentelle Features werden in einem separaten `experimental/`-Verzeichnis oder Research Branch entwickelt. Nicht direkt in den Stable Core mischen.

Ziel: Layer 1 wartbar halten; Layer 2 durch das Policy-Interface erweiterbar machen; Layer 3 isoliert fuer riskante Forschung.

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

### □ P1 Adaptive Policy System

Verschiedene Adaptive-Mechanismen (Recovery, Diversity, Pop-Resize, Lamarck Burst, Fitness Shaping, Online Tuning, Surrogate-Gating) sind derzeit als separate Regelkreise implementiert. Das fuehrt zu konkurrierenden Aktionen, schwerer Debugbarkeit und doppeltem Code.

**Aktueller Stand:** `AdaptiveController` passt Lamarck-Budget und Interspecies-Rate regelbasiert an. `OperatorScheduler` passt Mutations-Gewichte an. `AnomalyDetectorSet` emittiert Events. Kein einheitliches Interface, keine Policy-Reihenfolge, keine Konflikterkennung zwischen Policies.

*Das Adaptive Recovery System (P0) ist die erste konkrete Implementierung, die spaeter unter diesem Interface modelliert werden soll. Das Policy System ist die Vereinheitlichung, nicht der Ersatz.*

**Policy-Interface:**

```python
class AdaptivePolicy:
    def observe(ctx: TrainingContext) -> None: ...   # Zustand lesen, intern updaten
    def decide(ctx: TrainingContext) -> Action | None: ...  # Aktion vorschlagen oder None
    def apply(ctx: TrainingContext, action: Action) -> None: ...  # Aktion ausfuehren
```

- `TrainingContext`: enthaelt Generation, Fitness-Stats, IQR, Anomalie-Signale, aktive Policies, Letzte-Aktionen.
- `Action`: typisiertes Objekt mit `priority: int` und `conflict_group: str` (Policies derselben Gruppe exkludieren sich).
- `NeuroEvolution.register_policy(policy, enabled=True)`.
- `NeuroEvolution.set_policy_order(policy_names)`: legt Evaluierungsreihenfolge fest.
- Policies werden in definierter Reihenfolge aufgerufen; Konflikte (gleiche `conflict_group`) werden nach `priority` aufgeloest.
- Langfristig: Recovery, Diversity Injection, Population Resize, Lamarck Burst, Fitness Shaping, Online Tuning und Surrogate-Gating als registrierte Policies.

**Diagnostics:**

- `active_policies`: Liste aktiver Policies mit Reihenfolge.
- `last_policy_actions`: pro Policy die letzte ausgefuehrte Aktion.
- `policy_rewards`: Fitness-Delta nach jeder Policy-Aktion (Grundlage fuer spaetere Bandit-Evaluation).
- `policy_conflicts`: Faelle wo Policies derselben Gruppe konkurriert haben.

**Tests:**

- Policy wird registriert; `observe → decide → apply` wird in korrekter Reihenfolge aufgerufen.
- Deaktivierte Policy veraendert nichts am Trainingslauf.
- Zwei Policies mit gleichem `conflict_group` feuern nie gleichzeitig; hoehere `priority` gewinnt.
- Benutzerdefinierte Policy kann via `register_policy` eingebunden werden.

### □ P1 Modular Compatibility Distance / Genome Descriptor

Die aktuelle Kompatibilitaets-Distanz in der NEAT-Speziation ist auf Verbindungsanzahl und Gewichtsdifferenz beschraenkt. Sobald Genome komplexer werden — Module, Input-/Output-Gruppen, Conv-Bloecke, Plastizitaetsregeln — reicht diese einfache Distanz nicht mehr um Spezies korrekt zu trennen.

**Aktueller Stand:** Kompatibilitaets-Berechnung in `population.py`: `delta = c1 * excess + c2 * disjoint + c3 * mean_weight_diff`. Feste Formel; keine Plugin-Punkte; ignoriert Aktivierungsfunktionen, Module, Gruppen-Strukturen oder andere Genome-Erweiterungen.

- `GenomeDescriptor`-Klasse: beschreibt Topologie, Gewichte, Aktivierungsfunktionen, Module, Input-/Output-Gruppen, Conv-Bloecke, Plastizitaetskoeffizienten.
- `DistanceMetric`-Protokoll: `compute(desc_a: GenomeDescriptor, desc_b: GenomeDescriptor) -> float`.
- Eingebaute Metriken: `topology_distance` (bestehend, NEAT-Formel), `weight_distance`, `activation_distance` (Jaccard ueber Aktivierungstypen), `module_distance`, `group_distance` (Input-/Output-Grouper-Struktur).
- `NeuroEvolution.set_compatibility_metric(metric | ChainMetric([...], weights=[...]))`.
- Alte `c1/c2/c3`-NEAT-Formel bleibt als `topology_distance` Standard erhalten (kein Breaking Change).
- Besonders wichtig fuer spaetere Tasks: Input-Gruppierung, Output-Gruppierung, Shared Weights, ConvNEAT, STDP, Neuromodulation, DARTS.

**Tests:**

- Identische Genome haben Distanz 0 fuer alle Metriken.
- Gewichtsunterschiede erhoehen `weight_distance`, aber nicht `topology_distance`.
- Zusaetzliche Module erhoehen `module_distance`.
- Alte Kompatibilitaets-Berechnung mit `topology_distance` liefert identisches Ergebnis wie zuvor.
- `ChainMetric` kombiniert Metriken korrekt mit den konfigurierten Gewichten.

### ✓ P1 Experiment Tracking / Run Database

Es gibt kein zentrales `Run`-Objekt. Erkenntnisse aus verschiedenen Konfigurationen muessen manuell aus CSV-Logs zusammengesetzt werden. Reproduzierbare Re-Runs sind nicht moeglich ohne manuelle Konfigurationsdokumentation.

**Aktueller Stand:** `RunRecord` in `util/run_history.py` implementiert mit `load()`, `list_runs()`, `list_categories()`, `group_by_config()` — liest CSV-Logs aus `logs/`-Verzeichnis; 8 Tests in `test_diagnostics_features.py`. `save_config`/`load_config` speichert Konfiguration. Fehlend: vollstaendiges `Run`-Objekt mit `diagnostics`/`artifacts`-Feldern, SQLite-Layer, `reproduce_run()`.

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

Die Selektionsstrategie (Tournament) ist fest in `population.py` verdrahtet; kein Austausch ohne Code-Aenderung.

**Aktueller Stand:** Tournament-Selektion (k=3) ist als interne Closure `_selection_score` in `population.py` implementiert. Fitness-Sharing, Species-Budget-Aufteilung und Komplexitaetsstrafe sind eingebaut und nicht austauschbar. `OperatorScheduler` passt Mutations-Gewichte an, aber die Selektion selbst ist nicht als Plugin-Punkt ausgelegt.

- `SelectionStrategy`-Protokoll: `select(population, n) -> list[Genome]`.
- Eingebaute Strategien: `TournamentSelection(k=3)`, `ElitistSelection(top_frac=0.2)`, `FitnessProportional()`, `NoveltyOnlySelection()`, `RankSelection()`.
- `NeuroEvolution.set_selection_strategy(strategy)`.
- Per-Species-Ueberschreibung: `set_selection_strategy(strategy, species_id=...)`.
- Diagnostics: aktive Strategie, Durchschnittsfitness der selektierten vs. nicht-selektierten Genome.
- Tests: jede Strategie gibt korrekte Anzahl Genome zurueck; Fitness-Proportional ist stochastisch korrekt.

### ⚡ P1 Evaluation-Middleware-Stack

Evaluatoren sind einfache Callables ohne Kompositionsmechanismus; Caching, Normierung und Noise-Injection muessen pro Evaluator manuell implementiert werden.

**Aktueller Stand:** Basis implementiert in `evolution/eval_middleware.py`.
`NeuroEvolution.add_eval_middleware()` und `clear_eval_middleware()` sind
verfuegbar; Middleware laeuft in LIFO-Reihenfolge. Eingebaut sind
`CachingMiddleware`, `TimingMiddleware`, `RetryMiddleware`,
`ComponentMiddleware` und `CaseBatchMiddleware` inklusive Diagnostics. Offen:
`NoiseMiddleware` und GUI-Anzeige fuer Komponentenwerte.

- ✓ `EvalMiddleware`-Protokoll: `__call__(genome, eval_fn, ctx) -> float`.
- ✓ Eingebaute Middleware: `CachingMiddleware(maxsize=512)` (Genome-Hash → Fitness), `TimingMiddleware` (Eval-Zeit pro Genom), `RetryMiddleware(n=3, aggregation="mean")`.
- `NoiseMiddleware(sigma=0.05)` (Input-Perturbation).
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

### □ P1 Generationsreport / Run-Postmortem

Nach einem Trainingslauf gibt es keine strukturierte Zusammenfassung; Erkenntnisse muessen manuell aus Logs und CSV herausgezogen werden.

**Aktueller Stand:** `_write_run_summary()` schreibt nach Training eine kompakte Textzeile ins Log und in die CSV. Das `run_end`-Event traegt `stop_reason`, Iterations-Anzahl und Best-Fitness. Kein self-contained Bericht, keine Fitness-Kurven-Visualisierung, kein Mutations-Attribution-Block.

*Export-Layer: baut auf Experiment Tracking (Run-Objekt und RunDatabase) auf. Der Report liest Daten aus dem `Run`-Objekt und exportiert sie in ein menschenlesbares Format.*

- `yane.export_run_report(path, format="html"|"json"|"md")`: generiert Bericht nach `run_end`.
- Inhalte: Fitness-Kurve (SVG-Inline fuer HTML), beste Genome (Topologie + Score), Mutations-Attribution, Anomalie-Log, Konfiguration, Laufzeit, Recovery-Events.
- HTML-Report ist self-contained (kein externer CSS/JS); JSON-Report ist maschinenlesbar.
- GUI: „Report exportieren"-Button im Diagnostics-Tab; oeffnet gespeicherte Datei im Browser.
- `yane.set_report_autosave(path_template="{date}_{example}_report.html")`.

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

### □ P1 Hybrid Feature-Extractor API (Input-Vorverarbeitung)

NEAT eignet sich schlecht fuer hochdimensionale Eingaben (Bilder, Sensorrauschen), weil der Suchraum explodiert. Mit einer vorgeschalteten Transformation — CNN-Layer, PCA, Autoencoder — reduziert NEAT den Suchraum auf das semantisch relevante.

**Aktueller Stand:** `fitness_fn` erhaelt das Genome und ruft `genome.forward(raw_inputs)` selbst auf. Kein eingebauter Mechanismus um Inputs vor `forward()` zu transformieren; der Nutzer muss die Transformation manuell in den Evaluator einbauen — unklar fuer andere Runs, nicht im Checkpoint gespeichert.

- `NeuroEvolution.set_input_transform(fn)`: beliebige Funktion `(raw_inputs: list[float]) -> list[float]` wird vor jedem `genome.forward()` aufgerufen.
- Praxisfall 1: vortrainierter CNN-Truncated-Layer als Feature-Extraktor (z.B. ResNet18 bis Avg-Pool), NEAT evolviert nur den Kopf (10–512 Inputs).
- Praxisfall 2: PCA-Projektion auf 20 Komponenten aus einem Offline-Dataset.
- `n_inputs` der Population muss mit der Output-Dimension der Transformation uebereinstimmen; Validierung beim `configure()`-Aufruf.
- Transform wird im `save_config()`-Blob als Pickle-Referenz gespeichert (falls serialisierbar).
- Tests: Transform wird vor jedem `forward()` aufgerufen; Dimension-Mismatch loest `ValueError` aus; ohne Transform identisches Verhalten wie bisher.

### □ P1 Fitness-Landscape-Visualisierung (PCA / t-SNE)

Konvergenzverhalten und Populationsstruktur sind nur ueber Zahlen erkennbar; die geometrische Verteilung im Genotyp-Raum ist unsichtbar.

**Aktueller Stand:** Diagnostics-Tab zeigt skalare Metriken (Best-Fitness, IQR, Species-Count, Stagnation-Count). Pareto-Scatter und MAP-Elites-Heatmap existieren fuer QD-Laeufe. Keine geometrische Darstellung der Gesamtpopulation im Gewichts-/Topologie-Raum.

- Feature-Vektoren: `GenomeDescriptor` aus dem Modular-Compatibility-Distance-Task als feste Repr&auml;sentation verwenden (Topologie-Statistiken, Gewichts-Statistiken, Aktivierungsverteilung) — direkte Gewichtsvektoren sind wegen variabler Genome-Struktur nicht alignment-sicher.
- PCA (2 Komponenten, ohne externe Deps) &uuml;ber diese Descriptor-Vektoren; optionales t-SNE (via `scikit-learn` falls installiert) als Alternative.
- GUI: interaktive Scatter-Punktwolke im Diagnostics-Tab; Farbe = Fitness, Form = Species, Groesse = Komplexitaet.
- Hover-Tooltip mit Genome-ID, Species, Fitness, Nodes/Connections.
- Export: Snapshot als PNG; alle Projektionspunkte als CSV.
- Benchmark: Visualisierung auf XOR-Lauf zeigt Species-Cluster klar getrennt.

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

### ⚡ P1 Connection-Weight-Histogramm und Gewichtsgesundheit

Pathologien wie Vanishing/Exploding Weights oder symmetrische Gewichtsverteilungen sind in den Diagnostics unsichtbar.

**Aktueller Stand:** Core-Diagnostics implementiert unter `weight_health`:
Histogramm, Mean/Std, 5./95.-Perzentil, Dead-/Saturated-Fraction und Warnung
bei Kollaps/Explosion. Offen: GUI-Histogramm-Widget und `.npy`-Export.

- ✓ Pro Generation: Gewichtsverteilung der gesamten Population als Histogramm (20 Bins, -5 bis +5).
- ✓ Kennzahlen: Mean, Std, 5./95.-Perzentil, Anteil toter Gewichte (|w| < 0.01), Anteil saturierter Gewichte (|w| > 4.9).
- GUI: Histogramm-Widget im Diagnostics-Tab (aktualisiert sich waehrend Training).
- ⚡ Warnung wenn Std < 0.05 (Kollaps) oder > 3.0 (Explosion); noch ohne N-Generationen-Streak.
- Export: Gewichtsmatrix des besten Genoms als `.npy` (NumPy-Format) fuer externe Analyse.

### □ P1 Erweiterbare Aktivierungsfunktionen

Das Aktivierungsset (sigmoid, tanh, relu, leaky_relu, swish, linear) ist im Code fest verdrahtet.

**Aktueller Stand:** `ActivationType` ist ein Python-Enum in `util/activation.py`; `ACTIVATION_FNS` ist ein statisches Dict. 15 Aktivierungen sind implementiert (inkl. ELU, Gaussian, Sine, Cube). Neue Typen koennen nur durch Quellcode-Aenderung und Enum-Erweiterung hinzugefuegt werden; kein Laufzeit-Registrierungsmechanismus.

- `NeuroEvolution.register_activation(name, fn, backprop_fn=None)` registriert benutzerdefinierte Aktivierungen zur Laufzeit.
- Registrierte Funktionen werden beim Checkpoint mit gespeichert (Name → Lambda/Pickle).
- Vorschlaege fuer eingebaute Erweiterungen: GELU, Mish, SiLU.
- GUI-Checkbox-Liste zeigt verfuegbare Aktivierungen; Nutzer kann erlaubte Typen einschraenken.

### □ P1 Multi-Population Inselmodell

Einzelne Population konvergiert in lokale Optima; unabhaengige Inseln verbessern die Exploration.

**Aktueller Stand:** YANE verwendet eine einzige `Population`-Instanz; alle Genome konkurrieren global unter einer gemeinsamen Species-Struktur. Kein Mechanismus fuer mehrere parallele Populationen mit periodischem Genome-Austausch. Langtests zeigen besonders bei Pendulum und CarRacing Seed-Abhaengigkeit trotz erreichbarer Ziel-Fitness; das Inselmodell ist dort plausibler als noch mehr lokales Reward-Shaping.

- `NeuroEvolution.set_island_model(n_islands, migration_rate, migration_interval)`.
- Jede Insel laeuft als eigene `Population`-Instanz; periodisch werden die besten N Genome zwischen zunaechst zufaelligen, dann topologie-nah gewaehlten Inseln migriert.
- Diagnostics: Fitness pro Insel, Migrations-Events, Diversitaets-Abstand zwischen Inseln.
- Benchmark: Vergleich Einzel-Population vs. Inselmodell auf Pendulum,
  CarRacing, Regression 2→2 und Multiplication. Erfolgskriterien: hoehere
  Seed-Erfolgsrate oder niedrigere Median-Zeit bei gleichem Eval-Budget.

### □ P1 Hyperparameter-Suche

Nutzer muessen Parameter (Pop-Groesse, Target-Species, Lamarck-Steps usw.) manuell ausprobieren.

**Aktueller Stand:** Kein systematischer Such-Mechanismus im Core. Nutzer starten verschiedene Konfigurationen manuell und vergleichen CSV-Logs. `save_config`/`load_config` erleichtert das Speichern von Konfigurationen. Das Benchmark-Harness `benchmarks/long_examples.py` kann Beispiele bereits parallel mit CPU-/RAM-Headroom starten, Teilresultate schreiben und Berichte mergen; daraus soll die Such-Infrastruktur wiederverwendet werden. Baut auf Experiment Tracking (RunDatabase) auf.

- `yane.hyperparameter_search(param_grid, n_seeds, fitness_fn, budget_iterations)`: laeuft N Konfigurationen parallel und gibt Ranking zurueck.
- Strategien: Grid-Search, Random-Search, einfaches Bayes-Opt (via `scikit-optimize`).
- Jede Konfiguration wird als eigener `Run` in der RunDatabase gespeichert.
- Wiederverwendung des Long-Examples-Schedulers: `--parallel auto`, CPU-/RAM-
  Headroom, Worker-Pause/Resume, Teilresultate pro Run.
- GUI: eigener „Suche"-Tab mit Parameterraum-Editor und Ergebnis-Tabelle.
- Ergebnisse als CSV exportieren; beste Konfiguration direkt in Trainings-Tab uebertragen.
- Akzeptanzkriterium: Suche kann mindestens zwei Taxi-Evaluator-Varianten
  parallel ueber 3 Seeds vergleichen und eine nach Erfolgsrate + Median-Zeit
  sortierte Tabelle erzeugen.

### □ P1 Ensemble-Bewertung und -Deployment

Ein einzelnes Genom ist stochastisch; ein Ensemble aus den Top-K ist robuster.

**Aktueller Stand:** `get_best()` gibt das einzelne beste Genome zurueck. `population._evaluated` enthaelt alle bewerteten Genome sortierbar nach Fitness. Kein Ensemble-Wrapper, kein Aggregations-Mechanismus ueber mehrere Genome hinweg.

- `yane.get_ensemble(k=5)` → `EnsembleGenome`-Wrapper mit `forward(inputs)`.
- Strategien: Mittelwert der Outputs, Mehrheitsvoting (diskret), gewichtete Kombination nach Fitness.
- `EnsembleGenome.to_python()` und `to_onnx()` exportieren alle K Genome gemeinsam.
- GUI: Ensemble-Groesse im Inspect-Tab einstellen, Ensemble-Output als zusaetzliche Spalte anzeigen.

### ⚡ P1 Strukturierte / maschinenlesbare Protokollierung

**Weitgehend implementiert.** `set_log_format("jsonlines"|"csv"|"both")`, `set_tensorboard_logdir(path)` und `set_log_callbacks(on_generation=fn)` existieren und funktionieren. Fehlende Restpunkte:

- Fitness-History-CSV um Validierungs-Fitness-Spalte ergaenzen (Wert ist in `population_memory_info()` vorhanden, wird aber nicht ins CSV geschrieben).
- Ensemble-Fitness-Spalte ergaenzen (sobald Ensemble-Bewertung implementiert ist).

### □ P1 Erweiterte Genome-Analyse im Inspect (Sensitivitaet / Attribution)

Nutzer sehen Ausgaben, aber nicht WARUM das Genom so entscheidet.

**Aktueller Stand:** Inspect-Tab zeigt Roh-Output des Genoms fuer manuelle Eingaben und den interpretierten Aktions-Label (fuer Gym-Beispiele mit `action_display_fn`). Keine Input-Attribution, keine Sensitivity-Analyse, keine Visualisierung toter Knoten.

- Sensitivity-Analyse: fuer jeden Input-Kanal den Output bei +0.1 / -0.1 Perturbation messen → Einfluss-Score je Input.
- Inspect-Tab: Balken-Diagramm mit Input-Relevanz, farblich nach Einflussrichtung.
- „Toter Knoten"-Marker: Knoten/Verbindungen, die bei allen Test-Cases nie feuern, visuell hervorheben.
- Vergleich: gleiche Analyse fuer Genome am Anfang und am Ende des Trainings.

### □ P1 Plugin-System fuer benutzerdefinierte Evaluatoren

Eigene Umgebungen einzubinden erfordert bisher direktes Editieren der `examples.py`.

**Aktueller Stand:** Evaluatoren werden als `ExampleConfig`-Instanzen in `gui/examples.py` registriert. Kein Plugin-Protokoll, keine automatische Erkennung von Evaluator-Klassen, kein Plugin-Verzeichnis-Scan. Nutzer muessen YANE-Quellcode direkt editieren.

- `ExamplePlugin`-Protokoll: Python-Klasse mit `name`, `make_eval`, `n_inputs`, `n_outputs`, `target_fitness`.
- `yane.register_example(plugin)` registriert den Evaluator in der GUI-Liste.
- Plugins werden aus einem konfigurierbaren Plugin-Verzeichnis automatisch geladen (`~/.yane/plugins/`).
- Dokumentation + Beispiel-Plugin-Template als Quickstart.

### ⚡ P1 Lernkurven-Vergleich (mehrere Runs)

Einzelne Runs sind nicht repraesentativ; Vergleich verschiedener Konfigurationen ist nicht moeglich.

**Aktueller Stand:** `ComparisonTab` in `gui/tabs/comparison_tab.py` implementiert mit `refresh()`, `_reload_records()`, `_update_stats()`, `_export_png()`, `_export_csv()` — nutzt `RunRecord` fuer ueberlagerte Fitness-Kurven. Fehlend: Median/IQR-Baender ueber mehrere Seeds derselben Config, bestaetigt funktionierende PNG/CSV-Exporte. Baut auf Experiment Tracking (RunDatabase) auf.

*GUI-Layer: laedt mehrere `Run`-Objekte aus der RunDatabase und visualisiert sie vergleichend.*

- GUI: „Vergleich"-Ansicht mit ueberlagerten Fitness-Kurven (bis zu 4 Runs, farbcodiert).
- Statistische Zusammenfassung: Median, 25./75.-Perzentil ueber N Wiederholungen derselben Konfig.
- Export: Vergleichs-Plot als PNG, Rohdaten als CSV.

### □ P1 Fitness-Surrogate-Modell (Billigfilter vor teurer Evaluierung)

Teure Evaluierungen (Simulationen, Gym-Umgebungen) werden fuer jedes Genom gleich oft durchgefuehrt; ein billiges Surrogate-Modell koennte schwache Genome frueh aussortieren.

**Aktueller Stand:** `EvaluationRunner` evaluiert jedes Genome mit der echten `fitness_fn`. Kein Mechanismus um Genome vorab zu filtern. `population_memory_info()` liefert Fitness-Statistiken, aber kein Modell das Fitness aus Genome-Features vorhersagt.

*Architektur: Adaptive Evaluation Budgeting — teilt Budget-Verwaltung, Promotion-Fraktion und Diagnostics-Schema mit Anytime-Evaluation (P0). Surrogate filtert vorab per Modell-Vorhersage; Anytime bewertet die promovierten Genome mehrfach. Beide koennen kombiniert werden.*

- `NeuroEvolution.set_surrogate(enabled=True, warmup_evals=200, surrogate_frac=0.5)`.
- Surrogate: lineares Modell ueber Genome-Features (Gewichts-Statistiken, Topologie-Groesse, Aktivierungsverteilung).
- Training: Surrogate wird auf bereits bewerteten Genomen (Replay-Buffer der letzten 3 Generationen) trainiert.
- Nutzung: Surrogate filtert untere `surrogate_frac` aus; nur obere Fraktion wird real evaluiert.
- `surrogate_frac` adaptiv: steigt wenn Surrogate-Ranking gut vorhersagt (Spearman-Rho > 0.7), sinkt sonst.
- Diagnostics: `surrogate_spearman_rho`, `filtered_fraction`, `saved_real_evals`.
- Tests: Surrogate wird trainiert; Filterquote haelt `surrogate_frac` ein; Spearman-Rho wird korrekt berechnet.

### □ P1 Automatische Fitness-Shaping-Erkennung

Die Fitness-Landschaft (sparse, plateau, skewed) ist unsichtbar; der Nutzer muss manuell entscheiden welche Fitness-Transformationen noetig sind.

**Aktueller Stand:** `RankTransform`, `SigmaScaling`, `LinearNormalize`, `ClipTransform` und `ChainTransform` sind verfuegbar und per `set_fitness_transform()` setzbar. Der Nutzer muss aber selbst entscheiden welche Transformation geeignet ist; keine automatische Landschafts-Diagnose.

- `FitnessLandscapeAnalyzer.analyze(population) -> FitnessLandscapeReport` mit Diagnosen: Sparsity-Score, Plateau-Anteil, Schiefe, Cluster-Trennbarkeit.
- Automatische Empfehlungen: `"apply RankTransform"` bei starker Schiefe, `"apply SigmaScaling"` bei Plateau, `"inject diversity"` bei Sparsity.
- `NeuroEvolution.set_auto_fitness_shaping(enabled=True)`: wendet die empfohlenen Transformationen automatisch an.
- Analyse laeuft alle 50 Generationen; bei `enabled=True` wird die beste Transformation direkt gesetzt.
- GUI: Landscape-Report erscheint im Diagnostics-Tab; Empfehlung und angewendete Transformation werden angezeigt.
- Tests: Analyzer erkennt schief-verteilte Fitness; empfiehlt korrekte Transformation; Auto-Mode setzt Transform.

### □ P1 Online-Hyperparameter-Adaptation (Bandit-Tuning waehrend Training)

Statische Hyperparameter sind fuer alle Trainingsphasen gleich; im Unterschied zur einmaligen offline Suche laeuft das Bandit-Tuning innerhalb eines einzelnen Runs und passt sich dynamisch an die aktuelle Phase an.

**Aktueller Stand:** `AdaptiveController` passt einige Raten (Lamarck-Budget, Interspecies-Rate) regelbasiert an. `OperatorScheduler` passt Mutations-Gewichte per Success-Rate an. Kein Bandit-Algorithmus mit explizitem Exploration/Exploitation-Trade-off fuer beliebige Hyperparameter.

*Langfristig als Policy unter dem Adaptive Policy System (P1) modellieren — kein eigener Regelkreis neben den anderen Adaptive-Mechanismen.*

- `NeuroEvolution.set_online_tuning(enabled=True, params=["mutation_rate", "n_lamarck_steps"])`.
- UCB1-Bandit: diskrete Kandidatenwerte pro Parameter; Arm = Konfiguration; Reward = Fitness-Delta der letzten Generation.
- Exploration-Phase (erste 20% der Iterationen): gleichmaessige Stichprobe; danach Exploitation des bisher besten Arms.
- Diagnostics: aktuelle Werte, Arm-Rewards, Exploration-Rate, Anzahl Arm-Wechsel.
- Tests: Bandit waehlt mehrere Konfigurationen in Exploration-Phase; bester Arm wird danach haeufiger gewaehlt.

### □ P1 Weight-Inheritance beim Crossover (Lamarck-informierte Gewichts-Initialisierung)

Beim Crossover werden Gewichte neuer Verbindungen (die nur ein Elternteil hat) zufaellig initialisiert; Lamarck-optimierte Elterngewichte werden nicht weitervererbt.

**Aktueller Stand:** NEAT-Crossover in `genome.crossover()`: Matching-Connections erben Gewicht per 50/50-Zufallsauswahl von einem der Elternteile. Disjoint/excess-Gene des fitteren Elternteils behalten ihr Gewicht. Kein fitness-gewichtetes Blending; kein Mechanismus der Lamarck-optimierte Gewichte (die typischerweise besser konvergiert sind) bevorzugt.

- Bei strukturellem Crossover: Verbindungen die nur im besseren Elternteil vorhanden sind behalten ihr Gewicht (statt Zufalls-Init).
- Shared Connections: Gewicht = gewichteter Mittelwert beider Elternteile (nach Fitness gewichtet).
- `NeuroEvolution.set_weight_inheritance(enabled=True, blend_alpha=0.7)`: `blend_alpha` steuert Gewichtung des besseren Elternteils.
- Benchmark: Weight-Inheritance vs. Zufalls-Init auf XOR und CartPole (Konvergenzgeschwindigkeit).
- Tests: Shared Connections behalten Elterngewichte; Neue Connections behalten Elterngewicht statt 0; Blend liegt im erwarteten Bereich.

---

### □ P2 Genome-Codec-Protokoll (austauschbare Serialisierung)

Das Checkpoint-Format ist fest auf Pickle festgelegt; alternative Formate (JSON, MessagePack, komprimiertes Binaerformat) sind nicht einsteckbar.

**Aktueller Stand:** `evolution/checkpoint.py` schreibt/liest Pickles mit atomarem Schreiben (Temp-Datei). JSON-Sidecar fuer Metadaten existiert. Inline-Dokumentation erklaert warum Pickle verwendet wird. Kein austauschbares Codec-Interface, keine Migrationslogik.

- `GenomeCodec`-Protokoll: `encode(genome) -> bytes`, `decode(bytes) -> Genome`.
- Eingebaute Codecs: `PickleCodec` (Standard, bestehend), `JsonCodec` (menschenlesbar, nur einfache Genome), `MsgpackCodec` (kompakt, schnell).
- `yane.set_checkpoint_codec(codec)`.
- Checkpoint-Datei enthaelt Codec-Kennung im Header; Ladelogik waehlt automatisch den richtigen Codec.
- Migration: `yane.migrate_checkpoint(path, target_codec)` konvertiert bestehende Pickles.
- Tests: Round-trip encode → decode fuer alle Codecs auf Standard-Genomen.

### □ P2 Konfigurationsversionierung und Kompatibilitaets-Check

Checkpoints enthalten die Population, aber nicht den vollstaendigen Zustand der Konfiguration; spaeters Nachladen kann zu stillem Fehlverhalten fuehren.

**Aktueller Stand:** Checkpoint-Sidecar-JSON enthaelt `version`, `pop_size`, `n_inputs`, `n_outputs` und einen `requires_reattach`-Flag. GUI warnt bei Topologie-Mismatches. Kein SHA-256-Hash der vollstaendigen Konfiguration, kein strukturierter Diff bei Abweichungen, kein `CompatibilityLevel`.

- Jede `ExperimentPreset` erhaelt einen deterministischen Konfigurations-Hash (SHA-256 ueber kanonisches JSON).
- Beim `load_checkpoint()`: Hash wird mit dem gespeicherten verglichen; bei Abweichung erscheint strukturierter Diff der geaenderten Felder.
- `CompatibilityLevel`: `EXACT` (identisch), `COMPATIBLE` (nur unkritische Felder geaendert), `BREAKING` (Inputs/Outputs/Topologie-Constraints geaendert).
- GUI: Warn-Dialog bei `BREAKING`; Checkpoint-Metadaten zeigen Konfig-Hash und Aenderungs-Diff.
- CLI: `python -m yane.checkpoint --diff old.pkl new.pkl` zeigt Konfigurations-Unterschiede.

### □ P2 Transfer Learning / Genome Fine-Tuning

Wissen aus einem trainierten Genom soll auf eine neue Aufgabe uebertragen werden.

**Aktueller Stand:** `warm_start_from_checkpoint(path)` laedt Genome aus einem Checkpoint und passt Eingabe/Ausgabe-Dimension an (Knoten werden hinzugefuegt/entfernt). Kein selektives Einfrieren von Verbindungsgruppen, kein progressive-unfreeze-Mechanismus.

- `yane.load_genome_as_seed(genome, freeze_layers=[...])`: bestimmte Verbindungsgruppen koennen eingefroren werden.
- Lamarck-Feinabstimmung auf neuer Aufgabe ohne Topologie-Aenderung als erste Phase.
- Dann schrittweise Entsperren eingeforener Teile (progressive unfreeze).
- Benchmark: Transfer CartPole → LunarLander vs. Training from scratch.

### □ P2 Offene Evolution / Co-Evolution von Aufgabe und Agent (POET-aehnlich)

Statische Aufgaben fuehren zu Ueberanpassung; co-evolving Environments erzeugen robustere Agenten.

**Aktueller Stand:** `evolution/coevolution.py` und `evolution/curriculum.py` existieren. Curriculum stuetzt sich auf statische Schwierigkeits-Stufen. Kein `EnvironmentGenome`, keine co-evolutionaere Paarung zwischen Agent- und Umgebungs-Populationen.

- `EnvironmentGenome`: evolvierbare Umgebungsparameter (z. B. Hangneigung, Hindernisanzahl).
- Paarung: Agent-Genome werden auf ihren aktuellen Environment-Genome evaluiert.
- Archiv-Mechanismus: nur Agenten-Genome, die auf mindestens einem Environment gut abschneiden, ueberleben.
- Benchmark: Co-Evolution vs. Domain-Randomization auf BipedalWalker mit variabler Bodenbeschaffenheit.

### □ P2 YANE → PyTorch-Bruecke (NAS + Feinabstimmung)

Evoluted Architekturen koennen nicht mit Gradienten-Methoden weiter optimiert werden.

**Aktueller Stand:** `genome_to_python()` exportiert Genome als reines Python (nur `math`). `genome_to_numpy_weights()` liefert die Weight-Matrix. Kein PyTorch-Export, kein `nn.Module`-Wrapper, keine Gradient-Optimierung der evolvierten Topologie.

- `genome.to_torch_module()` → `torch.nn.Module` mit exakt derselben Topologie.
- Gewichte werden uebertragen; danach normales PyTorch-Training moeglich.
- Nutzung: YANE findet gute Architektur (NAS-Rolle), PyTorch optimiert Gewichte weiter.
- Export beruecksichtigt Memory-Knoten als `nn.GRUCell`-Aequivalent.

### □ P2 Genome-Phylogenie (Stammbaum der Innovationen)

Welche Mutation hat den entscheidenden Durchbruch geliefert? Das ist derzeit nicht nachverfolgbar.

**Aktueller Stand:** `evolution/innovation.py` verwaltet Innovation-Nummern fuer Knoten/Verbindungen. `mutation_tracking.py` trackt Mutations-Erfolgsraten pro Typ. Kein Eltern-ID-Tracking, kein Fitness-Delta-Attribution pro Innovation, kein Stammbaum-Objekt.

- `InnovationTracker` protokolliert fuer jede Innovation: Eltern-Genome-ID, Generation, Delta-Fitness.
- `genome.lineage()` gibt Vorfahren-Kette bis zum Urgenome zurueck.
- GUI: Stammbaum-Visualisierung (kollabierbar) mit markierten Schluessel-Mutationen.
- Analyse: welche Mutations-Operatoren fuehren am haeufigsten zu Fitness-Spruengen.

### □ P2 Verhaltensklonierung als Warm-Start

Evolution braucht viele Iterationen bis zu brauchbaren Loesungen; Demonstrationen koennen das beschleunigen.

**Aktueller Stand:** `warm_start_from_checkpoint(path)` erlaubt einen Head-Start aus einem vorherigen Lauf. Kein supervised Vortraining auf Demonstrations-Daten, kein Backprop-basiertes Behaviour-Cloning vor der Evolution.

- `yane.behaviour_clone(demonstrations, n_steps)`: supervised Vortraining des besten Genoms auf Demonstrations-Daten via Lamarck/Backprop.
- Demonstrationen als Liste von `(inputs, outputs)`-Paaren; kein RL-Umgebungsformat noetig.
- Geklontes Genom wird als initiales Seed fuer die Population verwendet.
- Benchmark: BC-Warm-Start vs. random-init auf LunarLander.

### ⚡ P2 Population-Size-Adaptation (Dynamische Pop-Groesse)

Die Populationsgroesse ist nach Start fest; in fruehen Phasen ist eine grosse Population fuer Exploration noetig, spaeter waere eine kleine Population effizienter.

**Aktueller Stand:** `set_adaptive_population(min_size, max_size, growth_rate=0.05, enabled=True)` in `neuro_evolution.py` implementiert — diversitaets- und stagnationsbasiertes Resize an Generationsgrenzen. API-Divergenz: Code nutzt `growth_rate`, Task spezifiziert `schedule`-Enum (`linear_decay`/`performance_based`). Fehlend: explizite Schedule-Modi, Diagnostics-Felder, Tests.

- `NeuroEvolution.set_adaptive_pop_size(min_pop=20, max_pop=500, schedule="linear_decay"|"performance_based")`.
- `linear_decay`: Pop-Groesse nimmt linear von `max_pop` zu `min_pop` ueber den Trainingsverlauf ab.
- `performance_based`: Pop-Groesse sinkt wenn Konvergenzrate hoch ist, steigt bei Stagnation.
- Groesse wird nur an Generationsgrenzen geaendert; Ueberschuss-Genome werden per Fitness-Selection eliminiert.
- Diagnostics: aktuelle Pop-Groesse, letzter Resize-Trigger, Groessen-Historie.
- Tests: Pop-Groesse bleibt in `[min_pop, max_pop]`; bei `linear_decay` sinkt sie monoton; kein Resize innerhalb einer Generation.

### □ P2 Gradient-gesteuerte Mutations-Richtung (Lamarck-Momentum)

Lamarck-Gradienten werden nach jedem Refinement-Schritt verworfen; sie koennten die Mutations-Richtung fuer die naechste Generation informieren.

**Aktueller Stand:** `LamarckRefiner` (Hill-Climb, NES, SA, CMA-ES) berechnet Gewichts-Deltas intern fuer das Refinement, gibt sie aber nach Abschluss nicht weiter. Mutation und Lamarck sind zwei komplett unabhaengige Mechanismen ohne Informationsfluss.

- `LamarckRefiner` speichert den Parameteraenderungs-Vektor (Gradient-Schaetzung) des letzten Schritts.
- Beim Mutieren: mit Wahrscheinlichkeit `momentum_prob` wird die Mutations-Richtung mit dem gespeicherten Momentum gewichtet.
- Momentum-Decay: Gradient-Information verfaellt exponentiell (`decay=0.9`).
- `NeuroEvolution.set_lamarck_momentum(enabled=True, momentum_prob=0.3, decay=0.9)`.
- Benchmark: Mit vs. ohne Momentum auf Symbolic-Regression und CartPole (Konvergenzgeschwindigkeit).
- Tests: Momentum-Vektor wird nach Lamarck-Schritt aktualisiert; bei `momentum_prob=0` kein Einfluss auf Mutation.

### □ P2 Automatisches Post-Training Pruning (Netzwerk-Komprimierung)

Trainierte Genome sind oft ueberdimensioniert; unnoetige Verbindungen und tote Knoten koennen ohne Leistungsverlust entfernt werden.

**Aktueller Stand:** `export_genome_weights()` gibt die Weight-Matrix des besten Genoms zurueck; Nutzer koennen Gewichte extern analysieren. `genome.connection_count` und `len(genome.nodes)` sind zugaenglich. Kein automatisches Pruning, keine `prune()`-Methode, kein Fitness-Rollback nach Komprimierung.

- `genome.prune(threshold=0.01, method="weight"|"activation_frequency")` entfernt Verbindungen unter Schwellenwert.
- `genome.compress(target_size)`: iteratives Pruning bis Zielgroesse erreicht (kleinstes Gewicht zuerst).
- `NeuroEvolution.set_post_training_pruning(enabled=True, threshold=0.01, max_drop_frac=0.02)`: automatisches Pruning nach `run_end`.
- Evaluierung: gepruntes Genom wird einmal neu bewertet; wenn Fitness-Drop groesser als `max_drop_frac` wird Pruning rueckgaengig gemacht.
- `genome.prune_stats()`: Anzahl entfernter Verbindungen/Knoten, Fitness-Delta, Komprimierungsrate.
- Tests: `prune()` entfernt Verbindungen unter Schwellenwert; `compress()` erreicht Zielgroesse; Fitness-Check aktiviert Rollback bei zu grossem Drop.

---

### □ P2 Differenzierbare Topologie-Suche (DARTS-Lite)

> **Experimentell (Layer 3):** Nur in `experimental/` oder Research Branch implementieren, nicht direkt in den Stable Core mischen.

NEAT sucht Topologien diskret durch Mutation; differenzierbare Relaxation erlaubt gradienten-basierte Architektursuche.

**Aktueller Stand:** Topologie-Suche erfolgt ausschliesslich durch diskrete Mutations-Operatoren (`add_connection`, `add_node`, `remove_connection`). Lamarck optimiert Gewichte mit Gradienten, aber nicht die Topologie-Entscheidungen. Kein kontinuierliches Gating-Signal.

- Kontinuierliche Relaxation: Kanten-Gewichte mit Gating-Sigmoid; niedrige Gates werden nach N Schritten geprunt.
- `DARTSOptimizer`: wechselt ab zwischen Lamarck-Schritt (Gewichte) und Architektur-Gradient (Gates).
- Nur fuer Feed-Forward-Genome ohne Memory-Nodes.
- `NeuroEvolution.set_darts_mode(enabled=True, prune_threshold=0.1)`.
- Benchmark: DARTS-Lite vs. Standard-NEAT auf Symbolic-Regression-Task.

### □ P2 Intrinsische Belohnung / Curiosity-Modul

> **Experimentell (Layer 3):** Nur in `experimental/` oder Research Branch implementieren, nicht direkt in den Stable Core mischen.

Sparse-Reward-Umgebungen (z. B. Maze) liefern keine nutzbare Fitness-Information; Curiosity kann die Exploration antreiben. Wichtig: bekannte strukturierte MDPs wie Taxi sollten zuerst mit strukturierten Evaluator-Komponenten, Subgoals und Graph-Distanz geloest werden. Curiosity ist fuer unbekannte oder schlecht modellierbare Umgebungen gedacht, nicht als Ersatz fuer vorhandene Aufgabenstruktur.

**Aktueller Stand:** `AdaptiveFitnessComponentWeights` ermoeglichen Zusatzkomponenten zur Task-Fitness (Komplexitaet, Novelty). Kein internes Vorhersage-Netz, kein Prediction-Error als intrinsischer Reward, keine Curiosity-Architektur.

- `CuriosityModule`: haelt ein kleines Vorhersage-Netz (auch ein YANE-Genom), das naechste Observations vorhersagt.
- Intrinsischer Reward = Vorhersagefehler; wird zur externen Fitness addiert (gewichtet, konfigurierbar).
- Das Vorhersage-Netz wird per Lamarck parallel zur Haupt-Population trainiert.
- `yane.set_curiosity(enabled=True, weight=0.3, network_size=8)`.
- Benchmark: Curiosity vs. kein Curiosity auf einem Sparse-Reward-Maze (eigene
  minimale Implementierung), nicht auf Taxi als Primaer-Benchmark.

### □ P2 Synaptische Plastizitaet (STDP / Hebbsches Lernen)

> **Experimentell (Layer 3):** Nur in `experimental/` oder Research Branch implementieren, nicht direkt in den Stable Core mischen.

Genome lernen derzeit nur durch Evolution, nicht durch Erfahrung innerhalb einer Episode.

**Aktueller Stand:** Lamarck-Refinement optimiert Gewichte zwischen Generationen (inter-lifetime). Innerhalb einer Episode (`genome.forward()`-Aufrufe) sind alle Gewichte unveraenderlich. `genome.reset()` setzt Memory-Knoten zurueck, aber keine evolvierte Plastizitaetsregel.

- Knoten/Verbindungen koennen evolvierte Hebb-Regel-Koeffizienten (A, B, C, D) tragen.
- Gewichte werden waehrend `genome.forward()` nach der STDP-Regel angepasst (intra-lifetime-learning).
- `genome.reset()` setzt Gewichte auf Basiswerte zurueck (Plastizitaet ist episoden-lokal).
- Benchmark: STDP vs. Lamarck auf Aufgaben mit veraenderlicher Umgebung (z. B. wechselnde XOR-Eingaenge).

### □ P2 Neuromodulation

> **Experimentell (Layer 3):** Nur in `experimental/` oder Research Branch implementieren, nicht direkt in den Stable Core mischen.

Modulatorische Signale erlauben kontextabhaengige Gewichtung ganzer Verbindungsgruppen.

**Aktueller Stand:** `NodeType` kennt `INPUT`, `OUTPUT`, `HIDDEN` und `MEMORY`. Kein `MODULATOR`-Knotentyp; Neuromodulation ist nicht im Aktivierungsfluss vorgesehen. Globale Lamarck-Signale beeinflussen das Training, aber keine intra-Netz-Modulation zur Laufzeit.

- Sonderknotentyp `Modulator`: sein Ausgabe-Wert skaliert eingehende Verbindungen anderer Knoten.
- Evolvierbar: welcher Knoten moduliert, welche Verbindungen beeinflusst werden, Staerke.
- Anwendung: schnelle Anpassung an wechselnde Aufgaben (Multi-Task-Szenarien).
- Benchmark: Modulation vs. kein Modulation auf einem Aufgaben-Wechsel-Szenario.

### □ P2 Evolutionaere Input-Gruppierung (Evolvable Input Aggregation Layer)

> **Experimentell (Layer 3):** Nur in `experimental/` oder Research Branch implementieren, nicht direkt in den Stable Core mischen.

NEAT behandelt jeden Input als unabhaengigen Knoten; bei hochdimensionalen Eingaben (Sensoren, Pixel-Gruppen) entstehen so viele Verbindungen, dass der Suchraum explodiert. Ein evolvierbarer Aggregations-Layer vor dem eigentlichen NEAT-Netz reduziert die Anzahl der effektiven Input-Knoten durch gelerntes Zusammenfassen von Inputs — ohne dass der Nutzer vorher wissen muss welche Inputs zusammengehoeren.

Verwandte Konzepte: Capsule Networks (Hinton 2017) gruppieren semantisch aequivalente Features; CoSyNE co-evolviert Eingabe-Gewichte; ES-HyperNEAT bestimmt Knoten-Positionen im Substrat automatisch. Das Neue hier ist die Kombination: evolvierbare Gruppen mit evolvierbaren Aggregations-Operatoren als dynamisch wachsender Pre-Layer, der vollstaendig in NEAT-Mutations- und Crossover-Logik integriert ist.

**Aktueller Stand:** Kein `InputGrouper` in YANE. Verwandte Bausteine vorhanden: `set_input_transform()` (P1-Task, manuell vorgegeben), HyperNEAT/CPPN (erzeugt raeumliche Gewichtsmuster fuer Hidden-Substrate, nicht fuer Input-Aggregation), `warm_start_from_checkpoint()` (kann Eingabe-Anzahl anpassen). Die dynamisch veraenderliche Input-Knoten-Anzahl ist der technisch schwierigste Teil — sie beruehrt Innovation-Tracking, Crossover-Alignment und Checkpoint-Kompatibilitaet.

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

### □ P2 Evolutionaere Output-Gruppierung (Evolvable Output Synergy Layer)

> **Experimentell (Layer 3):** Nur in `experimental/` oder Research Branch implementieren, nicht direkt in den Stable Core mischen.

Symmetrisch zur Input-Gruppierung — aber die externe Schnittstelle bleibt unveraendert. Von aussen sieht das Genome weiterhin N Ausgabe-Kanaele; intern werden Outputs die immer gemeinsam aktiviert werden unter einem geteilten Proto-Output-Knoten zusammengefasst. Das NEAT-Netz evolviert nur noch K < N Proto-Knoten, der `OutputGrouper` expandiert diese transparent zurueck auf die N extern erwarteten Ausgabe-Werte. Weder die Fitness-Funktion noch der Nutzer-Code muss geaendert werden.

Beispiel: Aktion 3 und Aktion 7 werden in einer Aufgabe stets gleichzeitig ausgefuehrt. Ohne Gruppierung evolviert NEAT zwei separate Output-Nodes mit eigenen Verbindungsbaumen — doppelter Suchaufwand. Mit Gruppierung teilen sich beide einen Proto-Knoten; das Netz lernt die Synergy einmal, der Grouper verteilt sie auf beide Slots.

Anwendungsfaelle: hochdimensionale Steuerung (Roboterarm mit 20 Gelenken, von denen 4 Gruppen immer synchron feuern), multi-label Klassifikation mit korrelierten Klassen, jede Aufgabe mit korrelierten Ausgabe-Kanaelen.

**Aktueller Stand:** Kein `OutputGrouper` in YANE. `genome.output_nodes` entspricht 1:1 den externen Ausgabe-Slots; `genome_to_python()` liest sie direkt. Kein Mechanismus fuer geteilte Proto-Knoten. Technische Herausforderungen sind identisch mit der Input-Gruppierung: dynamische interne Output-Knoten-Anzahl, Crossover-Alignment, Checkpoint-Kompatibilitaet.

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

### □ P2 Shared Weights (Weight-Sharing zwischen Verbindungen)

Klassische NEAT-Verbindungen haben je ein unabhaengiges Gewicht; fuer CNN-artige Strukturen muessen raeumlich aequivalente Verbindungen dasselbe Gewicht teilen. Ohne Weight-Sharing muss NEAT jedes Pixel-zu-Pixel-Gewicht einzeln evolvieren.

**Aktueller Stand:** Jede `Connection` hat ein eigenes `weight`-Attribut. Kein Konzept von Gewichts-Gruppen; Lamarck und Mutation behandeln jede Verbindung unabhaengig.

- `Connection.weight_group: str | None`: optionale Gruppen-ID; alle Verbindungen einer Gruppe teilen denselben skalaren Gewichtswert.
- `genome.weight_groups: dict[str, float]`: zentrales Register der Gruppen-Gewichte; `connection.weight` gibt den aktuellen Gruppen-Wert zurueck.
- Mutation und Lamarck operieren auf Gruppen (nicht auf einzelnen Verbindungen); ein Gruppen-Gewicht-Update gilt fuer alle N Verbindungen gleichzeitig.
- Gruppen sind evolvierbar: Verbindungen koennen ihre Gruppe wechseln oder eine neue Gruppe erzeugen.
- Checkpoint-kompatibel: `weight_groups`-Dict wird gespeichert.
- Benchmark: Shared-Weights vs. unabhaengige Gewichte auf MNIST (784 Inputs → 10 Outputs, flache Topologie).

### □ P2 Convolutional NEAT (CoDeepNEAT-inspiriert)

> **Experimentell (Layer 3):** Nur in `experimental/` oder Research Branch implementieren, nicht direkt in den Stable Core mischen.

NEAT sucht verbindungsweise; fuer Bildverarbeitung ist die sinnvolle Sucheinheit ein Conv-Block (Filter, Stride, Channels), nicht eine einzelne Gewichtsverbindung.

**Aktueller Stand:** `add_node` und `add_connection` als atomare Mutations-Operatoren. HyperNEAT (ueber CPPNs) kann raeumliche Gewichtsmuster erzeugen, aber keine echten Faltungsoperationen mit geteilten Gewichten und Stride.

- Neuer Knotentyp `CONV2D` mit evolvierbaren Parametern: `kernel_size`, `stride`, `out_channels`, `activation`.
- Weight-Sharing automatisch: alle raeumlichen Positionen eines Filters teilen dasselbe Gewicht (baut auf Shared-Weights-Task auf).
- Mutations-Operator `add_conv_block`: fuegt einen vollstaendigen Conv-Block (CONV2D + optionaler MaxPool-Knoten) als Einheit hinzu.
- `genome.forward_image(pixels, height, width, channels)`: korrekter Faltungs-Forward-Pass.
- `genome_to_python()` unterstuetzt CONV2D-Knoten (erzeugt `for`-Schleifen statt einzelner Verbindungen).
- Benchmark: Convolutional NEAT vs. HyperNEAT vs. flaches NEAT auf MNIST (Accuracy nach fester Evaluierungs-Anzahl).

### □ P2 ES-HyperNEAT (Evolvable Substrate HyperNEAT)

> **Experimentell (Layer 3):** Nur in `experimental/` oder Research Branch implementieren, nicht direkt in den Stable Core mischen.

Das aktuelle HyperNEAT-Substrat wird vom Nutzer manuell als Gitter-Koordinaten definiert; die CPPN weiss nicht wo sinnvolle Knoten-Positionen im geometrischen Raum liegen.

**Aktueller Stand:** `hyperneat_substrate(n_inputs, n_hidden, n_outputs)` erzeugt ein festes Schicht-Gitter. `generate_genome_from_cppn()` verwendet dieses Substrat. Die CPPN kann die Substrat-Struktur nicht beeinflussen.

- ES-HyperNEAT: CPPN-Output-Varianz bestimmt automatisch ob an einer Koordinate ein Knoten sinnvoll ist (hohe lokale Varianz → Knoten platzieren).
- `hyperneat_substrate(evolve=True)`: Substrat-Koordinaten werden nicht vorgegeben sondern per Quadtree aus der CPPN-Aktivierungslandschaft abgeleitet.
- Geometrische Biase bleiben erhalten: Eingabe-Knoten unten, Ausgabe-Knoten oben; CPPN kodiert raeumliche Beziehungen.
- `generate_genome_from_cppn(cppn, substrate, evolve_substrate=True)`: erweiterte Signatur.
- Benchmark: ES-HyperNEAT vs. festes Substrat auf einem 2D-Navigations-Task (Maze) und MNIST.
