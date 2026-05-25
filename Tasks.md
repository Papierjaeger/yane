# Tasks: YANE staerker machen

Diese Datei ist die aktuelle Roadmap fuer YANE. Offene und neue Tasks stehen
oben. Abgeschlossene Arbeit ist weiter unten nur noch kompakt zusammengefasst.

## Status

**Aktueller Stand:** Alle P0-Tasks abgeschlossen. Teststand: `813 passed`.

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
- Naechster Schwerpunkt: P1-Tasks (Landscape-Viz, Selektionsstrategie, Generationsreport, Eval-Middleware).

## Legende

- `P0`: hoher Hebel, nah am aktuellen Code, direkt nuetzlich
- `P1`: wichtiger Ausbau, mittlerer Aufwand
- `P2`: experimentell, Forschungsarbeit oder groesserer Umbau
- ✅: erledigt
- ⚡: teilweise erledigt
- 🔲: offen

---

## Offene Tasks

---

### ✅ P0 Event-System / Observer-Muster

Die Evolution hat kein entkoppeltes Benachrichtigungssystem; GUI, Logging und externe Integrationen greifen direkt auf interne Zustände zu.

- `yane.on(event, fn)` und `yane.off(event, fn)`: Callbacks fuer `"generation_end"`, `"new_best"`, `"species_created"`, `"species_extinct"`, `"stagnation"`, `"run_end"`.
- Callback-Parameter: vollstaendiges Diagnostics-Dict, Generations-Index, bestes Genom.
- Interne Nutzung: GUI-Worker, CSV-Logger und Adaptive Controller werden auf dieses System umgestellt.
- `yane.emit(event, payload)`: oeffentliche API fuer benutzerdefinierte Events aus Evaluatoren heraus.
- Schwacher Referenz-Mechanismus (`weakref`) damit registrierte GUI-Objekte keine Memory-Leaks verursachen.

### ✅ P0 Online-Anomalie-Detektion

Problematische Trainingslaeufe (Fitness-Einbruch, Diversitaets-Kollaps) werden erst im Nachhinein erkannt.

- Detektoren als leichtgewichtige Klassen mit `check(diagnostics) -> AnomalyReport | None`:
  - `FitnessCollapseDetector`: Best-Fitness faellt mehr als X% in N Generationen.
  - `DiversityCollapseDetector`: Durchschnittliche genetische Distanz unterschreitet Schwelle.
  - `HomogenizationDetector`: Anteil identischer Topologien ueberschreitet Schwelle.
  - `StuckSpeciationDetector`: Species-Anzahl konstant 1 fuer N Generationen ohne Verbesserung.
- `yane.set_anomaly_detectors([...])` oder `add_anomaly_detector(detector)`.
- GUI: farbiger Warnhinweis in der Statuszeile; Anomalie-Log im Diagnostics-Tab.
- Diagnostics: `anomalies_detected` (Zaehler) und `last_anomaly` (Event + Generation) in `population_memory_info()`.

### ✅ P0 Fitness-Transformer-API

Fitness-Shaping (Ranking, Clipping, Normierung) ist ueber mehrere Stellen im Code verstreut und nicht benutzerdefiniert austauschbar.

- `yane.set_fitness_transform(fn)`: Callable `(raw_fitnesses: list[float]) -> list[float]` wird nach jeder Evaluationsrunde angewendet.
- Eingebaute Transformer: `RankTransform`, `SigmaScaling`, `LinearNormalize(lo, hi)`, `ClipTransform(min, max)`.
- Transformer-Kette: `yane.set_fitness_transform(ChainTransform([RankTransform(), SigmaScaling()]))`.
- Diagnostics: Transformer-Name und Input/Output-Range per Generation.
- Bestehende Fitness-Sanitizer und Normierungs-Code wird auf diese API umgestellt.

---

### 🔲 P1 Fitness-Landscape-Visualisierung (PCA / t-SNE)

Konvergenzverhalten und Populationsstruktur sind nur ueber Zahlen erkennbar; die geometrische Verteilung im Genotyp-Raum ist unsichtbar.

- Genome einer Generation als Feature-Vektoren kodieren (Gewichtsvektor + Verbindungsmaske).
- PCA (2 Komponenten, ohne externe Deps) und optionales t-SNE (via `scikit-learn` falls installiert) zur Projektion.
- GUI: interaktive Scatter-Punktwolke im Diagnostics-Tab; Farbe = Fitness, Form = Species, Groesse = Komplexitaet.
- Hover-Tooltip mit Genome-ID, Species, Fitness, Nodes/Connections.
- Export: Snapshot als PNG; alle Projektionspunkte als CSV.
- Benchmark: Visualisierung auf XOR-Lauf zeigt Species-Cluster klar getrennt.

### 🔲 P1 Selektionsstrategie als Plugin

Die Selektionsstrategie (Tournament) ist fest in `population.py` verdrahtet; kein Austausch ohne Code-Aenderung.

- `SelectionStrategy`-Protokoll: `select(population, n) -> list[Genome]`.
- Eingebaute Strategien: `TournamentSelection(k=3)`, `ElitistSelection(top_frac=0.2)`, `FitnessProportional()`, `NoveltyOnlySelection()`, `RankSelection()`.
- `NeuroEvolution.set_selection_strategy(strategy)`.
- Per-Species-Ueberschreibung: `set_selection_strategy(strategy, species_id=...)`.
- Diagnostics: aktive Strategie, Durchschnittsfitness der selektierten vs. nicht-selektierten Genome.
- Tests: jede Strategie gibt korrekte Anzahl Genome zurueck; Fitness-Proportional ist stochastisch korrekt.

### 🔲 P1 Generationsreport / Run-Postmortem

Nach einem Trainingslauf gibt es keine strukturierte Zusammenfassung; Erkenntnisse muessen manuell aus Logs und CSV herausgezogen werden.

- `yane.export_run_report(path, format="html"|"json"|"md")`: generiert Bericht nach `run_end`.
- Inhalte: Fitness-Kurve (SVG-Inline fuer HTML), beste Genome (Topologie + Score), Mutations-Attribution, Anomalie-Log, Konfiguration, Laufzeit.
- HTML-Report ist self-contained (kein externer CSS/JS); JSON-Report ist maschinenlesbar.
- GUI: „Report exportieren"-Button im Diagnostics-Tab; oeffnet gespeicherte Datei im Browser.
- `yane.set_report_autosave(path_template="{date}_{example}_report.html")`.

### 🔲 P1 Evaluation-Middleware-Stack

Evaluatoren sind einfache Callables ohne Kompositionsmechanismus; Caching, Normierung und Noise-Injection muessen pro Evaluator manuell implementiert werden.

- `EvalMiddleware`-Protokoll: `__call__(genome, eval_fn, ctx) -> float`.
- Eingebaute Middleware: `CachingMiddleware(maxsize=512)` (Genome-Hash → Fitness), `NoiseMiddleware(sigma=0.05)` (Input-Perturbation), `TimingMiddleware` (Eval-Zeit pro Genom), `RetryMiddleware(n=3, aggregation="mean")`.
- `yane.add_eval_middleware(mw)` haengt in die Kette ein.
- Middleware-Reihenfolge: LIFO (zuletzt hinzugefuegt = aeussere Schicht).
- Diagnostics: Cache-Hit-Rate, Durchschnittliche Eval-Zeit, Retry-Rate.

### 🔲 P1 Populations-Filter und -Aggregatoren API

Analyse von Populationszustaenden erfordert direkten Zugriff auf interne Listen; keine saubere funktionale API.

- `population.filter(fn: Genome -> bool) -> list[Genome]`: selektiert Genome nach Praedikat.
- `population.map(fn: Genome -> T) -> list[T]`: transformiert Genome zu beliebigen Werten.
- `population.reduce(fn: (acc, Genome) -> acc, init) -> acc`: Fold ueber Population.
- `population.group_by(fn: Genome -> K) -> dict[K, list[Genome]]`: Gruppierung (z. B. nach Species oder Aktivierungstyp).
- `population.top_k(k, key=lambda g: g.fitness) -> list[Genome]`.
- Alle Methoden docstring-frei, aber mit Typ-Annotationen; kein Overhead wenn nicht verwendet.
- Tests: Filter, Map, Group-by auf kleiner Test-Population; Top-k-Reihenfolge korrekt.

### 🔲 P1 Connection-Weight-Histogramm und Gewichtsgesundheit

Pathologien wie Vanishing/Exploding Weights oder symmetrische Gewichtsverteilungen sind in den Diagnostics unsichtbar.

- Pro Generation: Gewichtsverteilung der gesamten Population als Histogramm (20 Bins, -5 bis +5).
- Kennzahlen: Mean, Std, 5./95.-Perzentil, Anteil toter Gewichte (|w| < 0.01), Anteil saturierter Gewichte (|w| > 4.9).
- GUI: Histogramm-Widget im Diagnostics-Tab (aktualisiert sich waehrend Training).
- Warnung wenn Std < 0.05 (Kollaps) oder > 3.0 (Explosion) fuer N aufeinanderfolgende Generationen.
- Export: Gewichtsmatrix des besten Genoms als `.npy` (NumPy-Format) fuer externe Analyse.

---

### 🔲 P2 Genome-Codec-Protokoll (austauschbare Serialisierung)

Das Checkpoint-Format ist fest auf Pickle festgelegt; alternative Formate (JSON, MessagePack, komprimiertes Binaerformat) sind nicht einsteckbar.

- `GenomeCodec`-Protokoll: `encode(genome) -> bytes`, `decode(bytes) -> Genome`.
- Eingebaute Codecs: `PickleCodec` (Standard, bestehend), `JsonCodec` (menschenlesbar, nur einfache Genome), `MsgpackCodec` (kompakt, schnell).
- `yane.set_checkpoint_codec(codec)`.
- Checkpoint-Datei enthaelt Codec-Kennung im Header; Ladelogik waehlt automatisch den richtigen Codec.
- Migration: `yane.migrate_checkpoint(path, target_codec)` konvertiert bestehende Pickles.
- Tests: Round-trip encode → decode fuer alle Codecs auf Standard-Genomen.

### 🔲 P2 Konfigurationsversionierung und Kompatibilitaets-Check

Checkpoints enthalten die Population, aber nicht den vollstaendigen Zustand der Konfiguration; spaeters Nachladen kann zu stillem Fehlverhalten fuehren.

- Jede `ExperimentPreset` erhaelt einen deterministischen Konfigurations-Hash (SHA-256 ueber kanonisches JSON).
- Beim `load_checkpoint()`: Hash wird mit dem gespeicherten verglichen; bei Abweichung erscheint strukturierter Diff der geaenderten Felder.
- `CompatibilityLevel`: `EXACT` (identisch), `COMPATIBLE` (nur unkritische Felder geaendert), `BREAKING` (Inputs/Outputs/Topologie-Constraints geaendert).
- GUI: Warn-Dialog bei `BREAKING`; Checkpoint-Metadaten zeigen Konfig-Hash und Aenderungs-Diff.
- CLI: `python -m yane.checkpoint --diff old.pkl new.pkl` zeigt Konfigurations-Unterschiede.

### 🔲 P2 Differenzierbare Topologie-Suche (DARTS-Lite)

NEAT sucht Topologien diskret durch Mutation; differenzierbare Relaxation erlaubt gradienten-basierte Architektursuche.

- Kontinuierliche Relaxation: Kanten-Gewichte mit Gating-Sigmoid; niedrige Gates werden nach N Schritten geprunt.
- `DARTSOptimizer`: wechselt ab zwischen Lamarck-Schritt (Gewichte) und Architektur-Gradient (Gates).
- Nur fuer Feed-Forward-Genome ohne Memory-Nodes.
- `NeuroEvolution.set_darts_mode(enabled=True, prune_threshold=0.1)`.
- Benchmark: DARTS-Lite vs. Standard-NEAT auf Symbolic-Regression-Task.

### 🔲 P2 Intrinsische Belohnung / Curiosity-Modul

Sparse-Reward-Umgebungen (z. B. Maze) liefern keine nutzbare Fitness-Information; Curiosity kann die Exploration antreiben.

- `CuriosityModule`: haelt ein kleines Vorhersage-Netz (auch ein YANE-Genom), das naechste Observations vorhersagt.
- Intrinsischer Reward = Vorhersagefehler; wird zur externen Fitness addiert (gewichtet, konfigurierbar).
- Das Vorhersage-Netz wird per Lamarck parallel zur Haupt-Population trainiert.
- `yane.set_curiosity(enabled=True, weight=0.3, network_size=8)`.
- Benchmark: Curiosity vs. kein Curiosity auf einem Sparse-Reward-Maze (eigene minimale Implementierung).

---

### ✅ P0 Genome-Export (Python-Funktion / to_numpy_weights)

Trainierte Genome sollten ohne YANE-Abhaengigkeit einsetzbar sein.

- `genome.to_python()` → standalone Python-Funktion (nur `math`, keine YANE-Importe); direkt in andere Projekte kopierbar.
- `genome.to_onnx()` → ONNX-Modell fuer Integration mit PyTorch, TensorFlow, Triton, Edge-Devices.
- GUI-Schaltflaeche „Export" im Inspect-Tab (Python-Datei oder `.onnx` speichern).
- `genome.to_numpy_weights()` → kompakte Weight-Matrix fuer einfaches Deployment.
- Regressionstest: exportierte Funktion gibt fuer alle Test-Cases dieselben Werte wie `genome.forward()`.

### ✅ P0 Gym-Inspect verbessern (Aktion + Reward anzeigen)

Inspect zeigt bei Gym-Beispielen rohe Netzwerkausgaben, nicht die interpretierte Aktion.

- Im Manual-Forward-Pass: berechnete Aktion als separates Feld anzeigen (z. B. „Aktion: 2 (rechts)" fuer diskrete Umgebungen, skalierter Wert fuer kontinuierliche).
- Optional: Einzelschritt in der Umgebung ausfuehren und Reward zurueckgeben.
- Render-Frame im Inspect anzeigen, wenn die Umgebung `render_mode="rgb_array"` unterstuetzt.
- `ExampleConfig`: optionales `action_display_fn` fuer benutzerdefinierte Aktionsdarstellung.

### ✅ P0 Validierungs-Set-Unterstuetzung

Overfitting ist auf Datensatz-Beispielen nicht erkennbar, weil Training- und Test-Daten identisch sind.

- `yane.set_validation_fn(fn)`: separates Evaluations-Callable, das nach jeder Generation auf dem besten Genom laeuft.
- Validierungs-Fitness wird in Diagnostics und CSV-Log separat protokolliert (nicht fuer Selektion genutzt).
- GUI: Validierungs-Fitness als zweite Linie im Fitness-Chart; Warnung wenn Val-Fitness deutlich schlechter als Train-Fitness.
- Fuer Regression-/Dataset-Beispiele: Train/Val-Split automatisch aus den Test-Cases generieren.

### ✅ P0 Vollstaendige Konfigurationspersistenz

Presets speichern nur einen Teil der Einstellungen; viele GUI-Felder (Pop-Groesse, Target, Memory-Checkbox usw.) werden nicht persistiert.

- „Konfiguration speichern / laden" als JSON-Datei: alle Training-Tab-Werte, alle Adaptive-Widgets, alle Forschungs-Checkboxen.
- Letzte Konfiguration beim Programmstart automatisch wiederherstellen (opt-in, `settings.json`).
- Preset-System um `target_fitness` und Populationsgroesse erweitern.
- CLI-Modus: `python -m yane.gui --config run.json` startet Training direkt ohne GUI-Interaktion.

---

### 🔲 P1 Erweiterbare Aktivierungsfunktionen

Das Aktivierungsset (sigmoid, tanh, relu, leaky_relu, swish, linear) ist im Code fest verdrahtet.

- `NeuroEvolution.register_activation(name, fn, backprop_fn=None)` registriert benutzerdefinierte Aktivierungen zur Laufzeit.
- Registrierte Funktionen werden beim Checkpoint mit gespeichert (Name → Lambda/Pickle).
- Vorschlaege fuer eingebaute Erweiterungen: ELU, GELU, Gaussian, Mish, SiLU.
- GUI-Checkbox-Liste zeigt verfuegbare Aktivierungen; Nutzer kann erlaubte Typen einschraenken.

### 🔲 P1 Multi-Population Inselmodell

Einzelne Population konvergiert in lokale Optima; unabhaengige Inseln verbessern die Exploration.

- `NeuroEvolution.set_island_model(n_islands, migration_rate, migration_interval)`.
- Jede Insel laeuft als eigene `Population`-Instanz; periodisch werden die besten N Genome zwischen zunaechst zufaelligen, dann topologie-nah gewaehlten Inseln migriert.
- Diagnostics: Fitness pro Insel, Migrations-Events, Diversitaets-Abstand zwischen Inseln.
- Benchmark: Vergleich Einzel-Population vs. Inselmodell auf XOR, CartPole, Multiplication.

### 🔲 P1 Hyperparameter-Suche

Nutzer muessen Parameter (Pop-Groesse, Target-Species, Lamarck-Steps usw.) manuell ausprobieren.

- `yane.hyperparameter_search(param_grid, n_seeds, fitness_fn, budget_iterations)`: laeuft N Konfigurationen parallel und gibt Ranking zurueck.
- Strategien: Grid-Search, Random-Search, einfaches Bayes-Opt (via `scikit-optimize`).
- GUI: eigener „Suche"-Tab mit Parameterraum-Editor und Ergebnis-Tabelle.
- Ergebnisse als CSV exportieren; beste Konfiguration direkt in Trainings-Tab uebertragen.

### 🔲 P1 Ensemble-Bewertung und -Deployment

Ein einzelnes Genom ist stochastisch; ein Ensemble aus den Top-K ist robuster.

- `yane.get_ensemble(k=5)` → `EnsembleGenome`-Wrapper mit `forward(inputs)`.
- Strategien: Mittelwert der Outputs, Mehrheitsvoting (diskret), gewichtete Kombination nach Fitness.
- `EnsembleGenome.to_python()` und `to_onnx()` exportieren alle K Genome gemeinsam.
- GUI: Ensemble-Groesse im Inspect-Tab einstellen, Ensemble-Output als zusaetzliche Spalte anzeigen.

### 🔲 P1 Strukturierte / maschinenlesbare Protokollierung

Das CSV-Log ist beschraenkt; kein strukturiertes Format fuer externe Tools.

- Option `yane.set_log_format("jsonlines")`: jede Generation als JSON-Objekt mit allen Diagnostics-Keys in eine `.jsonl`-Datei.
- Integration mit TensorBoard: `yane.set_tensorboard_logdir(path)` schreibt Skalare via `torch.utils.tensorboard` (optional).
- `yane.set_log_callbacks(on_generation=fn)`: benutzerdefinierter Callback mit dem vollstaendigen Diagnostics-Dict.
- Fitness-History-CSV wird um Validierungs-Fitness und Ensemble-Fitness ergaenzt.

### 🔲 P1 Erweiterte Genome-Analyse im Inspect (Sensitivitaet / Attribution)

Nutzer sehen Ausgaben, aber nicht WARUM das Genom so entscheidet.

- Sensitivity-Analyse: fuer jeden Input-Kanal den Output bei +0.1 / -0.1 Perturbation messen → Einfluss-Score je Input.
- Inspect-Tab: Balken-Diagramm mit Input-Relevanz, farblich nach Einflussrichtung.
- „Toter Knoten"-Marker: Knoten/Verbindungen, die bei allen Test-Cases nie feuern, visuell hervorheben.
- Vergleich: gleiche Analyse fuer Genome am Anfang und am Ende des Trainings.

### 🔲 P1 Plugin-System fuer benutzerdefinierte Evaluatoren

Eigene Umgebungen einzubinden erfordert bisher direktes Editieren der `examples.py`.

- `ExamplePlugin`-Protokoll: Python-Klasse mit `name`, `make_eval`, `n_inputs`, `n_outputs`, `target_fitness`.
- `yane.register_example(plugin)` registriert den Evaluator in der GUI-Liste.
- Plugins werden aus einem konfigurierbaren Plugin-Verzeichnis automatisch geladen (`~/.yane/plugins/`).
- Dokumentation + Beispiel-Plugin-Template als Quickstart.

### 🔲 P1 Lernkurven-Vergleich (mehrere Runs)

Einzelne Runs sind nicht repraesentativ; Vergleich verschiedener Konfigurationen ist nicht moeglich.

- `Run`-Objekte speichern Fitness-History, Konfiguration und Endresultat.
- GUI: „Vergleich"-Ansicht mit ueberlagerten Fitness-Kurven (bis zu 4 Runs, farbcodiert).
- Statistische Zusammenfassung: Median, 25./75.-Perzentil ueber N Wiederholungen derselben Konfig.
- Export: Vergleichs-Plot als PNG, Rohdaten als CSV.

---

### 🔲 P2 Synaptische Plastizitaet (STDP / Hebbsches Lernen)

Genome lernen derzeit nur durch Evolution, nicht durch Erfahrung innerhalb einer Episode.

- Knoten/Verbindungen koennen evolvierte Hebb-Regel-Koeffizienten (A, B, C, D) tragen.
- Gewichte werden waehrend `genome.forward()` nach der STDP-Regel angepasst (intra-lifetime-learning).
- `genome.reset()` setzt Gewichte auf Basiswerte zurueck (Plastizitaet ist episoden-lokal).
- Benchmark: STDP vs. Lamarck auf Aufgaben mit veraenderlicher Umgebung (z. B. wechselnde XOR-Eingaenge).

### 🔲 P2 Neuromodulation

Modulatorische Signale erlauben kontextabhaengige Gewichtung ganzer Verbindungsgruppen.

- Sonderknotentyp `Modulator`: sein Ausgabe-Wert skaliert eingehende Verbindungen anderer Knoten.
- Evolvierbar: welcher Knoten moduliert, welche Verbindungen beeinflusst werden, Staerke.
- Anwendung: schnelle Anpassung an wechselnde Aufgaben (Multi-Task-Szenarien).
- Benchmark: Modulation vs. kein Modulation auf einem Aufgaben-Wechsel-Szenario.

### 🔲 P2 Transfer Learning / Genome Fine-Tuning

Wissen aus einem trainierten Genom soll auf eine neue Aufgabe uebertragen werden.

- `yane.load_genome_as_seed(genome, freeze_layers=[...])`: bestimmte Verbindungsgruppen koennen eingefroren werden.
- Lamarck-Feinabstimmung auf neuer Aufgabe ohne Topologie-Aenderung als erste Phase.
- Dann schrittweise Entsperren eingeforener Teile (progressive unfreeze).
- Benchmark: Transfer CartPole → LunarLander vs. Training from scratch.

### 🔲 P2 Offene Evolution / Co-Evolution von Aufgabe und Agent (POET-aehnlich)

Statische Aufgaben fuehren zu Ueberanpassung; co-evolving Environments erzeugen robustere Agenten.

- `EnvironmentGenome`: evolvierbare Umgebungsparameter (z. B. Hangneigung, Hindernisanzahl).
- Paarung: Agent-Genome werden auf ihren aktuellen Environment-Genome evaluiert.
- Archiv-Mechanismus: nur Agenten-Genome, die auf mindestens einem Environment gut abschneiden, ueberleben.
- Benchmark: Co-Evolution vs. Domain-Randomization auf BipedalWalker mit variabler Bodenbeschaffenheit.

### 🔲 P2 YANE → PyTorch-Bruecke (NAS + Feinabstimmung)

Evoluted Architekturen koennen nicht mit Gradienten-Methoden weiter optimiert werden.

- `genome.to_torch_module()` → `torch.nn.Module` mit exakt derselben Topologie.
- Gewichte werden uebertragen; danach normales PyTorch-Training moeglich.
- Nutzung: YANE findet gute Architektur (NAS-Rolle), PyTorch optimiert Gewichte weiter.
- Export beruecksichtigt Memory-Knoten als `nn.GRUCell`-Aequivalent.

### 🔲 P2 Genome-Phylogenie (Stammbaum der Innovationen)

Welche Mutation hat den entscheidenden Durchbruch geliefert? Das ist derzeit nicht nachverfolgbar.

- `InnovationTracker` protokolliert fuer jede Innovation: Eltern-Genome-ID, Generation, Delta-Fitness.
- `genome.lineage()` gibt Vorfahren-Kette bis zum Urgenome zurueck.
- GUI: Stammbaum-Visualisierung (kollabierbar) mit markierten Schluessel-Mutationen.
- Analyse: welche Mutations-Operatoren fuehren am haeufigsten zu Fitness-Spruengen.

### 🔲 P2 Verhaltensklonierung als Warm-Start

Evolution braucht viele Iterationen bis zu brauchbaren Loesungen; Demonstrationen koennen das beschleunigen.

- `yane.behaviour_clone(demonstrations, n_steps)`: supervised Vortraining des besten Genoms auf Demonstrations-Daten via Lamarck/Backprop.
- Demonstrationen als Liste von `(inputs, outputs)`-Paaren; kein RL-Umgebungsformat noetig.
- Geklontes Genom wird als initiales Seed fuer die Population verwendet.
- Benchmark: BC-Warm-Start vs. random-init auf LunarLander.

---

## Abgeschlossen

### ✅ P2 Modul-Crossover und Modulbibliothek

- `ModuleBlueprint` und `ModuleLibrary` speichern wiederverwendbare Hidden-Module.
- `module_crossover()` kopiert kompatible Donor-Subgraphen in Recipient-Genome.
- Optionaler Population-Mutationsoperator fuegt Module aus der Bibliothek ein.
- `NeuroEvolution.set_module_library()` aktiviert Bibliothek und Insert-Rate.
- Diagnostics melden Modulanzahl, Inserts, Uses und Wiederverwendungsrate.
- Tests fuer Modul-Crossover, Bibliothek, Reinsert und Diagnostics.

### ✅ P2 Evolvierbare CPPNs

- `CPPNGenome` modelliert CPPNs als normale kleine YANE-Genome.
- CPPN-Outputs erzeugen sparse Weight-Patterns ueber `generate_weight_pattern()`.
- `hyperneat_substrate()` definiert Input/Hidden/Output-Substrate.
- `generate_genome_from_cppn()` decodiert CPPN-Muster zu normalen YANE-Genomen.
- `benchmarks/cppn_indirect_ablation.py` vergleicht direkte Kodierung mit CPPN-Substrat.
- Tests fuer CPPN-Muster, Substrat-Genome und indirekte Genome-Generierung.

### ✅ P2 Meta-adaptive Policies evolvieren

- `PolicyGenes` fuer Operator-Exploration, Lamarck-Budget und Interspecies-Rate.
- `MetaAdaptivePolicyEvolver` vergleicht globale und per-Species-Scores und mutiert Policy-Gene.
- Sicherheitsgrenzen ueber `PolicyGeneBounds`, Clamping und begrenzte Operator-Gewichte.
- `NeuroEvolution.set_meta_adaptive_policies()` aktiviert die Schicht und Diagnostics.
- `benchmarks/meta_policy_ablation.py` vergleicht feste Policy, handadaptive Policy und evolvierte Policy.
- Tests fuer API, Diagnostics, per-Species-Gene und Grenzwert-Clamping.

### ✅ P2 Evolvierbare Descriptor-Gewichte

- `AdaptiveFitnessComponentWeights` mit Gewichtshistorie, Diagnostics und Collapse-Floor.
- `NeuroEvolution.set_fitness_components()` addiert optionale gewichtete Komponenten zur Task-Fitness.
- Adaptive Updates laufen generationenweise bei Stagnation und nutzen Populationsvarianz/Kontributionsanteile.
- `benchmarks/descriptor_weight_ablation.py` vergleicht Task-only, feste und adaptive Descriptor-Kombinationen.
- Tests fuer Scalarization, Fitness-Shaping, Historie, Stagnationsupdate und Collapse-Schutz.

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

### ✅ P1 Checkpoint-Format langfristig haerten

- `tests/fixtures/checkpoint_v1.pkl` und `checkpoint_v2.pkl` als versionierte Regressionsfixtures.
- `TestCheckpointFixtures` in `test_checkpoint_migration.py`: laedt echte Fixture-Dateien, prueeft Migration und Sidecar.
- GUI: `_show_checkpoint_metadata()` zeigt nach dem Laden Version, Pop-Size, Inputs/Outputs; warnt bei `requires_reattach`.
- API: `GET /checkpoint/metadata?path=...` liest `.json`-Sidecar ohne Pickle zu laden; Fallback auf Live-Ableitung.
- `evolution/checkpoint.py`: Inline-Dokumentation erklaert warum Pickle (nicht JSON) verwendet wird + Security-Grenze.
- 3 neue GUI-Smoke-Tests, 3 neue API-Tests.

### ✅ P1 Remote/Distributed Evaluation

- `evolution/remote_evaluation.py`: Protokoll (`EvalJob`, `EvalResult`), `RemoteEvaluationClient` (HTTP, Round-Robin, Retry, Cancel) und `RemoteWorkerServer` (FastAPI, Auth-Token, Timeout).
- Security-Modell: Token-Pflicht dokumentiert; Pickle nur von authentifizierten Sendern entgegennehmen.
- `RemoteEvaluationClient` und `RemoteWorkerServer` in `__all__` exportiert.
- `benchmarks/remote_eval_bench.py`: vergleicht Sequential, Thread-Pool, Process-Pool und Remote-HTTP.
- 16 neue Tests in `test_remote_evaluation.py`.
