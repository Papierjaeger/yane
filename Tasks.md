# Tasks: YANE stärker machen

Diese Datei sammelt Ideen, Aufgaben und Forschungsrichtungen, um YANE allgemein leistungsfähiger zu machen. Es geht hier bewusst nicht um Spezialtuning einzelner Beispiele, sondern um Verbesserungen am Framework, damit größere, schwierigere und längere Aufgaben besser bewältigt werden können.

## Legende

- `P0`: hoher Hebel, nah am aktuellen Code, wahrscheinlich direkt nützlich
- `P1`: wichtiger Ausbau, mittlerer Aufwand
- `P2`: experimentell oder größerer Umbau
- ✅ = vollständig implementiert → Details stehen unter [Erledigte Aufgaben](#erledigte-aufgaben)
- ⚡ = teilweise implementiert → Rest-Aufgaben sind direkt unter dem jeweiligen Task beschrieben

---

## 1. Fitness, Evaluation und Training

### P1: Curriculum Learning

Schwere Aufgaben können in Stufen gelernt werden.

Aufgaben:

- Curriculum-Interface definieren.
- Fitnessfunktion kann aktuelle Stufe melden.
- Automatischer Stufenwechsel bei Ziel-Fitness.
- Population beim Stufenwechsel behalten.

Nutzen:

- Komplexe Aufgaben werden schrittweise lernbar.
- Besonders nützlich für lange Sequenzen und Control-Aufgaben.

### P1: Fitness-Landscape Diagnostics ✅

Wie verteilt sich Fitness in der Population? Gibt es Sprünge, Plateaus, bimodale Verteilungen?

**Bereits implementiert:** IQR der Fitness in `population_memory_info()`, Fitness-Histogramm (10 Bins) in GUI (`FitnessHistogram`-Widget), Plateau-Ratio (`stagnation_count / stagnation_threshold`) als GUI-Label, Fitness-Sprungrate (`jump_rate = n_new_best / n_submitted`) in `population_memory_info()` und GUI.

**Noch offen:** — (alle geplanten Diagnostics sind jetzt implementiert).

Nutzen:

- Sieht man sofort, ob die Population kollabiert (alle gleich gut), exploriert (breite Verteilung) oder stagniert.
- Hilft beim Tuning von Populationsgröße, Elitismus und Novelty-Gewicht.

### P2: Multi-Objective Optimization

Aktuell wird Fitness als einzelner Zahlenwert optimiert.

Aufgaben:

- Fitness als Vektor erlauben, z. B. Leistung, Geschwindigkeit, Größe, Stabilität.
- Pareto-Selektion implementieren.
- NSGA-II-artige Auswahl prüfen.
- GUI-Darstellung für Pareto-Front.

Nutzen:

- YANE muss nicht alle Ziele in eine fragile Fitnessformel pressen.
- Gute Basis für größere Aufgaben mit Trade-offs.

## 2. Evolution und Suchstrategie

### P1: Adaptive Strukturmutation — Restaufgaben ✅

**Bereits implementiert:** Drei neue Mutationsmechanismen (gewichtsbasierte Remove, Spike-Mutation, Rewiring), alle selbst-adaptiv. Erfolgsrate pro Mutationstyp wird getrackt und in GUI angezeigt („Mutation success"-Gruppe). Mutationstypen werden populationsweit nach historischem Nutzen gewichtet (`_apply_mutation_success_weights`). Per-Species-Mutationstendenzen: jede Species entwickelt eigene Biases (`mutation_biases`), die die Mutationsraten ihrer Genome modulieren (`_apply_species_mutation_biases`).

**Noch offen:** — (alle Restaufgaben sind jetzt implementiert).

Nutzen:

- Weniger blinde Mutation bei komplexen Aufgaben.
- Bessere Diagnose, welche Strukturmutation tatsächlich hilft.

### P1: Speciation robuster machen ✅

Die Speziation ist zentral für NEAT-artige Evolution.

**Bereits implementiert:** Hard-Cap gegen Species-Explosion, inkrementelle Speziation (O(0) Steady-State), Threshold-Adjustment-Diagnostics. Dynamische Ziel-Species: `target_species=0` → auto aus `sqrt(pop_size)`. Alternative Kompatibilitätsmetrik `"topology_no_disabled"` (ignoriert deaktivierte Connections, via `set_speciation_metric`). Kleine Species werden geschützt: Mindestalter (`_min_species_age` = 1 Generation) und Mindestgröße (`_min_species_size` = 2). Species-Historie: `created_at_spawn`, `parent_species_id`, `merge_count` pro Species (in `population_memory_info()` als `species_lineage`).

**Noch offen:** — (alle Restaufgaben sind jetzt implementiert).

Nutzen:

- Mehr strukturelle Vielfalt.
- Weniger lokales Festfahren.

### P1: Reproduktion per Species-Budget

Aktuell entsteht ein Kind per Tournament Selection aus der gesamten evaluierten Population.

Aufgaben:

- Spawn-Anteile pro Species nach Shared Fitness berechnen.
- Schwache/stagnierende Species kontrolliert reduzieren.
- Junge Species Mindestbudget geben.
- Crossover bevorzugt innerhalb derselben Species.

Nutzen:

- Näher an klassischem NEAT.
- Besserer Schutz neuer Strukturinnovationen.

### P2: Quality Diversity / MAP-Elites

YANE hat bereits Novelty Search. Der nächste Schritt wäre Quality Diversity.

Aufgaben:

- Behavior Descriptor API öffnen.
- Archiv nach Verhaltensdimensionen aufbauen.
- Pro Zelle bestes Genom behalten.
- Neue Kandidaten aus verschiedenen Archivbereichen erzeugen.

Nutzen:

- Evolution sammelt viele verschiedene brauchbare Lösungen.
- Sehr nützlich für Aufgaben mit sparse rewards.

### P2: Coevolution

Einige Aufgaben können durch Gegenspieler oder Aufgabenvariation stärker werden.

Aufgaben:

- Interface für Population gegen Population entwerfen.
- Kompetitive Fitness unterstützen.
- Hall-of-Fame-Gegner speichern.
- Rock-Paper-Scissors-Zyklen durch Archiv vermeiden.

Nutzen:

- Starke Methode für Strategien, Spiele und robuste Policies.

### P2: Adaptive Populationsgröße

Die Populationsgröße ist aktuell fix. Dynamische Anpassung könnte Ressourcen effizienter nutzen.

Aufgaben:

- Populationsgröße automatisch reduzieren, wenn Species sich stark konsolidiert hat (wenig Diversität braucht weniger Kandidaten).
- Populationsgröße erhöhen, wenn viele Species gleichzeitig wachsen (Exploration-Phase).
- Mindest- und Maximalgröße als Grenzen definieren.
- Größenveränderung schrittweise (kein sprunghafter Drop).

Nutzen:

- Weniger Evaluierungen in konvergierten Phasen, mehr Vielfalt in Explorationsphasen.
- Bessere Ressourcennutzung bei langen Runs.

### P2: Warm-Start und Transfer Learning

Trainiertes Bestes Genome oder ganze Population als Startpunkt für verwandte Aufgaben nutzen.

Aufgaben:

- `load_population(checkpoint)` als Startpunkt statt zufälliger Initialisierung.
- Filterung: nur Genome behalten, die auf neuer Aufgabe mindestens Baseline erreichen.
- Optionale Re-Initialisierung der Strategie-Gene (sigma, rates) bei Start.
- Beispiel: CartPole-trainiertes Netz als Startpunkt für Acrobot.

Nutzen:

- Verkürzte Konvergenzzeit bei verwandten Aufgaben.
- Sinnvoll für Curriculum Learning (automatischer Wechsel).

## 3. Netzwerkmodell

### P0: Netzwerkgröße kontrollierter wachsen lassen

Größere Aufgaben brauchen größere Netze, aber unkontrolliertes Wachstum wird teuer.

Aufgaben:

- Weiche Komplexitätsstrafe optional einführen.
- Inaktive Nodes/Connections erkennen.
- Pruning-Mutation gezielter auf wirkungslose Struktur anwenden.
- Netzkomplexität in Fitnessstatistik anzeigen.

Nutzen:

- Bessere Balance zwischen Ausdrucksstärke und Geschwindigkeit.

### P1: Normalisierung als Framework-Feature

Aktuell normalisieren Beispiele manuell.

Aufgaben:

- Input- und Output-Normalizer in `NeuroEvolution` oder ExampleConfig abstrahieren.
- Standardnormalisierer: min/max, z-score, clipping, running stats.
- Normalizer mit Genom/Experiment speichern.
- GUI-Inspect mit Rohwerten und normalisierten Werten vereinheitlichen.

Nutzen:

- Größere Aufgaben werden leichter korrekt skaliert.
- Weniger Beispiel-spezifischer Boilerplate.

### P1: Bessere Memory-Mechanismen ⚡

Persistente Node-Werte sind einfach und flexibel, aber schwer steuerbar.

**Bereits implementiert:** Persistente Nodes (`persistent=True`) behalten Werte über Forward-Passes hinweg, `reset()` löscht sie. Memory-Toggle in GUI.

**Wichtig für sequentielle Aufgaben:** Mit einem Tick-basierten Ansatz (ein Token/Zeichen pro Tick, ein "Output-relevant"-Flag als zweiter Input) kann YANE prinzipiell lernen, sich wie ein Sprachmodell zu verhalten — ohne festes Kontextfenster, da der interne State theoretisch über beliebig viele Ticks persistiert. Der entscheidende technische Engpass dabei ist Gating: ohne gezielte Schreib-/Vergess-Kontrolle kollabiert der State bei langen Sequenzen numerisch oder das Netz kann keine selektive Retention lernen. Gating (`value = gate * old + (1 - gate) * new`) ist daher der einzige echte technische Blocker für diese Aufgabenklasse.

Noch offen:

- Explizite Memory Nodes als eigener NodeType prüfen.
- Gating-Mechanismen implementieren: `value = gate * old + (1 - gate) * new`, wobei `gate` ein evolvierbarer Parameter oder ein weiterer Node-Output ist.
- Leaky memory als einfachere Variante: `value = alpha * old + new`, mit evolvierbarem Zerfall `alpha`.
- Mutation für `alpha` und Gate-Stärke.
- Reset-Regeln klarer visualisieren.

Nutzen:

- Bessere Sequenz- und Control-Fähigkeiten.
- Stabilere rekurrente Dynamik.
- Ermöglicht LLM-artiges Verhalten über Tick-basierte sequentielle Verarbeitung.

### P2: Modulare Subnetze

Für große Aufgaben können Module hilfreich sein.

Aufgaben:

- Gruppen von Nodes als Modul markieren.
- Mutation zum Duplizieren ganzer Module.
- Modul-Crossover erforschen.
- Wiederverwendbare Substrukturen speichern.

Nutzen:

- Skalierung auf komplexere Aufgaben.
- Evolution kann bereits gefundene Teilfunktionen wiederverwenden.

### P2: Indirekte Kodierung / CPPN

Direkte Kodierung jeder Connection skaliert schlecht für sehr große Netze.

Aufgaben:

- Indirekte Kodierung für Gewichtsmuster erforschen.
- CPPN/HyperNEAT-artige Verbindungsgenerierung prüfen.
- Koordinaten für Input/Output/Hidden Nodes definieren.

Nutzen:

- Potenziell viel bessere Skalierung bei großen, regelmäßigen Strukturen.
- Interessant für Bild- und Steuerungsaufgaben.

## 4. Optimierung und Hybrid-Training

### P1: Lamarckian Refinement — Restaufgaben ⚡

**Bereits implementiert:** Adaptives Lamarck (stagnationsbasiert 0→max_steps, Top-20%), expliziter Modus, per-Species Lamarck (Steps skalieren mit Species-Stagnation), `lamarck_per_species`-Diagnostics in GUI, `_lamarck_sigma`-Multiplier, kumulative Lamarck-Zeitmessung (`lamarck_time_ms`) in GUI.

**Noch offen:**

- Separate Sigma-Strategie für Lamarck (eigenes, evolvierbares sigma unabhängig von `sigma_global`, statt nur Multiplier).
- Tests für Zusammenspiel mit Strukturstagnation.

Nutzen:

- Bessere Diagnose des Lamarck-Overheads.
- Feinere Kontrolle der Schrittweite.

### P1: Lokale Optimierer für Gewichte

Statt reinem Hill-Climbing könnten bessere lokale Verfahren helfen.

Aufgaben:

- Simulated Annealing für Gewichte testen.
- Evolution Strategies für Gewichte testen.
- CMA-ES pro Topologie prüfen.
- Optionaler numerischer Gradienten-Free Optimizer.

Nutzen:

- Topologie wird evolutionär gesucht, Gewichte werden effizienter verfeinert.

### P2: Backprop-Hybrid für differenzierbare Teile

Wenn ein Netz azyklisch und differenzierbar ist, könnte Backprop optional helfen.

Aufgaben:

- Export nach PyTorch für kompatible Topologien prüfen.
- Aktivierungen mit Gradienten-Mapping definieren.
- Kurzes Gradient-Finetuning für Top-Genome.
- Ergebnis zurück in Genome schreiben.

Nutzen:

- Deutlich bessere Skalierung bei großen supervised Aufgaben.
- Bleibt optional, Neuroevolution bleibt Kernidee.

### P2: Natural Evolution Strategies (NES) als Lamarck-Alternative

Hill-Climbing ändert Gewichte zufällig und hält die bessere Variante. NES schätzt aus mehreren Perturbationen einen approximierten Fitness-Gradienten und macht gerichtete Schritte.

Aufgaben:

- `_nes_refine()` als Alternative zu `_lamarck_refine()` implementieren.
- k Perturbationen ± epsilon; Gradienten-Schätzung: `g = sum(f_i * noise_i) / (k * epsilon)`.
- Schrittweite adaptiv (analoges sigma_global-Konzept).
- Benchmark: NES vs. Hill-Climbing auf Acrobot / LunarLander.
- Umschaltbar via `set_lamarck(mode='nes')`.

Nutzen:

- Gerichtete Gewichtsoptimierung statt reinem Zufalls-Climbing.
- Skaliert besser mit Genomgröße (ein Schritt pro k Evaluierungen statt k Schritte).
- Theoretisch fundierter als blindes Hill-Climbing.

## 5. Parallelisierung und Performance

### P0: Einheitliche Worker-Abstraktion ✅

Die Trainingspfade unterscheiden sich aktuell: `train()`, GUI sequenziell, GUI multiprocess, manuell/API.

**Bereits implementiert:** `_run_evaluations()` als zentrale Evaluierungs-Pipeline, `EvaluationResult`-Dataclass für Rückgaben. Explizites Lamarck-Refinement aus `train()` in `EvaluationRunner.run()` verschoben — alle Pfade (train, GUI sequential, GUI multiprocess) durchlaufen jetzt dieselbe vollständige Pipeline: Lamarck → Evaluation → `_finalize_fitness`.

**Noch offen:** — (alle Restaufgaben sind jetzt implementiert).

Nutzen:

- Weniger doppelte Logik.
- Features greifen automatisch in GUI, Skripten und API.

### P0: Forward-Performance weiter messen ✅

Der Fast Path ist bereits optimiert, aber große Netze brauchen Benchmarks.

**Bereits implementiert:** `benchmarks/forward_bench.py` — Microbenchmarks für `forward()` über hidden-Node-Größen {10, 50, 200, 1000}, azyklisch vs. zyklisch, Median-µs/call, CLI mit `--sizes` und `--save`.

**Noch offen:** — (Basis-Benchmarks sind implementiert; Kosten-pro-Node/Regression-Tests können bei Bedarf ergänzt werden).

Nutzen:

- Optimierungen bleiben messbar.
- Große Aufgaben werden planbarer.

### P1: Vektorisierte Batch-Auswertung

Dataset-Aufgaben bewerten viele Samples pro Genom. Aktuell läuft das meist sampleweise.

Aufgaben:

- Optionales `forward_batch()` für azyklische Netze prüfen.
- NumPy-basierte Ausführung für feed-forward Topologien.
- Fallback auf normalen Forward bei Zyklen/Memory.

Nutzen:

- Viel schneller für Regression/Klassifikation.
- Wichtig für MNIST-artige Aufgaben.

### P1: Genom-Serialisierung optimieren

Größere Populationen und Multiprocessing leiden unter Pickle-Kosten.

Aufgaben:

- Kompaktere Serialisierung für Genome entwickeln.
- Nur aktive Struktur serialisieren.
- Optional shared immutable topology für verwandte Genome prüfen.
- Profiling für Multiprocessing-IPC.

Nutzen:

- Schnellere Parallelisierung.
- Weniger RAM-Druck.

### P2: GPU-Unterstützung für kompatible Netze

Nicht jedes evolvierte Netz passt gut auf GPU, aber Batch-Auswertung schon.

Aufgaben:

- Feed-forward-Genome in Matrixform exportieren.
- Batch-Auswertung auf NumPy/CuPy/PyTorch testen.
- GPU nur für große Batches aktivieren.

Nutzen:

- Größere supervised Aufgaben werden realistischer.

## 6. Robustheit und Reproduzierbarkeit

### P1: Sicherheitslimits für Werte ✅

Sehr große Gewichte/Aktivierungen können Netze instabil machen.

**Bereits implementiert:** `set_weight_clipping(w_max, b_max)` in `NeuroEvolution` — klemmt alle Gewichte auf `[-w_max, w_max]` und alle Biases auf `[-b_max, b_max]` nach jeder Mutation. `b_max` defaults zu `w_max`. `None` (default) deaktiviert Clipping.

**Noch offen:** Output-Sanitizing pro Forward; Zähler für Clipping-Ereignisse.

Nutzen:

- Stabilere Langzeitläufe.
- Weniger degenerative Genome.

## 7. API, GUI und Bedienbarkeit

### P0: GUI für fortgeschrittene Evolutionseinstellungen ✅

**Bereits in der GUI konfigurierbar:** Multi-eval, Aggregation, Sigma-Penalty, Lamarck-Modus/-Steps, Worker-Anzahl, Target-Species, Memory-Toggle, Population-Size, Max-Nodes/Connections, Target-Fitness, Checkpoint Save/Load, Normalization-Toggle, Fitness-Shaping-Toggle, Interspecies-Crossover-Rate, Convergence-Stop (eps + stagnation), Early-Stopping-Factor, Effizienzstrafe (max_ms + penalty), Elitismus (global + per species).

**Noch offen:** — (alle geplanten GUI-Einstellungen sind jetzt exponiert).

Nutzen:

- YANE wird als Experimentierwerkzeug deutlich brauchbarer.

### P1: API vollständiger machen ✅

**Bereits implementiert:** `POST /configure` mit `n_inputs`, `n_outputs`, `max_nodes`, `max_connections`, `n_initial_hidden`, `stateful`, `population_size`, `target_species`, `lamarck_steps`, `lamarck_sigma`, `lamarck_adaptive_max_steps`, `lamarck_adaptive_top_k`, `seed`. `GET /population/status`, `GET /population/best`, `GET /population/diagnostics`, `GET /population/best/export`, `POST /checkpoint/save`, `POST /checkpoint/load`.

**Noch offen:** — (alle geplanten API-Endpunkte sind jetzt implementiert).

Nutzen:

- Externe Tools können YANE ernsthaft steuern.

### P1: Genome visualisieren und inspizieren ✅

**Bereits implementiert:** NetworkCanvas (Topologie-Visualisierung), WeightHistogram (Gewichtsverteilung), FitnessChart, SpeciesChart, MutationRateChart. Connection-Gewichte farblich (blau = positiv, rot = negativ) und nach Stärke (Alpha + Linienbreite). Aktivierungsfunktion pro Node als Buchstaben-Label. Persistente Nodes in Violett mit gestricheltem Ring und Gedächtnis-Punkt. Deaktivierte Connections als graue gestrichelte Linie. Innovationsnummern als kleine Zahl unter jedem Node.

**Noch offen:** — (alle geplanten Visualisierungen sind jetzt implementiert).

Nutzen:

- Man versteht besser, welche Strukturen Evolution findet.

## 8. Benchmarking

### P1: Ablation Tests

Um zu wissen, welche Features wirklich helfen, sollten sie abschaltbar sein.

Aufgaben:

- Novelty Search an/aus.
- Speciation an/aus.
- Crossover an/aus.
- Lamarck an/aus.
- Memory an/aus.
- Diversity Injection an/aus.

Nutzen:

- Klarheit, welche Mechanismen bei welchen Aufgaben helfen.

## 9. Dokumentation

### P0: Dokumentation aktuell halten

Die technische Dokumentation sollte mit neuen Features mitwachsen.

Aufgaben:

- Jede neue Framework-Funktion in README und technischer Doku eintragen.
- Beispiele für empfohlene Einstellungen geben.
- Architekturdiagramm ergänzen.
- Glossar für NEAT/YANE-Begriffe anlegen.

Nutzen:

- YANE bleibt verstehbar, auch wenn es komplexer wird.

### P1: Entwicklerdokumentation für interne APIs

Aufgaben:

- Lifecycle eines Genoms dokumentieren.
- Population-Spawn-Zyklus dokumentieren.
- Worker-Pfade dokumentieren.
- Konventionen für Fitnessfunktionen festhalten.

Nutzen:

- Weniger Risiko bei größeren Refactors.

---

## Entwicklungsphasen (aktualisiert)

| Phase | Fokus | Enthaltene Tasks (offen) |
|---|---|---|
| **Phase 3** · Skalierung | Species-Budget-Reproduktion, Batch-Forward, Serialisierung, Worker-Pipeline | P1: Batch-Auswertung, Serialisierung |
| **Phase 4** · Schwierige Aufgaben | Curriculum, Multi-Objective, MAP-Elites, Memory/Gating, Hybrid-Optimierung | P1: Curriculum, Memory (Rest); P2: Multi-Obj, QD, NES, Backprop |

> Phase 1 und 2 sind vollständig abgeschlossen. Species-Budget-Reproduktion wurde bewusst von Phase 2 auf Phase 3 verschoben (erfordert Umbau von steady-state zu generationsbasiertem Spawn-Modell).

---

## TODO-Liste (nach Code-Analyse bereinigt)

### ✅ Vollständig implementiert

- [x] Automatische Effizienzbewertung in der Elternauswahl anwenden.
- [x] Bewertungszeit in den GUI-Worker-Pfaden messen.
- [x] Einheitliche Evaluation-Statistiken (mean/median/p95/max Eval-Zeit, Offspring-Zähler, Topologie-Historie).
- [x] GUI-Diagnostics für Eval-Zeit und Offspring-Zähler.
- [x] Elitismus explizit machen (elite_count, species_elite_count, set_elitism()).
- [x] Fitness-Sanitizing zentralisieren (nan/inf/clipping, Diagnose-Zähler).
- [x] Mehrfachbewertung pro Genom (set_multi_eval, mean/median/min, sigma-Penalty).
- [x] Species-Explosion beheben (aktives Mergen + Hard-Cap + Threshold-Kalibrierung).
- [x] Connection Enable/Disable (reversibles Deaktivieren, Forward-Filter, forward-Pfad compile-time).
- [x] Adaptive Strukturmutation (weight-biased remove, Spike-Mutation, Rewiring).
- [x] Adaptives Lamarckian Refinement (stagnationsbasiert 0→max_steps, Top-20%).
- [x] Per-Species adaptives Lamarck (Steps skalieren mit Species-Stagnation, Diagnostics in GUI).
- [x] Raten des besten Genoms in GUI anzeigen (sigma_global, Struktur-Raten, Weight/Bias/Activation-Raten).
- [x] Durchschnittliche Mutationsraten populationsweit ausgeben.
- [x] Mutationsraten-Historie in GUI plotten (MutationRateChart).
- [x] Inkrementelle Speziation (_assign_species O(0) im Steady-State, periodischer Vollscan).
- [x] Debug-Tab mit Live-Log, Toggle und Clipboard-Kopie.
- [x] Species-Hard-Cap gegen Pop-Overflow bei extremer Stagnation.
- [x] Threshold-Adjustment-Diagnostics (dbg_last_adj_n, dbg_adj_count) im Log und GUI.
- [x] UI: Collapsible Groups im LeftPanel, Übersetzung auf Englisch, Visual Polish.
- [x] EvaluationResult-Objekt (genome, fitness, elapsed_ms, n_lamarck_steps, stopped_early, raw_fitnesses).
- [x] Seed-Parameter für NeuroEvolution (set_seed).
- [x] Checkpoint-Speicherung (save_checkpoint / load_checkpoint, GUI Save/Load-Buttons).
- [x] Benchmark-Suite mit mehreren Seeds (benchmarks/run_suite.py).
- [x] API /configure erweitert (max_nodes, max_connections, n_initial_hidden, stateful, population_size, target_species, Lamarck, seed).
- [x] Fitness-Shaping (Rank-basierte Transformation vor Selektion, set_fitness_shaping).
- [x] Convergence Detection (set_convergence_stop, set_max_evaluations, on_stop-Callback, GUI-Stoppgrund).
- [x] Interspecies Crossover (~5%, set_interspecies_crossover, n_interspecies_crossover-Zähler).
- [x] Output-Scale als Strategie-Gen auf Output-Nodes (analog zu input_scale).
- [x] Ensemble-Inferenz vervollständigt (mode='mean'/'vote'/'weighted', GUI Top-5-Avg-Fitness).
- [x] Early Stopping pro Genom (Generator-Protokoll, auto-N-Kalibrierung, 20%-Warmup, Hochrechnung).
- [x] Strukturiertes Logging (logs/<kategorie>/<timestamp>/ mit run.log, config.json, best_genome.json, fitness_history.csv, Auto-Cleanup).
- [x] Testabdeckung: 359 Tests, pytest -m ci < 1 s, Coverage ≥ 80% für core/ und evolution/.
- [x] GUI: Advanced-Sektion mit Fitness-Shaping, Interspecies-Crossover, Convergence-Stop, Early-Stop, Effizienzstrafe, Elitismus.
- [x] API: Checkpoint-Endpunkte, Best-Genom-Export, Diagnostics-Endpunkt.
- [x] Fitness-Landscape: Histogramm (FitnessHistogram-Widget) + Plateau-Ratio + Sprungrate (jump_rate) in GUI.
- [x] Lamarck: Kumulative Refinement-Zeitmessung in GUI.
- [x] Adaptive Strukturmutation: Erfolgsrate-Tracking, Nutzen-Gewichtung, Per-Species-Tendenzen.
- [x] Weight/Bias-Clipping: set_weight_clipping(w_max, b_max) nach jeder Mutation.
- [x] Forward-Microbenchmarks: benchmarks/forward_bench.py (acyclic vs cyclic, n∈{10,50,200,1000}).

### ⚡ Teilweise implementiert (Rest siehe Task-Detail oben)

- [x] Fitness-Landscape Diagnostics: IQR + Histogramm + Plateau-Ratio + Sprungrate done.
- [x] Speciation robuster: Dynamische Ziel-Species, alternative Metrik, kleine Species geschützt, Stammbaum.
- [x] Worker-Abstraktion: _run_evaluations() + EvaluationResult done; explizites Lamarck in Runner verschoben, GUI-Pfad vereinheitlicht.
- [x] Genome-Visualisierung: Basis-Canvas + Gewichtsfarben + Aktivierungslabel + Persistent-Ring + Disabled-Connections + Innovationsnummern.
- [ ] Lamarck Rest: per-Species + Zeitmessung done; separate Sigma-Strategie fehlt.
- [ ] Memory-Mechanismen: Persistente Nodes done; Gating, Leaky Memory, eigener NodeType fehlen.

### 🔲 Noch nicht begonnen

- [ ] Curriculum Learning
- [ ] Multi-Objective Optimization
- [ ] Species-Budget-Reproduktion (verschoben auf Phase 3)
- [ ] Quality Diversity / MAP-Elites
- [ ] Coevolution
- [ ] Adaptive Populationsgröße
- [ ] Warm-Start / Transfer Learning
- [ ] Netzwerkgröße kontrollieren (Komplexitätsstrafe, Inactive-Detection)
- [ ] Normalisierung als Framework-Feature
- [ ] Modulare Subnetze
- [ ] CPPN / Indirekte Kodierung
- [ ] Lokale Optimierer (Simulated Annealing, CMA-ES)
- [ ] Backprop-Hybrid
- [ ] NES als Lamarck-Alternative
- [ ] Vektorisierte Batch-Auswertung
- [ ] Genom-Serialisierung optimieren
- [ ] GPU-Unterstützung
- [ ] Ablation Tests
- [ ] Dokumentation aktuell halten
- [ ] Entwicklerdokumentation

---

## Erledigte Aufgaben

### Early Stopping pro Genom bei Dataset-Aufgaben (P0)

Bei Klassifikations- und Regressions-Aufgaben wird jedes Genom auf allen Samples bewertet, auch wenn es nach wenigen Fällen offensichtlich schlecht ist.

**Design: optionales Generator-Protokoll (rückwärtskompatibel)**

Die bestehende Signatur `fitness_fn(genome) -> float` bleibt unverändert. Zusätzlich kann eine Fitnessfunktion ein **Generator** sein, der nach jedem Sample (oder Batch) einen Partial-Score yieldet. YANE erkennt Generator-Funktionen via `inspect.isgeneratorfunction()` und bricht die Auswertung ab, wenn die akkumulierte Partial-Fitness unter den Schwellwert fällt. Nicht-Generator-Funktionen laufen unverändert durch.

Beispiel (neues optionales Format):
```python
def evaluate(genome):
    fitness = 0.0
    for sample in dataset:
        genome.reset()
        outputs = genome.forward(sample["input"])
        fitness -= abs(outputs[0] - sample["output"][0])
        yield fitness  # YANE kann hier abbrechen
```

Implementierte Details:

- `set_early_stopping(factor: float | None = 1.0)` in `NeuroEvolution` — Abbruch wenn `partial_fitness < best_fitness - abs(best_fitness) * factor`. Formel ist sign-unabhängig: bei negativer Fitness (XOR) wie bei positiver (CartPole) und bei exponentiell wachsender Fitness korrekt.
- **Hochrechnung für Dataset-Aufgaben:** `estimated = partial_fitness * (N/k_sofar)` mit linearer Akkumulationsannahme.
- **Automatische N-Inferenz:** Erstes vollständiges Genom kalibriert N und schaltet Early Stopping frei. Prüfung startet ab 20% der aktuellen N.
- Abbruchzähler `n_early_stopped` in `population_memory_info()`; `stopped_early: bool` in `EvaluationResult`.
- `_early_stopping_n` in Checkpoint serialisiert.

### Fitness-Shaping / Rank-basierte Transformation (P1)

`set_fitness_shaping(enabled: bool)` in `NeuroEvolution`. Nach `_compute_shared_fitness()` werden alle `shared_fitness`-Werte durch Ränge ersetzt (linear: Rang 1 = schlechtestes Genom). In `_compute_selection_scores()` wird `rank(g.shared_fitness) / pop_size` statt `max(0.0, g.shared_fitness - min_fit + 1e-6)` verwendet. Fitness-Sharing bleibt aktiv — Rank-Shaping ersetzt nur den rohen `shared_fitness`-Wert.

### Convergence Detection und automatisches Training-Stop (P1)

`set_convergence_stop(fitness_spread_eps, min_stagnation)`: Training stoppt wenn `stagnation_count >= stagnation_threshold * min_stagnation` UND IQR der Fitness < `fitness_spread_eps`.
`set_max_evaluations(n)`: zusätzlich zu `set_max_iterations()`, zählt tatsächliche Evaluierungen.
`train(fitness_fn, on_stop: Callable[[str], None] | None = None)`: Callback mit Stoppgrund (`"target_reached"`, `"max_iterations"`, `"converged"`, `"max_evaluations"`).
GUI zeigt Stoppgrund an. `min_fitness` + `max_iterations` waren bereits vorhanden.

### Automatische Effizienz in der Elternauswahl

YANE misst Bewertungszeiten in `train()` und in den GUI-Worker-Pfaden. Die Rohfitness bleibt unverändert; Effizienz wird als eigene Variable geführt und wirkt dynamisch nur auf die Elternauswahl. Bei Stagnation sinkt die Effizienz-Relevanz bis auf `0`.

### Einheitliche Evaluation-Statistiken

Pro Trainingsschritt werden erfasst: mean/median/p95/max der Eval-Zeiten über alle evaluierten Genome, kumulative Offspring-Zähler (crossover / mutation / diversity injection), sowie eine Topologie-Historie des besten Genoms bei jeder Fitness-Verbesserung. Alle Werte fließen via `population_memory_info()` in die GUI.

### Elitismus explizit machen

`elite_count` (globale Top-k) und `species_elite_count` (bestes je Species) sind konfigurierbar via `set_elitism()`. Single-member-Species-Champions sind ebenfalls geschützt. Eliten werden nie gelöscht, aber weiterhin als Elternteile genutzt — nur Kopien mutieren.

### Fitness-Sanitizing zentralisieren

Zentrale `_apply_sanitize()`-Funktion in `NeuroEvolution`. Behandelt `nan` und `inf` defensiv (Fallback-Wert konfigurierbar), optionales Fitness-Clipping (clip_low, clip_high), Diagnose-Zähler für invalide und geclippte Werte (`n_invalid_fitness`, `n_clipped_fitness`). Konfigurierbar via `set_fitness_sanitizing()`.

### Mehrfachbewertung pro Genom

`set_multi_eval(n, aggregation, sigma_penalty)` erlaubt mehrere Episoden pro Genom. Aggregation: mean / median / min. Sigma-Penalty subtrahiert ein Vielfaches der Standardabweichung. Automatische Worker-Anpassung berücksichtigt den Mehraufwand. GUI konfigurierbar.

### Species-Explosion beheben

Die "try last species first"-Optimierung in `_assign_species()` verhinderte, dass ein steigender Kompatibilitäts-Threshold Species zusammenführen konnte — Genome landeten immer in ihrer alten Species. Fix: Aktives Mergen von Species-Paaren nach der Zuweisung, Hard-Cap gegen Pop-Overflow, kalibrierte Threshold-Anpassung (per-Evaluation-Frequenz). Species-Anzahl stabilisiert sich bei ~Ziel.

### Connection Enable/Disable

`enabled`-Flag auf `Connection`. Deaktivierte Verbindungen bleiben im Genom (reversibel), werden aber im Forward-Pass übergangen. Compile-time-Filter in allen Forward-Pfaden (acyclisch: Snapshot bei Compile, kein Runtime-Overhead; cyclisch: `if conn.enabled` in `fire()`). Selbst-adaptive Mutations-Raten. Crossover und Pickle unterstützen das Flag.

### Adaptive Strukturmutation

Drei neue Mutationsmechanismen, alle selbst-adaptiv:
- **Gewichtsbasiertes remove_connection**: Soft-min Sampling (Wahrscheinlichkeit ~ 1/|w|) bevorzugt das Entfernen unwichtiger Verbindungen.
- **Spike-Mutation**: Jede Connection kann ihr Gewicht komplett neu initialisieren (N(0, sigma_global)) statt nur zu perturbieren.
- **Rewiring**: Atomisches Remove-dann-Add; Netzgröße stabil, Topologie wird exploriert.

### Adaptives Lamarckian Refinement

Lamarck feuert automatisch ohne manuelle Konfiguration. Anzahl der Hill-Climbing-Schritte skaliert linear mit Stagnation (0 bei gutem Fortschritt → max_steps bei voller Stagnation). Nur Top-20% des evaluierten Pools werden verfeinert. Per-Species: Steps skalieren zusätzlich mit Species-Stagnation, `lamarck_per_species`-Diagnostics in GUI. Expliziter Modus (`set_lamarck(n_steps)`) bleibt für Override-Fälle erhalten.

### Mutationsraten-Visualisierung

Im LeftPanel werden die selbst-adaptiven Raten des aktuell besten Genoms angezeigt: `sigma_global`, die vier Struktur-Raten (Add/Remove Node, Add/Remove Connection) sowie populationsdurchschnittliche Weight-shift-rate, Weight-delta, Bias-rate, Bias-delta und Activation-Wechsel-rate. Populationsweite Durchschnitte zusätzlich zum Best-Genom. `MutationRateChart`-Sparkline in GUI für sigma_global + weight-rate-Historie.

### Inkrementelle Speziation

`_assign_species()` läuft im Steady-State in O(0) statt O(pop): Jedes neu evaluierte Genom cached seine letzte Species und wird beim nächsten Aufruf direkt dort eingesetzt, wenn die Kompatibilität noch stimmt. Ein periodischer Vollscan (jede k-te Generation) korrigiert Drift.

### Debug-Tab mit Live-Log

Debug-Tab in der GUI: zeilenweise Snapshot aller `population_memory_info()`-Werte alle 0.5 s, Toggle zum Pausieren, Clipboard-Kopie des vollständigen Logs.

### Species-Hard-Cap und Stabilitätsfixes

Mehrere aufeinanderfolgende Fixes für Randfälle bei Stagnation: Hard-Cap, Threshold-Freeze-Fix, Threshold-Schritt-Kalibrierung (per-Evaluation-Frequenz), Dead-Band entfernt. Diagnostics: `dbg_last_adj_n` und `dbg_adj_count` in `population_memory_info()` und im Debug-Tab sichtbar.

### UI-Verbesserungen

- **Collapsible Groups im LeftPanel**: Alle Diagnostik-Sektionen können ein- und ausgeklappt werden; Zustand bleibt erhalten.
- **Vollständige Übersetzung auf Englisch**: GUI-Texte, Labels und Tooltips durchgehend auf Englisch.
- **Visual Polish**: Chart-Ticks, Δ-Styling, Canvas-Legende, Progress-Form und Tab-Indikator verbessert; Inspect-Tab-Notification-Dot erscheint nur bei neuem Bestwert.

### Interspecies Crossover (P1)

`set_interspecies_crossover(rate: float)` in `NeuroEvolution`. Zweiter Elternteil wird mit `rate` Wahrscheinlichkeit aus anderer Species gewählt. Zähler `n_interspecies_crossover` in `population_memory_info()`. Offspring bekommt keine Sonderbehandlung bei Species-Zuweisung.

### Ensemble-Inferenz der Top-k Genome (P1)

`forward_ensemble(inputs, k, mode='mean'|'vote'|'weighted')` mit drei Modi: Mittelwert (default), Majority-Vote für binäre Klassifikation, fitness-gewichtetes Averaging. `get_ensemble(k)` gibt Top-k Genome zurück. GUI zeigt `top5_avg_fitness` im LeftPanel.

### Output-Scale als Strategie-Gen (P1)

`output_scale: float = 1.0` auf Output-Nodes (wie `input_scale` auf Input-Nodes). Evolvierbar via `mutation_output_scale`. Forward-Pass multipliziert Ausgabe-Aktivierung mit `output_scale`. In `node.py`: `_SLOT_DEFAULTS`, `copy()`, `mutate()`.

### Seeding zentralisieren (P0)

`NeuroEvolution(seed=...)` / `set_seed(seed)` setzt Python `random` und `numpy.random` reproduzierbar. Seed wird in Logs, `config.json` und GUI angezeigt. Gym-Env-Seeds liegen beim Nutzer.

### Checkpoints (P0)

`save_checkpoint(path)` / `load_checkpoint(path)`: Population + InnovationTracker atomar speichern und laden. GUI Save/Load-Buttons mit Dateidialog. API `POST /configure` akzeptiert `seed`-Parameter. `_early_stopping_n`, `_n_early_stopped` etc. werden mit serialisiert.

### API /configure erweitert (P1)

`POST /configure` akzeptiert: `n_inputs`, `n_outputs`, `max_nodes`, `max_connections`, `n_initial_hidden`, `stateful`, `population_size`, `target_species`, `lamarck_steps`, `lamarck_sigma`, `lamarck_adaptive_max_steps`, `lamarck_adaptive_top_k`, `seed`.

### API-Endpunkte vervollständigt (P1)

`POST /checkpoint/save` + `POST /checkpoint/load` zum Speichern/Laden der Population. `GET /population/best/export` liefert das beste Genom als JSON mit allen Nodes und Connections. `GET /population/diagnostics` gibt die vollen `population_memory_info()`-Daten zurück.

### GUI für fortgeschrittene Einstellungen vervollständigt (P0)

Neue „Advanced"-Sektion in den Training-Settings: Fitness-Shaping-Toggle, Interspecies-Crossover-Rate, Convergence-Stop (IQR-Schwelle × Stagnationsfaktor), Early-Stopping-Factor, Effizienzstrafe (max_ms × penalty), Elitismus (global + per species).

### Fitness-Landscape Diagnostics (P1)

Fitness-Histogramm (10 Bins) als `FitnessHistogram`-Widget in der GUI (Population-Sektion). Plateau-Ratio (`stagnation_count / stagnation_threshold`) als Label. IQR war bereits vorhanden.

### Lamarck Refinement-Zeitmessung (P1)

Kumulative `lamarck_time_ms`-Messung in `_lamarck_refine()`, exponiert in `population_memory_info()` und GUI („Time total (ms)"). Serialisiert in Checkpoints.

### Standard-Benchmark-Suite (P0)

`benchmarks/run_suite.py`: XOR, basic_multiplication, CartPole-v1, Acrobot-v1. Pro Benchmark n Runs mit Seeds 0..n-1, Time-to-Solve, Erfolgsrate. Ergebnisse als JSON in `benchmarks/results/`. `--fast`-Flag für CI-Suite (nur XOR + basic_multiplication).

### Strukturiertes Logging (P1)

`logs/<kategorie>/<timestamp>/`-Struktur mit `run.log`, `config.json`, `best_genome.json`, `fitness_history.csv`. `util/logger.py`: `setup_logging(name, log_root_override)`, Auto-Cleanup pro Kategorie (max 20). API-Server nutzt `"api"`-Kategorie. 359 Tests, `pytest -m ci` < 1 s, Coverage ≥ 80% für `core/` und `evolution/`.

