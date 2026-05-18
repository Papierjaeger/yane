# Tasks: YANE staerker machen

Diese Datei sammelt Ideen, Aufgaben und Forschungsrichtungen, um YANE allgemein leistungsfaehiger zu machen. Es geht hier bewusst nicht um Spezialtuning einzelner Beispiele, sondern um Verbesserungen am Framework, damit groessere, schwierigere und laengere Aufgaben besser bewaeltigt werden koennen.

## Prioritaeten

Legende:

- `P0`: hoher Hebel, nah am aktuellen Code, wahrscheinlich direkt nuetzlich
- `P1`: wichtiger Ausbau, mittlerer Aufwand
- `P2`: experimentell oder groesserer Umbau

## 1. Fitness, Evaluation und Training

### Erledigt: Automatische Effizienz in der Elternauswahl

YANE misst Bewertungszeiten inzwischen in `train()` und in den GUI-Worker-Pfaden. Die Rohfitness bleibt unverändert; Effizienz wird als eigene Variable geführt und wirkt dynamisch nur auf die Elternauswahl. Bei Stagnation sinkt die Effizienz-Relevanz bis auf `0`.

Offene Anschlussideen:

- API um optionale Bewertungszeit erweitern.
- GUI-Trends fuer Effizienz ueber Zeit plotten.
- Feste `EfficiencyPenalty` optional in der GUI konfigurierbar machen.

### P0: Einheitliche Evaluation-Statistiken

YANE sollte pro Training mehr interne Messwerte erfassen.

Aufgaben:

- Durchschnittliche Eval-Zeit pro Genom speichern.
- Median, p95 und Max-Eval-Zeit erfassen.
- Nodes/Connections des besten Genoms historisieren.
- Species-Zahl, Novelty-Gewicht, Kompatibilitaetsschwelle loggen.
- Anzahl Crossover vs. Mutation vs. Diversity Injection erfassen.
- GUI-Ansicht fuer diese Kennzahlen ergaenzen.

Nutzen:

- Man erkennt, warum ein Training stagniert.
- Performance-Probleme werden sichtbar.
- Groessere Experimente werden vergleichbarer.

### P0: Fitness-Sanitizing zentralisieren

Aktuell muss jede Fitnessfunktion selbst robust sein. Degenerative Netze koennen `nan`, `inf` oder extreme Werte erzeugen.

Aufgaben:

- Zentrale Funktion `sanitize_fitness(value)` einfuehren.
- `nan` und `inf` defensiv behandeln.
- Optional Fitness-Clipping konfigurierbar machen.
- Warnungen/Diagnose bei invaliden Fitnesswerten sammeln.

Nutzen:

- Mehr Stabilitaet bei grossen Suchraeumen.
- Weniger Abbrueche durch einzelne kaputte Genome.

### P1: Mehrfachbewertung pro Genom

Bei stochastischen Umgebungen ist eine einzelne Episode oft zu verrauscht.

Aufgaben:

- Optional `n_evaluations_per_genome` unterstuetzen.
- Mittelwert, Median oder Worst-Case als Fitness erlauben.
- Varianz als zusaetzliche Strafe verwenden.
- GUI/API konfigurierbar machen.

Nutzen:

- Robustere Policies.
- Weniger Overfitting auf zufaellige Episoden.

### P1: Curriculum Learning

Schwere Aufgaben koennen in Stufen gelernt werden.

Aufgaben:

- Curriculum-Interface definieren.
- Fitnessfunktion kann aktuelle Stufe melden.
- Automatischer Stufenwechsel bei Ziel-Fitness.
- Population beim Stufenwechsel behalten.

Nutzen:

- Komplexe Aufgaben werden schrittweise lernbar.
- Besonders nuetzlich fuer lange Sequenzen und Control-Aufgaben.

### P2: Multi-Objective Optimization

Aktuell wird Fitness als einzelner Zahlenwert optimiert.

Aufgaben:

- Fitness als Vektor erlauben, z. B. Leistung, Geschwindigkeit, Groesse, Stabilitaet.
- Pareto-Selektion implementieren.
- NSGA-II-artige Auswahl pruefen.
- GUI-Darstellung fuer Pareto-Front.

Nutzen:

- YANE muss nicht alle Ziele in eine fragile Fitnessformel pressen.
- Gute Basis fuer groessere Aufgaben mit Trade-offs.

## 2. Evolution und Suchstrategie

### P0: Elitismus explizit machen

Aktuell schuetzt `_prune()` das beste Genom und Species-Champions. Das sollte als klares Feature sichtbar und konfigurierbar sein.

Aufgaben:

- `elite_count` pro Population einfuehren.
- `species_elite_count` einfuehren.
- Dokumentieren, wann Eliten unveraendert erhalten bleiben.
- Tests fuer Elite-Erhalt ueber mehrere Spawn-Zyklen.

Nutzen:

- Weniger Risiko, gute Loesungen durch Zufall zu verlieren.
- Besser steuerbare Evolution.

### P0: Mutation- und Strategie-Gene sichtbar machen

Die selbstadaptierenden Raten sind stark, aber schwer zu beobachten.

Aufgaben:

- Durchschnittliche Mutationsraten pro Population ausgeben.
- Raten des besten Genoms anzeigen.
- Historie in GUI plotten.
- Warnung, wenn Raten an Minimum oder Maximum kleben.

Nutzen:

- Bessere Diagnose von Stagnation.
- Man sieht, ob Evolution eher Struktur oder Gewichte erforscht.

### P1: Adaptive Strukturmutation verbessern

Aktuell gibt es feste Strukturmutationstypen mit selbstadaptierenden Raten.

Aufgaben:

- Erfolgsrate einzelner Mutationstypen messen.
- Mutationstypen gewichten nach historischem Nutzen.
- Unterschiedliche Mutationstendenzen je Species erlauben.
- Strukturwachstum bei Stagnation aggressiver steuern.

Nutzen:

- Weniger blinde Mutation.
- Bessere Skalierung bei grossen Topologien.

### P1: Speciation robuster machen

Die Speziation ist zentral fuer NEAT-artige Evolution.

Aufgaben:

- Ziel-Species dynamisch an Populationsgroesse koppeln.
- Alternative Kompatibilitaetsmetriken testen.
- Per-Species-Stagnation aktiv fuer Spawn-Budget nutzen.
- Kleine Species nicht zu frueh verlieren lassen.
- Species-Historie und Stammbaum speichern.

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

- Naeher an klassischem NEAT.
- Besserer Schutz neuer Strukturinnovationen.

### P2: Quality Diversity / MAP-Elites

YANE hat bereits Novelty Search. Der naechste Schritt waere Quality Diversity.

Aufgaben:

- Behavior Descriptor API oeffnen.
- Archiv nach Verhaltensdimensionen aufbauen.
- Pro Zelle bestes Genom behalten.
- Neue Kandidaten aus verschiedenen Archivbereichen erzeugen.

Nutzen:

- Evolution sammelt viele verschiedene brauchbare Loesungen.
- Sehr nuetzlich fuer Aufgaben mit sparse rewards.

### P2: Coevolution

Einige Aufgaben koennen durch Gegenspieler oder Aufgabenvariation staerker werden.

Aufgaben:

- Interface fuer Population gegen Population entwerfen.
- Kompetitive Fitness unterstuetzen.
- Hall-of-Fame-Gegner speichern.
- Rock-Paper-Scissors-Zyklen durch Archiv vermeiden.

Nutzen:

- Starke Methode fuer Strategien, Spiele und robuste Policies.

## 3. Netzwerkmodell

### P0: Netzwerkgroesse kontrollierter wachsen lassen

Groessere Aufgaben brauchen groessere Netze, aber unkontrolliertes Wachstum wird teuer.

Aufgaben:

- Weiche Komplexitaetsstrafe optional einfuehren.
- Inaktive Nodes/Connections erkennen.
- Pruning-Mutation gezielter auf wirkungslose Struktur anwenden.
- Netzkomplexitaet in Fitnessstatistik anzeigen.

Nutzen:

- Bessere Balance zwischen Ausdrucksstaerke und Geschwindigkeit.

### P1: Connection Enable/Disable statt Loeschen

NEAT nutzt oft deaktivierbare Gene. Aktuell werden Connections entfernt.

Aufgaben:

- `enabled`-Flag fuer Connections einfuehren.
- Mutation zum Aktivieren/Deaktivieren.
- Crossover-Regeln fuer deaktivierte Gene.
- Forward ignoriert deaktivierte Connections.

Nutzen:

- Struktur kann temporaer ausgeschaltet werden, ohne historische Information zu verlieren.
- Crossover wird stabiler.

### P1: Normalisierung als Framework-Feature

Aktuell normalisieren Beispiele manuell.

Aufgaben:

- Input- und Output-Normalizer in `NeuroEvolution` oder ExampleConfig abstrahieren.
- Standardnormalisierer: min/max, z-score, clipping, running stats.
- Normalizer mit Genom/Experiment speichern.
- GUI-Inspect mit Rohwerten und normalisierten Werten vereinheitlichen.

Nutzen:

- Groessere Aufgaben werden leichter korrekt skaliert.
- Weniger Beispiel-spezifischer Boilerplate.

### P1: Bessere Memory-Mechanismen

Persistente Node-Werte sind einfach und flexibel, aber schwer steuerbar.

Aufgaben:

- Explizite Memory Nodes als eigener NodeType pruefen.
- Gating-Mechanismen fuer Memory testen.
- Leaky memory: `value = alpha * old + new`.
- Mutation fuer Memory-Zerfall `alpha`.
- Reset-Regeln klarer visualisieren.

Nutzen:

- Bessere Sequenz- und Control-Faehigkeiten.
- Stabilere rekurrente Dynamik.

### P2: Modulare Subnetze

Fuer grosse Aufgaben koennen Module hilfreich sein.

Aufgaben:

- Gruppen von Nodes als Modul markieren.
- Mutation zum Duplizieren ganzer Module.
- Modul-Crossover erforschen.
- Wiederverwendbare Substrukturen speichern.

Nutzen:

- Skalierung auf komplexere Aufgaben.
- Evolution kann bereits gefundene Teilfunktionen wiederverwenden.

### P2: Indirekte Kodierung / CPPN

Direkte Kodierung jeder Connection skaliert schlecht fuer sehr grosse Netze.

Aufgaben:

- Indirekte Kodierung fuer Gewichtsmuster erforschen.
- CPPN/HyperNEAT-artige Verbindungsgenerierung pruefen.
- Koordinaten fuer Input/Output/Hidden Nodes definieren.

Nutzen:

- Potenziell viel bessere Skalierung bei grossen, regelmaessigen Strukturen.
- Interessant fuer Bild- und Steuerungsaufgaben.

## 4. Optimierung und Hybrid-Training

### P0: Lamarckian Refinement besser integrieren

Das vorhandene Lamarckian Refinement ist stark, aber teuer.

Aufgaben:

- Refinement-Zeit messen und anzeigen.
- Nur Top-K-Genome refinieren.
- Refinement adaptiv bei Stagnation aktivieren.
- Separate Sigma-Strategie fuer Lamarck-Schritte.
- Tests fuer Zusammenspiel mit Strukturstagnation.

Nutzen:

- Schnellere Gewichtsanpassung ohne zu hohe Kosten.

### P1: Lokale Optimierer fuer Gewichte

Statt reinem Hill-Climbing koennten bessere lokale Verfahren helfen.

Aufgaben:

- Simulated Annealing fuer Gewichte testen.
- Evolution Strategies fuer Gewichte testen.
- CMA-ES pro Topologie pruefen.
- Optionaler numerischer Gradienten-Free Optimizer.

Nutzen:

- Topologie wird evolutionaer gesucht, Gewichte werden effizienter verfeinert.

### P2: Backprop-Hybrid fuer differenzierbare Teile

Wenn ein Netz azyklisch und differenzierbar ist, koennte Backprop optional helfen.

Aufgaben:

- Export nach PyTorch fuer kompatible Topologien pruefen.
- Aktivierungen mit Gradienten-Mapping definieren.
- Kurzes Gradient-Finetuning fuer Top-Genome.
- Ergebnis zurueck in Genome schreiben.

Nutzen:

- Deutlich bessere Skalierung bei grossen supervised Aufgaben.
- Bleibt optional, Neuroevolution bleibt Kernidee.

## 5. Parallelisierung und Performance

### P0: Einheitliche Worker-Abstraktion

Die Trainingspfade unterscheiden sich aktuell: `train()`, GUI sequenziell, GUI multiprocess, manuell/API.

Aufgaben:

- Gemeinsame Evaluation-Pipeline bauen.
- Timing, Fitness-Sanitizing, Effizienzstrafe und Logging zentral anwenden.
- Worker-Ergebnisse als Objekt zurueckgeben: `genome`, `fitness`, `elapsed_ms`, `metadata`.

Nutzen:

- Weniger doppelte Logik.
- Features greifen automatisch in GUI, Skripten und API.

### P0: Forward-Performance weiter messen

Der Fast Path ist bereits optimiert, aber grosse Netze brauchen Benchmarks.

Aufgaben:

- Microbenchmarks fuer `forward()` erstellen.
- Azyklische vs. zyklische Netze vergleichen.
- Kosten pro Node/Connection messen.
- Performance-Regressions in Tests aufnehmen.

Nutzen:

- Optimierungen bleiben messbar.
- Grosse Aufgaben werden planbarer.

### P1: Vektorisierte Batch-Auswertung

Dataset-Aufgaben bewerten viele Samples pro Genom. Aktuell laeuft das meist sampleweise.

Aufgaben:

- Optionales `forward_batch()` fuer azyklische Netze pruefen.
- NumPy-basierte Ausfuehrung fuer feed-forward Topologien.
- Fallback auf normalen Forward bei Zyklen/Memory.

Nutzen:

- Viel schneller fuer Regression/Klassifikation.
- Wichtig fuer MNIST-artige Aufgaben.

### P1: Genom-Serialisierung optimieren

Groessere Populationen und Multiprocessing leiden unter Pickle-Kosten.

Aufgaben:

- Kompaktere Serialisierung fuer Genome entwickeln.
- Nur aktive Struktur serialisieren.
- Optional shared immutable topology fuer verwandte Genome pruefen.
- Profiling fuer Multiprocessing-IPC.

Nutzen:

- Schnellere Parallelisierung.
- Weniger RAM-Druck.

### P2: GPU-Unterstuetzung fuer kompatible Netze

Nicht jedes evolvierte Netz passt gut auf GPU, aber Batch-Auswertung schon.

Aufgaben:

- Feed-forward-Genome in Matrixform exportieren.
- Batch-Auswertung auf NumPy/CuPy/PyTorch testen.
- GPU nur fuer grosse Batches aktivieren.

Nutzen:

- Groessere supervised Aufgaben werden realistischer.

## 6. Robustheit und Reproduzierbarkeit

### P0: Seeding zentralisieren

Reproduzierbarkeit ist fuer Experimente wichtig.

Aufgaben:

- `NeuroEvolution(seed=...)` unterstuetzen.
- Python `random`, NumPy und Gymnasium-Env-Seeds setzen.
- Seed in Logs und GUI anzeigen.
- Beispielruns reproduzierbar machen.

Nutzen:

- Bugs und Verbesserungen werden vergleichbar.
- Trainingsresultate lassen sich nachvollziehen.

### P0: Checkpoints

Lange Trainings brauchen Speicherung und Wiederaufnahme.

Aufgaben:

- Population inklusive InnovationTracker speichern.
- Bestes Genom separat speichern.
- Training aus Checkpoint fortsetzen.
- GUI-Buttons fuer Save/Load.
- API-Endpunkte fuer Checkpointing.

Nutzen:

- Lange Experimente gehen nicht verloren.
- Gute Genome koennen spaeter weiterentwickelt werden.

### P1: Experiment-Logging

Neben dem aktuellen Logger braucht YANE strukturierte Runs.

Aufgaben:

- Run-Verzeichnis pro Training.
- Konfiguration als JSON speichern.
- Fitnesshistorie als CSV/JSONL speichern.
- Best-Genome pro Intervall speichern.
- Optional TensorBoard-kompatible Logs.

Nutzen:

- Vergleich mehrerer Experimente.
- Grundlage fuer Benchmarking.

### P1: Sicherheitslimits fuer Werte

Sehr grosse Gewichte/Aktivierungen koennen Netze instabil machen.

Aufgaben:

- Optionales Weight-Clipping.
- Optionales Bias-Clipping.
- Output-Sanitizing pro Forward.
- Zaehler fuer Clipping-/Overflow-Ereignisse.

Nutzen:

- Stabilere Langzeitlaeufe.
- Weniger degenerative Genome.

## 7. API, GUI und Bedienbarkeit

### P0: GUI fuer fortgeschrittene Evolutionseinstellungen

Viele starke Features sind im Code, aber nicht sichtbar.

Aufgaben:

- Effizienzstrafe konfigurierbar machen.
- Lamarck-Sigma konfigurierbar machen.
- Population/Species/Novelty-Statistiken anzeigen.
- Mutationsraten des besten Genoms anzeigen.
- Checkpoint Save/Load.

Nutzen:

- YANE wird als Experimentierwerkzeug deutlich brauchbarer.

### P1: API vollstaendiger machen

Die API kann aktuell nur `n_inputs` und `n_outputs` konfigurieren.

Aufgaben:

- `max_nodes`, `max_connections`, `n_initial_hidden`, `stateful` in `/configure`.
- Population size, target species, Lamarck, Resource Limits.
- Bestes Genom exportieren.
- Checkpoints per API.
- Trainingsstatus mit Diagnostics.

Nutzen:

- Externe Tools koennen YANE ernsthaft steuern.

### P1: Genome visualisieren und inspizieren

Die Netzwerkvisualisierung kann mehr Diagnose liefern.

Aufgaben:

- Connection-Gewichte farblich und nach Staerke darstellen.
- Aktivierungsfunktionen pro Node anzeigen.
- Persistente Nodes markieren.
- Inaktive/nie getriggerte Struktur markieren.
- Innovationsnummern optional anzeigen.

Nutzen:

- Man versteht besser, welche Strukturen Evolution findet.

## 8. Benchmarking

### P0: Standard-Benchmark-Suite

YANE braucht feste Benchmarks, um echte Fortschritte zu messen.

Aufgaben:

- Kleine deterministische Benchmarks definieren.
- Mittelwert ueber mehrere Seeds messen.
- Erfolgskriterium und Time-to-Solve erfassen.
- CI-freundliche schnelle Benchmarks.
- Separate lange Benchmark-Suite fuer lokale Experimente.

Nutzen:

- Verbesserungen werden objektiver.
- Regressionen fallen schneller auf.

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
- Beispiele fuer empfohlene Einstellungen geben.
- Architekturdiagramm ergaenzen.
- Glossar fuer NEAT/YANE-Begriffe anlegen.

Nutzen:

- YANE bleibt verstehbar, auch wenn es komplexer wird.

### P1: Entwicklerdokumentation fuer interne APIs

Aufgaben:

- Lifecycle eines Genoms dokumentieren.
- Population-Spawn-Zyklus dokumentieren.
- Worker-Pfade dokumentieren.
- Konventionen fuer Fitnessfunktionen festhalten.

Nutzen:

- Weniger Risiko bei groesseren Refactors.

## 10. Moegliche grosse Entwicklungsphasen

### Phase 1: Stabilisieren und sichtbar machen

- Effizienzstrafe in alle Trainingspfade bringen.
- Evaluation-Statistiken zentralisieren.
- Seeding einfuehren.
- Checkpoints fuer Population und Best-Genome.
- GUI-Diagnostik erweitern.

### Phase 2: Evolution robuster machen

- Expliziter Elitismus.
- Species-Budget-Reproduktion.
- Adaptive Strukturmutation.
- Fitness-Sanitizing.
- Standard-Benchmarks und Ablations.

### Phase 3: Skalierung

- Batch-Forward fuer azyklische Netze.
- Bessere Serialisierung.
- Worker-Pipeline vereinheitlichen.
- Optional GPU/PyTorch-Export fuer grosse Batch-Aufgaben.

### Phase 4: Schwierige Aufgaben

- Curriculum Learning.
- Multi-Objective Optimization.
- Quality Diversity / MAP-Elites.
- Bessere Memory Nodes.
- Hybrid-Optimierung fuer Gewichte.

## 11. Erste konkrete TODO-Liste

- [x] Automatische Effizienzbewertung in der Elternauswahl anwenden.
- [x] Bewertungszeit in den GUI-Worker-Pfaden messen.
- [ ] `EvaluationResult`-Objekt einfuehren.
- [ ] Fitness-Sanitizing zentralisieren.
- [ ] Seed-Parameter fuer `NeuroEvolution` einfuehren.
- [ ] Checkpoint-Speicherung fuer Population und InnovationTracker.
- [ ] GUI-Diagnostics fuer Eval-Zeit, Species, Novelty und Topologie.
- [ ] Benchmark-Suite mit mehreren Seeds erstellen.
- [ ] Mutationsraten-Diagnostics implementieren.
- [ ] API-`/configure` um wichtige Konfigurationsparameter erweitern.
