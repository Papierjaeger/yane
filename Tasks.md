# Tasks: YANE staerker machen

Diese Datei sammelt Ideen, Aufgaben und Forschungsrichtungen, um YANE allgemein leistungsfaehiger zu machen. Es geht hier bewusst nicht um Spezialtuning einzelner Beispiele, sondern um Verbesserungen am Framework, damit groessere, schwierigere und laengere Aufgaben besser bewaeltigt werden koennen.

## Prioritaeten

Legende:

- `P0`: hoher Hebel, nah am aktuellen Code, wahrscheinlich direkt nuetzlich
- `P1`: wichtiger Ausbau, mittlerer Aufwand
- `P2`: experimentell oder groesserer Umbau

## 1. Fitness, Evaluation und Training


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

### P0: Early Stopping pro Genom bei Dataset-Aufgaben

Bei Klassifikations- und Regressions-Aufgaben wird jedes Genom auf allen Samples bewertet, auch wenn es nach wenigen Faellen offensichtlich schlecht ist.

**Design: optionales Generator-Protokoll (rueckwaertskompatibel)**

Die bestehende Signatur `fitness_fn(genome) -> float` bleibt unveraendert. Zusaetzlich kann eine Fitnessfunktion ein **Generator** sein, der nach jedem Sample (oder Batch) einen Partial-Score yieldet. YANE erkennt Generator-Funktionen via `inspect.isgeneratorfunction()` und bricht die Auswertung ab, wenn die akkumulierte Partial-Fitness unter `early_stop_threshold` faellt. Nicht-Generator-Funktionen laufen unveraendert durch.

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

Aufgaben:

- `set_early_stopping(factor: float | None = 1.0)` in `NeuroEvolution` — setzt den relativen Abbruchfaktor. Abbruch wenn `partial_fitness < best_fitness - abs(best_fitness) * factor`. Diese Formel ist sign-unabhaengig: bei negativer Fitness (XOR: best=-0.5, factor=1.0 → threshold=-1.0) wie bei positiver Fitness (CartPole: best=200, factor=1.0 → threshold=0) und bei exponentiell wachsender Fitness korrekt. Default `1.0` bedeutet: Abbruch sobald das Genom mehr als 100% der besten bekannten Fitness hinter dem Populationsbesten liegt.
- **Hochrechnung fuer Dataset-Aufgaben:** Da nach k von N Samples `partial_fitness ≈ k/N * final_fitness`, wird der Vergleich mit der extrapolierten Schaetzung durchgefuehrt: `estimated = partial_fitness * (N/k_sofar)`. Die Extrapolation setzt **lineare Akkumulation** voraus (Summe oder laufendes Mittel). Nicht-lineare Aggregationen (z.B. `max`, `sqrt`, Bonus am Ende) sind vom Nutzer eigenverantwortlich zu handhaben: entweder Generator nicht verwenden, oder bereits normalisierte Werte yielden (laufendes Mittel statt laufende Summe) — dann entfaellt die Hochrechnung und der Vergleich ist direkt.
- **Automatische N-Inferenz:** Jedes Genom, das nicht fruehzeitig abgebrochen wird, aktualisiert N mit der tatsaechlichen Yield-Anzahl dieses Runs. Damit passt sich N automatisch an, wenn eine Fitnessfunktion bei verschiedenen Runs unterschiedlich viele Samples ausfuehrt (z.B. adaptives Curriculum, zufaelliges Sampling). N startet als `None`; das erste vollstaendige Genome kalibriert N und schaltet Early Stopping frei. Pruefung startet erst ab 20% der aktuellen N (`k >= N // 5`). Fuer gym-Multi-Eval (Mittelwert ueber Episoden statt Summe) ist N = `n_evaluations` bereits bekannt — direkter Vergleich ohne Hochrechnung.
- In `_run_evaluations()`: wenn `inspect.isgeneratorfunction(fitness_fn)`, Generator statt direktem Aufruf verwenden. Nach jedem `yield` pruefen ob Partial-Fitness < threshold; falls ja, Generator schliessen und Partial-Fitness als Endwert verwenden.
- Abbruchzaehler `n_early_stopped` in `population_memory_info()` aufnehmen; ins `EvaluationResult`-Objekt als `stopped_early: bool`.
- Kompatibel mit Multi-Eval halten (alle n Durchlaeufe koennen individuell fruehzeitig abgebrochen werden).

Nutzen:

- Viel schnellere Population-Selektion bei grossen Datasets (XOR, MNIST).
- Macht groessere Datasets fuer YANE praktisch.

### P1: Fitness-Shaping / Rank-basierte Transformation

Rohe Fitness-Werte koennen schlecht skalieren: kleine Unterschiede bei hoher Fitness, riesige Unterschiede bei niedrigen Werten. Rank-Transformation macht Selection stabiler.

**Verhaeltnis zu Fitness-Sharing:** Fitness-Sharing (`_compute_shared_fitness()`) ist aktiv und teilt jeden Fitness-Wert durch die Species-Groesse, bevor Tournament-Selection laeuft. Fitness-Shaping ergaenzt das: die `shared_fitness`-Werte werden zusaetzlich in Raenge umgewandelt, bevor sie in die Selection-Formel einfliessen. Fitness-Sharing bleibt aktiv — Rank-Shaping ersetzt nur den rohen `shared_fitness`-Wert durch seinen Rang.

In der Selection-Formel (`_compute_selection_scores()`) wuerde:
```
max(0.0, g.shared_fitness - min_fit + 1e-6)
```
ersetzt durch:
```
rank(g.shared_fitness) / pop_size  # linear normiert auf [1/N, 1]
```

Aufgaben:

- Optionale Rank-Transformation: nach `_compute_shared_fitness()` alle `shared_fitness`-Werte durch Raenge ersetzen (linear: Rang 1 = schlechtestes Genom).
- Lineare oder nichtlineare Rank-Skala wahlweise (linear ausreichend fuer Start).
- `set_fitness_shaping(enabled: bool)` in `NeuroEvolution` API.
- Toggle in GUI.

Nutzen:

- Selection ist skalierungsinvariant — keine Empfindlichkeit gegenueber Fitness-Ausreissern.
- Stabiler bei sehr sparse Rewards und bei sehr dichter Fitness-Verteilung.
- Bekannt aus CMA-ES und OpenAI-ES als wichtige Robustheitsverbesserung.

### P1: Convergence Detection und automatisches Training-Stop

Aktuell laeuft Training bis der Nutzer haelt. Es gibt keine automatische Erkennung von Konvergenz.

**Bereits vorhanden:** `min_fitness` (Stopp wenn Fitness-Ziel erreicht) und `max_iterations` (Stopp nach N Schritten) existieren bereits als `set_min_fitness()` / `set_max_iterations()` in `NeuroEvolution`.

Noch fehlende Aufgaben:

- **Fitness-Spread-Konvergenz:** `set_convergence_stop(fitness_spread_eps: float, min_stagnation: float = 1.0)` — Training stoppt wenn `stagnation_count >= stagnation_threshold * min_stagnation` UND IQR der Fitness in der evaluierten Population < `fitness_spread_eps`. (IQR wird bereits in der Fitness-Landscape-Diagnostics-Task berechnet.)
- **`max_evaluations`:** Alias / Ergaenzung zu `max_iterations` der tatsaechliche Evaluierungen zaehlt (relevant fuer Multi-Eval: `max_evaluations = max_iterations × n_evaluations`). `set_max_evaluations(n: int)` hinzufuegen.
- **Callback:** `train(fitness_fn, on_stop: Callable[[str], None] | None = None)` — wird am Ende aufgerufen mit Grund (`"target_reached"`, `"max_iterations"`, `"converged"`, `"max_evaluations"`).
- GUI-Anzeige des Stoppgrundes und automatischer Stop (bereits `min_fitness` triggert GUI-Stop).

Nutzen:

- Skripte koennen unbeaufsichtigt laufen.
- Benchmark-Suite wird einfacher (kein manuelles Stoppen).

### P1: Fitness-Landscape Diagnostics

Wie verteilt sich Fitness in der Population? Gibt es Sprünge, Plateaus, bimodale Verteilungen?

Aufgaben:

- Fitness-Histogramm der aktuellen Population berechnen und in GUI anzeigen.
- Fitness-Sprungrate verfolgen (Anteil Evaluierungen, die neuen Bestwert setzen).
- Plateau-Erkennung: wie lange ist stagnation_count / max_size > 0.9?
- Fitness-Interquartilsabstand (IQR) als Diversitaetsmass anzeigen.

Nutzen:

- Sieht man sofort, ob die Population kollabiert (alle gleich gut), exploriert (breite Verteilung) oder stagniert.
- Hilft beim Tuning von Populationsgroesse, Elitismus und Novelty-Gewicht.

## 2. Evolution und Suchstrategie


### P1: Mutation- und Strategie-Gene sichtbar machen

Die selbstadaptierenden Raten sind stark, aber schwer zu beobachten. Raten des besten Genoms bereits in GUI (siehe Erledigte). Offen:

Aufgaben:

- Durchschnittliche Mutationsraten populationsweit ausgeben (nicht nur bestes Genom).
- Historie der Raten in GUI plotten.
- Warnung, wenn Raten an Minimum oder Maximum kleben.

Nutzen:

- Bessere Diagnose von Stagnation.
- Man sieht, ob Evolution eher Struktur oder Gewichte erforscht.

### P1: Adaptive Strukturmutation — Restaufgaben

Erste Verbesserungen implementiert (siehe Erledigte). Offen:

Aufgaben:

- Erfolgsrate einzelner Mutationstypen messen und anzeigen.
- Mutationstypen populationsweit nach historischem Nutzen gewichten.
- Unterschiedliche Mutationstendenzen je Species erlauben.

Nutzen:

- Weniger blinde Mutation bei komplexen Aufgaben.
- Bessere Diagnose, welche Strukturmutation tatsaechlich hilft.

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

### P1: Interspecies Crossover

Original-NEAT erlaubt gelegentliches Crossover zwischen verschiedenen Species. Aktuell findet Crossover nur innerhalb derselben Species statt.

**Species-Zuweisung des Offspring:** Ein Interspecies-Offspring bekommt kein Sonderbehandlung bei der Zuweisung — `_species_stale = True` wird gesetzt und das Offspring landet beim naechsten `_assign_species()`-Aufruf beim naechsten kompatiblen Repraesentanten (normales NEAT-Assignment). Es erhaelt nur keinen Species-Elitism-Schutz (d.h. es zaehlt nicht als "Champion" einer Species und ist nicht automatisch Elite).

Aufgaben:

- Kleinen Anteil (~5%) aller Crossover-Events interspecies erlauben: zweiter Elternteil wird aus einer anderen Species gewaehlt.
- Interspecies-Offspring wird als `n_interspecies_crossover` in `population_memory_info()` gezaehlt.
- Rate als fixer Hyperparameter (kein Strategie-Gen, um Komplexitaet niedrig zu halten).
- `set_interspecies_crossover(rate: float)` in `NeuroEvolution` API.

Nutzen:

- Kombiniert strukturelle Innovationen aus verschiedenen Niches.
- Kann lokale Optima aufbrechen, die Species-Isolation sonst schuetzt.

### P1: Per-Species adaptives Lamarck

Aktuell bekommt jedes Genome denselben (stagnationsbasierten) Lamarck-Aufwand. Species, die stark stagnieren, koennten mehr Refinement-Schritte bekommen.

Aufgaben:

- Pro Species `stagnation_count` verfolgen (bereits teilweise im Code).
- Lamarck-Steps pro Genome nach Species-Stagnation skalieren statt nach globaler Stagnation.
- Diagnostics: mittlere Lamarck-Steps je Species anzeigen.

Nutzen:

- Ressourcen werden dort eingesetzt, wo Verbesserung am staerksten blockiert ist.
- Feinere Kontrolle als globales Lamarck.

### P1: Ensemble-Inferenz der Top-k Genome

Das aktuelle "bestes Genom" ist ein einzelner Kandidat. Ein Ensemble der Top-k koennte robuster sein.

**Bereits implementiert:** `get_ensemble(k)` und `forward_ensemble(inputs, k)` mit simplem Averaging existieren in `neuro_evolution.py` (Zeilen 373–392).

Noch fehlende Aufgaben:

- `mode`-Parameter fuer `forward_ensemble(inputs, k, mode='mean')`: zusaetzlich `mode='vote'` fuer binaere Klassifikation (Majority-Vote ueber gerundete Outputs).
- GUI-Anzeige der Ensemble-Fitness: Mittelwert der Top-5-Fitness-Werte im LeftPanel anzeigen (ergaenzt den aktuellen Best-Fitness-Wert).
- Optional: Fitness-gewichtetes Averaging (`mode='weighted'`): Outputs werden nach relativer Fitness der k Genome gewichtet.

Nutzen:

- Robustere Inferenz bei stochastischen Aufgaben.
- Sofort nutzbar ohne zusaetzliches Training.

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

### P2: Adaptive Populationsgroesse

Die Populationsgroesse ist aktuell fix. Dynamische Anpassung koennte Ressourcen effizienter nutzen.

Aufgaben:

- Populationsgroesse automatisch reduzieren, wenn Species sich stark konsolidiert hat (wenig Diversitaet braucht weniger Kandidaten).
- Populationsgroesse erhoehen, wenn viele Species gleichzeitig wachsen (Exploration-Phase).
- Mindest- und Maximalgroesse als Grenzen definieren.
- Groessenveraenderung schrittweise (kein sprunghafter Drop).

Nutzen:

- Weniger Evaluierungen in konvergierten Phasen, mehr Vielfalt in Explorationsphasen.
- Bessere Ressourcennutzung bei langen Runs.

### P2: Warm-Start und Transfer Learning

Trainiertes Bestes Genome oder ganze Population als Startpunkt fuer verwandte Aufgaben nutzen.

Aufgaben:

- `load_population(checkpoint)` als Startpunkt statt zufaelliger Initialisierung.
- Filterung: nur Genome behalten, die auf neuer Aufgabe mindestens Baseline erreichen.
- Optionale Re-Initialisierung der Strategie-Gene (sigma, rates) bei Start.
- Beispiel: CartPole-trainiertes Netz als Startpunkt fuer Acrobot.

Nutzen:

- Verkuerzte Konvergenzzeit bei verwandten Aufgaben.
- Sinnvoll fuer Curriculum Learning (automatischer Wechsel).

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

**Wichtig fuer sequentielle Aufgaben:** Mit einem Tick-basierten Ansatz (ein Token/Zeichen pro Tick, ein "Output-relevant"-Flag als zweiter Input) kann YANE prinzipiell lernen, sich wie ein Sprachmodell zu verhalten — ohne festes Kontextfenster, da der interne State theoretisch ueber beliebig viele Ticks persistiert. Der entscheidende technische Engpass dabei ist Gating: ohne gezielte Schreib-/Vergess-Kontrolle kollabiert der State bei langen Sequenzen numerisch oder das Netz kann keine selektive Retention lernen. Gating (`value = gate * old + (1 - gate) * new`) ist daher der einzige echte technische Blocker fuer diese Aufgabenklasse.

Aufgaben:

- Explizite Memory Nodes als eigener NodeType pruefen.
- Gating-Mechanismen implementieren: `value = gate * old + (1 - gate) * new`, wobei `gate` ein evolvierbarer Parameter oder ein weiterer Node-Output ist.
- Leaky memory als einfachere Variante: `value = alpha * old + new`, mit evolvierbarerm Zerfall `alpha`.
- Mutation fuer `alpha` und Gate-Staerke.
- Reset-Regeln klarer visualisieren.

Nutzen:

- Bessere Sequenz- und Control-Faehigkeiten.
- Stabilere rekurrente Dynamik.
- Ermoeglicht LLM-artiges Verhalten ueber Tick-basierte sequentielle Verarbeitung.

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

### P1: Output-Scale als Strategie-Gen

Input-Nodes haben bereits einen evolvierbaren `input_scale`-Faktor fuer automatische Normalisierung. Output-Nodes haben kein Analoges.

Aufgaben:

- `output_scale: float = 1.0` auf Output-Nodes (wie `input_scale` auf Input-Nodes).
- Evolvierbar: Mutationsrate analog zu `input_scale`.
- Forward-Pass multipliziert Ausgabe-Aktivierung mit `output_scale`.
- GUI: Output-Scales neben Input-Scales anzeigen.

Nutzen:

- Netze koennen Ausgaben automatisch auf erwarteten Wertebereich skalieren.
- Besonders nuetzlich bei Continuous-Control (z. B. Aktion in [-1, 1] oder [0, 100]).
- Symmetrisch zur bestehenden Input-Scale-Mechanik — konsistentes Konzept.

## 4. Optimierung und Hybrid-Training

### P1: Lamarckian Refinement — Restaufgaben

Adaptive Integration implementiert (siehe Erledigte). Offen:

Aufgaben:

- Refinement-Zeit pro Schritt messen und in Statistiken anzeigen.
- Separate Sigma-Strategie fuer Lamarck (unabhaengig von sigma_global).
- Tests fuer Zusammenspiel mit Strukturstagnation.

Nutzen:

- Bessere Diagnose des Lamarck-Overheads.
- Feinere Kontrolle der Schrittweite.

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

### P2: Natural Evolution Strategies (NES) als Lamarck-Alternative

Hill-Climbing aendert Gewichte zufaellig und haelt die bessere Variante. NES schaetzt aus mehreren Perturbationen einen approximierten Fitness-Gradienten und macht gerichtete Schritte.

Aufgaben:

- `_nes_refine()` als Alternative zu `_lamarck_refine()` implementieren.
- k Perturbationen ± epsilon; Gradienten-Schaetzung: `g = sum(f_i * noise_i) / (k * epsilon)`.
- Schrittweite adaptiv (analoges sigma_global-Konzept).
- Benchmark: NES vs. Hill-Climbing auf Acrobot / LunarLander.
- Umschaltbar via `set_lamarck(mode='nes')`.

Nutzen:

- Gerichtete Gewichtsoptimierung statt reinem Zufalls-Climbing.
- Skaliert besser mit Genomgroesse (ein Schritt pro k Evaluierungen statt k Schritte).
- Theoretisch fundierter als blindes Hill-Climbing.

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

**Benchmark-Kandidaten und Erfolgskriterien:**

| Benchmark | Typ | Erfolgskriterium | Timeout | Seeds |
|---|---|---|---|---|
| XOR | deterministisch | fitness ≥ -0.1 | 30 s | 5 |
| basic_multiplication | deterministisch | fitness ≥ -5.0 | 60 s | 5 |
| CartPole-v1 | gym (stochastisch) | mittl. Episoden-Return ≥ 475 | 2 min | 5 |
| Acrobot-v1 | gym (stochastisch) | mittl. Return ≥ -100 | 5 min | 5 |

XOR und basic_multiplication sind rein deterministisch (festes Dataset). Gym-Benchmarks setzen den neuen `NeuroEvolution(seed=...)` voraus (Seeding-Task muss zuerst fertig sein).

Aufgaben:

- `benchmarks/run_suite.py`: laeuft alle Benchmarks und gibt Tabelle (Benchmark, gelöst, mittl. Iterationen, Laufzeit) aus.
- Pro Benchmark: n Runs mit Seeds 0..n-1, mittlere Time-to-Solve und Erfolgsrate messen.
- "Schnelle" CI-Suite: nur XOR + basic_multiplication (< 3 min gesamt).
- "Lange" lokale Suite: alle vier Benchmarks.
- Ergebnisse als JSON in `benchmarks/results/` speichern fuer Vergleiche.

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

## 10. Tests und Qualitaetssicherung

### P1: Testabdeckung ausbauen und Regressionstests etablieren

Aktuell gibt es schon einige Testdateien in `tests/`, aber neue Features werden nicht systematisch durch Tests abgesichert.

Aufgaben:

- Fuer jedes neue Framework-Feature einen mindestens grundlegenden Test schreiben (Richtlinie: kein P0/P1-Feature ohne Test).
- Regressionstests fuer bereits behobene Bugs (Species-Explosion, Threshold-Adjustment, etc.), damit sie nicht zurueckkehren.
- Parametrisierte Tests fuer Edge Cases: leere Population, single-member Species, extreme Fitness-Werte, NaN/Inf-Gewichte, maximale Netzwerkgroesse.
- CI-faehige Test-Suite: `pytest` mit `--durations=10` und Coverage-Report. Schnelle CI-Suite (< 2 min) als separates pytest-Mark (`-m "ci"`) definieren.
- Coverage-Ziel definieren (z.B. >= 80% fuer `core/` und `evolution/`).
- Property-based Testing mit `hypothesis` fuer Invarianten pruefen (z. B. `forward()`-Output hat immer Laenge `n_outputs`, Mutation erzeugt keine isolierten Nodes, Crossover erhaelt Kompatibilitaet nicht beliebig).

Nutzen:

- Refactoring-Risiko sinkt erheblich.
- Neue Contributors koennen sicher aendern.
- Regressionen werden vor dem Merge erkannt.

## 11. Strukturiertes Logging

### P1: Logs mit Kategorie- und Zeitstempel-Ordnern

Aktuell schreibt YANE Logs unsortiert. Ein strukturiertes Log-Verzeichnis erleichtert Debugging und Vergleich von Trainingslaeufen.

**Zielstruktur:**

```
logs/
   cartpole/
       2026-05-24_14-10-00/
           run.log
           best_genome.json
           config.json
       2026-05-24_15-30-00/
           ...
   xor/
       2026-05-24_14-10-00/
           ...
   benchmarks/
       2026-05-24_14-10-00/
           suite_results.json
           ...
   gui/
       2026-05-24_14-10-00/
           ...
```

Aufgaben:

- `util/logger.py` erweitern: `setup_logging(name: str, log_root: str | None = None)` erstellt `<projekt-root>/logs/<name>/<timestamp>/` und gibt den Pfad zurueck. Der `log_root` ist **immer relativ zum YANE-Projektverzeichnis**, nicht zum Home-Verzeichnis oder einem systemweiten tmp-Pfad. Default `None` → `logs/` im Projekt-Root. Ein abweichender Pfad ist nur ueber explizite Konfiguration moeglich.
- Timestamp-Format: `YYYY-MM-DD_HH-MM-SS` (ISO 8601-kompatibel, dateisystem-sicher).
- Automatische Kategorie-Erkennung: In `train()` und `NeuroEvolution` wird `name` aus der Fitnessfunktion oder einem optionalen `run_name`-Parameter abgeleitet; GUI nutzt `"gui"`; Benchmark-Suite nutzt `"benchmarks"`.
- Pro Run: `run.log` (Hauptlog), `config.json` (alle NeuroEvolution-Einstellungen serialisiert), `best_genome.json` (Bestes Genom am Ende des Runs).
- Fitness-Historie als `fitness_history.csv` (Iteration, Bestfitness, Meanfitness, Medianfitness, IQR) — baut auf der bestehenden `population_memory_info()` auf.
- Alte Logs automatisch aufrauemen: `max_log_dirs` pro Kategorie (Default 20), aelteste werden geloescht.
- GUI-Konfiguration: Log-Root-Pfad (relativ zum Projektverzeichnis) und Auto-Cleanup im Settings-Tab einstellbar. Standard: `logs/` im YANE-Projektordner — kein versteckter Pfad in `~/.yane/` oder `/tmp/`.
- `log_info` / `log_warning` / `log_error` in API-Endpunkten nutzen, sodass Server-Logs in `logs/api/` landen.

Nutzen:

- Trainingslaeufe sind dauerhaft und geordnet nachvollziehbar — alle Logs direkt im Projektordner, kein Suchen in versteckten Systemverzeichnissen.
- Kein manuelles Einrichten von Log-Verzeichnissen mehr noetig.
- Schnelles Auffinden des relevanten Runs ueber Kategorie + Zeitstempel.
- Automatisches Aufrauemen verhindert volle Platten bei langen Experiment-Serien.

## 12. Moegliche grosse Entwicklungsphasen

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

## 13. Erste konkrete TODO-Liste

- [x] Automatische Effizienzbewertung in der Elternauswahl anwenden.
- [x] Bewertungszeit in den GUI-Worker-Pfaden messen.
- [x] Einheitliche Evaluation-Statistiken (mean/median/p95/max Eval-Zeit, Offspring-Zähler, Topologie-Historie).
- [x] GUI-Diagnostics fuer Eval-Zeit und Offspring-Zähler.
- [x] Elitismus explizit machen (elite_count, species_elite_count, set_elitism()).
- [x] Fitness-Sanitizing zentralisieren (nan/inf/clipping, Diagnose-Zähler).
- [x] Mehrfachbewertung pro Genom (set_multi_eval, mean/median/min, sigma-Penalty).
- [x] Species-Explosion beheben (aktives Mergen + Dead-band-Hysterese).
- [x] Connection Enable/Disable (reversibles Deaktivieren, Forward-Filter, forward-Pfad compile-time).
- [x] Adaptive Strukturmutation (weight-biased remove, Spike-Mutation, Rewiring).
- [x] Adaptives Lamarckian Refinement (Top-K, stagnationsbasiert, 0→max_steps).
- [x] Raten des besten Genoms in GUI anzeigen (sigma_global, Struktur-Raten, Weight/Bias/Activation-Raten, Strategy-Genes).
- [x] Inkrementelle Speziation (_assign_species von O(pop) auf O(0) im Steady-State).
- [x] Debug-Tab mit Live-Log, Toggle und Clipboard-Kopie.
- [x] Species-Hard-Cap gegen Pop-Overflow bei extremer Stagnation.
- [x] Threshold-Adjustment-Diagnostics (dbg_last_adj_n, dbg_adj_count) im Log und GUI.
- [x] UI: Collapsible Groups im LeftPanel, Uebersetzung auf Englisch, Visual Polish.
- [ ] `EvaluationResult`-Objekt einfuehren: `genome`, `fitness: float`, `elapsed_ms: float`, `n_lamarck_steps: int`, `stopped_early: bool`, `raw_fitnesses: list[float]` (bei Multi-Eval die Einzelwerte vor Aggregation). Ersetzt die aktuellen `(Genome, float, float)`-Tupel in Worker-Pfaden.
- [ ] Seed-Parameter fuer `NeuroEvolution` einfuehren.
- [ ] Checkpoint-Speicherung fuer Population und InnovationTracker.
- [ ] Benchmark-Suite mit mehreren Seeds erstellen.
- [ ] Durchschnittliche Mutationsraten populationsweit ausgeben (Ergaenzung zu Best-Genome-Raten).
- [ ] Mutationsraten-Historie in GUI plotten.
- [ ] API-`/configure` um wichtige Konfigurationsparameter erweitern.
- [ ] Fitness-Shaping (Rank-basierte Transformation vor Selektion) einfuehren.
- [ ] Convergence Detection: `set_convergence_stop(fitness_spread_eps)`, `set_max_evaluations(n)`, `on_stop`-Callback in `train()`. (`min_fitness` + `max_iterations` bereits vorhanden.)
- [ ] Interspecies Crossover (kleiner Anteil, ~5%) implementieren.
- [ ] Output-Scale als Strategie-Gen auf Output-Nodes (analog zu input_scale).
- [ ] Ensemble-Inferenz vervollstaendigen: `mode`-Parameter fuer `forward_ensemble` (vote, weighted), GUI-Anzeige der Top-5-Durchschnittsfitness. (`get_ensemble` + Averaging bereits implementiert.)
- [ ] Early Stopping pro Genome bei schlechter Teilperformance: Generator-Protokoll, sign-unabhaengiger Abbruch wenn `partial_fitness < best_fitness - abs(best_fitness) * factor` (default 1.0), mit Hochrechnung fuer Dataset-Summen.
- [ ] Strukturiertes Logging: `logs/<kategorie>/<timestamp>/`-Struktur mit `run.log`, `config.json`, `best_genome.json`, `fitness_history.csv`; Auto-Cleanup pro Kategorie.
- [ ] Testabdeckung ausbauen: Regressionstests fuer Bugfixes, parametrisierte Edge-Case-Tests, CI-Suite (`-m "ci"`), Coverage-Ziel >= 80% fuer `core/` und `evolution/`.

---

## Erledigte Aufgaben

### Automatische Effizienz in der Elternauswahl

YANE misst Bewertungszeiten in `train()` und in den GUI-Worker-Pfaden. Die Rohfitness bleibt unveraendert; Effizienz wird als eigene Variable gefuehrt und wirkt dynamisch nur auf die Elternauswahl. Bei Stagnation sinkt die Effizienz-Relevanz bis auf `0`.

### Einheitliche Evaluation-Statistiken

Pro Trainingsschritt werden jetzt erfasst: mean/median/p95/max der Eval-Zeiten ueber alle evaluierten Genome, kumulative Offspring-Zaehler (crossover / mutation / diversity injection), sowie eine Topologie-Historie des besten Genoms bei jeder Fitness-Verbesserung. Alle Werte fliessen via `population_memory_info()` in die GUI (zwei neue Sektionen im LeftPanel).

### Elitismus explizit machen

`elite_count` (globale Top-k) und `species_elite_count` (bestes je Species) sind jetzt konfigurierbar via `set_elitism()`. Single-member-Species-Champions sind ebenfalls geschützt. Eliten werden nie gelöscht, aber weiterhin als Elternteile genutzt — nur Kopien mutieren. Defaults entsprechen dem bisherigen Verhalten.

### Fitness-Sanitizing zentralisieren

Zentrale `_apply_sanitize()`-Funktion in `NeuroEvolution`. Behandelt `nan` und `inf` defensiv (Fallback-Wert konfigurierbar), optionales Fitness-Clipping (clip_low, clip_high), Diagnose-Zaehler fuer invalide und geclippte Werte (`n_invalid_fitness`, `n_clipped_fitness`). Konfigurierbar via `set_fitness_sanitizing()`.

### Mehrfachbewertung pro Genom

`set_multi_eval(n, aggregation, sigma_penalty)` erlaubt mehrere Episoden pro Genom. Aggregation: mean / median / min. Sigma-Penalty subtrahiert ein Vielfaches der Standardabweichung. Automatische Worker-Anpassung beruecksichtigt den Mehraufwand. GUI konfigurierbar.

### Species-Explosion beheben

Die "try last species first"-Optimierung in `_assign_species()` verhinderte, dass ein steigender Kompatibilitaets-Threshold Species zusammenfuehren konnte — Genome landeten immer in ihrer alten Species. Fix: Aktives Mergen von Species-Paaren nach der Zuweisung (wenn zwei Repraesentanten naeher als der Threshold liegen), kombiniert mit Dead-band ±1 in der Threshold-Anpassung (verhindert sofortiges Rueckpendeln). Species-Anzahl stabilisiert sich jetzt bei ~Ziel statt bei 36+ einzufrieren.

### Connection Enable/Disable statt Loeschen

`enabled`-Flag auf `Connection`. Deaktivierte Verbindungen bleiben im Genom (reversibel), werden aber im Forward-Pass uebergangen. Compile-time-Filter in allen Forward-Pfaden (acyclisch: Snapshot bei Compile, kein Runtime-Overhead; cyclisch: `if conn.enabled` in `fire()`). Selbst-adaptive Mutations-Raten (`mutation_disable_connection`, `mutation_enable_connection`, initial 3% um Add/Remove-Balance nicht zu stoeren). Crossover und Pickle unterstuetzen das Flag.

### Adaptive Strukturmutation (A, 4, 5)

Drei neue Mutationsmechanismen, alle selbst-adaptiv:
- **A — Gewichtsbasiertes remove_connection**: Soft-min Sampling (Wahrscheinlichkeit ~ 1/|w|) bevorzugt das Entfernen unwichtiger Verbindungen.
- **4 — Spike-Mutation**: Jede Connection kann ihr Gewicht komplett neu initialisieren (N(0, sigma_global)) statt nur zu perturbieren. Rate adaptiert sich via `rate_mutation_rate`.
- **5 — Rewiring**: Atomisches Remove-dann-Add; Netzgroesse stabil, Topologie wird exploriert. Kein `_STRUCT_FLOOR` — rein selbst-adaptiv.

### Adaptives Lamarckian Refinement

Lamarck feuert jetzt automatisch ohne manuelle Konfiguration. Anzahl der Hill-Climbing-Schritte skaliert linear mit Stagnation (0 bei gutem Fortschritt → max_steps bei voller Stagnation). Nur Top-20% des evaluierten Pools werden verfeinert. Expliziter Modus (`set_lamarck(n_steps>0`) bleibt fuer Override-Faelle erhalten. Benchmark Acrobot: 2x schneller geloest, bessere Endfitness, 5% Eval-Overhead.

### Mutationsraten des besten Genoms in der GUI

Im LeftPanel werden die selbst-adaptiven Raten des aktuell besten Genoms angezeigt: `sigma_global` (globale Schrittweite), die vier Struktur-Raten (Add/Remove Node, Add/Remove Connection) sowie populationsdurchschnittliche Weight-shift-rate, Weight-delta, Bias-rate, Bias-delta und Activation-Wechsel-rate. Alle Werte stehen in collapsible Groups ("Structural rates", "Strategy genes", "Weight / Node rates").

### Inkrementelle Speziation

`_assign_species()` laeuft jetzt im Steady-State in O(0) statt O(pop): Jedes neu evaluierte Genom cached seine letzte Species und wird beim naechsten Aufruf direkt dort eingesetzt, wenn die Kompatibilitaet noch stimmt. Ein periodischer Vollscan (jede k-te Generation) korrigiert Drift. Spart signifikant CPU in grossen Populationen.

### Debug-Tab mit Live-Log

Neuer "Debug"-Tab in der GUI: zeilenweise Snapshot aller `population_memory_info()`-Werte alle 0.5 s, Toggle zum Pausieren, Clipboard-Kopie des vollstaendigen Logs. Ermoeglicht tieferes Debugging ohne Print-Statements oder externen Logger.

### Species-Hard-Cap und Stabilitaetsfixes

Mehrere aufeinanderfolgende Fixes fuer Randfaelle bei Stagnation:
- **Hard-Cap**: Species-Anzahl wird hart nach oben begrenzt, damit Pop-Overflow bei extremer Stagnation und vielen kleinen Species ausbleibt.
- **Threshold-Freeze-Fix**: Threshold wurde bei kleinem `target_species` nicht korrekt angepasst; Schutz wird jetzt nur fuer Species mit mehr als einem Mitglied angewendet.
- **Threshold-Schritt-Kalibrierung**: Schrittgroesse auf per-Evaluation-Frequenz kalibriert (nicht per-Schritt), damit Threshold-Anpassung bei verschiedenen Populationsgroessen stabil bleibt.
- **Dead-Band entfernt**: Hysterese erwies sich als schaedlich; Threshold-Anpassung laeuft jetzt ohne Dead-Band.
- **Diagnostics**: `dbg_last_adj_n` und `dbg_adj_count` in `population_memory_info()` und im Debug-Tab sichtbar.

### UI-Verbesserungen (Collapsible Groups, Englisch, Visual Polish)

- **Collapsible Groups im LeftPanel**: Alle Diagnostik-Sektionen koennen ein- und ausgeklappt werden; Zustand bleibt erhalten.
- **Vollstaendige Uebersetzung auf Englisch**: GUI-Texte, Labels und Tooltips durchgehend auf Englisch umgestellt.
- **Visual Polish**: Chart-Ticks, Δ-Styling, Canvas-Legende, Progress-Form und Tab-Indikator verbessert; Inspect-Tab-Notification-Dot erscheint nur bei neuem Bestwert.
