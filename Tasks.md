# Tasks: YANE staerker machen

Diese Datei ist die aktuelle Roadmap fuer YANE. Offene und neue Tasks stehen
oben. Abgeschlossene Arbeit ist weiter unten nur noch kompakt zusammengefasst.

## Status

**Aktueller Stand:** Die bisherigen Roadmap-Phasen sind abgeschlossen.

- Core-Evolution, Speciation, Mutation, Worker-Pipeline, GUI, API, Logging und Checkpoints sind implementiert.
- Multi-Objective, Quality Diversity, CMA-ES, Backprop-/Matrix-Bausteine, Presets, Benchmark-Gates und GUI-Smoke-Tests sind implementiert.
- Naechster Schwerpunkt: vorhandene Features adaptiv machen und in der GUI eindeutig sichtbar steuern.
- Letzter kompletter Testlauf: `595 passed`.

## Legende

- `P0`: hoher Hebel, nah am aktuellen Code, direkt nuetzlich
- `P1`: wichtiger Ausbau, mittlerer Aufwand
- `P2`: experimentell, Forschungsarbeit oder groesserer Umbau
- ✅: erledigt
- ⚡: teilweise erledigt
- 🔲: offen

---

## Offene Tasks: Adaptive YANE

### P0 🔲 Adaptive Control Layer einfuehren

YANE hat viele starke Einzelfunktionen. Der naechste grosse Schritt ist eine
gemeinsame adaptive Steuerung, die diese Funktionen anhand von Stagnation,
Diversity, Fitness-Trend, Kosten und Species-Zustand automatisch dosiert.

Aufgaben:

- Zentrale `AdaptiveController`-Komponente entwerfen.
- Einheitliche Signale definieren: Fitness-Trend, Plateau, Diversity, Species-Stagnation, Evaluation-Kosten, Best-Genome-Komplexitaet.
- Gemeinsames Policy-Format fuer adaptive Features definieren: `off`, `fixed`, `adaptive`, `auto`.
- Diagnostics fuer Policy-Entscheidungen sammeln: Grund, alte Rate, neue Rate, betroffene Species, Trigger-Signal.
- API- und Checkpoint-Kompatibilitaet fuer adaptive Policy-State sichern.

Nutzen:

- Adaptive Faehigkeiten werden ein Systemmerkmal statt einzelner Spezialfaelle.
- Neue adaptive Features koennen spaeter konsistent angeschlossen werden.

### P0 🔲 Lamarck-Modi adaptiv vereinheitlichen

In der GUI gibt es bereits `Adaptiv`, aber der Modus wirkt aktuell wie eine
Lamarck-Hill-Climbing-Option. NES, SA und CMA-ES sollten ebenfalls klar als
adaptive Varianten steuerbar sein.

Aufgaben:

- Lamarck-Modell klaeren: Optimierer `Hill-Climb`, `NES`, `SA`, `CMA-ES`; Zeitplan `aus`, `explizit`, `adaptiv`.
- Adaptive Varianten fuer NES, SA und CMA-ES implementieren oder vorhandene Pfade eindeutig anbinden.
- Per-Species-Entscheidung erlauben: manche Species bekommen lokale Suche, andere nur Mutation.
- Kostenbudget einfuehren: adaptive lokale Suche darf nicht unkontrolliert Evaluationen verbrennen.
- Diagnostics erweitern: Modus, Optimierer, Trigger, Schritte, Kosten, Verbesserung pro Optimierer.

Nutzen:

- Der Begriff `Adaptiv` ist nicht mehr nur an einen Lamarck-Spezialfall gekoppelt.
- Nutzer sehen klar, welche lokale Suche wann und warum aktiv war.

### P0 🔲 GUI fuer adaptive Features eindeutig machen

Adaptive Optionen muessen in der GUI sichtbar, unterscheidbar und nachvollziehbar
sein. Aktuell ist nicht immer klar, ob `Adaptiv` nur Lamarck betrifft oder ein
allgemeines Automatikverhalten meint.

Aufgaben:

- Eigene GUI-Sektion `Adaptive Control` bauen.
- Fuer jedes adaptive Feature denselben UI-Aufbau verwenden: Modus, Min/Max, Budget, aktueller Wert, letzter Trigger.
- Lamarck-GUI in zwei Controls splitten: `Optimierer` und `Zeitplan`.
- Live-Anzeige fuer adaptive Entscheidungen ergaenzen: Warum wurde etwas erhoeht, gesenkt oder deaktiviert?
- Tooltips und Labels so formulieren, dass `Adaptiv` nicht mit `Explizit` oder `Auto-Preset` verwechselt wird.
- Presets fuer adaptive Profile ergaenzen: konservativ, balanciert, aggressiv, analysefreundlich.

Nutzen:

- YANE wird als adaptives System bedienbar, nicht als Sammlung versteckter Schalter.
- GUI-Runs lassen sich besser erklaeren und debuggen.

### P0 🔲 Interspecies-Crossover adaptiv machen

Interspecies-Crossover ist aktuell als feste Rate steuerbar. Sinnvoller waere
eine adaptive Rate, die bei Stagnation steigt und bei zu viel Instabilitaet oder
Diversity-Verlust wieder sinkt.

Aufgaben:

- Adaptive Interspecies-Policy implementieren: Basisrate, Min/Max, Ramp-up, Cooldown.
- Trigger definieren: Species-Stagnation, globales Plateau, geringe Novelty, zu starke Species-Isolation.
- Schutzregeln einbauen: Eliten bewahren, inkompatible Eltern meiden, Rate bei Fitness-Einbruch senken.
- Diagnostics ergaenzen: aktuelle Rate, Cross-Species-Erfolg, Nachkommen-Fitness, verworfene Paarungen.
- GUI-Control ergaenzen: `Aus`, `Fix`, `Adaptiv`; mit Live-Rate und letzter Entscheidung.

Nutzen:

- YANE kann genetische Inseln gezielt verbinden, ohne permanent Struktur zu verwischen.
- Crossover wird vom Ablationsschalter zum aktiven Suchinstrument.

### P0 🔲 Adaptive Operator-Scheduler fuer Mutation, QD und Pruning

Viele Operatoren existieren bereits, aber ihre Aktivierung ist noch zu oft fest
oder lokal verteilt. Ein Scheduler sollte entscheiden, wann welcher Operator mehr
oder weniger Druck bekommt.

Aufgaben:

- Mutation-Operatoren adaptiv gewichten: Add-Node, Add-Connection, Rewire, Disable/Enable, Spike, Remove.
- Novelty/QD-Druck adaptiv steuern: mehr Exploration bei Plateau, weniger bei Zielnaehe.
- Pruning adaptiv dosieren: Komplexitaetsdruck erhoehen, wenn Fitness stagniert und Netzgroesse waechst.
- Population Size und Species Target in dieselbe Policy-Sicht integrieren.
- Operator-Erfolg pro Species und global tracken.

Nutzen:

- YANE reagiert besser auf unterschiedliche Phasen eines Laufs.
- Mehr Exploration, wenn sie gebraucht wird; mehr Verdichtung, wenn Netzwerke ausufern.

### P1 🔲 Adaptive Benchmark- und Ablation-Suite

Adaptive Features brauchen Benchmarks, die nicht nur Endfitness messen, sondern
auch Kosten, Stabilitaet und Entscheidungsqualitaet.

Aufgaben:

- Benchmark-Matrix bauen: fixed vs adaptive fuer Lamarck, Interspecies-Crossover, Mutation-Scheduler, QD-Druck.
- Kostenmetriken aufnehmen: Evaluationen, Lamarck-Schritte, Wall-Time, Fitnessgewinn pro Zusatzkosten.
- Ablations fuer jeden adaptiven Trigger ergaenzen.
- Gates fuer adaptive Profile kalibrieren.
- Markdown-Report um adaptive Decision-Summary erweitern.

Nutzen:

- Adaptive Features werden messbar statt nur plausibel.
- Schlechte Automatik laesst sich schnell erkennen.

### P1 🔲 Matrix-Forward automatisch im Training nutzen

`MatrixGenome`, `MatrixForwardCache` und Batch-Helfer existieren. Der naechste
Schritt ist die kontrollierte Integration in echte Evaluationspfade.

Aufgaben:

- Kompatible DAG-Subpopulationen waehrend Dataset-Evaluation erkennen.
- Matrixexport nur nutzen, wenn Batchgroesse und Topologie es rechtfertigen.
- Fallback auf `Genome.forward()` bei Zyklen, Memory oder unsupported Activation.
- Benchmark: sampleweises `forward()`, `forward_batch()`, Matrix-Forward.
- Diagnostics fuer Matrix-Cache-Hits/Misses ergaenzen.

Nutzen:

- Groessere supervised Aufgaben werden realistischer.
- GPU-/CuPy-Pfad bekommt einen praktischen Einstiegspunkt.

### P1 🔲 Preset-System fuer adaptive Profile erweitern

Presets existieren, aber sollten adaptive Policies als erstklassige Konfiguration
speichern koennen.

Aufgaben:

- Preset-Schema mit Version, Validierung und `adaptive_policies` einfuehren.
- Presets fuer konkrete Beispiele ergaenzen: XOR, Regression 2->2, CartPole, Acrobot.
- Adaptive Profile speichern: konservativ, balanciert, aggressiv, stabilitaetsorientiert.
- GUI-Preset-Editor verbessern: Name, Beschreibung, Save/Overwrite, adaptive Summary.
- Preset-Name und adaptive Policy-State in Logs und Checkpoints mitschreiben.

Nutzen:

- Adaptive Experimente werden reproduzierbar.
- GUI-Konfiguration wird schneller und weniger fehleranfaellig.

### P1 🔲 Stabilitaetslauf fuer GUI-Regression 2->2

Die offenen Crash-Logs deuten darauf hin, dass laengere GUI-Laeufe weiterhin
gesondert beobachtet werden sollten.

Aufgaben:

- Crash-Logs unter `logs/gui/Regression 2->2/...` auswerten.
- Reproduzierbaren GUI-Run fuer Regression 2->2 definieren.
- Headless/CLI-nahe Reproduktion bauen, falls der Crash nicht rein Qt-seitig ist.
- Crash-State-Snapshots und `run.log` mit Worker-Modus, Preset und adaptiven Feature-Flags korrelieren.
- Fix oder Guard implementieren, wenn eine konkrete Ursache sichtbar wird.

Nutzen:

- Erhoeht Vertrauen in lange GUI-Trainingslaeufe.
- Verhindert, dass neue adaptive Features alte Stabilitaetsprobleme ueberdecken.

### P1 🔲 Release-Cleanup und API-Konsistenz

Nach vielen schnellen Feature-Adds sollte die oeffentliche Oberflaeche einmal
geglättet werden.

Aufgaben:

- Public API auf Namenskonsistenz pruefen (`set_*`, `get_*`, Diagnostics-Keys).
- Adaptive Parameter einheitlich benennen: `*_mode`, `*_policy`, `*_min`, `*_max`, `*_budget`.
- README-Beispiele gegen aktuellen Code ausfuehren.
- Veraltete Kommentare/Docstrings entfernen oder aktualisieren.
- Importpfade und `__all__` fuer neue Module pruefen.
- Minimalen Release-Abschnitt in README ergaenzen.

Nutzen:

- YANE wirkt nach aussen weniger wie ein Forschungsnotizbuch.
- Neue Nutzer finden schneller den richtigen Einstieg.

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

### P1 🔲 Checkpoint-Format langfristig haerten

Checkpoint v2 mit Migration und Metadata-Sidecar ist implementiert. Langfristig
sollten Payloads robuster und besser inspizierbar werden.

Aufgaben:

- Kleine Fixture-Checkpoints fuer v1/v2 in Tests versionieren.
- JSON-Metadaten in GUI/API sichtbar machen.
- Optional: getrennte Speicherung von Config, Tracker, Population, QD-Archiv.
- Import-Warnungen fuer fehlende Descriptor-Callbacks in GUI anzeigen.
- Dokumentieren, welche Teile Pickle bleiben und warum.

Nutzen:

- Alte Laeufe bleiben langfristig nutzbar.
- Checkpoints werden besser debugbar.

### P1 🔲 Pareto- und MAP-Elites-Visualisierung polishen

Basisplots existieren. Fuer echte Analyse fehlen noch Interaktion und Export aus
der GUI.

Aufgaben:

- Hover/Tooltip fuer Pareto-Punkte und MAP-Elites-Zellen.
- Export-Buttons fuer QD-Archiv JSON/CSV in der GUI.
- Farbschema fuer Fitness/Complexity klarer trennen.
- Pareto-Plotachsen beschriften und skalieren.
- Optional: Klick auf Zelle/Punkt zeigt Genome im Inspect-Tab.

Nutzen:

- QD- und Multi-Objective-Laeufe werden besser interpretierbar.
- GUI wird als Analysewerkzeug staerker.

### P2 🔲 Evolvierbare Descriptor-Gewichte

Descriptor-Registry und Fitness-Komponenten sind vorhanden. Evolvierbare oder
adaptive Gewichtung ist noch Forschungsarbeit.

Aufgaben:

- Gewichtshistorie fuer Fitness-Komponenten speichern.
- Adaptive Gewichtung bei Stagnation testen.
- Descriptor-Kombinationen per Ablation benchmarken.
- Mechanismus gegen Descriptor-Collapse entwerfen.

Nutzen:

- Weniger manuelles Descriptor-Design.
- Bessere Archive fuer unbekannte Aufgaben.

### P2 🔲 Meta-adaptive Policies evolvieren

Wenn die handgebauten adaptiven Policies stabil sind, koennen ihre Parameter
selbst zum Evolutionsobjekt werden.

Aufgaben:

- Policy-Gene fuer Operator-Scheduler, Lamarck-Budget und Interspecies-Rate modellieren.
- Policy-Gene pro Species und global vergleichen.
- Sicherheitsgrenzen fuer extreme Policies einbauen.
- Meta-Ablation: feste Policy vs handadaptive Policy vs evolvierte Policy.

Nutzen:

- YANE lernt nicht nur Netzwerke, sondern auch bessere Suchstrategien.
- Gute Auto-Konfiguration kann ueber Aufgaben hinweg uebertragen werden.

### P2 🔲 Modul-Crossover und Modulbibliothek

Module koennen erkannt und dupliziert werden. Wiederverwendung ueber Genome und
Laeufe hinweg ist noch offen.

Aufgaben:

- Modul-Crossover zwischen kompatiblen Subgraphen erforschen.
- Gute Module in einer Bibliothek speichern.
- Mutationsoperator: Modul aus Bibliothek einfuegen.
- Diagnostics: Modulhaeufigkeit und Wiederverwendungsrate.

Nutzen:

- Evolution kann gefundene Teilfunktionen besser wiederverwenden.
- Skalierung auf komplexere Aufgaben koennte stabiler werden.

### P2 🔲 Evolvierbare CPPNs

Indirekte Kodierung kann Verbindungen aus Koordinaten erzeugen, aber die
CPPN-Funktion selbst ist noch nicht evolvierbar.

Aufgaben:

- CPPN-Genome als eigene kleine Netzklasse oder normales YANE-Genome modellieren.
- Weight-Pattern aus CPPN-Outputs erzeugen.
- HyperNEAT-artige Substrate fuer Inputs/Outputs definieren.
- Benchmark gegen direkte Kodierung auf regelmaessigen Aufgaben.

Nutzen:

- Bessere Skalierung bei grossen, geometrisch strukturierten Netzen.
- Interessant fuer Bild- und Steuerungsaufgaben.

---

## Naechste empfohlene Reihenfolge

1. **P0 Adaptive Control Layer einfuehren**
2. **P0 Lamarck-Modi adaptiv vereinheitlichen**
3. **P0 GUI fuer adaptive Features eindeutig machen**
4. **P0 Interspecies-Crossover adaptiv machen**
5. **P0 Adaptive Operator-Scheduler fuer Mutation, QD und Pruning**
6. **P1 Adaptive Benchmark- und Ablation-Suite**
7. **P1 Stabilitaetslauf fuer GUI-Regression 2->2**

---

## Abgeschlossene Capability-Gruppen

### Fitness, Evaluation und Training ✅

- Curriculum Learning mit Stage-Wechsel und Diagnostics.
- Fitness-Landscape-Diagnostics: IQR, Histogramm, Plateau-Ratio, Jump-Rate.
- Multi-eval mit mean/median/min und Sigma-Penalty.
- Early-Stopping-Protokoll fuer Generator-Fitnessfunktionen.
- Fitness-Shaping und zentrale Fitness-Sanitizing-Pipeline.
- Multi-Objective-Fitness inklusive Pareto-Rang, Crowding-Distance und Vector-Aggregation.

### Evolution und Suchstrategie ✅

- Robuste Speciation mit Hard-Cap, dynamischer Ziel-Species und alternativer Metrik.
- Species-budgetierte Reproduktion mit Spawn-Credits, Jugendbonus und Species-internem Tournament.
- Adaptive Strukturmutation: weight-biased remove, Spike, Rewiring, Enable/Disable.
- Mutation-Success-Tracking und per-Species-Mutation-Biases.
- Adaptive Populationsgroesse.
- Quality Diversity / MAP-Elites mit Archiv, Diagnostics, GUI-Control und Export.
- Coevolution-Bausteine: Hall-of-Fame, competitive fitness, mixed opponents.
- Async Evaluation Queue fuer lokale Future-basierte Batch-Auswertung.

### Netzwerkmodell ✅

- Aktive/inaktive Strukturdiagnostik und gezieltes Pruning.
- Normalizer-Framework mit Scale/MinMax/ZScore/Clip/RunningStats.
- Memory Nodes mit Leaky Memory, statischem und dynamischem Gating.
- Modulare Subnetze erkennen und duplizieren.
- Indirekte Kodierung / CPPN-artige Connection-Generierung.
- Matrixexport, MatrixForwardCache, kompatible Batch-Auswertung und optionaler CuPy-Pfad.

### Optimierung und Hybrid-Training ✅

- Lamarckian Refinement adaptiv, explizit und per Species.
- Evolvierbares `genome.lamarck_sigma`.
- Lokale Optimierer: Hill-Climbing, Simulated Annealing, NES, CMA-ES.
- Optionaler Backprop-Hook fuer kompatible feed-forward Genome.
- Benchmark zum Vergleich von Lamarck-Modi.

### Performance und Parallelisierung ✅

- Einheitliche Evaluation-Pipeline fuer train/API/GUI.
- GUI sequential, ThreadPool und Multiprocessing-Pfade.
- Forward-Microbenchmarks.
- Vektorisierter `forward_batch()`.
- Schlankerer Pickle-State fuer Genome.
- Serialization-/Matrixexport-Profiling-Skript.

### Robustheit und Reproduzierbarkeit ✅

- Seed-Handling fuer Python und NumPy.
- Checkpoints mit Migration v1->v2 und JSON-Metadaten-Sidecar.
- Warm-Start / Transfer Learning mit I/O-Adaption.
- Weight-/Bias-Clipping und Output-Sanitizing.
- ResourceGuard und Memory-Trim.
- Strukturierte Logs mit Config, Fitness-History, Best-Genome und Crash-State.

### GUI, API und Bedienbarkeit ✅

- PySide6-GUI mit Training, Inspect, API-Server und Debug-Tab.
- NetworkCanvas, WeightHistogram, FitnessChart, SpeciesChart, MutationRateChart.
- ParetoScatter und MAP-Elites-Heatmap.
- Advanced-Controls fuer Fitness-Shaping, Ablations, Lamarck-Modi, Multi-Objective, QD, Elitismus und mehr.
- GUI-Smoke- und Screenshot-Tests.
- FastAPI-Konfiguration, Population-Endpoints, Diagnostics, Best-Export, Checkpoint-Endpunkte.
- Preset-System fuer GUI und API.

### Benchmarking und Dokumentation ✅

- Benchmark-Suite mit Gate-Modus und Markdown-Report.
- Ablation-Schalter und Tests.
- README und technische Dokumentation fuer aktuelle Architektur.
- Entwicklerdokumentation: Genom-Lifecycle, Spawn-Zyklus, Worker-Pfade, Fitness-Konventionen.
