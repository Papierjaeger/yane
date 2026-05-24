# Technische Dokumentation: YANE

Diese Dokumentation beschreibt den aktuellen Stand des Projekts und erklärt, wie das neuronale Netz in YANE aufgebaut ist, wie es rechnet und wie es sich evolutionär weiterentwickelt.

## 1. Architektur

YANE besteht aus fünf Hauptbereichen:

- `core/`: Datenmodell des neuronalen Netzes (`Genome`, `Node`, `Connection`)
- `evolution/`: Mutation, Population, Speziation, Crossover, Innovation Tracking und Novelty Search
- `neuro_evolution.py`: öffentliche API und Trainingsloop
- `gui/`: PySide6-Oberfläche mit Beispielregistry und Visualisierung
- `api/`: FastAPI-Schnittstelle für externe Steuerung

Der zentrale Einstiegspunkt ist `NeuroEvolution`. Diese Klasse erzeugt eine `Population`, wählt Genome zur Bewertung aus, nimmt Fitnesswerte entgegen und steuert optionale Funktionen wie Ressourcenlimits, Effizienzstrafe und Lamarckian Refinement.

## 2. Genom und Netzwerkmodell

Ein `Genome` ist ein vollständiges neuronales Netz.

Es enthält:

- `nodes`: alle Nodes
- `input_nodes`: Eingabeknoten
- `output_nodes`: Ausgabeknoten
- Connections an jedem Node als ausgehende Kanten
- Fitness und Shared Fitness
- Strategie-Gene wie `crossover_prob`, `offspring_factor` und `sigma_global`
- Mutationsobjekte für Struktur und Strategieparameter
- optionale Limits `max_nodes` und `max_connections`
- `allow_memory`: steuert, ob Nodes `persist_value=True` halten dürfen; wird durch `stateful` in `configure()` gesetzt

### 2.1 Nodes

Eine `Node` hat folgende relevante Eigenschaften:

- `type`: `INPUT`, `HIDDEN` oder `OUTPUT`
- `value`: aktueller Aktivierungs-/Akkumulationswert
- `bias`: additiver Bias vor der Aktivierung
- `activation`: Aktivierungsfunktion
- `persist_value`: wenn `True`, bleibt der aktivierte Wert nach dem Feuern erhalten
- `max_triggers`: Schutzlimit für zyklische Netze im BFS-Forward-Modus
- `input_index`: Index des gelesenen Eingabewerts bei Input-Nodes
- `input_scale`: evolvierbarer Skalierungsfaktor für Input-Nodes (Standard `1.0`); multipliziert den Rohwert, damit das Netz Eingaben selbst normalisieren kann
- `connections`: ausgehende gewichtete Verbindungen

Input-Nodes werden bei `configure()` auf `LINEAR` gesetzt, damit Rohwerte unverändert ins Netz gelangen. Output-Nodes starten mit `SIGMOID`, weil `Node` standardmäßig diese Aktivierung verwendet. Aktivierungen können später mutieren.

### 2.2 Connections

Eine `Connection` verbindet einen Quell-Node mit einem Ziel-Node. Technisch liegt die Connection in `source.connections` und enthält:

- `target`: Ziel-Node
- `weight`: Gewicht
- `innovation`: historische Innovationsnummer
- `mutation`: selbstadaptierende Mutationsparameter für das Gewicht

Connections sind gerichtet. Zyklen, Selbstverbindungen und bidirektionale Verbindungen sind erlaubt.

### 2.3 Aktivierungsfunktionen

Aktuell verfügbare Aktivierungen:

- `linear`
- `sigmoid`
- `tanh`
- `relu`
- `binary`
- `leaky_relu`
- `elu`
- `swish`
- `softplus`
- `sine`
- `square`
- `abs`
- `gaussian`
- `cube`
- `cosine`

Mehrere Funktionen enthalten Clipping, damit große Werte nicht unnötig zu Overflow oder `inf` führen.

## 3. Initialisierung

Die Konfiguration erfolgt über:

```python
yane.configure(
    n_inputs,
    n_outputs,
    max_nodes=None,
    max_connections=None,
    n_initial_hidden=0,
    stateful=True,
)
```

Der Ablauf:

1. Ein leeres `Genome` wird erstellt.
2. Für jeden Input wird ein `INPUT`-Node mit `LINEAR`-Aktivierung angelegt.
3. Für jeden Output wird ein `OUTPUT`-Node angelegt.
4. Wenn `stateful=True`, bekommen Output-Nodes initial `persist_value=True`.
5. Wenn `n_initial_hidden > 0`, werden Hidden Nodes angelegt.
6. Bei initialen Hidden Nodes wird Input -> Hidden -> Output vollständig verbunden. Die Verbindungsgewichte werden mit `random.uniform(-1.0, 1.0)` initialisiert.
7. Ohne initiale Hidden Nodes startet das Genom ohne Connections.
8. Eine `Population` wird mit diesem Startgenom und einer Template-Kopie erzeugt.

Wichtig: Das normale Startgenom ist nicht voll verbunden. Die erste vielfältige Generation entsteht später durch `_bootstrap_initial_population()`, wobei zufällige Connections hinzugefügt werden.

## 4. Forward-Ausführung

YANE unterstützt zwei Arten der Netzausführung: vollständiger `forward()` und schrittweiser `tick()`.

### 4.1 `forward()`

`forward(data)` setzt Inputs, propagiert Signale durch das gesamte Netzwerk und gibt eine Liste der Outputwerte zurück.

Beim ersten Aufruf entscheidet YANE, ob das Netz azyklisch ist:

- Wenn kein Zyklus gefunden wird, baut `_build_exec_order()` eine topologische Reihenfolge.
- Danach wird mit `_compile_forward()` eine schnelle Closure erzeugt und für alle weiteren Aufrufe direkt gecacht (`_forward_dispatch`). Ab dem zweiten Aufruf gibt es kein Attribute-Lookup auf die Topologie mehr.
- Wenn ein Zyklus gefunden wird, verwendet das Genom dauerhaft `_bfs_forward()`.

### 4.2 Azyklischer Fast Path

Bei azyklischen Netzen:

1. Output-Nodes werden pro Aufruf auf `0.0` gesetzt.
2. Triviale Input-Nodes (`LINEAR`, Bias `0`, nicht persistent) leiten Werte direkt über ihre Connections weiter, multipliziert mit `input_scale`.
3. Andere Input-Nodes feuern über `fire_simple()`.
4. Hidden- und Output-Nodes feuern in topologischer Reihenfolge.
5. Outputs werden aus `output_nodes` gelesen.

Nicht persistente Hidden Nodes werden nach dem Feuern auf `0.0` zurückgesetzt. Persistente Hidden Nodes behalten ihren aktivierten Wert und können dadurch Gedächtnis abbilden.

### 4.3 Zyklischer BFS Path

Bei zyklischen Netzen:

1. Output-Nodes und nicht persistente Nodes werden zurückgesetzt.
2. Persistente Hidden Nodes behalten ihren Wert.
3. Inputs werden gesetzt und als getriggert markiert.
4. Getriggerte Nodes feuern in Wellen.
5. Pro Node begrenzt `max_triggers`, wie oft sie in einem Forward Pass feuern darf.

Damit können rekurrente Topologien entstehen, ohne dass Endlosschleifen den Lauf blockieren.

### 4.4 `tick()`

`tick()` arbeitet explizit schrittweise:

1. `set_inputs(data)` setzt Inputwerte und markiert Input-Nodes als getriggert.
2. `tick()` feuert alle aktuell getriggerten Nodes einmal.
3. Die Ziel-Nodes werden für den nächsten Tick getriggert.
4. `get_outputs()` liest die aktuellen Outputwerte.

Dieser Modus eignet sich für Simulationen, bei denen Zeit explizit modelliert werden soll.

## 5. Memory und Stateful-Verhalten

Mit `stateful=True` dürfen Nodes `persist_value=True` haben. Dadurch wird der aktivierte Wert nicht gelöscht, sondern in den nächsten Schritt übernommen.

Regeln:

- `genome.reset()` löscht alle Node-Werte und sollte am Episodenstart verwendet werden.
- In Dataset-Beispielen wird meist vor jedem Sample zurückgesetzt, damit einzelne Samples unabhängig bleiben.
- In Sequenz- und Gym-Aufgaben wird am Episodenanfang zurückgesetzt, aber nicht zwischen einzelnen Zeitschritten.
- Mit `stateful=False` setzt `Genome.mutate()` alle Nodes wieder auf `persist_value=False`, selbst wenn eine Mutation Persistenz aktivieren würde.

Memory entsteht also nicht durch eine separate RNN-Schicht, sondern durch persistente Node-Werte und erlaubte zyklische Connections.

## 6. Mutation

Jedes Genom kann Struktur, Gewichte, Nodes und eigene Strategieparameter mutieren.

### 6.1 Strukturmutationen

Strukturmutationen liegen in `evolution/smart_mutation.py`.

`add_node`:

- wählt eine bestehende Connection `A -> B`
- ersetzt sie durch `A -> N -> B`
- die erste Connection behält das alte Gewicht
- die zweite Connection bekommt Gewicht `1.0`
- der neue Hidden Node startet mit `LINEAR` und Bias `0.0`
- dadurch bleibt das Verhalten direkt nach dem Einfügen möglichst ähnlich

`remove_node`:

- entfernt einen zufälligen Hidden Node
- kann Bypass-Connections `A -> B` erzeugen
- Bypass-Gewicht ist `w_AN * w_NB`
- Wahrscheinlichkeit dafür ist `bypass_connection_prob`

`add_connection`:

- verbindet zwei zufällige Nodes
- Zyklen sind erlaubt
- doppelte Connections vom selben Source zum selben Target werden vermieden
- neues Gewicht wird mit `random.gauss(0.0, 0.3)` initialisiert

`remove_connection`:

- wählt bevorzugt Connections mit kleinem absolutem Gewicht aus (Soft-Min-Sampling: Wahrscheinlichkeit ∝ 1 / (|Gewicht| + ε))
- entfernt die gewählte Connection

`rewire_connection`:

- wählt eine Connection aus dem unteren Viertel nach absolutem Gewicht (Soft-Min-Sampling)
- entfernt diese Connection und fügt sofort eine neue zufällige Connection ein
- die Topologiegröße bleibt gleich; die Suchrichtung ändert sich

`disable_connection`:

- deaktiviert eine Connection (Soft-Min-Sampling nach absolutem Gewicht)
- die Connection bleibt im Genom gespeichert, wird aber beim Forward Pass übergangen
- reversibel durch `enable_connection`

`enable_connection`:

- reaktiviert eine zufällig gewählte deaktivierte Connection

### 6.2 Node- und Connection-Mutationen

Nodes mutieren:

- Bias
- Aktivierungsfunktion
- Persistenz
- `max_triggers`
- bei Input-Nodes zusätzlich `input_index` und `input_scale`

Connections mutieren:

- Gewicht

Alle kontinuierlichen Werte werden mit gaußschem Rauschen verändert. Die Schrittweite ist `mutation.value_delta * genome.sigma_global`.

### 6.3 Selbstadaptierende Mutationsraten

`Mutation` enthält:

- `shift_rate`: Wahrscheinlichkeit für Wertverschiebung
- `custom_rate`: Wahrscheinlichkeit für Enum-Wechsel, z. B. Aktivierung
- `bool_rate`: Wahrscheinlichkeit für Boolean-Flip
- `int_rate`: Wahrscheinlichkeit für Integer-Neuziehung
- `rate_mutation_rate`: Wahrscheinlichkeit, dass diese Raten selbst mutieren
- `value_delta`: Schrittweite für Wertmutationen

Die Raten werden vererbt und selbst mutiert. Dadurch sucht die Evolution nicht nur nach Netzwerkgewichten und Topologien, sondern auch nach passenden Mutationsparametern.

Eine Untergrenze `Mutation.MIN_RATE = 0.001` verhindert, dass Mutationen vollständig verschwinden. Für Strukturmutationen existiert zusätzlich ein Floor von `0.01`.

## 7. Crossover und Innovation Numbers

YANE nutzt NEAT-artige Innovation Numbers.

Der `InnovationTracker` vergibt:

- eindeutige IDs für Nodes
- eindeutige IDs für Connections
- stabile IDs für wiederholte Splits derselben Connection

Beim Crossover gilt:

- Der fittere Parent ist führend.
- Matching Genes mit gleicher Innovation Number können von beiden Eltern kommen.
- Disjoint und Excess Genes des fitteren Parents werden behalten.
- Disjoint und Excess Genes des schwächeren Parents werden verworfen.
- Strategie-Gene und Mutationsobjekte werden zufällig von einem der Eltern übernommen.

Dadurch können Topologien gekreuzt werden, ohne Connections nur anhand ihrer Listenposition falsch zuzuordnen.

## 8. Population und Evolution

Die `Population` verwaltet zwei Listen:

- `_unevaluated`: Genome, die noch bewertet werden müssen
- `_evaluated`: Genome mit Fitnesswert

### 8.1 Auswahl und Bewertung

`select_for_evaluation()` gibt das nächste unbewertete Genom zurück. Ist keines vorhanden, wird `_spawn_offspring()` aufgerufen.

`submit(genome, fitness)`:

1. speichert Fitness und Shared Fitness
2. verschiebt das Genom nach `_evaluated`
3. berechnet einen Behavior Descriptor für Novelty Search
4. aktualisiert Stagnationszähler
5. beschneidet die Population, falls sie zu groß wird

### 8.2 Bootstrap der Anfangspopulation

Wenn es noch keine bewerteten Genome gibt und neue Kandidaten benötigt werden, erzeugt `_bootstrap_initial_population()` zufällige Varianten des Template-Genoms.

Dabei werden zufällige Connections hinzugefügt. Die Anzahl liegt zwischen `n_outputs` und `min(n_inputs * n_outputs, max(n_inputs, 50))`.

### 8.3 Selektion

Neue Eltern werden per Tournament Selection gewählt:

- Turniergröße `k = 3`, falls genügend Genome vorhanden sind
- Fitness wird so verschoben, dass negative Fitnesswerte die Gewichtung nicht invertieren
- Auswahlwert enthält Shared Fitness, `offspring_factor`, Novelty-Bonus und Effizienz-Faktor

Formelhaft:

```text
selection_score =
  shifted_shared_fitness
  * offspring_factor
  * (1 + novelty_weight * novelty)
  * efficiency_factor
```

Der `efficiency_factor` ist `1 - efficiency_weight * (1 - efficiency_score)`. Details zur Berechnung stehen in Abschnitt 8.9.

### 8.4 Speziation

Genome werden nach NEAT-Kompatibilität gruppiert.

Kompatibilitätsdistanz:

```text
delta = c1 * E / N + c2 * D / N + c3 * W_bar
```

Mit:

- `E`: Excess Genes
- `D`: Disjoint Genes
- `W_bar`: durchschnittliche Gewichtsdifferenz passender Connections
- `c1 = 1.0`
- `c2 = 1.0`
- `c3 = 0.4`

Der Kompatibilitätsschwellwert startet bei `0.2` und wird automatisch angepasst, um ungefähr die Zielzahl von Species zu halten. Standardziel ist `5`, per `set_target_species(n)` änderbar. Der Schwellwert liegt stets zwischen `0.01` und `1.5`. Um Oszillation zu vermeiden, greift die Anpassung nur, wenn die tatsächliche Anzahl um mehr als `1` vom Ziel abweicht (Dead-Band von ±1).

### 8.5 Fitness Sharing

Innerhalb einer Species wird Fitness geteilt:

```text
shared_fitness = fitness / species_size
```

Das schützt strukturelle Nischen, weil ein einzelnes gutes Genom in einer kleinen Species nicht sofort gegen eine große Gruppe ähnlicher Genome untergeht.

### 8.6 Novelty Search

Für jedes bewertete Genom wird ein Behavior Descriptor berechnet:

1. Es gibt 10 feste Probe-Inputs aus `[-1, 1]`, erzeugt mit Seed `42`.
2. Das Genom wird für jeden Probe-Input zurückgesetzt und per `forward()` ausgewertet.
3. Alle Outputs werden zu einem Vektor zusammengefügt.
4. Novelty ist die mittlere Distanz zu anderen Populationsteilnehmern und einem Archiv.

Novelty wird auf `[0, 1]` normalisiert. Der Bonus steigt bei Stagnation:

```text
novelty_weight = 0.1 + 0.4 * min(1.0, stagnation_count / population_size)
```

Deskriptoren, deren Novelty den Schwellwert `0.3` überschreiten, werden in ein persistentes Archiv aufgenommen (maximal 200 Einträge; älteste Einträge werden bei Überlauf verworfen). Das Archiv verhindert, dass Novelty kollabiert, wenn die Population konvergiert. Die Novelty-Matrix wird alle `max_size / 2` Bewertungen neu berechnet statt bei jedem Submit, um den Rechenaufwand zu begrenzen.

### 8.7 Stagnation und Diversity Injection

Wenn `max_size` Bewertungen lang keine Fitnessverbesserung eintritt, injiziert YANE neue Diversität:

- expandierte Kopien des besten Genoms
- reine Gewichtsmutationen des besten Genoms
- frische Template-Genome mit zufälligen Connections
- stark mutierte Template-Genome

Zusätzlich gibt es strukturelle Stagnation. Wenn die Topologie des besten Genoms sehr lange gleich bleibt (Schwellwert: `3 × population_size` Bewertungen), werden gezielt Strukturvarianten erzeugt. Das ist wichtig, wenn Lamarckian Refinement die Gewichte verbessert, aber die Topologie nicht mehr wächst.

### 8.8 Pruning

Wenn die Population über `max_size` wächst:

- global bestes Genom bleibt geschützt
- Species-Champions aus Species mit mehreren Mitgliedern bleiben geschützt
- schlechteste nicht geschützte Genome werden entfernt
- entfernte Genome werden mit `_clear()` von Referenzen befreit

### 8.9 Automatische Effizienzbewertung

YANE speichert pro Genom optional die Bewertungszeit:

- `eval_time_ms`: Dauer der Fitnessbewertung
- `efficiency_score`: relative Geschwindigkeit innerhalb der evaluierten Population
- `selection_score`: zuletzt berechneter interner Auswahlwert

Der `efficiency_score` wird relativ zur aktuellen Population normalisiert:

```text
1.0 = schnellstes getimtes Genom
0.0 = langsamstes getimtes Genom
```

Genome ohne Timingdaten bleiben neutral bei `1.0`. Die Rohfitness wird dadurch nicht verändert.

Die Relevanz der Effizienz hängt von Stagnation ab:

```text
efficiency_weight = 0.5 * (1 - min(1, stagnation_count / stagnation_threshold))
```

Bei guter Entwicklung ist Effizienz also relevant. Bei Stagnation fällt die Effizienz-Relevanz bis auf `0`, damit langsamere oder größere Strukturvarianten nicht zu früh aus der Elternauswahl gedrängt werden.

Die Elternauswahl verwendet Effizienz als Multiplikator:

```text
efficiency_factor = 1 - efficiency_weight * (1 - efficiency_score)
```

Die bestehende feste `EfficiencyPenalty` bleibt separat verfügbar. Sie reduziert direkt die Fitness, wenn `set_efficiency_penalty(max_ms, penalty_per_ms)` gesetzt wurde. Die automatische Effizienzbewertung ist dagegen eine eigene Variable und wirkt nur in der Elternauswahl.

### 8.10 Detaillierter Spawning-Ablauf

Jedes Mal, wenn `select_for_evaluation()` aufgerufen wird und keine unbewerteten Genome vorhanden sind, erzeugt `_spawn_offspring()` ein neues Kandidatengenom. Dabei greift eine von vier Logiken.

**Erste Generation:**

Solange noch kein Genom bewertet wurde, füllt `_bootstrap_initial_population()` die Population. Das Template-Genom wird `max_size`-mal kopiert. Jede Kopie erhält eine zufällige Anzahl neuer Connections, gleichmäßig gezogen aus dem Bereich:

```text
[n_outputs, min(n_inputs × n_outputs, max(n_inputs, 50))]
```

**Diversitätsinjektion bei Fitness-Stagnation:**

Wenn seit `max_size` Bewertungen keine Fitnessverbesserung eingetreten ist, injiziert `_inject_fresh_genome()` ein neues Genom. Vier Strategien mit gleicher Wahrscheinlichkeit:

| Strategie | Beschreibung |
|---|---|
| 0 | Kopie des besten Genoms → 1–3 neue Connections → 1–3 weitere Mutationen |
| 1 | Kopie des besten Genoms → alle Gewichte mit gaußschem Rauschen (`sigma_global`) stören |
| 2 | Frisches Template → zufällige Connections (n_out bis min(n_in × n_out, 50)) → Gewichte neu initialisieren |
| 3 | Frisches Template → 3–8 vollständige Mutationen |

**Strukturelle Diversitätsinjektion:**

Wenn die Topologie (Anzahl Nodes und Connections) des besten Genoms seit `3 × max_size` Bewertungen unverändert ist, injiziert `_inject_structural_diversity()` gezielt eine strukturell abweichende Variante. Drei Strategien:

| Strategie | Beschreibung |
|---|---|
| 0 | Kopie des besten Genoms → 2–5 neue Connections zwingen |
| 1 | Kopie des besten Genoms → `add_node` → 1–3 weitere Connections |
| 2 | Frisches Template → zufällige Connections → Gewichte neu initialisieren |

**Normales Spawning:**

1. `_assign_species()` ordnet alle bewerteten Genome Species zu.
2. `_compute_shared_fitness()` berechnet Shared Fitness für alle Mitglieder.
3. `_compute_efficiency_scores()` normalisiert Bewertungszeiten, falls sich die Spannweite geändert hat.
4. Novelty wird neu berechnet, wenn seit der letzten Berechnung mindestens `max_size / 2` Bewertungen vergangen sind.
5. **Tournament Selection (k=3):** Drei zufällige Genome aus `_evaluated` werden per `selection_score` verglichen; das beste wird als Elternteil gewählt.
6. **Crossover oder Kopie:** Mit Wahrscheinlichkeit `crossover_prob` des Elternteils wird ein zweites Elternteil aus derselben Species gewählt. Gibt es dort keine anderen Genome, wird aus der gesamten Population gewählt. Das fittere der beiden Genome ruft `crossover(weaker)` auf. Andernfalls wird das erste Elternteil kopiert.
7. Das Kind wird mutiert (Abschnitt 6).
8. Das Kind erhält Fitness `0.0` und landet in `_unevaluated`.

### 8.11 Spezies-Lebenszyklus

**Spezies entsteht:**

In jedem Spawning-Zyklus ruft `_spawn_offspring()` die Funktion `_assign_species()` auf. Jedes bewertete Genome wird gegen die Repräsentanten aller bestehenden Species geprüft. Ist die Kompatibilitätsdistanz zu allen Repräsentanten ≥ Schwellwert, entsteht eine neue `Species`. Das Genome wird ihr erstes Mitglied und gleichzeitig ihr Repräsentant.

**Spezies wächst:**

Jedes weitere Genome mit einer Kompatibilitätsdistanz < Schwellwert zum Repräsentanten wird der Species zugewiesen. Nach jeder vollständigen Zuweisung wird der Repräsentant auf das aktuell fitteste Mitglied der Species aktualisiert.

**Spezies werden zusammengeführt:**

Wenn nach der Zuweisung die Anzahl der Species über `target_species` liegt, prüft `_assign_species()`, ob zwei Species-Repräsentanten selbst kompatibel sind (Distanz < Schwellwert). In diesem Fall wird die kleinere Species in die größere aufgenommen. Das Zusammenführen stoppt, sobald die Zielzahl erreicht ist, um kein Über-Merging zu verursachen.

**Spezies stirbt:**

`_prune()` wird nach jeder Fitnesseingabe aufgerufen, sobald die Population `max_size` übersteigt. Es wird das Genome mit der schlechtesten Fitness entfernt — sofern es nicht geschützt ist:

- **Global geschützt:** Die top-`elite_count` Genome nach Fitness (Standard: 1).
- **Species-geschützt:** Das jeweils fitteste Genome jeder Species (Standard: 1 pro Species).

Für jedes entfernte Genome:

1. Das Genome wird aus `_evaluated` und aus dem Behavior-Archiv gelöscht.
2. Es wird aus seiner Species entfernt.
3. Wenn die Species dadurch keine Mitglieder mehr hat, gilt sie als leer.
4. Am Ende jedes Speziation-Zyklus werden leere Species endgültig aus `_species` entfernt.

Species-Stagnation (wie viele Zyklen kein Fitnessfortschritt innerhalb der Species) wird für Diagnosezwecke getrackt, aber aktuell nicht als Auslöser für das Entfernen ganzer Species verwendet.

## 9. Lamarckian Refinement

Optional aktivierbar:

```python
yane.set_lamarck(n_steps=5, sigma=1.0)
```

Vor der eigentlichen Fitnessbewertung wird ein lokaler Hill-Climb auf Gewichten und Biases durchgeführt:

0. die Basisfitness des aktuellen Genoms wird einmalig berechnet
1. aktuelle Gewichte und Biases werden gespeichert
2. alle Gewichte und Biases werden mit Gaußrauschen verändert
3. die Fitness wird neu berechnet
4. nur bessere Änderungen bleiben erhalten
5. schlechtere Änderungen werden zurückgerollt

Schritte 1–5 werden `n_steps` Mal wiederholt. Danach folgt die reguläre Fitnessbewertung in der Trainingsschleife. Gesamtkosten pro Genom: `n_steps + 2` Fitnessfunktionsaufrufe (1 Basisfitness + `n_steps` Hill-Climb + 1 finale Bewertung).

Die Schrittweite ist:

```text
genome.sigma_global * yane._lamarck_sigma
```

Das ist Lamarckian, weil die verbesserten Gewichte direkt im Genom bleiben und weitervererbt werden.

## 10. Mehrfachbewertung pro Genom

Für stochastische Umgebungen kann jedes Genom mehrfach bewertet werden:

```python
yane.set_multi_eval(n=5, aggregation="mean", sigma_penalty=0.0)
```

Parameter:

- `n`: Anzahl der Bewertungen pro Genom (Standard: `1`)
- `aggregation`: wie die `n` Fitnesswerte zusammengefasst werden
  - `"mean"`: arithmetisches Mittel (Standard)
  - `"median"`: statistischer Median; robust gegen Ausreißer-Episoden
  - `"min"`: Worst-Case-Fitness; konservativste Wahl
- `sigma_penalty`: subtrahiert `sigma_penalty × Standardabweichung` vom aggregierten Wert; bestraft hochvariante Genome unabhängig von der Aggregationsmethode; bei `n=1` oder Standardabweichung `0` wirkungslos

Die `elapsed_ms` für die Effizienzbewertung ist die Summe aller `n` Einzelmessungen. Da alle Genome dieselbe Anzahl Bewertungen erhalten, bleiben die relativen Effizienzscores korrekt.

Kosten: `n` Fitnessfunktionsaufrufe pro Genom. Wenn Lamarckian Refinement aktiv ist, gilt `n_steps + 2` wie ohne Mehrfachbewertung — Lamarck nutzt nur Single-Evals für den Hill-Climb; die Mehrfachbewertung gilt nur für die abschließende „offizielle" Fitness.

Der manuelle Loop (`next_genome()` / `submit_fitness()`) ist nicht betroffen; dort aggregiert der Nutzer selbst.

## 11. Fitness-Konventionen

YANE gibt keine feste Fitness-Skala vor. Die Beispiele nutzen zwei Muster:

- Fehleraufgaben: Fitness ist negativer absoluter Fehler, besser ist näher an `0`
- Gym-Aufgaben: Fitness ist Reward oder geformter Reward, höher ist besser

Für Regressions- und Klassifikationsbeispiele wird häufig `genome.reset()` zwischen Samples aufgerufen. Für Sequenzen und Episoden wird nur am Anfang zurückgesetzt, damit Memory wirken kann.

## 12. Beispielkonfigurationen

Dieser Abschnitt beschreibt die aktuellen Beispiele aus `examples/` und `gui/examples.py`.

### 12.1 XOR

Ziel: XOR von zwei binären Eingaben lernen.

Dataset:

| Input | Output |
|---|---|
| `[0, 0]` | `[0]` |
| `[0, 1]` | `[1]` |
| `[1, 0]` | `[1]` |
| `[1, 1]` | `[0]` |

Skript-Konfiguration:

- `n_inputs=2`
- `n_outputs=1`
- `max_nodes=20`
- `max_connections=50`
- `stateful=True` implizit, aber Evaluation resetet jedes Sample
- `target_fitness=-0.1`
- keine Normalisierung nötig

GUI-Konfiguration:

- `n_initial_hidden=2`
- `stateful=False`

Fitness:

```text
fitness = -sum(abs(output - target))
```

### 12.2 Multiplication

Ziel: Multiplikationstabelle `0..9 * 0..9`.

Normalisierung:

- Input `a` wird zu `a / 9`
- Input `b` wird zu `b / 9`
- Output `a*b` wird zu `(a*b) / 81`

Grund:

Aktivierungsfunktionen arbeiten stabiler, wenn Werte in einem kleinen Bereich liegen. Ohne Normalisierung müssten Outputs bis `81` direkt erzeugt werden.

Skript-Konfiguration:

- `n_inputs=2`
- `n_outputs=1`
- `max_nodes=30`
- `max_connections=100`
- `target_fitness=-5.0`
- `stateful=True` implizit, aber jedes Sample wird zurückgesetzt

GUI-Konfiguration:

- `n_initial_hidden=3`
- `stateful=False`
- `input_scale=[9.0, 9.0]` für Rohwertanzeige im Inspect-Tab
- `output_scale=[81.0]`
- Normalisierung kann in `make_eval(..., normalize=False)` deaktiviert werden

Fitness:

```text
fitness = -sum(abs(normalized_output - normalized_target))
```

`-5.0` bedeutet über 100 Samples im Schnitt maximal ca. `0.05` normalisierter Fehler.

### 12.3 Regression 2->2

Ziel: eine kleine kontinuierliche 2-zu-2-Abbildung mit XOR-artigem Muster.

Dataset:

| Input | Output |
|---|---|
| `[0, 0]` | `[0, 1]` |
| `[0, 1]` | `[1, 0]` |
| `[1, 0]` | `[1, 1]` |
| `[1, 1]` | `[0, 0]` |

Skript-Konfiguration:

- `n_inputs=2`
- `n_outputs=2`
- `max_nodes=20`
- `max_connections=60`
- `target_fitness=-0.4`
- keine zusätzliche Normalisierung

GUI-Konfiguration:

- `max_nodes=25`
- `max_connections=80`
- `n_initial_hidden=4`
- `stateful=False`

Fitness:

```text
fitness = -sum(abs(output_i - target_i))
```

`-0.4` entspricht bei 4 Samples und 2 Outputs einem durchschnittlichen Fehler von ca. `0.05` pro Output.

### 12.4 Regression 3->3

Ziel: eine nichtlineare 3-zu-3-Abbildung über alle binären 3-Bit-Eingaben.

Dataset:

- 8 Samples
- Inputs aus `{0,1}^3`
- Outputs sind 3 Werte im Bereich `0/1`

Skript-Konfiguration:

- `n_inputs=3`
- `n_outputs=3`
- `max_nodes=20`
- `max_connections=60`
- `target_fitness=-5.0`
- keine zusätzliche Normalisierung

GUI-Konfiguration:

- `max_nodes=30`
- `max_connections=120`
- `n_initial_hidden=9`
- `stateful=False`

Die GUI gibt dem Beispiel mehr Hidden-Kapazität, weil drei Outputs parallel nichtlineare Teilfunktionen lernen müssen.

### 12.5 Sequence: Pi-Ziffern

Ziel: aus der aktuellen Pi-Ziffer die nächste Ziffer vorhersagen.

Rohsequenz beginnt:

```text
3 -> 1 -> 4 -> 1 -> 5 -> 9 -> 2 -> 6 -> 5 -> 3 ...
```

Normalisierung:

- Input-Ziffer `d` wird zu `d / 9`
- Output-Ziffer `next` wird zu `next / 9`

Skript-Konfiguration:

- `n_inputs=1`
- `n_outputs=1`
- `max_nodes=30`
- `max_connections=100`
- `DECIMAL_PLACES=10`
- `target_fitness=-10.0`
- `stateful=True` implizit

GUI-Konfiguration:

- `max_nodes=20`
- `max_connections=60`
- `stateful=True`
- `input_scale=[9.0]`
- `output_scale=[9.0]`
- Inspect nutzt die ersten 10 Sequenzsamples

Wichtig:

Bei der Evaluation wird `genome.reset()` nur einmal vor der Sequenz aufgerufen. Dadurch können persistente Hidden Nodes Informationen aus vorherigen Schritten behalten.

Fitness:

```text
fitness = -sum(abs(normalized_output - normalized_target))
```

### 12.6 MNIST

Ziel: handgeschriebene Ziffern klassifizieren.

Datei:

- erwartet `mnist_train.csv` im aktuellen Arbeitsverzeichnis
- Quelle laut Codekommentar: Kaggle MNIST CSV Dataset

Normalisierung:

- Pixelwerte `0..255` werden zu `pixel / 255.0`

Konfiguration:

- `n_inputs=784`
- `n_outputs=10`
- `max_nodes=200`
- `max_connections=1000`
- `max_iterations=100`
- `target_fitness=n_samples`

Fitness:

```text
fitness = Anzahl korrekt klassifizierter Samples
```

Output-Auswahl:

```python
predicted_label = outputs.index(max(outputs))
```

Hinweis: Dieses Beispiel ist deutlich größer als die kleinen Dataset-Beispiele und benötigt die externe CSV-Datei.

## 13. GUI-/Gymnasium-Beispiele

Die folgenden Beispiele werden in der GUI nur geladen, wenn `gymnasium` importierbar ist.

### 13.1 CartPole

- Environment: `CartPole-v1`
- Inputs: 4 Zustandswerte
- Outputs: 2 Aktionen
- `max_nodes=30`
- `max_connections=100`
- `target_fitness=1000`
- Action: Index des maximalen Outputs
- Reset: am Episodenanfang

### 13.2 Acrobot

- Environment: `Acrobot-v1`
- Inputs: 6
- Outputs: 3
- `max_nodes=30`
- `max_connections=100`
- `target_fitness=0`
- Action: Index des maximalen Outputs
- Reward-Shaping: maximale Endeffektorhöhe plus `10.0`, wenn gelöst

### 13.3 MountainCar Continuous

- Environment: `MountainCarContinuous-v0`
- Inputs: 2
- Outputs: 1
- `max_nodes=20`
- `max_connections=60`
- `target_fitness=10.0`
- Action-Skalierung: `output * 2 - 1`, geklemmt auf `[-1, 1]`
- Fitness: maximale Position plus `10.0`, wenn gelöst

### 13.4 MountainCar Discrete

- Environment: `MountainCar-v0`
- Inputs: 2
- Outputs: 3
- `max_nodes=20`
- `max_connections=60`
- `target_fitness=10.0`
- Action: Index des maximalen Outputs
- Fitness: maximale Position plus `10.0`, wenn gelöst

### 13.5 Pendulum

- Environment: `Pendulum-v1`
- Inputs: 3
- Outputs: 1
- `max_nodes=20`
- `max_connections=60`
- `n_initial_hidden=2`
- `target_fitness=-400`
- Action-Skalierung: `output * 4 - 2`, geklemmt auf `[-2, 2]`

### 13.6 LunarLander

- Environment: `LunarLander-v3`
- Inputs: 8
- Outputs: 4
- `max_nodes=40`
- `max_connections=150`
- `target_fitness=200`
- `early_stop=-200`
- Action: Index des maximalen Outputs

### 13.7 BipedalWalker

- Environment: `BipedalWalker-v3`
- Inputs: 24
- Outputs: 4
- `max_nodes=60`
- `max_connections=300`
- `n_initial_hidden=4`
- `target_fitness=0`
- `early_stop=-50`
- Action-Skalierung: pro Output `output * 2 - 1`, geklemmt auf `[-1, 1]`

### 13.8 CarRacing

- Environment: `CarRacing-v3`
- Rohbeobachtung: `96x96x3` Pixel
- Vorverarbeitung: Grayscale, Downsampling auf `12x12`
- Inputs: 144 normalisierte Pixelwerte `0..1`
- Outputs: 3
- `max_nodes=80`
- `max_connections=500`
- `target_fitness=100`
- Actions:
  - Steering: `output[0] * 2 - 1`, geklemmt auf `[-1, 1]`
  - Gas: `output[1]`, geklemmt auf `[0, 1]`
  - Brake: `output[2]`, geklemmt auf `[0, 1]`

### 13.9 Blackjack

- Environment: `Blackjack-v1`
- Inputs:
  - Player Sum `/31`
  - Dealer Card `/10`
  - Usable Ace als `0.0` oder `1.0`
- Outputs: 2 (`stick`, `hit`)
- `max_nodes=20`
- `max_connections=60`
- `target_fitness=-0.05`
- Fitness: durchschnittlicher Reward über 500 Episoden, in Demo 20 Episoden

### 13.10 Cliff Walking

- Environment: `CliffWalking-v1`
- Grid: `4x12`
- Inputs:
  - Row `/3`
  - Column `/11`
- Outputs: 4
- `max_nodes=30`
- `max_connections=100`
- `n_initial_hidden=4`
- `target_fitness=0.0`
- Reward-Shaping:
  - Bonus für Annäherung ans Ziel
  - kleinerer Malus fürs Entfernen
  - Bonus für neue beste Distanz
  - großer Bonus bei Zielerreichung

### 13.11 Frozen Lake

- Environment: `FrozenLake-v1`
- `is_slippery=False`
- Grid: `4x4`
- Inputs:
  - Row `/3`
  - Column `/3`
- Outputs: 4
- `max_nodes=20`
- `max_connections=60`
- `target_fitness=0.8`
- Fitness: Mittelwert über 20 Episoden, in Demo 5 Episoden
- Reward-Shaping: `0.1` pro Schritt näher ans Ziel

### 13.12 Taxi

- Environment: `Taxi-v4`
- Inputs:
  - Row `/4`
  - Column `/4`
  - Passenger Location `/4`
  - Destination `/3`
- Outputs: 6
- `max_nodes=40`
- `max_connections=150`
- `n_initial_hidden=6`
- `target_fitness=-120.0`
- Reward-Shaping:
  - Bonus für Bewegung zum Fahrgast oder Ziel
  - `+30` bei Pickup
  - `+50` bei erfolgreicher Lieferung

## 14. API-Server

FastAPI-App:

```bash
uvicorn yane.api.server:app --reload
```

Der Server hält eine globale `NeuroEvolution`-Instanz.

Setup-Route:

```http
POST /configure
```

Unterstützte Konfigurationsfelder sind unter anderem `n_inputs`, `n_outputs`,
`max_nodes`, `max_connections`, `n_initial_hidden`, `stateful`,
`population_size`, `target_species`, Lamarck-Parameter und `seed`.

Trainingsablauf per HTTP:

1. `POST /configure`
2. `POST /population/next`
3. beliebige Auswertung über `/network/forward` oder `/network/tick`
4. `POST /population/fitness`
5. Wiederholen

Zusätzlich verfügbar sind Status-/Diagnose-Endpunkte (`GET /population/status`,
`GET /population/best`, `GET /population/diagnostics`), Best-Genome-Export
(`GET /population/best/export`) und Checkpoint-Speicherung
(`POST /checkpoint/save`, `POST /checkpoint/load`).

## 15. Architekturdiagramm

```text
Fitness-Funktion / GUI / API
          |
          v
NeuroEvolution
  - Konfiguration, Stoppkriterien, Logging
  - EvaluationRunner, LamarckRefiner, Sanitizer
          |
          v
Population
  - evaluated / unevaluated Pools
  - Species, Novelty, Pareto-Shaping, Elternauswahl
  - Spawn, Mutation, Crossover, Pruning
          |
          v
Genome
  - Nodes + Connections
  - forward(), forward_batch(), tick()
  - Strategie-Gene und Runtime-Caches
```

## 16. Multi-Objective Optimization

Multi-Objective wird über `NeuroEvolution.set_multi_objective()` aktiviert.
Eine Fitnessfunktion darf dann statt eines Skalars einen Vektor liefern:

```python
yane.set_multi_objective(weights=(1.0, -0.01), maximize=(True, False))

def evaluate(genome):
    return (task_score(genome), genome.connection_count)
```

Der Vektor wird als `genome.objectives` gespeichert. Für bestehende APIs,
Logs und Stop-Kriterien wird zusätzlich eine skalare Fitness berechnet:

- mit `weights`: gewichtete Summe
- ohne `weights`: Summe der Objectives

Die Population kann die Selektion mit Pareto-Rang und Crowding-Distance formen.
Dadurch bleiben unterschiedliche Trade-off-Lösungen länger im Rennen, statt
dass eine einzelne handgebaute Fitnessformel alle Ziele vermischen muss.

## 17. Quality Diversity / MAP-Elites

Quality Diversity wird über `NeuroEvolution.set_quality_diversity()` aktiviert:

```python
yane.set_quality_diversity(
    descriptor_fn=lambda genome: (genome.connection_count,),
    bins=(20,),
    ranges=((0.0, 200.0),),
)
```

`descriptor_fn` erzeugt einen festen Behavior Descriptor. `MAPElitesArchive`
diskretisiert diesen Descriptor in Zellen und speichert pro Zelle eine Kopie des
besten Genoms. `Population.submit()` aktualisiert das Archiv nach jeder
Bewertung. Wenn die Population stagniert, darf `_inject_fresh_genome()` eine
Archiv-Elite kopieren, mutieren und als neuen Kandidaten einspeisen.

Diagnostics:

- `quality_diversity_enabled`
- `quality_diversity_cells`
- `quality_diversity_coverage`
- `n_quality_diversity_updates`
- `n_quality_diversity_injections`

Descriptor-Callbacks werden nicht in Checkpoints gepickelt, weil sie häufig
Closures sind. Das Archiv selbst bleibt erhalten; nach `load_checkpoint()` kann
der Callback erneut per `set_quality_diversity()` gesetzt werden.

## 18. Weitere Forschungsbausteine

### 18.1 Coevolution

`evolution/coevolution.py` enthält:

- `HallOfFame(max_size)`: speichert starke historische Gegner als Genomkopien
- `competitive_fitness(genome, opponents, match_fn, aggregation)`: bewertet ein
  Genom gegen mehrere Gegner
- `mixed_opponents(current_population, hall_of_fame, ...)`: mischt aktuelle und
  historische Gegner, um zyklisches Vergessen zu reduzieren

Damit lassen sich Population-gegen-Population-Experimente aufbauen, ohne den
Standard-Trainingspfad zu verändern.

### 18.2 Modulare Subnetze

`evolution/modularity.py` erkennt Hidden-Node-Komponenten per
`hidden_modules(genome)` und kann ein Modul mit `duplicate_module(genome,
tracker)` duplizieren. Dabei werden Hidden Nodes, interne Connections,
Eingänge ins Modul und Ausgänge aus dem Modul kopiert und mit neuen
Innovationsnummern versehen.

### 18.3 Indirekte Kodierung / CPPN

`evolution/indirect_encoding.py` bietet einen einfachen HyperNEAT-artigen
Einstieg:

- `layered_coordinates(genome)` weist Input/Hidden/Output-Nodes 2D-Koordinaten zu.
- `generate_connections_from_coordinates(...)` erzeugt Connections aus einer
  CPPN-artigen Gewichtsfunktion über Quell-/Zielkoordinaten und Distanz.

### 18.4 Matrix-, GPU- und Backprop-Export

`evolution/matrix_export.py` exportiert kompatible azyklische Genome ohne
persistente Hidden Nodes als `MatrixGenome` mit Adjazenzmatrix, Biasvektor,
Aktivierungen und Input/Output-Indizes. `forward_matrix()` läuft mit NumPy-API,
`forward_matrix_gpu()` nutzt CuPy, wenn installiert.

`evolution/backprop.py` enthält einen optionalen PyTorch-Hook
`backprop_finetune_linear_outputs(...)`. PyTorch ist keine harte Abhängigkeit;
ohne Installation wird ein klarer `ImportError` geworfen.

### 18.5 CMA-ES als Lamarck-Modus

`set_lamarck(mode="cma_es", cma_population=...)` und
`set_lamarck_adaptive(mode="cma_es", ...)` aktivieren eine kompakte
volle-Kovarianz-CMA-ES-Variante für die Gewichte und Biases einer festen
Topologie. Der Modus ist absichtlich lokal gehalten: Topologie bleibt
evolutionär, Parameter werden lamarckianisch verfeinert.

## 19. Genom-Lifecycle

Ein Genom durchläuft typischerweise diese Zustände:

1. `Population.select_for_evaluation()` gibt ein unbewertetes Genom zurück.
2. Die Fitnessfunktion ruft `forward()`, `forward_batch()` oder `tick()`.
3. `NeuroEvolution._finalize_fitness()` sanitizt Fitness, wendet optionale
   Effizienz-/Komplexitätsstrafen an und speichert Objective-Vektoren.
4. `Population.submit()` verschiebt das Genom in `_evaluated`, berechnet
   Verhalten/Novelty, weist Species zu und aktualisiert Stagnationszähler.
5. Wenn neue Kandidaten gebraucht werden, erzeugt `_spawn_offspring()` Kopien,
   Crossover-Kinder oder Diversity-Injektionen.
6. `Genome.mutate()` verändert Struktur, Gewichte, Aktivierungen, Memory-Gates
   und Strategie-Gene; danach werden Topologie-Caches invalidiert.
7. `_prune()` entfernt schlechte Genome, schützt aber globale und Species-Eliten.

## 20. Population-Spawn-Zyklus

Der Spawn-Zyklus ist steady-state: Es wird nicht zwingend eine ganze Generation
auf einmal ersetzt. Wenn `_unevaluated` leer ist, erzeugt die Population einen
neuen Kandidaten. Vor der Elternwahl werden Shared Fitness, optionales
Fitness-Shaping, Pareto-Shaping, Novelty und Effizienzscore aktualisiert.

Species erhalten Rolling Spawn-Credits aus durchschnittlicher Shared Fitness,
Stagnationsfaktor und Jugendbonus. Danach läuft das Parent-Tournament innerhalb
der gewählten Species. Crossover verwendet bevorzugt Eltern derselben Species;
Interspecies-Crossover ist nur mit konfigurierter Rate erlaubt.

## 21. Worker-Pfade

Alle automatischen Trainingspfade laufen durch dieselbe Evaluierungslogik:

- `train()` sequenziell über `EvaluationRunner.run()`
- GUI sequenziell über `train()` mit Iterationscallback
- GUI ThreadPool über `_run_evaluations()`
- GUI Multiprocessing über `_mp_evaluate()` plus `submit_fitness_batch()`

Für Multiprocessing werden Genome per Pickle über IPC verschickt. Rebuildbare
Runtime-Caches (`_compiled_forward`, `_forward_dispatch`, `_exec_order`,
`_reset_nodes`, `_innov_cache`, NumPy-Wertebuffer) werden im Pickle-State
entfernt und im Worker beim ersten `forward()` neu aufgebaut.

## 22. Fitness-Konventionen

- Höhere Fitness ist immer besser.
- Fehlerbasierte Beispiele nutzen häufig negative Fehler, bei denen `0` optimal ist.
- Fitnessfunktionen sollten `genome.reset()` am Episodenstart oder vor
  unabhängigen Samples aufrufen.
- NaN/Inf sollten vermieden werden; zusätzlich kann
  `set_fitness_sanitizing()` defensiv absichern.
- Für stochastische Aufgaben empfiehlt sich `set_multi_eval()` mit `median`
  oder `mean` plus optionaler Sigma-Strafe.
- Für Trade-offs empfiehlt sich `set_multi_objective()` statt einer schwer
  wartbaren Mischformel.

## 23. Glossar

- **Genom**: ein evolvierbares neuronales Netz inklusive Strategieparameter.
- **Node**: Neuron mit Aktivierung, Bias, optionalem Memory und ausgehenden Connections.
- **Connection**: gerichtete gewichtete Verbindung zu einem Ziel-Node.
- **Innovation Number**: historische ID für NEAT-kompatiblen Crossover.
- **Species**: Gruppe ähnlicher Topologien, die intern konkurrieren.
- **Shared Fitness**: Fitness nach Species-Sharing, damit große Species nicht dominieren.
- **Novelty**: Bonus für ungewöhnliches Verhalten auf festen Probeinputs.
- **Lamarckian Refinement**: lokale Gewichtsoptimierung, deren Ergebnis vererbt wird.
- **Pareto-Front**: Menge nicht-dominierter Lösungen bei mehreren Zielen.
- **Crowding-Distance**: NSGA-II-Tie-Breaker, der dünn besetzte Frontbereiche schützt.
- **MAP-Elites**: Quality-Diversity-Archiv mit der besten Lösung pro Verhaltenszelle.
- **CPPN**: Netzwerk/Funktion, die Gewichte aus geometrischen Koordinaten erzeugt.

## 24. Grenzen und Hinweise

- YANE ist stochastisch. Gleiche Einstellungen können je nach Seed unterschiedlich schnell konvergieren.
- Große Aufgaben wie MNIST oder pixelbasierte Kontrolle sind deutlich schwerer als die kleinen Dataset-Beispiele.
- Fitnessfunktionen sollten Ausgaben defensiv behandeln, wenn lineare Outputs sehr groß werden können.
- Für unabhängige Samples sollte `genome.reset()` vor jedem Sample aufgerufen werden.
- Für Sequenzen und Episoden sollte `genome.reset()` nur am Episodenstart aufgerufen werden.
- Normalisierung ist oft entscheidend, weil Aktivierungsfunktionen und Mutationsschrittweiten sonst in ungünstigen Skalen arbeiten.

## 25. Weitere öffentliche API-Methoden

### 25.1 Populationsgröße und Species

```python
yane.set_population_size(n)   # Standard: 100
yane.set_target_species(n)    # Standard: 5
```

`set_population_size(n)` legt die maximale Anzahl von Genomen in der Population fest. Kleinere Populationen trainieren schneller, finden aber öfter lokale Optima. `set_target_species(n)` setzt die Zielanzahl der Species; der Kompatibilitätsschwellwert wird automatisch angepasst. Höhere Werte erhalten mehr strukturelle Diversität.

### 25.2 Batch-Evaluation

```python
genomes = yane.next_genome_batch(n=4)
results = [(g, run_simulation(g)) for g in genomes]
yane.submit_fitness_batch(results)
```

Für manuelle Parallelisierung: `next_genome_batch(n)` gibt `n` unbewertete Genome zurück (spawnt bei Bedarf Nachkommen). `submit_fitness_batch(results)` reicht Fitnesswerte für eine Liste von `(Genom, Fitness)`-Paaren ein. Optional sind auch `(Genom, Fitness, Bewertungszeit_ms)`-Tupel möglich, damit die automatische Effizienzbewertung in eigenen Batch-Loops greift. Mindestens ein Genom muss vorher über `next_genome()` / `submit_fitness()` bewertet worden sein.

### 25.3 Speicherverwaltung

```python
yane.trim_memory()
```

Gibt freigegebene Heap-Seiten explizit ans Betriebssystem zurück (ruft `malloc_trim` auf Linux auf). Nützlich nach langen Trainingsläufen oder nach `set_resource_limits(max_process_gb=...)`.

## 26. Testabdeckung

Die Tests decken unter anderem ab:

- Initialisierung und Konfiguration
- Forward- und Tick-Modus
- Mutation und Mutationsraten
- Memory Nodes und `stateful=False`
- Connections und Crossover
- Innovation Tracking
- Population, Pruning und Resource-Stabilität
- Novelty und Ensemble
- Multi-Objective/Pareto-Helfer
- Checkpoints, Normalisierung, Curriculum und Batch-Forward

Ausführen:

```bash
pytest
```
