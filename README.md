# Yet Another Neuro Evolution (YANE)

YANE ist ein Python-Framework für **Neuroevolution**: Neuronale Netze werden nicht per Backpropagation trainiert, sondern durch Evolution verändert, bewertet, selektiert und über viele Generationen weiterentwickelt.

Das Projekt enthält:

- eine Python-API für eigene Trainingsschleifen
- eine PySide6-GUI mit Live-Fitnesskurve und Netzwerkvisualisierung
- einen FastAPI-Server für externe Steuerung per HTTP
- mehrere Dataset-, Sequenz- und Gymnasium-Beispiele
- Tests für Genome, Mutationen, Population, Innovation Tracking, Memory Nodes und Stabilität

**Voraussetzung:** Python 3.10+

## Installation

```bash
pip install -r requirements.txt
```

Abhängigkeiten: `numpy`, `PySide6`, `gymnasium`, `fastapi`, `uvicorn`, `pydantic`, `psutil`

## Schnellstart

### GUI starten

Aus dem Projektordner:

```bash
python run.py
```

Oder als Paket aus dem Elternordner von `yane/`:

```bash
python -m yane.gui
```

Die GUI bietet:

- Training-Tab mit Beispielauswahl, Fitnesskurve und Netzwerkansicht
- Inspect-Tab zum Prüfen des besten Genoms auf bekannten Testfällen
- `Run Best` für renderbare Gymnasium-Beispiele
- API-Server-Tab zum Starten der HTTP-Schnittstelle

### Beispiel per Skript trainieren

Aus dem Elternordner von `yane/`:

```bash
python -m yane.examples.XOR
python -m yane.examples.basic_multiplication
python -m yane.examples.simple_2_2_continuous
python -m yane.examples.simple_3_3_continuous
python -m yane.examples.sequence_recall_PI
```

## Grundprinzip

YANE verwaltet eine Population von **Genomen**. Ein Genom ist ein neuronales Netz aus Nodes und gewichteten Connections.

Der Trainingsablauf ist:

1. Ein Genom wird zur Bewertung ausgewählt.
2. Eine Fitness-Funktion bewertet, wie gut dieses Netz die Aufgabe löst.
3. Das Genom wird mit seiner Fitness in die Population zurückgegeben.
4. Sobald neue Kandidaten gebraucht werden, erzeugt die Population Nachkommen durch Mutation oder Crossover.
5. Schlechtere Genome werden verworfen, gute und strukturell interessante Genome bleiben erhalten.

Das Startgenom enthält Input- und Output-Nodes, aber standardmäßig **keine Connections**. Die erste vielfältige Population wird durch zufällige Strukturmutationen aufgebaut. Optional können mit `n_initial_hidden` von Anfang an Hidden Nodes eingefügt werden.

YANE ist NEAT-inspiriert und erweitert den Ansatz um selbstadaptierende Mutationsraten, Novelty Search, Memory Nodes, optionales Lamarckian Weight Refinement und Ressourcenlimits.

## Minimales Python-Beispiel

```python
from yane import NeuroEvolution

DATA = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0]),
]

yane = NeuroEvolution()
yane.configure(
    n_inputs=2,
    n_outputs=1,
    max_nodes=20,
    max_connections=50,
    n_initial_hidden=2,
    stateful=False,
)
yane.set_min_fitness(-0.1)
yane.set_resource_limits(max_process_gb=2.0)

def evaluate(genome):
    fitness = 0.0
    for inputs, target in DATA:
        genome.reset()
        output = genome.forward(inputs)
        fitness -= abs(output[0] - target[0])
    return fitness

iterations = yane.train(evaluate)
best = yane.get_best()
print(iterations, best.fitness)
```

Fitness ist in vielen Beispielen ein negativer Fehler. Je näher der Wert an `0` liegt, desto besser.

## Wichtige API

### Konfiguration

```python
yane.configure(
    n_inputs=2,
    n_outputs=1,
    max_nodes=30,
    max_connections=100,
    n_initial_hidden=3,
    stateful=True,
)
```

Parameter:

- `n_inputs`: Anzahl der Eingabewerte
- `n_outputs`: Anzahl der Ausgabewerte
- `max_nodes`: optionale Obergrenze für Nodes pro Genom
- `max_connections`: optionale Obergrenze für Connections pro Genom
- `n_initial_hidden`: Hidden Nodes im Startgenom
- `stateful`: erlaubt persistente Node-Werte und damit Gedächtnis über mehrere `forward()`-Aufrufe (Standard: `True`)

### Automatisches Training

```python
yane.set_min_fitness(-0.1)
yane.set_max_iterations(10_000)
n = yane.train(evaluate)
best = yane.get_best()
```

Das Training stoppt, wenn `min_fitness` erreicht oder `max_iterations` ausgeschöpft ist. Ohne beide Grenzen läuft es unbegrenzt.

### Manuelle Trainingsschleife

```python
genome = yane.next_genome()
fitness = run_simulation(genome)
yane.submit_fitness(fitness)
```

Diese Form ist nützlich für Simulationen, Episoden oder externe Systeme.

### Forward Mode

```python
outputs = genome.forward([0.5, 1.0])
```

`forward()` berechnet einen vollständigen Durchlauf. Für azyklische Netze wird eine schnelle topologische Ausführungsreihenfolge kompiliert. Bei Zyklen fällt YANE auf einen BFS-basierten Modus mit Trigger-Limit zurück.

### Tick Mode

```python
genome.set_inputs([0.5, 1.0])
genome.tick()
genome.tick()
outputs = genome.get_outputs()
```

`tick()` propagiert Signale schrittweise. Das ist praktisch, wenn man die Dynamik eines rekurrenten Netzes oder eine Simulation explizit takten möchte.

### Ensemble

```python
top = yane.get_ensemble(k=3)
outputs = yane.forward_ensemble([0.2, 0.8], k=3)
```

`forward_ensemble()` mittelt die Outputs der besten `k` Genome.

### Ressourcen und Effizienz

```python
yane.set_efficiency_penalty(max_ms=10.0, penalty_per_ms=0.5)
yane.set_resource_limits(min_free_gb=2.0, max_used_percent=85.0, max_process_gb=2.0)
print(yane.population_memory_info())
yane.trim_memory()
```

Die Effizienzstrafe reduziert Fitness für langsame Bewertungen. Die Ressourcenlimits pausieren Training bei knappem Systemspeicher oder verkleinern die Population, wenn der Prozess zu viel RAM nutzt. `trim_memory()` gibt explizit freigegebene Heap-Seiten ans Betriebssystem zurück und ist nützlich nach langen Trainingsläufen.

### Weitere Konfiguration

```python
yane.set_population_size(100)   # Standard: 100
yane.set_target_species(5)      # Standard: 5
```

- `set_population_size(n)`: Größe der Population.
- `set_target_species(n)`: Zielanzahl Species; der Kompatibilitätsschwellwert wird automatisch angepasst. Höhere Werte schützen mehr strukturelle Nischen (besonders nützlich für XOR-artige Aufgaben).

### Lamarckian Refinement

```python
yane.set_lamarck(n_steps=5, sigma=1.0)
```

Vor jeder Fitnessbewertung werden Gewichte und Biases mit `n_steps` lokalen Hill-Climb-Schritten verfeinert. Verbesserte Gewichte werden direkt ins Genom übernommen und weitervererbt. Kosten: `n_steps + 1` zusätzliche Fitnessfunktionsaufrufe pro Genom.

### Batch-API

```python
genomes = yane.next_genome_batch(n=4)
results = [(g, run_simulation(g)) for g in genomes]
yane.submit_fitness_batch(results)
```

Für manuelle Parallelisierung. Mindestens ein Genom muss vorher über `next_genome()` / `submit_fitness()` bewertet worden sein.

## Beispiele

| Beispiel | Inputs | Outputs | Normalisierung | Hidden Nodes in GUI | Stateful | Ziel-Fitness |
|---|---:|---:|---|---:|---|---:|
| XOR | 2 | 1 | nein, Werte 0/1 | 2 | nein | -0.1 |
| Multiplication | 2 | 1 | Inputs /9, Output /81 | 3 | nein | -5.0 |
| Regression 2->2 | 2 | 2 | nein, Werte 0/1 | 4 | nein | -0.4 |
| Regression 3->3 | 3 | 3 | nein, Werte 0/1 | 9 | nein | -5.0 |
| Sequence: Pi-Ziffern | 1 | 1 | Digit /9 | 0 | ja | -10.0 |
| MNIST | 784 | 10 | Pixel /255 | – | ja | Anzahl Samples |

MNIST ist nur als Skript verfügbar (`python -m yane.examples.MNIST`), nicht in der GUI; die Spalte „Hidden Nodes in GUI" entfällt daher.

Weitere GUI-Beispiele sind CartPole, Acrobot, MountainCar (Continuous), MountainCar (Discrete), Pendulum, LunarLander, BipedalWalker, CarRacing, Blackjack, Cliff Walking, Frozen Lake und Taxi. Details stehen in [TECHNISCHE_DOKUMENTATION.md](TECHNISCHE_DOKUMENTATION.md).

## Technische Dokumentation

Die ausführliche Beschreibung des Netzaufbaus, der Forward-Ausführung, Mutation, Crossover, Speziation, Novelty Search, Memory Nodes und aller Beispielkonfigurationen steht in:

[TECHNISCHE_DOKUMENTATION.md](TECHNISCHE_DOKUMENTATION.md)

## API-Server

Start:

```bash
uvicorn yane.api.server:app --reload
```

Wichtige Endpunkte:

| Methode | Pfad | Beschreibung |
|---|---|---|
| `POST` | `/configure?n_inputs=2&n_outputs=1` | Population initialisieren |
| `POST` | `/population/next` | nächstes Genom auswählen |
| `POST` | `/population/fitness` | Fitness für aktuelles Genom übergeben |
| `GET` | `/population/status` | Populationsstatus |
| `GET` | `/population/best` | bestes Genom |
| `POST` | `/network/inputs` | Inputs setzen |
| `POST` | `/network/tick` | einen Tick ausführen |
| `GET` | `/network/outputs` | Outputs lesen |
| `POST` | `/network/forward` | vollständigen Forward Pass ausführen |
| `POST` | `/network/reset` | Genomzustand zurücksetzen |

Swagger UI ist während des Serverbetriebs unter `http://127.0.0.1:8000/docs` erreichbar.

## Projektstruktur

```text
yane/
  __init__.py
  neuro_evolution.py          Haupt-API und Trainingsloop
  run.py                      GUI-Startdatei
  core/
    genome.py                 Netzwerk, Forward/Tick, Mutation, Crossover
    node.py                   Neuron, Aktivierung, Bias, Memory, Input-Scale
    connection.py             gewichtete Verbindung
  evolution/
    mutation.py               selbstadaptierende Mutationsparameter
    smart_mutation.py         Strukturmutationen
    innovation.py             Innovation Numbers für NEAT-Crossover
    population.py             Population, Speziation, Selection, Novelty
    efficiency_penalty.py     Laufzeitbasierte Fitnessstrafe
  util/
    activation.py             Aktivierungsfunktionen
    resource_guard.py         Speicherüberwachung
    logger.py                 Logging-Helfer
  gui/
    main.py
    window.py                 PySide6-Hauptfenster
    worker.py                 Trainings- und Demo-Threads
    canvas.py                 Netzwerk- und Fitnessvisualisierung
    examples.py               GUI-Beispielregistry
  api/
    server.py                 FastAPI-App
    models.py                 Pydantic-Modelle
    routes/
      network.py              Netzwerkendpunkte
      population.py           Populationsendpunkte
  examples/
    XOR/
    basic_multiplication/
    simple_2_2_continuous/
    simple_3_3_continuous/
    sequence_recall_PI/
    MNIST/
  tests/
```

## Tests

```bash
pytest
```

## Status

Aktive Entwicklung.
