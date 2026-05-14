# Yet Another Neuro Evolution (YANE)

A Python framework for **neuroevolution** — training neural networks through evolutionary algorithms instead of backpropagation.

**Requires Python 3.10+**

## How it works

YANE maintains a **population of genomes** (neural networks) and evolves them over time:

1. A genome is selected from the population and evaluated with a user-defined fitness function
2. Fitter genomes produce offspring through mutation (tournament selection used when the unevaluated pool is exhausted)
3. The population evolves until a target fitness is reached, or indefinitely

Mutations can modify:
- Network structure (add/remove nodes, add/remove connections)
- Node parameters (bias, activation function, input index)
- Whether a node carries its value to the next tick cycle
- **Mutation rates themselves** — rates self-adapt over time (meta-evolution), with a hard minimum floor

Supported activation functions: `Linear`, `Sigmoid`, `Tanh`, `ReLU`, `Binary`

Cycles are explicitly allowed — self-connections, bidirectional connections, loops. The `max_triggers` parameter per node prevents infinite loops in forward mode.

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `gymnasium`, `fastapi`, `uvicorn`, `pydantic`, `psutil`

## Usage

Add the **parent directory** of `yane/` to `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/parent
# or run an example directly:
cd /path/to/parent && python -m yane.examples.XOR
```

### Training

```python
from yane import NeuroEvolution

yane = NeuroEvolution()
yane.configure(n_inputs=2, n_outputs=1, max_nodes=50, max_connections=200)
yane.set_min_fitness(-0.1)   # stop when fitness >= this value
yane.set_max_iterations(10000)  # or stop after N evaluations (whichever comes first)
# omit both to train indefinitely

def evaluate(genome):
    fitness = 0.0
    for inputs, target in [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]:
        outputs = genome.forward(inputs)
        fitness -= abs(outputs[0] - target)
    return fitness

n = yane.train(evaluate)   # returns number of iterations performed
best = yane.get_best()
```

### Manual loop

For evaluation that spans multiple steps (e.g. a simulation):

```python
genome = yane.next_genome()
score = run_simulation(genome)   # your evaluation here
yane.submit_fitness(score)
```

### Tick mode (step-by-step propagation)

Each `tick()` fires all currently stimulated nodes once. Outputs may be empty until enough ticks have propagated through the network.

```python
genome = yane.next_genome()
genome.set_inputs([0.5, 1.0])
genome.tick()                    # input nodes fire → hidden nodes receive signal
genome.tick()                    # hidden nodes fire → output nodes receive signal
outputs = genome.get_outputs()
yane.submit_fitness(my_score)
```

### Efficiency penalty

Penalises genomes that take too long to evaluate, promoting smaller and faster networks:

```python
yane.set_efficiency_penalty(max_ms=10.0, penalty_per_ms=0.5)
```

### Resource limits

Training pauses automatically when system memory runs low and resumes when it recovers:

```python
yane.set_resource_limits(min_free_gb=2.0, max_used_percent=85.0)
```

### Diagnosing memory growth

```python
print(yane.population_memory_info())
# {'total_genomes': 100, 'avg_nodes_per_genome': 12.3, 'largest_genome_nodes': 47, ...}
```

### API server

```bash
cd /path/to/parent
uvicorn yane.api.server:app --reload
```

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/configure` | Initialise with n_inputs / n_outputs |
| `POST` | `/population/next` | Select next genome for evaluation |
| `POST` | `/population/fitness` | Submit fitness for current genome |
| `GET`  | `/population/status` | Population size, best fitness |
| `GET`  | `/population/best` | Best genome info |
| `POST` | `/network/inputs` | Set input values |
| `POST` | `/network/tick` | Execute one tick |
| `GET`  | `/network/outputs` | Read current outputs |
| `POST` | `/network/forward` | Full forward pass |
| `POST` | `/network/reset` | Reset state |

## Project structure

```
yane/                       ← the Python package (importable as `yane`)
  __init__.py
  neuro_evolution.py        # Main entry point
  core/
    genome.py               # Network (tick + forward mode, mutation, copy)
    node.py                 # Neuron with activation, bias, persist_value
    connection.py           # Weighted connection with self-adaptive mutation
  evolution/
    mutation.py             # Self-adaptive mutation rates
    smart_mutation.py       # Structure changes (insert/remove node, add/remove connection)
    population.py           # Population management & tournament selection
    efficiency_penalty.py   # Fitness penalty for slow networks
  util/
    activation.py           # Activation functions
    resource_guard.py       # System memory monitoring
  api/
    server.py               # FastAPI app
    models.py               # Pydantic schemas
    routes/
      network.py            # Tick/forward endpoints
      population.py         # Population management endpoints
  examples/
    XOR/
    MNIST/
    gym/
  tests/
```

## Status

Active development.
