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

Add the **parent directory** of `yane/` to `PYTHONPATH`, then import normally:

```bash
# from the parent of yane/
export PYTHONPATH=/path/to/parent
```

Or run examples directly:

```bash
cd /path/to/parent
python -m yane.examples.XOR
```

```python
from yane import NeuroEvolution

yane = NeuroEvolution()
yane.configure(n_inputs=2, n_outputs=1, max_nodes=50, max_connections=200)
yane.set_min_fitness(-0.1)   # optional — omit to train indefinitely

def evaluate(genome):
    fitness = 0.0
    for inputs, target in [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]:
        outputs = genome.forward(inputs)
        fitness -= abs(outputs[0] - target)
    return fitness

yane.train(evaluate)
best = yane.get_best()
```

### Tick mode (step-by-step)

```python
genome = yane.next_genome()
genome.set_inputs([0.5, 1.0])
genome.tick()               # one propagation step
outputs = genome.get_outputs()
yane.submit_fitness(my_score)
```

### Resource limits

Training pauses automatically when system memory is low and resumes when it recovers:

```python
yane.set_resource_limits(min_free_gb=2.0, max_used_percent=85.0)
```

### Efficiency penalty

```python
yane.set_efficiency_penalty(max_ms=10.0, penalty_per_ms=0.1)
```

### API server

```bash
cd src
uvicorn yane.api.server:app --reload
```

Endpoints:

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
