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

Dependencies: `numpy`, `PySide6`, `gymnasium`, `fastapi`, `uvicorn`, `pydantic`, `psutil`

## GUI

Launch the graphical interface:

```bash
cd /path/to/parent
python -m yane.gui
# or directly:
python run.py
```

The GUI provides:
- **Training tab** — configure and run training with live fitness chart and network visualisation
- **Inspect tab** — see how the best genome performs on known test cases after training
- **▶ Run Best** button (gym examples) — watch the best genome play an episode in real-time
- **API Server tab** — start the built-in HTTP server

Built-in examples: XOR, CartPole, Acrobot, MountainCar (Continuous).

## Usage (API / scripting)

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

```python
# Pause when system memory is low, resume when it recovers:
yane.set_resource_limits(min_free_gb=2.0, max_used_percent=85.0)

# Hard cap on yane's own RAM. When exceeded, the population is halved
# (keeping the best genomes) and the GC is forced until under budget:
yane.set_resource_limits(max_process_gb=1.0)
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

#### Example: training loop via HTTP (XOR)

```python
import requests

BASE = "http://127.0.0.1:8000"

XOR_DATA = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]

# 1. Configure once
requests.post(f"{BASE}/configure", params={"n_inputs": 2, "n_outputs": 1})

for _ in range(10_000):
    # 2. Select the next genome to evaluate
    requests.post(f"{BASE}/population/next")

    # 3. Evaluate: forward pass for each input, accumulate fitness
    fitness = 0.0
    for inputs, target in XOR_DATA:
        out = requests.post(f"{BASE}/network/forward", json={"data": inputs}).json()["outputs"][0]
        fitness -= abs(out - target)

    # 4. Submit fitness
    requests.post(f"{BASE}/population/fitness", json={"fitness": fitness})

# Check results
print(requests.get(f"{BASE}/population/status").json())
# {"size": 100, "evaluated": 99, "unevaluated": 1, "best_fitness": -0.07}
print(requests.get(f"{BASE}/population/best").json())
# {"fitness": -0.07, "n_nodes": 5, "n_connections": 6, "n_inputs": 2, "n_outputs": 1}
```

#### Example: tick mode (step-by-step, e.g. CartPole with recurrent network)

```python
import gymnasium as gym
import requests

BASE = "http://127.0.0.1:8000"

requests.post(f"{BASE}/configure", params={"n_inputs": 4, "n_outputs": 2})
requests.post(f"{BASE}/population/next")

env = gym.make("CartPole-v1")
obs, _ = env.reset()
total_reward = 0.0

for _ in range(500):
    # Feed current observation
    requests.post(f"{BASE}/network/inputs", json={"data": obs.tolist()})
    # Propagate one step
    requests.post(f"{BASE}/network/tick")
    # Read action
    outputs = requests.get(f"{BASE}/network/outputs").json()["outputs"]
    action = int(outputs[0] > 0.5)

    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

requests.post(f"{BASE}/population/fitness", json={"fitness": total_reward})
env.close()
```

The interactive API docs (Swagger UI) are available at `http://127.0.0.1:8000/docs` while the server is running.

## Project structure

```
yane/                       ← the Python package (importable as `yane`)
  __init__.py
  neuro_evolution.py        # Main entry point
  run.py                    # Launch the GUI
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
  gui/
    window.py               # Main application window (PySide6)
    worker.py               # Background training and episode runner threads
    canvas.py               # Network visualisation and fitness chart widgets
    examples.py             # Built-in example configurations (XOR, gym envs)
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
