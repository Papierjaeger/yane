# Yet Another Neuro Evolution (YANE)

A Python framework for **neuroevolution** — training neural networks through evolutionary algorithms instead of backpropagation.

## How it works

YANE maintains a **population of genomes** (neural networks) and evolves them over time:

1. A genome is selected from the population and evaluated with a user-defined fitness function
2. Fitter genomes reproduce via tournament selection, producing offspring through mutation
3. The population evolves until a target fitness is reached

Mutations can modify:
- Network structure (add/remove nodes and connections)
- Node parameters (bias, activation function, input index)
- Connection weights
- **Mutation rates themselves** — rates self-adapt over time (meta-evolution)

Supported activation functions: `Linear`, `Sigmoid`, `Tanh`, `ReLU`, `Binary`

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `gym`, `line_profiler`

## Usage

```python
from neural_network.yane import Yane

yane = Yane()
yane.set_min_fitness(100)

def fitness():
    # Evaluate yane.selected_candidate and return a fitness score
    return yane.selected_candidate.tick([0.5, 1.0])[0] * 100

yane.run(fitness)

best = yane.get_best_solution()
```

### Evaluating a genome manually

```python
output = genome.tick([input1, input2])  # returns list of output values
```

## Examples

| Example | Description |
|---------|-------------|
| `src/examples/XOR` | Learn the XOR function |
| `src/examples/MNIST` | Handwritten digit classification |
| `src/examples/gym` | OpenAI Gym environments |
| `src/examples/basic multiplication` | Simple arithmetic task |
| `src/examples/sequence_recall_PI` | Sequence memory task |

## Project structure

```
src/
  neural_network/
    yane.py          # Main entry point
    Population.py    # Population management & selection
    genome.py        # Genome (network structure & forward pass)
    node.py          # Neuron with activation & mutation
    connection.py    # Weighted connection between nodes
    mutation.py      # Self-adaptive mutation logic
    util/
      activation.py  # Activation functions
  examples/          # Runnable example tasks
docs/
  Training.md        # Notes on training setup
```

## Status

Active development — some features are still work in progress.
