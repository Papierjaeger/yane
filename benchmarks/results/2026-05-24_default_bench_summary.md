# GUI Default Benchmark Summary — 2026-05-24

Quick suite: 3 seeds per example, fixed short time budgets via
`benchmarks/default_bench.py`.

| Example | Baseline solved | Final solved | Baseline mean fitness | Final mean fitness | Notes |
|---|---:|---:|---:|---:|---|
| XOR | 1/3 | 3/3 | -0.698 | -0.053 | Better defaults: 4 hidden, target species 2, fitness shaping, Lamarck 2. |
| Regression 2→2 | 0/3 | 0/3 | -1.958 | -1.423 | Fitness improved, target still not reached in short budget. |
| Regression 3→3 | 0/3 | 0/3 | -8.299 | -7.534 | Fitness and accuracy improved, still difficult in short budget. |
| Multiplication | 0/3 | 0/3 | -11.585 | -9.266 | Fitness improved; exact table accuracy remains low in short budget. |
| Sequence: Pi-Ziffern | 0/3 | 0/3 | -0.241 | -0.342 | Accuracy unchanged at 0.20; smaller population/species is faster but noisier. |
| CartPole | 0/3 | 1/3 | 138.000 | 342.333 | Fixed GUI target/env horizon to standard 500-step task. |
| MountainCar (Discrete) | 2/3 | 1/3 | 7.042 | 3.411 | Existing defaults were already decent; no GUI tuning applied. |
| Frozen Lake | 0/3 | 1/3 | 0.367 | 0.867 | Added hidden capacity and lower target species. |

Result files:

- `benchmarks/results/2026-05-24_default_bench_baseline.json`
- `benchmarks/results/2026-05-24_default_bench_final.json`
