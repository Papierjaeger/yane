"""Warm-start / transfer-learning demo.

Pre-trains on CartPole-v1 (4 inputs → 1 output), then warm-starts Acrobot-v1
(6 inputs → 1 output) from that checkpoint.  The 2 extra Acrobot inputs are
appended as new input nodes with no connections; evolution develops them
from there.  The existing 4-input structure carries over its learned weights.

Compares against a cold-start Acrobot run to quantify the head start.

Usage:
    python -m yane.benchmarks.warm_start_demo
"""
from __future__ import annotations
import tempfile
import time
from pathlib import Path

import gymnasium as gym

from yane import NeuroEvolution

# ── Task parameters ──────────────────────────────────────────────────────────

_SEED        = 0
_POP         = 150
_CARTPOLE_TARGET = 450.0
_ACROBOT_TARGET  = -100.0
_CARTPOLE_MAX    = 20_000
_ACROBOT_MAX     = 20_000


# ── Fitness functions ─────────────────────────────────────────────────────────

def _cartpole_eval(seed: int):
    env = gym.make("CartPole-v1")

    def evaluate(genome) -> float:
        total = 0.0
        obs, _ = env.reset(seed=seed)
        genome.reset()
        done = False
        while not done:
            action = 1 if genome.forward(list(obs))[0] > 0.5 else 0
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        return total

    return evaluate


def _acrobot_eval(seed: int):
    env = gym.make("Acrobot-v1")

    def evaluate(genome) -> float:
        total = 0.0
        obs, _ = env.reset(seed=seed)
        genome.reset()
        done = False
        while not done:
            out = genome.forward(list(obs))[0]
            action = int(out * 2 + 1) % 3  # maps [-∞,∞] → {0,1,2}
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        return total

    return evaluate


# ── Training helpers ──────────────────────────────────────────────────────────

def train_cartpole(checkpoint_path: Path) -> None:
    yane = NeuroEvolution(seed=_SEED)
    yane.configure(n_inputs=4, n_outputs=1, max_nodes=20, max_connections=50)
    yane.set_population_size(_POP)
    yane.set_min_fitness(_CARTPOLE_TARGET)
    yane.set_max_iterations(_CARTPOLE_MAX)
    t0 = time.perf_counter()
    iters = yane.train(_cartpole_eval(_SEED))
    elapsed = time.perf_counter() - t0
    fit = yane.get_best().fitness
    solved = fit >= _CARTPOLE_TARGET
    print(f"  CartPole: {'solved' if solved else 'stopped'} after {iters} iters  "
          f"fitness={fit:.1f}  ({elapsed:.0f}s)")
    yane.save_checkpoint(checkpoint_path)
    print(f"  Checkpoint saved → {checkpoint_path.name}")


def run_acrobot(label: str, checkpoint: Path | None) -> tuple[int, float]:
    yane = NeuroEvolution(seed=_SEED)
    yane.configure(n_inputs=6, n_outputs=1, max_nodes=30, max_connections=80)
    yane.set_population_size(_POP)
    yane.set_min_fitness(_ACROBOT_TARGET)
    yane.set_max_iterations(_ACROBOT_MAX)
    if checkpoint is not None:
        n = yane.warm_start_from_checkpoint(checkpoint)
        print(f"  Imported {n} genomes from CartPole checkpoint "
              f"(4→1 adapted to 6→1).")
    t0 = time.perf_counter()
    iters = yane.train(_acrobot_eval(_SEED))
    elapsed = time.perf_counter() - t0
    fit = yane.get_best().fitness
    solved = fit >= _ACROBOT_TARGET
    print(f"  {label}: {'solved' if solved else 'stopped'} after {iters} iters  "
          f"fitness={fit:.1f}  ({elapsed:.0f}s)")
    return iters, fit


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "cartpole.pkl"

        print("━━━ Phase 1: Pre-train CartPole (4 inputs → 1 output) ━━━")
        train_cartpole(ckpt)

        print("\n━━━ Phase 2: Cold-start Acrobot (6 inputs → 1 output) ━━━")
        cold_iters, cold_fit = run_acrobot("Cold start", checkpoint=None)

        print("\n━━━ Phase 3: Warm-start Acrobot from CartPole checkpoint ━━━")
        warm_iters, warm_fit = run_acrobot("Warm start", checkpoint=ckpt)

        print("\n━━━ Summary ━━━")
        cold_solved = cold_fit >= _ACROBOT_TARGET
        warm_solved = warm_fit >= _ACROBOT_TARGET
        if cold_solved and warm_solved:
            ratio = cold_iters / max(1, warm_iters)
            diff  = cold_iters - warm_iters
            if diff > 0:
                print(f"Warm-start converged {ratio:.2f}× faster ({diff} fewer iterations).")
            else:
                print(f"Cold={cold_iters}  Warm={warm_iters} — similar convergence speed.")
        elif warm_solved and not cold_solved:
            print("Warm-start solved; cold-start did not reach target within limit.")
        elif cold_solved and not warm_solved:
            print("Cold-start solved; warm-start did not reach target within limit.")
        else:
            print(f"Neither reached target.  "
                  f"Cold fitness: {cold_fit:.1f}  Warm fitness: {warm_fit:.1f}")


if __name__ == "__main__":
    main()
