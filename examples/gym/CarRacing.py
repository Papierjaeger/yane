"""CarRacing example — drives a car around a track using image inputs.

Extra dependencies: pip install gymnasium[box2d] swig Pillow
"""
import numpy as np
import gymnasium as gym
from PIL import Image

from yane import NeuroEvolution

IMAGE_SIZE = (8, 8)
N_INPUTS = IMAGE_SIZE[0] * IMAGE_SIZE[1]   # 64 greyscale pixels


def _preprocess(state) -> list[float]:
    img = Image.fromarray(state).resize(IMAGE_SIZE).convert("L")
    return (np.array(img).flatten() / 255.0).tolist()


def main():
    env = gym.make("CarRacing-v2")

    yane = NeuroEvolution()
    yane.configure(n_inputs=N_INPUTS, n_outputs=3, max_nodes=50, max_connections=200)
    yane.set_min_fitness(100)

    def evaluate(genome):
        state, _ = env.reset()
        fitness = 0.0
        for _ in range(100):
            outputs = genome.forward(_preprocess(state))
            state, reward, terminated, truncated, _ = env.step(outputs)
            fitness += reward
            if terminated or truncated or fitness < 0:
                break
        return fitness

    n = yane.train(evaluate)
    env.close()
    print(f"Done in {n} iterations.")

    best = yane.get_best()
    env = gym.make("CarRacing-v2", render_mode="human")
    state, _ = env.reset()
    done = False
    while not done:
        outputs = best.forward(_preprocess(state))
        state, _, terminated, truncated, _ = env.step(outputs)
        done = terminated or truncated
    env.close()


if __name__ == "__main__":
    main()
