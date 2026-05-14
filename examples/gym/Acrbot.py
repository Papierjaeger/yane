"""Acrobot example — swings a two-link robot arm to reach a target height."""
import numpy as np
import gymnasium as gym

from yane import NeuroEvolution


def main():
    env = gym.make("Acrobot-v1")
    n_inputs = env.observation_space.shape[0]   # 6
    n_outputs = env.action_space.n               # 3

    yane = NeuroEvolution()
    yane.configure(n_inputs=n_inputs, n_outputs=n_outputs, max_nodes=30, max_connections=100)
    yane.set_resource_limits(max_process_gb=2.0)
    yane.set_min_fitness(-64)

    def evaluate(genome):
        state, _ = env.reset()
        done = False
        fitness = 0.0
        while not done:
            outputs = genome.forward(list(state))
            action = int(np.argmax(outputs))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            fitness += reward
            if fitness < -100:
                break
        return fitness

    n = yane.train(evaluate)
    env.close()
    print(f"Done in {n} iterations.")

    best = yane.get_best()
    env = gym.make("Acrobot-v1", render_mode="human")
    state, _ = env.reset()
    done = False
    while not done:
        outputs = best.forward(list(state))
        action = int(np.argmax(outputs))
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()


if __name__ == "__main__":
    main()
