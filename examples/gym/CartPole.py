"""CartPole example — balances a pole on a cart using gymnasium."""
import numpy as np
import gymnasium as gym

from yane import NeuroEvolution


def main():
    env = gym.make("CartPole-v1")
    n_inputs = env.observation_space.shape[0]   # 4
    n_outputs = env.action_space.n               # 2

    yane = NeuroEvolution()
    yane.configure(n_inputs=n_inputs, n_outputs=n_outputs, max_nodes=30, max_connections=100)
    yane.set_min_fitness(500)

    def evaluate(genome):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            outputs = genome.forward(list(state))
            action = int(np.argmax(outputs))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        return total_reward

    n = yane.train(evaluate)
    print(f"Done in {n} iterations.")

    best = yane.get_best()
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
