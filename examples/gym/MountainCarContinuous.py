"""MountainCarContinuous example — drives a car up a hill with continuous actions."""
import gymnasium as gym

from yane import NeuroEvolution

MAX_STEPS = 800


def main():
    env = gym.make("MountainCarContinuous-v0")
    n_inputs = env.observation_space.shape[0]   # 2
    n_outputs = env.action_space.shape[0]        # 1

    yane = NeuroEvolution()
    yane.configure(n_inputs=n_inputs, n_outputs=n_outputs, max_nodes=20, max_connections=60)
    yane.set_resource_limits(max_process_gb=2.0)
    yane.set_min_fitness(1000)

    def evaluate(genome):
        state, _ = env.reset()
        fitness = 0.0
        for _ in range(MAX_STEPS):
            action = genome.forward(list(state))
            state, reward, terminated, truncated, _ = env.step(action)
            fitness += reward + state[1]   # reward + velocity bonus
            if terminated or truncated:
                break
        return fitness

    n = yane.train(evaluate)
    env.close()
    print(f"Done in {n} iterations.")

    best = yane.get_best()
    env = gym.make("MountainCarContinuous-v0", render_mode="human")
    state, _ = env.reset()
    for _ in range(MAX_STEPS):
        action = best.forward(list(state))
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()


if __name__ == "__main__":
    main()
