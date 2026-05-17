"""MNIST example — classifies handwritten digits (0-9).

Dataset: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
"""
import numpy as np
import pandas as pd

from yane import NeuroEvolution


def read_mnist(path):
    data = pd.read_csv(path)
    return data.iloc[1:].values


def main():
    mnist_data = read_mnist("mnist_train.csv")
    n_samples = len(mnist_data)

    yane = NeuroEvolution()
    yane.configure(n_inputs=784, n_outputs=10, max_nodes=200, max_connections=1000)
    yane.set_resource_limits(max_process_gb=2.0)
    yane.set_min_fitness(n_samples)
    yane.set_max_iterations(100)

    def evaluate(genome):
        correct = 0
        for row in mnist_data:
            label = int(row[0])
            pixels = (row[1:] / 255.0).tolist()
            outputs = genome.forward(pixels)
            if outputs.index(max(outputs)) == label:
                correct += 1
        return correct

    yane.train(evaluate)

    best = yane.get_best()
    print(f"Best accuracy: {best.fitness}/{n_samples}")


if __name__ == "__main__":
    main()
