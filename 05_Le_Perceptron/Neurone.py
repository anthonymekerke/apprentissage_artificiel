import numpy as np

class Neurone:
    def __init__(self, ls):
        self.bias = 0.5
        self.output = 0
        self.weights = np.random.uniform(0, 1, 2)
        self.learning_step = ls

    def compute_output(self, example):
        sigma = 0
        for i in range(len(example)):
            sigma += (self.weights[i] * example[i])

        sigma -= self.bias

        self.output = 1 if sigma > 0 else -1

    def update(self, example, etiquette):
        self.bias = self.bias + self.learning_step * (etiquette - self.output) * -0.5

        for i in range(self.weights.size):
            self.weights[i] = self.weights[i] + self.learning_step * (etiquette - self.output) * example[i]
