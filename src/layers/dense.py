import numpy as np


class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.random.randn(1, output_size)
        self.input = None
        self.output = None

    def forward(self, X):
        self.input = X
        self.output = np.dot(self.weights, self.input) + self.bias
