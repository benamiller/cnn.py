import numpy as np


class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.random.randn(1, output_size)
        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient):
        input_gradient = np.dot(output_gradient, self.weights.T)
        weights_gradient = np.dot(self.input.T, output_gradient)
        bias_gradient = np.sum(output_gradient, axis=0, keepDims=True)

        self.weights_gradient = weights_gradient
        self.bias_gradient = bias_gradient

        return input_gradient
