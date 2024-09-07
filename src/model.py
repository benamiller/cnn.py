import numpy as np

class Model:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss):
        self.loss = loss

    def set_optimizer(self, optimizer):
        valid_optimizers = ['adam', 'sgd']

        if optimizer not in valid_optimizers:
            return Error("Optimizer not valid. Ensure optimizer is one of: " + valid_optimizers)

        self.optimizer = optimizer

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, gradients):

    def train(self, iterations):

    def predict(self, X):
        return self.forward(X)

