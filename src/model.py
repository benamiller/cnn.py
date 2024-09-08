import numpy as np

class Model:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss):
        valid_losses = ['mse', 'cross_entropy']

        if loss not in valid_losses:
            return Error("LosS not valid. Ensure loss is one of: " + valid_losses)

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
        gradients = X
        for layer in reversed(self.layers):
            gradients = layer.backward(gradients)

        return gradients

    def train(self, iterations):


    def predict(self, X):
        return self.forward(X)

