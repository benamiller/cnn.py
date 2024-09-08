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

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, gradients):
        gradients = x
        for layer in reversed(self.layers):
            gradients = layer.backward(gradients)

        return gradients

    def train(self, X, Y, epochs, batch_size):
        for epoch in range(epochs):
            for i in range(0, len(x), batch_size):
                x = X[i:i + batch_size]
                y = X[i:i + batch_size]

                forward_output = self.forward(x)

                loss = self.loss.forward(forward_output, y)

                output_gradient = self.loss.backward()
                self.backward(forward_output)

                self.optimizer.update(self.layers)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")


    def predict(self, X):
        return self.forward(X)

