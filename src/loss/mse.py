import numpy as np

class MSE:
    def __init__(self, input, labels):
        self.input = input
        self.labels = labels

    def get_mean_squared_error(self, input=self.input, labels=self.labels):
        
        aggregating_delta = 0

        for i, input in enumerate(self.inputs):
            aggregating_delta += (input - self.labels[i])**2

        return aggregating_delta


