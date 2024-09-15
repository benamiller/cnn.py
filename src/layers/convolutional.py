import numpy as np


class Convolutional:
    def __init__(filters, filter_size, channels):
        self.weights = np.random.randn(filters, filter_size[0], filter_size[1], channels) * 0.01
        self.bias = np.random.randn(1, filters)
        self.input = None
        self.output = None

