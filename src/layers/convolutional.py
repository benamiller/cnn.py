import numpy as np


class Convolutional:
    def __init__(input_shape, filters, filter_size, stride=1, padding=0):
        self.input_channels, self.input_height, self.input_width = input_shape
        self.filter_size = filter_size
        self.filters = filters
        self.stride = stride
        self.padding = padding

        self.output_height = ((self.input_height - self.filter_size + 2 * padding) // stride) + 1
        self.output_width = ((self.input_width - self.filter_size + 2 * padding) // stride) + 1

        self.weights = np.random.randn(filters, filter_size, filter_size, self.input_channels) * 0.01
        self.bias = np.random.randn(1, filters)
        self.input = None
        self.output = None

