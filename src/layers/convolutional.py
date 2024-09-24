import numpy as np


class Convolutional:
    def __init__(self, input_shape, filters, filter_size, stride=1, padding=0):
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
        self.padded_input
        self.output = None

    def forward(self, input):
        self.input = input
        batch_size = input.shape[0]
        
        if self.padding > 0:
            self.padded_input = np.pad(input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            self.padded_input = input
        
        self.outputs = np.zeros((batch_size, self.filters, self.output_height, self.output_width))

        for h in range(self.output_height):
            for w in range(self.output_width):
                filter_shadow = self.padded_input[:,:, h*self.stride:h*self.stride+self.filter_size, w*self.stride:w*self.stride+self.filter_size]

                for f in range(self.filters):
                    self.output[:, f, h, w] = np.sum(filter_shadow * self.weights[f], axis=(1, 2, 3)) + self.bias[f]

        return self.output

    def backward(self, gradient, learning_rate):
        batch_size = gradient.shape[0]
        input_gradient = np.zeros_like(self.padded_input)
        weights_gradient = np.zeros_like(self.weights)
        bias_gradient = np.zeros_like(self.bias)

        for h in range(self.output_height):
            for w in range(self.output_width):
                filter_shadow = self.padded_input[:,:, h*self.stride:h*self.stride+self.filter_size, w*self.stride:w*self.stride+self.filter_size] 

        return input_gradient * self.weights



        # Return the previous gradient, multiplied by our weights to get how previous layer affects our input, which in turn affects output of computation graph
