import numpy as np


def models():
    class BaseNetwork(object):
        def __init__(self, input_dim=784, hidden_dim=15, output_dim=10, num_hidden_layers=2):
            """ Initialize the BaseNetwork. """
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.num_hidden_layers = num_hidden_layers
            # layer sizes calculation
            layer_sizes = [self.input_dim] + [self.hidden_dim] * \
                self.num_hidden_layers + [self.output_dim]
            # initializing random weights and biases
            self.weights = [np.random.randn(
                layer_sizes[i+1], layer_sizes[i]) for i in range(len(layer_sizes)-1)]
            self.biases = [np.random.randn(layer_sizes[i+1])
                           for i in range(len(layer_sizes)-1)]

        def feedforward(self, a):
            """ Return the output of the network if "a" is input. """
            for b, w in zip(self.biases, self.weights):
                a = sigmoid(np.dot(w, a)+b)
            return a

    def sigmoid(z):
        """ Compute the sigmoid (logistic) function element-wise."""
        return 1.0/(1.0+np.exp(-z))
