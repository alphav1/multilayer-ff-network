import numpy as np
import pytorch


def models():
    class BaseNetwork(object):
        def __init__(self, input_dim=784, hidden_dim=15, output_dim=10, num_hidden_layers=2):
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.num_hidden_layers = num_hidden_layers

        def feedforward(self, a):
            pass

    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))
