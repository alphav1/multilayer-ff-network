import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class BaseNetwork:
    """Base class for a Multilayer Feed-Forward Network."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_hidden_layers: int, initialization_method: str = 'xavier'):
        # network dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers

        # for initialization
        # storing weights and biases corresponding to each dimension
        self.params: Dict[str, torch.Tensor] = {}
        self.grads: Dict[str, torch.Tensor] = {}  # gradients for changes
        # cache for storing details for backpropagation
        self.cache: Dict[str, torch.Tensor] = {}

        self._initialize_parameters(initialization_method)

        # batch normalization parameters (placeholder)

    def _initialize_parameters(self, method: str):
        # initialize the base parameters - weights and biases
        layer_dims = [self.input_dim] + \
                     [self.hidden_dim] * self.num_hidden_layers + \
                     [self.output_dim]  # number of total dimensions & layers

        for i in range(1, len(layer_dims)):  # loop through weight matrices
            prev_dim, current_dim = layer_dims[i-1], layer_dims[i]
            f_in = prev_dim
            f_out = current_dim

            if method == 'xavier':
                # uniform xavier initialization, from lecture slides
                # from the formula wij ~ U[-sqrt(6 / (f_in + f_out)), sqrt(6 / (f_in + f_out))]
                limit = (6.0 / (f_in + f_out)) ** 0.5
                W = torch.empty(prev_dim, current_dim)
                nn.init.uniform_(W, -limit, limit)
            elif method == 'he':
                std = (2.0 / f_in) ** 0.5  # w ~ N(0, sqrt(1/ fin))
                W = torch.randn(prev_dim, current_dim) * std
            else:
                std = 1.0 / (f_in ** 0.5)
                W = torch.randn(prev_dim, current_dim) * std

            # We use nn.init to get the tensors, then register them for manual gradients.
            b = torch.zeros(current_dim)

            # The parameters must be registered as Tensors with requires_grad=True
            # to be recognized by torch.optim
            self.params[f'W{i}'] = nn.Parameter(W, requires_grad=True)
            self.params[f'b{i}'] = nn.Parameter(b, requires_grad=True)

            # Store references to make access cleaner for optimization step
            setattr(self, f'W{i}', self.params[f'W{i}'])
            setattr(self, f'b{i}', self.params[f'b{i}'])

            # Initialize gradients to None
            setattr(self, f'dW{i}', None)
            setattr(self, f'db{i}', None)

        print(
            f"Network parameters initialized with {method.capitalize()} for {self.num_hidden_layers} hidden layer(s).")

    # Activation functions
    @staticmethod
    def sigmoid(Z: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.exp(-Z))

    @staticmethod
    def relu(Z: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.tensor(0.0), Z)

    @staticmethod
    def tanh(Z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(Z)

    # Derivatives of the activation functions
    @staticmethod
    def sigmoid_derivative(Z: torch.Tensor) -> torch.Tensor:
        s = 1 / (1 + torch.exp(-Z))
        return s * (1 - s)

    @staticmethod
    def relu_derivative(Z: torch.Tensor) -> torch.Tensor:
        return (Z > 0).float()

    @staticmethod
    def tanh_derivative(Z: torch.Tensor) -> torch.Tensor:
        return 1 - torch.tanh(Z)**2

    @staticmethod
    def softmax(Z: torch.Tensor) -> torch.Tensor:
        """ Computes the Softmax activation for multi-class classification. """
        # Subtract max for numerical stability
        exp_Z = torch.exp(Z - torch.max(Z, dim=1, keepdim=True)[0])
        return exp_Z / torch.sum(exp_Z, dim=1, keepdim=True)

    @staticmethod
    def get_parameters(self) -> List[nn.Parameter]:
        """Returns a list of all nn.Parameter objects for PyTorch optimizer."""
        return list(self.params.values())

    def forward(self, X: torch.Tensor, hidden_activation: str = 'sigmoid', output_activation: str = 'softmax') -> torch.Tensor:
        self.cache.clear()
        self.cache['A0'] = X
        A = X

        activation_funcs = {
            'relu': self.relu,
            'sigmoid': self.sigmoid,
            'tanh': self.tanh,
            'softmax': self.softmax,
            'identity': lambda z: z
        }

        # Hidden layers
        for i in range(1, self.num_hidden_layers + 1):
            W = self.params[f'W{i}']
            b = self.params[f'b{i}']
            Z = A @ W + b
            A = activation_funcs[hidden_activation](Z)

            self.cache[f'Z{i}'] = Z
            self.cache[f'A{i}'] = A

        # Output layer
        output_layer_idx = self.num_hidden_layers + 1
        W_out = self.params[f'W{output_layer_idx}']
        b_out = self.params[f'b{output_layer_idx}']
        Z_out = A @ W_out + b_out
        A_out = activation_funcs[output_activation](Z_out)

        self.cache[f'Z{output_layer_idx}'] = Z_out
        self.cache[f'A{output_layer_idx}'] = A_out

        return A_out

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss. This method should be overridden by a subclass.
        """
        raise NotImplementedError(
            "Loss function must be implemented in a subclass.")

    def backward(self, y_true: torch.Tensor, hidden_activation: str = 'sigmoid', loss_mode: str = 'softmax_ce'):
        """
        Performs the backward pass, calculating gradients based on the specified loss mode.

        Args:
            y_true (torch.Tensor): The true labels for the batch.
            hidden_activation (str): The activation function used in the hidden layers.
            loss_mode (str): Determines how the initial gradient is calculated.
                             'softmax_ce' for Softmax with Cross-Entropy loss.
                             'mse' for Mean Squared Error loss with a linear output.
        """
        self.grads.clear()
        num_layers = self.num_hidden_layers + 1
        m = y_true.shape[0]

        # --- 1. Calculate the initial gradient dZ for the OUTPUT LAYER ---
        y_pred = self.cache[f'A{num_layers}']

        if loss_mode == 'softmax_ce':
            # For Softmax + Cross-Entropy, dZ = y_pred - y_true_one_hot
            y_true_one_hot = nn.functional.one_hot(
                y_true, num_classes=self.output_dim).float()
            dZ = y_pred - y_true_one_hot
        elif loss_mode == 'mse':
            # For linear output + MSE, dZ = y_pred - y_true
            # Assumes y_true is already in the correct format (not class indices)
            dZ = y_pred - y_true
        else:
            raise ValueError(f"Unsupported loss_mode: {loss_mode}")

        # Get the activation from the previous layer (last hidden layer)
        A_prev = self.cache[f'A{num_layers - 1}']

        # Gradient for the last layer's weights and biases
        self.grads[f'dW{num_layers}'] = (1/m) * A_prev.T @ dZ
        self.grads[f'db{num_layers}'] = (1/m) * torch.sum(dZ, dim=0)

        # Assign gradient to the .grad attribute for the optimizer
        self.params[f'W{num_layers}'].grad = self.grads[f'dW{num_layers}']
        self.params[f'b{num_layers}'].grad = self.grads[f'db{num_layers}']

        # --- 2. Propagate gradient to HIDDEN LAYERS ---
        dA_prev = dZ @ self.params[f'W{num_layers}'].T

        # Map activation function names to their derivative functions
        activation_derivatives = {
            'relu': self.relu_derivative,
            'sigmoid': self.sigmoid_derivative,
            'tanh': self.tanh_derivative
        }
        g_prime = activation_derivatives[hidden_activation]

        # Loop backwards from the last hidden layer to the first
        for i in range(self.num_hidden_layers, 0, -1):
            # dZ for the current hidden layer: dA_prev * g'(Z)
            # This is the chain rule: (dL/dA) * (dA/dZ) = dL/dZ
            Z = self.cache[f'Z{i}']
            dZ = dA_prev * g_prime(Z)

            # Get activation from the previous layer (or input X if it's the first hidden layer)
            A_prev = self.cache[f'A{i - 1}']

            # Calculate gradients for W and b for the current layer i
            self.grads[f'dW{i}'] = (1/m) * A_prev.T @ dZ
            self.grads[f'db{i}'] = (1/m) * torch.sum(dZ, dim=0)

            # Assign to .grad attribute
            self.params[f'W{i}'].grad = self.grads[f'dW{i}']
            self.params[f'b{i}'].grad = self.grads[f'db{i}']

            # If we are not at the first layer, calculate the gradient to pass to the next layer back
            if i > 1:
                dA_prev = dZ @ self.params[f'W{i}'].T


class MyFFNetworkForClassification(BaseNetwork):
    """A feed-forward network specialized for classification tasks."""

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Computes the Cross-Entropy loss for multi-class classification.

        Args:
            y_pred (torch.Tensor): The predicted probabilities from the softmax output.
            y_true (torch.Tensor): The true labels (as class indices).

        Returns:
            torch.Tensor: The mean cross-entropy loss.
        """
        m = y_true.shape[0]
        # We use y_true as indices to select the predicted probabilities for the correct classes.
        log_likelihood = -torch.log(y_pred[range(m), y_true])
        loss = torch.sum(log_likelihood) / m
        return loss


class MyFFNetworkForRegression(BaseNetwork):
    """A feed-forward network specialized for regression tasks."""

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Computes the Mean Squared Error (MSE) loss for regression.

        Args:
            y_pred (torch.Tensor): The predicted values.
            y_true (torch.Tensor): The true values.

        Returns:
            torch.Tensor: The mean MSE loss.
        """
        m = y_true.shape[0]
        loss = torch.sum((y_pred - y_true)**2) / m
        return loss
