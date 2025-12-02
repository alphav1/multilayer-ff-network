import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import os


class Autoencoder:
    """
    Autoencoder with 2 encoder layers and 2 decoder layers.
    Uses manual gradient computation for learning.
    """

    def __init__(self, input_dim: int, hidden_dims: list,
                 initialization_method: str = 'xavier',
                 use_batch_norm: bool = False,
                 add_noise: bool = False,
                 noise_std: float = 0.1):
        """
        Args:
            input_dim: Dimension of input (e.g., 784 for 28x28 images)
            hidden_dims: List of hidden layer dimensions [encoder1, encoder2/bottleneck, decoder1]
                        decoder2 output will be same as input_dim
            initialization_method: Weight initialization ('xavier', 'he', 'constant')
            use_batch_norm: Whether to use batch normalization
            add_noise: Whether to add noise to input (denoising autoencoder)
            noise_std: Standard deviation of Gaussian noise
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_batch_norm
        self.add_noise = add_noise
        self.noise_std = noise_std

        # Verify we have correct number of dimensions
        assert len(
            hidden_dims) == 3, "hidden_dims should be [encoder1, bottleneck, decoder1]"

        # Full architecture: input -> encoder1 -> bottleneck -> decoder1 -> output
        self.layer_dims = [input_dim] + hidden_dims + [input_dim]

        # Store parameters and gradients
        self.params: Dict[str, torch.Tensor] = {}
        self.grads: Dict[str, torch.Tensor] = {}
        self.cache: Dict[str, torch.Tensor] = {}

        # Batch normalization parameters
        self.bn_params: Dict[str, torch.Tensor] = {}
        self.bn_cache: Dict[str, torch.Tensor] = {}

        self._initialize_parameters(initialization_method)

    def _initialize_parameters(self, method: str):
        """Initialize weights and biases for all layers"""
        for i in range(1, len(self.layer_dims)):
            prev_dim, current_dim = self.layer_dims[i-1], self.layer_dims[i]

            if method == 'xavier':
                limit = (6.0 / (prev_dim + current_dim)) ** 0.5
                W = torch.empty(prev_dim, current_dim)
                nn.init.uniform_(W, -limit, limit)
            elif method == 'he':
                std = (2.0 / prev_dim) ** 0.5
                W = torch.randn(prev_dim, current_dim) * std
            elif method == 'constant':
                W = torch.full((prev_dim, current_dim), 0.01)
            else:
                std = 1.0 / (prev_dim ** 0.5)
                W = torch.randn(prev_dim, current_dim) * std

            b = torch.zeros(current_dim)

            self.params[f'W{i}'] = nn.Parameter(W, requires_grad=True)
            self.params[f'b{i}'] = nn.Parameter(b, requires_grad=True)

            # Batch normalization (except for output layer)
            if self.use_batch_norm and i < len(self.layer_dims) - 1:
                self.bn_params[f'gamma{i}'] = nn.Parameter(
                    torch.ones(current_dim), requires_grad=True)
                self.bn_params[f'beta{i}'] = nn.Parameter(
                    torch.zeros(current_dim), requires_grad=True)
                self._register_buffer(
                    f'running_mean{i}', torch.zeros(current_dim))
                self._register_buffer(
                    f'running_var{i}', torch.ones(current_dim))

        print(
            f"Autoencoder initialized with {method.capitalize()} initialization")
        print(f"Architecture: {' -> '.join(map(str, self.layer_dims))}")

    def _register_buffer(self, name, tensor):
        """Helper to register non-learnable buffers"""
        setattr(self, name, tensor)

    @staticmethod
    def relu(Z: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.tensor(0.0), Z)

    @staticmethod
    def relu_derivative(Z: torch.Tensor) -> torch.Tensor:
        return (Z > 0).float()

    @staticmethod
    def sigmoid(Z: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.exp(-Z))

    @staticmethod
    def sigmoid_derivative(Z: torch.Tensor) -> torch.Tensor:
        s = 1 / (1 + torch.exp(-Z))
        return s * (1 - s)

    def encode(self, X: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Encode input through encoder layers to get bottleneck representation.
        """
        A = X
        num_encoder_layers = 2  # encoder1 and encoder2 (bottleneck)

        for i in range(1, num_encoder_layers + 1):
            W, b = self.params[f'W{i}'], self.params[f'b{i}']
            Z = A @ W + b

            if self.use_batch_norm:
                gamma, beta = self.bn_params[f'gamma{i}'], self.bn_params[f'beta{i}']
                running_mean = getattr(self, f'running_mean{i}')
                running_var = getattr(self, f'running_var{i}')
                epsilon = 1e-5

                if training:
                    batch_mean = Z.mean(dim=0)
                    batch_var = Z.var(dim=0, unbiased=False)
                    momentum = 0.9

                    running_mean.data = momentum * \
                        running_mean.data + (1 - momentum) * batch_mean
                    running_var.data = momentum * \
                        running_var.data + (1 - momentum) * batch_var

                    Z_hat = (Z - batch_mean) / torch.sqrt(batch_var + epsilon)
                else:
                    Z_hat = (Z - running_mean) / \
                        torch.sqrt(running_var + epsilon)

                Z = gamma * Z_hat + beta

            A = self.relu(Z)

        return A  # Bottleneck representation

    def forward(self, X: torch.Tensor, activation: str = 'relu',
                output_activation: str = 'sigmoid', training: bool = True) -> torch.Tensor:
        """
        Full forward pass through encoder and decoder.
        Returns reconstructed input.
        """
        self.cache.clear()

        # Add noise if enabled (denoising autoencoder)
        if training and self.add_noise:
            noise = torch.randn_like(X) * self.noise_std
            X_noisy = X + noise
            self.cache['X_original'] = X  # Store clean version for loss
            X = X_noisy

        self.cache['A0'] = X
        A = X

        activation_funcs = {
            'relu': self.relu,
            'sigmoid': self.sigmoid,
            'identity': lambda z: z
        }

        act_func = activation_funcs[activation]
        out_func = activation_funcs[output_activation]

        # Forward through all layers
        num_layers = len(self.layer_dims) - 1
        for i in range(1, num_layers + 1):
            W, b = self.params[f'W{i}'], self.params[f'b{i}']
            Z = A @ W + b
            self.cache[f'Z_pre_bn{i}'] = Z

            # Batch normalization (not on output layer)
            if self.use_batch_norm and i < num_layers:
                gamma, beta = self.bn_params[f'gamma{i}'], self.bn_params[f'beta{i}']
                running_mean = getattr(self, f'running_mean{i}')
                running_var = getattr(self, f'running_var{i}')
                epsilon = 1e-5

                if training:
                    batch_mean = Z.mean(dim=0)
                    batch_var = Z.var(dim=0, unbiased=False)
                    momentum = 0.9

                    running_mean.data = momentum * \
                        running_mean.data + (1 - momentum) * batch_mean
                    running_var.data = momentum * \
                        running_var.data + (1 - momentum) * batch_var

                    Z_hat = (Z - batch_mean) / torch.sqrt(batch_var + epsilon)

                    self.bn_cache[f'Z_hat{i}'] = Z_hat
                    self.bn_cache[f'batch_var{i}'] = batch_var
                    self.bn_cache[f'epsilon{i}'] = epsilon
                    self.bn_cache[f'gamma{i}'] = gamma
                    self.bn_cache[f'beta{i}'] = beta
                else:
                    Z_hat = (Z - running_mean) / \
                        torch.sqrt(running_var + epsilon)

                Z = gamma * Z_hat + beta

            self.cache[f'Z{i}'] = Z

            # Use different activation for output layer
            if i == num_layers:
                A = out_func(Z)
            else:
                A = act_func(Z)

            self.cache[f'A{i}'] = A

        return A

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Mean Squared Error loss for reconstruction.
        """
        # If using denoising autoencoder, compare reconstruction with clean input
        if 'X_original' in self.cache:
            y_true = self.cache['X_original']

        m = y_true.shape[0]
        loss = torch.sum((y_pred - y_true)**2) / m
        return loss

    def backward(self, y_true: torch.Tensor, activation: str = 'relu'):
        """
        Backward pass to compute gradients.
        """
        self.grads.clear()

        # If using denoising autoencoder, use clean input for gradient
        if 'X_original' in self.cache:
            y_true = self.cache['X_original']

        num_layers = len(self.layer_dims) - 1
        m = y_true.shape[0]

        # Output layer gradient (MSE with sigmoid output)
        y_pred = self.cache[f'A{num_layers}']
        # Gradient of MSE: 2/m * (y_pred - y_true)
        # Combined with sigmoid derivative for efficiency
        dZ = (2 / m) * (y_pred - y_true) * \
            self.sigmoid_derivative(self.cache[f'Z{num_layers}'])

        # Activation derivative function
        if activation == 'relu':
            g_prime = self.relu_derivative
        elif activation == 'sigmoid':
            g_prime = self.sigmoid_derivative
        else:
            def g_prime(z): return torch.ones_like(z)

        # Backpropagate through all layers
        for i in range(num_layers, 0, -1):
            A_prev = self.cache[f'A{i-1}']

            # Compute weight and bias gradients
            self.grads[f'dW{i}'] = A_prev.T @ dZ
            self.grads[f'db{i}'] = torch.sum(dZ, dim=0)

            self.params[f'W{i}'].grad = self.grads[f'dW{i}']
            self.params[f'b{i}'].grad = self.grads[f'db{i}']

            # Propagate to previous layer (if not at input)
            if i > 1:
                dA_prev = dZ @ self.params[f'W{i}'].T

                # Handle batch normalization
                if self.use_batch_norm and i - 1 < num_layers:
                    dZ_activated = dA_prev * g_prime(self.cache[f'Z{i-1}'])

                    # Backprop through batch norm
                    Z_hat = self.bn_cache[f'Z_hat{i-1}']
                    gamma = self.bn_cache[f'gamma{i-1}']
                    batch_var = self.bn_cache[f'batch_var{i-1}']
                    epsilon = self.bn_cache[f'epsilon{i-1}']
                    Z_pre_bn = self.cache[f'Z_pre_bn{i-1}']

                    dbeta = torch.sum(dZ_activated, dim=0)
                    dgamma = torch.sum(dZ_activated * Z_hat, dim=0)
                    self.bn_params[f'beta{i-1}'].grad = dbeta
                    self.bn_params[f'gamma{i-1}'].grad = dgamma

                    dZ_hat = dZ_activated * gamma
                    inv_std = 1. / torch.sqrt(batch_var + epsilon)
                    dvar = torch.sum(dZ_hat * (Z_pre_bn - Z_pre_bn.mean(dim=0))
                                     * -0.5 * (batch_var + epsilon)**(-1.5), dim=0)
                    dmean = torch.sum(dZ_hat * -inv_std, dim=0) + dvar * \
                        torch.mean(-2. * (Z_pre_bn -
                                   Z_pre_bn.mean(dim=0)), dim=0)
                    dZ = dZ_hat * inv_std + \
                        (dvar * 2 * (Z_pre_bn - Z_pre_bn.mean(dim=0)) / m) + (dmean / m)
                else:
                    dZ = dA_prev * g_prime(self.cache[f'Z{i-1}'])

    def save_model(self, filepath: str):
        """Save model parameters to file"""
        save_dict = {
            'params': {k: v.data for k, v in self.params.items()},
            'bn_params': {k: v.data for k, v in self.bn_params.items()} if self.use_batch_norm else {},
            'config': {
                'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'use_batch_norm': self.use_batch_norm,
                'add_noise': self.add_noise,
                'noise_std': self.noise_std
            }
        }

        # Also save running statistics if using batch norm
        if self.use_batch_norm:
            running_stats = {}
            for i in range(1, len(self.layer_dims) - 1):
                running_stats[f'running_mean{i}'] = getattr(
                    self, f'running_mean{i}')
                running_stats[f'running_var{i}'] = getattr(
                    self, f'running_var{i}')
            save_dict['running_stats'] = running_stats

        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str, initialization_method: str = 'xavier'):
        """Load model parameters from file"""
        checkpoint = torch.load(filepath)
        config = checkpoint['config']

        model = cls(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            initialization_method=initialization_method,
            use_batch_norm=config['use_batch_norm'],
            add_noise=config['add_noise'],
            noise_std=config['noise_std']
        )

        # Load parameters
        for k, v in checkpoint['params'].items():
            model.params[k].data = v

        if config['use_batch_norm']:
            for k, v in checkpoint['bn_params'].items():
                model.bn_params[k].data = v

            # Load running statistics
            if 'running_stats' in checkpoint:
                for k, v in checkpoint['running_stats'].items():
                    setattr(model, k, v)

        print(f"Model loaded from {filepath}")
        return model
