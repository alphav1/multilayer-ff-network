import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class BaseNetwork:
    """Base class for a Multilayer Feed-Forward Network."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_hidden_layers: int, initialization_method: str = 'xavier', use_batch_norm: bool = True):
        # network dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        # for batch normalization
        self.use_batch_norm = use_batch_norm

        # for initialization
        # storing weights and biases corresponding to each dimension
        self.params: Dict[str, torch.Tensor] = {}
        self.grads: Dict[str, torch.Tensor] = {}  # gradients for changes
        # cache for storing details for backpropagation
        self.cache: Dict[str, torch.Tensor] = {}

        # Batch normalization parameters and cache
        self.bn_params: Dict[str, torch.Tensor] = {}
        self.bn_cache: Dict[str, torch.Tensor] = {}

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
                # allocates a tensor of shape (prev_dim, curr_dim)
                W = torch.empty(prev_dim, current_dim)
                # fills in-place the tensor W with uniform distributions
                nn.init.uniform_(W, -limit, limit)
            elif method == 'he':
                std = (2.0 / f_in) ** 0.5  # w ~ N(0, sqrt(1/ fin))
                # creates a 2d tensor of size prev_dim x curr_dim,
                W = torch.randn(prev_dim, current_dim) * std
                # samples from a normal distribution with mean 0, std deviation 1,
                # std scales all the values by a standard deviation factor std
            else:
                std = 1.0 / (f_in ** 0.5)
                W = torch.randn(prev_dim, current_dim) * std

            # We use nn.init to get the tensors, then register them for manual gradients.
            # tensor filled with zeros, of size current dimensions
            b = torch.zeros(current_dim)

            # The parameters must be registered as Tensors with requires_grad=True
            # to be recognized by torch.optim
            # trainable parameters W, B for layer i
            self.params[f'W{i}'] = nn.Parameter(W, requires_grad=True)
            # grad tells PyTorch to track the gradients
            self.params[f'b{i}'] = nn.Parameter(b, requires_grad=True)

            # Store references to make access cleaner for optimization step
            # creates instance attributes dynamically
            setattr(self, f'W{i}', self.params[f'W{i}'])
            setattr(self, f'b{i}', self.params[f'b{i}'])  # called like self.W1

            # Initialize gradients to None
            setattr(self, f'dW{i}', None)
            setattr(self, f'db{i}', None)

            # Initialize Batch Norm parameters if enabled (not for the output layer)
            if self.use_batch_norm and i < len(layer_dims) - 1:
                # Learnable scale (gamma) and shift (beta) parameters
                self.bn_params[f'gamma{i}'] = nn.Parameter(
                    torch.ones(current_dim), requires_grad=True)
                self.bn_params[f'beta{i}'] = nn.Parameter(
                    torch.zeros(current_dim), requires_grad=True)
                setattr(self, f'gamma{i}', self.bn_params[f'gamma{i}'])
                setattr(self, f'beta{i}', self.bn_params[f'beta{i}'])

                # Non-learnable running mean and variance for inference
                self.register_buffer(
                    f'running_mean{i}', torch.zeros(current_dim))
                self.register_buffer(
                    f'running_var{i}', torch.ones(current_dim))

        print(
            f"Network parameters initialized with {method.capitalize()} for {self.num_hidden_layers} hidden layer(s).")
        if self.use_batch_norm:
            print("Batch Normalization is enabled.")

    # Helper to register buffers like running_mean/var
    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    # Activation functions
    @staticmethod
    def sigmoid(Z: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.exp(-Z))  # 1 / 1 + e^(-z)

    @staticmethod
    def relu(Z: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.tensor(0.0), Z)  # z if z >= 0, 0 if z < 0

    @staticmethod
    def tanh(Z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(Z)  # tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))

    # Derivatives of the activation functions
    @staticmethod
    def sigmoid_derivative(Z: torch.Tensor) -> torch.Tensor:
        s = 1 / (1 + torch.exp(-Z))  # chain rule - base function
        return s * (1 - s)  # chain rule used

    @staticmethod
    def relu_derivative(Z: torch.Tensor) -> torch.Tensor:
        return (Z > 0).float()  # derivative is 1 if z > 0, 0 if z <= 0

    @staticmethod
    def tanh_derivative(Z: torch.Tensor) -> torch.Tensor:
        return 1 - torch.tanh(Z)**2  # 1 / cosh^2(z) = 1 - tanh^2(z)

    @staticmethod
    def softmax(Z: torch.Tensor) -> torch.Tensor:
        """ Computes the Softmax activation for multi-class classification. """
        # convert the vector of numbers into a probability distribution where all values sum to 1
        # Z - torch.max(Z, dim=1, keepdim=True)[0] --> subtract the maximum value along dimension 1 (rows) from each element
        # doesn't change the final probabilities, but prevents numerical overflow for exponents
        # exp_Z = torch.exp(Z - max_Z) applies e^x to each element - subtraction prevents too large values
        exp_Z = torch.exp(Z - torch.max(Z, dim=1, keepdim=True)[0])
        # lastly, normalization divides each exponentaial by the sum of all exponentials in its row, ensuring all values sum to 1
        # dim= 1 indicates operating along rows, and keepdim=True maintains the dimensions
        return exp_Z / torch.sum(exp_Z, dim=1, keepdim=True)

    @staticmethod
    def get_parameters(self) -> List[nn.Parameter]:
        return list(self.params.values())

    def forward(self, X: torch.Tensor, hidden_activation: str = 'sigmoid', output_activation: str = 'softmax', training: bool = True) -> torch.Tensor:
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
            W, b = self.params[f'W{i}'], self.params[f'b{i}']
            Z = A @ W + b
            self.cache[f'Z_pre_bn{i}'] = Z  # Cache Z before BN

            # z_normalize = gamma * (z-score-normalized) + beta
            if self.use_batch_norm:
                # initialize the params
                gamma, beta = self.bn_params[f'gamma{i}'], self.bn_params[f'beta{i}']
                running_mean, running_var = getattr(
                    self, f'running_mean{i}'), getattr(self, f'running_var{i}')
                epsilon = 1e-5  # if variance is zero, divide by zero, need to add a small value

                # during training use the formula z_score-normalized = (Z - batch_mean) / (sqrt(batch_variance + epsilon))
                if training:
                    # E(x)
                    batch_mean = Z.mean(dim=0)
                    batch_var = Z.var(dim=0, unbiased=False)

                    # by using the running mean and running var instead of the batch mean and batch var, we create a smooth stable average
                    # of what the mean and variance look like across the entire dataset - best for new guesses during the testing phase
                    momentum = 0.9  # hyperparameter used to updated the running mean and running variance during training

                    # during the evaluation (when training=False), we can't calculate a batch mean or batch variance for a small sample
                    # we need instead, an estimate of the mean and variance for each neuron that represents the overall statistics of the training data

                    running_mean.data = momentum * \
                        running_mean.data + \
                        (1 - momentum) * \
                        batch_mean  # exponential moving average of the mean
                    # update the mean = keep e.g. 0.9 of the old mean, and we take 10% of the mean from the current batch
                    running_var.data = momentum * \
                        running_var.data + (1 - momentum) * batch_var

                    # and then use the z_normalized formula
                    Z_hat = (Z - batch_mean) / torch.sqrt(batch_var + epsilon)

                    self.bn_cache[f'Z_hat{i}'] = Z_hat
                    self.bn_cache[f'batch_var{i}'] = batch_var
                    self.bn_cache[f'epsilon{i}'] = epsilon
                else:
                    # if not training, do not update the running ones,
                    # just use them as they are - for unsees, individual samples
                    Z_hat = (Z - running_mean) / \
                        torch.sqrt(running_var + epsilon)

                # normalize with gamma and beta
                Z = gamma * Z_hat + beta

            # for the inner hidden layers
            self.cache[f'Z{i}'] = Z
            A = activation_funcs[hidden_activation](Z)
            self.cache[f'A{i}'] = A

        # for the final output layer, once the loop finishes
        # layer identification - get the output layer
        output_layer_idx = self.num_hidden_layers + 1
        # retrieve the weights and biases
        W_out, b_out = self.params[f'W{output_layer_idx}'], self.params[f'b{output_layer_idx}']
        Z_out = A @ W_out + b_out  # perform the linear computation for the layer - Z = AW + b
        # apply the specific activation function to the linear transformation
        A_out = activation_funcs[output_activation](Z_out)
        # this is important since the output layer computation often uses a different activation function

        # this is the output of the linear calculation of the layer (initial, raw score)
        self.cache[f'Z{output_layer_idx}'] = Z_out
        # this is the activation - firing strength
        self.cache[f'A{output_layer_idx}'] = A_out

        return A_out  # this is the final prediction of the neural network, after all forward loops

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
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
        self.grads.clear()  # clear the collection stored in self.grads() - gradients for changes
        num_layers = self.num_hidden_layers + 1
        # get the size of the first dimension of the y_true array - the ground truth, the correct answers to the input data
        m = y_true.shape[0]

        # --- 1. Calculate the initial gradient dZ for the OUTPUT LAYER ---
        # y_pred is the predicted output from the cache, specifically the activations from the final layer
        y_pred = self.cache[f'A{num_layers}']

        # now, we calculate the dZ = the derivative of the Loss with respect to Z
        # it represents how much the final loss (the error between the expected, perfect output, and the predicted) change
        # if we change the pre-activation output Z of the final layer

        # this works differently for different losses -
        # this tells us the direction and magnitude of the error right before the final activation function was applied

        if loss_mode == 'softmax_ce':
            # For Softmax + Cross-Entropy, dZ = y_pred - y_true_one_hot
            # here, we use the mathematical simplification of the derivative calculations
            # the full cross-entropy loss is calculated with L_batch = - 1/N sum N i=1 [log(p_i, c_i)]
            # then, the softmax is calculated with softmax(z)_i = e_z_i / sum j e_z_j
            # when we calculate the loss partial derivative with respect to the pre-activation output,
            # (if i make a small change to the raw score of z_k, how much does total loss change)
            # the calculation ends up simplifiying to softmax(z)_k - y_k
            # which is why the dZ = y_pred (contains the softmax probabilities) - y_true_one_hot (one_hot encoded truth)
            # which gives us the gradient directly
            y_true_one_hot = nn.functional.one_hot(
                y_true, num_classes=self.output_dim).float()
            dZ = y_pred - y_true_one_hot
        elif loss_mode == 'mse':
            # For linear output + MSE, dZ = y_pred - y_true
            # Assumes y_true is already in the correct format (not class indices)
            dZ = y_pred - y_true
        elif loss_mode == 'sigmoid_bce':
            dZ = y_pred - y_true
        else:
            raise ValueError(f"Unsupported loss_mode: {loss_mode}")

        # first, we get the activation from the previous layer - last hidden layer before output
        A_prev = self.cache[f'A{num_layers - 1}']

        # then get the gradient for the last (previous) layer's weights and biases
        # calculate the gradient for the output layer's weights dW
        self.grads[f'dW{num_layers}'] = (1/m) * A_prev.T @ dZ
        # chain rule (dLoss / dZ) * (dZ / dW)
        # (dLoss / dZ) is dZ
        # (dZ / dW) is how the output Z changes with respect to the weights W - simply the input to the layer - A_prev
        # then transpose for multiplication, and average across all batch samples (1/m)

        # then we calculate the gradient for the output layers bias db
        # the gradient of the loss with respect to the bias is just dZ itself
        # we sum dZ gradients for all samples in the batch, and average across by m (1/m)
        self.grads[f'db{num_layers}'] = (1/m) * torch.sum(dZ, dim=0)

        # assign gradient to the .grad attribute for the optimizer
        self.params[f'W{num_layers}'].grad = self.grads[f'dW{num_layers}']
        self.params[f'b{num_layers}'].grad = self.grads[f'db{num_layers}']
        # when we then call optimizer.step() the optimizer will look for this parameter and update it accordingly

        # --- 2. Propagate gradient to HIDDEN LAYERS ---
        dA_prev = dZ @ self.params[f'W{num_layers}'].T
        # this calculates the gradient of the loss with respect to the activations of the previous layers (dA_prev)
        # this again is the chain rule (dLoss / dZ) * (dZ / dA_prev)
        # dL/dz is our dZ
        # dZ/dA_prev (how the output Z changes with respect tot the input A_prev) is the transpose of the weights W.T

        # this new dA_prev is now the error signal that is fed into the for loop - repeating for each hidden layer, moving backwards

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
            # Z = self.cache[f'Z{i}']
            # dZ = dA_prev * g_prime(Z)

            # # Get activation from the previous layer (or input X if it's the first hidden layer)
            # A_prev = self.cache[f'A{i - 1}']

            # # Calculate gradients for W and b for the current layer i
            # self.grads[f'dW{i}'] = (1/m) * A_prev.T @ dZ
            # self.grads[f'db{i}'] = (1/m) * torch.sum(dZ, dim=0)

            # # Assign to .grad attribute
            # self.params[f'W{i}'].grad = self.grads[f'dW{i}']
            # self.params[f'b{i}'].grad = self.grads[f'db{i}']

            # # If we are not at the first layer, calculate the gradient to pass to the next layer back
            # if i > 1:
            #     dA_prev = dZ @ self.params[f'W{i}'].T

            dZ_activated = dA_prev * g_prime(self.cache[f'Z{i}'])

            if self.use_batch_norm:
                Z_hat = self.bn_cache[f'Z_hat{i}']
                gamma = self.bn_cache[f'gamma{i}']
                batch_var = self.bn_cache[f'batch_var{i}']
                epsilon = self.bn_cache[f'epsilon{i}']
                Z_pre_bn = self.cache[f'Z_pre_bn{i}']

                # Gradients for gamma and beta
                dbeta = torch.sum(dZ_activated, dim=0)
                dgamma = torch.sum(dZ_activated * Z_hat, dim=0)
                self.bn_params[f'beta{i}'].grad = dbeta
                self.bn_params[f'gamma{i}'].grad = dgamma

                # Backprop through normalization
                dZ_hat = dZ_activated * gamma
                inv_std = 1. / torch.sqrt(batch_var + epsilon)
                dvar = torch.sum(dZ_hat * (Z_pre_bn - Z_pre_bn.mean(dim=0))
                                 * -0.5 * (batch_var + epsilon)**(-1.5), dim=0)
                dmean = torch.sum(dZ_hat * -inv_std, dim=0) + dvar * \
                    torch.mean(-2. * (Z_pre_bn - Z_pre_bn.mean(dim=0)), dim=0)
                dZ = dZ_hat * inv_std + \
                    (dvar * 2 * (Z_pre_bn - Z_pre_bn.mean(dim=0)) / m) + (dmean / m)
            else:
                dZ = dZ_activated

            A_prev = self.cache[f'A{i - 1}']
            self.grads[f'dW{i}'] = (1/m) * A_prev.T @ dZ
            self.grads[f'db{i}'] = (1/m) * torch.sum(dZ, dim=0)
            self.params[f'W{i}'].grad = self.grads[f'dW{i}']
            self.params[f'b{i}'].grad = self.grads[f'db{i}']

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

        # softmax for multiclass classification is used for problems like mnist, and the loss is calculated with cross-entropy
        # cross-entropy loss at a single point is calculated with the following
        # : because the true distribution has a probability of 1 for the correct class c, and 0 for others, the formula becomes
        # L (loss for the single data point) = -log(p_c) (p_c is the models predicted probability for the correct class)

        # then, we calculate the the formula for a batch of samples N and take the average:
        # L_batch = - 1/N sum N i=1 [log(p_i, c_i)]

        m = y_true.shape[0]
        # We use y_true as indices to select the predicted probabilities for the correct classes.

        # This is the key part: it calculates log(p_i, c_i) for every sample
        # y_pred[range(m), y_true] cleverly selects the predicted probability
        # of the correct class for each sample in the batch.

        log_likelihood = -torch.log(y_pred[range(m), y_true] + 1e-12)

        # This is the sum and average: (1/N) * Σ(...)
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


class MyFFNetworkForBinaryClassification(BaseNetwork):
    """A feed-forward network specialized for binary classification tasks."""

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        m = y_true.shape[0]
        epsilon = 1e-12
        y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)
        bce_loss = - (y_true * torch.log(y_pred) +
                      (1 - y_true) * torch.log(1 - y_pred))
        return torch.sum(bce_loss) / m
