"""
A multi-layer perceptron (MLP) built from raw tensors, without torch.nn.

Architecture (identical to models/mlp_nn.py):

    image (28x28) -> flatten -> [784]
        -> Linear -> [128] -> ReLU
        -> Linear -> [64]  -> ReLU
        -> Linear -> [10]     (one score, or "logit", per digit)

Every Linear layer is just a matrix multiply plus a bias: y = x @ W + b.
"""

import math

import torch


def init_weight(n_in, n_out):
    """
    Create a weight matrix of shape [n_in, n_out] with small random values.

    We use "He initialization": random numbers scaled by sqrt(2 / n_in).
    This keeps the signal from shrinking or exploding as it passes through
    ReLU layers. Starting with all zeros would not work, because every
    neuron would compute the same thing and learn the same thing.

    requires_grad=True tells PyTorch to track this tensor so it can compute
    gradients for it when we call loss.backward().
    """
    return (torch.randn(n_in, n_out) * math.sqrt(2 / n_in)).requires_grad_()


def init_bias(n_out):
    return torch.zeros(n_out, requires_grad=True)


def relu(x):
    """ReLU: keep positive values, replace negative values with 0."""
    return x.clamp(min=0)


class ScratchMLP:
    def __init__(self):
        # Layer 1: 784 inputs -> 128 hidden neurons
        self.W1 = init_weight(784, 128)
        self.b1 = init_bias(128)
        # Layer 2: 128 -> 64 hidden neurons
        self.W2 = init_weight(128, 64)
        self.b2 = init_bias(64)
        # Layer 3: 64 -> 10 outputs (one per digit)
        self.W3 = init_weight(64, 10)
        self.b3 = init_bias(10)

    def forward(self, images):
        # images: [batch, 1, 28, 28] -> x: [batch, 784]
        x = images.reshape(images.shape[0], -1)

        x = relu(x @ self.W1 + self.b1)   # [batch, 128]
        x = relu(x @ self.W2 + self.b2)   # [batch, 64]
        logits = x @ self.W3 + self.b3    # [batch, 10]
        # No softmax here: the loss function applies it for us.
        return logits

    def __call__(self, images):
        # Lets us write model(images) instead of model.forward(images),
        # the same way nn.Module models work.
        return self.forward(images)

    def parameters(self):
        """All the tensors the model learns."""
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def state_dict(self):
        """The learned tensors, by name, for saving to disk."""
        return {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "W3": self.W3, "b3": self.b3,
        }

    def load_state_dict(self, state):
        """Replace this model's tensors with ones loaded from disk."""
        for name, tensor in state.items():
            setattr(self, name, tensor.clone().requires_grad_())
