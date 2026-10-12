"""
The same multi-layer perceptron (MLP) as models/mlp_scratch.py, built with torch.nn.

Architecture:

    image (28x28) -> flatten -> [784]
        -> Linear -> [128] -> ReLU
        -> Linear -> [64]  -> ReLU
        -> Linear -> [10]     (one score, or "logit", per digit)

Compare this with mlp_scratch.py: nn.Linear creates and initializes the
weight and bias for us, and nn.Module keeps track of all the parameters.
"""

from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),          # [batch, 1, 28, 28] -> [batch, 784]
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),     # no softmax: the loss function applies it for us
        )

    def forward(self, images):
        return self.layers(images)
