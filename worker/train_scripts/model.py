"""
A real PyTorch MLP, architected on the fly from whatever the user
submitted: number and width of hidden layers, per-layer dropout, hidden
activation, and output activation. Nothing here is fixed or pre-built --
every job constructs its own nn.Module from scratch.
"""
from typing import List

import torch
from torch import nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


class ConfigurableMLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        hidden_layers: List[int],
        dropout: List[float],
        activation: str,
        output_activation: str,
    ):
        super().__init__()
        act_cls = _ACTIVATIONS[activation]

        layers = []
        prev_size = n_features
        for width, drop_rate in zip(hidden_layers, dropout):
            layers.append(nn.Linear(prev_size, width))
            layers.append(act_cls())
            if drop_rate > 0:
                layers.append(nn.Dropout(drop_rate))
            prev_size = width

        self.hidden = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_size, n_outputs)
        self.output_activation = output_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits/values. Output activation is applied
        separately (see apply_output_activation) so that loss functions
        which expect raw logits (e.g. cross-entropy) still work correctly.
        """
        x = self.hidden(x)
        return self.output_layer(x)

    def apply_output_activation(self, logits: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "softmax":
            return torch.softmax(logits, dim=-1)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(logits)
        return logits  # linear
