from models.ssl.base.projection import ProjectionHead

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(ProjectionHead):
    """ MLP projection network for transforming representation space to contrastive space 
    """
    def __init__(
            self,
            in_dim,
            out_dim=128,
            hidden_dim=None,
            num_layers=2,
            dropout=0.0,
            activation="relu",
            use_layernorm=False,
            normalise=True,
    ):
        super().__init__(in_dim, out_dim, normalise)

        hidden_dim = hidden_dim or in_dim

        act_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "leaky_relu": lambda: nn.LeakyReLU(0.01),
            "tanh": nn.Tanh,
            "identity": nn.Identity,
        }
        if activation not in act_map:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []

        # Optional normalization layer before hidden layers
        if use_layernorm:
            layers.append(nn.LayerNorm(in_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_map[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalise:
            return F.normalize(self.net(x), dim=-1)
        else:
            return self.net(x)