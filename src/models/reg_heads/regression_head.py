""" Regression head for supervised learning of molecule embeddings (after unsupervised pretraining)
"""

import torch
import torch.nn as nn

class RegressionHead(nn.Module):
    """
    Flexible regression head for pretrained embeddings.

    Supports:
    - linear (num_layers=1)
    - MLP (num_layers > 1)
    - optional normalization (useful for pretrained embeddings)
    """

    def __init__(self, 
                 embed_dim,
                 output_dim=1,
                 hidden_dim=None,
                 num_layers=3,
                 dropout=0.0,
                 activation="relu",
                 use_layernorm=False
                 ):
        super().__init__()

        hidden_dim = hidden_dim or embed_dim

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
        in_dim = embed_dim

        # Optional normalization layer before hidden layers
        if use_layernorm:
            layers.append(nn.LayerNorm(embed_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_map[activation]())  # new instance each time
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))

        self.model = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, embed_dim) or (embed_dim,)
        returns: (batch,) or (batch, output_dim)
        """
        # single input sample
        if x.dim() == 1:
            x = x.unsqueeze(0)

        out = self.model(x)

        # fix output shape for single output regression
        if self.output_dim == 1:
            out = out.squeeze(-1)

        return out