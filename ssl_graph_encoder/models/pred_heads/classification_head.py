""" Classification head
    Basically the same as regression head, but may diverge later
    If I add class weighting, label smoothing, etc.
"""

import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    """
    Flexible classification head for pretrained embeddings.

    Supports:
    - linear (num_layers=1)
    - MLP (num_layers > 1)
    - optional embeddig normalization (useful for pretrained embeddings)
    """

    def __init__(self,
                 embed_dim,
                 num_classes=1,
                 hidden_dim=None,
                 num_layers=3,
                 dropout=0.0,
                 activation="relu",
                 use_layernorm=False,
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
            layers.append(act_map[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # Logit output layer
        layers.append(nn.Linear(in_dim, num_classes))

        self.model = nn.Sequential(*layers)
        self.output_dim = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, embed_dim) or (embed_dim,)
        returns:
            - (batch,)            if num_classes == 1
            - (batch, num_classes) otherwise

            Outputs are logits, use sigmoid/softmax *outside* the model if needed
        """
        # if single input sample
        if x.dim() == 1:
            x = x.unsqueeze(0)

        logits = self.model(x)

        if self.output_dim == 1:
            logits = logits.squeeze(-1)

        return logits

