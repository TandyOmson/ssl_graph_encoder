""" Encoder agnostic projection head used only during SSL
During contrastive learning, projection heads are applied to the encoded representations h, and z=g(h) (views of graphs passed through the encoder)
The representations are undergo a non-linear transformation into another latent spae where the contrastive loss is calculated.
A rationale for this is provided by Chen et al. 2020 alongside demonstrating clear benefits in their data 
THE PROJECTION HEAD IS DISCARDED AFTER TRAINING - i.e. h is the output of the encoder in downstream tasks
"""
import torch
from abc import ABC, abstractmethod

class ProjectionHead(torch.nn.Module, ABC):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project encoded representations into latent space
        for contrastive loss computation.
        """
        pass