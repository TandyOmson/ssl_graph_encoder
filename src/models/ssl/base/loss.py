""" objectives for self-supervised graph learning
    There are essentially two types of self-supervised learning,
    - Contrastive 
    (generating multiple representations and maximising mutual information)
    - Predictive (AKA Generative), using either
        - self generated labels, such as in masked node prediction
        - "microlabels" referring to properties of nodes/subgraphs that in theory effect graph properties that aren't labelled)
"""
import torch
from abc import ABC, abstractmethod

class ContrastiveLoss(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """ Compute contrastive loss
        """
        pass