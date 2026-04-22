""" Abstract base class for pure encoder - generates embeddings from graph data (2D or 3D)
"""
import torch
from abc import ABC, abstractmethod

class GraphEncoder(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, data):
        """
        Returns graph-level embeddings
        """
        pass