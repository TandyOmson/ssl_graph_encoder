""" Abstract base class for pure encoder - generates embeddings from graph data (2D or 3D)
"""
import torch
from abc import ABC, abstractmethod

class GraphEncoder(torch.nn.Module, ABC):
    """ Generic graph encoder ABC
    """
    def __init__(self, feat_dim, embed_dim, **kwargs):
        super().__init__()

        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.config = kwargs
        self.output_dim = embed_dim # this may be overriden e.g. num_layers*embed dim in gcn or gin

    @abstractmethod
    def forward(self, data):
        """
        Returns graph-level embeddings
        """
        pass