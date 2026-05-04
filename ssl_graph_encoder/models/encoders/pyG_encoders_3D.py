"""
Pure encoders for generating embeddings from graph data (2D and 3D)
Abstract base class is the outline
Can mainly import models from torch_geometric.nn.models
"""

from ssl_graph_encoder.models.encoders.base import GraphEncoder
from torch_geometric.nn.models import SchNet
from torch_geometric.nn import global_mean_pool, global_add_pool
import torch
from torch.nn import Linear

class SchNetEncoder(GraphEncoder):
    """ 3D SchNet Adapted from torch_geometric
    """
    def __init__(self, feat_dim, embed_dim, hidden_channels=64, num_interactions=6):
        super().__init__(feat_dim, embed_dim)
        self.model = SchNetAdaptor(
            hidden_channels=hidden_channels,
            num_interactions=num_interactions,
            readout="add",
        )
        # By default, the final lin1 and lin2 have fixed dimensions
        self.model.lin1 = Linear(hidden_channels, hidden_channels//2)
        self.model.lin2 = Linear(hidden_channels//2, embed_dim)


    def forward(self, data):
        # PyG SchNet expects atomic numbers (as longs) and positions
        # In preprocessing, z must be set as a tensor of atomic nums, dtype long
        out = self.model(data.z, data.pos, data.batch)
        
        return out  
    
class SchNetAdaptor(SchNet):
    """ Subclassing SchNet to expose node embeddings
    """
    def __init__(self, hidden_channels, **args):
        super().__init__(hidden_channels=hidden_channels, **args)
        
    def forward(self, z, pos, batch=None, return_node=False):
        batch = torch.zeros_like(z) if batch is None else batch

        # initial embedding based on atomic numbers
        h = self.embedding(z)
        # adding interaction graph based on atomic number (rotationally invariant)
        edge_index, edge_weight = self.interaction_graph(pos, batch)
        # expanding rotationally invariant layer with radial basis functions
        edge_attr = self.distance_expansion(edge_weight)

        # interaction blocks
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        # atomwise layer
        h = self.lin1(h)
        # shifted softplus
        h = self.act(h)
        ## atomwise layer
        h = self.lin2(h)
        # pooling
        if self.readout == "add":
            g = global_add_pool(h, batch)
        else:
            g = global_mean_pool(h, batch)
        # graph and node level embeddings, or just graph level embeddings
        return (g, h) if return_node else g  # graph_emb, node_emb
