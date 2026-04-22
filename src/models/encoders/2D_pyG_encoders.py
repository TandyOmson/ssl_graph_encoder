"""
Pure encoders for generating embeddings from graph data (2D and 3D)
Abstract base class is the outline
Can mainly import models from torch_geometric.nn.models
"""
from models.encoders.base import GraphEncoder

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class GCNEncoder(GraphEncoder):
    """ 2D Graph convolutional network
    """
    def __init__(self, feat_dim, hidden_dim, num_layers=3):
        super().__init__(feat_dim, hidden_dim)
        self.output_dim = hidden_dim*num_layers
        self.num_layers = num_layers
        
        # build network
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(feat_dim, hidden_dim))
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = nn.ReLU()(x)
            xs.append(x)

        xpool = [global_mean_pool(x, batch) for x in xs]
        global_rep = torch.cat(xpool, 1)

        return global_rep
    
from torch_geometric.nn import GINEConv

class GINEEncoder(torch.nn.Module):
    """ 2D GINE implementation, allows for using edge features
    """
    def __init__(self, in_dim, hidden_dim, num_layers=3):
        super().__init__(in_dim, hidden_dim)
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList([
            GINEConv(torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            ))
            for _ in range(num_layers)
        ])

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        return global_mean_pool(x, batch)
