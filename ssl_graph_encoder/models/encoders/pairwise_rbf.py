""" Pairwise radial basis function encoder
    Baseline 1: Does atomic position alone solve my task?
"""

from ssl_graph_encoder.models.encoders.base import GraphEncoder
import torch
import torch.nn as nn
from torch.nn import Linear, SiLU
from torch_geometric.nn import global_mean_pool, global_add_pool

class PairwiseRBFEncoder(GraphEncoder):
    """ 3D pairwise radial basis function encoder
    """
    def __init__(self, feat_dim, embed_dim, hidden_channels=64, cutoff=10.0, num_gaussians=50, mlp_layers=3):
        super().__init__(feat_dim, embed_dim)     

        self.rbf_layer = RBFLayer(num_basis=num_gaussians, cutoff=cutoff)

        # MLP
        layers = []
        in_dim = num_gaussians

        for _ in range(mlp_layers-1):
            layers.append(Linear(in_dim, hidden_channels))
            layers.append(SiLU())
            in_dim = hidden_channels
        layers.append(Linear(in_dim, embed_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, batch):
        pos = batch.pos
        edge_index = batch.edge_index
        batch_index = batch.batch

        row, col = edge_index
        dists = (pos[row] - pos[col]).norm(dim=-1)

        # expand over
        rbf = self.rbf_layer(dists)
        # cutoff weighting
        mask = (dists < self.rbf_layer.cutoff).float()
        rbf = rbf * mask.unsqueeze(-1)

        node_rbf = torch.zeros(
            (pos.size(0), rbf.size(-1)),
            device=pos.device
        )
        node_rbf.index_add_(0, row, rbf)

        graph_feat = global_mean_pool(node_rbf, batch_index)
        emb = self.mlp(graph_feat)

        return emb

class RBFLayer(nn.Module):
    def __init__(self, num_basis=32, cutoff=5.0, gamma=10.0):
        super().__init__()
        self.num_basis = num_basis # number of gaussians
        self.cutoff = cutoff # defines centres
        self.gamma = gamma # defines width param

        centers = torch.linspace(0, cutoff, num_basis)
        self.register_buffer("centers", centers)

    def forward(self, d):
        """
        d: distances tensor, shape [...,]
        returns: RBF expansion, shape [..., num_basis]
        """
        d_expanded = d.unsqueeze(-1) 
        rbf = torch.exp(-self.gamma * (d_expanded - self.centers) ** 2)
        return rbf