""" Pairwise radial basis function encoder
    Baseline 1: Does atomic position alone solve my task?
"""

from ssl_graph_encoder.models.encoders.base import GraphEncoder
import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph
from torch.nn import Linear, SiLU
from torch_geometric.nn import global_mean_pool, global_add_pool

class PairwiseRBFEncoder(GraphEncoder):
    """ 3D pairwise radial basis function encoder
    """
    def __init__(self, feat_dim, embed_dim, hidden_channels=64, cutoff=5.0, num_gaussians=50, mlp_layers=3):
        super().__init__(feat_dim, embed_dim)     

        # cutoff of pairwise distances is same as cutoff defining the limit of the rbf centres in the radial basis expansion
        self.rbf_layer = RBFLayer(num_basis=num_gaussians, cutoff=cutoff)

        self.cutoff = cutoff

        # MLP (defined per graph)
        layers = []
        in_dim = num_gaussians
        for _ in range(mlp_layers-1):
            layers.append(Linear(in_dim, hidden_channels))
            layers.append(SiLU()) # sigmoid linear unit activation "swish"
            in_dim = hidden_channels
        layers.append(Linear(in_dim, embed_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, batch):
        pos = batch.pos
        batch_index = batch.batch

        # edge indices are all pairwise distances with a cutoff r
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch_index, loop=False)

        # all start and end indices for bonds
        row, col = edge_index
        # gives pairwise atomic distances
        dists = (pos[row] - pos[col]).norm(dim=-1)

        # expand over
        rbf = self.rbf_layer(dists)

        node_rbf = torch.zeros(
            (pos.size(0), rbf.size(-1)),
            device=pos.device
        )
        node_rbf.index_add_(0, row, rbf)

        graph_feat = global_mean_pool(node_rbf, batch_index)
        emb = self.mlp(graph_feat)

        return emb

class RBFLayer(nn.Module):
    def __init__(self, num_basis=32, cutoff=5.0):
        super().__init__()
        self.num_basis = num_basis # number of gaussians
        self.cutoff = cutoff # defines centre

        # gamma (width param/smoothness) based on num gaussians and cutoff
        self.gamma = 1/(2*((cutoff/num_basis)**2)) # may choose to tune this

        centres = torch.linspace(0, cutoff, num_basis)
        self.register_buffer("centres", centres)

    def forward(self, d):
        """
        d: distances tensor, shape [num_edges]
        returns: RBF expansion, shape [num_edges, num_basis]
        """
        d_expanded = d.unsqueeze(-1) # converts to column vector (1, num_bonds)
        rbf = torch.exp(-self.gamma * (d_expanded - self.centres) ** 2)
        return rbf