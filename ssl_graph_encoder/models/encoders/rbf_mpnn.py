""" GCN with radial basis expansion distance weighting and message passing
    Baseline 3    
"""
from ssl_graph_encoder.models.encoders.base import GraphEncoder
import torch
import torch.nn as nn
from torch.nn import Linear, SiLU
from torch_geometric.nn import global_mean_pool, global_add_pool

class RBFMPNNEncoder(GraphEncoder):
    """RBF-based message passing network (SchNet-lite)"""

    def __init__(
        self,
        feat_dim,
        embed_dim,
        hidden_channels=64,
        num_layers=4,
        num_gaussians=32,
        cutoff=5.0,
    ):
        super().__init__(feat_dim, embed_dim)

        # embeddings
        self.embedding = nn.Embedding(100, hidden_channels)

        # RBF expansion
        self.rbf_layer = RBFLayer(
            num_basis=num_gaussians,
            cutoff=cutoff
        )

        # MPNN layers
        self.layers = nn.ModuleList([
            RBFMPNNLayer(hidden_channels, num_gaussians)
            for _ in range(num_layers)
        ])

        self.act = SiLU()

        # final projection
        self.mlp = nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            SiLU(),
            Linear(hidden_channels, embed_dim)
        )

    def forward(self, batch):
        pos = batch.pos
        z = batch.z
        edge_index = batch.edge_index
        batch_idx = batch.batch

        # node features
        h = self.embedding(z)  # [N, hidden]

        row, col = edge_index

        # distances
        dists = (pos[row] - pos[col]).norm(dim=-1)  # [E]

        # RBF edge features
        rbf = self.rbf_layer(dists)  # [E, K]

        # message passing
        for layer in self.layers:
            h = layer(h, edge_index, rbf)
            h = self.act(h)

        # pooling
        graph_emb = global_mean_pool(h, batch_idx)

        # projection
        emb = self.mlp(graph_emb)

        return emb

class RBFLayer(nn.Module):
    def __init__(self, num_basis=32, cutoff=5.0, gamma=10.0):
        super().__init__()

        self.num_basis = num_basis
        self.cutoff = cutoff
        self.gamma = gamma

        centers = torch.linspace(0, cutoff, num_basis)
        self.register_buffer("centers", centers)

    def forward(self, d):
        d_expanded = d.unsqueeze(-1)  # [..., 1]
        rbf = torch.exp(-self.gamma * (d_expanded - self.centers) ** 2)
        return rbf

class RBFMPNNLayer(nn.Module):
    def __init__(self, hidden_dim, rbf_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            Linear(hidden_dim + rbf_dim, hidden_dim),
            SiLU(),
            Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h, edge_index, rbf):
        row, col = edge_index

        # concatenate neighbor features + geometric features
        m_ij = self.mlp(torch.cat([h[col], rbf], dim=-1))  # [E, hidden]

        # aggregate
        out = torch.zeros_like(h)
        out.index_add_(0, row, m_ij.to(out.dtype))

        # residual update
        return h + out
