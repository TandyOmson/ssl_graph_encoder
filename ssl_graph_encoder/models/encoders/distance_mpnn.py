""" GCN with distance weighting 
    Baseline 2: Uses atomic number and distance based edge weights
    Simple message passing
""" 
from ssl_graph_encoder.models.encoders.base import GraphEncoder
import torch
import torch.nn as nn
from torch.nn import Linear, SiLU
from torch_geometric.nn import global_mean_pool, global_add_pool

class DistanceGCNEncoder(GraphEncoder):
    """Simple distance-weighted GCN"""

    def __init__(
        self,
        feat_dim,
        embed_dim,
        hidden_channels=64,
        num_layers=4,
    ):
        super().__init__(feat_dim, embed_dim)

        # atomic number embedding
        self.embedding = nn.Embedding(100, hidden_channels)

        # GCN layers
        self.layers = nn.ModuleList([
            DistanceGCNLayer(hidden_channels)
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
        pos = batch.pos                     # [N, 3]
        z = batch.z                         # [N]
        edge_index = batch.edge_index
        batch_idx = batch.batch

        # initial node features
        h = self.embedding(z)               # [N, hidden]

        row, col = edge_index

        # distances on edges
        dists = (pos[row] - pos[col]).norm(dim=-1)  # [E]

        # message passing
        for layer in self.layers:
            h = layer(h, edge_index, dists)
            h = self.act(h)

        # graph pooling
        graph_emb = global_mean_pool(h, batch_idx)

        # projection
        emb = self.mlp(graph_emb)

        return emb

class DistanceGCNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lin = Linear(hidden_dim, hidden_dim)

    def forward(self, h, edge_index, dists):
        row, col = edge_index

        # message: W h_j
        m_ij = self.lin(h[col])  # [E, hidden]

        # distance weights (Gaussian)
        w_ij = torch.exp(-dists**2).unsqueeze(-1)  # [E, 1]

        m_ij = w_ij * m_ij

        # aggregate
        out = torch.zeros_like(h)
        out.index_add_(0, row, m_ij)

        # residual update
        return h + out
