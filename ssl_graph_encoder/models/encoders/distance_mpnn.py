""" GCN with distance weighting 
    Baseline 2: Uses atomic number and distance based edge weights
    Simple message passing
""" 
from ssl_graph_encoder.models.encoders.base import GraphEncoder
import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph
from torch.nn import Linear, SiLU
from torch_geometric.nn import global_mean_pool, global_add_pool

class DistanceGCNEncoder(GraphEncoder):
    """Simple distance-weighted GCN"""

    def __init__(
        self,
        feat_dim,
        embed_dim,
        cutoff=5.0,
        hidden_channels=64,
        num_layers=4,
    ):
        super().__init__(feat_dim, embed_dim)

        # set this to just above bond distance to get bonds only
        self.cutoff = cutoff

        # atomic number embedding (100 is a very safe upper bound for size of dictionary of tokens i.e. vocab length)
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
        batch_index = batch.batch

        # initial node features
        h = self.embedding(z)               # [N, hidden]

        # edge indices are all pairwise distances with a cutoff r
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch_index, loop=False)

        # all start and end indices for bonds
        row, col = edge_index
        # gives pairwise atomic distances
        dists = (pos[row] - pos[col]).norm(dim=-1)

        # message passing
        for layer in self.layers:
            h = layer(h, edge_index, dists)
            h = self.act(h)

        # graph pooling
        graph_emb = global_mean_pool(h, batch_index)

        # projection
        emb = self.mlp(graph_emb)

        return emb

class DistanceGCNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lin = Linear(hidden_dim, hidden_dim)

    def forward(self, h, edge_index, dists):
        row, col = edge_index

        # message (gets hidden embeddings for the source node of each edge)
        m_ij = self.lin(h[col])  # [E, hidden]
        # gaussia distance weights
        w_ij = torch.exp(-dists**2).unsqueeze(-1)  # [E, 1]
        m_ij = w_ij * m_ij

        # aggregate
        out = torch.zeros_like(h)
        out.index_add_(0, row, m_ij)

        # residual update
        return h + out
