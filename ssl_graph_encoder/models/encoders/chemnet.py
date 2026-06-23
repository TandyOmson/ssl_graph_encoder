""" ChemNet: encoder for finetuning ONLY (supervised training)
"""

from ssl_graph_encoder.models.encoders.base import GraphEncoder
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

class ChemNetEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        num_layers=4,
        num_atom_types=100,
        embed_dim=128,      # latent space
        output_dim=1        # prediction target
    ):
        super().__init__()

        self.embedding = nn.Embedding(num_atom_types, hidden_dim)

        self.layers = nn.ModuleList([
            ChemNetLayer(hidden_dim)
            for _ in range(num_layers)
        ])

        # latent projection
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU()
        )

        # supervised head
        self.pred_head = nn.Linear(embed_dim, output_dim)

    def forward(self, batch, return_embedding=False):
        x = batch.z
        edge_index = batch.edge_index
        batch_idx = batch.batch

        # initial features
        h = self.embedding(x)

        # message passing
        for layer in self.layers:
            h = layer(h, edge_index)

        # graph pooling
        graph_emb = global_mean_pool(h, batch_idx)

        # latent space
        z = self.project(graph_emb)   # [B, embed_dim]

        if return_embedding:
            return z

        # supervised prediction
        out = self.pred_head(z)
        return out

class ChemNetLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h, edge_index):
        row, col = edge_index

        # concatenate central + neighbour (important difference from GCN)
        m_ij = self.mlp(torch.cat([h[row], h[col]], dim=-1))

        out = torch.zeros_like(h)
        out.index_add_(0, row, m_ij)

        return h + out
