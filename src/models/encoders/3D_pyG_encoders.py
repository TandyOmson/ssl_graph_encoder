"""
Pure encoders for generating embeddings from graph data (2D and 3D)
Abstract base class is the outline
Can mainly import models from torch_geometric.nn.models
"""

from models.encoders.base import GraphEncoder
from torch_geometric.nn.models import SchNet

class SchNetEncoder(GraphEncoder):
    """ 3D SchNet Adapted from torch_geometric
    """
    def __init__(self, hidden_channels=128, num_interactions=6):
        super().__init__()
        self.model = SchNet(
            hidden_channels=hidden_channels,
            num_interactions=num_interactions,
            readout="add",
        )

    def forward(self, data):
        # PyG SchNet expects atomic numbers and positions
        return self.model(data.x, data.pos, data.batch)
