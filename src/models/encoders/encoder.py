"""
Encoder for molecule embeddings (may make this class based later)
GNN from Dig (Dive into Graphs)
encoder GNN choices:
- gcn (graph convolutional network)
- gin (graph isomorphic network)
- resgcn (GCN with residual based attention mechanism)
"""
from dig.sslgraph.utils import Encoder

class digEncoder(Encoder):
    """ Dig encoder subclass for flexibility later
    """
    def __init__(self, feat_dim, hidden_dim, gnn="resgcn", num_layers=5):
        super().__init__(feat_dim, 
                         hidden_dim, # this is embed_dim
                         gnn=gnn, 
                         n_layers=num_layers
                         )
        if gnn == "gcn" or gnn == "gin":
            self.output_dim = hidden_dim * num_layers
        else:
            self.output_dim = hidden_dim