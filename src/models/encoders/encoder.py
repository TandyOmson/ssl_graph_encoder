"""
Encoder for molecule embeddings (may make this class based later)
GNN from Dig (Dive into Graphs)
"""
from dig.sslgraph.utils import Encoder

class digEncoder(Encoder):
    """ Dig encoder subclass for flexibility later
    """
    def __init__(self, feat_dim, hidden_dim, **config):
        super().__init__(feat_dim, 
                         hidden_dim, # this is embed_dim
                         gnn=config.get("gnn", "resgcn"), 
                         n_layer=config.get("num_layers", 5)
                         )
        # need to do other self.prop statements if extending later