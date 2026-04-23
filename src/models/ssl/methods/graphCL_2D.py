"""
torch_geometric implementation of GraphCL 
"""
from models.ssl.base.ssl import ContrastiveSSL
from models.ssl.projection.mlp import MLP
from models.ssl.augmentation.graph_views import DigNodeDropping, DigSubgraph
from models.ssl.loss.nt_xent import InfoNCE

class GraphCL(ContrastiveSSL):
    def __init__(self, 
                 encoder_out_dim,
                 aug_1=None,
                 aug_2=None,
                 aug_ratio=0.1
                 ):
        
        projector = MLP(encoder_out_dim)
        augmentors = [DigNodeDropping(ratio=aug_ratio), 
                      DigSubgraph(ratio=aug_ratio),
                      ]
        loss_fn = InfoNCE(temperature=0.5, normalize=True)

        super().__init__(encoder_out_dim, projector, augmentors, loss_fn)
        if len(self.augmentors) != 2:
            raise NotImplementedError
        self.aug_1 , self.aug_2 = self.augmentors
        
    def training_step(self, data, encoder):
        """ Data may be a batch
        """
        view_1 = self.aug_1(data)
        view_2 = self.aug_2(data)

        h1 = encoder(view_1)
        h2 = encoder(view_2)

        z1 = self.projector(h1)
        z2 = self.projector(h2)

        loss = self.loss_fn([z1, z2])

        return loss 