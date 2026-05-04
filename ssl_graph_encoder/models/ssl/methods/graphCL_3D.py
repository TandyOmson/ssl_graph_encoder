"""
torch_geometric implementation of GraphCL 
"""
import torch
from torch_geometric.data import InMemoryDataset
from collections import defaultdict
from ssl_graph_encoder.models.ssl.base.ssl import ContrastiveSSL
from ssl_graph_encoder.models.ssl.projection.mlp import MLP
from ssl_graph_encoder.models.ssl.augmentation.conf_views_3d import AlternativeConformer
from ssl_graph_encoder.models.ssl.loss.nt_xent import InfoNCE

class GraphCL(ContrastiveSSL):
    def __init__(self, 
                 encoder_out_dim,
                 other_confs_file=None,
                 ):
        # other_conf_file created in preprocessing and matched to best conformers
        # use scripts/prepare_3D_from_smiles_embed.py

        conf_data, conf_slices = torch.load(other_confs_file)
        conf_dataset = InMemoryDataset()
        conf_dataset.data, conf_dataset.slices = conf_data, conf_slices
        
        # fast lookup
        other_confs_dict = defaultdict(list)
        for g in conf_dataset:
            other_confs_dict[int(g.sample_id)].append(g)

        projector = MLP(encoder_out_dim)
        augmentors = [AlternativeConformer(other_confs_dict),
                      AlternativeConformer(other_confs_dict),
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