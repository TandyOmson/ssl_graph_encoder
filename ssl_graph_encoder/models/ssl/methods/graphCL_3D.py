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
                 device,
                 other_confs_file=None,
                 ):
        # other_conf_file created in preprocessing and matched to best conformers
        # use scripts/prepare_3D_from_smiles_embed.py

        conf_data, conf_slices = torch.load(other_confs_file, map_location="cpu")
        conf_dataset = InMemoryDataset()
        conf_dataset.data, conf_dataset.slices = conf_data, conf_slices
        
        # fast lookup
        other_confs_dict = defaultdict(list)
        for idx, g in enumerate(conf_dataset):
            other_confs_dict[int(g.sample_id)].append(idx)

        projector = MLP(encoder_out_dim)
        aug_1 = AlternativeConformer(other_confs_dict, conf_dataset)
        aug_2 = AlternativeConformer(other_confs_dict, conf_dataset)
        loss_fn = InfoNCE(temperature=0.5, normalize=True)

        super().__init__(encoder_out_dim, device, projector, [aug_1, aug_2], loss_fn)
        if len(self.augmentors) != 2:
            raise NotImplementedError
        self.aug_1 = aug_1
        self.aug_2 = aug_2
        
    def training_step(self, batch, encoder):
        """ Data may be a batch
        """
        view_1, view_2 = batch

        device = next(encoder.parameters()).device

        h1 = encoder(view_1.to(device, non_blocking=True))
        h2 = encoder(view_2.to(device, non_blocking=True))

        z1 = self.projector(h1)
        z2 = self.projector(h2)

        loss = self.loss_fn([z1, z2])
        return loss
