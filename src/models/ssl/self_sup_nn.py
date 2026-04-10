"""
NN for self supervised pretraining of molecule embeddings (contains train(), behaves like criterion from pytorch)
from dig (dive into graphs):
graphCL
GRACE
InfoGraph
MVGRL
Can also choose from JSE_loss and NCE_loss objectives, depending on the NN
"""

import torch.nn as nn
from dig.sslgraph.method import Contrastive
from dig.sslgraph.method.contrastive.views_fn import NodeAttrMask, EdgePerturbation, UniformSample, RWSample, RandomView

class GraphCL(Contrastive):

    def __init__(self, dim, **config):

        views_fn = []

        for aug in [config.get("aug_1"), config.get("aug_2")]:
            if aug is None:
                views_fn.append(lambda x: x)
            elif aug == 'dropN':
                views_fn.append(UniformSample(ratio=config.get("aug_ratio", 0.2)))
            elif aug == 'permE':
                views_fn.append(EdgePerturbation(ratio=config.get("aug_ratio", 0.2)))
            elif aug == 'subgraph':
                views_fn.append(RWSample(ratio=config.get("aug_ratio", 0.2)))
            elif aug == 'maskN':
                views_fn.append(NodeAttrMask(mask_ratio=config.get("aug_ratio", 0.2)))
            elif aug == 'random2':
                canditates = [UniformSample(ratio=config.get("aug_ratio", 0.2)),
                              RWSample(ratio=config.get("aug_ratio", 0.2))]
                views_fn.append(RandomView(canditates))
            elif aug == 'random4':
                canditates = [UniformSample(ratio=config.get("aug_ratio", 0.2)),
                              RWSample(ratio=config.get("aug_ratio", 0.2)),
                              EdgePerturbation(ratio=config.get("aug_ratio", 0.2))]
                views_fn.append(RandomView(canditates))
            elif aug == 'random3':
                canditates = [UniformSample(ratio=config.get("aug_ratio", 0.2)),
                              RWSample(ratio=config.get("aug_ratio", 0.2)),
                              EdgePerturbation(ratio=config.get("aug_ratio", 0.2)),
                              NodeAttrMask(mask_ratio=config.get("aug_ratio", 0.2))]
                views_fn.append(RandomView(canditates))
            else:
                raise Exception
                
        super(GraphCL, self).__init__(objective=config.get("objective"),
                                      views_fn=views_fn,
                                      z_dim=dim,
                                      proj='MLP', # irrelevant, removed after pretraining
                                      )

    def train(self, encoders, data_loader, optimizer, epochs, per_epoch_out=False):
        # GraphCL removes projection heads after pre-training
        for enc, proj in super(GraphCL, self).train(encoders, data_loader,
                                                    optimizer, epochs, per_epoch_out):
            yield enc