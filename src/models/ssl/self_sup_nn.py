"""
NN for self supervised pretraining of molecule embeddings (contains train(), behaves like criterion from pytorch)
from dig (dive into graphs):
graphCL
GRACE
InfoGraph
MVGRL
Can also choose from JSE_loss and NCE_loss objectives, depending on the NN
"""

from dig.sslgraph.method import Contrastive
from dig.sslgraph.method.contrastive.views_fn import NodeAttrMask, EdgePerturbation, UniformSample, RWSample, RandomView

class GraphCL(Contrastive):

    def __init__(self,
                 dim,
                 aug_1=None,
                 aug_2=None,
                 aug_ratio=0.2,
                 objective='JSE'
                 ):

        views_fn = []
        self.objective = objective

        for aug in [aug_1, aug_2]:
            if aug is None:
                views_fn.append(lambda x: x)
            elif aug == 'dropN':
                views_fn.append(UniformSample(ratio=aug_ratio))
            elif aug == 'permE':
                views_fn.append(EdgePerturbation(ratio=aug_ratio))
            elif aug == 'subgraph':
                views_fn.append(RWSample(ratio=aug_ratio))
            elif aug == 'maskN':
                views_fn.append(NodeAttrMask(mask_ratio=aug_ratio))
            elif aug == 'random2':
                canditates = [UniformSample(ratio=aug_ratio),
                              RWSample(ratio=aug_ratio),]
                views_fn.append(RandomView(canditates))
            elif aug == 'random4':
                canditates = [UniformSample(ratio=aug_ratio),
                              RWSample(ratio=aug_ratio),
                              EdgePerturbation(ratio=aug_ratio)]
                views_fn.append(RandomView(canditates))
            elif aug == 'random3':
                canditates = [UniformSample(ratio=aug_ratio),
                              RWSample(ratio=aug_ratio),
                              EdgePerturbation(ratio=aug_ratio),
                              NodeAttrMask(mask_ratio=aug_ratio)]
                views_fn.append(RandomView(canditates))
            else:
                raise Exception
                
        super(GraphCL, self).__init__(objective=self.objective,
                                      views_fn=views_fn,
                                      z_dim=dim,
                                      proj='MLP', # irrelevant, removed after pretraining
                                      )

    def train(self, encoders, data_loader, optimizer, epochs, per_epoch_out=False):
        # GraphCL removes projection heads after pre-training
        for enc, proj in super(GraphCL, self).train(encoders, data_loader,
                                                    optimizer, epochs, per_epoch_out):
            yield enc