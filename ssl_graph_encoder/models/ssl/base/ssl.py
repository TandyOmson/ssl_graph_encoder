""" Abstract base class for contrastive self-supervised learning training loop classes
    SSL training loop exists to define the contrastive objective for training an encoder, this consists of:
    - Augmentations
    - Projection heads for tranforming from encoded representations of views to latent space where constrastive objective is calculated
    - Contrastive loss
"""
from tqdm import trange

from abc import ABC, abstractmethod
from ssl_graph_encoder.models.ssl.base.projection import ProjectionHead
from ssl_graph_encoder.models.ssl.base.augmentation import ViewAugmentor
from ssl_graph_encoder.models.ssl.base.loss import ContrastiveLoss

class ContrastiveSSL(ABC):
    def __init__(self, 
                 encoder_out_dim : int,
                 projector: ProjectionHead, 
                 augmentors: ViewAugmentor, 
                 loss_fn : ContrastiveLoss,
                 ):
        self.encoder_out_dim = encoder_out_dim
        self.projector = projector
        self.augmentors = augmentors
        self.loss_fn = loss_fn

    @abstractmethod
    def training_step(self, batch):
        """ Computes contrastive loss for a single batch
        """
        pass

    def pretrain(self, 
                 encoder, 
                 data_loader, 
                 optimizer, 
                 epochs, 
                 ):
        """ Runs self supervised pretraining on the encoder
            Must yield the encoder
        """
        encoder.train()
        self.projector.train()

        # MUST ADD PROJECTOR TO OPTIMIZER PARAMS FOR THEM TO BE TRAINED
        if self.projector is not None:
            optimizer.add_param_group({
                "params": self.projector.parameters()
            })

        with trange(epochs) as t:
            for epoch in t:
                train_loss = 0
                t.set_description('Pretraining: epoch %d' % (epoch+1))
                for batch in data_loader:
                    loss = self.training_step(batch, encoder)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss
                train_loss /= len(data_loader.dataset)
                t.set_postfix(loss=f'{train_loss:.4f}')

                # encoder must be yielded to remove projection head
                yield encoder
