""" Abstract base class for contrastive self-supervised learning training loop classes
    SSL training loop exists to define the contrastive objective for training an encoder, this consists of:
    - Augmentations
    - Projection heads for tranforming from encoded representations of views to latent space where constrastive objective is calculated
    - Contrastive loss
"""
from abc import ABC, abstractmethod

class ContrastiveSSL(ABC):
    def __init__(self, encoder, projector, augmentors, loss_fn):
        self.encoder = encoder
        self.projector = projector
        self.augmentor = augmentors
        self.loss_fn = loss_fn

    @abstractmethod
    def training_step(self, batch):
        pass