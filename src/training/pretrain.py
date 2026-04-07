""" Pretraining embeddings (unsupervised)
    "Molecules that are similar should have similar embeddings"
"""
import logging
import importlib

import torch

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def load_class(class_path):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

class GraphEncoder:
    """ Encoder class containing encoder and training method
    """
    def __init__(self, encoder, ssl):
        self.encoder = encoder
        self.ssl = ssl

def build_pretrain_encoder(feat_dim, embed_dim, config):
    """ Generic factory function
        Build an encoder model
    """
    encoderClass = load_class(config["encoder"]["class_path"])
    sslClass = load_class(config["ssl_nn"]["class_path"])

    encoder = encoderClass(feat_dim, embed_dim, **config["encoder"])
    ssl = sslClass(embed_dim, **config["ssl_nn"])

    return GraphEncoder(encoder, ssl)

class PretrainTrainer:
    """ Trainer for self-supervised pretraining molecule embeddings
    """

    def __init__(self, lr, epochs):
        self.lr = float(lr)
        self.epochs = int(epochs)

    def fit(self, graph_encoder_ssl, dataloader):
        """ Train encoder in-place
        """
        # set training mode (not necessary but nice to read)
        graph_encoder_ssl.encoder.train()
        optimizer = torch.optim.Adam(graph_encoder_ssl.encoder.parameters(), lr=self.lr)

        for _ in graph_encoder_ssl.ssl.train(graph_encoder_ssl.encoder, dataloader, optimizer, epochs=self.epochs):
            pass

        return graph_encoder_ssl.encoder

def pretrain(graph_encoder_ssl, dataloader, config):
    """ self-supervised pretraining molecule embeddings
    """

    optimizer = torch.optim.Adam(graph_encoder_ssl.encoder.parameters(), lr=float(config["pretrain"]["lr"]))
    
    graph_encoder_ssl.encoder.train()
    log.info("Start pretraining...")

    for enc in graph_encoder_ssl.ssl.train(graph_encoder_ssl.encoder, dataloader, optimizer, epochs=config["pretrain"]["epochs"]):
        pass

    return graph_encoder_ssl.encoder