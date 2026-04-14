""" Pretraining embeddings (unsupervised)
    "Molecules that are similar should have similar embeddings"
"""
import logging
import warnings

from training.training_helpers import load_class, filter_class_config

log = logging.getLogger(__name__)

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

    enc_args = filter_class_config(encoderClass, **config["encoder"]["kwargs"])
    encoder = encoderClass(feat_dim, embed_dim, **enc_args)
    ssl_args = filter_class_config(sslClass, **config["ssl_nn"]["kwargs"])
    ssl = sslClass(encoder.output_dim, **ssl_args)

    return GraphEncoder(encoder, ssl)

class PretrainTrainer:
    """ Trainer for self-supervised pretraining molecule embeddings
    """

    def __init__(self, device, config):
        self.epochs = config["pretrain"]["epochs"]
        self.device = device

        optim_cfg = config["pretrain"]["optimizer"]
        self.optim_class = load_class(optim_cfg["class_path"])
        self.optim_kwargs = optim_cfg.get("kwargs", {})

    def fit(self, graph_encoder_ssl, dataloader):
        """ Construct optimizer from config, train encoder in-place
        """
        graph_encoder_ssl.encoder = graph_encoder_ssl.encoder.to(self.device)
        graph_encoder_ssl.ssl = graph_encoder_ssl.ssl.to(self.device)
        graph_encoder_ssl.encoder.train()

        optimizer = self.optim_class(graph_encoder_ssl.encoder.parameters(), **self.optim_kwargs)

        # This is a horrible way of catching an irrelevant warning for a (not even real) bug that is written into dig (converts tensor to float for printing)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", 
                                    message="Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.", 
                                    category=UserWarning
                                    )
            for _ in graph_encoder_ssl.ssl.train(graph_encoder_ssl.encoder, dataloader, optimizer, epochs=self.epochs):
                pass

        return graph_encoder_ssl.encoder
