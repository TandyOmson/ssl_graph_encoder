""" Pretraining embeddings (unsupervised)
    "Molecules that are similar should have similar embeddings"
"""
import logging
import importlib
import inspect

log = logging.getLogger(__name__)

def load_class(class_path):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def filter_class_config(cls, **config):
    sig = inspect.signature(cls.__init__)
    params = sig.parameters

    # Check whether __init__ accepts **kwargs
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in params.values()
    )

    # If it does, pass everything through
    if has_var_kw:
        return dict(config)

    # Otherwise, filter strictly
    valid_keys = set(params) - {"self"}
    return {k: v for k, v in config.items() if k in valid_keys}

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

    encoder = encoderClass(feat_dim, embed_dim, **filter_class_config(encoderClass, **config["encoder"]["kwargs"]))
    ssl = sslClass(embed_dim, **filter_class_config(sslClass, **config["ssl_nn"]["kwargs"]))

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
        """ Train encoder in-place
        """
        graph_encoder_ssl.encoder = graph_encoder_ssl.encoder.to(self.device)
        graph_encoder_ssl.ssl = graph_encoder_ssl.ssl.to(self.device)
        graph_encoder_ssl.encoder.train()

        optimizer = self.optim_class(graph_encoder_ssl.encoder.parameters(), **self.optim_kwargs)

        for _ in graph_encoder_ssl.ssl.train(graph_encoder_ssl.encoder, dataloader, optimizer, epochs=self.epochs):
            pass

        return graph_encoder_ssl.encoder
