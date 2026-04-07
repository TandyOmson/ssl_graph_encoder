""" Training for finetuning the pretrained encoder using labels
"""
import logging
import importlib

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

def load_class(class_path):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

class GraphRegressionModel(nn.Module):
    """ Final model class including encoder and regression head
    """
    # may move optimizer and criterion here later as class attribute
    def __init__(self, encoder, reg_head):
        super().__init__()
        self.encoder = encoder
        self.reg_head = reg_head

    def forward(self, data):
        # extract node embeddings
        graph_emb = self.encoder(data)

        # handle DIG encoder tuple output
        if isinstance(graph_emb, tuple):
            graph_emb = graph_emb[0]

        # pass through regression head
        return self.reg_head(graph_emb)

def build_finetune_model(encoder, config):
    """ Generic factory function
        Build a fine-tuning model for regression on pretrained graph embeddings.
        Can freeze the encoder and add a flexible regression head or retrain encoder
    """

    if config["finetune"]["freeze_encoder"]:
        for p in encoder.parameters():
            p.requires_grad = False

    reg_headClass = load_class(config["regression_head"]["class_path"])

    reg_head = reg_headClass(config["encoder"]["embed_dim"], **config["regression_head"])

    return GraphRegressionModel(encoder, reg_head)

class FinetuneTrainer:
    """ Trainer for supervised fine-tuning encoder + regression head model 
    """
    def __init__(self, device, config):
        self.epochs = config["finetune"]["epochs"]
        self.device = device

        optim_cfg = config["finetune"]["optimizer"]
        self.optim_class  = load_class(optim_cfg["class_path"])
        self.optim_kwargs = optim_cfg.get("kwargs", {})

        crit_cfg = config["finetune"]["criterion"]
        self.crit_class = load_class(crit_cfg["class_path"])
        self.crit_kwargs = crit_cfg.get("kwargs", {})

    def fit(self, model, train_loader, val_loader):
        """ Train model in-place
        """
        optimizer = self.optim_class(
            filter(lambda p: p.requires_grad, model.parameters()), 
            **self.optim_kwargs
        )
        criterion = self.crit_class(**self.crit_kwargs)

        model = model.to(self.device)
        for epoch in range(self.epochs):
            model.train()
            train_loss = 0
            for data in train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, data.y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data in val_loader:
                    pred = model(data)
                    val_loss += criterion(pred, data.y)
            val_loss /= len(val_loader.dataset)
            print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Train Loss: {train_loss:.4f}")

        return model, model.encoder
