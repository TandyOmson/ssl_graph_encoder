""" Training for finetuning the pretrained encoder using labels
"""
import logging
import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

log = logging.getLogger(__name__)

import torch
import torch.nn as nn

def load_class(class_path):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def create_finetune_model(encoder, config):
    """ Generic factory function
        Build a fine-tuning model for regression on pretrained graph embeddings.
        Can freeze the encoder and add a flexible regression head or retrain encoder
    """

    if config["finetune"]["freeze_encoder"]:
        for p in encoder.parameters():
            p.requires_grad = False

    reg_headClass = load_class(config["regression_head"]["class_path"])

    reg_head = reg_headClass(config["encoder"]["embed_dim"], **config["regression_head"])

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

    return GraphRegressionModel(encoder, reg_head)

def finetune(model, train_loader, val_loader, config):
    """ Fine-tune the model on the given dataset.
    """
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=float(config["finetune"]["lr"])
    )
    criterion = nn.MSELoss()

    for epoch in range(config["finetune"]["epochs"]):
        model.train()
        train_loss = 0
        for data in train_loader:
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