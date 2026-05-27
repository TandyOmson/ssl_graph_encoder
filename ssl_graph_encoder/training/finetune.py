""" Training for finetuning the pretrained encoder using labels
"""
import logging
from tqdm import trange

from ssl_graph_encoder.training.training_helpers import load_class, filter_class_config, EarlyStoppingCallback, ReportMetricsCallback

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

class GraphRegressionModel(nn.Module):
    """ Final model class including encoder and regression head
    """
    # may move optimizer and criterion here later as class attribute
    def __init__(self, encoder, pred_head):
        super().__init__()
        self.encoder = encoder
        self.pred_head = pred_head

    def forward(self, data):
        # extract node embeddings
        graph_emb = self.encoder(data)

        # handle DIG encoder tuple output
        if isinstance(graph_emb, tuple):
            graph_emb = graph_emb[0]

        # pass through regression head
        return self.pred_head(graph_emb)

def build_finetune_model(encoder, config):
    """ Generic factory function
        Build a fine-tuning model for regression on pretrained graph embeddings.
        Can freeze the encoder and add a flexible regression head or retrain encoder
    """

    if config["finetune"]["freeze_encoder"]:
        for p in encoder.parameters():
            p.requires_grad = False

    pred_headClass = load_class(config["prediction_head"]["class_path"])

    pred_args = filter_class_config(pred_headClass, **config["prediction_head"]["kwargs"])
    pred_head = pred_headClass(encoder.output_dim, **pred_args)

    return GraphRegressionModel(encoder, pred_head)

class FinetuneTrainer:
    """ Trainer for supervised fine-tuning encoder + prediction head model 
    """
    def __init__(self, device, config):
        self.epochs = config["finetune"]["epochs"]
        self.device = device
        # might just make this self.callback_cfg so I can have callback options in the yaml, hardcoding for now
        self.use_callbacks = config.get("tuning", {}).get("run_tuning", False)

        optim_cfg = config["finetune"]["optimizer"]
        self.optim_class  = load_class(optim_cfg["class_path"])
        self.optim_kwargs = optim_cfg.get("kwargs", {})

        crit_cfg = config["finetune"]["criterion"]
        self.crit_class = load_class(crit_cfg["class_path"])
        self.crit_kwargs = crit_cfg.get("kwargs", {})

    def fit(self, model, train_loader, val_loader):
        """ Construct optimizer and criterion from config, train model in-place
        """
        optimizer = self.optim_class(
            filter(lambda p: p.requires_grad, model.parameters()), 
            **self.optim_kwargs
        )
        criterion = self.crit_class(**self.crit_kwargs)

        # training callbacks on_epoch_end take a metrics dict and epoch number, return True to signal early stopping
        callbacks = [
            EarlyStoppingCallback(monitor="val_loss", mode="min", patience=10, min_delta=0.001, logger=log),
            ReportMetricsCallback(every_n_epochs=5, keys=["val_loss"], logger=log)
        ]   

        # separate a train() function in the model class with "yield model, metrics" if this gets to messy
        model = model.to(self.device)
        with trange(self.epochs) as t:
            for epoch in t:
                model.train()
                train_loss = 0
                stop = False
                t.set_description('Finetuning: epoch %d' % (epoch+1))
                for data in train_loader:
                    data = data.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs, data.y.float())
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                train_loss /= len(train_loader.dataset)

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for data in val_loader:
                        data = data.to(self.device)
                        pred = model(data)
                        val_loss += criterion(pred, data.y.float())
                val_loss /= len(val_loader.dataset)
                t.set_postfix(val_loss=f'{val_loss:.4f}', train_loss=f'{train_loss:.4f}')

                if self.use_callbacks:
                    for cb in callbacks:
                        # only early stopping will return True
                        if cb.on_epoch_end(epoch, {"val_loss": val_loss}):
                            stop = True
                
                if stop:
                    break

        return model, model.encoder
