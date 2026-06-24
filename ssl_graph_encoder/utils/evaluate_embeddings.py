""" Functions for evaluating embeddings
"""
import torch

from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
import numpy as np


def _get_module_device(module):
    first_param = next(module.parameters(), None)
    return first_param.device if first_param is not None else torch.device("cpu")


def _batch_to_device(batch, device):
    return batch.to(device) if hasattr(batch, "to") else batch

def encoder_embeddings_out(trained_encoder, dataloader, sample_size=np.inf):
    """ Extract embeddings from the trained encoder, save them as npy 
    """
    trained_encoder.eval()
    device = _get_module_device(trained_encoder)
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            if len(embeddings) < sample_size:
                batch = _batch_to_device(batch, device)
                batch_embeddings = trained_encoder(batch)
                embeddings.append(batch_embeddings)
                if batch.y is not None:
                    labels.append(batch.y)
                else:
                    labels.append(torch.zeros(batch_embeddings.shape[0]))  # dummy labels if not available
            else:
                break
    
    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)
    embeddings_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()

    return embeddings_np, labels_np

def compute_embedding_splits(encoder, loaders):
    train_loader, test_loader, val_loader = loaders

    train_embed, train_labels = encoder_embeddings_out(encoder, train_loader, sample_size=4000)
    test_embed, test_labels = encoder_embeddings_out(encoder, test_loader, sample_size=500)
    val_embed, val_labels = encoder_embeddings_out(encoder, val_loader, sample_size=500)

    embeddings = np.concatenate([train_embed, test_embed, val_embed], axis=0)
    labels = np.concatenate([train_labels, test_labels, val_labels], axis=0)

    split_idxs = [
        np.arange(0, train_embed.shape[0]),
        np.arange(train_embed.shape[0], train_embed.shape[0] + test_embed.shape[0]),
        np.arange(train_embed.shape[0] + test_embed.shape[0], embeddings.shape[0]),
    ]

    return embeddings, labels, split_idxs, test_embed

def evaluate_full_model(model, val_loader):
    """ Evaluate RMSE and R2 of the full model on the test set
    """
    model.eval()
    device = _get_module_device(model)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in val_loader:
            data = _batch_to_device(data, device)
            preds = model(data)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

    all_labels = np.concatenate(all_labels).flatten()
    all_preds = np.concatenate(all_preds).flatten()

    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    r2 = r2_score(all_labels, all_preds)
    return rmse, r2

def evaluate_full_model_classification(model, val_loader):
    """ Evaluate accuaracy and F1 of the full model on the test set
    """
    model.eval()
    device = _get_module_device(model)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in val_loader:
            data = _batch_to_device(data, device)
            logits = model(data)

            # binary 
            if logits.ndim == 1 or logits.shape[-1] == 1:
                # convert logits to predicted class labels (the logits loss function applies a sigmoid in class classification)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()

            # multiclass
            else:
                # multiclass case is using CrossEntropyLoss which applies softmax
                preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

    all_labels = np.concatenate(all_labels).flatten()
    all_preds = np.concatenate(all_preds).flatten()

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return acc, f1