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

def local_spearman(embeddings, labels, k=100):
    """
    Compute Spearman correlation between embedding distances and activity differences
    restricted to each point's k nearest neighbors.
    """
    N = embeddings.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # Remove self-distance (first column)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    emb_dist_list = []
    label_diff_list = []

    for i in range(N):
        emb_dist_list.extend(distances[i])
        label_diff_list.extend(np.abs(labels[i] - labels[indices[i]]))

    spearman_corr, _ = spearmanr(emb_dist_list, label_diff_list)
    return spearman_corr

def supervised_embedding_eval(embeddings, labels):
    """Evaluate embeddings against labels:
    Ridge regression RMSE/R2
    kNN regression RMSE/R2
    Spearman correlation between embedding distances and label differences
    """

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )

    # Standardize embeddings
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Ridge Regression (global)
    lr = Ridge(alpha=1.0)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)

    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    lr_r2 = r2_score(y_test, lr_preds)

    # kNN Regression (local)
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_preds = knn.predict(X_test)

    knn_rmse = np.sqrt(mean_squared_error(y_test, knn_preds))
    knn_r2 = r2_score(y_test, knn_preds)

    # Spearman correlation (embedding distances vs activity differences)
    spearman_corr = local_spearman(embeddings, labels, k=10)

    return lr_rmse, lr_r2, knn_rmse, knn_r2, spearman_corr


def unsupervised_embedding_eval(embeddings):
    """Evaluate embeddings without labels:
    Spread (mean and median pairwise distances)
    Distance distribution (coefficient of variation)
    """

    pairwise_distances = pdist(embeddings, metric='euclidean')
    distances_flat = pairwise_distances.flatten()

    mean_distance = np.mean(distances_flat)
    median_distance = np.median(distances_flat)
    distance_variation = np.std(distances_flat) / mean_distance if mean_distance > 0 else 0

    return mean_distance, median_distance, distance_variation

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

def encoder_embeddings_out(trained_encoder, dataloader, sample_size=np.inf):
    """ Extract embeddings from the trained encoder, save them as npy 
    """
    trained_encoder.eval()
    device = _get_module_device(trained_encoder)
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            while len(embeddings) < sample_size:
                batch = _batch_to_device(batch, device)
                batch_embeddings = trained_encoder(batch)
                embeddings.append(batch_embeddings)
                if batch.y is not None:
                    labels.append(batch.y)
                else:
                    labels.append(torch.zeros(batch_embeddings.shape[0]))  # dummy labels if not available
    
    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)
    embeddings_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()

    return embeddings_np, labels_np

#
# Evaluation methods for classification tasks
#
def supervised_embedding_eval_classification(embeddings, labels):
    """ Evaluate embeddings against discrete labels for classification
        Rigde classification
        KNN classification
    """

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )

    # Standardize embeddings
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Ridge Classification (global)
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(X_train, y_train)
    clf_preds = clf.predict(X_test)

    lr_acc = accuracy_score(y_test, clf_preds)
    lr_f1 = f1_score(y_test, clf_preds, average='weighted') # average only applies if multi-class

    # kNN Classification (local)
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(X_train, y_train)
    knn_preds = knn_clf.predict(X_test)

    knn_acc = accuracy_score(y_test, knn_preds)
    knn_f1 = f1_score(y_test, knn_preds, average='weighted')

    return lr_acc, lr_f1, knn_acc, knn_f1

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