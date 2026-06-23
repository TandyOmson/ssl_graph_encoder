from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
import numpy as np

## Unsupervised metrics
##  Each method has embeddings as a single positional arg and kwargs 
def pairwise_distances(embeddings, metric="euclidean", max_samples=500):
    # max samples is included because it is especially expensive
    embeddings = embeddings[np.random.permutation(embeddings.shape[0])[:max_samples]]
    pairwise_distances = pdist(embeddings, metric=metric).flatten()

    mean_distance = np.mean(pairwise_distances)
    median_distance = np.median(pairwise_distances)
    distance_variation = np.std(pairwise_distances) / mean_distance if mean_distance > 0 else 0

    return {"mean_pdist" : mean_distance, "median_pdist" : median_distance, "pdist_var" : distance_variation}

## Supervised metrics
## Each method has embeddings, labels and split_idxs as positional args and kwargs 
def local_spearman(embeddings, labels, split_idxs, k=100):
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

def ridge_reg(embeddings, labels, split_idxs):
    X_train, X_test, X_val = embeddings[split_idxs[0],:], embeddings[split_idxs[1],:], embeddings[split_idxs[2],:]
    y_train, y_test, y_val = labels[split_idxs[0]], labels[split_idxs[1]], labels[split_idxs[2]]

    # optimize alpha using val set
    best_res = np.inf
    best_alpha = 0
    for alpha in np.linspace(0.1, 10, 50):
        lr = Ridge(alpha=alpha)
        lr.fit(X_train, y_train)
        res = mean_squared_error(y_val, lr.predict(X_val))
        if res < best_res:
            best_res = res
            best_alpha = alpha

    lr = Ridge(alpha=best_alpha)
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)

    lr_rmse = np.sqrt(mean_squared_error(y_test, preds))
    lr_r2 = r2_score(y_test, preds)
    return {"lr_rmse": lr_rmse, "lr_r2": lr_r2}

def knn_reg(embeddings, labels, split_idxs):
    X_train, X_test, X_val = embeddings[split_idxs[0],:], embeddings[split_idxs[1],:], embeddings[split_idxs[2],:]
    y_train, y_test, y_val = labels[split_idxs[0]], labels[split_idxs[1]], labels[split_idxs[2]]

    # optimize using val set
    best_res = np.inf
    best_k = 0
    for k in np.arange(5, 25, 5):
        knn = KNeighborsRegressor(n_neighbors=int(k))
        knn.fit(X_train, y_train)
        res = mean_squared_error(y_val, knn.predict(X_val))
        if res < best_res:
            best_res = res
            best_k = int(k)

    knn = KNeighborsRegressor(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)

    knn_rmse = np.sqrt(mean_squared_error(y_test, preds))
    knn_r2 = r2_score(y_test, preds)
    return {"knn_rmse": knn_rmse, "knn_r2": knn_r2}
