import argparse
from pprint import pprint
import numpy as np
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

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

    return {"lr_rmse": lr_rmse, 
            "lr_r2": lr_r2, 
            "knn_rmse": knn_rmse,
            "knn_r2": knn_r2, 
            "spearman_local": spearman_corr,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smis", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--embs", type=str, nargs="+", required=True, help="other .np embedding files")

    args = parser.parse_args()

    smis = np.array([i.strip() for i in open(args.smis, "r").readlines()])
    labels = np.array([float(i.strip()) for i in open(args.labels, "r").readlines()])
    other_embs = [np.load(i) for i in args.embs]

    # ignore any samples with nan in any of the embeddings/fingerprints
    mask = np.isfinite(labels)
    for d in other_embs:
        mask &= np.isfinite(d)[:,0]
    other_embs = [d[mask,:] for d in other_embs]
    labels = labels[mask]
    smis = smis[mask]

    for i in other_embs:
        assert len(labels) == len(smis)
        assert i.shape[0] == len(smis)

    print(f"metrics on {len(smis)} molecules")
    metrics = {}
    for f, i in zip(args.embs , other_embs):
        f = f.split("/")[-1].split(".")[-2]
        metrics[f] = supervised_embedding_eval(i, labels)

    pprint(metrics)
