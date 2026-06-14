""" Functions used for calculating pairwise distances and comparing with a reference
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist
import argparse

from pprint import pprint

def normalise(x):
    return (x-np.min(x)) / (np.max(x) - np.min(x))

def compute_pairwise_comparison_metrics(d_pred, d_ref):
    d_pred = np.asarray(d_pred)
    d_ref = np.asarray(d_ref)

    assert d_pred.shape == d_ref.shape

    mse = np.mean((d_pred - d_ref) ** 2)
    mae = np.mean(np.abs(d_pred - d_ref))

    pearson = pearsonr(d_pred, d_ref)[0]
    spearman = spearmanr(d_pred, d_ref)[0]

    return {
        "MSE": mse,
        "MAE": mae,
        "Pearson": pearson,
        "Spearman" : spearman
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smis", type=str, required=True)
    parser.add_argument("--ref", type=str, required=True, help=".np file with embeddings of reference")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--embs", type=list, required=True, help="other .np embedding files")

    args = parser.parse_args()

    smis = [i.strip() for i in open(args.smis, "r").readlines()]
    ref_embs = np.load(args.ref)
    other_embs = [np.load(i) for i in args.embs]

    assert int(ref_embs.shape[0]) == len(smis) 
    for i in other_embs:
        assert other_embs.shape[0] == ref_embs.shape[0]
    
    # get sample if size is specified
    if args.sample_size is not None:
        sample_idxs = np.random.randint(0, len(smis))
        smis_sample = [smis[i] for i in sample_idxs]
        ref_embs_sample = normalise(ref_embs[sample_idxs])
        other_embs_samples = []
        for i in other_embs:
            other_embs_samples.append(normalise(i[sample_idxs]))
    
    d_ref = pdist(ref_embs_sample, metric="euclidian")
    other_ds = []
    for i in other_embs_samples:
        other_ds.append(pdist(i, metric="euclidean"))

    metrics = {}
    for f, i in zip(args.embs.split(".")[-2] , other_ds):
        metrics[f] = compute_pairwise_comparison_metrics(d_ref, i)

    pprint(metrics)