import numpy as np
from scipy.spatial.distance import pdist
import argparse
from pprint import pprint

def normalise(x):
    return (x-np.min(x)) / (np.max(x) - np.min(x))

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

    return {"mean_dist": mean_distance,
            "median_distance": median_distance,
            "dist_variation": distance_variation
            }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smis", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--embs", type=str, nargs="+", required=True, help="other .np embedding files")

    args = parser.parse_args()

    smis = np.array([i.strip() for i in open(args.smis, "r").readlines()])
    other_embs = [np.load(i) for i in args.embs]

    # ignore any samples with nan in any of the embeddings/fingerprints
    mask = np.isfinite(other_embs[0])[:,0]
    for d in other_embs[1:]:
        mask &= np.isfinite(d)[:,0]
    other_embs = [d[mask,:] for d in other_embs]
    smis = smis[mask]
    
    for i in other_embs:
        assert i.shape[0] == len(smis)

    if args.sample_size > len(smis):
        args.sample_size = len(smis)
    # get sample if size is specified
    if args.sample_size is not None:
        sample_idxs = [np.random.randint(0, len(smis)) for i in range(args.sample_size)]
        smis_sample = [smis[i] for i in sample_idxs]
        other_embs_samples = []
        for i in other_embs:
            other_embs_samples.append(normalise(i[sample_idxs]))

    print(f"metrics on {len(smis)} molecules")
    metrics = {}
    for f, i in zip(args.embs, other_embs_samples):
        f = f.split("/")[-1].split(".")[-2]
        metrics[f] = unsupervised_embedding_eval(i)

    pprint(metrics)
