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
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--embs", type=list, required=True, help="other .np embedding files")

    args = parser.parse_args()

    smis = [i.strip() for i in open(args.smis, "r").readlines()]
    other_embs = [np.load(i) for i in args.embs]

    for i in other_embs:
        assert other_embs.shape[0] == len(smis)

    # get sample if size is specified
    if args.sample_size is not None:
        sample_idxs = np.random.randint(0, len(smis))
        smis_sample = [smis[i] for i in sample_idxs]
        other_embs_samples = []
        for i in other_embs:
            other_embs_samples.append(normalise(i[sample_idxs]))

    metrics = {}
    for f, i in zip(args.embs.split(".")[-2] , other_embs_samples):
        metrics[f] = unsupervised_embedding_eval(i)

    pprint(metrics)
