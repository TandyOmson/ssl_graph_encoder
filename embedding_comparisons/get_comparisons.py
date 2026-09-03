from scipy import linalg
import scipy as sp
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import sys
from sklearn.metrics import pairwise_distances

class FrechetDistance:
    def __init__(self):
        pass

    def evaluate(self, gen_embs, ref_embs):
        gen_embs = np.asarray(gen_embs)
        ref_embs = np.asarray(ref_embs)

        gen_mu = np.mean(gen_embs, axis=0)
        ref_mu = np.mean(ref_embs, axis=0)

        gen_cov = np.cov(gen_embs.T)
        ref_cov = np.cov(ref_embs.T)

        fd = self.frechet_distance(gen_mu, gen_cov, ref_mu, ref_cov)

        return fd
    
    def frechet_distance(self, mu1, cov1, mu2, cov2, eps=1e-6):
        diff = mu1 - mu2

        covmean, _ = sp.linalg.sqrtm(cov1.dot(cov2), disp=False)
        is_real = np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3)
        if not is_real:
            offset = np.eye(cov1.shape[0]) * eps
            covmean = sp.linalg.sqrtm((cov1 + offset).dot(cov2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return float(diff.dot(diff) + np.trace(cov1) + np.trace(cov2) - 2 * tr_covmean)

def tanimoto_matrix_block(fps, block_size=512):
    """Compute Tanimoto similarity matrix in blocks with progress bar."""
    N = fps.shape[0]
    sims = np.zeros((N, N), dtype=np.float32)

    norms = fps.sum(axis=1)

    for i in tqdm(range(0, N, block_size), desc="Tanimoto blocks"):
        i_end = min(i + block_size, N)

        block = fps[i:i_end]  # (B, F)

        dot = block @ fps.T  # (B, N)

        denom = (
            norms[i:i_end, None] +
            norms[None, :] - dot
        )

        eps = 1e-8
        sims[i:i_end] = dot / (denom + eps)

    return sims

def calculate_corr(emb_sim, fp_sim):
    """
    Correlation between embedding-space similarity and fingerprint-space similarity.
    Uses progress bars for large datasets.
    """
    # --- correlation ---
    print("Computing correlation...")

    #n_pairs = emb_sim.shape[0]

    #rng = np.random.default_rng(0)
    #i = rng.integers(0, emb_sim.shape[0]), n_pairs)
    #j = rng.integers(0, emb_sim.shape[0]), n_pairs)
    
    #mask = i != j
    #i = i[mask]
    #j = j[mask]

    #emb_sim = emb_sim[i, j]
    #fp_sim = fp_sim[i, j]
    
    #corr, _ = spearmanr(
    #    emb_sim,
    #    fp_sim
    #)

    corr, _ = spearmanr(
        emb_sim[np.triu_indices_from(emb_sim, k=1)],
        fp_sim[np.triu_indices_from(fp_sim, k=1)]
    )
    return corr

def calculate_stats(gen_emb, ref_emb, ref_fps, n_top=10,
                    n_bootstrap=100, sample_size=1000):

    # ---
    # FRECHET DISTANCE BOOTSTRAP
    # ---

    fd_calc = FrechetDistance()

    gen_emb = np.asarray(gen_emb)
    ref_emb = np.asarray(ref_emb)

    rng = np.random.default_rng(0)

    # split fd
    print("split FD")
    split_fd = []

    #for i in range(n_bootstrap):
    #
    #    idx = rng.permutation(len(ref_emb))
    #    half = len(idx) // 2
    #
    #    A = ref_emb[idx[:half]]
    #    B = ref_emb[idx[half:]]
    #
    #    split_fd.append(fd_calc.evaluate(A, B))
    #    print(f"bootstrap {i+1}", end="\r")
    #    
    #split_fd = np.array(split_fd)

    # PRECOMPUTING SIMILARITIES
    
    # cosine embedding similarity
    print("Computing embedding similarity matrix...")
    norms = np.linalg.norm(ref_emb, axis=1, keepdims=True)
    embs_norm = ref_emb / (norms + 1e-8)
    emb_sim = embs_norm @ embs_norm.T
    print("cosine dim", emb_sim.shape)

    # pairwise Euclidean distances
    dists = pairwise_distances(
        ref_emb,
        metric="euclidean"
    )
    # unique distances only
    #dists = dists[np.triu_indices_from(dists, k=1)]
    print("euclid dim", dists.shape)
    #print(dists)

    # ecfp4 tanimoto similarity
    print("Computing ecfp4 tanimoto similarity matrix...")
    fp_sim = tanimoto_matrix_block(ref_fps, block_size=512)
    fp_sim = fp_sim / np.linalg.norm(fp_sim)
    print("tanimoto dim", fp_sim.shape)

    # ---
    # COMPARING EMBEDDINGS DISTRIBUTIONS
    # ---

    # Average euclidian distance in local neighbourhood in embedded space per sample
    print("local embedding distances")
    local_emb_dist = []
    for n in range(sample_size):
        i = rng.integers(len(ref_emb))
        emb_idx = np.argpartition(
            dists[i],
            -(n_top + 1)
        )[-(n_top + 1):]

        emb_idx = emb_idx[emb_idx != i]
        local_emb_dist.append(
            dists[i, emb_idx].mean()
        )
        print(f"sample {n+1}", end="\r")

    local_emb_dist = np.asarray(local_emb_dist)

    # Average cosine similarity in local neighbourhood in embedded space per sample
    print("local embedding cosine similarity")
    local_emb_cosine_sim = []
    for n in range(sample_size):
        i = rng.integers(len(ref_emb))
        emb_idx = np.argpartition(
            emb_sim[i],
            -(n_top + 1)
        )[-(n_top + 1):]
        emb_idx = emb_idx[emb_idx != i]
        local_emb_cosine_sim.append(
            emb_sim[i, emb_idx].mean()
        )
        print(f"sample {n+1}", end="\r")

    local_emb_cosine_sim = np.asarray(local_emb_cosine_sim)
    
    # PCA eigenvalues
    # covariance matrix
    cov = np.cov(ref_emb, rowvar=False)

    # eigenvalues (sorted descending)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]

    # participation ratio (effective dimensionality)
    pr = (eigvals.sum() ** 2) / np.sum(eigvals ** 2)

    # fraction of variance explained
    explained = eigvals / eigvals.sum()

    # cumulative variance
    cum_explained = np.cumsum(explained)

    # dimensions required for 90% variance
    n90 = np.searchsorted(cum_explained, 0.90) + 1

    # ---
    # COMPARING WITH ECFP4
    # ---

    # Distribution of average ecfp4 tanimoto similarities for top n neighours in embedded space per sample 
    tanimoto_av = []
    embedding_tanimoto = []

    for n in range(sample_size):
    
        i = rng.integers(len(ref_emb))
    
        emb_idx = np.argpartition(
            emb_sim[i],
            -(n_top + 1)
        )[-(n_top + 1):]
    
        emb_idx = emb_idx[emb_idx != i]
    
        embedding_tanimoto.append(
            fp_sim[i, emb_idx].mean()
        )
        print(f"sample {n+1}", end="\r")

    tanimoto_av = np.asarray(embedding_tanimoto)
    
    # Distribution of overlaps of top-n neighbours in ecfp4 tanimoto and embedded space
    print("neighbour overlap")
    neighbour_overlap = []

    for n in range(sample_size):

        i = rng.integers(len(ref_emb))

        emb_idx = np.argpartition(
            emb_sim[i], -(n_top + 1)
        )[-(n_top + 1):]

        fp_idx = np.argpartition(
            fp_sim[i], -(n_top + 1)
        )[-(n_top + 1):]

        emb_set = set(emb_idx) - {i}
        fp_set = set(fp_idx) - {i}

        neighbour_overlap.append(
            len(emb_set & fp_set) / n_top
        )
        print(f"sample {n+1}", end="\r")

    neighbour_overlap = np.asarray(neighbour_overlap)

    # Spearman correlation between pairwise distances in embedding space and ecfp4 tanimoto space
    corr = calculate_corr(emb_sim, fp_sim)

    return {
        #"split_fd": split_fd,
        "local_emb_dist": local_emb_dist,
        "local_emb_cosine_sim": local_emb_cosine_sim,
        "emb_dist": dists[np.triu_indices_from(dists, k=1)][:1000000],
        "emb_cosine": emb_sim[np.triu_indices_from(emb_sim, k=1)][:1000000],
        "tanimoto_av": tanimoto_av,
        "participation_ratio": pr,
        "dimensions_90_variance": n90,
        "eigenvalues": eigvals,
        "total_variance": eigvals.sum(),
        "neighbour_overlap": neighbour_overlap,
        "corr": corr,
    }

gen_npzfile = sys.argv[1]
ref_npzfile = sys.argv[2]

gen_emb_dict = np.load(gen_npzfile)
ref_emb_dict = np.load(ref_npzfile)

all_stats = {}
for encoder in gen_emb_dict.keys():
    print(encoder)
    gen_emb = gen_emb_dict[encoder]
    ref_emb = ref_emb_dict[encoder]

    stats = calculate_stats(gen_emb, ref_emb, ref_emb_dict["benchmark_ecfp4"], n_top=250, n_bootstrap=5000, sample_size=8000)
    all_stats[encoder] = stats
    
# flatten into savez-friendly format
save_dict = {}
for model_name, d in all_stats.items():
    for k, v in d.items():
        save_dict[f"{model_name}_{k}"] = v

np.savez("stats.npz", **save_dict)
