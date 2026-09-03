from scipy import linalg
import scipy as sp
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import sys

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

def rbf_mmd(X, Y, sigma=None):
    """
    Unbiased RBF-kernel MMD^2.

    Parameters
    ----------
    X : (n,d) array
    Y : (m,d) array
    sigma : float or None
        RBF bandwidth. If None, use median heuristic.

    Returns
    -------
    float
        MMD^2
    """

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    n = len(X)
    m = len(Y)

    # --------------------------------------------------
    # Median heuristic
    # --------------------------------------------------
    if sigma is None:
        Z = np.vstack([X, Y])

        # pairwise squared distances
        D = cdist(Z, Z, metric="sqeuclidean")

        # exclude diagonal
        D = D[np.triu_indices_from(D, k=1)]

        sigma = np.sqrt(0.5 * np.median(D))

        # numerical safeguard
        sigma = max(sigma, 1e-8)

    gamma = 1.0 / (2.0 * sigma**2)

    # --------------------------------------------------
    # Kernel matrices
    # --------------------------------------------------
    Kxx = np.exp(-gamma * cdist(X, X, metric="sqeuclidean"))
    Kyy = np.exp(-gamma * cdist(Y, Y, metric="sqeuclidean"))
    Kxy = np.exp(-gamma * cdist(X, Y, metric="sqeuclidean"))

    # remove diagonals for unbiased estimate
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    mmd2 = (
        Kxx.sum() / (n * (n - 1))
        + Kyy.sum() / (m * (m - 1))
        - 2.0 * Kxy.mean()
    )

    return float(mmd2)

def query_to_set(query_emb, set_embs, n_top, dtype="real"):
    if dtype == "real":
        dot = set_embs @ query_emb
        denom = np.sum(set_embs**2, axis=1) + np.dot(query_emb, query_emb) - dot
        eps = 1e-8
        sims = dot / (denom + eps)
    elif dtype == "bit":
        inter = np.logical_and(set_embs, query_emb).sum(axis=1)
        union = np.logical_or(set_embs, query_emb).sum(axis=1)
        sims = inter / (union + 1e-8)
    else:
        raise ValueError

    n_top = min(n_top, len(sims))
    idx = np.argpartition(sims, -n_top)[-n_top:]
    return sims[idx].mean()

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

def calculate_corr(embs, fps, block_size=512):
    """
    Correlation between embedding-space similarity and fingerprint-space similarity.
    Uses progress bars for large datasets.
    """
    # --- embedding similarity ---
    print("Computing embedding similarity matrix...")
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_norm = embs / (norms + 1e-8)
    emb_sim = embs_norm @ embs_norm.T
    
    # --- fingerprint similarity (Tanimoto) ---
    print("Computing fingerprint similarity matrix...")
    fp_sim = tanimoto_matrix_block(fps, block_size=block_size)
    #fp_sim = fp_sim / np.linalg.norm(fp_sim)

    # --- correlation ---
    print("Computing correlation...")

    n_pairs = 200000

    rng = np.random.default_rng(0)
    i = rng.integers(0, len(embs), n_pairs)
    j = rng.integers(0, len(embs), n_pairs)
    
    mask = i != j
    i = i[mask]
    j = j[mask]
    
    corr, _ = spearmanr(
        emb_sim[i, j],
        fp_sim[i, j]
    )
    return corr

def calculate_stats(gen_emb, ref_emb, ref_fps, n_top=10,
                    n_bootstrap=100, sample_size=1000):

    fd_calc = FrechetDistance()

    gen_emb = np.asarray(gen_emb)
    ref_emb = np.asarray(ref_emb)

    rng = np.random.default_rng(0)

    # --------------------------------------------------------
    # 1. Distribution of FD and MMD for random split of ref+gen
    # --------------------------------------------------------

    print("split FD")
    split_fd = []
    #split_mmd = []

    for i in range(n_bootstrap):

        idx = rng.permutation(len(ref_emb))
        half = len(idx) // 2

        A = ref_emb[idx[:half]]
        B = ref_emb[idx[half:]]

        split_fd.append(fd_calc.evaluate(A, B))
        print(f"bootstrap {i+1}", end="\r")

        # simple linear-kernel MMD
        #split_mmd.append(rbf_mmd(A, B))
        
    split_fd = np.array(split_fd)
    #split_mmd = np.array(split_mmd)

    # --------------------------------------------------------
    # 2. FD and MMD of ref_emb and gen_emb
    # --------------------------------------------------------

    print("FD")
    fd = fd_calc.evaluate(gen_emb, ref_emb)

    #mmd = rbf_mmd(gen_emb, ref_emb)

    # --------------------------------------------------------
    # 3. Distribution of average top n_top tanimoto similarities
    # --------------------------------------------------------

    print("calculating tanimoto blocks")
    emb_norm = ref_emb / (
        np.linalg.norm(ref_emb, axis=1, keepdims=True) + 1e-8
    )
    emb_sim = emb_norm @ emb_norm.T

    fp_sim = tanimoto_matrix_block(ref_fps)

    
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
        print(f"neighbour {n+1}", end="\r")

    tanimoto_av = np.asarray(embedding_tanimoto)
    
    # --------------------------------------------------------
    # 4. Overlap of top-n neighbours
    # --------------------------------------------------------

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
        print(f"neighbour {n+1}", end="\r")

    neighbour_overlap = np.asarray(neighbour_overlap)

    # --------------------------------------------------------
    # 5. Continuity proxy
    # --------------------------------------------------------

    corr = calculate_corr(ref_emb, ref_fps)

    return {
        "split_fd": split_fd,
        #"split_mmd": split_mmd,
        "fd": fd,
        #"mmd": mmd,
        "tanimoto_av": tanimoto_av,
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

    stats = calculate_stats(gen_emb, ref_emb, ref_emb_dict["benchmark_ecfp4"], n_top=50, n_bootstrap=1000, sample_size=1000)
    all_stats[encoder] = stats
    
# flatten into savez-friendly format
save_dict = {}
for model_name, d in all_stats.items():
    for k, v in d.items():
        save_dict[f"{model_name}_{k}"] = v

np.savez("stats.npz", **save_dict)
