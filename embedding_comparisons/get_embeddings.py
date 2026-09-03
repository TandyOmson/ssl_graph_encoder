""" Comparing encoder embeddings with each other and with ECFP4
"""
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from scipy.linalg import sqrtm
from joblib import Parallel, delayed

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray

from ssl_graph_encoder.utils.get_ssl_embeddings_no_global import sslEmbeddings, sslEmbeddingsLM
from ssl_graph_encoder.utils.frechet_distance import FrechetDistance
from ssl_graph_encoder.utils.model_io import load_pretrained_encoder
from ssl_graph_encoder.utils.smiles_to_graph import SmilesToGraph

def canonicalise(smis):
    out = []
    for s in smis:
        try:
            out.append(Chem.CanonSmiles(s))
        except:
            continue
    return out

def get_embeddings(smis, encoder):
    failure_idxs = []
    
    embs = []
    emb_dim = 256
    for count, smi in enumerate(smis):
        print(f"gen mols {count} of {len(smis)}", end="\r")
        try:
            emb = encoder.get_embeddings(smi)
            if np.isnan(emb[0]).any():
                raise Exception
            embs.append(emb[0])
            emb_dim = len(emb[0])
        except:
            print("embedding failed for", count, smi)
            embs.append(np.array([np.nan]*emb_dim))
            failure_idxs.append(count)
            continue
    return np.array(embs), failure_idxs

def get_embeddings_lm(data, encoder):
    embs = []
    for i, d in enumerate(data):
        print(f"{i}/{len(data)}", end="\r")
        with torch.no_grad():
            emb = encoder(d)
        embs.append(emb[0].detach().cpu().numpy())

    return np.array(embs)

# ECFP4
fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def smiles_to_ecfp4(smiles_list):
    fps = np.zeros((len(smiles_list), 2048), dtype=np.uint8)
    fail_idxs = []
    
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fail_idxs.append(i)
            continue
        fp = fpgen.GetFingerprint(mol)
        ConvertToNumpyArray(fp, fps[i])

    return fps, fail_idxs

########################
### Comparison Tasks ###
########################
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

def make_fast_delta_fd(gen_emb, ref_emb):
    """
    Fast Frechet delta without sqrtm.
    Uses trace-based approximation:
        ||μ1-μ2||^2 + Tr(Σ1) + Tr(Σ2)
    """

    # --- precompute gen stats ---
    n = len(gen_emb)
    mu_g = gen_emb.mean(axis=0)
    Xc = gen_emb - mu_g
    cov_g = (Xc.T @ Xc) / (n - 1)
    tr_cov_g = np.trace(cov_g)

    # --- precompute ref stats ---
    m = len(ref_emb)
    mu_r = ref_emb.mean(axis=0)
    Yc = ref_emb - mu_r
    cov_r = (Yc.T @ Yc) / (m - 1)
    tr_cov_r = np.trace(cov_r)

    # frechet distance of full gen set and ref set
    diff = mu_g - mu_r
    base_fd = diff @ diff + tr_cov_g + tr_cov_r

    # frechet distance of gen set (without sample x_i) and ref set
    def delta_fd(x_i):
        # leave-one-out mean
        mu_i = (n * mu_g - x_i) / (n - 1)

        # leave-one-out covariance trace (cheap!)
        delta = x_i - mu_g
        tr_cov_i = (n / (n - 1)) * tr_cov_g - (delta @ delta) / (n - 1)

        # FD approx
        diff_i = mu_i - mu_r
        fd_i = diff_i @ diff_i + tr_cov_i + tr_cov_r

        return fd_i - base_fd

    return delta_fd

def calculate_stats(gen_emb, ref_emb, n_top=1):
    N = len(gen_emb)
    eps = 1e-8

    assert np.isfinite(gen_emb).all()
    assert np.isfinite(ref_emb).all()

    # precompute all similarities
    print("computing sims")
    dot_gen = gen_emb @ gen_emb.T
    norm_gen = np.sum(gen_emb**2, axis=1)
    denom_gen = norm_gen[:,None] + norm_gen[None, :] - dot_gen
    sim_gen = dot_gen / (denom_gen + eps)

    print("computing ref sims")
    dot_ref = gen_emb @ ref_emb.T
    norm_ref = np.sum(ref_emb**2, axis=1)
    denom_ref = norm_gen[:, None] + norm_ref[None, :] - dot_ref
    sim_ref = dot_ref / (denom_ref + eps)

    print("computing cov and mu")
    av_ref = np.empty(N)
    av_gen = np.empty(N)
    delta_fd = np.empty(N)
    delta_fd_fn = make_fast_delta_fd(gen_emb, ref_emb)
    
    def process_i(i):
        # ref
        row = sim_ref[i]
        idx = np.argpartition(row, -n_top)[-n_top:]
        av_r = row[idx].mean()
    
        # gen excl self
        rowg = sim_gen[i].copy()
        old = rowg[i]
        rowg[i] = -np.inf
        idx = np.argpartition(rowg, -n_top)[-n_top:]
        av_g = rowg[idx].mean()
        rowg[i] = old
    
        # delta FD
        d = delta_fd_fn(gen_emb[i])
    
        return av_r, av_g, d
    
    results = Parallel(n_jobs=44, prefer="processes")(
        delayed(process_i)(i) for i in range(N)
    )
    
    av_ref, av_gen, delta_fd = map(np.array, zip(*results))
    
    return av_ref, av_gen, delta_fd

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
    corr = np.corrcoef(
        emb_sim.ravel(),
        fp_sim.ravel()
    )[0, 1]

    return corr

# load SMILES
#gen_smis = [i.strip() for i in open("random_transformer_gen.smi")]
#ref_smis = [i.strip() for i in open("training_set.smi")]
gen_smis = [i.strip() for i in open("gen_test.smi")]
ref_smis = [i.strip() for i in open("ref_test.smi")]

gen_smis = canonicalise(gen_smis)
ref_smis = canonicalise(ref_smis)

gen_mask = np.ones(len(gen_smis), dtype=bool)
ref_mask = np.ones(len(ref_smis), dtype=bool)
print("SMILES loaded")
#########################
### Generate encoding ###
#########################
names = ["benchmark_ecfp4", "language_model", "schnet"]
lm = [True, True, False]
stg_file = "/home/tcl25/Dself_sup_graph_learning/ssl_graph_encoder/data/datasets/hydros_minimal_split/hydros_minimal_stg_config.json"
logs_dir="/home/tcl25/Dself_sup_graph_learning/ssl_graph_encoder/logs/hydros_final"
encoder_files = [f"{logs_dir}/{i}/finetuned_encoder.pt" for i in names]

stg = SmilesToGraph.from_config(Path(stg_file))

# compute embeddings
gen_embeddings = []
ref_embeddings = []
print("generating embeddings")
for enc_file, is_lm in zip(encoder_files, lm):
    print(enc_file)
    if is_lm:
        print("language_model")
        encoder = sslEmbeddingsLM(enc_file, stg_file)
    else:
        encoder = sslEmbeddings(enc_file, stg_file)
    #for name, p in encoder.encoder.named_parameters():
    #    print(name, p.flatten()[:5])
    #    break
    gen_embs, gen_fails = get_embeddings(gen_smis, encoder)
    ref_embs, ref_fails = get_embeddings(ref_smis, encoder)
    
    gen_embeddings.append(gen_embs)
    ref_embeddings.append(ref_embs)

    gen_mask[gen_fails] = False
    ref_mask[ref_fails] = False
    
# remove failures in any from from embeddings
new_ref = []
new_gen = []
for count, (ref_arr, gen_arr) in enumerate(zip(ref_embeddings, gen_embeddings)):
    new_gen.append(gen_arr[gen_mask,:])
    new_ref.append(ref_arr[ref_mask,:])

    print(ref_arr.shape)
    print(gen_arr.shape)
    
ref_embeddings = new_ref
gen_embeddings = new_gen

gen_dict = {}
for model_name, d in zip(names, gen_embeddings):
    gen_dict[f"{model_name}"] = d

ref_dict = {}
for model_name, d in zip(names, ref_embeddings):
    ref_dict[f"{model_name}"] = d

np.savez("test_gen_embeddings.npz", **gen_dict)
np.savez("test_ref_embeddings.npz", **ref_dict)

#all_stats = {}
#for gen, ref, name in zip(gen_embeddings, ref_embeddings, names):    
#    print("calculating stats for:", name)
#
#    stats = calculate_stats(gen, ref)
#    print("stats done")
#    gen_corr = calculate_corr(gen, gen_embeddings[-1])
#    ref_corr = calculate_corr(ref, ref_embeddings[-1])
#    print("corr done")
#
#    # unpack stats for clarity
#    av_ref, av_gen, delta_fd = stats
#
#    all_stats[name] = {
#        "av_ref": np.array(av_ref),
#        "av_gen": np.array(av_gen),
#        "delta_fd": np.array(delta_fd),
#        "gen_corr": gen_corr,
#        "ref_corr": ref_corr,
#    }
#
## flatten into savez-friendly format
#save_dict = {}
#for model_name, d in all_stats.items():
#    for k, v in d.items():
#        save_dict[f"{model_name}_{k}"] = v
#
#np.savez("stats.npz", **save_dict)

# plot distributions of av_ref, av_gen and delta_fd across the generated set
#import matplotlib.pyplot as plt
#
#av_ref, av_gen, delta_fd = all_stats["schnet"]["stats"]
#
#fig, ax = plt.subplots(1, 3)
#ax[0].hist(av_ref)
#ax[1].hist(av_gen)
#ax[2].hist(delta_fd)
#plt.show()
