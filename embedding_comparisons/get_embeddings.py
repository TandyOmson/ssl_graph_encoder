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

# load SMILES
gen_smis = [i.strip() for i in open("random_transformer_gen.smi")]
ref_smis = [i.strip() for i in open("training_set.smi")]
#gen_smis = [i.strip() for i in open("gen_test.smi")]
#ref_smis = [i.strip() for i in open("ref_test.smi")]

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
        
        gen_embs, gen_fails = get_embeddings(gen_smis, encoder)
        ref_embs, ref_fails = get_embeddings(ref_smis, encoder)
    
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

np.savez("gen_embeddings.npz", **gen_dict)
np.savez("ref_embeddings.npz", **ref_dict)

