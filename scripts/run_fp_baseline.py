""" Baseline fingerprint model
    atom pair fingerprints
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from src.utils.evaluate_embeddings import supervised_embedding_eval, unsupervised_embedding_eval

datadir = "/home/andyt/DProjects/Dself_sup_graph_learning/molecule_embedding_proj/data"

smis = [i.rstrip() for i in open(f'{datadir}/raw/hydros_minimal.smi', 'r').readlines()]
mols = [Chem.MolFromSmiles(smi) for smi in smis]
fpgen = rdFingerprintGenerator.GetAtomPairGenerator()
embeddings = np.array([fpgen.GetFingerprint(mol) for mol in mols])

affins_df = pd.read_csv(f"{datadir}/raw/vinardo_affins.csv", index_col=0)
affins_df = affins_df[affins_df["pose_1"].between(-20, 0)]
labels = affins_df["pose_1"].values

embeddings = embeddings[np.array(affins_df.index)-1,:]

lr_rmse, lr_r2, knn_rmse, knn_r2, spearman_corr = supervised_embedding_eval(embeddings, labels)
spread_mean, spread_median, dist_cv = unsupervised_embedding_eval(embeddings)

print(f"Ridge RMSE: {lr_rmse:.4f}, R2: {lr_r2:.4f}")
print(f"kNN RMSE: {knn_rmse:.4f}, R2: {knn_r2:.4f}")
print(f"Spearman correlation: {spearman_corr:.4f}")
print(f"Spread (mean): {spread_mean:.4f}")
print(f"Spread (median): {spread_median:.4f}")
print(f"Distance CV: {dist_cv:.4f}")

np.savez(f"{datadir}/processed/fp_baseline.npz", embeddings=embeddings, labels=labels)