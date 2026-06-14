""" Retrieve SSL embeddings for a given model for a given set of SMILES
"""

from ssl_graph_encoder.utils.get_ssl_embeddings import sslEmbeddings
import argparse
from rdkit import Chem
import torch
import traceback
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_file", type=str, required=True)
    parser.add_argument("--smiles_to_graph_file", type=str, required=True)
    parser.add_argument("--smis", type=str, required=True)
    parser.add_argument("--outfile", type=str, required=True)

    args = parser.parse_args()

    print("loading and canonicalising smiles")
    smis = [i.rstrip() for i in open(args.smis, "r").readlines()]
    canon_smis = []
    for i in smis:
        try:
            smi = Chem.CanonSmiles(i)
            canon_smis.append(smi)
        except:
            continue

    smis = canon_smis
    
    print(f"loaded {len(smis)} SMILES")
    print("loading model")
    embedder = sslEmbeddings(args.encoder_file, args.smiles_to_graph_file)
    
    print("generating embeddings")
    embs = []
    emb_dim = 0
    for count, smi in enumerate(smis):
        print(f"gen mols {count} of {len(smis)}", end="\r")
        try:
            with torch.no_grad():
                emb = embedder.get_embeddings(smi)
            embs.append(emb[0])
            emb_dim = len(emb[0])
        except:
            print("embedding failed for", count, smi)
            embs.append(np.array([np.nan]*emb_dim))

    np.save(args.outfile, embs)

    
