""" Retrieves ChemNet embeddings
"""
import argparse
from rdkit import Chem
import numpy as np
from fcd import load_ref_model, get_predictions, canonical_smiles
import torch
import traceback

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    model = load_ref_model()

    print("generating embeddings")
    embs = get_predictions(model, smis)

    np.save(args.outfile, embs)