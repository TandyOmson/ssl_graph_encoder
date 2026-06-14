""" Retrieves ECFP4 fingerprints
"""
import argparse
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np

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
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)    

    print("generating embeddings")
    embs = [fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smis]

    np.save(args.outfile, embs)