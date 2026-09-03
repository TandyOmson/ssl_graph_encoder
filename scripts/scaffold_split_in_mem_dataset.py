""" Applies scaffold splitting to dataset from .pt file
"""
import argparse
import torch
from torch_geometric.data import InMemoryDataset
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import random

def generate_scaffold(mol):
    scaffold =  MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    if not scaffold:
        return Chem.MolToSmiles(mol)
    return scaffold

def rdkit_scaffold_split(mols, frac_train=0.8, frac_test=0.1, frac_val=0.1):

    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6

    rng = random.Random(42)

    # 1. Group by scaffold
    scaffold_to_indices = {}
    for i, mol in enumerate(mols):
        scaffold = generate_scaffold(mol)
        if not scaffold:
            scaffold = str(i)  # optional fix for acyclic molecules
        scaffold_to_indices.setdefault(scaffold, []).append(i)

    scaffold_sets = list(scaffold_to_indices.values())

    # 2. Sort by size (important)
    scaffold_sets = sorted(scaffold_sets, key=len, reverse=True)

    # 3. Introduce slight randomness
    noisy_scaffold_sets = []
    for s in scaffold_sets:
        noise = rng.uniform(-0.5, 0.5)
        noisy_scaffold_sets.append((-(len(s) + noise), s))

    noisy_scaffold_sets.sort()  # still roughly size-based but jittered
    scaffold_sets = [s for _, s in noisy_scaffold_sets]

    # 4. Assign splits
    n_total = len(mols)
    train_cutoff = frac_train * n_total
    val_cutoff = (frac_train + frac_val) * n_total

    train_idx, val_idx, test_idx = [], [], []

    for s in scaffold_sets:
        if len(train_idx) + len(s) <= train_cutoff:
            train_idx.extend(s)
        elif len(train_idx) + len(val_idx) + len(s) <= val_cutoff:
            val_idx.extend(s)
        else:
            test_idx.extend(s)

    return train_idx, val_idx, test_idx, scaffold_to_indices


parser = argparse.ArgumentParser()
parser.add_argument("--smi", required=True, help="Original .smi file used to generate 3d data")
parser.add_argument("--pt", required=True, help="3d .pt file with dataset")
parser.add_argument("--labels", type=bool, default=False, help="output SMILES with labels")
args = parser.parse_args()

smis = [i.strip() for i in open(args.smi, "r").readlines()]

data, slices = torch.load(args.pt, weights_only=False)
base_name = ".".join(args.pt.split(".")[:-1])

# filter for failures
smis = [smis[i] for i in data.sample_id]
mols = [Chem.MolFromSmiles(s) for s in smis]

train_idx, val_idx, test_idx, _ = rdkit_scaffold_split(mols)

# recreate datasets with specified indices
dataset = InMemoryDataset()
dataset.data = data
dataset.slices = slices

# save SMILES
train_smis = [smis[i] for i in train_idx]
val_smis = [smis[i] for i in val_idx]
test_smis  = [smis[i] for i in test_idx]

graphs = [dataset.get(i) for i in train_idx]
for count, i in enumerate(train_smis):
    graphs[count].smiles = i
train_data, train_slices = InMemoryDataset.collate(graphs)
torch.save((train_data, train_slices), f"{base_name}_train.pt")    

graphs = [dataset.get(i) for i in test_idx]
for count, i in enumerate(test_smis):
    graphs[count].smiles = i
test_data, test_slices = InMemoryDataset.collate(graphs)
torch.save((test_data, test_slices), f"{base_name}_test.pt")    

graphs = [dataset.get(i) for i in val_idx]
for count, i in enumerate(val_smis):
    graphs[count].smiles = i
val_data, val_slices = InMemoryDataset.collate(graphs)
torch.save((val_data, val_slices), f"{base_name}_val.pt")

with open(f"{base_name}_train.smi", "w") as f:
    f.write("\n".join(train_smis) + "\n")

with open(f"{base_name}_val.smi", "w") as f:
    f.write("\n".join(val_smis) + "\n")

with open(f"{base_name}_test.smi", "w") as f:
    f.write("\n".join(test_smis) + "\n")


