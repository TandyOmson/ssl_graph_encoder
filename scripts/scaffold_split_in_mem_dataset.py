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

def rdkit_scaffold_split(mols, frac_train=0.8, frac_test=0.1, frac_valid=0.1):

    assert abs(frac_train + frac_valid + frac_test - 1.0) < 1e-6

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
    valid_cutoff = (frac_train + frac_valid) * n_total

    train_idx, valid_idx, test_idx = [], [], []

    for s in scaffold_sets:
        if len(train_idx) + len(s) <= train_cutoff:
            train_idx.extend(s)
        elif len(train_idx) + len(valid_idx) + len(s) <= valid_cutoff:
            valid_idx.extend(s)
        else:
            test_idx.extend(s)

    return train_idx, valid_idx, test_idx, scaffold_to_indices


parser = argparse.ArgumentParser()
parser.add_argument("--smi", required=True, help="Original .smi file used to generate 3d data")
parser.add_argument("--pt", required=True, help="3d .pt file with dataset")
parser.add_argument("--confs", required=True, help="3d .pt file with alternative conformers")
args = parser.parse_args()

smis = [i.strip() for i in open(args.smi, "r").readlines()]

data, slices = torch.load(args.pt, weights_only=False)
base_name = ".".join(args.pt.split(".")[:-1])

confs_data, confs_slices = torch.load(args.confs, weights_only=False)
base_name_confs = ".".join(args.confs.split(".")[:-1])

# filter for failures
smis = [smis[i] for i in data.sample_id]
mols = [Chem.MolFromSmiles(s) for s in smis]

train_idx, valid_idx, test_idx, _ = rdkit_scaffold_split(mols)

# recreate datasets with specified indices
dataset = InMemoryDataset()
dataset.data = data
dataset.slices = slices

confs_dataset = InMemoryDataset()
confs_dataset.data = confs_data
confs_dataset.slices = confs_slices

graphs = [dataset.get(i) for i in train_idx]
train_data, train_slices = InMemoryDataset.collate(graphs)
torch.save((train_data, train_slices), f"{base_name}_train.pt")    

graphs = [dataset.get(i) for i in test_idx]
test_data, test_slices = InMemoryDataset.collate(graphs)
torch.save((test_data, test_slices), f"{base_name}_test.pt")    

graphs = [dataset.get(i) for i in valid_idx]
valid_data, valid_slices = InMemoryDataset.collate(graphs)
torch.save((valid_data, valid_slices), f"{base_name}_val.pt")    

mol_to_confs = {}
for i in range(confs_dataset.len()):
    data_i = confs_dataset.get(i)
    sid = int(data_i.sample_id)

    if sid not in mol_to_confs:
        mol_to_confs[sid] = []
    mol_to_confs[sid].append(i)

# --- Expand splits from molecule → conformers ---
train_confs_idx = []
for i in train_idx:
    if i in mol_to_confs:
        train_confs_idx.extend(mol_to_confs[i])

valid_confs_idx = []
for i in valid_idx:
    if i in mol_to_confs:
        valid_confs_idx.extend(mol_to_confs[i])

test_confs_idx = []
for i in test_idx:
    if i in mol_to_confs:
        test_confs_idx.extend(mol_to_confs[i])

# --- Save conformer datasets ---
graphs = [confs_dataset.get(i) for i in train_confs_idx]
train_data, train_slices = InMemoryDataset.collate(graphs)
torch.save((train_data, train_slices), f"{base_name_confs}_train.pt")

graphs = [confs_dataset.get(i) for i in test_confs_idx]
test_data, test_slices = InMemoryDataset.collate(graphs)
torch.save((test_data, test_slices), f"{base_name_confs}_test.pt")

graphs = [confs_dataset.get(i) for i in valid_confs_idx]
valid_data, valid_slices = InMemoryDataset.collate(graphs)
torch.save((valid_data, valid_slices), f"{base_name_confs}_val.pt")

# --- Save SMILES --- #
train_smis = [smis[i] for i in train_idx]
valid_smis = [smis[i] for i in valid_idx]
test_smis  = [smis[i] for i in test_idx]

with open(f"{base_name}_train.smi", "w") as f:
    f.write("\n".join(train_smis) + "\n")

with open(f"{base_name}_val.smi", "w") as f:
    f.write("\n".join(valid_smis) + "\n")

with open(f"{base_name}_test.smi", "w") as f:
    f.write("\n".join(test_smis) + "\n")
