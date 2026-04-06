""" Data preprocessing for molecule embedding
    - Load SMILES, convert to rkdit mol objects
    - Generate graphs from rdkit mols
    - Can include atom features, bond adjacency matrix, bond features
"""

import torch
from torch.utils.data import random_split
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem

def atom_features(atom):
    return torch.tensor([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),
        atom.GetNumImplicitHs(),
        int(atom.GetIsAromatic())
    ], dtype=torch.float)

def bond_features(bond):
    return torch.tensor([
        bond.GetBondTypeAsDouble(),
        bond.GetIsConjugated(),
        bond.IsInRing()
    ], dtype=torch.float)

def smi_to_mol(smi, add_hs=False):
    """ Including to add custom sanitization later
    """
    mol = Chem.MolFromSmiles(smi)
    if add_hs:
        mol = Chem.AddHs(mol)
    return mol

def mol_to_graph(mol):
    # Node features
    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()])

    # Get a 2D tensor, first row is source nodes, second is target nodes
    # A sparse version of the edge adj. matrix
    edge_attrs = []
    edges = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
    
        bf = bond_features(bond)
    
        edges.append([i, j])
        edge_attrs.append(bf)
    
        edges.append([j, i])
        edge_attrs.append(bf)
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attrs)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class MoleculeDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__()
        self.data, self.slices = self.collate(data_list)

def split_dataset(dataset, val_frac=0.1, test_frac=0.1, random_seed=42):
    """ Split dataset into train/val/test sets
    """
    N = len(dataset)
    val_size = int(val_frac * N)
    test_size = int(test_frac *N)
    train_size = N - val_size - test_size

    generator = torch.Generator().manual_seed(random_seed)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        lengths=[train_size, val_size, test_size],
        generator=generator
    )
    return train_dataset, val_dataset, test_dataset