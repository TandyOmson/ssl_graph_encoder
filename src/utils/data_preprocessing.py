""" Data preprocessing for molecule embedding
    - Load SMILES, convert to rkdit mol objects
    - Generate graphs from rdkit mols
    - Can include atom features, bond adjacency matrix, bond features
"""

import torch
from torch.utils.data import random_split
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
import pandas as pd
import numpy as np

#
# Function for loading molecule data as SMILES
#
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

def read_smiles_file(smi_file, add_hs=False):
    """ Read a file of smiles
    """
    smis = [i.rstrip() for i in open(smi_file, 'r').readlines()]
    mols = [smi_to_mol(smi, add_hs=add_hs) for smi in smis]

    return mols

def read_affins_csv(affins_csv):
    """ Slightly hardcoded for now 
    """
    affins_df = pd.read_csv(affins_csv, index_col=0)
    affins_df = affins_df[affins_df["pose_1"].between(-20, 0)]
    labels = affins_df["pose_1"].values

    return labels, affins_df.index.values

#
# Functions for loading graph data
#
def load_graphs_from_dataset(node_attr, node_idxs, edges, label_file):
    """ REDUNDANT, KEEPING FOR LATER
        Based on NCI-1 dataset, load graphs and labels from given files
    """
    # should do a check to confirm that node_gen and idx_gen are the same length
    # each line i has value of ith node: format node_attr1, node_attr2, ...
    node_gen = ([float(i) for i in line.strip().split(",")][1:] for line in open(node_attr, "r"))
    # each line i has value of ith node: format graph_idx it belongs to
    idx_gen = (float(line.strip()[0]) for line in open(node_idxs, "r"))
    # each line i has value of ith edge: format source_node_idx, target_node_idx
    edges = ([float(i) for i in line.strip().split(",")] for line in open(edges, "r"))

    all_graphs = []
    edge_indexes = []
    
    # temp lists/counters
    graph_nodes = []
    graph_idx = 1
    node_count = 0
    while True:
        try:
            idx = next(idx_gen)
            if idx != graph_idx:
                # triggered once end of a graph is reached
                # node idxs for current graph are between node_count and node_count + len(graph_nodes)
                # use to construct edge index
                node_count += len(graph_nodes)
                this_node_edges = []
                while True:
                    edge = next(edges)
                    if edge[1] < node_count:
                        this_node_edges.append(edge)
                    else:
                        edge_indexes.append(torch.tensor(this_node_edges, dtype=torch.long).t().contiguous())
                        this_node_edges = []
                        this_node_edges.append(edge)
                        break

                graph_idx += 1
                all_graphs.append(torch.stack(graph_nodes))
                graph_nodes = []                
                
            node_attr = next(node_gen)
            graph_nodes.append(torch.tensor(node_attr, dtype=torch.long))
        except StopIteration:
            break

    graphs = [Data(x=nodes, edge_index=edge_index) for nodes, edge_index in zip(all_graphs, edge_indexes)]

    # NEED TO ADD CHECK FOR INT OR FLOAT, OR JUST HANDLE CLASSIFICATION
    labels = np.array([int(line.strip()[0]) for line in open(label_file, "r")])
    return graphs, labels

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