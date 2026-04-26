""" Prepares a graph dataset from SMILES (with or without labels)
    Takes the format of an InMemoryDataset as defined in torch_geometric

    - Load Graph Specification (and label spec if relevant)
    - Load SMILES/.sdf file
    - Convert to rdkit molecules objects
    - Convert to graphs (torch_geometric Data objects)
    - If labels provided, load labels and add to Data objects
    - Save as InMemoryDataset (data, slices) to output path
    - InMemoryDataset saves and loads based on whether the output dir exists!
"""

import argparse

import torch
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
import pandas as pd
import numpy as np

class FeatureSpec:
    def __init__(self, name, func, *, vocab=None):
        self.name = name
        self.func = func
        self.vocab = vocab
        self.is_categorical = vocab is not None

        if self.is_categorical:
            self.index = {v: i for i, v in enumerate(vocab)}
            self.dim = len(vocab)
        else:
            self.dim = 1

    def __call__(self, obj):
        """ one hot encoding for a sample for given feature spec
        """
        value = self.func(obj)

        if not self.is_categorical:
            return [float(value)]

        vec = [0] * self.dim
        if value in self.index:
            vec[self.index[value]] = 1
        return vec
    
def smi_to_mol(smi, add_hs=False):
    mol = Chem.MolFromSmiles(smi)
    if add_hs:
        mol = Chem.AddHs(mol)
    # room to add custom sanitization functions
    return mol

def mol_to_graph(mol, node_specs=None, edge_specs=None, pos_3d=False):
    
    node_features = []
    if node_specs:
        for atom in mol.GetAtoms():
            features = sum((f(atom) for f in node_specs), [])
            node_features.append(features)

        node_attr = torch.tensor(node_features, dtype=torch.float32)
    else:
        node_attr = torch.zeros((mol.GetNumAtoms(), 1), dtype=torch.float32)

    edge_features = []
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
    
        # Two way message passing requires both edges and their features!
        edges.append([i, j])
        edges.append([j, i])

        if edge_specs:
            features = sum((f(bond) for f in edge_specs), [])
            edge_features.append(features)
            edge_features.append(features)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    data = Data(x=node_attr, edge_index=edge_index)

    if edge_specs and edge_features:
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        data.edge_attr = edge_attr

    if pos_3d:
        if mol.GetConformer().Is3D():
            pos = np.array([[p.x, p.y, p.z] for p in mol.GetConformer().GetPositions()])
            pos = torch.tensor(pos, dtype=torch.float32)
            data.pos = pos
        else:
            print("3D was specified, but no coordinates were found in a data object...")
            raise Exception

    return data

def add_nitrogen_charges(m):
    m.UpdatePropertyCache(strict=False)
    ps = Chem.DetectChemistryProblems(m)
    if not ps:
        Chem.SanitizeMol(m)
        return m
    for p in ps:
        if p.GetType()=='AtomValenceException':
            at = m.GetAtomWithIdx(p.GetAtomIdx())
            if at.GetAtomicNum()==7 and at.GetFormalCharge()==0 and at.GetExplicitValence()==4:
                at.SetFormalCharge(1)
            if at.GetAtomicNum()==7 and at.GetFormalCharge()==0:
                bondcount = 0
                for b in at.GetBonds():
                    bondcount += b.GetBondTypeAsDouble()
                if int(bondcount) > 3:
                    at.SetFormalCharge(1)
                
    Chem.SanitizeMol(m)
    return m

def get_vocab(mols):
    species = set()
    for m in mols:
        for a in m.GetAtoms():
            species.add(a.GetAtomicNum())
    vocab = list(species)
    vocab.sort()
    return vocab

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, required=False, help="Path to .csv file with columns as labels, index is sample index")
    parser.add_argument("--mols", type=str, required=True, help="Path to file with SMILES, or .sdf file if --pos_3d True")
    parser.add_argument("--output", type=str, required=True, help="Path to output .pt file to save dataset in")
    parser.add_argument("--pos_3d", type=bool, default=False, help="Whether to include 3D positions in graph data")

    args = parser.parse_args()

    if not args.pos_3d:
        smis = [i.strip() for i in open(args.mols, 'r').readlines()]
        mols = [smi_to_mol(smi, add_hs=True) for smi in smis]
    else:
        try:
            mols = Chem.SDMolSupplier(args.mols, sanitize=False, removeHs=False)
            mols = [add_nitrogen_charges(m) for m in mols]
        except:
            print("pos 3d was selected, cannot read .sdf from --mols argument")
            raise Exception
    
    vocab = get_vocab(mols)

    # GRAPH SPECIFICATION
    # node/atom features: accept rdkit atom object
    # Atom identity (categorical → one-hot)
    node_features = [
        FeatureSpec(
            name="species",
            func=lambda atom: atom.GetAtomicNum(),
            vocab=vocab
        ),

        # Local topology (numeric)
        FeatureSpec(
            name="degree",
            func=lambda atom: atom.GetDegree(),
        ),

        FeatureSpec(
            name="implicit_hydrogens",
            func=lambda atom: atom.GetNumImplicitHs(),
        ),

        FeatureSpec(
            name="explicit_hydrogens",
            func=lambda atom: atom.GetNumExplicitHs(),
        ),

        # Electronic properties (numeric)
        FeatureSpec(
            name="formal_charge",
            func=lambda atom: atom.GetFormalCharge(),
        ),

        # Aromaticity (binary scalar, numeric)
        FeatureSpec(
            name="is_aromatic",
            func=lambda atom: int(atom.GetIsAromatic()),
        ),

        # Hybridization (categorical → one-hot)
        FeatureSpec(
            name="hybridization",
            func=lambda atom: atom.GetHybridization(),
            vocab=[
                Chem.HybridizationType.SP,
                Chem.HybridizationType.SP2,
                Chem.HybridizationType.SP3,
                Chem.HybridizationType.SP3D,
                Chem.HybridizationType.SP3D2,
            ],
        ),
    ]

    node_feature_dim = sum(f.dim for f in node_features)
    
    # edge/bond features: accept rdkit bond object
    edge_features = [
        # Bond type (categorical → one-hot)
        FeatureSpec(
            name="bond_type",
            func=lambda bond: bond.GetBondType(),
            vocab=[
                Chem.BondType.SINGLE,
                Chem.BondType.DOUBLE,
                Chem.BondType.TRIPLE,
                Chem.BondType.AROMATIC,
            ],
        ),

        # Conjugation (binary scalar)
        FeatureSpec(
            name="is_conjugated",
            func=lambda bond: int(bond.GetIsConjugated()),
        ),

        # Aromaticity (binary scalar)
        FeatureSpec(
            name="is_aromatic",
            func=lambda bond: int(bond.GetIsAromatic()),
        ),

        # Ring membership (binary scalar)
        FeatureSpec(
            name="is_in_ring",
            func=lambda bond: int(bond.IsInRing()),
        ),
    ]



    # TEMPORARY, later can create a config file that chooses node and edge features
    node_features_active = {name: True for name in [i.name for i in node_features]}
    edge_features_active = {name: True for name in [i.name for i in edge_features]}

    # for now, selecting manually
    node_features_active = {
        "species": True,        
    }
    edge_features_active = {
    }

    active_node_specs = [
        n for n in node_features
        if node_features_active.get(n.name, False)
    ]

    active_edge_specs = [
        n for n in edge_features
        if edge_features_active.get(n.name, False)
    ]

    print("Node feature dimension:", len(active_node_specs))
    print("Edge feature dimension:", len(active_edge_specs))
    if len(active_edge_specs) == 0:
        active_edge_specs = None

    graphs = [
        mol_to_graph(mol, node_specs=active_node_specs, edge_specs=active_edge_specs, pos_3d=args.pos_3d)
        for mol in mols
    ] # torch_geometric.data.Data objects

    # load labels
    labels_df = pd.read_csv(args.labels, index_col=0)
    if labels_df.index[0] == 1:
        labels_df.index = labels_df.index - 1
    # apply desired filters to prepare label file
    labels = labels_df["labels"].values

    graphs = [graphs[i] for i in labels_df.index]

    for data, y in zip(graphs, labels):
        data.y = torch.tensor(y, dtype=torch.float32).view(1)
    
    ys = [data.y for data in graphs]
    assert all(y.shape == ys[0].shape for y in ys)
    assert all(y.dtype == ys[0].dtype for y in ys)

    data, slices = InMemoryDataset.collate(graphs)
    torch.save((data, slices), args.output)
