""" DEPRECATED: use prepare_dataset_save_config.py

 Prepares a graph dataset from SMILES (with or without labels)
    - Load Graph Specification
    - Load SMILES
    - Convert to rdkit molecule objects
    - Embed n conformers using MMFF94
    - Convert all conformers to graphs
    - Create data objects with ONLY conformer 1 and add labels (if provided)
    - Save conformers 2 to n in a separate file
    - Save dataset as InMemoryDataset (data, slices)
"""

import argparse

import torch
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from tqdm import tqdm

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

def mol_to_graph(mol, node_specs=None, edge_specs=None, pos_3d=False, conf_id=0):
    
    node_features = []
    if node_specs:
        for atom in mol.GetAtoms():
            features = sum((f(atom) for f in node_specs), [])
            node_features.append(features)

        node_attr = torch.tensor(node_features, dtype=torch.float32)
    else:
        node_attr = torch.zeros((mol.GetNumAtoms(), 1), dtype=torch.float32)

    atomic_nums = []
    for atom in mol.GetAtoms():
        atomic_nums.append(atom.GetAtomicNum())
    atomic_nums = torch.tensor(atomic_nums, dtype=torch.long)

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
    
    data = Data(x=node_attr, z=atomic_nums, edge_index=edge_index)

    if edge_specs and edge_features:
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        data.edge_attr = edge_attr

    if pos_3d:
        if mol.GetConformer().Is3D():
            pos = np.array([[p[0], p[1], p[2]] for p in mol.GetConformer(conf_id).GetPositions()])
            pos = torch.tensor(pos, dtype=torch.float32)
            data.pos = pos
        else:
            print("3D was specified, but no coordinates were found in a data object...")
            raise Exception

    return data

def get_vocab(mols):
    species = set()
    for m in mols:
        for a in m.GetAtoms():
            species.add(a.GetAtomicNum())
    vocab = list(species)
    vocab.sort()
    return vocab

if __name__ == "__main__":
    print("DEPRECATED: use prepare_dataset_save_config.py instead")

    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, required=False, help="Path to .csv file with columns as labels, index is sample index")
    parser.add_argument("--smi", type=str, required=True, help="Path to file with SMILES")
    parser.add_argument("--output", type=str, required=True, help="Path to output .pt file to save dataset in")
    parser.add_argument("--conf_out", type=str, required=True, help="Path to output .pt file to save graphs of conformers 2 to n")
    parser.add_argument("--n_confs", type=int, required=True, help="Total number of conformers to generate")
    parser.add_argument("--save_every", type=int, required=False, help="For large files, save embedded structures to output every n molecules")

    args = parser.parse_args()
    
    smis = [i.strip() for i in open(args.smi, 'r').readlines()]
    mols = [smi_to_mol(smi, add_hs=True) for smi in smis]
    print(f"made {len(mols)} mol objects")

    species_vocab = get_vocab(mols)
    print("vocab", species_vocab)
    # GRAPH SPECIFICATION
    # node/atom features: accept rdkit atom object
    # Atom identity (categorical → one-hot)
    node_features = [
        FeatureSpec(
            name="species",
            func=lambda atom: atom.GetAtomicNum(),
            vocab=species_vocab
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

    # Embed conformers
    print("saving atomic num longs to data.z")
    graphs = []
    graphs_conf_pool = []
    failure_ids = []
    for sample_id, m in tqdm(enumerate(mols), total=len(mols), desc="Embedding confs"):
        res = AllChem.EmbedMultipleConfs(m, numConfs=args.n_confs, numThreads=0)
        if list(res) == []:
            res  = AllChem.EmbedMultipleConfs(m, numConfs=args.n_confs, useBasicKnowledge=False, numThreads=0)
            if list(res) == []:
                print(f"complete embed failure for sample {sample_id}")
                failure_ids.append(sample_id)
                continue

        # sorted conf ids
        mmff = AllChem.MMFFOptimizeMoleculeConfs(m)
        res = list(res)

        pairs = list(zip(res, mmff))
        pairs.sort(key=lambda x: x[1][1])  # sort by energy

        conf_ids = [cid for cid, _ in pairs]
        g0 = mol_to_graph(m, node_specs=active_node_specs, edge_specs=active_edge_specs, pos_3d=True, conf_id=conf_ids[0])
        g0.sample_id = sample_id
        graphs.append(g0)

        for idx in conf_ids[1:]:
            g = mol_to_graph(m, node_specs=active_node_specs, edge_specs=active_edge_specs, pos_3d=True, conf_id=idx)
            g.sample_id = sample_id # save flattened, load by sample_id later
            graphs_conf_pool.append(g)

        if args.save_every is not None and sample_id % args.save_every == 0:
            data, slices = InMemoryDataset.collate(graphs)
            torch.save((data, slices), args.output)

            data, slices = InMemoryDataset.collate(graphs_conf_pool)
            torch.save((data, slices), args.conf_out)

    if args.labels is not None:
        print(f"loading labels from {args.labels}")
        # load labels
        labels_df = pd.read_csv(args.labels, index_col=0)
        if labels_df.index[0] == 1:
            labels_df.index = labels_df.index - 1
        
        # remove failed samples
        labels_df = labels_df.loc[~labels_df.index.isin(failure_ids)] 
        label_map = labels_df["labels"].to_dict()         
        labeled_ids = set(label_map.keys())
        
        # remove graph with no labels
        graphs = [g for g in graphs if g.sample_id in labeled_ids]
        graphs_conf_pool = [
            g for g in graphs_conf_pool if g.sample_id in labeled_ids
        ]
        
        # attach labels
        for g in graphs:
            g.y = torch.tensor(label_map[g.sample_id], dtype=torch.float32).view(1)
        
        assert len({g.sample_id for g in graphs}) == len(graphs)
        
        ys = [data.y for data in graphs]
        assert all(y.shape == ys[0].shape for y in ys)
        assert all(y.dtype == ys[0].dtype for y in ys)

    data, slices = InMemoryDataset.collate(graphs)
    torch.save((data, slices), args.output)

    data, slices = InMemoryDataset.collate(graphs_conf_pool)
    torch.save((data, slices), args.conf_out)
