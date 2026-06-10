""" Data preprocessing for molecule embedding
    - Load SMILES, convert to rkdit mol objects
    - Generate graphs from rdkit mols
    - Can include atom features, bond adjacency matrix, bond features
"""

import torch
import torch_geometric
from torch.utils.data import random_split
from torch_geometric.data import InMemoryDataset

class MoleculeDataset(InMemoryDataset):
    def __init__(self, path, aug_1=None, aug_2=None):
        super().__init__()
        # required to avoid a weights_only=False error, may need to rethink if adding extra data (such as attr names)
        torch.serialization.add_safe_globals([
            torch_geometric.data.data.DataEdgeAttr, 
            torch_geometric.data.data.DataTensorAttr,
            torch_geometric.data.storage.GlobalStorage
            ])
        self.data, self.slices = torch.load(path)

        self.aug_1 = aug_1
        self.aug_2 = aug_2
        self.return_views = True

    def get(self, idx):
        data = super().get(idx)

        if self.aug_1 is None or self.aug_2 is None:
            return data

        view1 = self.aug_1.aug_func(data)
        view2 = self.aug_2.aug_func(data)

        return view1, view2

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