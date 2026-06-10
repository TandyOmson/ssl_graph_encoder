""" Dataset parts (for large sets) have to be joined by reloading and collating
    Remember must also be done for conformers
"""
import sys
import torch
from torch_geometric.data import InMemoryDataset

all_graphs = []
part_files = sys.argv[1:]

print(f"joining {len(part_files)} datasets")
for f in part_files:
    data, slices = torch.load(f, weights_only=False)

    dataset = InMemoryDataset()
    dataset.data = data
    dataset.slices = slices

    graphs = [dataset.get(i) for i in range(dataset.len())]
    all_graphs.extend(graphs)

final_data, final_slices = InMemoryDataset.collate(all_graphs)
torch.save((final_data, final_slices), f"{part_files[0].split('.')[0]}_combined.pt")
