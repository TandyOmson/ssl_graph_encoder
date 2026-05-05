import argparse
import pandas as pd
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset

from ssl_graph_encoder.utils.smiles_to_graph import SmilesToGraph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, required=True,
                        help="CSV with column 'labels', index is sample index")
    parser.add_argument("--smi", type=str, required=True, help="Path to file with SMILES")
    parser.add_argument("--output", type=str, required=True, help="Output .pt for primary graphs (conf 0)")
    parser.add_argument("--conf_out", type=str, required=True, help="Output .pt for conf pool graphs (conf 1..n)")
    parser.add_argument("--n_confs", type=int, required=True, help="Total number of conformers to generate")
    parser.add_argument("--config_out", type=str, required=True, help="Where to save SmilesToGraph config JSON")

    args = parser.parse_args()

    smis = [i.strip() for i in open(args.smi, "r", encoding="utf-8").readlines() if i.strip()]

    # Choose your feature set here
    atom_features = ["species"]  # add more if needed
    bond_features = []           # e.g. ["bond_type", "is_in_ring"]

    # Fit vocabs from the training SMILES set, then save the config for deployment
    stg = SmilesToGraph.fit_from_smiles(
        smis,
        atom_feature_names=atom_features,
        bond_feature_names=bond_features,
        add_hs=True,
        use_3d=True,
        max_confs=args.n_confs,
        include_unk=False,  # or True if you want explicit UNK bucket
    )
    stg.save_config(args.config_out)

    graphs = []
    graphs_conf_pool = []
    failure_ids = []

    for sample_id, smi in tqdm(list(enumerate(smis)), total=len(smis), desc="SMILES->graphs"):
        try:
            g0, pool = stg.smiles_to_graphs(smi, return_all_confs=True, optimise_mmff=True)
        except Exception:
            failure_ids.append(sample_id)
            continue

        g0.sample_id = sample_id
        graphs.append(g0)

        for g in pool:
            g.sample_id = sample_id
            graphs_conf_pool.append(g)

    # labels
    labels_df = pd.read_csv(args.labels, index_col=0)
    if len(labels_df.index) > 0 and labels_df.index[0] == 1:
        labels_df.index = labels_df.index - 1

    labels_df = labels_df.loc[~labels_df.index.isin(failure_ids)]
    label_map = labels_df["labels"].to_dict()
    labeled_ids = set(label_map.keys())

    graphs = [g for g in graphs if g.sample_id in labeled_ids]
    graphs_conf_pool = [g for g in graphs_conf_pool if g.sample_id in labeled_ids]

    for g in graphs:
        g.y = torch.tensor(label_map[g.sample_id], dtype=torch.float32).view(1)

    assert len({g.sample_id for g in graphs}) == len(graphs)

    data, slices = InMemoryDataset.collate(graphs)
    torch.save((data, slices), args.output)

    data, slices = InMemoryDataset.collate(graphs_conf_pool)
    torch.save((data, slices), args.conf_out)

    print("Saved:")
    print("  transformer config:", args.config_out)
    print("  main dataset:", args.output)
    print("  conf pool:", args.conf_out)

# EXAMPLE EXTERNAL USAGE
# -> load encoder class
# from ssl_graph_encoder.utils.smiles_to_graph import SmilesToGraph
#
# stg = SmilesToGraph.from_config("smiles_to_graph_config.json")
#
# g0, _ = stg.smiles_to_graphs("CCO", return_all_confs=False)
# -> feed g0 to pretrained encoder