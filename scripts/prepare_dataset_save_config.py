import argparse
import pandas as pd
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset
from joblib import Parallel, delayed
from pathlib import Path
from ssl_graph_encoder.utils.smiles_to_graph import SmilesToGraph

def stream_smis(path):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            smi = line.strip()
            if smi:
                yield i, smi

def process_smiles(sample_id, smi):
        try:
            # _STG is global SmilesToGraph instance
            g0, pool = _STG.smiles_to_graphs(smi, return_all_confs=True, optimise_mmff=True)
        except Exception:
            return None

        g0.sample_id = sample_id
        for g in pool:
            g.sample_id = sample_id 

        return g0, pool

def apply_labels_and_filter(graphs, graphs_conf_pool, label_map, labeled_ids):
    if label_map is None:
        return graphs, graphs_conf_pool

    # filter graphs with labels only
    graphs = [g for g in graphs if g.sample_id in labeled_ids]
    graphs_conf_pool = [
        g for g in graphs_conf_pool if g.sample_id in labeled_ids
    ]

    # attach labels
    for g in graphs:
        g.y = torch.tensor(
            label_map[g.sample_id],
            dtype=torch.float32
        ).view(1)

    return graphs, graphs_conf_pool

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, required=False, help="OPTIONAL: CSV with column 'labels', index is sample index")
    parser.add_argument("--smi", type=str, required=True, help="Path to file with SMILES")
    parser.add_argument("--output", type=str, required=True, help="Output .pt for primary graphs (conf 0)")
    parser.add_argument("--conf_out", type=str, required=True, help="Output .pt for conf pool graphs (conf 1..n)")
    parser.add_argument("--n_confs", type=int, required=True, help="Total number of conformers to generate")
    parser.add_argument("--config_out", type=str, required=True, help="Where to save SmilesToGraph config JSON")
    parser.add_argument("--save_every", type=int, default=50000, help="Splits up large jobs")
    parser.add_argument("--n_workers", type=int, default=1, help="Number of joblib workers for embedding step")

    args = parser.parse_args()

    args.smi = Path(args.smi)
    args.output = Path(args.output)
    args.conf_out = Path(args.conf_out)
    args.config_out = Path(args.config_out)

    smis = [i.strip() for i in open(args.smi, "r", encoding="utf-8").readlines() if i.strip()]

    # Choose your feature set here
    atom_features = ["species"]  # add more if needed
    bond_features = []           # e.g. ["bond_type", "is_in_ring"]

    print("atom features:", atom_features)
    print("bond features:", bond_features)

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
    print("saved SMILES->Graph config:", args.config_out)

    label_map = None
    labeled_ids = None

    # preprocess labels
    if args.labels is not None:
        labels_df = pd.read_csv(args.labels, index_col=0)

        if len(labels_df.index) > 0 and labels_df.index[0] == 1:
            labels_df.index = labels_df.index - 1

        label_map = labels_df["labels"].to_dict()
        labeled_ids = set(label_map.keys())

    _STG = stg # for parallel processing
    print(f"Embedding {len(smis)} molecules with {args.n_confs} conformers each...")

    results = Parallel(
        n_jobs=args.n_workers,
        backend="threading",
        return_as="generator",
        batch_size=8,
    )(
        delayed(process_smiles)(i, smi)
        for i, smi in stream_smis(args.smi)
    )

    graphs = []
    graphs_conf_pool = []
    failure_ids = []

    buffer_size = args.save_every
    chunk_idx = 0

    for sample_id, res in tqdm(
        zip(range(len(smis)), results),
        total=len(smis),
        desc="SMILES->Graphs",
    ):
        if res is None:
            failure_ids.append(sample_id)
            continue

        g0, pool = res
        
        graphs.append(g0)
        graphs_conf_pool.extend(pool)

        if len(graphs) >= buffer_size:
            print(f"Saving chunk {chunk_idx}...")

            graphs_proc = []
            if args.labels is not None:
                graphs, graphs_conf_pool = apply_labels_and_filter(
                        graphs,
                        graphs_conf_pool,
                        label_map,
                        labeled_ids
                    )
            data, slices = InMemoryDataset.collate(graphs)
            torch.save((data, slices), f"{args.output}.part{chunk_idx}")

            data, slices = InMemoryDataset.collate(graphs_conf_pool)
            torch.save((data, slices), f"{args.conf_out}.part{chunk_idx}")

            graphs.clear()
            graphs_conf_pool.clear()

            chunk_idx += 1
        
    if graphs:
        print(f"Saving final chunk {chunk_idx}...")

        graphs_proc = []
        if args.labels is not None:
            graphs, graphs_conf_pool = apply_labels_and_filter(
                graphs,
                graphs_conf_pool,
                label_map,
                labeled_ids
            )

        data, slices = InMemoryDataset.collate(graphs)
        torch.save((data, slices), f"{args.output}.part{chunk_idx}")

        data, slices = InMemoryDataset.collate(graphs_conf_pool)
        torch.save((data, slices), f"{args.conf_out}.part{chunk_idx}")

    print("Printing failure ids to embed_failures.txt")
    with open("embed_failures.txt", "w") as fw:
        for i in failure_ids:
            fw.write(f"{i}\n")
    print("Done")

# EXAMPLE EXTERNAL USAGE
# -> load encoder class
# from ssl_graph_encoder.utils.smiles_to_graph import SmilesToGraph
#
# stg = SmilesToGraph.from_config("smiles_to_graph_config.json")
#
# g0, _ = stg.smiles_to_graphs("CCO", return_all_confs=False)
# -> feed g0 to pretrained encoder