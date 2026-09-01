"""
Build a self-contained SMILES encoder (bidirectional GRU)
Compatible with load_pretrained_encoder pattern
"""
from pathlib import Path
import argparse
import torch
import torch.nn as nn
from ssl_graph_encoder.models.encoders.ecfp4_benchmark_encoder import ECFP4Encoder

def load_smiles(file):
    with open(file, "r") as f:
        return [line.strip() for line in f if line.strip()]

def build_vocab(smiles_list):
    charset = set()
    for smi in smiles_list:
        charset.update(list(smi))

    charset = sorted(list(charset))

    stoi = {c: i + 1 for i, c in enumerate(charset)}  # 0 = padding
    unk_idx = len(stoi) + 1

    return stoi, unk_idx

def main(args):
    smiles = load_smiles(args.smiles_file)
    # vocab
    stoi, unk_idx = build_vocab(smiles)
    # model
    model = ECFP4Encoder(
        feat_dim=None,
        embed_dim=args.embed_dim,
    )

    # ===== payload that matches your loader =====
    payload = {
        "encoder_class_path": "ssl_graph_encoder.models.encoders.ecfp4_benchmark_encoder.ECFP4Encoder",
        "feat_dim": None,  # no node features like GNN
        "embed_dim": model.embed_dim,
        "encoder_kwargs": {
            "kwargs": {
            }
        },
        "state_dict": model.state_dict(),
    }

    torch.save(payload, Path(args.outfile))
    print(f"Saved encoder to {args.outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_file", required=True)
    parser.add_argument("--outfile", required=True)

    parser.add_argument("--embed_dim", type=int, default=128)

    args = parser.parse_args()
    main(args)
