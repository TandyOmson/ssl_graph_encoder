"""
Build a self-contained SMILES encoder (bidirectional GRU)
Compatible with load_pretrained_encoder pattern
"""
from pathlib import Path
import argparse
import torch
import torch.nn as nn
from ssl_graph_encoder.models.encoders.smiles_benchmark_encoder import SmilesEncoder

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
    model = SmilesEncoder(
        feat_dim=None,
        embed_dim=args.embed_dim,
        stoi=stoi,
        unk_idx=unk_idx,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        max_len=args.max_len,
    )

    # initialise
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # ===== payload that matches your loader =====
    payload = {
        "encoder_class_path": "ssl_graph_encoder.models.encoders.smiles_benchmark_encoder.SmilesEncoder",
        "feat_dim": None,  # no node features like GNN
        "embed_dim": model.output_dim,
        "encoder_kwargs": {
            "kwargs": {
                "stoi": stoi,
                "unk_idx": unk_idx,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "max_len": args.max_len,
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
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=120)

    args = parser.parse_args()
    main(args)
