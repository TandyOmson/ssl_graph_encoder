"""
Build a self-contained SMILES encoder (bidirectional GRU)
Compatible with load_pretrained_encoder pattern
"""
from pathlib import Path
import argparse
import torch
import torch.nn as nn
from ssl_graph_encoder.models.encoders.ecfp4_benchmark_encoder import ECFP4Encoder

def main(args):
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
    parser.add_argument("--outfile", required=True)

    parser.add_argument("--embed_dim", type=int, default=128)

    args = parser.parse_args()
    main(args)
