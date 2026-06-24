"""
Build a randomly initialised SMILES encoder
This program is NOT optimised for language models.
This script outputs a langauge model with random parameters to be loaded as a pretrained encoder in finetuning only runs, for the purpose of benchmarking.
It ingests raw SMILES.

Bidirectional GRU
- Takes raw SMILES strings
- Handles tokenisation internally
- Saves EVERYTHING to a single .pt file
- Output: encoder(smiles_list) -> (B, D)
"""

from pathlib import Path
import argparse
import torch
import torch.nn as nn


# =========================
# Utilities
# =========================

def load_smiles(file):
    with open(file, "r") as f:
        smiles = [line.strip() for line in f if line.strip()]
    return smiles


def build_vocab(smiles_list):
    charset = set()
    for smi in smiles_list:
        charset.update(list(smi))

    charset = sorted(list(charset))

    # reserve:
    # 0 = padding
    # len(stoi)+1 = UNK
    stoi = {c: i + 1 for i, c in enumerate(charset)}
    unk_idx = len(stoi) + 1

    return stoi, unk_idx


# =========================
# Encoder (self-contained)
# =========================

class SmilesEncoder(nn.Module):
    def __init__(
        self,
        stoi,
        unk_idx,
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        max_len=120,
    ):
        super().__init__()

        self.stoi = stoi
        self.unk_idx = unk_idx
        self.max_len = max_len

        vocab_size = unk_idx + 1  # include UNK

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.rnn = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.output_dim = hidden_dim * 2

    def encode_smiles(self, smiles_list):
        batch_tokens = []

        for smi in smiles_list:
            tokens = [
                self.stoi.get(c, self.unk_idx)
                for c in smi[:self.max_len]
            ]

            if len(tokens) < self.max_len:
                tokens += [0] * (self.max_len - len(tokens))

            batch_tokens.append(tokens)

        return torch.tensor(batch_tokens, dtype=torch.long)

    def forward(self, smiles_list):
        """
        smiles_list: List[str]
        returns: (B, D)
        """

        x = self.encode_smiles(smiles_list)
        x = x.to(next(self.parameters()).device)

        x = self.embedding(x)                  # (B, L, E)
        _, h = self.rnn(x)                    # (layers*2, B, H)

        h = h.view(self.rnn.num_layers, 2, x.size(0), -1)
        h = h[-1]                             # (2, B, H)
        h = torch.cat([h[0], h[1]], dim=-1)   # (B, 2H)

        return h


# =========================
# Main
# =========================

def main(args):
    smiles = load_smiles(args.smiles_file)

    # build vocab
    stoi, unk_idx = build_vocab(smiles)

    # build model
    model = SmilesEncoder(
        stoi=stoi,
        unk_idx=unk_idx,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        max_len=args.max_len,
    )

    # initialise weights
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # save EVERYTHING to one file
    payload = {
        "model_state_dict": model.state_dict(),
        "config": {
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "max_len": args.max_len,
        },
        "stoi": stoi,
        "unk_idx": unk_idx,
    }

    torch.save(payload, args.outfile)

    print(f"Saved encoder to {args.outfile}")
    print(f"Vocab size: {len(stoi)} (UNK index = {unk_idx})")


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
