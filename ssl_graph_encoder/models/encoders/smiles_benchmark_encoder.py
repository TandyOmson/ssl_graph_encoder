import torch
import torch.nn as nn

class SmilesEncoder(nn.Module):
    def __init__(
        self,
        feat_dim,
        embed_dim,
        stoi=None,
        unk_idx=None,
        hidden_dim=256,
        num_layers=2,
        max_len=120,
    ):
        super().__init__()

        self.stoi = stoi
        self.unk_idx = unk_idx
        self.max_len = max_len

        vocab_size = unk_idx + 1

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

    def forward(self, data):
        smiles = data.smiles
        x = self.encode_smiles(smiles)
        x = x.to(next(self.parameters()).device)

        x = self.embedding(x)
        _, h = self.rnn(x)

        h = h.view(self.rnn.num_layers, 2, x.size(0), -1)
        h = h[-1]
        h = torch.cat([h[0], h[1]], dim=-1)

        return h