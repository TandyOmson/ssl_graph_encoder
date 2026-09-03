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
        max_len=120,
    ):
        super().__init__()

        self.stoi = stoi
        self.unk_idx = unk_idx
        self.max_len = max_len

        vocab_size = unk_idx + 1

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0,
        )

        self.conv1 = nn.Conv1d(
            embed_dim,
            256,
            kernel_size=5,
            padding=2,
        )

        self.conv2 = nn.Conv1d(
            256,
            256,
            kernel_size=5,
            padding=2,
        )

        self.selu = nn.SELU()

        self.pool = nn.MaxPool1d(
            kernel_size=2,
            stride=2,
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        #self.fc = nn.Linear(
        #    hidden_dim,
        #    feat_dim,
        #)

        self.output_dim = hidden_dim*2

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

        # (B, L, E)
        x = self.embedding(x)

        # (B, E, L)
        x = x.transpose(1, 2)

        x = self.selu(self.conv1(x))
        x = self.selu(self.conv2(x))

        #x = self.pool(x)

        # (B, L, C)
        x = x.transpose(1, 2)

        out, (h_n, _) = self.lstm(x)
        #h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        h = out.mean(dim=1)

        # final LSTM layer hidden state
        #h = h_n[-1]

        return h
