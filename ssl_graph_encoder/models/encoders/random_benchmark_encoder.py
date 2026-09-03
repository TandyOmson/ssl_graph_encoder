import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np

class RandomEncoder(nn.Module):
    def __init__(self, feat_dim, embed_dim):
        self.fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self.embed_dim = embed_dim
        self.output_dim=512
        
        super().__init__()
        self.register_buffer("_dummy", torch.empty(0))

    def encode_smiles(self, smiles_list):
        batch_fps = []

        for smi in smiles_list:
            batch_fps.append(np.random.rand(512))

        return torch.tensor(batch_fps, dtype=torch.float32)

    def forward(self, data):
        smiles = data.smiles
        x = self.encode_smiles(smiles)
        x = x.to(self._dummy.device)

        return x
    
