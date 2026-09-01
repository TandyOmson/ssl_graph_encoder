import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

class ECFP4Encoder:
    def __init__(self, feat_dim, embed_dim):
        self.fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        super().__init__()

    def encode_smiles(self, smiles_list):
        batch_fps = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                fp = [0] * 2048
            else:
                fp = self.fpgen.GetFingerprint(mol)
                fp = list(fp.ToBitString())
                fp = [int(bit) for bit in fp]

            batch_fps.append(fp)

        return torch.tensor(batch_fps, dtype=torch.long)

    def forward(self, data):
        smiles = data.smiles
        x = self.encode_smiles(smiles)
        x = x.to(next(self.parameters()).device)

        return x
    