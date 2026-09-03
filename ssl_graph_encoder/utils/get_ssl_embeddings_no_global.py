""" Gets embeddings from SSL graph encoder 
    The global variables are only used dor multiprocessing. Its a singleton pattern so this is needed for analysis scripts that run multiple encoders
"""

from ssl_graph_encoder.utils.smiles_to_graph import SmilesToGraph
from ssl_graph_encoder.utils.model_io import load_pretrained_encoder

import torch
from torch_geometric.data import Data
from pathlib import Path
import numpy as np

def get_model(encoderfile, stgfile, map_location):
    # modelfile includes encoder class path
    _model, _ = load_pretrained_encoder(Path(encoderfile), map_location)
    _model.eval()
    _stg = SmilesToGraph.from_config(Path(stgfile))
    return _model, _stg

class sslEmbeddings:
    def __init__(self, encoder_file, smiles_to_graph_file, map_location="cpu"):
        self.encoder, self.stg = get_model(encoder_file, smiles_to_graph_file, map_location)

    def get_embeddings(self, smi):
        z, _ = self.stg.smiles_to_graphs(smi, return_all_confs=False)
        with torch.no_grad():
            emb = self.encoder(z)
        
        return emb.to("cpu").numpy().astype(np.float64)

class sslEmbeddingsLM:
    def __init__(self, encoder_file, smiles_to_graph_file, map_location="cpu"):
        self.encoder, self.stg = get_model(encoder_file, smiles_to_graph_file, map_location)

    def get_embeddings(self, smi):
        data = Data()
        data.smiles = smi
        with torch.no_grad():
            emb = self.encoder(data)
        
        return emb
