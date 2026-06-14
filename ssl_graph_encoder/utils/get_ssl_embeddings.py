""" Gets embeddings from SSL graph encoder 
"""

from ssl_graph_encoder.utils.smiles_to_graph import SmilesToGraph
from ssl_graph_encoder.utils.model_io import load_pretrained_encoder

import torch
from pathlib import Path
import numpy as np

_model = None
_stg = None

def get_model(encoderfile, stgfile, map_location):
    global _model
    global _stg
    if _model is None:
        # modelfile includes encoder class path
        _model, _ = load_pretrained_encoder(Path(encoderfile), map_location)
        _model.eval()
    if _stg is None:
        _stg = SmilesToGraph.from_config(Path(stgfile))
    return _model, _stg

class sslEmbeddings:
    def __init__(self, encoder_file, smiles_to_graph_file, map_location="cpu"):
        self.encoder, self.stg = get_model(encoder_file, smiles_to_graph_file, map_location)

    def get_embeddings(self, smi):
        z, _ = _stg.smiles_to_graphs(smi, return_all_confs=False)
        with torch.no_grad():
            emb = self.encoder(z)
        
        return emb.to("cpu").numpy().astype(np.float64)