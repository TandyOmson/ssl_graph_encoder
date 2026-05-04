""" Baseline NN model
    DEPRECATED: Can now use run.py with config/defaults.yaml
"""
import logging
import pandas as pd
import numpy as np
import argparse
import yaml

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch_geometric.loader import DataLoader

from ssl_graph_encoder.utils.data_preprocessing import smi_to_mol, mol_to_graph, MoleculeDataset, split_dataset
from ssl_graph_encoder.training.pretrain import create_pretrain_encoder, pretrain
from ssl_graph_encoder.training.finetune import create_finetune_model, finetune
from ssl_graph_encoder.utils.evaluate_embeddings import supervised_embedding_eval, unsupervised_embedding_eval, evaluate_full_model, encoder_embeddings_out

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Get config
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config/defaults.yaml", help="Path to config file")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# Load data
datadir = config["datadir"]

smis = [i.rstrip() for i in open(config["smi_file"], 'r').readlines()]
mols = [smi_to_mol(smi, add_hs=True) for smi in smis]
graphs = [mol_to_graph(mol) for mol in mols]

affins_df = pd.read_csv(config["affins_csv"], index_col=0)
affins_df = affins_df[affins_df["pose_1"].between(-20, 0)]
labels = affins_df["pose_1"].values

# Restrict to molecules that have affinity labels
graphs = [graphs[i-1] for i in affins_df.index]

log.info(f"Loaded graphs and labels. Number of samples: {len(labels)}")

for data, y in zip(graphs, labels):
    data.y = torch.tensor(y, dtype=torch.float).view(1)
dataset = MoleculeDataset(graphs)
dataloader = DataLoader(dataset, 
                        batch_size=config["pretrain"]["batch_size"], 
                        shuffle=True, 
                        num_workers=config["pretrain"]["num_workers"]
                        )

# Pretrain encoder
log.info("Starting pretraining...")

feat_dim = dataloader.dataset[0].x.shape[1]
embed_dim = config["encoder"]["embed_dim"]

graph_encoder_ssl = create_pretrain_encoder(feat_dim, embed_dim, config)
# encoder is just the encoder part of graph_encoder_ssl, to be passed to finetune
encoder = pretrain(graph_encoder_ssl, dataloader, config)
embeddings, labelsout = encoder_embeddings_out(encoder, dataloader, outfile=None)

lr_rmse, lr_r2, knn_rmse, knn_r2, spearman_corr = supervised_embedding_eval(embeddings, labelsout)
spread_mean, spread_median, dist_cv = unsupervised_embedding_eval(embeddings)

print("Pretrained embedding stats:")
print(f"Ridge RMSE: {lr_rmse:.4f}, R2: {lr_r2:.4f}")
print(f"kNN RMSE: {knn_rmse:.4f}, R2: {knn_r2:.4f}")
print(f"Spearman correlation: {spearman_corr:.4f}")
print(f"Spread (mean): {spread_mean:.4f}")
print(f"Spread (median): {spread_median:.4f}")
print(f"Distance CV: {dist_cv:.4f}")

log.info("saving pretrained embeddings and labels...")
np.savez(f"{datadir}/processed/nn_baseline_pretrained.npz", embeddings=embeddings, labels=labelsout)

# split dataset for finetuning, remake loaders
train_dataset, val_dataset, test_dataset = split_dataset(dataset, val_frac=0.1, test_frac=0.1)
train_loader = DataLoader(train_dataset, batch_size=config["finetune"]["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Fine tune encoder with regression head. Need to remake dataloader if labels not already included
model = create_finetune_model(encoder, config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model, encoder = finetune(model, train_loader, val_loader, config)
embeddings, labelsout = encoder_embeddings_out(encoder, dataloader, outfile=None)

lr_rmse, lr_r2, knn_rmse, knn_r2, spearman_corr = supervised_embedding_eval(embeddings, labelsout)
spread_mean, spread_median, dist_cv = unsupervised_embedding_eval(embeddings)

print("Finetuned embedding stats:")
print(f"Ridge RMSE: {lr_rmse:.4f}, R2: {lr_r2:.4f}")
print(f"kNN RMSE: {knn_rmse:.4f}, R2: {knn_r2:.4f}")
print(f"Spearman correlation: {spearman_corr:.4f}")
print(f"Spread (mean): {spread_mean:.4f}")
print(f"Spread (median): {spread_median:.4f}")
print(f"Distance CV: {dist_cv:.4f}")

log.info("saving fine-tuned embeddings and labels...")
np.savez(f"{datadir}/processed/nn_baseline_finetuned.npz", embeddings=embeddings, labels=labelsout)

rmse, r2 = evaluate_full_model(model, test_loader)
print("Full Model Stats (using regression head):")
print(f"RMSE: {rmse:.4f}, R2: {r2:.4f}")