""" Script for running manual process development for graph network
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import logging
import yaml
import copy
import pandas as pd
import pprint

import torch
from torch_geometric.loader import DataLoader

from src.utils.data_preprocessing import smi_to_mol, mol_to_graph, MoleculeDataset, split_dataset
from src.utils.evaluate_embeddings import supervised_embedding_eval, unsupervised_embedding_eval, evaluate_full_model, encoder_embeddings_out
from src.training.pretrain import create_pretrain_encoder, pretrain
from src.training.finetune import create_finetune_model, finetune

class Objective:
    """ Manual process development for graph model
        Handles:
            experiments - testing baseline models (baseline config)
            increments - incremental changes on baseline models (manual edits to baseline config)
            hyperparameter tuning - automatic parameter testing (optuna automatically edits baseline config)
        
        High level execution:
        objective = Objective(config, device)
        score = objective()      # manual run (experiments and increments)
        score = objective(trial) # optuna run (hyperparameter tuning)
    """
    def __init__(self, base_config, device, base_name="baseline_nn"):
        self.base_config = base_config
        self.device = device
        self.base_name = base_name

        # Output options

        # Runtime state
        self.splitter = None
        self.encoder = None # acts as reference to self.graph_encoder_ssl.encoder
        self.graph_encoder_ssl = None
        self.model = None
        # later can add pretrain optimizer, finetune optimizer and finetune criterion

    def __call__(self, trial=None):
        """ Evaluates a model
        """
        config = self.make_config(trial)

        self.prepare_data(config)

        if config["pretrain"]:
            self.build_graph_encoder_ssl(config)
            self.run_pretrain(config)
            embeddings, labelsout = encoder_embeddings_out(self.encoder, self.dataloader, outfile=f"{config['datadir']}/processed/{self.base_name}_pretrained.npz")
            pretrain_encoder_stats = self.evaluate_encoder(embeddings, labelsout)
            pprint.pprint(pretrain_encoder_stats, width=1)

        self.build_model(config)
        self.run_finetune(config)
        embeddings, labelsout = encoder_embeddings_out(self.encoder, self.dataloader, outfile=f"{config['datadir']}/processed/{self.base_name}_finetuned.npz")
        finetune_encoder_stats = self.evaluate_encoder(embeddings, labelsout)
        pprint.pprint(finetune_encoder_stats)

        metrics = self.evaluate_model()
        
        # negative?
        return metrics[config["objective"]]
    
    def make_config(self, trial):
        config = copy.deepcopy(self.base_config)

        if trial is not None:
            # optuna trial logic
            config["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        return config
    
    # Essential methods for call
    def prepare_data(self, config):
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
        self.dataloader = DataLoader(dataset, 
                                batch_size=config["pretrain"]["batch_size"], 
                                shuffle=True, 
                                num_workers=config["pretrain"]["num_workers"]
                                )

        # split dataset for finetuning, remake loaders
        train_dataset, val_dataset, test_dataset = split_dataset(dataset, val_frac=0.1, test_frac=0.1)
        self.train_loader = DataLoader(train_dataset, batch_size=config["finetune"]["batch_size"], shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)    
    
        return 
    
    def build_graph_encoder_ssl(self, config):
        feat_dim = self.dataloader.dataset[0].x.shape[1]
        embed_dim = config["encoder"]["embed_dim"]
        self.graph_encoder_ssl = create_pretrain_encoder(feat_dim, embed_dim, config)
        return

    def run_pretrain(self, config):
        self.encoder = pretrain(self.graph_encoder_ssl, self.dataloader, config)
        return
    
    def build_model(self, config):
        self.model = create_finetune_model(self.encoder, config)
        self.model.to(self.device)
        return
    
    def run_finetune(self, config):
        self.model, self.encoder = finetune(self.model, self.train_loader, self.val_loader, config)
        return
    
    @staticmethod
    def evaluate_encoder(embeddings, labels):
        ridge_rmse, ridge_r2, knn_rmse, knn_r2, spearman_corr = supervised_embedding_eval(embeddings, labels)
        spread_mean, spread_median, dist_cv = unsupervised_embedding_eval(embeddings)
        return {"ridge_rmse" : ridge_rmse, "ridge_r2": ridge_r2, "knn_rmse" : knn_rmse, "knn_r2" : knn_r2, "spearman_corr" : spearman_corr,
                "spread_mean" : spread_mean, "spread_median" : spread_median, "dist_csv" : dist_cv}

    def evaluate_model(self):
        rmse, r2 = evaluate_full_model(self.model, self.test_loader)
        return {"rmse" : rmse, "r2" : r2}
    
    # Monitoring, logging, checkpointing, storage and metadata

# using optuna
# study.optimize(Objective(dataset, BASE_CONFIG), n_trials=50)

logging.basicConfig(level=logging.info)
log = logging.getLogger("RUN")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/defaults.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    objective = Objective(config, device, base_name="baseline_nn")
    score = objective()
    log.info(f"DONE. Metric: {score:.4f}")