""" Script for running manual process development for graph network
"""

import sys
from pathlib import Path

import argparse
import logging
import yaml
import json
import copy
import pandas as pd
import numpy as np
import pprint

import torch
from torch_geometric.loader import DataLoader
import optuna

from src.utils.data_preprocessing import smi_to_mol, mol_to_graph, MoleculeDataset, split_dataset
from src.utils.evaluate_embeddings import supervised_embedding_eval, unsupervised_embedding_eval, evaluate_full_model, encoder_embeddings_out
from src.training.pretrain import build_pretrain_encoder, PretrainTrainer
from src.training.finetune import build_finetune_model, FinetuneTrainer
from src.utils.tuning_helpers import apply_search_space, BestTrialCallback
from src.utils.model_io import save_pretrained_encoder, save_full_model

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
    def __init__(self, base_config, device, base_name=None):
        self.base_config = base_config
        self.device = device
        self.base_name = base_name

    def __call__(self, trial=None):
        """ Evaluates a model
        """
        # base config to outdir/base_config.yaml
        with open(self.base_config["outdir"] / "base_config.yaml", "w") as fw:
            yaml.dump(self.base_config, fw, sort_keys=False)
        
        config = self.make_config(trial)

        self.prepare_data(config)

        results = {}

        if config["pretrain"]:
            self.build_graph_encoder_ssl(config)
            self.run_pretrain(config)
            embeddings, labelsout = encoder_embeddings_out(self.encoder, self.dataloader)
            pretrain_encoder_stats = self.evaluate_encoder(embeddings, labelsout)
            metrics = pretrain_encoder_stats # only relevant if not finetuning

            # I/O
            np.savez(f"{config['datadir']}/processed/{self.base_name}_pretrained.npz",
                     embeddings=embeddings, 
                     labels=labelsout
                     )
            results["pretrain"] = metrics
            log.debug("\n" + pprint.pformat(pretrain_encoder_stats, width=1))
            save_pretrained_encoder(f"{config['datadir']}/models/{self.base_name}_pretrained_encoder.pt",
                                    self.encoder,
                                    config,
                                    extra={"ridge_rmse":results["pretrain"]["ridge_rmse"]},
                                    )

        if config["finetune"]:
            self.build_model(config)
            self.run_finetune(config)
            embeddings, labelsout = encoder_embeddings_out(self.encoder, self.dataloader)
            finetune_encoder_stats = self.evaluate_encoder(embeddings, labelsout)
            
            # I/O
            np.savez(f"{config['datadir']}/processed/{self.base_name}_finetuned.npz",
                     embeddings=embeddings, 
                     labels=labelsout
                     )
            results["finetune"] = metrics
            log.debug("\n" + pprint.pformat(finetune_encoder_stats, width=1))
            metrics = self.evaluate_model()
            save_full_model(f"{config['datadir']}/models/{self.base_name}_model.pt",
                            self.model.encoder,
                            self.model.reg_head,
                            config,
                            extra={"rmse":metrics[config["objective"]]}
                            )
        
        # if hyperparameter trial, output trial score and params to .csv
        if trial is not None:
            with open(config["outdir"] / "tuning.csv", "a") as fa:
                csvline = f"\n{trial.number},{metrics[config['objective']]}"
                for v in trial.params.values():
                    csvline += f",{v}"
                fa.write(csvline)
        
        results["score"] = metrics[config["objective"]]
        with open(config["outdir"] / "result.json", "w") as fw:
            json.dump(results, fw)

        log.info(f"DONE. Metric: {results['score']:.4f}")

        return metrics[config["objective"]]
    
    def make_config(self, trial):
        config = copy.deepcopy(self.base_config)

        if trial is not None:
            search_space = config.get("tuning", {}).get("search_space", {})
            config = apply_search_space(config, trial, search_space)
            log.info("Running trial " + f"{trial.number}")
            log.debug("Hyperparameters:\n" + pprint.pformat(trial.params))
            with open(self.base_config["outdir"] / "trial_config.yaml", "w") as fw:
                yaml.dump(config, fw, sort_keys=False)

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
        feat_dim = self.dataloader.dataset[0].x.size(-1)
        config["feat_dim"] = feat_dim
        embed_dim = config["encoder"]["embed_dim"]
        self.graph_encoder_ssl = build_pretrain_encoder(feat_dim, embed_dim, config)
        return

    def run_pretrain(self, config):
        pretrain_trainer = PretrainTrainer(self.device, config)
        self.encoder = pretrain_trainer.fit(self.graph_encoder_ssl, self.dataloader)
        return
    
    def build_model(self, config):
        self.model = build_finetune_model(self.encoder, config)
        return
    
    def run_finetune(self, config):
        finetune_trainer = FinetuneTrainer(self.device, config)
        self.model, self.encoder = finetune_trainer.fit(self.model, self.train_loader, self.val_loader)
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
    
def setup_logging(log_dir):
    """ configure logging
    """
    log_dir.mkdir(parents=False, exist_ok=True)

    log = logging.getLogger()
    log.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # stdout (debug level) 
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    log.addHandler(console)

    # run log 
    run_handler = logging.FileHandler(log_dir / "run.log", mode="w")
    run_handler.setLevel(logging.INFO)
    run_handler.setFormatter(formatter)
    log.addHandler(run_handler)

    # detail log 
    detail_handler = logging.FileHandler(log_dir / "detail.log", mode="w")
    detail_handler.setLevel(logging.DEBUG)
    detail_handler.setFormatter(formatter)
    log.addHandler(detail_handler)
    return

log = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/defaults.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set paths for input and output
    config["smi_file"] = Path(config["smi_file"])
    config["affins_csv"] = Path(config["affins_csv"])
    config["outdir"] = Path(config["outdir"])
    config["datadir"] = Path(config["datadir"])

    setup_logging(config["outdir"])

    if config["device"] == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            log.info("device is cuda")
        else:
            log.warning("cuda selected, put not available, defaulting to cpu")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        log.info("device is cpu")

    # single baseline run
    if not config.get("tuning", None).get("run_tuning", None):
        log.info("Hyperparameter tuning is OFF")
        config.pop("tuning")
        objective = Objective(config, device, base_name=config["run_name"])
        score = objective()

    # hyperparameter tuning run (set tuning in config)
    else:
        log.info("Hyperparameter tuning is ON")
        # create .csv for score, tuning parameters
        with open(config["outdir"] / "tuning.csv", "w") as fw:
            headerline = "trial,score"
            for param in config["tuning"]["search_space"]:
                headerline += f",{param['name']}"
            fw.write(headerline)
        
        best_trial_cb = BestTrialCallback(config)
        study  = optuna.create_study(direction="minimize")
        study.optimize(Objective(config, device, base_name=config["run_name"]), n_trials=config["tuning"]["n_trials"], callbacks=[best_trial_cb])