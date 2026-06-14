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
from torch_geometric.datasets import TUDataset
import optuna

# form the install sll_graph_encoder package
from ssl_graph_encoder.utils.data_preprocessing import MoleculeDataset, split_dataset
from ssl_graph_encoder.utils.evaluate_embeddings import supervised_embedding_eval, unsupervised_embedding_eval, evaluate_full_model, encoder_embeddings_out, supervised_embedding_eval_classification, evaluate_full_model_classification
from ssl_graph_encoder.training.pretrain import build_pretrain_encoder, PretrainTrainer
from ssl_graph_encoder.training.finetune import build_finetune_model, FinetuneTrainer
from ssl_graph_encoder.utils.tuning_helpers import apply_search_space, BestTrialCallback, LogDistributionsOnce, log_trial_metrics_and_params
from ssl_graph_encoder.utils.model_io import save_pretrained_encoder, save_full_model, load_pretrained_encoder

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

        self.classification = config.get("classification", False)
        if self.classification:
            log.info("Running in classification mode")
        else:
            log.info("Running in regression mode")
        self.prepare_data(config)

        results = {}

        if config.get("pretrain", False):
            self.build_graph_encoder_ssl(config)
            self.run_pretrain(config)
            self.prepare_data(config) # Reset the dataloader due to a quirk in pyG where if dataloader.dataset.get sees a list, it will always output a list
            
            save_pretrained_encoder(f"{config['datadir']}/models/{self.base_name}_pretrained_encoder.pt",
                                    self.encoder,
                                    config,
                                    #extra={"ridge_rmse":results["pretrain"]["ridge_rmse"]},
                                    )
            
            if config["pretrain"].get("ignore_labels", False) or not hasattr(self.dataloader.dataset[0], 'y'):
                log.warning("ignore_labels is True in pretrain; skipping supervised embedding evaluation metrics")
                embeddings, _ = encoder_embeddings_out(self.encoder, self.dataloader, sample_size=10000)
                pretrain_encoder_stats = self.evaluate_encoder_unsupervised_only(embeddings)
                metrics = pretrain_encoder_stats

                # I/O
                np.savez(f"{config['datadir']}/processed/{self.base_name}_pretrained.npz",
                        embeddings=embeddings,
                        )
            else:
                embeddings, labelsout = encoder_embeddings_out(self.encoder, self.dataloader, sample_size=10000)
                pretrain_encoder_stats = self.evaluate_encoder(embeddings, labelsout, classification=self.classification)
                metrics = pretrain_encoder_stats # only relevant if not finetuning

                # I/O
                np.savez(f"{config['datadir']}/processed/{self.base_name}_pretrained.npz",
                        embeddings=embeddings, 
                        labels=labelsout
                        )
            results["pretrain"] = metrics
            log.debug("PRETRAIN EMBEDDING STATS:\n" + pprint.pformat(pretrain_encoder_stats, width=1))
        # if there is no pretraining, a trained encoder file is expected
        else:
            if config["trained_encoder_file"] is not None:
                config["trained_encoder_file"] = Path(["trained_encoder_file"])
            else:
                raise FileNotFoundError("Expected trained_encoder_file in config if not pretraining")
            
            self.encoder, payload = load_pretrained_encoder(config["trained_encoder_file"])

        if config.get("finetune", False):
            self.build_model(config)
            self.run_finetune(config)

            save_full_model(f"{config['datadir']}/models/{self.base_name}_model.pt",
                            self.model.encoder,
                            self.model.pred_head,
                            config,
                            #extra={"rmse":metrics[config["objective"]]}
                            )
            
            save_pretrained_encoder(f"{config['datadir']}/models/{self.base_name}_finetuned_encoder.pt",
                                    self.encoder,
                                    config,
                                    )
        

            embeddings, labelsout = encoder_embeddings_out(self.encoder, self.dataloader, sample_size=10000)
            finetune_encoder_stats = self.evaluate_encoder(embeddings, labelsout, classification=self.classification)
            
            # I/O
            np.savez(f"{config['datadir']}/processed/{self.base_name}_finetuned.npz",
                     embeddings=embeddings, 
                     labels=labelsout
                     )
            results["finetune"] = finetune_encoder_stats
            log.debug("FINETUNE EMBEDDING STATS:\n" + pprint.pformat(finetune_encoder_stats, width=1))
            metrics = self.evaluate_model(classification=self.classification)
            
        # if hyperparameter trial, output trial score, other metrics and params to .csv
        if trial is not None:
            pretrain_encoder_stats = {f"pretrain_{k}": v for k, v in pretrain_encoder_stats.items()}
            finetune_encoder_stats = {f"finetune_{k}": v for k, v in finetune_encoder_stats.items()}
            all_metrics = {**pretrain_encoder_stats, **finetune_encoder_stats}

            log_trial_metrics_and_params(config["outdir"] / "tuning.csv", 
                                         trial.number, 
                                         metrics[config["objective"]], 
                                         all_metrics, 
                                         trial.params
                                         )
        
        results["score"] = metrics[config["objective"]]
        with open(config["outdir"] / "result.json", "w") as fw:
            json.dump(results, fw)

        log.info(f"DONE. Metric: {results['score']:.4f}\n")

        return metrics[config["objective"]]
    
    def make_config(self, trial):
        config = copy.deepcopy(self.base_config)

        if trial is not None:
            search_space = config.get("tuning", {}).get("search_space", {})
            config = apply_search_space(config, trial, search_space)
            log.info("Running trial " + f"{trial.number}")
            log.info("Hyperparameters:\n" + pprint.pformat(trial.params))
            with open(self.base_config["outdir"] / "trial_config.yaml", "w") as fw:
                yaml.dump(config, fw, sort_keys=False)

        return config
    
    def prepare_data(self, config):
        if config["download"]:
            dataset = TUDataset(config["datadir"] / "raw", name=config["download_name"], use_node_attr=True)
        else:
            dataset = MoleculeDataset(config["datafile"])

        self.dataloader = DataLoader(dataset, 
                                batch_size=config["pretrain"]["batch_size"], 
                                shuffle=True, 
                                num_workers=config["pretrain"].get("loader_worker_num", 4),
                                pin_memory=True,
                                persistent_workers=True,
                                )

        # split dataset for finetuning, remake loaders
        train_dataset, val_dataset, test_dataset = split_dataset(dataset, val_frac=0.1, test_frac=0.1)
        self.train_loader = DataLoader(train_dataset, batch_size=config["pretrain"]["batch_size"], shuffle=True)
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
    def evaluate_encoder_unsupervised_only(embeddings):
        """ Evaluate the encoder embedddings using only unsupervised metrics (no labels)
        """
        spread_mean, spread_median, dist_cv = unsupervised_embedding_eval(embeddings)
        return {"spread_mean" : spread_mean, "spread_median" : spread_median, "dist_csv" : dist_cv}

    @staticmethod
    def evaluate_encoder(embeddings, labels, classification=False):
        """ Evaludate the encoder embedddings irrespective of any prediction head
        """
        spread_mean, spread_median, dist_cv = unsupervised_embedding_eval(embeddings)
        
        # Check for missing labels
        if np.isnan(labels).any():
            return {"spread_mean" : spread_mean, "spread_median" : spread_median, "dist_csv" : dist_cv}

        if classification:
            ridge_acc, ridge_f1, knn_acc, knn_f1 = supervised_embedding_eval_classification(embeddings, labels)
            return {"ridge_acc" : ridge_acc, "ridge_f1": ridge_f1, "knn_acc" : knn_acc, "knn_f1" : knn_f1,
                    "spread_mean" : spread_mean, "spread_median" : spread_median, "dist_csv" : dist_cv}
        else:
            ridge_rmse, ridge_r2, knn_rmse, knn_r2, spearman_corr = supervised_embedding_eval(embeddings, labels)
            return {"ridge_rmse" : ridge_rmse, "ridge_r2": ridge_r2, "knn_rmse" : knn_rmse, "knn_r2" : knn_r2, "spearman_corr" : spearman_corr,
                    "spread_mean" : spread_mean, "spread_median" : spread_median, "dist_csv" : dist_cv}

    def evaluate_model(self, classification=False):
        """ Evaluate full mode with encoder and prediction head using the test set
        """
        if classification:
            acc, f1 = evaluate_full_model_classification(self.model, self.test_loader)
            return {"accuracy" : acc, "f1_score" : f1}
        else:
            rmse, r2 = evaluate_full_model(self.model, self.test_loader)
            return {"rmse" : rmse, "r2" : r2}
    
def setup_logging(log_dir):
    """ configure logging
    """
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    
    try:
        log_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        raise Exception(f"Log directory {log_dir} already exists. Exiting...")

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

    # pretrain and finetune loggers
    pretrain_logger = logging.getLogger("pretrain")
    pretrain_logger.setLevel(logging.INFO)

    data_handler = logging.FileHandler(log_dir / "pretrain.dat", mode="w")
    data_handler.setFormatter(logging.Formatter("%(message)s"))

    pretrain_logger.addHandler(data_handler)
    pretrain_logger.propagate = False # prevent duplication to root logs

    finetune_logger = logging.getLogger("finetune")
    finetune_logger.setLevel(logging.INFO)

    data_handler = logging.FileHandler(log_dir / "finetune.dat", mode="w")
    data_handler.setFormatter(logging.Formatter("%(message)s"))

    finetune_logger.addHandler(data_handler)
    finetune_logger.propagate = False # prevent duplication to root logs

    return

log = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/defaults.yaml", help="Path to config file")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set paths for input and output
    config["outdir"] = Path(config["outdir"])
    config["datadir"] = Path(config["datadir"])
    if config["datafile"] is not None:
        config["datafile"] = Path(config["datafile"])

    setup_logging(config["outdir"])

    if args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            log.info(f"device is cuda ({torch.cuda.get_device_name(device.index)})")
        else:
            log.warning("cuda selected but CUDA is not available; defaulting to cpu")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        log.info("device is cpu")

    config["device"] = device.type

    # single baseline run
    if not config.get("tuning", None):
        log.info("Hyperparameter tuning is OFF")
        objective = Objective(config, device, base_name=config["run_name"])
        score = objective()

    # hyperparameter tuning run (set tuning in config)
    else:
        log.info("Hyperparameter tuning is ON")
        
        # optuna callbacks
        best_trial_cb = BestTrialCallback(config) # changes files to best (otherwise trials overwrite standard I/O)
        log_dist_cb = LogDistributionsOnce() # writes the parameter distributions (search space) to log.info after first trial 

        study  = optuna.create_study(direction="minimize")
        study.optimize(Objective(config, device, base_name=config["run_name"]), 
                       n_trials=config["tuning"]["n_trials"], 
                       callbacks=[best_trial_cb, log_dist_cb]
                       )

        log.info(f"Tuning complete. Best trial: {study.best_trial.number}")
        log.info(f"Best trial params: {study.best_trial.params}")
        log.info(f"Best trial value: {study.best_trial.value}")
