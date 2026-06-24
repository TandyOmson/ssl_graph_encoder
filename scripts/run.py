""" Training for graph networks
"""
from pathlib import Path
import argparse
import logging
import yaml
import json
import copy
import numpy as np
import pprint

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
import optuna

# from the installed sll_graph_encoder package
from ssl_graph_encoder.utils.data_preprocessing import MoleculeDataset, SmilesDataset, split_dataset, make_loader
from ssl_graph_encoder.utils.evaluate_embeddings import evaluate_full_model, evaluate_full_model_classification, encoder_embeddings_out, compute_embedding_splits
from ssl_graph_encoder.training.pretrain import build_pretrain_encoder, PretrainTrainer
from ssl_graph_encoder.training.finetune import build_finetune_model, FinetuneTrainer
from ssl_graph_encoder.utils.tuning_helpers import apply_search_space, BestTrialCallback, LogDistributionsOnce, log_trial_metrics_and_params
from ssl_graph_encoder.utils.model_io import save_pretrained_encoder, save_full_model, load_pretrained_encoder, setup_logging
from ssl_graph_encoder.utils.evalutation import EmbeddingEvaluator

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
    def __init__(self, base_config, device):
        self.base_config = base_config
        self.device = device

    def __call__(self, trial=None):
        """ Evaluates a model
        """
        config = self.make_config(trial)
        self.embed_evaluator = EmbeddingEvaluator(config["metrics"])
        
        if not config.get("use_smiles", False):
            self.prepare_data(config)
        else:
            # This is only an option so that I can sneak in language models to finetuning step for like-for-like comparisons. Dont use regularly
            self.prepare_smiles_data(config)

        results = {}
        if config.get("pretrain", False):
            log.info("=== PRETRAIN START ===")
            metrics = self.run_pretrain(config)
            results["pretrain"] = metrics
            
        # if there is no pretraining, a file with encoder architecture is expected
        else:
            if config["trained_encoder_file"] is not None:
                config["trained_encoder_file"] = Path(config["trained_encoder_file"])
            else:
                raise FileNotFoundError("Expected trained_encoder_file in config if not pretraining")
            
            self.encoder, payload = load_pretrained_encoder(config["trained_encoder_file"])

        if config.get("finetune", False):
            log.info("=== FINETUNE START ===")
            metrics = self.run_finetune(config)
            results["finetune"] = metrics
        
        results["score"] = metrics[config["objective"]]
        with open(config["outdir"] / "result.json", "w") as fw:
            json.dump(results, fw)

        log.info(f"DONE. Metric: {results['score']:.4f}\n")

        return metrics[config["objective"]], results
    
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

            train_dataset, val_dataset, test_dataset = split_dataset(dataset, val_frac=0.1, test_frac=0.1)
            self.train_loader = DataLoader(train_dataset, batch_size=config["pretrain"]["batch_size"], shuffle=True)
            self.test_loader = DataLoader(test_dataset, batch_size=config["pretrain"]["batch_size"], shuffle=False) 
            self.val_loader = DataLoader(val_dataset, batch_size=config["pretrain"]["batch_size"], shuffle=False)

        else:
            train_dataset = MoleculeDataset(config["train_file"])
            test_dataset = MoleculeDataset(config["test_file"])
            val_dataset = MoleculeDataset(config["val_file"])

            self.train_loader = make_loader(train_dataset, config, shuffle=True)
            self.test_loader  = make_loader(test_dataset, config, shuffle=False)
            self.val_loader = make_loader(val_dataset, config, shuffle=False) 
        return 
    
    def prepare_smiles_data(self, config):
        # again, this is only used to sneak in language models to finetuning step, not for regular use
        train_dataset = SmilesDataset(config["train_file"])
        test_dataset = SmilesDataset(config["test_file"])
        val_dataset = SmilesDataset(config["val_file"])

        self.train_loader = make_loader(train_dataset, config, shuffle=True)
        self.test_loader  = make_loader(test_dataset, config, shuffle=False)
        self.val_loader = make_loader(val_dataset, config, shuffle=False) 
        return 

    def run_pretrain(self, config):
        feat_dim = self.train_loader.dataset[0].x.size(-1)
        config["feat_dim"] = feat_dim
        self.graph_encoder_ssl = build_pretrain_encoder(feat_dim, config["encoder"]["embed_dim"], config)

        pretrain_trainer = PretrainTrainer(self.device, config)
        self.encoder = pretrain_trainer.fit(self.graph_encoder_ssl, self.train_loader)
        
        self.prepare_data(config) # Reset the dataloader due to a quirk in pyG where if dataloader.dataset.get sees a list, it will always output a list
        
        save_pretrained_encoder(f"{config['outdir']}/pretrained_encoder.pt",
                                self.encoder,
                                config,
                                )
        
        if config["pretrain"].get("ignore_labels", False) or not hasattr(self.train_loader.dataset[0], 'y'):
            log.warning("ignore_labels is True in pretrain; skipping supervised embedding evaluation metrics")
            embeddings, _ = encoder_embeddings_out(self.encoder, self.test_loader, sample_size=5000)
            self.embed_evaluator.unsupervised_eval(embeddings)
            metrics = self.embed_evaluator.results # only relevant if not finetuning

            # I/O
            np.savez(f"{config['outdir']}/embedding_sample_pretrained.npz",
                    embeddings=embeddings,
                    )
        else:
            embeddings, labels, split_idxs, test_embed = compute_embedding_splits(self.encoder, (self.train_loader, self.test_loader, self.val_loader))
            self.embed_evaluator.unsupervised_eval(test_embed)
            self.embed_evaluator.supervised_eval(embeddings, labels, split_idxs)
            
            metrics = self.embed_evaluator.results # only relevant if not finetuning
            # I/O
            np.savez(f"{config['outdir']}/embedding_sample_pretrained.npz",
                    embeddings=embeddings, 
                    labels=labels
                    )
        log.debug("PRETRAIN EMBEDDING STATS:\n" + pprint.pformat(self.embed_evaluator.results, width=1))
        return metrics

    def run_finetune(self, config):
        self.model = build_finetune_model(self.encoder, config)

        finetune_trainer = FinetuneTrainer(self.device, config)
        self.model, self.encoder = finetune_trainer.fit(self.model, self.train_loader, self.test_loader)

        save_full_model(f"{config['outdir']}/model.pt",
                        self.model.encoder,
                        self.model.pred_head,
                        config,
                        )
        
        save_pretrained_encoder(f"{config['outdir']}/finetuned_encoder.pt",
                                self.encoder,
                                config,
                                )

        embeddings, labels, split_idxs, test_embed = compute_embedding_splits(self.encoder, (self.train_loader, self.test_loader, self.val_loader))
        self.embed_evaluator.supervised_eval(embeddings, labels, split_idxs)
        metrics = self.embed_evaluator.results

        # I/O
        np.savez(f"{config['outdir']}/embedding_sample_finetuned.npz",
                    embeddings=embeddings, 
                    labels=labels
                    )
        log.debug("FINETUNE EMBEDDING STATS:\n" + pprint.pformat(metrics, width=1))
        metrics = self.evaluate_model(classification=config.get("classification", False))
        return metrics

    def evaluate_model(self, classification=False):
        """ Evaluate full mode with encoder and prediction head using the test set
        """
        if classification:
            acc, f1 = evaluate_full_model_classification(self.model, self.test_loader)
            return {"accuracy" : acc, "f1_score" : f1}
        else:
            rmse, r2 = evaluate_full_model(self.model, self.test_loader)
            return {"rmse" : rmse, "r2" : r2}
    
log = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, default="config/defaults.yaml", help="Path to config file")
    parser.add_argument("--outdir", required=True, default="name of output directory in logs")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set paths for input and output
    config["outdir"] = Path(args.outdir)
    config["datadir"] = Path(config["datadir"])

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
    
    # base config to outdir/base_config.yaml
    with open(config["outdir"] / "base_config.yaml", "w") as fw:
        yaml.dump(config, fw, sort_keys=False)

    classification = config.get("classification", False)
    if classification:
        log.info("Running in classification mode")
    else:
        log.info("Running in regression mode")

    # single baseline run
    if not config.get("tuning", None):
        log.info("Hyperparameter tuning is OFF")
        objective = Objective(config, device)
        score, _ = objective()

    # hyperparameter tuning run (set tuning in config)
    else:
        log.info("Hyperparameter tuning is ON")

        def optuna_wrapper(trial):
            objective = Objective(config, device)
            metrics, results = objective(trial)
            pretrain_encoder_stats = {f"pretrain_{k}": v for k, v in results["pretrain"].items()}
            finetune_encoder_stats = {f"finetune_{k}": v for k, v in results["finetune"].items()}
            all_metrics = {**pretrain_encoder_stats, **finetune_encoder_stats}

            log_trial_metrics_and_params(config["outdir"] / "tuning.csv", 
                                         trial.number, 
                                         metrics[config["objective"]], 
                                         all_metrics, 
                                         trial.params
                                         )
            return metrics[config["objective"]]

        # optuna callbacks
        best_trial_cb = BestTrialCallback(config) # changes files to best (otherwise trials overwrite standard I/O)
        log_dist_cb = LogDistributionsOnce() # writes the parameter distributions (search space) to log.info after first trial 

        study  = optuna.create_study(direction="minimize")
        study.optimize(optuna_wrapper, 
                       n_trials=config["tuning"]["n_trials"], 
                       callbacks=[best_trial_cb, log_dist_cb]
                       )

        log.info(f"Tuning complete. Best trial: {study.best_trial.number}")
        log.info(f"Best trial params: {study.best_trial.params}")
        log.info(f"Best trial value: {study.best_trial.value}")
