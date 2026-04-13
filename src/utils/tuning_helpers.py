""" Helper function for hyperparameter tuning
    Currently just config reading with pyaml
"""

from pathlib import Path
import optuna
import pprint

def set_tuning_param_by_path(config, path, suggestion):
    """ Set config['a']['b']['c'] given "name" field 'a.b.c'  
    """
    keys = path.split(".")
    current = config
    for k in keys[:-1]:
        current = current[k]
    current[keys[-1]] = suggestion

def suggest_tuning_param_from_config(trial, spec):
    """ Creates suggestions for optuna trial object from each tuning parameter in config
    """
    t = spec["type"]

    if t == "float":
        return trial.suggest_float(
            spec["name"], float(spec["low"]), float(spec["high"]), log=spec.get("log", False)
        )
    if t == "int":
        return trial.suggest_int(
            spec["name"], int(spec["low"]), int(spec["high"]), step=spec.get("step", 1)
        )
    if t == "categorical":
        return trial.suggest_categorical(
            spec["name"], spec["choices"]
        )
    
    raise ValueError(f"Uknown spec type: {t}")

def apply_search_space(config, trial, search_space):
    """ Changes options in config to trial suggestions
    """
    for spec in search_space:
        suggestion = suggest_tuning_param_from_config(trial, spec)
        set_tuning_param_by_path(config, spec["name"], suggestion)
    return config

import logging
log = logging.getLogger(__name__)

class BestTrialCallback:
    """ Stateful callback for monitoring if this trial is the best one
        Standard I/O operations are always called, just add best to filenames if best model
    """
    def __init__(self, config):
        # best seen trial number
        self.best_seen = {"number" : None}
        # need config for I/O information
        self.config = config

    # Types encfored by optuna.study.optimize(callbacks=[])
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
        if study.best_trial.number == trial.number and self.best_seen["number"] != trial.number:
            log.info(f"Found best trial number {trial.number}, changing files to _best")
            self.best_seen["number"] = trial.number
            # If True, promote trial to best by changing filenames from I/O to _best.
            # logs/{base_name}/config.yaml
            src = Path(self.config["outdir"] / "trial_config.yaml")
            src.replace(src.with_stem(f"config_best"))
            # logs/{base_name}/result.json
            src = Path(self.config["outdir"] / "result.json")
            src.replace(src.with_stem(f"result_best"))
            # data/processed/{base_name}_pretrained.npz
            src = Path(self.config["datadir"] / "processed" / f"{self.config['run_name']}_pretrained.npz")
            src.replace(src.with_stem(f"{self.config['run_name']}_pretrained_best"))
            # data/processed/{base_name}_finetuned.npz
            src = Path(self.config["datadir"] / "processed" / f"{self.config['run_name']}_finetuned.npz")
            src.replace(src.with_stem(f"{self.config['run_name']}_finetuned_best"))
            # data/models/{base_name}_pretrained_encoder.pt
            src = Path(self.config["datadir"] / "models" / f"{self.config['run_name']}_pretrained_encoder.pt")
            src.replace(src.with_stem(f"{self.config['run_name']}_pretrained_encoder_best"))
            # data/models/{base_name}_model.pt
            src = Path(self.config["datadir"] / "models" / f"{self.config['run_name']}_model.pt")
            src.replace(src.with_stem(f"{self.config['run_name']}_model_best"))
            return 
        else:
            return 
        
class LogDistributionsOnce:
    """ Write tuning param disributions (search space) to log.info after the first trial is complete
    """
    def __init__(self):
        self.logged = False

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        if self.logged:
            return
        
        log.info(f"search space:\n" + pprint.pformat(study.trials[0].distributions))
        self.logged = True
