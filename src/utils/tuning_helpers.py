""" Helper function for hyperparameter tuning
    Currently just config reading with pyaml
"""

import optuna

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

class BestTrialCallback:
    """ Stateful callback for monitoring if this trial is the best one
        Standard I/O operations are always called, just add best to filenames if best model
    """
    def __init__(self, config):
        # best seen trial number
        self.best_seen = {"number"}
        # need config for I/O information
        self.config = config


    # Types encfored by optuna.study.optimize(callbacks=[])
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
        if study.best_trial.number == trial.number and self.best_seen["number"] != trial.number:
            self.best_seen["number"] = trial.number
            # If True, promote trial to best by changing filenames from I/O to _best_{trial.number}.
            # result.json
            # data/processed/{base_name}_pretrained.npz
            # data/processed/{base_name}_finetuned.npz
            # data/models/{base_name}.hdf5
            return 
        else:
            return 